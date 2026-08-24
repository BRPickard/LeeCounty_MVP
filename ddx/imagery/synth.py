"""Procedural pre/post imagery for offline demos and tests.

This renders a plausible four-band scene over a real Lee County AOI: parcels
become lots, tax-roll footprints become roofs, the gaps between parcels become
roads, and everything else is vegetation. A post-event scene applies a damage
swath — torn roofs, debris, defoliation and a flooded low corridor.

It exists so the full pipeline (search, read, normalise, detect, roll up,
report) can be exercised end to end with no imagery subscription and no
network. It is a plumbing demo, not a validation of the detector against real
sensor physics; every scene it produces is flagged ``synthetic``.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage

from shapely.geometry import Point

from ..geo import WGS84, Grid, reproject, reproject_many
from .base import ALL_BANDS

# Damage states used as demo ground truth.
INTACT, MINOR, MAJOR, DESTROYED = "intact", "minor", "major", "destroyed"
TRUTH_STATES = (INTACT, MINOR, MAJOR, DESTROYED)

# Rough reflectance signatures, ordered (blue, green, red, nir).
SPECTRA: dict[str, tuple[float, float, float, float]] = {
    "vegetation": (0.035, 0.075, 0.045, 0.360),
    "grass":      (0.055, 0.105, 0.080, 0.290),
    "bare":       (0.130, 0.155, 0.185, 0.235),
    "asphalt":    (0.085, 0.090, 0.095, 0.110),
    "roof_light": (0.290, 0.300, 0.305, 0.330),
    "roof_dark":  (0.115, 0.120, 0.125, 0.150),
    "roof_metal": (0.330, 0.345, 0.340, 0.380),
    "debris":     (0.185, 0.190, 0.200, 0.215),
    "water":      (0.075, 0.085, 0.055, 0.020),
    "cloud":      (0.850, 0.860, 0.870, 0.880),
}


def _seed_from(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _unit_hash(*parts: Any) -> float:
    """Deterministic float in [0, 1) from arbitrary keys."""
    return (_seed_from(*parts) % 1_000_003) / 1_000_003.0


def _smooth_noise(shape: tuple[int, int], scale_px: float, rng: np.random.Generator,
                  octaves: int = 3) -> np.ndarray:
    """Fractal-ish smooth noise in 0..1, generated coarse then upsampled."""
    h, w = shape
    total = np.zeros(shape, dtype="float32")
    weight = 0.0
    for octave in range(octaves):
        scale = max(2.0, scale_px / (2 ** octave))
        ch, cw = max(2, int(h / scale)), max(2, int(w / scale))
        coarse = rng.random((ch, cw), dtype="float32")
        zoomed = ndimage.zoom(coarse, (h / ch, w / cw), order=1, mode="nearest")
        zoomed = zoomed[:h, :w]
        if zoomed.shape != shape:  # zoom can be off by a pixel
            pad_h, pad_w = h - zoomed.shape[0], w - zoomed.shape[1]
            zoomed = np.pad(zoomed, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")
            zoomed = zoomed[:h, :w]
        amp = 0.5 ** octave
        total += zoomed * amp
        weight += amp
    total /= weight
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / (hi - lo + 1e-6)


@dataclass
class SynthEvent:
    """A simulated disaster: when it hit, and where the worst damage ran.

    The swath is anchored in real-world coordinates rather than to the AOI, so
    zooming into one neighbourhood shows only the part of the storm that
    actually crossed it — the same behaviour a real event footprint has.
    """

    id: str
    name: str
    date: dt.date
    kind: str = "hurricane"
    centre: tuple[float, float] = (-79.17, 35.47)   # lon, lat of the track
    bearing_deg: float = 40.0                       # track direction, degrees from north
    half_width_m: float = 5200.0                    # distance to the 1/e damage contour
    severity: float = 1.0                           # 0..1.5 multiplier on damage odds
    river: tuple[float, float] = (-79.17, 35.47)    # anchor for the synthetic drainage
    flood_width_m: float = 260.0                    # flooded corridor half-width

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "date": self.date.isoformat(),
                "kind": self.kind, "severity": self.severity,
                "centre": list(self.centre), "bearing_deg": self.bearing_deg,
                "half_width_km": round(self.half_width_m / 1000.0, 2)}


DEFAULT_EVENT = SynthEvent(
    id="demo-hurricane-2025",
    name="Demo Hurricane (synthetic)",
    date=dt.date(2025, 9, 18),
    centre=(-79.17, 35.47),
    bearing_deg=40.0,
    half_width_m=5200.0,
    severity=1.0,
)


def _grid_xy(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Projected x/y coordinate arrays for every pixel centre on the grid."""
    a, _, c, _, e, f = grid.transform
    xs = c + a * (np.arange(grid.width, dtype="float64") + 0.5)
    ys = f + e * (np.arange(grid.height, dtype="float64") + 0.5)
    return np.meshgrid(xs, ys)


def swath_intensity(grid: Grid, event: SynthEvent) -> np.ndarray:
    """0..1 damage intensity across the grid, peaking on the storm centreline."""
    xx, yy = _grid_xy(grid)
    cx, cy = reproject(Point(*event.centre), WGS84, grid.crs).coords[0]
    theta = math.radians(event.bearing_deg)
    # Unit normal to the track; the dot product gives cross-track distance.
    nx, ny = math.cos(theta), -math.sin(theta)
    cross = (xx - cx) * nx + (yy - cy) * ny
    core = np.exp(-(cross / max(50.0, event.half_width_m)) ** 2).astype("float32")

    # Along-track gusts so damage is patchy rather than a clean gradient.
    rng = np.random.default_rng(_seed_from(event.id, "swath"))
    gust = 0.60 + 0.75 * _smooth_noise(grid.shape, max(20.0, min(grid.shape) / 6), rng)
    return np.clip(core * gust * event.severity, 0.0, 1.0)


def flood_mask(grid: Grid, event: SynthEvent, intensity: np.ndarray) -> np.ndarray:
    """Corridor along a synthetic watercourse that takes on water.

    The channel is a fixed sinusoid in projected coordinates, so neighbouring
    AOIs see the same river in the same place.
    """
    if event.flood_width_m <= 0:
        return np.zeros(grid.shape, dtype=bool)
    xx, yy = _grid_xy(grid)
    ax, ay = reproject(Point(*event.river), WGS84, grid.crs).coords[0]
    u = (xx - ax) / 1000.0
    channel = ay + 1400.0 * np.sin(u / 3.1) + 520.0 * np.sin(u / 0.9 + 1.7)
    width = event.flood_width_m * (0.55 + 1.45 * np.clip(intensity, 0, 1))
    return (np.abs(yy - channel) < width) & (intensity > 0.15)


def building_damage(building_id: str, event: SynthEvent, intensity: float) -> str:
    """Ground-truth damage state for one structure under a simulated event."""
    roll = _unit_hash(event.id, building_id)
    p = float(np.clip(intensity, 0.0, 1.0)) * event.severity
    if roll > p * 0.92:
        return INTACT
    if roll > p * 0.62:
        return MINOR
    if roll > p * 0.32:
        return MAJOR
    return DESTROYED


def _paint(out: np.ndarray, mask: np.ndarray, material: str, jitter: np.ndarray | None = None,
           blend: float = 1.0) -> None:
    """Blend a material's spectrum into the four-band stack where mask is set."""
    if not mask.any():
        return
    spectrum = SPECTRA[material]
    for b in range(4):
        value = spectrum[b]
        if jitter is not None:
            value = value * (0.85 + 0.3 * jitter[mask])
            out[b][mask] = out[b][mask] * (1 - blend) + value * blend
        else:
            out[b][mask] = out[b][mask] * (1 - blend) + value * blend


def _rasterize(geoms: Iterable, grid: Grid, values: Iterable[int] | None = None,
               dtype: str = "int32", fill: int = 0) -> np.ndarray:
    shapes = list(zip(geoms, values)) if values is not None else [(g, 1) for g in geoms]
    shapes = [(g, v) for g, v in shapes if g is not None and not g.is_empty]
    if not shapes:
        return np.full(grid.shape, fill, dtype=dtype)
    return rasterize(
        shapes, out_shape=grid.shape, transform=rasterio.Affine(*grid.transform),
        fill=fill, dtype=dtype, all_touched=False,
    )


def _phenology(date: dt.date) -> float:
    """Seasonal vegetation vigour, 0.55 in winter to 1.0 at midsummer."""
    day = date.timetuple().tm_yday
    return 0.55 + 0.45 * (0.5 - 0.5 * math.cos(2 * math.pi * (day - 20) / 365.25))


@dataclass
class SceneRenderRequest:
    grid: Grid
    date: dt.date
    scene_id: str
    parcels: Sequence[Any] = field(default_factory=list)
    buildings: Sequence[Any] = field(default_factory=list)
    event: SynthEvent | None = None
    post_event: bool = False
    cloud_cover: float = 0.0
    gain: float = 1.0
    offset: float = 0.0


def render_scene(req: SceneRenderRequest) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Render a synthetic scene.

    Returns ``(data, valid, truth)`` where data is (4, h, w) float32 in band
    order blue, green, red, nir; valid is the usable-pixel mask; and truth maps
    building id to its ground-truth damage state (empty for pre-event scenes).
    """
    grid = req.grid
    h, w = grid.shape
    rng = np.random.default_rng(_seed_from(req.scene_id, grid.transform[2], grid.transform[5]))
    stable = np.random.default_rng(_seed_from("basemap", grid.transform[2], grid.transform[5]))

    out = np.zeros((4, h, w), dtype="float32")
    texture = _smooth_noise((h, w), 6.0, stable)

    # 1. Background: woods and fields, with roads in the parcel gaps.
    _paint(out, np.ones((h, w), dtype=bool), "vegetation", jitter=texture)
    field_mask = _smooth_noise((h, w), max(30.0, min(h, w) / 6), stable) > 0.62
    _paint(out, field_mask, "grass", jitter=texture)

    parcel_geoms = reproject_many([p.geometry for p in req.parcels], WGS84, grid.crs)
    if parcel_geoms:
        lots = _rasterize(parcel_geoms, grid).astype(bool)
        # Road surface is the gap *between* neighbouring lots, not every pixel
        # outside the parcel fabric (which would pave the next county over).
        near_lots = ndimage.binary_dilation(lots, iterations=max(1, int(24.0 / grid.gsd)))
        _paint(out, near_lots & ~lots, "asphalt", jitter=texture, blend=0.85)
        # Mown yards inside developed lots.
        yard_noise = _smooth_noise((h, w), 12.0, stable)
        _paint(out, lots & (yard_noise > 0.45), "grass", jitter=texture, blend=0.7)

    # 2. Roofs. Materials and damage are rasterised as class codes in two
    # passes rather than one mask per building, which keeps a county-wide AOI
    # with tens of thousands of structures tractable.
    intensity = (swath_intensity(grid, req.event)
                 if (req.event and req.post_event) else np.zeros((h, w), dtype="float32"))
    truth: dict[str, str] = {}
    roof_index = np.zeros((h, w), dtype="int32")

    # Below roughly 5 m/pixel individual roofs are sub-pixel; skip drawing them.
    draw_roofs = req.buildings and grid.gsd <= 5.0
    if draw_roofs:
        roof_geoms = reproject_many([b.geometry for b in req.buildings], WGS84, grid.crs)
        keep = [(b, g) for b, g in zip(req.buildings, roof_geoms) if g is not None]
        materials, damages = [], []
        a, _, c, _, e, f = grid.transform
        for b, g in keep:
            materials.append(1 if _unit_hash(b.id, "mat") > 0.72
                             else 2 if _unit_hash(b.id, "mat") > 0.36 else 3)
            if req.event and req.post_event:
                cx, cy = g.centroid.x, g.centroid.y
                col = int(np.clip((cx - c) / a, 0, w - 1))
                row = int(np.clip((cy - f) / e, 0, h - 1))
                state = building_damage(b.id, req.event, float(intensity[row, col]))
                truth[b.id] = state
                damages.append(TRUTH_STATES.index(state))
            else:
                damages.append(0)

        geoms_only = [g for _, g in keep]
        roof_index = _rasterize(geoms_only, grid, materials)
        for code, material in ((1, "roof_metal"), (2, "roof_light"), (3, "roof_dark")):
            _paint(out, roof_index == code, material, jitter=texture)

        if req.event and req.post_event:
            damage_raster = _rasterize(geoms_only, grid, [d + 1 for d in damages])
            for code, frac in ((MINOR, 0.30), (MAJOR, 0.65), (DESTROYED, 0.95)):
                mask = damage_raster == (TRUTH_STATES.index(code) + 1)
                if not mask.any():
                    continue
                _paint(out, mask, "debris", jitter=texture, blend=frac)
                speckle = rng.normal(0.0, 0.030 * frac, size=int(mask.sum())).astype("float32")
                for band in range(4):
                    out[band][mask] += speckle
                if code == DESTROYED:
                    _paint(out, mask, "bare", jitter=texture, blend=0.55)
    elif req.buildings and req.event and req.post_event:
        # Still record ground truth even when roofs are not drawn.
        a, _, c, _, e, f = grid.transform
        for b in req.buildings:
            pt = b.geometry.representative_point()
            gx, gy = reproject(pt, WGS84, grid.crs).coords[0]
            col = int(np.clip((gx - c) / a, 0, w - 1))
            row = int(np.clip((gy - f) / e, 0, h - 1))
            truth[b.id] = building_damage(b.id, req.event, float(intensity[row, col]))

    # 3. Storm effects on vegetation and low ground (not roofs, not pavement).
    if req.event and req.post_event:
        pre_ndvi = (out[3] - out[2]) / (out[3] + out[2] + 1e-6)
        vegetated = (roof_index == 0) & (pre_ndvi > 0.25)
        defoliation = np.clip(intensity * 0.75, 0.0, 0.85)
        out[3][vegetated] *= (1.0 - 0.60 * defoliation)[vegetated]
        out[2][vegetated] *= (1.0 + 0.50 * defoliation)[vegetated]
        out[1][vegetated] *= (1.0 + 0.15 * defoliation)[vegetated]

        # Wind-thrown debris scattered over open ground near the worst damage.
        debris_field = (intensity > 0.55) & (roof_index == 0) & \
                       (_smooth_noise(grid.shape, 5.0, rng) > 0.86)
        _paint(out, debris_field, "debris", jitter=texture, blend=0.5)

        flooded = flood_mask(grid, req.event, intensity)
        if flooded.any():
            _paint(out, flooded, "water", jitter=texture, blend=0.88)

    # 4. Seasonal vigour, per-scene radiometry, sensor noise.
    vigour = _phenology(req.date)
    green_mask = roof_index == 0
    out[3][green_mask] *= 0.55 + 0.45 * vigour
    out[2][green_mask] *= 1.15 - 0.15 * vigour

    out *= req.gain
    out += req.offset
    out += rng.normal(0.0, 0.006, size=out.shape).astype("float32")

    # 5. Clouds, which also punch holes in the valid mask.
    valid = np.ones((h, w), dtype=bool)
    if req.cloud_cover > 0.5:
        cloud_field = _smooth_noise((h, w), max(30.0, min(h, w) / 4),
                                    np.random.default_rng(_seed_from(req.scene_id, "cloud")))
        threshold = np.quantile(cloud_field, 1.0 - min(0.95, req.cloud_cover / 100.0))
        clouds = cloud_field > threshold
        if clouds.any():
            _paint(out, clouds, "cloud", jitter=texture, blend=0.9)
            valid &= ~ndimage.binary_erosion(clouds, iterations=1)

    np.clip(out, 0.0, 1.2, out=out)
    return out, valid, truth


def band_order() -> tuple[str, ...]:
    return ALL_BANDS  # blue, green, red, nir
