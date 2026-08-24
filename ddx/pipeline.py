"""End-to-end assessment: AOI + two dates in, parcel damage table out."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config
from .assess import AssessmentResult, assess
from .buildings import fetch_buildings
from .change import DetectorConfig, detect_change
from .geo import (BBox, Grid, bbox_area_km2, buffer_bbox, make_grid,
                  union_bbox)
from .imagery import ALL_BANDS, BandStack, Scene, read_scene, resolve_scene
from .parcels import Parcel, ParcelStore
from .render import damage_ramp, encode_png, stretch
from .imagery.raster import write_geotiff

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


class PipelineError(RuntimeError):
    """Raised for user-correctable problems: AOI too big, no parcels, bad dates."""


@dataclass
class AssessmentRequest:
    """Everything needed to run one assessment."""

    pre_scene_id: str
    post_scene_id: str
    bbox: BBox | None = None
    parcel_ids: list[int] = field(default_factory=list)
    gsd: float | None = None
    building_source: str = "auto"
    detector: dict[str, Any] = field(default_factory=dict)
    pre_provider: str | None = None
    post_provider: str | None = None
    save_rasters: bool = True

    def detector_config(self) -> DetectorConfig:
        cfg = DetectorConfig()
        for key, value in (self.detector or {}).items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, type(getattr(cfg, key))(value)
                        if not isinstance(getattr(cfg, key), tuple) else tuple(value))
        return cfg

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_aoi(request: AssessmentRequest, store: ParcelStore
                ) -> tuple[BBox, list[Parcel]]:
    """Work out the analysis extent and the parcels inside it."""
    if request.parcel_ids:
        parcels = store.get_many(request.parcel_ids, geometry=True)
        if not parcels:
            raise PipelineError("none of the requested parcel ids exist")
        bbox = union_bbox([p.geometry.bounds for p in parcels if p.geometry])
        # A little margin so edge pixels and co-registration have context.
        bbox = buffer_bbox(bbox, 40.0)
    elif request.bbox:
        bbox = request.bbox
        parcels = store.in_bbox(bbox, geometry=True)
        if not parcels:
            raise PipelineError("no parcels intersect that area")
    else:
        raise PipelineError("give either a bbox or a list of parcel ids")

    area = bbox_area_km2(bbox)
    if area > config.MAX_AOI_KM2:
        raise PipelineError(
            f"area of interest is {area:.0f} km², over the {config.MAX_AOI_KM2:.0f} km² "
            "limit; draw a smaller box or select fewer parcels")
    if len(parcels) > config.MAX_PARCELS_PER_JOB:
        raise PipelineError(
            f"{len(parcels)} parcels selected, over the "
            f"{config.MAX_PARCELS_PER_JOB} limit for one run")
    return bbox, parcels


def choose_gsd(pre: Scene, post: Scene, requested: float | None, bbox: BBox) -> float:
    """Pick an analysis resolution both scenes can support, within the pixel budget."""
    native = max(pre.gsd or config.DEFAULT_ANALYSIS_GSD,
                 post.gsd or config.DEFAULT_ANALYSIS_GSD)
    gsd = requested or native
    # Never claim more detail than the coarser input actually has.
    gsd = max(gsd, native)
    grid = make_grid(bbox, gsd)
    pixels = grid.width * grid.height
    if pixels > config.MAX_ANALYSIS_PIXELS:
        scale = math.sqrt(pixels / config.MAX_ANALYSIS_PIXELS)
        gsd = gsd * scale
        log.info("coarsening analysis grid to %.2f m to stay within the pixel budget", gsd)
    return gsd


def run_assessment(request: AssessmentRequest, store: ParcelStore | None = None,
                   progress: ProgressFn | None = None,
                   artifact_dir: Path | None = None) -> AssessmentResult:
    """Run the full pipeline and return the assessment."""
    store = store or ParcelStore()
    if not store.exists():
        raise PipelineError(
            "no parcel database; run scripts/ingest_parcels.py first")

    def step(message: str, fraction: float) -> None:
        log.info("[%3.0f%%] %s", fraction * 100, message)
        if progress:
            progress(message, fraction)

    step("Resolving area of interest", 0.02)
    bbox, parcels = resolve_aoi(request, store)

    step("Locating imagery", 0.06)
    pre_scene = resolve_scene(request.pre_scene_id, bbox, request.pre_provider)
    post_scene = resolve_scene(request.post_scene_id, bbox, request.post_provider)
    if pre_scene.datetime > post_scene.datetime:
        pre_scene, post_scene = post_scene, pre_scene

    gsd = choose_gsd(pre_scene, post_scene, request.gsd, bbox)
    grid = make_grid(bbox, gsd)
    step(f"Reading pre-event scene {pre_scene.date} at {gsd:.2f} m", 0.12)
    pre = read_scene(pre_scene, grid, bands=ALL_BANDS)
    step(f"Reading post-event scene {post_scene.date}", 0.36)
    post = read_scene(post_scene, grid, bands=ALL_BANDS)

    if pre.coverage < 0.02 or post.coverage < 0.02:
        raise PipelineError(
            "the selected imagery barely covers this area — pick scenes whose "
            "footprint overlaps the parcels")

    step("Detecting change", 0.58)
    change = detect_change(pre, post, request.detector_config())

    step("Collecting building footprints", 0.72)
    building_set = fetch_buildings(bbox, parcels, source=request.building_source)

    step("Scoring parcels and structures", 0.84)
    result = assess(change, parcels, building_set, request.detector_config())
    result.summary["aoi"] = {
        "bbox": list(bbox),
        "area_km2": round(bbox_area_km2(bbox), 3),
        "gsd": round(gsd, 3),
        "grid": [grid.width, grid.height],
    }
    result.summary["pre_scene"] = pre_scene.as_dict()
    result.summary["post_scene"] = post_scene.as_dict()

    if request.save_rasters and artifact_dir is not None:
        step("Writing rasters", 0.94)
        try:
            save_artifacts(artifact_dir, grid, pre, post, change, result)
        except Exception as exc:      # artifacts are a convenience, not the answer
            log.warning("could not write rasters: %s", exc)
            result.warnings.append(f"raster export failed: {exc}")

    step("Done", 1.0)
    return result


def save_artifacts(directory: Path, grid: Grid, pre: BandStack, post: BandStack,
                   change: Any, result: AssessmentResult) -> None:
    """Persist display rasters so chips and downloads work after the job ends."""
    directory.mkdir(parents=True, exist_ok=True)
    pre_rgb = stretch(pre.rgb(), pre.valid)
    post_rgb = stretch(post.rgb(), post.valid)
    write_geotiff(str(directory / "pre_rgb.tif"),
                  np.transpose(pre_rgb, (2, 0, 1)), grid, "uint8",
                  band_names=["red", "green", "blue"])
    write_geotiff(str(directory / "post_rgb.tif"),
                  np.transpose(post_rgb, (2, 0, 1)), grid, "uint8",
                  band_names=["red", "green", "blue"])
    score_u8 = np.clip(change.score * 255, 0, 255).astype("uint8")
    score_u8[~change.valid] = 0
    write_geotiff(str(directory / "score.tif"), score_u8, grid, "uint8",
                  nodata=0, band_names=["damage_score"])
    overlay = damage_ramp(change.score, change.config, change.valid)
    (directory / "overlay.png").write_bytes(encode_png(overlay))
    (directory / "grid.json").write_text(json.dumps({
        "crs": grid.crs, "transform": list(grid.transform),
        "width": grid.width, "height": grid.height,
    }, indent=2))
