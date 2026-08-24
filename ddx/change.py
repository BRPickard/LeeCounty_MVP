"""Pre/post change detection.

The pipeline is deliberately explainable rather than learned: every number a
user sees traces back to a named quantity (spectral change, structural loss,
vegetation loss, water gain) they can reason about and argue with. Order of
operations matters and is the same one a photogrammetrist would use by hand:

1. intersect the usable-pixel masks of both dates
2. correct sub-pixel misregistration between the two acquisitions
3. normalise the post date's radiometry onto the pre date, using pixels that
   did not change, so sun angle and atmosphere do not read as damage
4. derive per-pixel change measures
5. combine them into a damage score, down-weighting change that is obviously
   vegetation phenology rather than structural loss

Thresholds live in :class:`DetectorConfig` because they are sensor-dependent.
The shipped values are calibrated against the synthetic demo (see
``scripts/calibrate.py``); recalibrate before trusting them on a new sensor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from .geo import Grid
from .imagery.base import BandStack

# Damage classes, ordered from least to most severe.
NONE, POSSIBLE, MODERATE, SEVERE, DESTROYED = (
    "none", "possible", "moderate", "severe", "destroyed")
DAMAGE_CLASSES = (NONE, POSSIBLE, MODERATE, SEVERE, DESTROYED)
DAMAGED_CLASSES = (MODERATE, SEVERE, DESTROYED)

DAMAGE_COLORS = {
    NONE: "#2f9e44",
    POSSIBLE: "#fab005",
    MODERATE: "#fd7e14",
    SEVERE: "#e03131",
    DESTROYED: "#862e9c",
}


@dataclass
class DetectorConfig:
    """Tunable constants for the detector."""

    # Reflectance change that counts as a fully saturated spectral signal.
    magnitude_ref: float = 0.11
    # SSIM loss that counts as a fully saturated structural signal.
    structure_ref: float = 0.45
    structure_window: int = 7
    # Weights; they sum to 1.
    weight_magnitude: float = 0.55
    weight_structure: float = 0.45
    # How strongly change on pixels that were vegetation before the event is
    # discounted when scoring structures (0 = no discount, 1 = ignore entirely).
    vegetation_discount: float = 0.65
    veg_ndvi_threshold: float = 0.30
    # Flooding: post-date water index level and its rise from the pre date.
    flood_ndwi_level: float = 0.05
    flood_ndwi_rise: float = 0.10
    # Vegetation loss reported per parcel.
    veg_loss_ndvi_drop: float = 0.18
    # Maximum misregistration searched for, in pixels.
    max_shift_px: int = 6
    # Class breaks applied to an aggregated damage score.
    class_breaks: tuple[float, float, float, float] = (0.16, 0.33, 0.55, 0.78)

    def classify(self, score: float) -> str:
        if score is None or not np.isfinite(score):
            return NONE
        a, b, c, d = self.class_breaks
        if score < a:
            return NONE
        if score < b:
            return POSSIBLE
        if score < c:
            return MODERATE
        if score < d:
            return SEVERE
        return DESTROYED

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = DetectorConfig()


# --- primitives --------------------------------------------------------------
def _safe_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference index with a guarded denominator."""
    denom = a + b
    out = np.zeros_like(a, dtype="float32")
    ok = np.abs(denom) > 1e-6
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return np.clip(out, -1.0, 1.0)


def ndvi(stack: BandStack) -> np.ndarray | None:
    if not stack.has("nir", "red"):
        return None
    return _safe_index(stack.band("nir"), stack.band("red"))


def ndwi(stack: BandStack) -> np.ndarray | None:
    """McFeeters NDWI: positive over open water."""
    if not stack.has("green", "nir"):
        return None
    return _safe_index(stack.band("green"), stack.band("nir"))


def estimate_shift(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray,
                   max_shift: int = 6) -> tuple[int, int]:
    """Integer (row, col) offset of ``moving`` relative to ``reference``.

    Phase correlation on the overlapping centre. Real pre/post pairs from
    different orbits are routinely off by a pixel or two, and an uncorrected
    shift shows up as a bright halo of "structural loss" on every roof edge.
    """
    h, w = reference.shape
    if min(h, w) < 32 or max_shift <= 0:
        return (0, 0)
    # Work on the largest centred power-of-two-ish window we can afford.
    size = int(min(h, w, 1024))
    r0, c0 = (h - size) // 2, (w - size) // 2
    a = np.where(valid, reference, 0.0)[r0:r0 + size, c0:c0 + size].astype("float64")
    b = np.where(valid, moving, 0.0)[r0:r0 + size, c0:c0 + size].astype("float64")
    if a.std() < 1e-6 or b.std() < 1e-6:
        return (0, 0)
    a = a - a.mean()
    b = b - b.mean()
    window = np.outer(np.hanning(size), np.hanning(size))
    fa = np.fft.rfft2(a * window)
    fb = np.fft.rfft2(b * window)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    cross = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 1e-12)
    corr = np.fft.irfft2(cross, s=(size, size))
    corr = np.fft.fftshift(corr)
    centre = size // 2
    lo, hi = centre - max_shift, centre + max_shift + 1
    patch = corr[lo:hi, lo:hi]
    peak = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return (int(peak[0] - max_shift), int(peak[1] - max_shift))


def apply_shift(arr: np.ndarray, shift: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    """Shift an array by whole pixels, filling the vacated edge."""
    dr, dc = shift
    if dr == 0 and dc == 0:
        return arr
    out = np.full_like(arr, fill)
    src_r = slice(max(0, -dr), arr.shape[0] - max(0, dr))
    dst_r = slice(max(0, dr), arr.shape[0] - max(0, -dr))
    src_c = slice(max(0, -dc), arr.shape[1] - max(0, dc))
    dst_c = slice(max(0, dc), arr.shape[1] - max(0, -dc))
    out[dst_r, dst_c] = arr[src_r, src_c]
    return out


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Least-squares gain/offset on centred data, or None if degenerate."""
    if x.size < 64:
        return None
    xm, ym = float(x.mean()), float(y.mean())
    xc = x - xm
    var = float((xc * xc).mean())
    if var < 1e-8:
        return None
    gain = float((xc * (y - ym)).mean() / var)
    if not np.isfinite(gain):
        return None
    return gain, ym - gain * xm


def invariant_pixels(pre: BandStack, post: BandStack, valid: np.ndarray,
                     cfg: "DetectorConfig", keep: float = 0.4) -> np.ndarray:
    """Pick pseudo-invariant features: surfaces that plausibly did not change.

    Selection is deliberately made *against identity* — the pixels whose raw
    pre/post difference is smallest — rather than against a fitted line.
    Selecting against a fitted line is self-reinforcing: an early bad gain
    picks the pixels that agree with it, which confirms the bad gain. Since
    surface-reflectance products are already atmospherically corrected, the
    true gain sits near 1, so scoring candidates against identity is both
    unbiased enough and immune to that feedback.

    Vegetation is excluded first: leaf-on/leaf-off and storm defoliation move
    a lot of reflectance without any sensor difference to correct.
    """
    shared = [b for b in pre.bands if b in post.bands]
    if not shared:
        return valid
    base = valid.copy()
    pre_ndvi, post_ndvi = ndvi(pre), ndvi(post)
    if pre_ndvi is not None and post_ndvi is not None:
        non_veg = (pre_ndvi < cfg.veg_ndvi_threshold) & (post_ndvi < cfg.veg_ndvi_threshold)
        if int((base & non_veg).sum()) >= 2048:
            base &= non_veg
    if int(base.sum()) < 256:
        return valid

    diff = np.stack([post.band(b) - pre.band(b) for b in shared])
    magnitude = np.sqrt((diff ** 2).mean(axis=0))
    cutoff = float(np.quantile(magnitude[base], keep))
    chosen = base & (magnitude <= cutoff)
    return chosen if int(chosen.sum()) >= 256 else base


def normalize_to(reference: np.ndarray, moving: np.ndarray, valid: np.ndarray,
                 iterations: int = 2, keep_fraction: float = 0.8,
                 invariant: np.ndarray | None = None,
                 gain_limits: tuple[float, float] = (0.6, 1.7),
                 ) -> tuple[np.ndarray, tuple[float, float]]:
    """Fit ``moving`` onto ``reference`` with a robust per-band gain and offset.

    The fit runs on the pseudo-invariant set chosen by :func:`invariant_pixels`,
    then trims the worst-fitting fifth once to shed any survivors that did
    change. A fit implying a gain outside ``gain_limits`` is rejected and the
    moving image returned untouched: over a corrected surface-reflectance
    product the true gain is near 1, so a wild estimate means the invariant
    set was contaminated, and a bad normalisation fabricates damage
    everywhere it is applied.
    """
    base = valid if invariant is None else (valid & invariant)
    if int(base.sum()) < 256:
        base = valid
    mask = base.copy()

    fit: tuple[float, float] | None = None
    for _ in range(max(1, iterations)):
        candidate = _fit_line(moving[mask], reference[mask])
        if candidate is None:
            break
        fit = candidate
        gain, offset = fit
        residual = np.abs(reference - (moving * gain + offset))
        cutoff = np.quantile(residual[base], keep_fraction)
        new_mask = base & (residual <= cutoff)
        if int(new_mask.sum()) < 256:
            break
        mask = new_mask

    if fit is None:
        return moving.copy(), (1.0, 0.0)
    gain, offset = fit
    if not (gain_limits[0] <= gain <= gain_limits[1]):
        return moving.copy(), (1.0, 0.0)
    return (moving * gain + offset).astype("float32"), (float(gain), float(offset))


def ssim_map(a: np.ndarray, b: np.ndarray, window: int = 7,
             data_range: float = 0.5) -> np.ndarray:
    """Local structural similarity in [-1, 1], computed with box filters."""
    window = max(3, window | 1)
    filt = lambda z: ndimage.uniform_filter(z.astype("float64"), size=window, mode="nearest")
    mu_a, mu_b = filt(a), filt(b)
    saa = filt(a * a) - mu_a * mu_a
    sbb = filt(b * b) - mu_b * mu_b
    sab = filt(a * b) - mu_a * mu_b
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    out = np.divide(num, den, out=np.ones_like(num), where=np.abs(den) > 1e-12)
    return np.clip(out, -1.0, 1.0).astype("float32")


# --- result ------------------------------------------------------------------
@dataclass
class ChangeResult:
    """Per-pixel change layers for one pre/post pair on one grid."""

    grid: Grid
    valid: np.ndarray = field(repr=False)
    score: np.ndarray = field(repr=False)
    magnitude: np.ndarray = field(repr=False)
    structure_loss: np.ndarray = field(repr=False)
    brightness_delta: np.ndarray = field(repr=False)
    ndvi_delta: np.ndarray | None = field(default=None, repr=False)
    ndwi_delta: np.ndarray | None = field(default=None, repr=False)
    flooded: np.ndarray | None = field(default=None, repr=False)
    vegetation_loss: np.ndarray | None = field(default=None, repr=False)
    pre_vegetation: np.ndarray | None = field(default=None, repr=False)
    config: DetectorConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0

    def summary(self) -> dict[str, Any]:
        v = self.valid
        if not v.any():
            return {"coverage": 0.0, "mean_score": None}
        return {
            "coverage": round(self.coverage, 4),
            "mean_score": round(float(self.score[v].mean()), 4),
            "p95_score": round(float(np.quantile(self.score[v], 0.95)), 4),
            "flooded_fraction": (round(float(self.flooded[v].mean()), 4)
                                 if self.flooded is not None else None),
            "vegetation_loss_fraction": (round(float(self.vegetation_loss[v].mean()), 4)
                                         if self.vegetation_loss is not None else None),
            **self.diagnostics,
        }


def detect_change(pre: BandStack, post: BandStack,
                  config: DetectorConfig | None = None) -> ChangeResult:
    """Compare two co-gridded scenes and produce per-pixel change layers."""
    cfg = config or DEFAULT_CONFIG
    if pre.shape != post.shape:
        raise ValueError("pre and post stacks must share a grid")

    shared_bands = [b for b in pre.bands if b in post.bands]
    if not shared_bands:
        raise ValueError("pre and post scenes have no bands in common")

    valid = pre.valid & post.valid
    pre_gray = pre.brightness()
    post_gray = post.brightness()

    # 2. co-registration
    shift = estimate_shift(pre_gray, post_gray, valid, max_shift=cfg.max_shift_px)
    if shift != (0, 0):
        post_data = np.stack([apply_shift(post.band(b), shift) for b in post.bands])
        post = BandStack(post_data, post.bands, apply_shift(post.valid.astype("float32"),
                                                            shift) > 0.5,
                         post.grid, post.scene)
        valid = pre.valid & post.valid
        post_gray = post.brightness()

    # 3. radiometric normalisation, per band, onto the pre date.
    # Normalised difference indices are ratios, so they survive an
    # un-normalised gain well enough to pick the invariant surfaces first.
    invariant = invariant_pixels(pre, post, valid, cfg)

    gains: dict[str, list[float]] = {}
    norm_post: dict[str, np.ndarray] = {}
    for band in shared_bands:
        fitted, (gain, offset) = normalize_to(pre.band(band), post.band(band), valid,
                                              invariant=invariant)
        norm_post[band] = fitted
        gains[band] = [round(gain, 4), round(offset, 5)]
    post_norm = BandStack(np.stack([norm_post[b] for b in shared_bands]),
                          tuple(shared_bands), post.valid, post.grid, post.scene)
    pre_sub = BandStack(np.stack([pre.band(b) for b in shared_bands]),
                        tuple(shared_bands), pre.valid, pre.grid, pre.scene)

    # 4. change measures
    diff = post_norm.data - pre_sub.data
    magnitude = np.sqrt((diff ** 2).mean(axis=0)).astype("float32")

    post_gray_norm = post_norm.brightness()
    pre_gray_sub = pre_sub.brightness()
    structure_loss = np.clip(
        1.0 - ssim_map(pre_gray_sub, post_gray_norm, window=cfg.structure_window),
        0.0, 1.0).astype("float32")
    brightness_delta = (post_gray_norm - pre_gray_sub).astype("float32")

    pre_ndvi, post_ndvi = ndvi(pre_sub), ndvi(post_norm)
    ndvi_delta = ((post_ndvi - pre_ndvi).astype("float32")
                  if pre_ndvi is not None and post_ndvi is not None else None)
    pre_ndwi, post_ndwi = ndwi(pre_sub), ndwi(post_norm)
    ndwi_delta = ((post_ndwi - pre_ndwi).astype("float32")
                  if pre_ndwi is not None and post_ndwi is not None else None)

    flooded = None
    if post_ndwi is not None and ndwi_delta is not None:
        flooded = ((post_ndwi > cfg.flood_ndwi_level)
                   & (ndwi_delta > cfg.flood_ndwi_rise) & valid)
    vegetation_loss = None
    pre_vegetation = None
    if pre_ndvi is not None and ndvi_delta is not None:
        pre_vegetation = (pre_ndvi > cfg.veg_ndvi_threshold) & valid
        vegetation_loss = (pre_vegetation & (ndvi_delta < -cfg.veg_loss_ndvi_drop))

    # 5. combined structural damage score
    score = (cfg.weight_magnitude * np.clip(magnitude / cfg.magnitude_ref, 0.0, 1.0)
             + cfg.weight_structure * np.clip(structure_loss / cfg.structure_ref, 0.0, 1.0))
    if pre_vegetation is not None and cfg.vegetation_discount > 0:
        # Leaf-off/leaf-on and storm defoliation both move a lot of reflectance
        # without a building being touched; discount change on formerly green
        # pixels so the structural score stays about structures.
        score = np.where(pre_vegetation, score * (1.0 - cfg.vegetation_discount), score)
    if flooded is not None:
        # Standing water over what was not vegetation is itself damage evidence.
        score = np.maximum(score, np.where(flooded, 0.45, 0.0))
    score = np.clip(score, 0.0, 1.0).astype("float32")
    score[~valid] = 0.0

    return ChangeResult(
        grid=pre.grid, valid=valid, score=score, magnitude=magnitude,
        structure_loss=structure_loss, brightness_delta=brightness_delta,
        ndvi_delta=ndvi_delta, ndwi_delta=ndwi_delta, flooded=flooded,
        vegetation_loss=vegetation_loss, pre_vegetation=pre_vegetation,
        config=cfg,
        diagnostics={
            "bands": shared_bands,
            "registration_shift_px": list(shift),
            "radiometric_fit": gains,
            "invariant_pixels": (int(invariant.sum()) if invariant is not None else None),
            "pre_scene": pre.scene.id,
            "post_scene": post.scene.id,
            "pre_date": pre.scene.date,
            "post_date": post.scene.date,
            "synthetic": bool(pre.scene.synthetic or post.scene.synthetic),
        },
    )
