import numpy as np
import pytest

from ddx.change import (DEFAULT_CONFIG, DetectorConfig, apply_shift,
                        detect_change, estimate_shift, invariant_pixels,
                        ndvi, ndwi, normalize_to, ssim_map)
from ddx.imagery.base import ALL_BANDS, BandStack, Scene


def make_stack(grid, values, scene_id="s", date="2025-01-01", valid=None):
    """Build a four-band stack from a dict of band -> array or scalar."""
    data = np.stack([np.broadcast_to(np.asarray(values[b], dtype="float32"),
                                     grid.shape).copy() for b in ALL_BANDS])
    scene = Scene(id=scene_id, provider="test", datetime=f"{date}T00:00:00Z")
    mask = np.ones(grid.shape, dtype=bool) if valid is None else valid
    return BandStack(data, ALL_BANDS, mask, grid, scene)


def test_estimate_and_apply_shift_round_trip():
    rng = np.random.default_rng(3)
    reference = rng.random((256, 256)).astype("float32")
    moved = apply_shift(reference, (4, -3))
    valid = np.ones((256, 256), dtype=bool)
    shift = estimate_shift(reference, moved, valid, max_shift=8)
    assert shift == (-4, 3)
    # Applying the estimate to the moved image restores the reference.
    restored = apply_shift(moved, shift)
    core = slice(16, -16)
    assert np.allclose(restored[core, core], reference[core, core], atol=1e-6)


def test_estimate_shift_returns_zero_on_flat_input():
    flat = np.zeros((64, 64), dtype="float32")
    assert estimate_shift(flat, flat, np.ones((64, 64), bool)) == (0, 0)


def test_normalize_recovers_gain_and_offset():
    rng = np.random.default_rng(0)
    reference = rng.random((200, 200)).astype("float32")
    moving = (reference * 1.2 + 0.03).astype("float32")
    _, (gain, offset) = normalize_to(reference, moving, np.ones((200, 200), bool))
    assert gain == pytest.approx(1 / 1.2, rel=0.02)
    assert offset == pytest.approx(-0.03 / 1.2, abs=0.005)


def test_normalize_rejects_implausible_gain():
    rng = np.random.default_rng(1)
    reference = rng.random((100, 100)).astype("float32")
    moving = (reference * 9.0).astype("float32")     # far outside the guard
    fitted, (gain, offset) = normalize_to(reference, moving, np.ones((100, 100), bool))
    assert (gain, offset) == (1.0, 0.0)
    assert np.array_equal(fitted, moving)


def test_ssim_is_one_for_identical_and_lower_for_scrambled():
    rng = np.random.default_rng(2)
    a = rng.random((80, 80)).astype("float32")
    assert ssim_map(a, a).mean() == pytest.approx(1.0, abs=1e-6)
    b = rng.random((80, 80)).astype("float32")
    assert ssim_map(a, b).mean() < 0.5


def test_indices():
    grid_shape = (4, 4)
    scene = Scene(id="x", provider="t", datetime="2025-01-01T00:00:00Z")
    data = np.stack([np.full(grid_shape, v, "float32") for v in (0.05, 0.1, 0.05, 0.4)])
    stack = BandStack(data, ALL_BANDS, np.ones(grid_shape, bool), None, scene)
    assert ndvi(stack)[0, 0] == pytest.approx((0.4 - 0.05) / (0.4 + 0.05))
    assert ndwi(stack)[0, 0] == pytest.approx((0.1 - 0.4) / (0.1 + 0.4))


def test_no_change_between_identical_scenes(grid):
    values = {"blue": 0.06, "green": 0.09, "red": 0.07, "nir": 0.32}
    pre = make_stack(grid, values, "pre")
    rng = np.random.default_rng(5)
    textured = {b: np.full(grid.shape, v) + rng.normal(0, 0.004, grid.shape)
                for b, v in values.items()}
    pre = make_stack(grid, textured, "pre")
    post = make_stack(grid, textured, "post")
    result = detect_change(pre, post)
    assert result.score.max() < DEFAULT_CONFIG.class_breaks[0]
    assert result.coverage == pytest.approx(1.0)


def test_detects_a_patch_that_turned_to_rubble(grid):
    rng = np.random.default_rng(6)
    base = {b: np.full(grid.shape, v, "float32") + rng.normal(0, 0.004, grid.shape)
            for b, v in (("blue", 0.06), ("green", 0.09), ("red", 0.07), ("nir", 0.32))}
    pre = make_stack(grid, base, "pre")
    after = {b: v.copy() for b, v in base.items()}
    patch = (slice(40, 70), slice(40, 70))
    for band, value in (("blue", 0.19), ("green", 0.20), ("red", 0.21), ("nir", 0.22)):
        after[band][patch] = value + rng.normal(0, 0.02, (30, 30))
    post = make_stack(grid, after, "post")

    result = detect_change(pre, post)
    inside = result.score[patch].mean()
    outside = np.delete(result.score.ravel(), np.ravel_multi_index(
        np.mgrid[patch].reshape(2, -1), result.score.shape)).mean()
    assert inside > DEFAULT_CONFIG.class_breaks[1]
    assert inside > outside * 3


def test_flood_flag_needs_water_signature(grid):
    rng = np.random.default_rng(7)
    base = {b: np.full(grid.shape, v, "float32") + rng.normal(0, 0.003, grid.shape)
            for b, v in (("blue", 0.07), ("green", 0.10), ("red", 0.08), ("nir", 0.30))}
    pre = make_stack(grid, base, "pre")
    after = {b: v.copy() for b, v in base.items()}
    patch = (slice(10, 40), slice(10, 40))
    after["nir"][patch] = 0.02      # water absorbs near infrared
    after["green"][patch] = 0.09
    post = make_stack(grid, after, "post")

    result = detect_change(pre, post)
    assert result.flooded is not None
    assert result.flooded[patch].mean() > 0.9
    assert result.flooded.mean() < 0.2


def test_invariant_selection_prefers_unchanged_pixels(grid):
    rng = np.random.default_rng(8)
    base = {b: np.full(grid.shape, v, "float32") + rng.normal(0, 0.002, grid.shape)
            for b, v in (("blue", 0.10), ("green", 0.11), ("red", 0.12), ("nir", 0.14))}
    pre = make_stack(grid, base, "pre")
    after = {b: v.copy() for b, v in base.items()}
    changed = (slice(0, grid.height // 2), slice(None))
    for band in after:
        after[band][changed] += 0.2
    post = make_stack(grid, after, "post")

    mask = invariant_pixels(pre, post, np.ones(grid.shape, bool), DEFAULT_CONFIG)
    # Almost everything selected should come from the untouched half.
    assert mask[changed].sum() < mask.sum() * 0.05


def test_mismatched_grids_are_rejected(grid):
    from ddx.geo import make_grid
    other = make_grid((-79.19, 35.47, -79.185, 35.475), 1.0)
    values = {"blue": 0.06, "green": 0.09, "red": 0.07, "nir": 0.32}
    with pytest.raises(ValueError):
        detect_change(make_stack(grid, values), make_stack(other, values))


def test_classify_covers_every_break():
    config = DetectorConfig()
    assert config.classify(0.0) == "none"
    assert config.classify(0.99) == "destroyed"
    assert config.classify(float("nan")) == "none"
    assert config.classify(None) == "none"
