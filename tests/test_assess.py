import numpy as np
import pytest

from ddx.assess import (LOSS_FACTOR, assess, developed_core, focus_window_px)
from ddx.buildings.attribute import AttributeBuildingSource
from ddx.change import DEFAULT_CONFIG, ChangeResult
from ddx.geo import WGS84, reproject


def flat_change(grid, score_value=0.0, **layers):
    """A ChangeResult with uniform layers, for exercising the rollup."""
    shape = grid.shape
    base = {
        "magnitude": np.zeros(shape, "float32"),
        "structure_loss": np.zeros(shape, "float32"),
        "brightness_delta": np.zeros(shape, "float32"),
        "ndvi_delta": np.zeros(shape, "float32"),
        "flooded": np.zeros(shape, bool),
        "vegetation_loss": np.zeros(shape, bool),
    }
    base.update(layers)
    return ChangeResult(
        grid=grid, valid=np.ones(shape, bool),
        score=np.full(shape, score_value, "float32"), config=DEFAULT_CONFIG,
        diagnostics={"synthetic": False}, **base)


def test_focus_window_scales_with_structure_and_resolution():
    assert focus_window_px(120, 1.0) > focus_window_px(120, 4.0)
    assert focus_window_px(400, 1.0) >= focus_window_px(100, 1.0)
    assert focus_window_px(1, 1.0) >= 3
    assert focus_window_px(1e6, 0.1) <= 25


def test_developed_core_shrinks_only_large_parcels(store, sample_bbox):
    parcel = store.in_bbox(sample_bbox)[0]
    utm = reproject(parcel.geometry, WGS84.to_string(), "EPSG:32617")
    small = developed_core(utm, 150.0)
    assert small.area <= utm.area
    big = utm.buffer(60)      # push it well over the large-parcel threshold
    focused = developed_core(big, 150.0)
    assert focused.area < big.area * 0.6


def test_quiet_scene_reports_no_damage(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    result = assess(flat_change(grid, 0.0), parcels, buildings)
    assert result.summary["buildings"] == len(buildings)
    assert result.summary["buildings_damaged"] == 0
    assert all(a.worst_class == "none" for a in result.parcels)
    assert result.summary["estimated_loss_usd"] == 0.0


def test_uniform_severe_change_flags_every_structure(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    result = assess(flat_change(grid, 0.9), parcels, buildings)
    assert result.summary["buildings_damaged"] == len(buildings)
    assert result.summary["buildings_by_class"]["destroyed"] == len(buildings)
    # Every parcel with an appraised building value contributes a loss estimate.
    assert result.summary["estimated_loss_usd"] > 0


def test_damage_is_localised_to_the_parcel_it_touches(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    target = parcels[0]
    score = np.zeros(grid.shape, "float32")
    from ddx.zonal import rasterize_labels
    hit = rasterize_labels([reproject(target.geometry, WGS84.to_string(), grid.crs)],
                           grid, all_touched=True)
    score[hit == 1] = 0.95
    result = assess(flat_change(grid, 0.0, ), parcels, buildings)
    result_hit = assess(
        ChangeResult(grid=grid, valid=np.ones(grid.shape, bool), score=score,
                     magnitude=np.zeros(grid.shape, "float32"),
                     structure_loss=np.zeros(grid.shape, "float32"),
                     brightness_delta=np.zeros(grid.shape, "float32"),
                     config=DEFAULT_CONFIG, diagnostics={}),
        parcels, buildings)
    damaged = [a for a in result_hit.parcels if a.worst_class != "none"]
    assert [a.parcel.id for a in damaged] == [target.id]
    assert result.summary["buildings_damaged"] == 0


def test_flood_fraction_is_reported(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    flooded = np.ones(grid.shape, bool)
    result = assess(flat_change(grid, 0.1, flooded=flooded), parcels, buildings)
    assert all(a.flooded_fraction == pytest.approx(1.0) for a in result.parcels)
    assert result.summary["parcels_flooded"] == len(parcels)


def test_approximate_footprints_produce_a_warning(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    result = assess(flat_change(grid), parcels, buildings)
    assert any("approximate" in w for w in result.warnings)
    assert result.summary["footprints_approximate"] is True


def test_loss_factors_are_monotonic():
    order = ["none", "possible", "moderate", "severe", "destroyed"]
    values = [LOSS_FACTOR[c] for c in order]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_feature_collections_are_well_formed(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    result = assess(flat_change(grid, 0.5), parcels, buildings)
    parcel_fc = result.feature_collection()
    assert parcel_fc["type"] == "FeatureCollection"
    assert len(parcel_fc["features"]) == len(parcels)
    assert parcel_fc["features"][0]["geometry"] is not None
    assert result.feature_collection(geometry=False)["features"][0]["geometry"] is None
    building_fc = result.building_feature_collection()
    assert len(building_fc["features"]) == len(buildings)
    assert "damage_class" in building_fc["features"][0]["properties"]
