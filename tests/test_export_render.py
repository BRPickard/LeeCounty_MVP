import csv
import io

import numpy as np
import pytest

from ddx.assess import assess
from ddx.buildings.attribute import AttributeBuildingSource
from ddx.change import DEFAULT_CONFIG, ChangeResult
from ddx.export import buildings_csv, parcels_csv
from ddx.render import (damage_ramp, downsample, encode_png, side_by_side,
                        stretch)


@pytest.fixture
def result(store, sample_bbox, grid):
    parcels = store.in_bbox(sample_bbox)
    buildings = AttributeBuildingSource().fetch(sample_bbox, parcels)
    shape = grid.shape
    change = ChangeResult(
        grid=grid, valid=np.ones(shape, bool),
        score=np.full(shape, 0.85, "float32"),
        magnitude=np.full(shape, 0.2, "float32"),
        structure_loss=np.full(shape, 0.4, "float32"),
        brightness_delta=np.zeros(shape, "float32"),
        flooded=np.zeros(shape, bool),
        vegetation_loss=np.zeros(shape, bool),
        config=DEFAULT_CONFIG, diagnostics={})
    return assess(change, parcels, buildings)


def test_parcel_csv_has_a_row_per_parcel(result):
    rows = list(csv.DictReader(io.StringIO(parcels_csv(result))))
    assert len(rows) == len(result.parcels)
    assert rows[0]["Parcel damage class"] in (
        "none", "possible", "moderate", "severe", "destroyed")
    assert "PIN" in rows[0] and "Estimated structure loss (USD)" in rows[0]


def test_parcel_csv_is_sorted_worst_first(result):
    rows = list(csv.DictReader(io.StringIO(parcels_csv(result))))
    order = ["destroyed", "severe", "moderate", "possible", "none"]
    ranks = [order.index(r["Parcel damage class"]) for r in rows]
    assert ranks == sorted(ranks)


def test_building_csv_has_a_row_per_structure(result):
    rows = list(csv.DictReader(io.StringIO(buildings_csv(result))))
    total = sum(len(a.buildings) for a in result.parcels)
    assert len(rows) == total
    assert rows[0]["Footprint approximate"] == "yes"


def test_png_encoding_shapes():
    for array in (np.zeros((4, 5), "uint8"),
                  np.zeros((4, 5, 3), "uint8"),
                  np.zeros((4, 5, 4), "uint8")):
        blob = encode_png(array)
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 5, 2), "uint8"))


def test_stretch_spans_the_display_range():
    rng = np.random.default_rng(0)
    rgb = rng.random((30, 30, 3)).astype("float32") * 0.3
    out = stretch(rgb)
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0


def test_stretch_blacks_out_invalid_pixels():
    rgb = np.full((10, 10, 3), 0.5, "float32")
    valid = np.ones((10, 10), bool)
    valid[:5] = False
    out = stretch(rgb, valid)
    assert (out[:5] == 0).all()


def test_damage_ramp_is_transparent_where_undamaged():
    score = np.array([[0.0, 0.5, 0.99]], dtype="float32")
    rgba = damage_ramp(score, DEFAULT_CONFIG)
    assert rgba[0, 0, 3] == 0          # class "none" is see-through
    assert rgba[0, 2, 3] > 0
    assert tuple(rgba[0, 2, :3]) == (0x86, 0x2e, 0x9c)   # destroyed purple


def test_side_by_side_matches_heights_and_channels():
    joined = side_by_side([np.zeros((10, 4, 3), "uint8"),
                           np.zeros((12, 6, 4), "uint8")])
    assert joined.shape[0] == 10
    assert joined.shape[2] == 4
    with pytest.raises(ValueError):
        side_by_side([])


def test_downsample_caps_the_long_edge():
    assert downsample(np.zeros((3000, 1000, 3), "uint8"), 1000).shape[0] <= 1000
