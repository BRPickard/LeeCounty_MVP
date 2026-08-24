import numpy as np
import pytest
from shapely.geometry import box

from ddx.zonal import ZonalIndex, rasterize_labels, stats_table


@pytest.fixture
def labels():
    grid = np.zeros((6, 6), dtype="int32")
    grid[0:3, 0:3] = 1
    grid[0:2, 4:6] = 2
    grid[4:6, 1:4] = 3
    return grid


def test_counts_and_means(labels):
    values = np.arange(36, dtype="float32").reshape(6, 6)
    index = ZonalIndex(labels)
    assert list(index.counts) == [9, 4, 6]
    means = index.mean(values)
    assert means[0] == pytest.approx(values[0:3, 0:3].mean())
    assert means[1] == pytest.approx(values[0:2, 4:6].mean())


def test_valid_mask_excludes_pixels(labels):
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 0] = False
    index = ZonalIndex(labels, valid)
    assert index.counts[0] == 8


def test_quantiles_and_max(labels):
    values = np.zeros((6, 6), dtype="float32")
    values[0:3, 0:3] = np.arange(9).reshape(3, 3)
    index = ZonalIndex(labels)
    assert index.quantile(values, 0.5)[0] == pytest.approx(4.0)
    assert index.max(values)[0] == pytest.approx(8.0)
    assert index.quantile(values, 0.0)[0] == pytest.approx(0.0)


def test_fraction_of_boolean_layer(labels):
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:3, 0:2] = True          # 6 of zone 1's 9 pixels
    index = ZonalIndex(labels)
    assert index.fraction(mask)[0] == pytest.approx(6 / 9)
    assert index.fraction(mask)[1] == pytest.approx(0.0)


def test_empty_zone_yields_nan(labels):
    labels = labels.copy()
    labels[labels == 2] = 0        # zone 2 disappears but label 3 still exists
    index = ZonalIndex(labels)
    means = index.mean(np.ones((6, 6), dtype="float32"))
    assert np.isnan(means[1])
    assert means[2] == pytest.approx(1.0)


def test_rasterize_labels_matches_geometry(grid):
    from ddx.geo import WGS84, grid_bbox_wgs84, reproject
    west, south, east, north = grid_bbox_wgs84(grid)
    half = box(west, south, (west + east) / 2, north)
    projected = reproject(half, WGS84.to_string(), grid.crs)
    labels = rasterize_labels([projected], grid)
    coverage = (labels == 1).mean()
    assert 0.4 < coverage < 0.6


def test_stats_table_includes_quantiles(labels):
    values = np.arange(36, dtype="float32").reshape(6, 6)
    index = ZonalIndex(labels)
    table = stats_table(index, {"score": values},
                        quantiles=[("score_p90", "score", 0.9)])
    assert "pixels" in table and "score" in table and "score_p90" in table
    assert table["score_p90"][0] >= table["score"][0]
