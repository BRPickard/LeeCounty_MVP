import numpy as np
import pytest

from ddx.geo import make_grid
from ddx.imagery.raster import (auto_scale, is_remote, read_band_to_grid,
                                write_geotiff)


@pytest.fixture
def checkerboard(tmp_path, sample_bbox):
    grid = make_grid(sample_bbox, 1.0)
    yy, xx = np.mgrid[0:grid.height, 0:grid.width]
    data = ((xx // 20 + yy // 20) % 2 * 200 + 30).astype("uint8")
    path = tmp_path / "board.tif"
    write_geotiff(str(path), data, grid, "uint8")
    return path, grid


def test_scale_guessing():
    assert auto_scale("uint8") == 255.0
    assert auto_scale("uint16") == 10000.0
    assert auto_scale("float32") == 1.0
    assert auto_scale("uint16", declared=2.0) == 2.0


def test_remote_detection():
    assert is_remote("https://example.invalid/x.tif")
    assert is_remote("s3://bucket/x.tif")
    assert not is_remote("/local/x.tif")


def test_read_onto_the_same_grid_preserves_values(checkerboard):
    path, grid = checkerboard
    values, valid = read_band_to_grid(str(path), grid)
    assert valid.all()
    assert values.min() == pytest.approx(30 / 255, abs=0.02)
    assert values.max() == pytest.approx(230 / 255, abs=0.02)


def test_read_onto_a_coarser_offset_grid(checkerboard, sample_bbox):
    path, _ = checkerboard
    west, south, east, north = sample_bbox
    inner = make_grid((west + 0.0002, south + 0.0002, east - 0.0002, north - 0.0002), 4.0)
    values, valid = read_band_to_grid(str(path), inner)
    assert valid.mean() > 0.99
    assert 0.05 < values[valid].mean() < 0.95


def test_partial_overlap_marks_the_gap_invalid(checkerboard, sample_bbox):
    path, _ = checkerboard
    west, south, east, north = sample_bbox
    shifted = make_grid((west + (east - west) / 2, south, east + (east - west) / 2, north), 2.0)
    _, valid = read_band_to_grid(str(path), shifted)
    assert 0.2 < valid.mean() < 0.8


def test_disjoint_extent_returns_all_invalid(checkerboard):
    path, _ = checkerboard
    far = make_grid((-100.0, 20.0, -99.99, 20.01), 2.0)
    values, valid = read_band_to_grid(str(path), far)
    assert not valid.any()
    assert not values.any()


def test_nodata_pixels_are_excluded(tmp_path, sample_bbox):
    grid = make_grid(sample_bbox, 1.0)
    data = np.full(grid.shape, 7, dtype="uint8")
    blanked = grid.height // 4
    data[:blanked, :] = 0
    path = tmp_path / "nd.tif"
    write_geotiff(str(path), data, grid, "uint8", nodata=0)
    _, valid = read_band_to_grid(str(path), grid)
    assert valid.mean() == pytest.approx(1 - blanked / grid.height, abs=0.02)


def test_multiband_write_and_band_selection(tmp_path, sample_bbox):
    grid = make_grid(sample_bbox, 2.0)
    data = np.stack([np.full(grid.shape, v, "uint8") for v in (10, 100, 200)])
    path = tmp_path / "rgb.tif"
    write_geotiff(str(path), data, grid, "uint8", band_names=["red", "green", "blue"])
    for index, expected in enumerate((10, 100, 200), start=1):
        values, _ = read_band_to_grid(str(path), grid, band_index=index)
        assert values.mean() == pytest.approx(expected / 255, abs=0.01)
