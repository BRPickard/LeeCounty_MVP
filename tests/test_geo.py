import math

import pytest
from shapely.geometry import Point, box

from ddx.geo import (WGS84, area_m2, bbox_area_km2, buffer_bbox, grid_bbox_wgs84,
                     make_grid, parse_bbox, reproject, utm_crs_for, valid)


def test_utm_zone_selection():
    assert utm_crs_for(-79.17, 35.47).to_epsg() == 32617   # zone 17 north
    assert utm_crs_for(-79.17, -35.47).to_epsg() == 32717  # southern hemisphere
    assert utm_crs_for(2.35, 48.85).to_epsg() == 32631     # Paris, zone 31


def test_reprojection_round_trips():
    point = Point(-79.17, 35.47)
    utm = utm_crs_for(point.x, point.y).to_string()
    there = reproject(point, WGS84.to_string(), utm)
    back = reproject(there, utm, WGS84.to_string())
    assert back.x == pytest.approx(point.x, abs=1e-9)
    assert back.y == pytest.approx(point.y, abs=1e-9)


def test_area_matches_known_acreage():
    # 0.01 degree of latitude is about 1110 m; check the area is in range.
    geom = box(-79.17, 35.47, -79.16, 35.48)
    area = area_m2(geom)
    assert 800_000 < area < 1_100_000


def test_grid_covers_bbox_and_snaps_to_pixels():
    bbox = (-79.19, 35.47, -79.17, 35.49)
    grid = make_grid(bbox, 2.0)
    assert grid.gsd == 2.0
    assert grid.pixel_area_m2 == pytest.approx(4.0)
    covered = grid_bbox_wgs84(grid)
    assert covered[0] <= bbox[0] and covered[1] <= bbox[1]
    assert covered[2] >= bbox[2] and covered[3] >= bbox[3]
    # Origin sits on a whole multiple of the pixel size.
    assert grid.transform[2] % 2.0 == pytest.approx(0.0)


def test_grid_is_stable_between_calls():
    bbox = (-79.19, 35.47, -79.17, 35.49)
    assert make_grid(bbox, 2.0) == make_grid(bbox, 2.0)


def test_bbox_area_and_buffer():
    bbox = (-79.19, 35.47, -79.17, 35.49)
    assert bbox_area_km2(bbox) == pytest.approx(4.0, rel=0.1)
    grown = buffer_bbox(bbox, 100.0)
    assert grown[0] < bbox[0] and grown[3] > bbox[3]
    assert bbox_area_km2(grown) > bbox_area_km2(bbox)


def test_parse_bbox_orders_and_validates():
    assert parse_bbox("-79.1,35.5,-79.2,35.4") == (-79.2, 35.4, -79.1, 35.5)
    assert parse_bbox([1, 2, 3, 4]) == (1, 2, 3, 4)
    with pytest.raises(ValueError):
        parse_bbox("1,2,3")
    with pytest.raises(ValueError):
        parse_bbox("-200,0,10,10")


def test_valid_repairs_self_intersection():
    bowtie = valid(__import__("shapely.geometry", fromlist=["Polygon"]).Polygon(
        [(0, 0), (1, 1), (1, 0), (0, 1)]))
    assert bowtie.is_valid
