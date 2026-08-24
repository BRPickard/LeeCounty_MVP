"""Geometry and CRS helpers.

All geometry that crosses an API boundary is WGS84 (EPSG:4326) lon/lat.
Anything that needs metres (areas, footprint sizes, raster grids) is done in a
local UTM zone picked from the AOI centroid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

WGS84 = CRS.from_epsg(4326)

BBox = tuple[float, float, float, float]  # west, south, east, north


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def reproject(geom: BaseGeometry, src: str, dst: str) -> BaseGeometry:
    """Reproject a shapely geometry between two CRS given as strings."""
    if str(src) == str(dst):
        return geom
    tf = _transformer(str(src), str(dst))
    return shapely_transform(lambda x, y, z=None: tf.transform(x, y), geom)


def utm_crs_for(lon: float, lat: float) -> CRS:
    """Return the UTM CRS covering a point, north or south as appropriate."""
    zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def utm_crs_for_bbox(bbox: BBox) -> CRS:
    w, s, e, n = bbox
    return utm_crs_for((w + e) / 2.0, (s + n) / 2.0)


def bbox_of(geom: BaseGeometry) -> BBox:
    minx, miny, maxx, maxy = geom.bounds
    return (minx, miny, maxx, maxy)


def bbox_geom(bbox: BBox) -> BaseGeometry:
    w, s, e, n = bbox
    return box(w, s, e, n)


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        raise ValueError("union_bbox needs at least one bbox")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def buffer_bbox(bbox: BBox, metres: float) -> BBox:
    """Grow a WGS84 bbox by an approximate number of metres on every side."""
    w, s, e, n = bbox
    lat = (s + n) / 2.0
    dlat = metres / 111_320.0
    dlon = metres / max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    return (w - dlon, s - dlat, e + dlon, n + dlat)


def bbox_area_km2(bbox: BBox) -> float:
    """Approximate area of a WGS84 bbox in square kilometres."""
    w, s, e, n = bbox
    lat = math.radians((s + n) / 2.0)
    width_km = (e - w) * 111.320 * math.cos(lat)
    height_km = (n - s) * 110.574
    return abs(width_km * height_km)


def area_m2(geom_wgs84: BaseGeometry) -> float:
    """Area of a WGS84 geometry in square metres, via the local UTM zone."""
    if geom_wgs84.is_empty:
        return 0.0
    c = geom_wgs84.centroid
    utm = utm_crs_for(c.x, c.y)
    return float(reproject(geom_wgs84, WGS84, utm).area)


def valid(geom: BaseGeometry) -> BaseGeometry:
    """Best-effort repair of self-intersecting rings from source data."""
    if geom.is_valid:
        return geom
    fixed = geom.buffer(0)
    return fixed if not fixed.is_empty else geom


def to_geojson_geometry(geom: BaseGeometry) -> dict:
    return mapping(geom)


def from_geojson_geometry(obj: dict) -> BaseGeometry:
    return shape(obj)


def reproject_many(geoms: Sequence[BaseGeometry], src: str, dst: str) -> list[BaseGeometry]:
    """Reproject many geometries in one pass instead of one call each.

    Transforming 30,000 parcels individually spends most of its time in
    per-call setup; bundling them into a collection cuts that to one.
    """
    kept = [g for g in geoms if g is not None and not g.is_empty]
    if not kept:
        return []
    return list(reproject(GeometryCollection(kept), src, dst).geoms)


def round_geometry(geom: BaseGeometry, ndigits: int = 6) -> BaseGeometry:
    """Drop coordinate precision to keep GeoJSON payloads small (~0.1 m at 6dp)."""
    return shapely_transform(lambda x, y, z=None: (round(x, ndigits), round(y, ndigits)), geom)


@dataclass(frozen=True)
class Grid:
    """An analysis raster grid in a projected CRS."""

    crs: str
    transform: tuple[float, float, float, float, float, float]  # affine a,b,c,d,e,f
    width: int
    height: int

    @property
    def gsd(self) -> float:
        return abs(self.transform[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def pixel_area_m2(self) -> float:
        return abs(self.transform[0] * self.transform[4])


def make_grid(bbox: BBox, gsd: float, crs: CRS | None = None) -> Grid:
    """Build a north-up analysis grid covering a WGS84 bbox at a given GSD."""
    crs = crs or utm_crs_for_bbox(bbox)
    geom = reproject(bbox_geom(bbox), WGS84, crs)
    minx, miny, maxx, maxy = geom.bounds
    # snap outward to whole pixels so repeated runs share pixel edges
    minx = math.floor(minx / gsd) * gsd
    miny = math.floor(miny / gsd) * gsd
    maxx = math.ceil(maxx / gsd) * gsd
    maxy = math.ceil(maxy / gsd) * gsd
    width = max(1, int(round((maxx - minx) / gsd)))
    height = max(1, int(round((maxy - miny) / gsd)))
    return Grid(crs=crs.to_string(), transform=(gsd, 0.0, minx, 0.0, -gsd, maxy),
                width=width, height=height)


def grid_bounds(grid: Grid) -> tuple[float, float, float, float]:
    a, _, c, _, e, f = grid.transform
    return (c, f + e * grid.height, c + a * grid.width, f)


def grid_bbox_wgs84(grid: Grid) -> BBox:
    return bbox_of(reproject(bbox_geom(grid_bounds(grid)), grid.crs, WGS84))


def parse_bbox(text: str | Sequence[float]) -> BBox:
    """Parse a 'w,s,e,n' string or 4-sequence into a validated bbox."""
    if isinstance(text, str):
        parts = [p.strip() for p in text.split(",")]
    else:
        parts = list(text)
    if len(parts) != 4:
        raise ValueError("bbox must have 4 values: west,south,east,north")
    w, s, e, n = (float(p) for p in parts)
    if w > e:
        w, e = e, w
    if s > n:
        s, n = n, s
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        raise ValueError("bbox out of WGS84 range")
    return (w, s, e, n)
