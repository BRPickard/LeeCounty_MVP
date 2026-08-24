"""Real footprints loaded from a vector file.

Point this at Microsoft Building Footprints for North Carolina, an OSM
extract, or a county structures layer. Anything OGR can read works; footprints
are spatially joined to parcels by greatest overlap.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

from shapely.geometry import shape as shapely_shape
from shapely.strtree import STRtree

from ..geo import WGS84, BBox, bbox_geom, reproject, valid
from .base import STRUCTURE, Building, BuildingSet

BUILDING_KEYS = ("building", "BUILDING", "type", "TYPE", "class", "CLASS")
HEIGHT_KEYS = ("height", "HEIGHT", "HGT")
YEAR_KEYS = ("year_built", "YEAR_BUILT", "YRBLT", "release")


def _iter_geojson(path: Path) -> Iterator[tuple[Any, dict]]:
    text = path.read_text()
    # GeoJSONSeq / newline-delimited GeoJSON, as Microsoft publishes it.
    stripped = text.lstrip()
    if not stripped.startswith("{") or '"FeatureCollection"' not in stripped[:400]:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                continue
            if feat.get("geometry"):
                yield feat["geometry"], feat.get("properties") or {}
        return
    doc = json.loads(text)
    for feat in doc.get("features", []):
        if feat.get("geometry"):
            yield feat["geometry"], feat.get("properties") or {}


def _iter_shapefile(path: Path) -> Iterator[tuple[Any, dict]]:
    import shapefile  # pyshp
    from pyproj import CRS

    reader = shapefile.Reader(str(path))
    prj = path.with_suffix(".prj")
    src_crs = CRS.from_wkt(prj.read_text()) if prj.exists() else WGS84
    needs_reproject = src_crs.to_epsg() != 4326
    for srec in reader.iterShapeRecords():
        geo = srec.shape.__geo_interface__
        if not geo or not geo.get("coordinates"):
            continue
        geom = shapely_shape(geo)
        if needs_reproject:
            geom = reproject(geom, src_crs.to_string(), WGS84.to_string())
        yield geom, srec.record.as_dict()


def iter_features(path: Path) -> Iterator[tuple[Any, dict]]:
    """Yield (geometry-or-geojson, properties) pairs from a footprint file."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        target = path.parent / f"{path.stem}__unzipped"
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
        shps = sorted(target.rglob("*.shp"))
        if shps:
            yield from _iter_shapefile(shps[0])
            return
        for geo in sorted(target.rglob("*.geojson")) + sorted(target.rglob("*.json")):
            yield from _iter_geojson(geo)
        return
    if suffix in (".shp",):
        yield from _iter_shapefile(path)
        return
    yield from _iter_geojson(path)


def _first(props: dict, keys: Sequence[str]) -> Any:
    for k in keys:
        if k in props and props[k] not in (None, "", 0):
            return props[k]
    return None


class VectorBuildingSource:
    """Load footprints from a file and join them to parcels."""

    name = "footprint-file"
    approximate = False

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def available(self) -> bool:
        return self.path.exists()

    def fetch(self, bbox: BBox, parcels: Sequence[Any]) -> BuildingSet:
        if not self.available():
            raise FileNotFoundError(f"footprint file not found: {self.path}")
        clip = bbox_geom(bbox)
        parcel_geoms = [p.geometry for p in parcels if p.geometry is not None]
        parcel_ids = [p.id for p in parcels if p.geometry is not None]
        tree = STRtree(parcel_geoms) if parcel_geoms else None

        buildings: list[Building] = []
        for i, (geo, props) in enumerate(iter_features(self.path)):
            geom = geo if hasattr(geo, "geom_type") else shapely_shape(geo)
            if geom.is_empty or not geom.intersects(clip):
                continue
            geom = valid(geom)
            parcel_id = None
            if tree is not None:
                best_overlap = 0.0
                for idx in tree.query(geom):
                    candidate = parcel_geoms[idx]
                    if not candidate.intersects(geom):
                        continue
                    overlap = candidate.intersection(geom).area
                    if overlap > best_overlap:
                        best_overlap, parcel_id = overlap, parcel_ids[idx]
            buildings.append(Building(
                id=str(props.get("id") or props.get("OBJECTID") or f"fp{i}"),
                parcel_id=parcel_id,
                geometry=geom,
                source=self.name,
                kind=STRUCTURE,
                approximate=False,
                description=_first(props, BUILDING_KEYS),
                year_built=_first(props, YEAR_KEYS),
                attrs={"height": _first(props, HEIGHT_KEYS)},
            ))
        return BuildingSet(
            buildings=buildings,
            source=f"{self.name}:{self.path.name}",
            approximate=False,
            note=f"Mapped footprints from {self.path.name}, joined to parcels by overlap.",
        )
