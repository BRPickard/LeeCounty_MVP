"""Building footprint sources."""
from __future__ import annotations

from typing import Any, Sequence

from .. import config
from ..geo import BBox
from .attribute import AttributeBuildingSource
from .base import (DWELLING, OUTBUILDING, STRUCTURE, Building, BuildingSet,
                   BuildingSource)
from .osm import OSMBuildingSource
from .vector import VectorBuildingSource

__all__ = [
    "Building", "BuildingSet", "BuildingSource", "DWELLING", "OUTBUILDING",
    "STRUCTURE", "AttributeBuildingSource", "OSMBuildingSource",
    "VectorBuildingSource", "get_source", "list_sources", "fetch_buildings",
]

# Minimum OSM footprints per 100 parcels before we trust OSM over the tax roll.
OSM_COVERAGE_THRESHOLD = 25.0


def get_source(name: str, **kwargs: Any) -> BuildingSource:
    name = (name or "auto").lower()
    if name in ("tax-roll", "attribute", "parcel"):
        return AttributeBuildingSource(**kwargs)
    if name in ("file", "vector", "footprint-file"):
        path = kwargs.pop("path", None) or config.BUILDING_FILE
        if not path:
            raise ValueError("building source 'file' needs DDX_BUILDING_FILE or path=")
        return VectorBuildingSource(path, **kwargs)
    if name in ("osm", "openstreetmap", "overpass"):
        return OSMBuildingSource(**kwargs)
    raise ValueError(f"unknown building source: {name}")


def list_sources() -> list[dict[str, Any]]:
    """Describe the sources this deployment can use, for the UI's picker."""
    out = [{
        "id": "auto",
        "label": "Auto (best available)",
        "approximate": not bool(config.BUILDING_FILE),
        "available": True,
    }, {
        "id": "tax-roll",
        "label": "County tax roll (approximate footprints)",
        "approximate": True,
        "available": True,
    }]
    file_source = VectorBuildingSource(config.BUILDING_FILE) if config.BUILDING_FILE else None
    out.append({
        "id": "file",
        "label": f"Footprint file ({config.BUILDING_FILE or 'not configured'})",
        "approximate": False,
        "available": bool(file_source and file_source.available()),
    })
    out.append({
        "id": "osm",
        "label": "OpenStreetMap footprints",
        "approximate": False,
        "available": bool(config.OVERPASS_URL),
    })
    return out


def fetch_buildings(bbox: BBox, parcels: Sequence[Any], source: str = "auto",
                    **kwargs: Any) -> BuildingSet:
    """Fetch footprints, falling back to tax-roll approximations when needed.

    ``auto`` prefers a configured footprint file, then OSM if it returns
    meaningful coverage, then the tax roll.
    """
    source = (source or "auto").lower()
    if source != "auto":
        return get_source(source, **kwargs).fetch(bbox, parcels)

    if config.BUILDING_FILE:
        candidate = VectorBuildingSource(config.BUILDING_FILE)
        if candidate.available():
            try:
                return candidate.fetch(bbox, parcels)
            except Exception as exc:  # fall through to the next source
                _warn("footprint file", exc)

    if config.OVERPASS_URL and parcels:
        try:
            osm = OSMBuildingSource().fetch(bbox, parcels)
            per_100 = 100.0 * len(osm) / max(1, len(parcels))
            if per_100 >= OSM_COVERAGE_THRESHOLD:
                return osm
        except Exception as exc:
            _warn("OpenStreetMap", exc)

    return AttributeBuildingSource().fetch(bbox, parcels)


def _warn(what: str, exc: Exception) -> None:
    import logging
    logging.getLogger(__name__).warning("%s unavailable (%s); falling back", what, exc)
