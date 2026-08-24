"""Footprints from OpenStreetMap via Overpass.

Useful when no authoritative footprint layer is at hand. Coverage varies by
area, so the returned set reports how many footprints landed on parcels and
callers can fall back to tax-roll approximations if that looks thin.
"""
from __future__ import annotations

from typing import Any, Sequence

import requests
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from .. import config
from ..geo import BBox, valid
from .base import STRUCTURE, Building, BuildingSet

QUERY = """
[out:json][timeout:{timeout}];
(
  way["building"]({south},{west},{north},{east});
  relation["building"]({south},{west},{north},{east});
);
out geom;
"""


class OSMBuildingSource:
    """Fetch building=* ways/relations from an Overpass endpoint."""

    name = "openstreetmap"
    approximate = False

    def __init__(self, url: str | None = None, timeout: int = 90):
        self.url = url or config.OVERPASS_URL
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.url)

    def _polygons(self, element: dict) -> list[Polygon]:
        if element.get("type") == "way":
            geom = element.get("geometry") or []
            if len(geom) >= 4:
                return [Polygon([(p["lon"], p["lat"]) for p in geom])]
            return []
        rings: list[Polygon] = []
        for member in element.get("members", []):
            if member.get("role") not in ("outer", "", None):
                continue
            geom = member.get("geometry") or []
            if len(geom) >= 4:
                rings.append(Polygon([(p["lon"], p["lat"]) for p in geom]))
        return rings

    def fetch(self, bbox: BBox, parcels: Sequence[Any]) -> BuildingSet:
        west, south, east, north = bbox
        body = QUERY.format(south=south, west=west, north=north, east=east,
                            timeout=self.timeout)
        resp = requests.post(self.url, data={"data": body}, timeout=self.timeout + 15)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])

        parcel_geoms = [p.geometry for p in parcels if p.geometry is not None]
        parcel_ids = [p.id for p in parcels if p.geometry is not None]
        tree = STRtree(parcel_geoms) if parcel_geoms else None

        buildings: list[Building] = []
        for element in elements:
            tags = element.get("tags") or {}
            for j, poly in enumerate(self._polygons(element)):
                if poly.is_empty or poly.area <= 0:
                    continue
                poly = valid(poly)
                parcel_id = None
                if tree is not None:
                    best = 0.0
                    for idx in tree.query(poly):
                        candidate = parcel_geoms[idx]
                        if not candidate.intersects(poly):
                            continue
                        overlap = candidate.intersection(poly).area
                        if overlap > best:
                            best, parcel_id = overlap, parcel_ids[idx]
                buildings.append(Building(
                    id=f"osm-{element.get('type','w')}{element.get('id')}-{j}",
                    parcel_id=parcel_id,
                    geometry=poly,
                    source=self.name,
                    kind=STRUCTURE,
                    approximate=False,
                    description=tags.get("building") if tags.get("building") != "yes" else None,
                    year_built=_as_int(tags.get("start_date")),
                    attrs={k: v for k, v in tags.items() if k in ("name", "amenity", "levels")},
                ))
        matched = sum(1 for b in buildings if b.parcel_id is not None)
        return BuildingSet(
            buildings=buildings,
            source=self.name,
            approximate=False,
            note=(f"OpenStreetMap footprints ({matched} of {len(buildings)} matched to a "
                  "parcel). OSM building coverage is uneven outside towns."),
        )


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None
