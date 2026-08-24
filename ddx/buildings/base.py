"""Building footprint model and source interface.

Footprint quality drives what the assessment can honestly claim, so every
building carries the source it came from and whether its geometry is a real
surveyed/mapped footprint or an approximation derived from tax-roll
attributes. Downstream code branches on ``approximate`` rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from shapely.geometry.base import BaseGeometry

from ..geo import BBox, area_m2, to_geojson_geometry

# Structure kinds we distinguish in reporting.
DWELLING = "dwelling"
OUTBUILDING = "outbuilding"
STRUCTURE = "structure"      # mapped footprint of unknown use


@dataclass
class Building:
    """One structure associated with a parcel."""

    id: str
    parcel_id: int | None
    geometry: BaseGeometry = field(repr=False)
    source: str = "unknown"
    kind: str = STRUCTURE
    approximate: bool = False
    area_m2: float | None = None
    description: str | None = None
    year_built: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.area_m2 is None:
            self.area_m2 = area_m2(self.geometry)

    def as_feature(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        props: dict[str, Any] = {
            "id": self.id,
            "parcel_id": self.parcel_id,
            "source": self.source,
            "kind": self.kind,
            "approximate": self.approximate,
            "area_m2": round(self.area_m2 or 0.0, 1),
            "description": self.description,
            "year_built": self.year_built,
        }
        if extra:
            props.update(extra)
        return {
            "type": "Feature",
            "id": self.id,
            "properties": props,
            "geometry": to_geojson_geometry(self.geometry),
        }


@dataclass
class BuildingSet:
    """Buildings for an AOI plus provenance for the UI to display."""

    buildings: list[Building]
    source: str
    approximate: bool
    note: str = ""

    def __len__(self) -> int:
        return len(self.buildings)

    def by_parcel(self) -> dict[int | None, list[Building]]:
        out: dict[int | None, list[Building]] = {}
        for b in self.buildings:
            out.setdefault(b.parcel_id, []).append(b)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "approximate": self.approximate,
            "note": self.note,
            "count": len(self.buildings),
        }


class BuildingSource(Protocol):
    """Contract for anything that can supply footprints for an AOI."""

    name: str
    approximate: bool

    def available(self) -> bool: ...

    def fetch(self, bbox: BBox, parcels: Sequence[Any]) -> BuildingSet: ...
