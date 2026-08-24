"""Footprints approximated from county tax-roll attributes.

Lee County's parcel table records heated living area (``dwel_SFLA``) and
outbuilding area (``ob_AREA``) but no footprint geometry. That is enough to
know *how many* structures a parcel holds and roughly how big they are, which
is what parcel-level damage counting needs, but the placement is a guess.
Everything produced here is flagged ``approximate=True`` so the assessment
samples the parcel's developed core rather than pretending to know where the
roof is.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from shapely.affinity import rotate, translate
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from ..geo import WGS84, BBox, reproject, utm_crs_for
from .base import DWELLING, OUTBUILDING, Building, BuildingSet

SQFT_TO_M2 = 0.09290304

# Living area over footprint area, by dwelling style. A two-storey colonial
# reports twice the heated area its roof covers; a ranch reports about one.
STOREY_FACTOR: dict[str, float] = {
    "RANCH": 1.0,
    "CONVENTIONAL": 1.0,
    "MOBILE HOME": 1.0,
    "SINGLE WIDE": 1.0,
    "DOUBLE WIDE": 1.0,
    "MODULAR": 1.0,
    "CONTEMPORARY": 1.15,
    "SPLIT LEVEL": 1.4,
    "CAPE COD": 1.5,
    "CONDO / TOWNHOUSE": 1.6,
    "COLONIAL": 1.9,
    "TWO STORY": 1.9,
    "OLD STYLE": 1.6,
}
DEFAULT_STOREY_FACTOR = 1.2

# ob_DESCRIB values that are site improvements, not structures. Counting a
# parking lot as a damaged building would be worse than not counting it.
NON_BUILDING_TOKENS = (
    "PAVING", "ASPHALT", "CONCRETE", "FENCE", "WELL", "SEPTIC", "POOL",
    "HOMESITE", "M.H. SPACES", "SPACES", "TENNIS", "DRIVE", "WALK", "CANOPY LT",
    "LIGHT POLE", "SIGN", "TANK", "SILO PIT", "LAGOON",
)


def is_building_description(desc: str | None) -> bool:
    if not desc:
        return False
    upper = desc.upper()
    return not any(tok in upper for tok in NON_BUILDING_TOKENS)


def footprint_m2_from_living_area(sqft: float, style: str | None) -> float:
    factor = STOREY_FACTOR.get((style or "").strip().upper(), DEFAULT_STOREY_FACTOR)
    return max(20.0, sqft * SQFT_TO_M2 / factor)


def _rect(centre: Point, area: float, aspect: float, angle: float) -> BaseGeometry:
    """An area-preserving rectangle centred on a point, in metres."""
    w = math.sqrt(area * aspect)
    h = area / w
    rect = box(centre.x - w / 2, centre.y - h / 2, centre.x + w / 2, centre.y + h / 2)
    return rotate(rect, angle, origin=centre, use_radians=False)


def _stable_jitter(seed: int, spread: float) -> tuple[float, float, float]:
    """Deterministic pseudo-random offset and rotation from an integer seed."""
    h = (seed * 2654435761) & 0xFFFFFFFF
    dx = ((h & 0xFF) / 255.0 - 0.5) * 2 * spread
    dy = (((h >> 8) & 0xFF) / 255.0 - 0.5) * 2 * spread
    angle = ((h >> 16) & 0xFF) / 255.0 * 180.0
    return dx, dy, angle


def buildings_for_parcel(parcel: Any, jitter: float = 0.0) -> list[Building]:
    """Approximate footprints for one parcel from its tax-roll attributes."""
    geom = parcel.geometry
    if geom is None or geom.is_empty:
        return []
    centre_ll = geom.representative_point()
    utm = utm_crs_for(centre_ll.x, centre_ll.y)
    parcel_utm = reproject(geom, WGS84, utm)
    centre = parcel_utm.representative_point()
    parcel_area = parcel_utm.area

    out: list[Building] = []
    specs: list[tuple[str, float, str | None, int | None]] = []

    if parcel.dwel_sfla and parcel.dwel_sfla > 0:
        specs.append((
            DWELLING,
            footprint_m2_from_living_area(float(parcel.dwel_sfla), parcel.dwel_desc),
            parcel.dwel_desc,
            parcel.dwel_yrblt,
        ))
    if parcel.ob_area and parcel.ob_area > 0 and is_building_description(parcel.ob_desc):
        specs.append((
            OUTBUILDING,
            max(10.0, float(parcel.ob_area) * SQFT_TO_M2),
            parcel.ob_desc,
            parcel.ob_yrblt,
        ))

    for i, (kind, area, desc, year) in enumerate(specs):
        # Never let a guessed footprint spill outside its own parcel.
        area = min(area, max(20.0, parcel_area * 0.6))
        dx, dy, angle = _stable_jitter((parcel.id or 0) * 7 + i * 13, jitter)
        # Outbuildings sit behind the dwelling; nudge them off the centre.
        if kind == OUTBUILDING:
            dx += math.sqrt(area) * 1.6
            dy -= math.sqrt(area) * 1.2
        pt = Point(centre.x + dx, centre.y + dy)
        rect = _rect(pt, area, aspect=1.35, angle=angle)
        if not parcel_utm.contains(rect.centroid):
            rect = translate(rect, centre.x - rect.centroid.x, centre.y - rect.centroid.y)
        # Keep the drawn outline inside the parcel, but keep reporting the
        # tax-roll area: the clip is a display nicety, not a measurement.
        if not parcel_utm.contains(rect):
            clipped = rect.intersection(parcel_utm)
            if not clipped.is_empty and clipped.area >= 0.4 * rect.area:
                rect = clipped
        out.append(Building(
            id=f"p{parcel.id}-{kind[:3]}{i}",
            parcel_id=parcel.id,
            geometry=reproject(rect, utm, WGS84),
            source="tax-roll",
            kind=kind,
            approximate=True,
            area_m2=area,
            description=desc,
            year_built=year,
            attrs={"living_area_sqft": parcel.dwel_sfla if kind == DWELLING else None},
        ))
    return out


class AttributeBuildingSource:
    """Derives approximate footprints from parcel attributes. Always available."""

    name = "tax-roll"
    approximate = True

    def __init__(self, jitter: float = 0.0):
        self.jitter = jitter

    def available(self) -> bool:
        return True

    def fetch(self, bbox: BBox, parcels: Sequence[Any]) -> BuildingSet:
        buildings: list[Building] = []
        for parcel in parcels:
            buildings.extend(buildings_for_parcel(parcel, jitter=self.jitter))
        return BuildingSet(
            buildings=buildings,
            source=self.name,
            approximate=True,
            note=("Structure counts come from the county tax roll (dwelling living "
                  "area and outbuilding area). Footprint locations are approximate, "
                  "so damage is scored over each parcel's developed core rather than "
                  "an exact roof outline."),
        )
