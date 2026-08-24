"""Tabular exports of an assessment."""
from __future__ import annotations

import csv
import io
from typing import Any

from .assess import AssessmentResult
from .change import DAMAGE_CLASSES

PARCEL_COLUMNS = [
    ("pin", "PIN"),
    ("parid", "Parcel ID"),
    ("address", "Situs address"),
    ("owner", "Owner"),
    ("acres", "Acres"),
    ("damage_class", "Parcel damage class"),
    ("parcel_score", "Parcel damage score"),
    ("buildings_total", "Buildings (total)"),
    ("buildings_damaged", "Buildings damaged (moderate+)"),
    ("buildings_changed", "Buildings changed (any)"),
    *[(f"buildings_{c}", f"Buildings {c}") for c in DAMAGE_CLASSES],
    ("flooded_fraction", "Flooded fraction of parcel"),
    ("vegetation_loss_fraction", "Vegetation loss fraction"),
    ("changed_area_m2", "Changed area (m2)"),
    ("apr_bldg", "Appraised building value"),
    ("estimated_loss_usd", "Estimated structure loss (USD)"),
    ("coverage", "Usable pixel coverage"),
    ("approximate_footprints", "Footprints approximate"),
    ("dwel_desc", "Dwelling type"),
    ("dwel_yrblt", "Year built"),
    ("dwel_sfla", "Living area (sqft)"),
    ("lon", "Longitude"),
    ("lat", "Latitude"),
    ("tax_card", "Tax card URL"),
]

BUILDING_COLUMNS = [
    ("id", "Structure ID"),
    ("parcel_pin", "Parcel PIN"),
    ("parcel_address", "Situs address"),
    ("kind", "Structure kind"),
    ("description", "Description"),
    ("year_built", "Year built"),
    ("area_m2", "Footprint area (m2)"),
    ("damage_class", "Damage class"),
    ("score", "Damage score"),
    ("confidence", "Confidence"),
    ("approximate", "Footprint approximate"),
    ("pixels", "Pixels sampled"),
    ("flooded_fraction", "Flooded fraction"),
    ("source", "Footprint source"),
]


def _write(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([label for _, label in columns])
    for row in rows:
        writer.writerow([_fmt(row.get(key)) for key, _ in columns])
    return buffer.getvalue()


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return round(value, 4)
    return value


def parcels_csv(result: AssessmentResult) -> str:
    """One row per parcel, ordered worst damage first."""
    order = {c: i for i, c in enumerate(DAMAGE_CLASSES)}
    rows = []
    for assessment in sorted(result.parcels,
                             key=lambda a: (-order.get(a.worst_class, 0),
                                            -(a.parcel_score or 0))):
        props = assessment.as_properties()
        counts = props.pop("building_counts", {})
        for cls in DAMAGE_CLASSES:
            props[f"buildings_{cls}"] = counts.get(cls, 0)
        rows.append(props)
    return _write(rows, PARCEL_COLUMNS)


def buildings_csv(result: AssessmentResult) -> str:
    """One row per structure."""
    rows = []
    for assessment in result.parcels:
        for b in assessment.buildings:
            rows.append({
                "id": b.building.id,
                "parcel_pin": assessment.parcel.pin,
                "parcel_address": assessment.parcel.address,
                "kind": b.building.kind,
                "description": b.building.description,
                "year_built": b.building.year_built,
                "area_m2": round(b.building.area_m2 or 0.0, 1),
                "source": b.building.source,
                **b.as_properties(),
            })
    return _write(rows, BUILDING_COLUMNS)
