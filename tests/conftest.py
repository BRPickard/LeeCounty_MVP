"""Shared fixtures.

Most tests build their own tiny parcel database so they run anywhere; the
few that need the real Lee County layer skip cleanly when it is absent.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from shapely.geometry import box

from ddx.geo import make_grid
from ddx.parcels import ParcelStore, parcel_row

# A block of eight lots in Lee County, laid out two rows by four columns.
ORIGIN_LON, ORIGIN_LAT = -79.1800, 35.4780
LOT = 0.0006  # about 55 m


@pytest.fixture(scope="session")
def sample_bbox() -> tuple[float, float, float, float]:
    return (ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 4 * LOT, ORIGIN_LAT + 2 * LOT)


@pytest.fixture
def store(tmp_path: Path) -> ParcelStore:
    """A small parcel store with a mix of dwellings, outbuildings and vacant lots."""
    parcel_store = ParcelStore(tmp_path / "parcels.sqlite")
    parcel_store.reset()
    rows = []
    for i in range(8):
        col, row_index = i % 4, i // 4
        west = ORIGIN_LON + col * LOT
        south = ORIGIN_LAT + row_index * LOT
        geom = box(west, south, west + LOT * 0.9, south + LOT * 0.9)
        attrs = {
            "pin": f"TEST-{i:04d}",
            "parid": f"P{i}",
            "address": f"{100 + i} TEST ST",
            "owner": f"OWNER {i}",
            "subdivision": "TESTWOOD",
            "acres": 0.3,
            "apr_land": 30000.0,
            "apr_bldg": 150000.0 if i < 6 else 0.0,
            "apr_total": 180000.0 if i < 6 else 30000.0,
            "dwel_desc": "RANCH" if i < 6 else None,
            "dwel_yrblt": 1985 if i < 6 else None,
            "dwel_sfla": 1600 if i < 6 else None,
            "ob_desc": "UTILITY SHED FRAME" if i < 3 else
                       ("PAVING ASPHALT PARKING LIGHT" if i == 3 else None),
            "ob_yrblt": 1990 if i < 4 else None,
            "ob_area": 300 if i < 4 else None,
            "tax_card": f"https://example.invalid/card/{i}",
        }
        rows.append(parcel_row(i + 1, geom, attrs))
    parcel_store.insert_many(rows)
    parcel_store.conn.commit()
    parcel_store.set_meta("dataset_name", "Test County")
    parcel_store.set_meta("feature_count", len(rows))
    yield parcel_store
    parcel_store.close()


@pytest.fixture
def grid(sample_bbox):
    return make_grid(sample_bbox, 1.0)


@pytest.fixture(scope="session")
def real_store() -> ParcelStore:
    parcel_store = ParcelStore()
    if not parcel_store.exists():
        pytest.skip("no ingested parcel database; run scripts/ingest_parcels.py")
    return parcel_store


@pytest.fixture(scope="session")
def event_date() -> dt.date:
    from ddx.imagery.synth import DEFAULT_EVENT
    return DEFAULT_EVENT.date
