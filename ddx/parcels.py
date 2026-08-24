"""Parcel storage backed by SQLite + R-tree.

The county shapefile is a static reference layer, so a single-file SQLite
database with an R-tree index is plenty: 35k parcels answer a bbox query in
low single-digit milliseconds and the file ships anywhere without a server.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from shapely import wkb
from shapely.geometry.base import BaseGeometry

from . import config
from .geo import BBox, area_m2, bbox_geom, round_geometry, to_geojson_geometry

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS parcels (
    id           INTEGER PRIMARY KEY,
    pin          TEXT,
    parid        TEXT,
    address      TEXT,
    owner        TEXT,
    subdivision  TEXT,
    acres        REAL,
    apr_land     REAL,
    apr_bldg     REAL,
    apr_total    REAL,
    dwel_desc    TEXT,
    dwel_yrblt   INTEGER,
    dwel_sfla    INTEGER,
    ob_desc      TEXT,
    ob_yrblt     INTEGER,
    ob_area      INTEGER,
    tax_card     TEXT,
    area_m2      REAL,
    lon          REAL,
    lat          REAL,
    minx         REAL,
    miny         REAL,
    maxx         REAL,
    maxy         REAL,
    geom         BLOB NOT NULL,
    geom_simple  BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parcels_pin ON parcels(pin);
CREATE INDEX IF NOT EXISTS idx_parcels_parid ON parcels(parid);

CREATE VIRTUAL TABLE IF NOT EXISTS parcels_rtree USING rtree(
    id, minx, maxx, miny, maxy
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Columns returned for list/summary views (geometry excluded).
SUMMARY_COLUMNS = (
    "id", "pin", "parid", "address", "owner", "subdivision", "acres",
    "apr_land", "apr_bldg", "apr_total", "dwel_desc", "dwel_yrblt", "dwel_sfla",
    "ob_desc", "ob_yrblt", "ob_area", "tax_card", "area_m2", "lon", "lat",
)


@dataclass
class Parcel:
    id: int
    pin: str | None
    parid: str | None
    address: str | None
    owner: str | None
    subdivision: str | None
    acres: float | None
    apr_land: float | None
    apr_bldg: float | None
    apr_total: float | None
    dwel_desc: str | None
    dwel_yrblt: int | None
    dwel_sfla: int | None
    ob_desc: str | None
    ob_yrblt: int | None
    ob_area: int | None
    tax_card: str | None
    area_m2: float | None
    lon: float | None
    lat: float | None
    geometry: BaseGeometry | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.address or self.pin or self.parid or f"parcel {self.id}"

    def properties(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in SUMMARY_COLUMNS}
        d["label"] = self.label
        return d

    def to_feature(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        props = self.properties()
        if extra:
            props.update(extra)
        return {
            "type": "Feature",
            "id": self.id,
            "properties": props,
            "geometry": to_geojson_geometry(self.geometry) if self.geometry else None,
        }


def _row_to_parcel(row: sqlite3.Row, geom_key: str | None = None) -> Parcel:
    kwargs = {k: row[k] for k in SUMMARY_COLUMNS}
    geom = None
    if geom_key is not None and row[geom_key] is not None:
        geom = wkb.loads(row[geom_key])
    return Parcel(**kwargs, geometry=geom)


class ParcelStore:
    """Read/write access to the parcel database."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.PARCEL_DB)
        self._conn: sqlite3.Connection | None = None

    # -- connection ----------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ParcelStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def exists(self) -> bool:
        if not self.path.exists():
            return False
        try:
            return self.count() > 0
        except sqlite3.DatabaseError:
            return False

    # -- writing -------------------------------------------------------------
    def initialize(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def reset(self) -> None:
        self.initialize()
        self.conn.executescript("DELETE FROM parcels; DELETE FROM parcels_rtree;")
        self.conn.commit()

    def insert_many(self, rows: Sequence[dict[str, Any]]) -> None:
        cols = [
            "id", "pin", "parid", "address", "owner", "subdivision", "acres",
            "apr_land", "apr_bldg", "apr_total", "dwel_desc", "dwel_yrblt",
            "dwel_sfla", "ob_desc", "ob_yrblt", "ob_area", "tax_card",
            "area_m2", "lon", "lat", "minx", "miny", "maxx", "maxy",
            "geom", "geom_simple",
        ]
        placeholders = ",".join("?" for _ in cols)
        self.conn.executemany(
            f"INSERT OR REPLACE INTO parcels ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(r.get(c) for c in cols) for r in rows],
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO parcels_rtree (id, minx, maxx, miny, maxy) VALUES (?,?,?,?,?)",
            [(r["id"], r["minx"], r["maxx"], r["miny"], r["maxy"]) for r in rows],
        )

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value) if not isinstance(value, str) else value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def all_meta(self) -> dict[str, Any]:
        return {r["key"]: self.get_meta(r["key"]) for r in
                self.conn.execute("SELECT key FROM meta")}

    # -- reading -------------------------------------------------------------
    def count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM parcels").fetchone()["n"])

    def extent(self) -> BBox | None:
        row = self.conn.execute(
            "SELECT MIN(minx) a, MIN(miny) b, MAX(maxx) c, MAX(maxy) d FROM parcels"
        ).fetchone()
        if row is None or row["a"] is None:
            return None
        return (row["a"], row["b"], row["c"], row["d"])

    def get(self, parcel_id: int, geometry: bool = True) -> Parcel | None:
        row = self.conn.execute("SELECT * FROM parcels WHERE id = ?", (parcel_id,)).fetchone()
        return _row_to_parcel(row, "geom" if geometry else None) if row else None

    def get_many(self, ids: Sequence[int], geometry: bool = True,
                 simplified: bool = False) -> list[Parcel]:
        if not ids:
            return []
        out: list[Parcel] = []
        key = ("geom_simple" if simplified else "geom") if geometry else None
        for chunk in _chunks(list(ids), 900):
            marks = ",".join("?" for _ in chunk)
            rows = self.conn.execute(
                f"SELECT * FROM parcels WHERE id IN ({marks})", chunk
            ).fetchall()
            out.extend(_row_to_parcel(r, key) for r in rows)
        order = {pid: i for i, pid in enumerate(ids)}
        out.sort(key=lambda p: order.get(p.id, 0))
        return out

    def find_by_pin(self, pin: str, geometry: bool = True) -> list[Parcel]:
        rows = self.conn.execute(
            "SELECT * FROM parcels WHERE pin = ? OR parid = ?", (pin, pin)
        ).fetchall()
        return [_row_to_parcel(r, "geom" if geometry else None) for r in rows]

    def search_text(self, query: str, limit: int = 25) -> list[Parcel]:
        like = f"%{query.strip().upper()}%"
        rows = self.conn.execute(
            "SELECT * FROM parcels WHERE UPPER(address) LIKE ? OR UPPER(owner) LIKE ?"
            " OR pin LIKE ? OR parid LIKE ? LIMIT ?",
            (like, like, like, like, limit),
        ).fetchall()
        return [_row_to_parcel(r, None) for r in rows]

    def ids_in_bbox(self, bbox: BBox, limit: int | None = None) -> list[int]:
        w, s, e, n = bbox
        sql = ("SELECT id FROM parcels_rtree WHERE maxx >= ? AND minx <= ?"
               " AND maxy >= ? AND miny <= ? ORDER BY id")
        params: list[Any] = [w, e, s, n]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [int(r["id"]) for r in self.conn.execute(sql, params)]

    def in_bbox(self, bbox: BBox, limit: int | None = None, geometry: bool = True,
                simplified: bool = False, precise: bool = True) -> list[Parcel]:
        """Parcels intersecting a bbox.

        The R-tree gives bbox-level candidates; ``precise`` filters those down
        to true geometric intersection.
        """
        ids = self.ids_in_bbox(bbox, limit=None if precise else limit)
        parcels = self.get_many(ids, geometry=geometry or precise, simplified=simplified)
        if precise:
            clip = bbox_geom(bbox)
            parcels = [p for p in parcels
                       if p.geometry is not None and p.geometry.intersects(clip)]
            if limit is not None:
                parcels = parcels[:limit]
            if not geometry:
                for p in parcels:
                    p.geometry = None
        return parcels

    def in_geometry(self, geom: BaseGeometry, limit: int | None = None,
                    geometry: bool = True) -> list[Parcel]:
        """Parcels intersecting an arbitrary WGS84 polygon."""
        candidates = self.get_many(self.ids_in_bbox(geom.bounds), geometry=True)
        hit = [p for p in candidates if p.geometry is not None and p.geometry.intersects(geom)]
        if limit is not None:
            hit = hit[:limit]
        if not geometry:
            for p in hit:
                p.geometry = None
        return hit

    def iter_all(self, geometry: bool = False) -> Iterator[Parcel]:
        for row in self.conn.execute("SELECT * FROM parcels ORDER BY id"):
            yield _row_to_parcel(row, "geom" if geometry else None)


def _chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def parcel_row(pid: int, geom: BaseGeometry, attrs: dict[str, Any],
               simplify_tolerance: float = 1e-5) -> dict[str, Any]:
    """Build an insertable row from a WGS84 geometry plus attributes."""
    minx, miny, maxx, maxy = geom.bounds
    centroid = geom.representative_point()
    simple = geom.simplify(simplify_tolerance, preserve_topology=True)
    if simple.is_empty:
        simple = geom
    row = {
        "id": pid,
        "area_m2": area_m2(geom),
        "lon": round(centroid.x, 7),
        "lat": round(centroid.y, 7),
        "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
        "geom": wkb.dumps(round_geometry(geom, 7)),
        "geom_simple": wkb.dumps(round_geometry(simple, 6)),
    }
    row.update(attrs)
    return row
