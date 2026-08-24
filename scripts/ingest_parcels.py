#!/usr/bin/env python3
"""Load a county parcel shapefile into the ddx parcel database.

Usage:
    python scripts/ingest_parcels.py Parcels.zip
    python scripts/ingest_parcels.py path/to/Parcels.shp --db data/parcels.sqlite

Accepts either a zipped shapefile or an unzipped .shp. Geometry is reprojected
to WGS84; attributes are mapped onto the ddx parcel schema (see FIELD_MAP).
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shapefile  # pyshp
from pyproj import CRS
from shapely.geometry import shape as shapely_shape

from ddx import config
from ddx.geo import WGS84, reproject, valid
from ddx.parcels import ParcelStore, parcel_row

# ddx column <- candidate shapefile field names, first match wins.
FIELD_MAP: dict[str, tuple[str, ...]] = {
    "pin": ("PIN", "PARCEL_PIN", "PARCELID", "PARCEL_ID"),
    "parid": ("PARID", "PARCELNO", "ACCOUNT"),
    "address": ("PropAddr", "SITEADDR", "SITUS", "PHYSADDR", "ADDRESS"),
    "owner": ("Owner1", "OWNER", "OWNERNAME", "OWNER_NAME"),
    "subdivision": ("SUBDIV", "SUBDIVISION"),
    "acres": ("ACRES", "GIS_ACRES", "ACREAGE"),
    "apr_land": ("APRLAND", "LANDVAL", "LAND_VALUE"),
    "apr_bldg": ("APRBLDG", "BLDGVAL", "IMPR_VALUE"),
    "apr_total": ("APRTOT", "TOTALVAL", "TOTAL_VALUE"),
    "dwel_desc": ("dwel_DESCR", "DWELL_DESC", "BLDG_DESC"),
    "dwel_yrblt": ("dwel_YRBLT", "YRBLT", "YEAR_BUILT"),
    "dwel_sfla": ("dwel_SFLA", "SFLA", "HEATED_SQFT", "LIVING_AREA"),
    "ob_desc": ("ob_DESCRIB", "OB_DESC"),
    "ob_yrblt": ("ob_YRBLT",),
    "ob_area": ("ob_AREA",),
    "tax_card": ("TaxCard", "TAXCARD", "TAX_URL"),
}

TEXT_COLUMNS = {"pin", "parid", "address", "owner", "subdivision",
                "dwel_desc", "ob_desc", "tax_card"}
INT_COLUMNS = {"dwel_yrblt", "dwel_sfla", "ob_yrblt", "ob_area"}
FLOAT_COLUMNS = {"acres", "apr_land", "apr_bldg", "apr_total"}


def resolve_shapefile(src: Path, workdir: Path) -> Path:
    """Return a path to a .shp, unzipping into workdir if needed."""
    if src.is_dir():
        shps = sorted(src.glob("*.shp"))
        if not shps:
            raise SystemExit(f"no .shp found in directory {src}")
        return shps[0]
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            zf.extractall(workdir)
        shps = sorted(workdir.rglob("*.shp"))
        if not shps:
            raise SystemExit(f"no .shp inside {src}")
        return shps[0]
    if src.suffix.lower() == ".shp":
        return src
    raise SystemExit(f"unsupported input: {src}")


def source_crs(shp: Path) -> CRS:
    prj = shp.with_suffix(".prj")
    if not prj.exists():
        print(f"  ! no .prj beside {shp.name}; assuming EPSG:4326", file=sys.stderr)
        return WGS84
    return CRS.from_wkt(prj.read_text().strip())


def build_getter(fields: list[str]):
    """Map ddx columns onto the actual field names present in this shapefile."""
    lookup = {f.upper(): f for f in fields}
    resolved: dict[str, str] = {}
    for col, candidates in FIELD_MAP.items():
        for cand in candidates:
            if cand.upper() in lookup:
                resolved[col] = lookup[cand.upper()]
                break
    return resolved


def coerce(col: str, value):
    if value is None:
        return None
    if col in TEXT_COLUMNS:
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8", "replace")
        text = str(value).strip()
        return text or None
    if col in INT_COLUMNS:
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
        return n if n != 0 else None
    if col in FLOAT_COLUMNS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="Parcels.zip, Parcels.shp, or a directory")
    ap.add_argument("--db", type=Path, default=config.PARCEL_DB, help="output SQLite path")
    ap.add_argument("--name", default="", help="dataset label, e.g. 'Lee County, NC'")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="stop after N features (testing)")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ddx-ingest-"))
    try:
        shp = resolve_shapefile(args.source, tmp)
        crs = source_crs(shp)
        print(f"source : {shp}")
        print(f"crs    : {crs.name}")

        reader = shapefile.Reader(str(shp))
        total = len(reader)
        print(f"features: {total}")

        field_names = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
        resolved = build_getter(field_names)
        missing = sorted(set(FIELD_MAP) - set(resolved))
        print(f"mapped : {len(resolved)} attribute columns"
              + (f" (unmapped: {', '.join(missing)})" if missing else ""))

        store = ParcelStore(args.db)
        store.reset()

        src_crs = crs.to_string()
        dst_crs = WGS84.to_string()
        needs_reproject = crs.to_epsg() != 4326

        batch: list[dict] = []
        written = skipped = 0
        for i, srec in enumerate(reader.iterShapeRecords()):
            if args.limit and i >= args.limit:
                break
            geo = srec.shape.__geo_interface__
            if not geo or not geo.get("coordinates"):
                skipped += 1
                continue
            try:
                geom = valid(shapely_shape(geo))
                if needs_reproject:
                    geom = reproject(geom, src_crs, dst_crs)
                if geom.is_empty or geom.area <= 0:
                    skipped += 1
                    continue
            except Exception as exc:  # keep going; report at the end
                print(f"  ! feature {i} geometry error: {exc}", file=sys.stderr)
                skipped += 1
                continue

            rec = srec.record.as_dict()
            attrs = {col: coerce(col, rec.get(src_field))
                     for col, src_field in resolved.items()}
            batch.append(parcel_row(i + 1, geom, attrs))

            if len(batch) >= args.batch:
                store.insert_many(batch)
                store.conn.commit()
                written += len(batch)
                batch.clear()
                print(f"  .. {written}/{total}", end="\r", flush=True)

        if batch:
            store.insert_many(batch)
            store.conn.commit()
            written += len(batch)

        extent = store.extent()
        store.set_meta("source_file", str(args.source))
        store.set_meta("source_crs", crs.name)
        store.set_meta("dataset_name", args.name or shp.stem)
        store.set_meta("feature_count", written)
        store.set_meta("extent", list(extent) if extent else None)
        store.set_meta("ingested_at", dt.datetime.now(dt.timezone.utc).isoformat())

        print(f"\nwrote {written} parcels to {args.db} ({skipped} skipped)")
        if extent:
            print("extent : %.5f, %.5f, %.5f, %.5f" % extent)
        store.close()
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
