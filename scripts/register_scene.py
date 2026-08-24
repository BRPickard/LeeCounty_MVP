#!/usr/bin/env python3
"""Register imagery you already have so the app can use it as a date.

Any georeferenced raster GDAL can read works: NAIP tiles, Maxar Open Data
COGs (local or by URL), a county orthophoto, a drone mosaic.

    # four-band NAIP tile, bands in the usual R,G,B,NIR order
    python scripts/register_scene.py naip_2022.tif --date 2022-05-14 --platform NAIP

    # a Maxar Open Data COG straight off S3, no download
    python scripts/register_scene.py \\
        https://maxar-opendata.s3.amazonaws.com/events/.../103001....tif \\
        --date 2024-09-28 --platform "Maxar WorldView-3" --bands red,green,blue

    python scripts/register_scene.py --list
    python scripts/register_scene.py --remove naip-2022

Entries land in data/scenes/manifest.json, which the ``local`` imagery
provider reads. Band order is detected from band descriptions when the file
has them, and can always be overridden with --bands.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddx import config
from ddx.imagery.base import ALL_BANDS
from ddx.imagery.local import (describe_geotiff, is_remote, load_manifest,
                               manifest_path, save_manifest)

# What each band count most likely means when the file does not say.
BY_COUNT = {
    1: ["red"],
    3: ["red", "green", "blue"],
    4: ["red", "green", "blue", "nir"],
}
# Common description spellings mapped onto canonical band names.
ALIASES = {
    "b": "blue", "blue": "blue", "b02": "blue", "band_2": "blue",
    "g": "green", "green": "green", "b03": "green", "band_3": "green",
    "r": "red", "red": "red", "b04": "red", "band_1": "red",
    "n": "nir", "nir": "nir", "b08": "nir", "infrared": "nir",
    "near-infrared": "nir", "band_4": "nir",
}


def infer_bands(info: dict, override: str | None) -> list[str]:
    if override:
        names = [b.strip().lower() for b in override.split(",") if b.strip()]
        bad = [b for b in names if b not in ALL_BANDS]
        if bad:
            raise SystemExit(f"unknown band name(s): {', '.join(bad)}; "
                             f"use any of {', '.join(ALL_BANDS)}")
        return names
    described = [ALIASES.get(d.strip().lower()) for d in info["descriptions"]]
    if all(described) and len(set(described)) == len(described):
        return described  # type: ignore[return-value]
    fallback = BY_COUNT.get(info["count"])
    if not fallback:
        raise SystemExit(
            f"cannot guess band order for a {info['count']}-band file; pass --bands")
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="path or URL of the raster")
    ap.add_argument("--date", help="acquisition date, YYYY-MM-DD")
    ap.add_argument("--id", help="scene id (default: filename plus date)")
    ap.add_argument("--platform", default="", help='e.g. "NAIP", "Maxar WorldView-3"')
    ap.add_argument("--collection", default="local")
    ap.add_argument("--bands", help="comma separated band order, e.g. red,green,blue,nir")
    ap.add_argument("--cloud", type=float, default=None, help="cloud cover percent")
    ap.add_argument("--scale", type=float, default=None,
                    help="divide pixel values by this to get 0-1 reflectance")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true", help="show registered scenes")
    ap.add_argument("--remove", metavar="SCENE_ID", help="delete a scene entry")
    args = ap.parse_args()

    doc = load_manifest()
    scenes = doc.setdefault("scenes", [])

    if args.list:
        if not scenes:
            print(f"no scenes registered in {manifest_path()}")
        for scene in scenes:
            print(f"  {scene['id']:<28} {scene['datetime'][:10]}  "
                  f"{scene.get('platform', ''):<22} "
                  f"{', '.join(scene.get('assets', {}))}")
        return 0

    if args.remove:
        before = len(scenes)
        doc["scenes"] = [s for s in scenes if s["id"] != args.remove]
        if len(doc["scenes"]) == before:
            raise SystemExit(f"no scene with id {args.remove!r}")
        save_manifest(doc)
        print(f"removed {args.remove}")
        return 0

    if not args.source or not args.date:
        ap.error("source and --date are required (or use --list / --remove)")

    try:
        date = dt.date.fromisoformat(args.date)
    except ValueError as exc:
        raise SystemExit(f"--date must be YYYY-MM-DD: {exc}") from exc

    source = args.source
    if not is_remote(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        source = path
    print(f"inspecting {source} …")
    info = describe_geotiff(source if is_remote(str(source)) else Path(source))
    bands = infer_bands(info, args.bands)
    if len(bands) > info["count"]:
        raise SystemExit(f"asked for {len(bands)} bands but the file has {info['count']}")

    href = str(source)
    if not is_remote(href):
        # Store relative to the scene directory when the file lives under it.
        try:
            href = str(Path(href).relative_to(Path(config.SCENE_DIR)))
        except ValueError:
            pass

    scene_id = args.id or f"{Path(str(source)).stem}-{date.isoformat()}"
    entry = {
        "id": scene_id,
        "datetime": f"{date.isoformat()}T00:00:00Z",
        "collection": args.collection,
        "platform": args.platform,
        "gsd": round(info["gsd"], 4),
        "cloud_cover": args.cloud,
        "bbox": [round(v, 6) for v in info["bbox"]],
        "assets": {band: {"href": href, "band": i}
                   for i, band in enumerate(bands, start=1)},
        "scale": args.scale,
        "note": args.note,
    }
    doc["scenes"] = [s for s in scenes if s["id"] != scene_id] + [entry]
    doc["scenes"].sort(key=lambda s: s["datetime"])
    save_manifest(doc)

    print(f"registered {scene_id}")
    print(f"  date     {date}")
    print(f"  bands    {', '.join(bands)} (of {info['count']} in file, {info['dtype']})")
    print(f"  gsd      {info['gsd']:.3f} m   crs {info['crs']}")
    print(f"  bbox     {', '.join(f'{v:.5f}' for v in info['bbox'])}")
    print(f"  manifest {manifest_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
