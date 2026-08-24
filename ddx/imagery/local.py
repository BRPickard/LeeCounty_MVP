"""Imagery already on disk: NAIP tiles, Maxar Open Data COGs, drone orthos.

Scenes are described by ``data/scenes/manifest.json``::

    {"scenes": [
      {"id": "naip-2022", "datetime": "2022-05-14T00:00:00Z", "platform": "NAIP",
       "gsd": 0.6, "assets": {"red": {"href": "naip_2022.tif", "band": 1},
                              "green": {"href": "naip_2022.tif", "band": 2},
                              "blue":  {"href": "naip_2022.tif", "band": 3},
                              "nir":   {"href": "naip_2022.tif", "band": 4}}}
    ]}

``scripts/register_scene.py`` writes those entries for you. Relative hrefs
resolve against the manifest's directory; absolute paths and https:// URLs
work too, so a Maxar Open Data COG can be referenced without downloading it.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import config
from ..geo import BBox, Grid, bbox_geom
from .base import ALL_BANDS, BandStack, ProviderError, Scene
from .raster import is_remote, open_raster, read_band_to_grid

MANIFEST_NAME = "manifest.json"


def manifest_path(scene_dir: Path | None = None) -> Path:
    return Path(scene_dir or config.SCENE_DIR) / MANIFEST_NAME


def load_manifest(scene_dir: Path | None = None) -> dict[str, Any]:
    path = manifest_path(scene_dir)
    if not path.exists():
        return {"scenes": []}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{path} is not valid JSON: {exc}") from exc


def save_manifest(doc: dict[str, Any], scene_dir: Path | None = None) -> Path:
    path = manifest_path(scene_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def _asset(entry: Any) -> tuple[str, int]:
    """Normalise an asset entry to (href, band index)."""
    if isinstance(entry, str):
        return entry, 1
    return entry["href"], int(entry.get("band", 1))


class LocalProvider:
    """Serve scenes listed in the local manifest."""

    name = "local"

    def __init__(self, scene_dir: Path | str | None = None):
        self.scene_dir = Path(scene_dir or config.SCENE_DIR)

    def available(self) -> bool:
        return bool(load_manifest(self.scene_dir).get("scenes"))

    def _resolve(self, href: str) -> str:
        if is_remote(href) or Path(href).is_absolute():
            return href
        return str((self.scene_dir / href).resolve())

    def _scenes(self) -> list[Scene]:
        out: list[Scene] = []
        for raw in load_manifest(self.scene_dir).get("scenes", []):
            assets = {b: self._resolve(_asset(a)[0]) for b, a in raw.get("assets", {}).items()}
            out.append(Scene(
                id=raw["id"],
                provider=self.name,
                datetime=raw["datetime"],
                collection=raw.get("collection", "local"),
                platform=raw.get("platform", ""),
                gsd=raw.get("gsd"),
                cloud_cover=raw.get("cloud_cover"),
                bbox=tuple(raw["bbox"]) if raw.get("bbox") else None,
                assets=assets,
                synthetic=bool(raw.get("synthetic")),
                extra={"raw_assets": raw.get("assets", {}),
                       "scale": raw.get("scale"), "note": raw.get("note", "")},
            ))
        return out

    def search(self, bbox: BBox, start: dt.date, end: dt.date, limit: int = 50,
               **kwargs: Any) -> list[Scene]:
        aoi = bbox_geom(bbox)
        hits: list[Scene] = []
        for scene in self._scenes():
            date = dt.date.fromisoformat(scene.date)
            if not (start <= date <= end):
                continue
            if scene.bbox and not bbox_geom(scene.bbox).intersects(aoi):
                continue
            hits.append(scene)
        hits.sort(key=lambda s: s.datetime)
        return hits[:limit]

    def read(self, scene: Scene, grid: Grid,
             bands: Sequence[str] = ALL_BANDS) -> BandStack:
        raw_assets = scene.extra.get("raw_assets") or {}
        scale = scene.extra.get("scale")
        wanted = [b for b in bands if b in scene.assets]
        if not wanted:
            raise ProviderError(f"scene {scene.id} has none of the bands {list(bands)}")

        layers, masks = [], []
        for band in wanted:
            href = scene.assets[band]
            _, index = _asset(raw_assets.get(band, href))
            values, valid = read_band_to_grid(href, grid, band_index=index, scale=scale)
            layers.append(values)
            masks.append(valid)
        data = np.stack(layers).astype("float32")
        valid = np.logical_and.reduce(masks)
        return BandStack(data=data, bands=tuple(wanted), valid=valid, grid=grid, scene=scene)


def describe_geotiff(path: Path) -> dict[str, Any]:
    """Inspect a GeoTIFF so it can be registered without hand-writing JSON."""
    with open_raster(str(path)) as src:
        descriptions = [(d or "").lower() for d in src.descriptions]
        bounds = src.bounds
        from rasterio.warp import transform_bounds
        wgs = transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21)
        return {
            "count": src.count,
            "dtype": src.dtypes[0],
            "descriptions": descriptions,
            "bbox": list(wgs),
            "gsd": abs(src.transform.a),
            "crs": str(src.crs),
        }
