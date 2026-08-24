"""Real satellite imagery from any STAC API.

Ships with presets for the two open archives that need no account:
Element 84's Earth Search (Sentinel-2 L2A, Landsat) and Microsoft's Planetary
Computer. Both serve cloud-optimised GeoTIFFs, so reads are windowed against
the AOI rather than downloading whole scenes.

Sentinel-2 is 10 m: excellent for flood extent, defoliation and neighbourhood
scale destruction, but a single-family roof is one to two pixels. For
per-building damage grading, register sub-metre imagery through the ``local``
provider instead (see docs/IMAGERY.md).
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .. import config
from ..geo import BBox, Grid
from .base import ALL_BANDS, BandStack, ProviderError, Scene
from .raster import read_band_to_grid

log = logging.getLogger(__name__)

# Sentinel-2 scene classification values that are not usable ground:
# 0 nodata, 1 saturated, 3 cloud shadow, 8/9 cloud medium+high, 10 cirrus.
SCL_BAD = (0, 1, 3, 8, 9, 10)


@dataclass
class CollectionPreset:
    """How to read one collection from one endpoint."""

    collection: str
    label: str
    band_assets: dict[str, str]
    gsd: float
    cloud_field: str = "eo:cloud_cover"
    mask_asset: str | None = None
    mask_bad_values: tuple[int, ...] = ()
    platform_field: str = "platform"
    extra_query: dict[str, Any] = field(default_factory=dict)


EARTH_SEARCH_PRESETS = {
    "sentinel-2-l2a": CollectionPreset(
        collection="sentinel-2-l2a",
        label="Sentinel-2 L2A (10 m, ~5 day revisit)",
        band_assets={"blue": "blue", "green": "green", "red": "red", "nir": "nir"},
        gsd=10.0,
        mask_asset="scl",
        mask_bad_values=SCL_BAD,
    ),
    "landsat-c2-l2": CollectionPreset(
        collection="landsat-c2-l2",
        label="Landsat 8/9 Collection 2 L2 (30 m)",
        band_assets={"blue": "blue", "green": "green", "red": "red", "nir": "nir08"},
        gsd=30.0,
    ),
}

PLANETARY_PRESETS = {
    "sentinel-2-l2a": CollectionPreset(
        collection="sentinel-2-l2a",
        label="Sentinel-2 L2A (10 m, ~5 day revisit)",
        band_assets={"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"},
        gsd=10.0,
        mask_asset="SCL",
        mask_bad_values=SCL_BAD,
    ),
    "naip": CollectionPreset(
        collection="naip",
        label="NAIP aerial (0.6-1 m, every 2-3 years, US only)",
        band_assets={"red": "image", "green": "image", "blue": "image", "nir": "image"},
        gsd=0.6,
        cloud_field="",
    ),
}

ENDPOINTS: dict[str, dict[str, Any]] = {
    "earth-search": {
        "url": config.EARTH_SEARCH_URL,
        "label": "Earth Search (AWS open data)",
        "presets": EARTH_SEARCH_PRESETS,
        "sign": False,
    },
    "planetary": {
        "url": config.PLANETARY_URL,
        "label": "Microsoft Planetary Computer",
        "presets": PLANETARY_PRESETS,
        "sign": True,
    },
}

# NAIP packs all four bands in one asset, in this order.
NAIP_BAND_INDEX = {"red": 1, "green": 2, "blue": 3, "nir": 4}


class StacProvider:
    """Search and read a STAC API."""

    def __init__(self, endpoint: str = "earth-search", collection: str = "sentinel-2-l2a",
                 url: str | None = None, max_cloud: float = 80.0):
        cfg = ENDPOINTS.get(endpoint)
        if cfg is None:
            raise ValueError(f"unknown STAC endpoint {endpoint!r}")
        self.endpoint = endpoint
        self.url = url or cfg["url"]
        self.sign = cfg["sign"]
        presets = cfg["presets"]
        if collection not in presets:
            raise ValueError(f"{endpoint} has no preset for collection {collection!r}")
        self.preset = presets[collection]
        self.max_cloud = max_cloud
        self.name = f"{endpoint}:{collection}"
        self._client = None

    def available(self) -> bool:
        return bool(self.url)

    # -- internals -----------------------------------------------------------
    def _open(self):
        if self._client is None:
            try:
                from pystac_client import Client
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ProviderError("pystac-client is required for STAC providers") from exc
            try:
                self._client = Client.open(self.url)
            except Exception as exc:
                raise ProviderError(f"cannot reach STAC API {self.url}: {exc}") from exc
        return self._client

    def _sign_item(self, item):
        if not self.sign:
            return item
        try:
            import planetary_computer
        except ImportError:
            log.warning("planetary_computer not installed; PC asset URLs will not be signed")
            return item
        return planetary_computer.sign(item)

    # -- provider API --------------------------------------------------------
    def search(self, bbox: BBox, start: dt.date, end: dt.date, limit: int = 50,
               max_cloud: float | None = None, **kwargs: Any) -> list[Scene]:
        client = self._open()
        query: dict[str, Any] = dict(self.preset.extra_query)
        cloud_limit = self.max_cloud if max_cloud is None else max_cloud
        if self.preset.cloud_field and cloud_limit < 100:
            query[self.preset.cloud_field] = {"lte": cloud_limit}
        try:
            search = client.search(
                collections=[self.preset.collection],
                bbox=list(bbox),
                datetime=f"{start.isoformat()}/{end.isoformat()}",
                query=query or None,
                max_items=limit,
            )
            items = list(search.items())
        except Exception as exc:
            raise ProviderError(f"STAC search failed on {self.url}: {exc}") from exc

        scenes: list[Scene] = []
        for item in items:
            props = item.properties
            assets: dict[str, str] = {}
            asset_meta: dict[str, dict[str, Any]] = {}
            for band, key in self.preset.band_assets.items():
                asset = item.assets.get(key)
                if asset is None:
                    continue
                assets[band] = asset.href
                extra = asset.extra_fields or {}
                raster = (extra.get("raster:bands") or [{}])[0]
                asset_meta[band] = {
                    "key": key,
                    "scale": raster.get("scale"),
                    "offset": raster.get("offset", 0.0) or 0.0,
                    "index": NAIP_BAND_INDEX.get(band, 1)
                    if self.preset.collection == "naip" else 1,
                }
            if not assets:
                continue
            mask_asset = None
            if self.preset.mask_asset:
                a = item.assets.get(self.preset.mask_asset)
                mask_asset = a.href if a else None

            scenes.append(Scene(
                id=item.id,
                provider=self.name,
                datetime=(props.get("datetime") or f"{item.datetime:%Y-%m-%dT%H:%M:%SZ}"),
                collection=self.preset.collection,
                platform=str(props.get(self.preset.platform_field, "") or ""),
                gsd=props.get("gsd") or self.preset.gsd,
                cloud_cover=props.get(self.preset.cloud_field) if self.preset.cloud_field else None,
                bbox=tuple(item.bbox) if item.bbox else None,
                assets=assets,
                preview=(item.assets.get("thumbnail").href
                         if item.assets.get("thumbnail") else None),
                extra={"asset_meta": asset_meta, "mask_asset": mask_asset,
                       "stac_endpoint": self.endpoint, "item_id": item.id},
            ))
        scenes.sort(key=lambda s: s.datetime)
        return scenes

    def read(self, scene: Scene, grid: Grid,
             bands: Sequence[str] = ALL_BANDS) -> BandStack:
        item = None
        if self.sign:
            try:
                item = self._sign_item(self._open().get_collection(
                    scene.collection).get_item(scene.extra.get("item_id", scene.id)))
            except Exception as exc:
                log.warning("could not re-sign %s: %s", scene.id, exc)

        asset_meta = scene.extra.get("asset_meta") or {}
        wanted = [b for b in bands if b in scene.assets]
        if not wanted:
            raise ProviderError(f"scene {scene.id} has none of the bands {list(bands)}")

        layers, masks = [], []
        for band in wanted:
            meta = asset_meta.get(band, {})
            href = scene.assets[band]
            if item is not None and meta.get("key") and meta["key"] in item.assets:
                href = item.assets[meta["key"]].href
            values, valid = read_band_to_grid(
                href, grid,
                band_index=int(meta.get("index", 1)),
                scale=(1.0 / meta["scale"]) if meta.get("scale") else None,
                offset=float(meta.get("offset") or 0.0) / float(meta["scale"])
                if meta.get("scale") and meta.get("offset") else 0.0,
            )
            layers.append(values)
            masks.append(valid)

        data = np.stack(layers).astype("float32")
        valid = np.logical_and.reduce(masks)

        mask_href = scene.extra.get("mask_asset")
        if item is not None and self.preset.mask_asset in (item.assets or {}):
            mask_href = item.assets[self.preset.mask_asset].href
        if mask_href and self.preset.mask_bad_values:
            try:
                from rasterio.enums import Resampling
                scl, scl_valid = read_band_to_grid(
                    mask_href, grid, scale=1.0, resampling=Resampling.nearest)
                codes = np.rint(scl).astype("int16")
                bad = np.isin(codes, self.preset.mask_bad_values) & scl_valid
                valid &= ~bad
            except Exception as exc:
                log.warning("cloud mask unavailable for %s: %s", scene.id, exc)

        return BandStack(data=data, bands=tuple(wanted), valid=valid, grid=grid, scene=scene)


def list_presets() -> list[dict[str, Any]]:
    """Everything the UI can offer as a real-imagery source."""
    out = []
    for endpoint, cfg in ENDPOINTS.items():
        for key, preset in cfg["presets"].items():
            out.append({
                "id": f"{endpoint}:{key}",
                "endpoint": endpoint,
                "endpoint_label": cfg["label"],
                "collection": key,
                "label": preset.label,
                "gsd": preset.gsd,
                "requires_network": True,
            })
    return out
