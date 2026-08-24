"""Windowed, reprojecting reads from local or remote rasters onto an analysis grid.

The important trick here is the decimated read: when a source pixel is much
finer than the target grid, we let GDAL serve the request from an overview
level instead of pulling full-resolution bytes we are only going to average
away. For a remote COG that is the difference between a few hundred kilobytes
and a few hundred megabytes.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from ..geo import Grid, grid_bounds
from .base import ProviderError

# GDAL settings that make remote COG access sane: do not list the whole
# bucket directory, keep a chunk cache, and do not sign anonymous requests.
REMOTE_ENV: dict[str, Any] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    "AWS_NO_SIGN_REQUEST": "YES",
}


def is_remote(href: str) -> bool:
    return href.startswith(("http://", "https://", "s3://", "/vsi"))


@contextmanager
def open_raster(href: str) -> Iterator[rasterio.DatasetReader]:
    """Open a local path or remote COG URL with COG-friendly GDAL settings."""
    env = REMOTE_ENV if is_remote(href) else {}
    try:
        with rasterio.Env(**env):
            with rasterio.open(href) as src:
                yield src
    except RasterioIOError as exc:
        raise ProviderError(f"cannot open raster {href}: {exc}") from exc


def auto_scale(dtype: str, declared: float | None = None) -> float:
    """Divisor that brings raw pixel values to roughly 0..1 reflectance."""
    if declared:
        return declared
    dtype = str(dtype)
    if dtype.startswith("uint8") or dtype.startswith("int8"):
        return 255.0
    if dtype.startswith("uint16") or dtype.startswith("int16"):
        return 10000.0  # Sentinel-2 / Landsat style scaling
    return 1.0


def _decimation(src_window: Window, dst_shape: tuple[int, int]) -> tuple[int, int]:
    """Output shape for a decimated read that still oversamples the target."""
    h, w = dst_shape
    # Read at ~2x the target sampling so the warp has something to average.
    out_h = int(min(max(1, math.ceil(src_window.height)), max(1, h * 2)))
    out_w = int(min(max(1, math.ceil(src_window.width)), max(1, w * 2)))
    return out_h, out_w


def read_band_to_grid(
    href: str,
    grid: Grid,
    band_index: int = 1,
    scale: float | None = None,
    offset: float = 0.0,
    nodata: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one band of a raster onto ``grid``.

    Returns ``(values, valid)`` where values is float32 on the grid and valid
    is a boolean mask of pixels backed by real source data.
    """
    dst = np.zeros(grid.shape, dtype="float32")
    dst_valid = np.zeros(grid.shape, dtype=bool)

    with open_raster(href) as src:
        src_bounds = transform_bounds(grid.crs, src.crs, *grid_bounds(grid), densify_pts=21)
        window = from_bounds(*src_bounds, transform=src.transform)
        # Pad by a pixel so bilinear resampling has neighbours at the edges.
        window = Window(window.col_off - 1, window.row_off - 1,
                        window.width + 2, window.height + 2)
        full = Window(0, 0, src.width, src.height)
        try:
            window = window.intersection(full)
        except Exception:
            return dst, dst_valid  # no overlap between scene and AOI
        window = window.round_offsets(op="floor").round_lengths(op="ceil")
        if window.width <= 0 or window.height <= 0:
            return dst, dst_valid

        out_shape = _decimation(window, grid.shape)
        src_nodata = nodata if nodata is not None else src.nodatavals[band_index - 1]
        arr = src.read(
            band_index,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.average if out_shape[0] < window.height else Resampling.nearest,
            masked=True,
        )
        src_transform = src.window_transform(window) * rasterio.Affine.scale(
            window.width / out_shape[1], window.height / out_shape[0]
        )
        divisor = auto_scale(src.dtypes[band_index - 1], scale)
        # Cast before filling: an integer masked array cannot hold NaN.
        values = np.ma.asarray(arr).astype("float32").filled(np.nan)
        if src_nodata is not None:
            values = np.where(values == src_nodata, np.nan, values)
        values = (values + offset) / divisor
        src_crs = src.crs

    source_valid = np.isfinite(values).astype("float32")
    values = np.nan_to_num(values, nan=0.0)

    reproject(
        source=values, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=rasterio.Affine(*grid.transform), dst_crs=grid.crs,
        resampling=resampling, src_nodata=None, dst_nodata=None,
    )
    valid_f = np.zeros(grid.shape, dtype="float32")
    reproject(
        source=source_valid, destination=valid_f,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=rasterio.Affine(*grid.transform), dst_crs=grid.crs,
        resampling=Resampling.nearest,
    )
    dst_valid = valid_f > 0.5
    dst[~dst_valid] = 0.0
    return dst, dst_valid


def write_geotiff(path: str, data: np.ndarray, grid: Grid, dtype: str = "uint8",
                  nodata: float | None = None, band_names: list[str] | None = None) -> None:
    """Write a (bands, h, w) array as a tiled, overview-bearing GeoTIFF."""
    if data.ndim == 2:
        data = data[None, ...]
    profile = {
        "driver": "GTiff", "height": grid.height, "width": grid.width,
        "count": data.shape[0], "dtype": dtype, "crs": grid.crs,
        "transform": rasterio.Affine(*grid.transform),
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "predictor": 2,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names[: data.shape[0]], start=1):
                dst.set_band_description(i, name)
        factors = [f for f in (2, 4, 8, 16) if min(grid.width, grid.height) // f >= 32]
        if factors:
            dst.build_overviews(factors, Resampling.average)
