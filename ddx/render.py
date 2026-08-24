"""Image encoding for the web UI: PNG chips and damage overlays.

Written against zlib/struct rather than an imaging library so the service has
one less binary dependency; the arrays involved are small (map chips), and
rasterio already covers everything geospatial.
"""
from __future__ import annotations

import struct
import zlib
from typing import Sequence

import numpy as np

from .change import DAMAGE_COLORS, ChangeResult, DetectorConfig


def encode_png(array: np.ndarray) -> bytes:
    """Encode an (h, w), (h, w, 3) or (h, w, 4) uint8 array as a PNG."""
    if array.ndim == 2:
        array = array[:, :, None]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype("uint8")
    height, width, channels = array.shape
    color_type = {1: 0, 3: 2, 4: 6}.get(channels)
    if color_type is None:
        raise ValueError(f"cannot encode {channels}-channel image")

    stride = width * channels
    raw = np.empty((height, stride + 1), dtype="uint8")
    raw[:, 0] = 0                       # filter type 0 (None) per scanline
    raw[:, 1:] = array.reshape(height, stride)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + chunk(b"IEND", b""))


def stretch(rgb: np.ndarray, valid: np.ndarray | None = None,
            low: float = 2.0, high: float = 98.0, gamma: float = 1.35,
            per_band: bool = False) -> np.ndarray:
    """Percentile-stretch a float (h, w, 3) array to display uint8.

    The default pools all three bands into one histogram. Stretching each band
    to its own percentiles is the usual remote-sensing habit, but over a
    vegetated scene the red band occupies a narrow low range and gets blown
    out, turning woodland magenta. A shared stretch keeps the colour balance
    recognisable, which matters when a human is comparing before and after.
    """
    out = np.zeros(rgb.shape, dtype="uint8")
    sample = rgb[valid] if valid is not None and valid.any() else rgb.reshape(-1, rgb.shape[-1])
    if sample.size == 0:
        return out
    axis = 0 if per_band else None
    lo = np.percentile(sample, low, axis=axis)
    hi = np.percentile(sample, high, axis=axis)
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    scaled = np.clip((rgb - lo) / span, 0.0, 1.0) ** (1.0 / max(gamma, 1e-6))
    out = (scaled * 255).astype("uint8")
    if valid is not None:
        out[~valid] = 0
    return out


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def damage_ramp(score: np.ndarray, config: DetectorConfig,
                valid: np.ndarray | None = None, alpha: int = 200) -> np.ndarray:
    """Classify a score raster into the damage palette as an RGBA overlay."""
    breaks = config.class_breaks
    classes = np.digitize(score, breaks)   # 0..4 matching DAMAGE_CLASSES
    palette = np.array([_hex_to_rgb(DAMAGE_COLORS[c]) for c in
                        ("none", "possible", "moderate", "severe", "destroyed")],
                       dtype="uint8")
    rgba = np.zeros(score.shape + (4,), dtype="uint8")
    rgba[..., :3] = palette[classes]
    opacity = np.where(classes == 0, 0, alpha).astype("uint8")
    if valid is not None:
        opacity = np.where(valid, opacity, 0).astype("uint8")
    rgba[..., 3] = opacity
    return rgba


def change_overlay(change: ChangeResult, alpha: int = 200) -> np.ndarray:
    return damage_ramp(change.score, change.config, change.valid, alpha=alpha)


def downsample(array: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Nearest-neighbour decimation so chips stay small over the wire."""
    h, w = array.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_dim)))
    return array[::step, ::step]


def side_by_side(images: Sequence[np.ndarray], gap: int = 4,
                 fill: int = 255) -> np.ndarray:
    """Join equal-height uint8 images horizontally with a separator."""
    images = [im for im in images if im is not None and im.size]
    if not images:
        raise ValueError("nothing to join")
    height = min(im.shape[0] for im in images)
    channels = max(im.shape[2] if im.ndim == 3 else 1 for im in images)

    def conform(im: np.ndarray) -> np.ndarray:
        im = im[:height]
        if im.ndim == 2:
            im = im[:, :, None]
        if im.shape[2] == 1 and channels > 1:
            im = np.repeat(im, channels, axis=2)
        if im.shape[2] == 3 and channels == 4:
            im = np.concatenate([im, np.full(im.shape[:2] + (1,), 255, "uint8")], axis=2)
        return im

    parts: list[np.ndarray] = []
    spacer = np.full((height, gap, channels), fill, dtype="uint8")
    for i, im in enumerate(images):
        if i:
            parts.append(spacer)
        parts.append(conform(im))
    return np.concatenate(parts, axis=1)
