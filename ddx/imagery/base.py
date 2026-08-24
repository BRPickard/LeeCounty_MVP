"""Imagery provider interface.

A provider does two things: find scenes over an AOI in a date range, and read
a scene onto a target analysis grid. Everything downstream (change detection,
zonal statistics) works on the grid, so providers of wildly different
resolution and CRS are interchangeable.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from ..geo import BBox, Grid

# Canonical band names. Providers map their own asset keys onto these.
BLUE, GREEN, RED, NIR = "blue", "green", "red", "nir"
RGB = (RED, GREEN, BLUE)
ALL_BANDS = (BLUE, GREEN, RED, NIR)


@dataclass
class Scene:
    """One acquisition that can be read onto a grid."""

    id: str
    provider: str
    datetime: str                      # ISO 8601, UTC
    collection: str = ""
    platform: str = ""
    gsd: float | None = None           # metres per pixel, native
    cloud_cover: float | None = None   # percent 0-100
    bbox: BBox | None = None
    assets: dict[str, str] = field(default_factory=dict)  # band -> href
    preview: str | None = None
    synthetic: bool = False            # True for demo imagery; surfaced in the UI
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.datetime[:10]

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(b for b in ALL_BANDS if b in self.assets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "datetime": self.datetime,
            "date": self.date,
            "collection": self.collection,
            "platform": self.platform,
            "gsd": self.gsd,
            "cloud_cover": self.cloud_cover,
            "bbox": list(self.bbox) if self.bbox else None,
            "bands": list(self.bands),
            "preview": self.preview,
            "synthetic": self.synthetic,
            "label": self.label(),
            **({"note": self.extra["note"]} if "note" in self.extra else {}),
        }

    def label(self) -> str:
        bits = [self.date]
        if self.platform:
            bits.append(self.platform)
        elif self.collection:
            bits.append(self.collection)
        if self.gsd:
            bits.append(f"{self.gsd:g} m")
        if self.cloud_cover is not None:
            bits.append(f"{self.cloud_cover:.0f}% cloud")
        if self.synthetic:
            bits.append("SYNTHETIC")
        return " · ".join(bits)


@dataclass
class BandStack:
    """Scene pixels resampled onto an analysis grid.

    ``data`` is (bands, height, width) float32 scaled to roughly 0..1
    reflectance. ``valid`` is a boolean mask of pixels with real data (False
    for nodata, off-footprint, or cloud-masked pixels).
    """

    data: np.ndarray
    bands: tuple[str, ...]
    valid: np.ndarray
    grid: Grid
    scene: Scene

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"expected (bands, h, w), got {self.data.shape}")
        if len(self.bands) != self.data.shape[0]:
            raise ValueError("band name count does not match array depth")
        if self.valid.shape != self.data.shape[1:]:
            raise ValueError("valid mask shape does not match raster shape")

    def band(self, name: str) -> np.ndarray | None:
        if name not in self.bands:
            return None
        return self.data[self.bands.index(name)]

    def has(self, *names: str) -> bool:
        return all(n in self.bands for n in names)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1:]

    @property
    def coverage(self) -> float:
        """Fraction of grid pixels holding usable data."""
        return float(self.valid.mean()) if self.valid.size else 0.0

    def brightness(self) -> np.ndarray:
        """Mean of the visible bands, or of whatever bands exist."""
        idx = [self.bands.index(b) for b in RGB if b in self.bands]
        if not idx:
            idx = list(range(len(self.bands)))
        return self.data[idx].mean(axis=0)

    def rgb(self) -> np.ndarray:
        """(h, w, 3) array for display; falls back to grayscale replication."""
        if self.has(*RGB):
            return np.stack([self.band(b) for b in RGB], axis=-1)
        gray = self.brightness()
        return np.stack([gray] * 3, axis=-1)


class ImageryProvider(Protocol):
    """Contract every imagery source implements."""

    name: str

    def available(self) -> bool:
        """Whether this provider is usable right now (config/network present)."""

    def search(self, bbox: BBox, start: dt.date, end: dt.date,
               limit: int = 50, **kwargs: Any) -> list[Scene]:
        """Scenes intersecting bbox with an acquisition date in [start, end]."""

    def read(self, scene: Scene, grid: Grid,
             bands: Sequence[str] = ALL_BANDS) -> BandStack:
        """Read a scene onto the analysis grid."""


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a search or read."""
