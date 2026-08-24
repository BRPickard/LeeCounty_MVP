"""Fast grouped statistics over a label raster.

Zonal stats for thousands of small polygons is the inner loop of the whole
assessment, so it avoids per-zone Python: one rasterisation, one lexsort, then
index arithmetic for counts, sums and quantiles.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.features import rasterize

from .geo import Grid


def rasterize_labels(geoms: Sequence, grid: Grid, labels: Sequence[int] | None = None,
                     all_touched: bool = False) -> np.ndarray:
    """Burn geometries (already in the grid CRS) into an int32 label raster.

    Zero means "no zone", so labels start at 1. Later geometries win overlaps,
    which matches the caller's ordering intent.
    """
    labels = list(labels) if labels is not None else list(range(1, len(geoms) + 1))
    shapes = [(g, int(v)) for g, v in zip(geoms, labels)
              if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype="int32")
    return rasterize(shapes, out_shape=grid.shape,
                     transform=rasterio.Affine(*grid.transform),
                     fill=0, dtype="int32", all_touched=all_touched)


class ZonalIndex:
    """Pixel groupings for one label raster, reusable across many value layers."""

    def __init__(self, labels: np.ndarray, valid: np.ndarray | None = None):
        flat = labels.ravel()
        if valid is not None:
            flat = np.where(valid.ravel(), flat, 0)
        keep = flat > 0
        self.n_labels = int(flat.max()) if flat.size else 0
        self._pixel_labels = flat[keep]
        self._pixel_index = np.flatnonzero(keep)
        order = np.argsort(self._pixel_labels, kind="stable")
        self._order = order
        self._sorted_labels = self._pixel_labels[order]
        # Group boundaries in the sorted array, for labels 1..n_labels.
        edges = np.searchsorted(self._sorted_labels,
                                np.arange(1, self.n_labels + 2), side="left")
        self._starts = edges[:-1]
        self._ends = edges[1:]
        self.counts = (self._ends - self._starts).astype("int64")

    def __len__(self) -> int:
        return self.n_labels

    def _gather(self, values: np.ndarray) -> np.ndarray:
        return values.ravel()[self._pixel_index][self._order]

    def sum(self, values: np.ndarray) -> np.ndarray:
        gathered = self._gather(values).astype("float64")
        cumulative = np.concatenate([[0.0], np.cumsum(gathered)])
        return cumulative[self._ends] - cumulative[self._starts]

    def mean(self, values: np.ndarray) -> np.ndarray:
        totals = self.sum(values)
        out = np.full(self.n_labels, np.nan)
        nonzero = self.counts > 0
        out[nonzero] = totals[nonzero] / self.counts[nonzero]
        return out

    def fraction(self, mask: np.ndarray) -> np.ndarray:
        """Share of each zone's valid pixels where the boolean mask is set."""
        return self.mean(mask.astype("float32"))

    def quantile(self, values: np.ndarray, q: float) -> np.ndarray:
        """Per-zone quantile via one lexsort; nearest-rank, no interpolation."""
        gathered = self._gather(values).astype("float64")
        # Sort values within each group by sorting on (label, value) together.
        inner = np.lexsort((gathered, self._sorted_labels))
        ordered = gathered[inner]
        out = np.full(self.n_labels, np.nan)
        nonzero = self.counts > 0
        if not nonzero.any():
            return out
        offsets = np.rint(q * (self.counts[nonzero] - 1)).astype("int64")
        picks = self._starts[nonzero] + offsets
        out[nonzero] = ordered[picks]
        return out

    def max(self, values: np.ndarray) -> np.ndarray:
        return self.quantile(values, 1.0)


def stats_table(index: ZonalIndex, layers: dict[str, np.ndarray],
                quantiles: Iterable[tuple[str, str, float]] = ()) -> dict[str, np.ndarray]:
    """Means for each named layer, plus any requested per-zone quantiles."""
    out: dict[str, np.ndarray] = {"pixels": index.counts.astype("float64")}
    for name, layer in layers.items():
        if layer is None:
            continue
        if layer.dtype == bool:
            out[name] = index.fraction(layer)
        else:
            out[name] = index.mean(layer)
    for out_name, layer_name, q in quantiles:
        layer = layers.get(layer_name)
        if layer is not None:
            out[out_name] = index.quantile(layer, q)
    return out
