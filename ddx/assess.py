"""Turn per-pixel change into per-building and per-parcel answers.

Two sampling regimes, chosen by footprint quality:

* **Mapped footprints** — score the roof outline itself. Each structure gets
  its own answer.
* **Tax-roll approximations** — the county knows a parcel holds a 1,600 sq ft
  ranch and a shed, but not where they sit. Scoring a guessed rectangle would
  invent precision, so those parcels are scored over their *developed core*
  (the built-up part of the lot) and every structure on the parcel inherits
  that result, flagged approximate.

Either way the reported building counts come from a real source, and the
confidence field says how much the pixel evidence backs them up.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from .buildings.base import Building, BuildingSet
from .change import (DAMAGE_CLASSES, DAMAGED_CLASSES, NONE, ChangeResult,
                     DetectorConfig)
from .geo import WGS84, reproject_many
from .parcels import Parcel
from .zonal import ZonalIndex, rasterize_labels

# Rough share of a structure's assessed value lost at each damage class. These
# are planning figures for triage, not appraisals.
LOSS_FACTOR: dict[str, float] = {
    "none": 0.0, "possible": 0.05, "moderate": 0.25, "severe": 0.60, "destroyed": 1.0,
}

HIGH, MEDIUM, LOW = "high", "medium", "low"

# Sampling geometry for approximate footprints.
CORE_INSET_M = 3.0          # pull back from the lot line to avoid road bleed
CORE_MIN_RADIUS_M = 16.0
CORE_AREA_FACTOR = 3.2      # homestead disc area relative to built area
LARGE_PARCEL_M2 = 4000.0    # above this, focus on the homestead, not the field


@dataclass
class BuildingAssessment:
    building: Building
    score: float | None
    damage_class: str
    pixels: int
    confidence: str
    approximate: bool
    flooded_fraction: float | None = None
    magnitude: float | None = None
    structure_loss: float | None = None
    ndvi_delta: float | None = None

    def as_properties(self) -> dict[str, Any]:
        return {
            "score": _round(self.score, 3),
            "damage_class": self.damage_class,
            "pixels": self.pixels,
            "confidence": self.confidence,
            "approximate": self.approximate,
            "flooded_fraction": _round(self.flooded_fraction, 3),
            "magnitude": _round(self.magnitude, 4),
            "structure_loss": _round(self.structure_loss, 3),
            "ndvi_delta": _round(self.ndvi_delta, 3),
        }

    def as_feature(self) -> dict[str, Any]:
        return self.building.as_feature(self.as_properties())


@dataclass
class ParcelAssessment:
    parcel: Parcel
    buildings: list[BuildingAssessment] = field(default_factory=list)
    parcel_score: float | None = None
    worst_class: str = NONE
    building_counts: dict[str, int] = field(default_factory=dict)
    flooded_fraction: float | None = None
    vegetation_loss_fraction: float | None = None
    changed_area_m2: float | None = None
    estimated_loss_usd: float | None = None
    pixels: int = 0
    coverage: float | None = None
    note: str = ""

    @property
    def building_count(self) -> int:
        return len(self.buildings)

    @property
    def damaged_building_count(self) -> int:
        return sum(1 for b in self.buildings if b.damage_class in DAMAGED_CLASSES)

    @property
    def changed_building_count(self) -> int:
        """Buildings showing any change at all, including 'possible'."""
        return sum(1 for b in self.buildings if b.damage_class != NONE)

    def as_properties(self) -> dict[str, Any]:
        props = self.parcel.properties()
        props.update({
            "parcel_score": _round(self.parcel_score, 3),
            "damage_class": self.worst_class,
            "buildings_total": self.building_count,
            "buildings_damaged": self.damaged_building_count,
            "buildings_changed": self.changed_building_count,
            "building_counts": self.building_counts,
            "flooded_fraction": _round(self.flooded_fraction, 3),
            "vegetation_loss_fraction": _round(self.vegetation_loss_fraction, 3),
            "changed_area_m2": _round(self.changed_area_m2, 1),
            "estimated_loss_usd": _round(self.estimated_loss_usd, 0),
            "coverage": _round(self.coverage, 3),
            "approximate_footprints": any(b.approximate for b in self.buildings),
            "note": self.note,
        })
        return props

    def as_feature(self, geometry: bool = True) -> dict[str, Any]:
        feature = self.parcel.to_feature(self.as_properties())
        if not geometry:
            feature["geometry"] = None
        return feature


@dataclass
class AssessmentResult:
    parcels: list[ParcelAssessment]
    change: ChangeResult
    building_set: BuildingSet
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def feature_collection(self, geometry: bool = True) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [p.as_feature(geometry=geometry) for p in self.parcels],
            "properties": self.summary,
        }

    def building_feature_collection(self) -> dict[str, Any]:
        return {
            "type": "FeatureCollection",
            "features": [b.as_feature() for p in self.parcels for b in p.buildings],
        }


def _round(value: Any, digits: int) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def developed_core(parcel_utm: BaseGeometry, built_area_m2: float) -> BaseGeometry:
    """The part of a lot where its structures plausibly sit.

    Small lots: the whole lot, pulled back from the boundary. Large lots: a
    disc around the lot's interior point, sized to the built area — so a
    twenty-acre farm is not scored on twenty acres of soybeans.
    """
    core = parcel_utm.buffer(-CORE_INSET_M)
    if core.is_empty or core.area < parcel_utm.area * 0.15:
        core = parcel_utm
    if parcel_utm.area <= LARGE_PARCEL_M2:
        return core
    radius = max(CORE_MIN_RADIUS_M, math.sqrt(max(built_area_m2, 40.0) * CORE_AREA_FACTOR))
    disc = core.representative_point().buffer(radius)
    clipped = disc.intersection(core)
    return clipped if not clipped.is_empty else core


def _confidence(pixels: int, approximate: bool, gsd: float,
                footprint_m2: float | None) -> str:
    if pixels <= 0:
        return LOW
    # How many pixels the structure itself spans, independent of the zone used.
    spanned = (footprint_m2 or 0.0) / max(gsd * gsd, 1e-6)
    if approximate:
        return MEDIUM if pixels >= 12 and spanned >= 4 else LOW
    if pixels >= 20 and spanned >= 12:
        return HIGH
    if pixels >= 6 and spanned >= 2:
        return MEDIUM
    return LOW



# Window sizes (pixels) used to look for a building-sized patch of change.
FOCUS_WINDOWS = (3, 5, 7, 9, 13, 17, 25)


def focus_window_px(built_area_m2: float, gsd: float) -> int:
    """Odd window roughly the size of the structure being looked for."""
    side_px = math.sqrt(max(built_area_m2, 25.0)) / max(gsd, 1e-6)
    target = max(3, int(round(side_px)) | 1)
    for w in FOCUS_WINDOWS:
        if target <= w:
            return w
    return FOCUS_WINDOWS[-1]


def _windowed_mean(score: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Mean score over a window, ignoring invalid pixels."""
    weight = valid.astype("float32")
    total = ndimage.uniform_filter(score * weight, size=window, mode="nearest")
    count = ndimage.uniform_filter(weight, size=window, mode="nearest")
    out = np.zeros_like(score)
    ok = count > 0.15
    out[ok] = total[ok] / count[ok]
    return out


def focus_scores(score: np.ndarray, valid: np.ndarray, index: ZonalIndex,
                 zone_windows: dict[int, int]) -> dict[int, float]:
    """Strongest building-sized patch of change inside each approximate zone.

    Averaging a whole half-acre lot buries a damaged roof in lawn. Asking
    instead "is there a *structure-sized* patch of change anywhere on this
    lot?" is both closer to the question and far more sensitive, at the cost
    of picking up a large debris pile or a felled tree stand — which is why
    these results stay flagged approximate.
    """
    by_window: dict[int, list[int]] = defaultdict(list)
    for zone_id, window in zone_windows.items():
        by_window[window].append(zone_id)
    out: dict[int, float] = {}
    for window, zone_ids in by_window.items():
        smoothed = _windowed_mean(score, valid, window)
        peaks = index.max(smoothed)
        for zone_id in zone_ids:
            if 0 <= zone_id < peaks.size:
                out[zone_id] = _nan_to_none(peaks[zone_id])
    return out


def assess(change: ChangeResult, parcels: Sequence[Parcel], building_set: BuildingSet,
           config: DetectorConfig | None = None) -> AssessmentResult:
    """Roll per-pixel change up to buildings and parcels."""
    cfg = config or change.config
    grid = change.grid
    gsd = grid.gsd
    warnings: list[str] = []

    by_parcel = building_set.by_parcel()
    parcels = [p for p in parcels if p.geometry is not None]
    parcel_utm = dict(zip([p.id for p in parcels],
                          reproject_many([p.geometry for p in parcels], WGS84, grid.crs)))

    # --- build the sampling zones -------------------------------------------
    zone_geoms: list[BaseGeometry] = []
    zone_owner: list[tuple[int, str]] = []   # (parcel_id, building_id or "")
    exact_zone: list[bool] = []
    zone_built_area: list[float] = []

    for parcel in parcels:
        geom_utm = parcel_utm.get(parcel.id)
        if geom_utm is None or geom_utm.is_empty:
            continue
        buildings = by_parcel.get(parcel.id, [])
        exact = [b for b in buildings if not b.approximate]
        approx = [b for b in buildings if b.approximate]

        if exact:
            for geom, b in zip(reproject_many([b.geometry for b in exact], WGS84, grid.crs),
                               exact):
                zone_geoms.append(geom)
                zone_owner.append((parcel.id, b.id))
                exact_zone.append(True)
                zone_built_area.append(b.area_m2 or 0.0)
        if approx:
            # Look for the largest structure the parcel is known to hold.
            built = max((b.area_m2 or 0.0) for b in approx)
            zone_geoms.append(developed_core(geom_utm, sum(b.area_m2 or 0.0 for b in approx)))
            zone_owner.append((parcel.id, ""))
            exact_zone.append(False)
            zone_built_area.append(built)

    # --- zonal statistics ----------------------------------------------------
    layers = {
        "score": change.score,
        "magnitude": change.magnitude,
        "structure_loss": change.structure_loss,
        "ndvi_delta": change.ndvi_delta,
        "flooded": change.flooded,
        "vegetation_loss": change.vegetation_loss,
    }

    zone_stats: dict[int, dict[str, float]] = {}
    if zone_geoms:
        # Small footprints can fall between pixel centres; all_touched keeps
        # them from vanishing at coarse resolution.
        labels = rasterize_labels(zone_geoms, grid, all_touched=True)
        index = ZonalIndex(labels, change.valid)
        computed = {name: (index.fraction(layer) if layer is not None and layer.dtype == bool
                           else index.mean(layer) if layer is not None else None)
                    for name, layer in layers.items()}
        score_p75 = index.quantile(change.score, 0.75)
        approx_windows = {i: focus_window_px(zone_built_area[i], gsd)
                          for i in range(len(zone_geoms)) if not exact_zone[i]}
        focus = focus_scores(change.score, change.valid, index, approx_windows)
        for i in range(len(zone_geoms)):
            zone_stats[i] = {
                "pixels": int(index.counts[i]) if i < len(index.counts) else 0,
                "score_mean": _nan_to_none(computed["score"][i]) if computed["score"] is not None else None,
                "score_p75": _nan_to_none(score_p75[i]),
                "focus": focus.get(i),
                "exact": exact_zone[i],
                "magnitude": _nan_to_none(computed["magnitude"][i]),
                "structure_loss": _nan_to_none(computed["structure_loss"][i]),
                "ndvi_delta": (_nan_to_none(computed["ndvi_delta"][i])
                               if computed["ndvi_delta"] is not None else None),
                "flooded": (_nan_to_none(computed["flooded"][i])
                            if computed["flooded"] is not None else None),
            }

    zone_lookup: dict[tuple[int, str], int] = {
        owner: i for i, owner in enumerate(zone_owner)}

    # --- whole-parcel statistics --------------------------------------------
    parcel_stats: dict[int, dict[str, float]] = {}
    if parcels:
        geoms = [parcel_utm[p.id] for p in parcels if p.id in parcel_utm]
        ids = [p.id for p in parcels if p.id in parcel_utm]
        plabels = rasterize_labels(geoms, grid, all_touched=True)
        pindex_all = ZonalIndex(plabels)          # every pixel in the parcel
        pindex = ZonalIndex(plabels, change.valid)  # only usable pixels
        p_score = pindex.mean(change.score)
        p_flood = pindex.fraction(change.flooded) if change.flooded is not None else None
        p_veg = (pindex.fraction(change.vegetation_loss)
                 if change.vegetation_loss is not None else None)
        changed_mask = change.score >= cfg.class_breaks[1]
        p_changed = pindex.fraction(changed_mask)
        for i, pid in enumerate(ids):
            total_px = int(pindex_all.counts[i]) if i < len(pindex_all.counts) else 0
            used_px = int(pindex.counts[i]) if i < len(pindex.counts) else 0
            parcel_stats[pid] = {
                "score": _nan_to_none(p_score[i]),
                "flooded": _nan_to_none(p_flood[i]) if p_flood is not None else None,
                "veg_loss": _nan_to_none(p_veg[i]) if p_veg is not None else None,
                "changed_fraction": _nan_to_none(p_changed[i]),
                "pixels": used_px,
                "coverage": (used_px / total_px) if total_px else 0.0,
            }

    # --- assemble ------------------------------------------------------------
    assessments: list[ParcelAssessment] = []
    for parcel in parcels:
        pstat = parcel_stats.get(parcel.id, {})
        buildings = by_parcel.get(parcel.id, [])
        approx_stat = zone_stats.get(zone_lookup.get((parcel.id, "")), None)

        building_results: list[BuildingAssessment] = []
        for b in buildings:
            stat = (zone_stats.get(zone_lookup[(parcel.id, b.id)])
                    if (parcel.id, b.id) in zone_lookup else approx_stat)
            if stat is None:
                building_results.append(BuildingAssessment(
                    building=b, score=None, damage_class=NONE, pixels=0,
                    confidence=LOW, approximate=b.approximate))
                continue
            if stat.get("exact"):
                # p75 resists a single bright edge pixel deciding a whole roof.
                score = stat["score_p75"] if stat["pixels"] >= 4 else stat["score_mean"]
            else:
                # Approximate zone: the strongest structure-sized patch on the lot.
                score = stat.get("focus")
                if score is None:
                    score = stat["score_p75"] if stat["pixels"] >= 4 else stat["score_mean"]
            building_results.append(BuildingAssessment(
                building=b,
                score=score,
                damage_class=cfg.classify(score) if score is not None else NONE,
                pixels=int(stat["pixels"]),
                confidence=_confidence(int(stat["pixels"]), b.approximate, gsd, b.area_m2),
                approximate=b.approximate,
                flooded_fraction=stat.get("flooded"),
                magnitude=stat.get("magnitude"),
                structure_loss=stat.get("structure_loss"),
                ndvi_delta=stat.get("ndvi_delta"),
            ))

        counts = {c: 0 for c in DAMAGE_CLASSES}
        for br in building_results:
            counts[br.damage_class] += 1
        worst = NONE
        for c in DAMAGE_CLASSES:
            if counts.get(c):
                worst = c
        if not building_results:
            # No structures: the parcel still gets a land-change verdict.
            worst = cfg.classify(pstat.get("score"))

        loss = None
        if parcel.apr_bldg and building_results:
            total_area = sum(br.building.area_m2 or 0.0 for br in building_results) or 1.0
            loss = sum(
                (br.building.area_m2 or 0.0) / total_area
                * LOSS_FACTOR.get(br.damage_class, 0.0) * float(parcel.apr_bldg)
                for br in building_results
            )

        changed_area = None
        if pstat.get("changed_fraction") is not None and parcel.area_m2:
            changed_area = pstat["changed_fraction"] * float(parcel.area_m2)

        note = ""
        if building_results and all(br.approximate for br in building_results):
            note = "Structure locations approximate; scored over the parcel's developed core."

        assessments.append(ParcelAssessment(
            parcel=parcel,
            buildings=building_results,
            parcel_score=pstat.get("score"),
            worst_class=worst,
            building_counts=counts,
            flooded_fraction=pstat.get("flooded"),
            vegetation_loss_fraction=pstat.get("veg_loss"),
            changed_area_m2=changed_area,
            estimated_loss_usd=loss,
            pixels=int(pstat.get("pixels", 0)),
            coverage=pstat.get("coverage"),
            note=note,
        ))

    low_coverage = [a for a in assessments if (a.coverage or 0) < 0.5]
    if low_coverage:
        warnings.append(
            f"{len(low_coverage)} of {len(assessments)} parcels had under 50% usable "
            "pixels in one or both dates (cloud, shadow or scene edge).")
    if building_set.approximate:
        warnings.append(building_set.note)
    if change.diagnostics.get("synthetic"):
        warnings.append("At least one date is SYNTHETIC demo imagery, not a real acquisition.")

    return AssessmentResult(
        parcels=assessments,
        change=change,
        building_set=building_set,
        summary=summarize(assessments, change, building_set),
        warnings=warnings,
    )


def summarize(assessments: Sequence[ParcelAssessment], change: ChangeResult,
              building_set: BuildingSet) -> dict[str, Any]:
    """Headline numbers for the results panel."""
    counts = {c: 0 for c in DAMAGE_CLASSES}
    parcel_counts = {c: 0 for c in DAMAGE_CLASSES}
    for a in assessments:
        parcel_counts[a.worst_class] += 1
        for c, n in a.building_counts.items():
            counts[c] += n

    total_buildings = sum(counts.values())
    damaged = sum(counts[c] for c in DAMAGED_CLASSES)
    flooded_parcels = sum(1 for a in assessments if (a.flooded_fraction or 0) > 0.05)
    loss = sum(a.estimated_loss_usd or 0.0 for a in assessments)
    return {
        "parcels": len(assessments),
        "buildings": total_buildings,
        "buildings_by_class": counts,
        "parcels_by_class": parcel_counts,
        "buildings_damaged": damaged,
        "buildings_changed": total_buildings - counts[NONE],
        "parcels_affected": len(assessments) - parcel_counts[NONE],
        "parcels_flooded": flooded_parcels,
        "estimated_loss_usd": round(loss, 0) if loss else 0.0,
        "footprint_source": building_set.source,
        "footprints_approximate": building_set.approximate,
        "change": change.summary(),
    }


def _nan_to_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else value
