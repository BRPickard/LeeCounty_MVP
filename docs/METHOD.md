# How the damage assessment works

Every number the app reports traces back to a named, inspectable quantity.
There is no learned model in the loop, which makes the output arguable — a
GIS analyst can disagree with a specific parcel and see exactly which
measurement produced it.

## The pipeline

```
parcels (SQLite + R-tree)
        │
        ├── AOI: drawn box or picked parcels
        │
imagery provider ──► pre scene ─┐
                                ├──► analysis grid (local UTM, fixed GSD)
imagery provider ──► post scene ┘
        │
        ▼
  1. mask intersection      usable pixels on both dates
  2. co-registration        integer pixel shift via phase correlation
  3. radiometric match      per-band gain/offset on invariant surfaces
  4. change measures        spectral magnitude, SSIM loss, NDVI, NDWI
  5. damage score           weighted blend, vegetation-discounted
        │
        ▼
  zonal statistics ──► per structure ──► per parcel ──► CSV / GeoJSON / map
```

## 1. Usable pixels

A pixel counts only if both dates have real data there: inside the scene
footprint, not nodata, and — for Sentinel-2 — not flagged as cloud, cloud
shadow or cirrus by the scene classification band. Parcels whose coverage
falls below 50% are counted in a warning rather than silently scored on three
pixels.

## 2. Co-registration

Two acquisitions from different orbits are routinely misaligned by a pixel or
two. Uncorrected, that misalignment paints a halo of "structural loss" around
every roof edge in the scene. The detector estimates an integer (row, column)
offset by phase correlation on the scene centre and shifts the post date onto
the pre date before anything else is measured. The applied shift is reported
in the job diagnostics; a shift larger than a pixel or two on imagery that
should be well registered is a sign something is wrong with the inputs.

## 3. Radiometric normalisation

Sun angle, atmosphere and sensor differences move reflectance between dates.
Left alone, that difference reads as damage everywhere.

The correction is a per-band gain and offset fitted on *pseudo-invariant
features*: surfaces that plausibly did not change. Two design choices matter:

- **Vegetation is excluded first.** Leaf-on/leaf-off and storm defoliation
  move a great deal of reflectance with no sensor difference to correct, so
  the candidate set keeps only pixels that are non-vegetated on both dates.
- **Candidates are scored against identity, not against a fitted line.**
  Selecting the pixels that best fit an initial estimate is self-reinforcing:
  a bad first gain selects the pixels that agree with it, which confirms the
  bad gain. Since surface-reflectance products are already atmospherically
  corrected, the true gain sits near 1, so ranking candidates by their raw
  pre/post difference is both unbiased enough and immune to that feedback.

A fit implying a gain outside 0.6–1.7 is rejected and that band is left
uncorrected, because a bad normalisation fabricates damage across the whole
scene. Rejections show up in the job's `radiometric_fit` diagnostics as
`[1.0, 0.0]`.

## 4. Change measures

| Measure | Definition | What it catches |
|---|---|---|
| Spectral magnitude | RMS of the per-band difference after normalisation | Roof colour change, debris, bare ground where a building stood |
| Structural loss | `1 − SSIM` on brightness, 7-pixel window | Loss of the regular geometry of an intact roof |
| NDVI delta | `(NIR−R)/(NIR+R)` post minus pre | Defoliation, tree fall, crop loss |
| NDWI delta | `(G−NIR)/(G+NIR)` post minus pre | Standing water where there was none |

## 5. The damage score

```
score = 0.55 · clip(magnitude / 0.11) + 0.45 · clip(structural_loss / 0.45)
score = score · (1 − 0.65)          where the pixel was vegetation before
score = max(score, 0.45)            where the pixel is newly flooded
```

The vegetation discount is the important term. A hurricane strips leaves off
half a county; without the discount every wooded parcel reads as destroyed.
Discounting change on formerly green pixels keeps the *structural* score about
structures, while vegetation loss is reported separately as its own parcel
statistic.

Score breaks map to classes at 0.16 / 0.33 / 0.55 / 0.78 →
`none, possible, moderate, severe, destroyed`.

## 6. Rolling up to structures

Two sampling regimes, chosen by footprint quality:

**Mapped footprints** (a footprint file or OpenStreetMap) — the roof outline
is sampled directly and the structure's score is the 75th percentile of its
pixels. A percentile rather than a mean, so one bright edge pixel cannot
decide a whole roof; the 75th rather than the max, so noise cannot either.

**Tax-roll approximations** (the default for Lee County, which publishes no
footprint layer) — the county records that a parcel holds a 1,600 sq ft ranch
and a 300 sq ft shed, but not where they sit. Scoring a guessed rectangle
would invent precision that does not exist. Instead the parcel's *developed
core* is sampled — the lot pulled back from its boundary, and for lots over
about an acre a disc around the interior point sized to the built area, so a
twenty-acre farm is not scored on twenty acres of soybeans.

Within that core the score is not the mean, which would bury a damaged roof
in lawn, but the strongest **structure-sized patch**: the score is smoothed
with a window matched to the largest structure the parcel is known to hold,
and the peak of that smoothed surface is taken. The question being asked is
"is there a building-sized patch of change on this lot?", which is much closer
to what the tax roll can actually support.

Every structure carries a `confidence` field (high / medium / low) driven by
how many pixels it spans and whether its footprint is mapped or approximate.
Approximate structures never reach `high`.

## 7. Parcel roll-up

Each parcel reports structure counts by damage class, the worst class present,
flooded and vegetation-loss fractions over the whole lot, changed area, and a
rough structure loss estimate: the parcel's appraised building value
apportioned across its structures by footprint area and multiplied by a
per-class factor (0 / 0.05 / 0.25 / 0.60 / 1.0). That figure is a triage aid
for prioritising field inspection, not an appraisal.

## Calibration and its limits

`scripts/calibrate.py` scores the detector against the demo provider's ground
truth, which is known exactly because the simulation created it. On the
shipped demo the detector reaches roughly **0.82 F1** for major-or-destroyed
structures (recall ≈ 0.89, precision ≈ 0.76) using tax-roll approximate
footprints.

**That number measures the plumbing, not the physics.** It says the score
separates damaged from intact structures in a simulation whose damage was
painted by the same codebase. The shipped thresholds are calibrated on
synthetic imagery and must be re-tuned against real labelled imagery — an
xBD-style annotated event, or field-verified inspections from a real
declaration — before any operational use. `scripts/calibrate.py` is the place
to do that.

## Known limitations

- **Resolution governs everything.** At Sentinel-2's 10 m a single-family roof
  is one to two pixels: flood extent, defoliation and neighbourhood-scale
  destruction are credible; per-roof damage grading is not. Sub-metre imagery
  (NAIP, Maxar Open Data, aerial or drone) is what makes structure-level
  grading meaningful.
- **Nadir only.** Change detection compares reflectance from above. A house
  with intact roof and destroyed walls reads as undamaged; so does most flood
  damage below the roofline.
- **Approximate footprints inherit a parcel-level answer.** Every structure on
  such a parcel shares one score, and a large debris pile or a stand of felled
  trees inside the developed core can raise it.
- **Shadow and viewing geometry** are not modelled. Two acquisitions at very
  different sun angles will show elevated change along building edges.
- **The loss estimate is a planning figure.** It comes from the tax roll's
  appraised building value, not from an inspection.
