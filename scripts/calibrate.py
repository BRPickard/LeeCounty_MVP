#!/usr/bin/env python3
"""Score the detector against the demo's ground truth.

The synthetic provider knows exactly which structures it damaged, so the
detector can be measured rather than eyeballed. Use this when changing
detector weights or class breaks:

    python scripts/calibrate.py                 # report current settings
    python scripts/calibrate.py --sweep         # search better class breaks

The numbers are only as good as the simulation: they say the pipeline is
wired up and the score separates damaged from intact *in the demo*. Real
sensors need real labelled events before these thresholds mean anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ddx.change import DAMAGE_CLASSES, DAMAGED_CLASSES, DetectorConfig
from ddx.imagery import get_provider, search_scenes
from ddx.pipeline import AssessmentRequest, run_assessment

# Demo truth states that a responder would want flagged.
TRUTH_DAMAGED = ("major", "destroyed")
TRUTH_ANY = ("minor", "major", "destroyed")

DEFAULT_AOI = (-79.205, 35.455, -79.160, 35.492)


def binary_metrics(truth: dict[str, str], pred: dict[str, str],
                   truth_positive=TRUTH_DAMAGED,
                   pred_positive=DAMAGED_CLASSES) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for key, actual in truth.items():
        if key not in pred:
            continue
        is_pos = actual in truth_positive
        said_pos = pred[key] in pred_positive
        if is_pos and said_pos:
            tp += 1
        elif is_pos:
            fn += 1
        elif said_pos:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / total if total else 0.0, "n": total}


def confusion(truth: dict[str, str], pred: dict[str, str]) -> None:
    order_t = ["intact", "minor", "major", "destroyed"]
    header = "truth / pred"
    print("\n" + header.rjust(14) + "".join(f"{c:>11}" for c in DAMAGE_CLASSES))
    for t in order_t:
        row = Counter(pred[k] for k, v in truth.items() if v == t and k in pred)
        print(f"{t:>14}", "".join(f"{row.get(c, 0):>11}" for c in DAMAGE_CLASSES))


def run(aoi, gsd: float, days_before: int, days_after: int, detector: dict | None = None):
    provider = get_provider("demo")
    event = provider.event.date
    start = event - dt.timedelta(days=days_before + 20)
    end = event + dt.timedelta(days=days_after + 20)
    scenes, _ = search_scenes(aoi, start, end, providers=["demo"], limit=60)
    pre = [s for s in scenes if not s.extra.get("post_event")]
    post = [s for s in scenes if s.extra.get("post_event")]
    if not pre or not post:
        raise SystemExit("demo provider returned no pre/post pair for that window")
    pre_scene = min(pre, key=lambda s: (s.cloud_cover or 0))
    post_scene = min(post, key=lambda s: (s.cloud_cover or 0))
    print(f"pre : {pre_scene.label()}\npost: {post_scene.label()}")

    result = run_assessment(AssessmentRequest(
        pre_scene_id=pre_scene.id, post_scene_id=post_scene.id, bbox=aoi,
        gsd=gsd, building_source="tax-roll", detector=detector or {},
        save_rasters=False,
    ))
    truth = provider.truth_for(post_scene)
    scores = {b.building.id: b.score for p in result.parcels for b in p.buildings}
    pred = {b.building.id: b.damage_class for p in result.parcels for b in p.buildings}
    return result, truth, pred, scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", default=",".join(str(v) for v in DEFAULT_AOI))
    ap.add_argument("--gsd", type=float, default=1.0)
    ap.add_argument("--days-before", type=int, default=10)
    ap.add_argument("--days-after", type=int, default=10)
    ap.add_argument("--sweep", action="store_true", help="search better class breaks")
    args = ap.parse_args()

    aoi = tuple(float(v) for v in args.bbox.split(","))
    result, truth, pred, scores = run(aoi, args.gsd, args.days_before, args.days_after)

    print(f"\nbuildings scored: {len(pred)}   with ground truth: "
          f"{len(set(truth) & set(pred))}")
    print("truth distribution:", dict(Counter(truth.values())))
    print("predicted distribution:", dict(Counter(pred.values())))
    confusion(truth, pred)

    for label, positives in (("major+destroyed", TRUTH_DAMAGED),
                             ("any damage", TRUTH_ANY)):
        m = binary_metrics(truth, pred, positives)
        print(f"\n[{label}] precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"f1 {m['f1']:.3f}  accuracy {m['accuracy']:.3f}  (n={m['n']})")

    usable = {k: v for k, v in scores.items() if v is not None and k in truth}
    if usable:
        import numpy as np
        keys = list(usable)
        values = np.array([usable[k] for k in keys], dtype="float64")
        states = np.array([truth[k] for k in keys])
        print("\n score distribution by ground-truth state")
        print(f"{'state':>10} {'n':>6} {'p10':>7} {'median':>7} {'p90':>7}")
        for state in ("intact", "minor", "major", "destroyed"):
            sel = values[states == state]
            if sel.size:
                print(f"{state:>10} {sel.size:>6} {np.quantile(sel, .1):>7.3f} "
                      f"{np.median(sel):>7.3f} {np.quantile(sel, .9):>7.3f}")

    if args.sweep:
        import numpy as np
        # Only the "moderate" break decides damaged-vs-not, so sweep that one
        # directly instead of walking the whole four-dimensional grid.
        keys = [k for k in scores if scores[k] is not None and k in truth]
        values = np.array([scores[k] for k in keys])
        positive = np.array([truth[k] in TRUTH_DAMAGED for k in keys])
        print("\nsweeping the damaged threshold (class_breaks[1]) ...")
        best = None
        for threshold in np.arange(0.05, 0.85, 0.005):
            said = values >= threshold
            tp = int((said & positive).sum())
            fp = int((said & ~positive).sum())
            fn = int((~said & positive).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if best is None or f1 > best[0]:
                best = (f1, threshold, precision, recall)
        f1, threshold, precision, recall = best
        print(f"best threshold {threshold:.3f}: precision {precision:.3f} "
              f"recall {recall:.3f} f1 {f1:.3f}")
        print(f"(currently class_breaks[1] = {DetectorConfig().class_breaks[1]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
