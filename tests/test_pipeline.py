import datetime as dt

import pytest

from ddx.geo import make_grid
from ddx.imagery import get_provider, search_scenes
from ddx.pipeline import (AssessmentRequest, PipelineError, choose_gsd,
                          resolve_aoi, run_assessment)


def demo_pair(bbox, event_date):
    scenes, _ = search_scenes(bbox, event_date - dt.timedelta(days=20),
                              event_date + dt.timedelta(days=20), providers=["demo"])
    before = [s for s in scenes if not s.extra["post_event"]]
    after = [s for s in scenes if s.extra["post_event"]]
    return before[-1], after[0]


def test_resolve_aoi_from_parcel_ids(store, sample_bbox):
    request = AssessmentRequest(pre_scene_id="a", post_scene_id="b", parcel_ids=[1, 2])
    bbox, parcels = resolve_aoi(request, store)
    assert {p.id for p in parcels} == {1, 2}
    assert bbox[0] < sample_bbox[0]      # buffered outward


def test_resolve_aoi_rejects_an_empty_area(store):
    request = AssessmentRequest(pre_scene_id="a", post_scene_id="b",
                                bbox=(-100.0, 20.0, -99.99, 20.01))
    with pytest.raises(PipelineError, match="no parcels"):
        resolve_aoi(request, store)


def test_resolve_aoi_rejects_unknown_ids(store):
    request = AssessmentRequest(pre_scene_id="a", post_scene_id="b", parcel_ids=[9999])
    with pytest.raises(PipelineError, match="parcel ids"):
        resolve_aoi(request, store)


def test_resolve_aoi_enforces_the_area_limit(store, monkeypatch):
    import ddx.config as config
    monkeypatch.setattr(config, "MAX_AOI_KM2", 0.001)
    request = AssessmentRequest(pre_scene_id="a", post_scene_id="b", parcel_ids=[1])
    with pytest.raises(PipelineError, match="limit"):
        resolve_aoi(request, store)


def test_resolve_aoi_needs_an_area(store):
    with pytest.raises(PipelineError, match="bbox or a list"):
        resolve_aoi(AssessmentRequest(pre_scene_id="a", post_scene_id="b"), store)


def test_choose_gsd_never_claims_more_detail_than_the_input(event_date):
    pre, post = demo_pair((-79.19, 35.47, -79.17, 35.49), event_date)
    pre.gsd, post.gsd = 10.0, 30.0
    bbox = (-79.19, 35.47, -79.17, 35.49)
    assert choose_gsd(pre, post, requested=1.0, bbox=bbox) == 30.0
    assert choose_gsd(pre, post, requested=50.0, bbox=bbox) == 50.0


def test_choose_gsd_respects_the_pixel_budget(monkeypatch, event_date):
    import ddx.config as config
    monkeypatch.setattr(config, "MAX_ANALYSIS_PIXELS", 10_000)
    pre, post = demo_pair((-79.19, 35.47, -79.17, 35.49), event_date)
    pre.gsd = post.gsd = 1.0
    coarse = choose_gsd(pre, post, None, (-79.25, 35.42, -79.10, 35.55))
    assert coarse > 1.0
    grid = make_grid((-79.25, 35.42, -79.10, 35.55), coarse)
    assert grid.width * grid.height <= config.MAX_ANALYSIS_PIXELS * 1.05


def test_detector_overrides_are_applied():
    request = AssessmentRequest(pre_scene_id="a", post_scene_id="b",
                                detector={"magnitude_ref": 0.2, "class_breaks": [.1, .2, .3, .4]})
    config = request.detector_config()
    assert config.magnitude_ref == 0.2
    assert config.class_breaks == (.1, .2, .3, .4)
    assert config.classify(0.25) == "moderate"


@pytest.mark.slow
@pytest.mark.usefixtures("real_store")
def test_end_to_end_demo_run_detects_the_simulated_event(event_date, tmp_path):
    bbox = (-79.190, 35.470, -79.178, 35.480)
    pre, post = demo_pair(bbox, event_date)
    result = run_assessment(AssessmentRequest(
        pre_scene_id=pre.id, post_scene_id=post.id, bbox=bbox, gsd=2.0,
        building_source="tax-roll"), artifact_dir=tmp_path)

    summary = result.summary
    assert summary["parcels"] > 50
    assert summary["buildings"] > 20
    assert summary["buildings_damaged"] > 0
    assert summary["estimated_loss_usd"] > 0
    assert any("SYNTHETIC" in w for w in result.warnings)

    # Artifacts the UI needs.
    for name in ("pre_rgb.tif", "post_rgb.tif", "score.tif", "overlay.png", "grid.json"):
        assert (tmp_path / name).exists(), name

    # The detector should agree with the simulation more often than not.
    truth = get_provider("demo").truth_for(post)
    predicted = {b.building.id: b.damage_class
                 for p in result.parcels for b in p.buildings}
    damaged_truth = {k for k, v in truth.items() if v in ("major", "destroyed")}
    flagged = {k for k, v in predicted.items()
               if v in ("moderate", "severe", "destroyed")}
    overlap = damaged_truth & flagged
    assert len(damaged_truth) > 10
    assert len(overlap) / len(damaged_truth) > 0.6      # recall
    assert len(overlap) / max(1, len(flagged)) > 0.5    # precision


@pytest.mark.slow
@pytest.mark.usefixtures("real_store")
def test_identical_scenes_report_almost_no_damage(event_date):
    bbox = (-79.190, 35.470, -79.180, 35.478)
    pre, _ = demo_pair(bbox, event_date)
    result = run_assessment(AssessmentRequest(
        pre_scene_id=pre.id, post_scene_id=pre.id, bbox=bbox, gsd=2.0,
        building_source="tax-roll", save_rasters=False))
    assert result.summary["buildings_damaged"] == 0
    assert result.change.diagnostics["registration_shift_px"] == [0, 0]
