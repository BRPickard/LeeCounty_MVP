import datetime as dt

import numpy as np
import pytest

from ddx.geo import make_grid
from ddx.imagery import (DemoProvider, LocalProvider, ProviderError,
                         get_provider, list_providers, resolve_scene,
                         search_scenes)
from ddx.imagery.base import ALL_BANDS, BandStack, Scene
from ddx.imagery.local import load_manifest, save_manifest
from ddx.imagery.stac import StacProvider, list_presets


def test_scene_label_and_serialisation():
    scene = Scene(id="x", provider="p", datetime="2024-03-04T10:00:00Z",
                  platform="sentinel-2a", gsd=10, cloud_cover=12.4,
                  assets={"red": "a", "nir": "b"})
    assert scene.date == "2024-03-04"
    assert scene.bands == ("red", "nir")
    assert "12% cloud" in scene.label()
    assert scene.as_dict()["bands"] == ["red", "nir"]


def test_band_stack_validates_shapes():
    scene = Scene(id="x", provider="p", datetime="2024-01-01T00:00:00Z")
    with pytest.raises(ValueError):
        BandStack(np.zeros((2, 4, 4), "float32"), ("red",),
                  np.ones((4, 4), bool), None, scene)
    with pytest.raises(ValueError):
        BandStack(np.zeros((1, 4, 4), "float32"), ("red",),
                  np.ones((3, 3), bool), None, scene)


def test_demo_search_spans_the_event(event_date):
    provider = DemoProvider()
    scenes = provider.search((-79.19, 35.47, -79.17, 35.49),
                             event_date - dt.timedelta(days=30),
                             event_date + dt.timedelta(days=30))
    assert scenes
    assert any(not s.extra["post_event"] for s in scenes)
    assert any(s.extra["post_event"] for s in scenes)
    assert all(s.synthetic for s in scenes)
    # Dates come back in order and inside the window.
    dates = [s.date for s in scenes]
    assert dates == sorted(dates)


def test_demo_search_is_stable_across_windows(event_date):
    provider = DemoProvider()
    bbox = (-79.19, 35.47, -79.17, 35.49)
    wide = {s.date for s in provider.search(bbox, event_date - dt.timedelta(days=60),
                                            event_date + dt.timedelta(days=60))}
    narrow = {s.date for s in provider.search(bbox, event_date - dt.timedelta(days=20),
                                              event_date + dt.timedelta(days=20))}
    assert narrow <= wide


@pytest.mark.usefixtures("real_store")
def test_demo_read_produces_a_plausible_scene(event_date):
    provider = DemoProvider()
    bbox = (-79.185, 35.475, -79.178, 35.481)
    grid = make_grid(bbox, 2.0)
    scene = provider.search(bbox, event_date - dt.timedelta(days=10), event_date)[-1]
    stack = provider.read(scene, grid)
    assert stack.data.shape == (4, grid.height, grid.width)
    assert stack.bands == ALL_BANDS
    assert 0.0 <= stack.data.min() and stack.data.max() <= 1.3
    # Vegetation should dominate a pre-event scene.
    ndvi = (stack.band("nir") - stack.band("red")) / (stack.band("nir") + stack.band("red"))
    assert ndvi.mean() > 0.2


@pytest.mark.usefixtures("real_store")
def test_demo_post_event_scene_carries_ground_truth(event_date):
    provider = DemoProvider()
    bbox = (-79.185, 35.475, -79.178, 35.481)
    grid = make_grid(bbox, 2.0)
    post = [s for s in provider.search(bbox, event_date, event_date + dt.timedelta(days=20))
            if s.extra["post_event"]][0]
    provider.read(post, grid)
    truth = provider.truth_for(post)
    assert truth
    assert set(truth.values()) <= {"intact", "minor", "major", "destroyed"}


def test_local_provider_round_trip(tmp_path, sample_bbox):
    from ddx.imagery.raster import write_geotiff
    grid = make_grid(sample_bbox, 1.0)
    data = np.stack([np.full(grid.shape, v, "uint8") for v in (60, 90, 120, 200)])
    raster = tmp_path / "ortho.tif"
    write_geotiff(str(raster), data, grid, "uint8")

    provider = LocalProvider(tmp_path)
    save_manifest({"scenes": [{
        "id": "ortho-2024", "datetime": "2024-05-01T00:00:00Z", "platform": "test",
        "gsd": 1.0, "bbox": list(sample_bbox),
        "assets": {b: {"href": "ortho.tif", "band": i}
                   for i, b in enumerate(("red", "green", "blue", "nir"), start=1)},
    }]}, tmp_path)

    assert provider.available()
    scenes = provider.search(sample_bbox, dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    assert len(scenes) == 1
    stack = provider.read(scenes[0], grid)
    assert stack.has(*ALL_BANDS)
    assert stack.band("red").mean() == pytest.approx(60 / 255, abs=0.01)
    assert stack.band("nir").mean() == pytest.approx(200 / 255, abs=0.01)


def test_local_provider_filters_by_date_and_extent(tmp_path, sample_bbox):
    provider = LocalProvider(tmp_path)
    save_manifest({"scenes": [{
        "id": "far-away", "datetime": "2024-05-01T00:00:00Z",
        "bbox": [10.0, 40.0, 10.1, 40.1], "assets": {"red": "x.tif"},
    }]}, tmp_path)
    assert provider.search(sample_bbox, dt.date(2024, 1, 1), dt.date(2024, 12, 31)) == []
    assert provider.search((10.0, 40.0, 10.1, 40.1),
                           dt.date(2023, 1, 1), dt.date(2023, 12, 31)) == []


def test_empty_manifest_means_unavailable(tmp_path):
    assert not LocalProvider(tmp_path).available()
    assert load_manifest(tmp_path) == {"scenes": []}


def test_provider_registry_resolves_ids():
    assert get_provider("demo").name == "demo"
    assert get_provider("local").name == "local"
    assert isinstance(get_provider("earth-search:sentinel-2-l2a"), StacProvider)
    with pytest.raises(ValueError):
        get_provider("nonsense-provider")


def test_stac_presets_are_declared():
    ids = {p["id"] for p in list_presets()}
    assert "earth-search:sentinel-2-l2a" in ids
    assert "planetary:naip" in ids
    with pytest.raises(ValueError):
        StacProvider(endpoint="earth-search", collection="not-a-collection")


def test_provider_listing_includes_availability_and_windows():
    entries = {e["id"]: e for e in list_providers()}
    assert entries["demo"]["available"] is True
    assert entries["demo"]["synthetic"] is True
    assert entries["demo"]["default_window"]["pre_end"] < \
           entries["demo"]["default_window"]["post_start"]


def test_search_reports_provider_failures_without_dying(monkeypatch, event_date):
    class Broken:
        name = "broken"

        def available(self):
            return True

        def search(self, *a, **k):
            raise RuntimeError("network is down")

    import ddx.imagery as imagery
    monkeypatch.setitem(imagery._CACHE, "broken", Broken())
    scenes, problems = search_scenes((-79.19, 35.47, -79.17, 35.49),
                                     event_date - dt.timedelta(days=20), event_date,
                                     providers=["demo", "broken"])
    assert scenes                      # demo still returned results
    assert problems and problems[0]["provider"] == "broken"


def test_resolve_scene_rebuilds_demo_ids_after_cache_eviction(event_date):
    import ddx.imagery as imagery
    bbox = (-79.19, 35.47, -79.17, 35.49)
    scenes, _ = search_scenes(bbox, event_date - dt.timedelta(days=20), event_date,
                              providers=["demo"])
    imagery._SCENES.clear()
    assert resolve_scene(scenes[0].id, bbox).id == scenes[0].id
    with pytest.raises(ProviderError):
        resolve_scene("no-such-scene", bbox)
