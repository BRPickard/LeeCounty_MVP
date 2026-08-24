import datetime as dt
import time

import pytest
from fastapi.testclient import TestClient

from ddx.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def ready(client):
    body = client.get("/api/health").json()
    if body["status"] != "ok":
        pytest.skip("no ingested parcel database")
    return body


def test_health_reports_the_dataset(client, ready):
    assert ready["parcels_loaded"] > 0
    assert len(ready["extent"]) == 4
    assert {c["id"] for c in ready["damage_classes"]} == {
        "none", "possible", "moderate", "severe", "destroyed"}
    assert "max_aoi_km2" in ready["limits"]


def test_parcels_in_bbox(client, ready):
    west, south, east, north = ready["extent"]
    lon = (west + east) / 2
    lat = (south + north) / 2
    response = client.get(f"/api/parcels?bbox={lon-0.004},{lat-0.004},{lon+0.004},{lat+0.004}")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "FeatureCollection"
    assert body["properties"]["count"] >= 0
    for feature in body["features"][:5]:
        assert feature["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert "label" in feature["properties"]


def test_bad_bbox_is_rejected(client, ready):
    assert client.get("/api/parcels?bbox=1,2,3").status_code == 400
    assert client.get("/api/parcels?bbox=-200,0,10,10").status_code == 400


def test_huge_bbox_is_refused(client, ready):
    assert client.get("/api/parcels?bbox=-100,20,-70,45").status_code == 400


def test_parcel_detail_and_missing_id(client, ready):
    body = client.get("/api/parcels/1").json()
    assert body["id"] == 1
    assert client.get("/api/parcels/99999999").status_code == 404


def test_imagery_listing_and_scene_search(client, ready):
    providers = client.get("/api/imagery/providers").json()
    assert any(p["id"] == "demo" and p["available"] for p in providers)

    demo = next(p for p in providers if p["id"] == "demo")
    window = demo["default_window"]
    west, south, east, north = ready["extent"]
    lon, lat = (west + east) / 2, (south + north) / 2
    bbox = f"{lon-0.004},{lat-0.004},{lon+0.004},{lat+0.004}"
    body = client.get(f"/api/imagery/scenes?bbox={bbox}"
                      f"&start={window['pre_start']}&end={window['pre_end']}"
                      "&providers=demo").json()
    assert body["scenes"]
    assert all(s["synthetic"] for s in body["scenes"])


def test_bad_dates_are_rejected(client, ready):
    assert client.get("/api/imagery/scenes?bbox=-79.19,35.47,-79.17,35.49"
                      "&start=nonsense&end=2025-01-01").status_code == 400


def test_building_sources_listed(client):
    ids = {s["id"] for s in client.get("/api/buildings/sources").json()}
    assert "tax-roll" in ids and "auto" in ids


def test_assess_requires_an_area(client, ready):
    response = client.post("/api/assess", json={
        "pre_scene_id": "demo-2025-09-12", "post_scene_id": "demo-2025-09-19"})
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/api/assess/deadbeef").status_code == 404


@pytest.mark.slow
def test_full_assessment_round_trip(client, ready):
    providers = client.get("/api/imagery/providers").json()
    window = next(p for p in providers if p["id"] == "demo")["default_window"]
    west, south, east, north = ready["extent"]
    lon, lat = (west + east) / 2, (south + north) / 2
    bbox = [lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004]
    query = ",".join(str(v) for v in bbox)

    before = client.get(f"/api/imagery/scenes?bbox={query}&start={window['pre_start']}"
                        f"&end={window['pre_end']}&providers=demo").json()["scenes"]
    after = client.get(f"/api/imagery/scenes?bbox={query}&start={window['post_start']}"
                       f"&end={window['post_end']}&providers=demo").json()["scenes"]
    assert before and after

    submitted = client.post("/api/assess", json={
        "pre_scene_id": before[-1]["id"], "post_scene_id": after[0]["id"],
        "bbox": bbox, "gsd": 3.0, "building_source": "tax-roll"})
    assert submitted.status_code == 202
    job_id = submitted.json()["id"]

    deadline = time.time() + 240
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/assess/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status["status"] == "done", status.get("error")

    summary = status["summary"]
    assert summary["parcels"] > 0
    assert set(summary["buildings_by_class"]) == {
        "none", "possible", "moderate", "severe", "destroyed"}
    assert any("SYNTHETIC" in w for w in status["warnings"])

    parcels = client.get(f"/api/assess/{job_id}/parcels.geojson").json()
    assert len(parcels["features"]) == summary["parcels"]
    props = parcels["features"][0]["properties"]
    assert {"damage_class", "buildings_total", "buildings_damaged"} <= set(props)

    buildings = client.get(f"/api/assess/{job_id}/buildings.geojson").json()
    assert len(buildings["features"]) == summary["buildings"]

    csv_body = client.get(f"/api/assess/{job_id}/parcels.csv").text
    assert csv_body.splitlines()[0].startswith("PIN,")
    assert len(csv_body.splitlines()) == summary["parcels"] + 1
    assert client.get(f"/api/assess/{job_id}/buildings.csv").status_code == 200

    overlay = client.get(f"/api/assess/{job_id}/overlay.png")
    assert overlay.status_code == 200 and overlay.content[:4] == b"\x89PNG"

    parcel_id = parcels["features"][0]["id"]
    chip = client.get(f"/api/assess/{job_id}/chip/{parcel_id}.png")
    assert chip.status_code == 200 and chip.content[:4] == b"\x89PNG"

    assert client.get(f"/api/assess/{job_id}/raster/score.tif").status_code == 200
    assert client.get(f"/api/assess/{job_id}/raster/evil.tif").status_code == 404


@pytest.mark.slow
def test_job_reports_a_useful_error_for_an_empty_area(client, ready):
    response = client.post("/api/assess", json={
        "pre_scene_id": "demo-2025-09-12", "post_scene_id": "demo-2025-09-19",
        "bbox": [-100.0, 20.0, -99.99, 20.01]})
    assert response.status_code == 202
    job_id = response.json()["id"]
    deadline = time.time() + 60
    status = {}
    while time.time() < deadline:
        status = client.get(f"/api/assess/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.4)
    assert status["status"] == "error"
    assert "parcel" in (status["error"] or "").lower()
