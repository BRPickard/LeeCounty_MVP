import pytest

from ddx.parcels import ParcelStore


def test_counts_and_extent(store, sample_bbox):
    assert store.count() == 8
    west, south, east, north = store.extent()
    assert west == pytest.approx(sample_bbox[0], abs=1e-6)
    assert north <= sample_bbox[3] + 1e-6


def test_bbox_query_is_precise(store, sample_bbox):
    everything = store.in_bbox(sample_bbox)
    assert len(everything) == 8
    # A box over the leftmost column only.
    narrow = (sample_bbox[0] - 1e-5, sample_bbox[1] - 1e-5,
              sample_bbox[0] + 0.0004, sample_bbox[3])
    subset = store.in_bbox(narrow)
    assert 0 < len(subset) < 8
    assert all(p.geometry.intersects(
        __import__("shapely.geometry", fromlist=["box"]).box(*narrow)) for p in subset)


def test_bbox_query_excludes_bbox_only_overlaps(store):
    # Far away: nothing should come back.
    assert store.in_bbox((-80.0, 34.0, -79.9, 34.1)) == []


def test_limit_is_respected(store, sample_bbox):
    assert len(store.in_bbox(sample_bbox, limit=3)) == 3


def test_lookup_by_id_and_pin(store):
    parcel = store.get(1)
    assert parcel is not None and parcel.pin == "TEST-0000"
    assert parcel.geometry is not None
    assert store.get(999) is None
    assert len(store.find_by_pin("TEST-0003")) == 1
    assert len(store.find_by_pin("P3")) == 1


def test_get_many_preserves_requested_order(store):
    parcels = store.get_many([5, 2, 7])
    assert [p.id for p in parcels] == [5, 2, 7]


def test_text_search_matches_address_and_owner(store):
    assert len(store.search_text("TEST ST")) == 8
    assert len(store.search_text("owner 4")) == 1


def test_area_is_computed_on_insert(store):
    parcel = store.get(1)
    assert parcel.area_m2 > 0
    assert parcel.lon < 0 and parcel.lat > 0


def test_missing_database_reports_not_exists(tmp_path):
    assert not ParcelStore(tmp_path / "absent.sqlite").exists()


def test_feature_shape(store):
    feature = store.get(1).to_feature({"extra": 1})
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["extra"] == 1
    assert feature["properties"]["label"] == "100 TEST ST"
