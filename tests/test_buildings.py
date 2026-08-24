import json

import pytest
from shapely.geometry import box, mapping

from ddx.buildings import fetch_buildings, get_source, list_sources
from ddx.buildings.attribute import (AttributeBuildingSource,
                                     footprint_m2_from_living_area,
                                     is_building_description)
from ddx.buildings.base import DWELLING, OUTBUILDING
from ddx.buildings.vector import VectorBuildingSource


def test_site_improvements_are_not_buildings():
    assert is_building_description("DETACHED FRAME GARAGE")
    assert is_building_description("UTILITY SHED FRAME")
    assert not is_building_description("PAVING ASPHALT PARKING LIGHT")
    assert not is_building_description("M.H. SPACES (NO PARK) HOMESITE")
    assert not is_building_description(None)


def test_storey_factor_shrinks_multi_storey_footprints():
    ranch = footprint_m2_from_living_area(1600, "RANCH")
    colonial = footprint_m2_from_living_area(1600, "COLONIAL")
    assert ranch > colonial
    assert ranch == pytest.approx(1600 * 0.09290304, rel=1e-6)


def test_tax_roll_footprints_count_and_classify(store, sample_bbox):
    parcels = store.in_bbox(sample_bbox)
    result = AttributeBuildingSource().fetch(sample_bbox, parcels)
    assert result.approximate is True
    kinds = [b.kind for b in result.buildings]
    # Six parcels have a dwelling; three have a shed (the fourth is paving).
    assert kinds.count(DWELLING) == 6
    assert kinds.count(OUTBUILDING) == 3
    assert all(b.approximate for b in result.buildings)


def test_footprints_stay_inside_their_parcel(store, sample_bbox):
    parcels = {p.id: p for p in store.in_bbox(sample_bbox)}
    result = AttributeBuildingSource(jitter=6.0).fetch(sample_bbox, list(parcels.values()))
    for building in result.buildings:
        parcel = parcels[building.parcel_id]
        assert parcel.geometry.buffer(1e-9).contains(building.geometry)


def test_footprints_are_deterministic(store, sample_bbox):
    parcels = store.in_bbox(sample_bbox)
    first = AttributeBuildingSource(jitter=5.0).fetch(sample_bbox, parcels)
    second = AttributeBuildingSource(jitter=5.0).fetch(sample_bbox, parcels)
    assert [b.geometry.wkt for b in first.buildings] == \
           [b.geometry.wkt for b in second.buildings]


def test_by_parcel_grouping(store, sample_bbox):
    parcels = store.in_bbox(sample_bbox)
    grouped = AttributeBuildingSource().fetch(sample_bbox, parcels).by_parcel()
    assert len(grouped[1]) == 2      # dwelling plus shed
    assert 7 not in grouped or not grouped[7]   # vacant lot


def test_vector_source_reads_geojson_and_joins_to_parcels(store, sample_bbox, tmp_path):
    parcels = store.in_bbox(sample_bbox)
    target = parcels[0]
    centre = target.geometry.representative_point()
    footprint = box(centre.x - 0.00005, centre.y - 0.00005,
                    centre.x + 0.00005, centre.y + 0.00005)
    path = tmp_path / "footprints.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"id": "fp1"},
                      "geometry": mapping(footprint)}],
    }))

    result = VectorBuildingSource(path).fetch(sample_bbox, parcels)
    assert len(result) == 1
    assert result.approximate is False
    assert result.buildings[0].parcel_id == target.id
    assert result.buildings[0].approximate is False


def test_vector_source_reads_newline_delimited_geojson(store, sample_bbox, tmp_path):
    parcels = store.in_bbox(sample_bbox)
    centre = parcels[0].geometry.representative_point()
    footprint = box(centre.x - 0.00004, centre.y - 0.00004,
                    centre.x + 0.00004, centre.y + 0.00004)
    path = tmp_path / "ms.geojsonl"
    path.write_text(json.dumps({"type": "Feature", "properties": {},
                                "geometry": mapping(footprint)}) + "\n")
    assert len(VectorBuildingSource(path).fetch(sample_bbox, parcels)) == 1


def test_missing_vector_file_is_reported(tmp_path):
    source = VectorBuildingSource(tmp_path / "nope.geojson")
    assert not source.available()
    with pytest.raises(FileNotFoundError):
        source.fetch((0, 0, 1, 1), [])


def test_auto_falls_back_to_tax_roll_without_network(store, sample_bbox, monkeypatch):
    import ddx.config as config
    monkeypatch.setattr(config, "BUILDING_FILE", "")
    monkeypatch.setattr(config, "OVERPASS_URL", "")
    result = fetch_buildings(sample_bbox, store.in_bbox(sample_bbox), source="auto")
    assert result.source == "tax-roll"
    assert result.approximate is True


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        get_source("carrier-pigeon")


def test_source_listing_shape():
    ids = {entry["id"] for entry in list_sources()}
    assert {"auto", "tax-roll", "file", "osm"} <= ids
