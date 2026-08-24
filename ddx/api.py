"""HTTP API and static hosting for the damage assessment app."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .buildings import list_sources as list_building_sources
from .change import DAMAGE_CLASSES, DAMAGE_COLORS, DetectorConfig
from .export import buildings_csv, parcels_csv
from .geo import BBox, bbox_area_km2, parse_bbox
from .imagery import list_providers, search_scenes
from .jobs import DONE, registry
from .parcels import ParcelStore
from .pipeline import AssessmentRequest, PipelineError
from .render import encode_png, side_by_side

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Parcel Damage Assessment",
    version="0.1.0",
    description="Pre/post satellite change detection rolled up to parcels and structures.",
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

_store: ParcelStore | None = None


def store() -> ParcelStore:
    global _store
    if _store is None:
        _store = ParcelStore()
    return _store


# --- models ------------------------------------------------------------------
class AssessBody(BaseModel):
    pre_scene_id: str
    post_scene_id: str
    bbox: list[float] | None = None
    parcel_ids: list[int] = Field(default_factory=list)
    gsd: float | None = Field(default=None, gt=0.05, le=250)
    building_source: str = "auto"
    detector: dict[str, Any] = Field(default_factory=dict)
    pre_provider: str | None = None
    post_provider: str | None = None


# --- metadata ----------------------------------------------------------------
@app.get("/api/health")
def health() -> dict[str, Any]:
    parcel_store = store()
    ready = parcel_store.exists()
    return {
        "status": "ok" if ready else "no-parcels",
        "parcels_loaded": parcel_store.count() if ready else 0,
        "dataset": parcel_store.get_meta("dataset_name") if ready else None,
        "extent": parcel_store.extent() if ready else None,
        "source_crs": parcel_store.get_meta("source_crs") if ready else None,
        "ingested_at": parcel_store.get_meta("ingested_at") if ready else None,
        "limits": {
            "max_aoi_km2": config.MAX_AOI_KM2,
            "max_parcels_per_job": config.MAX_PARCELS_PER_JOB,
            "max_analysis_pixels": config.MAX_ANALYSIS_PIXELS,
        },
        "damage_classes": [{"id": c, "color": DAMAGE_COLORS[c]} for c in DAMAGE_CLASSES],
        "detector_defaults": DetectorConfig().as_dict(),
    }


@app.get("/api/imagery/providers")
def imagery_providers() -> list[dict[str, Any]]:
    return list_providers()


@app.get("/api/buildings/sources")
def building_sources() -> list[dict[str, Any]]:
    return list_building_sources()


# --- parcels -----------------------------------------------------------------
@app.get("/api/parcels")
def parcels_in_bbox(
    bbox: str = Query(..., description="west,south,east,north in WGS84"),
    limit: int = Query(4000, ge=1, le=20000),
    simplified: bool = True,
) -> dict[str, Any]:
    aoi = _parse_bbox(bbox)
    parcel_store = _ready_store()
    area = bbox_area_km2(aoi)
    if area > config.MAX_AOI_KM2 * 4:
        raise HTTPException(400, f"area {area:.0f} km² is too large to draw; zoom in")
    found = parcel_store.in_bbox(aoi, limit=limit, geometry=True, simplified=simplified)
    truncated = len(found) >= limit
    return {
        "type": "FeatureCollection",
        "features": [p.to_feature() for p in found],
        "properties": {"count": len(found), "truncated": truncated,
                       "area_km2": round(area, 3)},
    }


@app.get("/api/parcels/search")
def parcel_search(q: str = Query(..., min_length=2), limit: int = Query(25, le=100)):
    parcel_store = _ready_store()
    return [p.properties() for p in parcel_store.search_text(q, limit=limit)]


@app.get("/api/parcels/{parcel_id}")
def parcel_detail(parcel_id: int) -> dict[str, Any]:
    parcel = _ready_store().get(parcel_id)
    if parcel is None:
        raise HTTPException(404, f"parcel {parcel_id} not found")
    return parcel.to_feature()


# --- imagery -----------------------------------------------------------------
@app.get("/api/imagery/scenes")
def imagery_scenes(
    bbox: str = Query(...),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(...),
    providers: str | None = Query(None, description="comma separated provider ids"),
    limit: int = Query(40, ge=1, le=200),
    max_cloud: float = Query(80.0, ge=0, le=100),
) -> dict[str, Any]:
    aoi = _parse_bbox(bbox)
    try:
        start_date = dt.date.fromisoformat(start)
        end_date = dt.date.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(400, f"dates must be YYYY-MM-DD: {exc}") from exc
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    ids = [p.strip() for p in providers.split(",")] if providers else None
    scenes, problems = search_scenes(aoi, start_date, end_date, providers=ids,
                                     limit=limit, max_cloud=max_cloud)
    return {
        "scenes": [s.as_dict() for s in scenes],
        "problems": problems,
        "window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
    }


# --- assessment --------------------------------------------------------------
@app.post("/api/assess", status_code=202)
def submit_assessment(body: AssessBody) -> dict[str, Any]:
    _ready_store()
    if not body.bbox and not body.parcel_ids:
        raise HTTPException(400, "give either bbox or parcel_ids")
    bbox: BBox | None = None
    if body.bbox:
        try:
            bbox = parse_bbox(body.bbox)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    request = AssessmentRequest(
        pre_scene_id=body.pre_scene_id,
        post_scene_id=body.post_scene_id,
        bbox=bbox,
        parcel_ids=body.parcel_ids,
        gsd=body.gsd,
        building_source=body.building_source,
        detector=body.detector,
        pre_provider=body.pre_provider,
        post_provider=body.post_provider,
    )
    job = registry.submit(request)
    return job.as_dict()


@app.get("/api/assess/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    return _job(job_id).as_dict()


@app.get("/api/assess")
def job_list(limit: int = Query(20, le=100)) -> list[dict[str, Any]]:
    return [j.as_dict() for j in registry.recent(limit)]


@app.get("/api/assess/{job_id}/parcels.geojson")
def job_parcels(job_id: str, geometry: bool = True):
    job = _finished(job_id)
    result = registry.result(job_id)
    if result is not None:
        return JSONResponse(result.feature_collection(geometry=geometry))
    return _file_or_404(job_id, "parcels.geojson", "application/geo+json")


@app.get("/api/assess/{job_id}/buildings.geojson")
def job_buildings(job_id: str):
    _finished(job_id)
    result = registry.result(job_id)
    if result is not None:
        return JSONResponse(result.building_feature_collection())
    return _file_or_404(job_id, "buildings.geojson", "application/geo+json")


@app.get("/api/assess/{job_id}/parcels.csv")
def job_parcels_csv(job_id: str):
    _finished(job_id)
    result = registry.result(job_id)
    body = parcels_csv(result) if result is not None else _read_artifact(job_id, "parcels.csv")
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="damage-parcels-{job_id}.csv"'})


@app.get("/api/assess/{job_id}/buildings.csv")
def job_buildings_csv(job_id: str):
    _finished(job_id)
    result = registry.result(job_id)
    if result is None:
        raise HTTPException(410, "structure table is no longer cached; re-run the assessment")
    return PlainTextResponse(buildings_csv(result), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="damage-structures-{job_id}.csv"'})


@app.get("/api/assess/{job_id}/overlay.png")
def job_overlay(job_id: str):
    _finished(job_id)
    return _file_or_404(job_id, "overlay.png", "image/png")


@app.get("/api/assess/{job_id}/raster/{name}")
def job_raster(job_id: str, name: str):
    _finished(job_id)
    if name not in ("pre_rgb.tif", "post_rgb.tif", "score.tif"):
        raise HTTPException(404, "unknown raster")
    return _file_or_404(job_id, name, "image/tiff")


@app.get("/api/assess/{job_id}/chip/{parcel_id}.png")
def job_chip(job_id: str, parcel_id: int, size: int = Query(320, ge=64, le=1024),
             pad: float = Query(25.0, ge=0, le=200)):
    """Pre / post / damage strip for one parcel, cut from the job's rasters."""
    _finished(job_id)
    parcel = _ready_store().get(parcel_id)
    if parcel is None or parcel.geometry is None:
        raise HTTPException(404, f"parcel {parcel_id} not found")

    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    panels: list[np.ndarray] = []
    for name in ("pre_rgb.tif", "post_rgb.tif", "score.tif"):
        path = registry.artifact(job_id, name)
        if path is None:
            continue
        with rasterio.open(path) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *parcel.geometry.bounds,
                                      densify_pts=21)
            west, south, east, north = bounds
            window = from_bounds(west - pad, south - pad, east + pad, north + pad,
                                 transform=src.transform)
            try:
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
            except Exception:
                continue
            if window.width < 1 or window.height < 1:
                continue
            # Scale the crop to the requested chip size, up or down, keeping
            # the aspect ratio so the three panels line up.
            scale = size / max(window.height, window.width)
            out_h = max(1, int(round(window.height * scale)))
            out_w = max(1, int(round(window.width * scale)))
            arr = src.read(window=window, out_shape=(src.count, out_h, out_w))
        if src.count >= 3:
            panels.append(np.transpose(arr[:3], (1, 2, 0)))
        else:
            from .render import damage_ramp
            score = arr[0].astype("float32") / 255.0
            rgba = damage_ramp(score, DetectorConfig(), score > 0, alpha=255)
            panels.append(rgba[..., :3])

    if not panels:
        raise HTTPException(404, "no imagery available for that parcel in this job")
    return Response(encode_png(side_by_side(panels)), media_type="image/png")


# --- helpers -----------------------------------------------------------------
def _parse_bbox(text: str) -> BBox:
    try:
        return parse_bbox(text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _ready_store() -> ParcelStore:
    parcel_store = store()
    if not parcel_store.exists():
        raise HTTPException(
            503, "no parcel data loaded; run scripts/ingest_parcels.py first")
    return parcel_store


def _job(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


def _finished(job_id: str):
    job = _job(job_id)
    if job.status != DONE:
        raise HTTPException(409, f"job is {job.status}: {job.error or job.message}")
    return job


def _file_or_404(job_id: str, name: str, media_type: str):
    path = registry.artifact(job_id, name)
    if path is None:
        raise HTTPException(404, f"{name} not available for job {job_id}")
    return FileResponse(path, media_type=media_type)


def _read_artifact(job_id: str, name: str) -> str:
    path = registry.artifact(job_id, name)
    if path is None:
        raise HTTPException(404, f"{name} not available for job {job_id}")
    return path.read_text()


@app.exception_handler(PipelineError)
def _pipeline_error(request, exc: PipelineError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
