"""Offline demo provider: procedurally generated pre/post scenes.

Search returns a plausible acquisition calendar around a simulated event;
read renders the scene over the real parcel fabric. Nothing here touches the
network, so the app is fully usable — and testable — with no imagery account.
Every scene is flagged ``synthetic`` and the UI labels it accordingly.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Sequence

import numpy as np

from ..geo import BBox, Grid, grid_bbox_wgs84
from .base import ALL_BANDS, BandStack, Scene
from .synth import (DEFAULT_EVENT, SceneRenderRequest, SynthEvent, _unit_hash,
                    render_scene)

REVISIT_DAYS = 12          # Sentinel-2-like cadence
PLATFORM = "demo-sat"


class DemoProvider:
    """Synthetic imagery over the loaded parcel layer."""

    name = "demo"

    def __init__(self, event: SynthEvent | None = None, gsd: float = 1.0,
                 parcel_store: Any | None = None, building_source: Any | None = None):
        self.event = event or DEFAULT_EVENT
        self.gsd = gsd
        self._store = parcel_store
        self._buildings = building_source

    # -- lazy deps so importing this module stays cheap ----------------------
    @property
    def store(self):
        if self._store is None:
            from ..parcels import ParcelStore
            self._store = ParcelStore()
        return self._store

    @property
    def building_source(self):
        if self._buildings is None:
            from ..buildings.attribute import AttributeBuildingSource
            # Jitter keeps the rendered roofs from sitting exactly on the
            # footprints the analysis samples, so the demo is not self-fulfilling.
            self._buildings = AttributeBuildingSource(jitter=7.0)
        return self._buildings

    def available(self) -> bool:
        return True

    # -- provider API --------------------------------------------------------
    def _scene_for(self, date: dt.date, bbox: BBox) -> Scene:
        sid = f"demo-{date.isoformat()}"
        cloud = round(float(_unit_hash(sid, "cloud")) ** 2.2 * 70.0, 1)
        post = date > self.event.date
        return Scene(
            id=sid,
            provider=self.name,
            datetime=f"{date.isoformat()}T15:42:00Z",
            collection="demo-optical",
            platform=PLATFORM,
            gsd=self.gsd,
            cloud_cover=cloud,
            bbox=bbox,
            assets={b: f"synthetic://{sid}/{b}" for b in ALL_BANDS},
            synthetic=True,
            extra={
                "post_event": post,
                "event": self.event.as_dict(),
                "note": ("Synthetic imagery generated locally for demonstration. "
                         "Not a real acquisition."),
            },
        )

    def search(self, bbox: BBox, start: dt.date, end: dt.date, limit: int = 50,
               **kwargs: Any) -> list[Scene]:
        # Anchor the calendar to a fixed epoch so scene dates are stable
        # regardless of the window the user asks for.
        epoch = dt.date(2020, 1, 1)
        first = start + dt.timedelta(days=(-(start - epoch).days) % REVISIT_DAYS)
        scenes: list[Scene] = []
        date = first
        while date <= end and len(scenes) < limit:
            scenes.append(self._scene_for(date, bbox))
            date += dt.timedelta(days=REVISIT_DAYS)
        # Always offer a clean shot either side of the event; a demo where the
        # only post-event scene is 70% cloud is a bad first impression.
        for offset, label in ((-6, "pre"), (4, "post")):
            special = self.event.date + dt.timedelta(days=offset)
            if start <= special <= end and all(s.date != special.isoformat() for s in scenes):
                scene = self._scene_for(special, bbox)
                scene.cloud_cover = 1.0 if label == "pre" else 3.0
                scenes.append(scene)
        scenes.sort(key=lambda s: s.datetime)
        return scenes[:limit]

    def read(self, scene: Scene, grid: Grid,
             bands: Sequence[str] = ALL_BANDS) -> BandStack:
        bbox = grid_bbox_wgs84(grid)
        parcels = self.store.in_bbox(bbox, geometry=True) if self.store.exists() else []
        buildings = self.building_source.fetch(bbox, parcels).buildings if parcels else []
        date = dt.date.fromisoformat(scene.date)
        post = bool(scene.extra.get("post_event", date > self.event.date))
        # Per-scene gain/offset stands in for atmosphere and sun angle, so the
        # normalisation step downstream has something real to correct.
        gain = 0.94 + 0.14 * _unit_hash(scene.id, "gain")
        offset = -0.006 + 0.016 * _unit_hash(scene.id, "offset")

        data, valid, truth = render_scene(SceneRenderRequest(
            grid=grid, date=date, scene_id=scene.id, parcels=parcels,
            buildings=buildings, event=self.event, post_event=post,
            cloud_cover=scene.cloud_cover or 0.0, gain=gain, offset=offset,
        ))
        scene.extra["truth"] = truth
        order = list(ALL_BANDS)
        keep = [b for b in bands if b in order]
        idx = [order.index(b) for b in keep]
        return BandStack(data=np.ascontiguousarray(data[idx]), bands=tuple(keep),
                         valid=valid, grid=grid, scene=scene)

    def truth_for(self, scene: Scene) -> dict[str, str]:
        """Ground-truth damage states from the last render of this scene."""
        return dict(scene.extra.get("truth") or {})
