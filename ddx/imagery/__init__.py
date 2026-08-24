"""Imagery providers and the registry that dispatches between them."""
from __future__ import annotations

import datetime as dt
import logging
from collections import OrderedDict
from typing import Any, Sequence

from .. import config
from ..geo import BBox, Grid
from .base import (ALL_BANDS, BLUE, GREEN, NIR, RED, RGB, BandStack,
                   ImageryProvider, ProviderError, Scene)
from .demo import DemoProvider
from .local import LocalProvider
from .stac import ENDPOINTS, StacProvider, list_presets

__all__ = [
    "ALL_BANDS", "BLUE", "GREEN", "RED", "NIR", "RGB", "BandStack", "Scene",
    "ImageryProvider", "ProviderError", "DemoProvider", "LocalProvider",
    "StacProvider", "get_provider", "list_providers", "search_scenes",
    "read_scene", "default_provider_ids", "resolve_scene", "get_scene",
    "remember_scenes",
]

log = logging.getLogger(__name__)
_CACHE: dict[str, Any] = {}

# Scenes returned by a search are remembered so a job can be submitted by
# scene id alone. Bounded so a long-running server does not grow without end.
_SCENES: "OrderedDict[str, Scene]" = OrderedDict()
SCENE_CACHE_SIZE = 4000


def remember_scenes(scenes: Sequence[Scene]) -> None:
    for scene in scenes:
        _SCENES[scene.id] = scene
        _SCENES.move_to_end(scene.id)
    while len(_SCENES) > SCENE_CACHE_SIZE:
        _SCENES.popitem(last=False)


def get_scene(scene_id: str) -> Scene | None:
    scene = _SCENES.get(scene_id)
    if scene is not None:
        _SCENES.move_to_end(scene_id)
    return scene


def resolve_scene(scene_id: str, bbox: BBox | None = None,
                  provider_id: str | None = None) -> Scene:
    """Look up a scene by id, re-searching its provider if it has aged out."""
    scene = get_scene(scene_id)
    if scene is not None:
        return scene
    # Demo scene ids carry their own date, so they can always be rebuilt.
    if scene_id.startswith("demo-") and bbox is not None:
        date = dt.date.fromisoformat(scene_id.removeprefix("demo-"))
        scenes = get_provider("demo").search(bbox, date, date, limit=4)
        for candidate in scenes:
            if candidate.id == scene_id:
                remember_scenes([candidate])
                return candidate
    if provider_id and bbox is not None:
        provider = get_provider(provider_id)
        today = dt.date.today()
        scenes = provider.search(bbox, dt.date(2015, 1, 1), today, limit=500)
        remember_scenes(scenes)
        for candidate in scenes:
            if candidate.id == scene_id:
                return candidate
    raise ProviderError(
        f"scene {scene_id!r} is no longer cached; search for imagery again")


def default_provider_ids() -> list[str]:
    return [p.strip() for p in config.IMAGERY_PROVIDERS.split(",") if p.strip()]


def get_provider(provider_id: str) -> ImageryProvider:
    """Instantiate (and memoise) a provider from its id.

    Ids are ``demo``, ``local``, or ``<stac-endpoint>:<collection>``.
    """
    provider_id = (provider_id or "demo").strip()
    if provider_id in _CACHE:
        return _CACHE[provider_id]

    if provider_id == "demo":
        provider: Any = DemoProvider()
    elif provider_id == "local":
        provider = LocalProvider()
    elif ":" in provider_id:
        endpoint, collection = provider_id.split(":", 1)
        provider = StacProvider(endpoint=endpoint, collection=collection)
    elif provider_id in ENDPOINTS:
        provider = StacProvider(endpoint=provider_id)
    else:
        raise ValueError(f"unknown imagery provider: {provider_id}")

    _CACHE[provider_id] = provider
    return provider


def list_providers() -> list[dict[str, Any]]:
    """Describe every selectable imagery source, for the UI's source picker."""
    entries: list[dict[str, Any]] = []
    local = LocalProvider()
    entries.append({
        "default_window": None,
        "id": "local",
        "label": "Registered local imagery (NAIP / Maxar / drone)",
        "gsd": None,
        "requires_network": False,
        "available": local.available(),
        "synthetic": False,
        "note": "Scenes listed in data/scenes/manifest.json.",
    })
    demo = DemoProvider()
    event = demo.event.date
    entries.append({
        "id": "demo",
        "label": "Demo imagery (synthetic, offline)",
        "gsd": 1.0,
        "requires_network": False,
        "available": True,
        "synthetic": True,
        "note": ("Procedurally generated over the real parcel fabric around a "
                 f"simulated event on {event:%d %b %Y}. Use it to exercise the "
                 "workflow; it is not a real acquisition."),
        # Pre-fill the date pickers either side of the simulated event.
        "default_window": {
            "pre_start": str(event - dt.timedelta(days=40)),
            "pre_end": str(event - dt.timedelta(days=1)),
            "post_start": str(event + dt.timedelta(days=1)),
            "post_end": str(event + dt.timedelta(days=40)),
        },
        "event": demo.event.as_dict(),
    })
    today = dt.date.today()
    generic_window = {
        "pre_start": str(today - dt.timedelta(days=120)),
        "pre_end": str(today - dt.timedelta(days=30)),
        "post_start": str(today - dt.timedelta(days=29)),
        "post_end": str(today),
    }
    for preset in list_presets():
        entries.append({**preset, "available": True, "synthetic": False,
                        "default_window": generic_window,
                        "note": f"Served by {preset['endpoint_label']}."})

    enabled = set(default_provider_ids())
    for entry in entries:
        entry["enabled"] = entry["id"] in enabled or not enabled
    return entries


def search_scenes(bbox: BBox, start: dt.date, end: dt.date,
                  providers: Sequence[str] | None = None, limit: int = 40,
                  **kwargs: Any) -> tuple[list[Scene], list[dict[str, str]]]:
    """Search several providers, returning scenes plus any per-provider errors.

    A provider that is unreachable (no network, API down) must not sink the
    whole search: its failure is reported alongside whatever else came back.
    """
    ids = list(providers) if providers else default_provider_ids()
    scenes: list[Scene] = []
    problems: list[dict[str, str]] = []
    for pid in ids:
        try:
            provider = get_provider(pid)
            if not provider.available():
                continue
            scenes.extend(provider.search(bbox, start, end, limit=limit, **kwargs))
        except Exception as exc:
            log.warning("imagery search failed for %s: %s", pid, exc)
            problems.append({"provider": pid, "error": str(exc)})
    scenes.sort(key=lambda s: s.datetime)
    remember_scenes(scenes)
    return scenes, problems


def read_scene(scene: Scene, grid: Grid,
               bands: Sequence[str] = ALL_BANDS) -> BandStack:
    """Read a scene onto a grid using whichever provider produced it."""
    return get_provider(scene.provider).read(scene, grid, bands=bands)
