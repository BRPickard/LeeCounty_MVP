"""Runtime configuration.

Everything is overridable by environment variable so the same code runs on a
laptop, in a container, or in CI without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default))).expanduser().resolve()


# --- storage -----------------------------------------------------------------
DATA_DIR = _path("DDX_DATA_DIR", REPO_ROOT / "data")
PARCEL_DB = _path("DDX_PARCEL_DB", DATA_DIR / "parcels.sqlite")
JOB_DIR = _path("DDX_JOB_DIR", DATA_DIR / "jobs")
SCENE_DIR = _path("DDX_SCENE_DIR", DATA_DIR / "scenes")
CACHE_DIR = _path("DDX_CACHE_DIR", DATA_DIR / "cache")

# --- imagery -----------------------------------------------------------------
# Comma separated list of enabled providers, highest priority first.
IMAGERY_PROVIDERS = os.environ.get("DDX_IMAGERY_PROVIDERS", "local,demo,earth-search:sentinel-2-l2a")
EARTH_SEARCH_URL = os.environ.get(
    "DDX_EARTH_SEARCH_URL", "https://earth-search.aws.element84.com/v1"
)
PLANETARY_URL = os.environ.get(
    "DDX_PLANETARY_URL", "https://planetarycomputer.microsoft.com/api/stac/v1"
)
MAXAR_OPEN_DATA_URL = os.environ.get(
    "DDX_MAXAR_URL", "https://maxar-opendata.s3.amazonaws.com/events/catalog.json"
)

# --- buildings ---------------------------------------------------------------
BUILDING_SOURCE = os.environ.get("DDX_BUILDING_SOURCE", "auto")
BUILDING_FILE = os.environ.get("DDX_BUILDING_FILE", "")
OVERPASS_URL = os.environ.get("DDX_OVERPASS_URL", "https://overpass-api.de/api/interpreter")

# --- analysis limits ---------------------------------------------------------
# Guardrails so a careless bbox does not try to pull half a state.
MAX_AOI_KM2 = float(os.environ.get("DDX_MAX_AOI_KM2", "150"))
MAX_PARCELS_PER_JOB = int(os.environ.get("DDX_MAX_PARCELS_PER_JOB", "5000"))
MAX_ANALYSIS_PIXELS = int(os.environ.get("DDX_MAX_ANALYSIS_PIXELS", str(12_000_000)))
# Target ground sample distance (metres) for the analysis grid. Imagery finer
# than this is read at native resolution then downsampled to this grid.
DEFAULT_ANALYSIS_GSD = float(os.environ.get("DDX_ANALYSIS_GSD", "2.0"))

for _d in (DATA_DIR, JOB_DIR, SCENE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
