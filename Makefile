# Parcel damage assessment MVP
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PORT ?= 8000
PARCELS ?= Parcels.zip

.PHONY: help install ingest serve test test-fast calibrate lint clean

help:
	@echo "make install                 create the venv and install dependencies"
	@echo "make ingest PARCELS=file.zip load a county parcel shapefile"
	@echo "make serve [PORT=8000]       run the web app"
	@echo "make test                    run the full test suite"
	@echo "make test-fast               skip the end-to-end runs"
	@echo "make calibrate               score the detector against the demo ground truth"
	@echo "make clean                   remove generated data and caches"

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

ingest:
	$(BIN)/python scripts/ingest_parcels.py $(PARCELS) --name "Lee County, NC"

serve:
	$(BIN)/python -m uvicorn ddx.api:app --host 127.0.0.1 --port $(PORT) --reload

test:
	$(BIN)/python -m pytest -q

test-fast:
	$(BIN)/python -m pytest -q -m "not slow"

calibrate:
	$(BIN)/python scripts/calibrate.py --sweep

clean:
	rm -rf data/jobs data/cache __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
