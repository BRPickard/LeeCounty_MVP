# Choosing imagery

The app treats imagery as a plug-in. Which source you pick decides what the
assessment can honestly claim, so this is the most consequential choice in the
whole workflow.

| Source | Resolution | Cost / access | Good for |
|---|---|---|---|
| Demo (built in) | 1 m synthetic | free, offline | Learning the workflow, testing, demos |
| Sentinel-2 L2A | 10 m | free, no account | Flood extent, defoliation, neighbourhood destruction |
| Landsat 8/9 | 30 m | free, no account | Regional extent only |
| NAIP | 0.6–1 m | free, US only | Excellent baseline; flown every 2–3 years, so rarely a *post* date |
| Maxar Open Data | 30–50 cm | free for declared disasters | The realistic post-event source for structure grading |
| Commercial tasking | 30–50 cm | paid | Post-event imagery on demand |

## The resolution problem, plainly

At 10 m a single-family roof is one to two pixels. The change detector will
tell you a parcel changed; it cannot tell you which of the three structures on
it lost a roof. Structure-level grading needs sub-metre imagery.

The practical pattern for a US county is a **NAIP pre-date paired with a Maxar
Open Data post-date**: both free, both sub-metre. Maxar releases open imagery
for most major declared disasters within days.

## Built-in STAC sources

Sentinel-2 and Landsat work with no configuration — pick them in the source
dropdown and search. They come from Element 84's Earth Search or Microsoft's
Planetary Computer, both public STAC APIs serving cloud-optimised GeoTIFFs, so
only the pixels over your AOI are read.

Planetary Computer asset URLs need signing. Install the optional helper:

```bash
pip install planetary-computer
```

Without it, Earth Search still works unsigned.

## Registering imagery you already have

Anything GDAL can read — a NAIP tile, a Maxar COG, a county orthophoto, a
drone mosaic — can become a selectable date:

```bash
# a four-band NAIP tile on disk
python scripts/register_scene.py naip_2022.tif --date 2022-05-14 --platform NAIP

# a Maxar Open Data COG straight off S3, no download
python scripts/register_scene.py \
    "https://maxar-opendata.s3.amazonaws.com/events/<event>/ard/<...>.tif" \
    --date 2024-09-28 --platform "Maxar WorldView-3" --bands red,green,blue

python scripts/register_scene.py --list
python scripts/register_scene.py --remove naip-2022
```

Band order is read from the file's band descriptions when it has them and can
always be forced with `--bands`. Entries are written to
`data/scenes/manifest.json`, which the `local` provider serves.

### Finding Maxar Open Data for an event

Browse <https://www.maxar.com/open-data> for the event, then take the COG URLs
from its STAC catalogue at
`https://maxar-opendata.s3.amazonaws.com/events/<event-id>/collection.json`.
Register the pre- and post-event tiles covering your AOI with the command
above. Nothing needs downloading — the reads are windowed over HTTP.

## RGB-only imagery

Drone and some aerial mosaics have no near-infrared band. Everything still
runs, with two consequences: no NDVI, so the vegetation discount that keeps
storm defoliation out of the structural score is unavailable; and no NDWI, so
flooding is not flagged. Expect more false positives over wooded parcels.

## Band scaling

Pixel values are converted to roughly 0–1 reflectance automatically: 8-bit is
divided by 255, 16-bit by 10,000 (Sentinel-2 / Landsat convention), floats are
taken as-is. STAC items carrying `raster:bands` scale and offset — which
includes Sentinel-2 processing baseline 04.00 and later — use those instead.
Override with `--scale` when registering a scene with unusual scaling.
