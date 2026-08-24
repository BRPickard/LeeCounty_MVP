# The Lee County parcel layer

The shipped dataset is **Lee County, North Carolina** (county seat Sanford) —
not Lee County, Florida. The projection in `Parcels.prj` is NAD83 / North
Carolina State Plane (ftUS) and the tax card URLs point at
`taxaccess.leecountync.gov`.

| | |
|---|---|
| Features | 35,409 polygons (1 skipped: empty geometry) |
| Source CRS | NAD83 / North Carolina (ftUS), reprojected to EPSG:4326 on ingest |
| Extent | −79.3598, 35.3071 → −78.9703, 35.6279 |
| Exported | ArcGIS Pro, 18 Aug 2026 |

## What the attributes give the assessment

The tax roll is doing real work here, because the county publishes no building
footprint layer.

| Column | Coverage | Used for |
|---|---|---|
| `PIN`, `PARID` | 35,407 / 35,251 | Parcel identity, joins back to county systems |
| `PropAddr` | 35,251 | Labels, the exported damage table |
| `Owner1` | 35,248 | Notification lists |
| `dwel_SFLA` | 21,890 | Dwelling **count and size** — the primary structure source |
| `dwel_DESCR` | 21,890 | Storey factor: living area → footprint area |
| `dwel_YRBLT` | 21,890 | Vintage, useful for triage |
| `ob_AREA`, `ob_DESCRIB` | 15,777 | Outbuildings, after filtering site improvements |
| `APRBLDG` | 25,459 non-zero | Structure loss estimate |
| `TaxCard` | all | Deep link to the county's tax card |

`ob_DESCRIB` mixes real structures ("DETACHED FRAME GARAGE", "UTILITY SHED
FRAME") with site improvements ("PAVING ASPHALT PARKING LIGHT", "M.H. SPACES
(NO PARK) HOMESITE"). The latter are filtered out — counting a parking lot as
a damaged building would be worse than not counting it.

Living area is not footprint area: a two-storey colonial reports twice the
area its roof covers. `dwel_DESCR` drives a storey factor (ranch 1.0, cape cod
1.5, colonial 1.9, and so on) before the footprint is estimated.

## Loading a different county

Nothing above is Lee-specific except the field names, and those are matched
against a list of common alternatives (`FIELD_MAP` in
`scripts/ingest_parcels.py`). Any parcel shapefile with a `.prj` works:

```bash
python scripts/ingest_parcels.py OtherCounty.zip --name "Other County, ST"
```

The ingest reports how many attribute columns it mapped and which it could
not. Add spellings to `FIELD_MAP` if your county names things differently.

## Provenance

`Parcels.zip` at the repository root is the county export used to build the
shipped database — public record data from Lee County's GIS, exported from
ArcGIS Pro on 18 Aug 2026. It is committed so `make ingest` works out of the
box. Replace it with a fresher export whenever the county publishes one; the
ingest is idempotent and rebuilds the database from scratch each run.
