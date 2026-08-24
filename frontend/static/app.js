/* Parcel damage assessment — map UI.
 *
 * No build step and no framework: one module-scoped state object, explicit
 * render functions, and Leaflet for the map. Everything the user picks lives
 * in `state`; every DOM update flows through a render* function so there is
 * one place to look when the screen disagrees with the data.
 */
(() => {
"use strict";

const CLASS_COLORS = {
  none: "#2f9e44", possible: "#fab005", moderate: "#fd7e14",
  severe: "#e03131", destroyed: "#862e9c",
};
const CLASS_ORDER = ["destroyed", "severe", "moderate", "possible", "none"];
const CLASS_LABELS = {
  none: "No change detected", possible: "Possible change", moderate: "Moderate damage",
  severe: "Severe damage", destroyed: "Destroyed",
};
const PARCEL_ZOOM = 15;

const state = {
  health: null,
  providers: [],
  provider: null,
  mode: null,               // "draw" | "pick" | null
  bbox: null,               // [w, s, e, n]
  parcelIds: new Set(),
  scenes: [],
  job: null,
  results: null,            // GeoJSON FeatureCollection
  buildings: new Map(),     // parcel id -> [building features]
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat();
const money = new Intl.NumberFormat(undefined, {
  style: "currency", currency: "USD", maximumFractionDigits: 0,
});

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* not JSON; keep the status line */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ---------------------------------------------------------------- map --- */
let map, parcelLayer, selectionLayer, resultLayer, overlayLayer;

function initMap() {
  map = L.map("map", { zoomControl: true, preferCanvas: true })
         .setView([35.47, -79.17], 12);
  // Basemap is best-effort: the app is fully usable without tiles (offline
  // deployments, restricted networks), so a tile failure is not an error.
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "© OpenStreetMap contributors", crossOrigin: true,
  }).addTo(map);

  parcelLayer = L.geoJSON(null, {
    style: () => ({ color: "#6b7686", weight: 1, opacity: 0.55, fillOpacity: 0.05,
                    fillColor: "#4c8dff" }),
    onEachFeature: (feature, layer) => {
      layer.on("click", () => onParcelClick(feature));
      layer.bindTooltip(feature.properties.label || `Parcel ${feature.id}`,
                        { sticky: true });
    },
  }).addTo(map);
  selectionLayer = L.layerGroup().addTo(map);
  resultLayer = L.geoJSON(null, {
    style: styleResult,
    onEachFeature: (feature, layer) => {
      layer.on("click", () => showDetail(feature.id));
      layer.bindTooltip(resultTooltip(feature), { sticky: true });
    },
  }).addTo(map);

  map.on("moveend", refreshParcels);
  installBoxDraw();
}

function styleResult(feature) {
  const cls = feature.properties.damage_class || "none";
  return {
    color: CLASS_COLORS[cls], weight: 1,
    opacity: cls === "none" ? 0.35 : 0.9,
    fillColor: CLASS_COLORS[cls],
    fillOpacity: cls === "none" ? 0.06 : 0.45,
  };
}

function resultTooltip(feature) {
  const p = feature.properties;
  return `<strong>${p.label || p.pin || feature.id}</strong><br>` +
         `${CLASS_LABELS[p.damage_class] || p.damage_class}<br>` +
         `${p.buildings_damaged}/${p.buildings_total} structures damaged`;
}

/* Drag a rectangle on the map without pulling in a drawing plugin. */
function installBoxDraw() {
  let origin = null, rect = null;
  const container = map.getContainer();

  const point = (event) => map.mouseEventToLatLng(event);

  container.addEventListener("mousedown", (event) => {
    if (state.mode !== "draw" || event.button !== 0) return;
    event.preventDefault();
    origin = point(event);
    rect = L.rectangle([origin, origin],
                       { color: "#4c8dff", weight: 1.5, dashArray: "5 4", fillOpacity: 0.08 })
            .addTo(selectionLayer);
  }, true);

  container.addEventListener("mousemove", (event) => {
    if (!origin || !rect) return;
    rect.setBounds(L.latLngBounds(origin, point(event)));
  }, true);

  const finish = (event) => {
    if (!origin || !rect) return;
    const bounds = L.latLngBounds(origin, point(event));
    origin = null; rect = null;
    if (bounds.getNorth() - bounds.getSouth() < 1e-5) { setMode(null); return; }
    setBBox([bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()]);
    setMode(null);
  };
  container.addEventListener("mouseup", finish, true);
}

function setMode(mode) {
  state.mode = state.mode === mode ? null : mode;
  $("btn-draw").classList.toggle("active", state.mode === "draw");
  $("btn-pick").classList.toggle("active", state.mode === "pick");
  if (state.mode === "draw") {
    map.dragging.disable();
    showHint("Drag on the map to draw the area of interest");
  } else {
    map.dragging.enable();
    showHint(state.mode === "pick"
      ? "Click parcels to add or remove them from the selection" : null);
  }
  if (state.mode === "pick") refreshParcels(true);
}

function showHint(text) {
  const hint = $("map-hint");
  hint.textContent = text || "";
  hint.hidden = !text;
}

/* ------------------------------------------------------------ parcels --- */
let parcelRequest = 0;

async function refreshParcels(force) {
  if (state.results && force !== true) { parcelLayer.clearLayers(); return; }
  if (map.getZoom() < PARCEL_ZOOM) {
    parcelLayer.clearLayers();
    if (!state.results) showHint(state.mode ? null : "Zoom in to see parcels");
    return;
  }
  const bounds = map.getBounds();
  const bbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
  const token = ++parcelRequest;
  try {
    const data = await api(`/api/parcels?bbox=${bbox.join(",")}&limit=3000`);
    if (token !== parcelRequest) return;   // a later pan already won
    parcelLayer.clearLayers();
    parcelLayer.addData(data);
    parcelLayer.eachLayer((layer) => {
      if (state.parcelIds.has(layer.feature.id)) highlightPicked(layer);
    });
  } catch (error) {
    console.warn("parcel load failed", error);
  }
}

function highlightPicked(layer) {
  layer.setStyle({ color: "#4c8dff", weight: 2, fillOpacity: 0.35 });
}

function onParcelClick(feature) {
  if (state.mode !== "pick") {
    if (state.results) showDetail(feature.id);
    return;
  }
  if (state.parcelIds.has(feature.id)) state.parcelIds.delete(feature.id);
  else state.parcelIds.add(feature.id);
  state.bbox = null;
  selectionLayer.clearLayers();
  parcelLayer.eachLayer((layer) => {
    if (state.parcelIds.has(layer.feature.id)) highlightPicked(layer);
    else parcelLayer.resetStyle(layer);
  });
  renderAreaSummary();
}

function setBBox(bbox) {
  state.bbox = bbox;
  state.parcelIds.clear();
  selectionLayer.clearLayers();
  L.rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
              { color: "#4c8dff", weight: 2, fillOpacity: 0.05 }).addTo(selectionLayer);
  renderAreaSummary();
}

function clearSelection() {
  state.bbox = null;
  state.parcelIds.clear();
  state.scenes = [];
  selectionLayer.clearLayers();
  $("scene-picker").hidden = true;
  renderAreaSummary();
  refreshParcels(true);
}

function areaKm2(bbox) {
  const midLat = ((bbox[1] + bbox[3]) / 2) * Math.PI / 180;
  return Math.abs((bbox[2] - bbox[0]) * 111.32 * Math.cos(midLat) *
                  (bbox[3] - bbox[1]) * 110.57);
}

function renderAreaSummary() {
  const el = $("area-summary");
  const limit = state.health ? state.health.limits.max_aoi_km2 : 150;
  let ready = false;
  if (state.bbox) {
    const km2 = areaKm2(state.bbox);
    ready = km2 <= limit;
    el.textContent = `Box selected — ${km2.toFixed(1)} km²` +
      (ready ? "" : ` (over the ${limit} km² limit; draw a smaller box)`);
    el.classList.toggle("error", !ready);
  } else if (state.parcelIds.size) {
    ready = true;
    el.textContent = `${state.parcelIds.size} parcel${state.parcelIds.size === 1 ? "" : "s"} selected`;
    el.classList.remove("error");
  } else {
    el.textContent = "No area selected yet.";
    el.classList.remove("error");
  }
  $("btn-search").disabled = !ready;
  $("btn-run").disabled = !ready || !$("pre-scene").value || !$("post-scene").value;
}

function selectionBBox() {
  if (state.bbox) return state.bbox;
  if (!state.parcelIds.size) return null;
  let west = 180, south = 90, east = -180, north = -90;
  parcelLayer.eachLayer((layer) => {
    if (!state.parcelIds.has(layer.feature.id)) return;
    const b = layer.getBounds();
    west = Math.min(west, b.getWest()); south = Math.min(south, b.getSouth());
    east = Math.max(east, b.getEast()); north = Math.max(north, b.getNorth());
  });
  return west <= east ? [west, south, east, north] : null;
}

/* ------------------------------------------------------------ imagery --- */
function renderProviders() {
  const select = $("provider-select");
  select.innerHTML = "";
  state.providers.forEach((provider) => {
    const option = document.createElement("option");
    option.value = provider.id;
    option.textContent = provider.label + (provider.available ? "" : " — not configured");
    option.disabled = !provider.available;
    select.appendChild(option);
  });
  const preferred = state.providers.find((p) => p.available && p.id === "demo")
                 || state.providers.find((p) => p.available);
  if (preferred) select.value = preferred.id;
  onProviderChange();
}

function onProviderChange() {
  const provider = state.providers.find((p) => p.id === $("provider-select").value);
  state.provider = provider || null;
  $("provider-note").textContent = provider ? provider.note : "";
  const window = provider && provider.default_window;
  if (window) {
    $("pre-start").value = window.pre_start;
    $("pre-end").value = window.pre_end;
    $("post-start").value = window.post_start;
    $("post-end").value = window.post_end;
  }
  $("scene-picker").hidden = true;
  state.scenes = [];
  renderAreaSummary();
}

async function findImagery() {
  const bbox = selectionBBox();
  if (!bbox) return;
  const provider = $("provider-select").value;
  const cloud = $("cloud-range").value;
  const note = $("imagery-note");
  note.textContent = "Searching…";
  note.classList.remove("error");
  $("btn-search").disabled = true;

  const query = (start, end) =>
    `/api/imagery/scenes?bbox=${bbox.join(",")}&start=${start}&end=${end}` +
    `&providers=${encodeURIComponent(provider)}&max_cloud=${cloud}`;
  try {
    const [before, after] = await Promise.all([
      api(query($("pre-start").value, $("pre-end").value)),
      api(query($("post-start").value, $("post-end").value)),
    ]);
    fillScenes($("pre-scene"), before.scenes);
    fillScenes($("post-scene"), after.scenes);
    $("scene-picker").hidden = false;

    const problems = [...before.problems, ...after.problems];
    if (!before.scenes.length || !after.scenes.length) {
      note.textContent = "No scenes in one of those windows — widen the dates" +
        (problems.length ? ` (${problems[0].error})` : "") + ".";
      note.classList.add("error");
    } else {
      note.textContent = `${before.scenes.length} before, ${after.scenes.length} after.` +
        (problems.length ? ` Some sources failed: ${problems[0].error}` : "");
    }
  } catch (error) {
    note.textContent = `Imagery search failed: ${error.message}`;
    note.classList.add("error");
  } finally {
    $("btn-search").disabled = false;
    renderAreaSummary();
  }
}

function fillScenes(select, scenes) {
  select.innerHTML = "";
  scenes.forEach((scene) => {
    const option = document.createElement("option");
    option.value = scene.id;
    option.dataset.provider = scene.provider;
    option.textContent = scene.label;
    select.appendChild(option);
  });
  // Default to the least cloudy scene in each window.
  const best = scenes.reduce((acc, scene) =>
    (acc === null || (scene.cloud_cover ?? 100) < (acc.cloud_cover ?? 100)) ? scene : acc, null);
  if (best) select.value = best.id;
}

/* --------------------------------------------------------- assessment --- */
async function runAssessment() {
  const bbox = state.parcelIds.size ? null : state.bbox;
  const body = {
    pre_scene_id: $("pre-scene").value,
    post_scene_id: $("post-scene").value,
    building_source: $("building-source").value || "auto",
    pre_provider: $("pre-scene").selectedOptions[0]?.dataset.provider,
    post_provider: $("post-scene").selectedOptions[0]?.dataset.provider,
  };
  if (bbox) body.bbox = bbox;
  else body.parcel_ids = [...state.parcelIds];
  const gsd = parseFloat($("gsd").value);
  if (!Number.isNaN(gsd)) body.gsd = gsd;

  $("btn-run").disabled = true;
  $("run-error").hidden = true;
  $("progress").hidden = false;
  setProgress(0.02, "submitting");
  try {
    const job = await api("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.job = job;
    pollJob(job.id);
  } catch (error) {
    failRun(error.message);
  }
}

function setProgress(fraction, message) {
  $("progress-fill").style.width = `${Math.round(fraction * 100)}%`;
  $("progress-text").textContent = message || "";
}

function failRun(message) {
  $("progress").hidden = true;
  const el = $("run-error");
  el.textContent = message;
  el.hidden = false;
  $("btn-run").disabled = false;
}

function pollJob(jobId) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    try {
      const job = await api(`/api/assess/${jobId}`);
      state.job = job;
      setProgress(job.progress, job.message);
      if (job.status === "done") {
        await loadResults(jobId);
      } else if (job.status === "error") {
        failRun(job.error || "assessment failed");
      } else {
        pollJob(jobId);
      }
    } catch (error) {
      failRun(error.message);
    }
  }, 700);
}

async function loadResults(jobId) {
  const [parcels, buildings] = await Promise.all([
    api(`/api/assess/${jobId}/parcels.geojson`),
    api(`/api/assess/${jobId}/buildings.geojson`).catch(() => ({ features: [] })),
  ]);
  state.results = parcels;
  state.buildings = new Map();
  buildings.features.forEach((feature) => {
    const key = feature.properties.parcel_id;
    if (!state.buildings.has(key)) state.buildings.set(key, []);
    state.buildings.get(key).push(feature);
  });

  parcelLayer.clearLayers();
  selectionLayer.clearLayers();
  resultLayer.clearLayers();
  resultLayer.addData(parcels);
  const bounds = resultLayer.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [24, 24] });

  $("progress").hidden = true;
  $("btn-run").disabled = false;
  renderResults(jobId);
  showHint(null);
  $("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResults(jobId) {
  const summary = state.job.summary || {};
  const section = $("step-results");
  section.hidden = false;

  const cards = [
    ["Parcels assessed", fmt.format(summary.parcels || 0)],
    ["Structures", fmt.format(summary.buildings || 0)],
    ["Structures damaged", fmt.format(summary.buildings_damaged || 0)],
    ["Parcels affected", fmt.format(summary.parcels_affected || 0)],
    ["Parcels flooded", fmt.format(summary.parcels_flooded || 0)],
    ["Est. structure loss", money.format(summary.estimated_loss_usd || 0)],
  ];
  $("summary-cards").innerHTML = cards.map(([label, value]) =>
    `<div class="card"><div class="value">${value}</div><div class="label">${label}</div></div>`
  ).join("");

  const counts = summary.buildings_by_class || {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  $("class-bar").innerHTML = CLASS_ORDER.slice().reverse().map((cls) =>
    `<div style="width:${(counts[cls] || 0) / total * 100}%;background:${CLASS_COLORS[cls]}"
          title="${CLASS_LABELS[cls]}: ${counts[cls] || 0}"></div>`
  ).join("");
  $("class-legend").innerHTML = CLASS_ORDER.map((cls) =>
    `<li><span class="swatch" style="background:${CLASS_COLORS[cls]}"></span>` +
    `${CLASS_LABELS[cls]} — ${fmt.format(counts[cls] || 0)}</li>`
  ).join("");

  $("warnings").innerHTML = (state.job.warnings || [])
    .map((w) => `<p>${escapeHtml(w)}</p>`).join("");

  $("dl-parcels-csv").href = `/api/assess/${jobId}/parcels.csv`;
  $("dl-buildings-csv").href = `/api/assess/${jobId}/buildings.csv`;
  $("dl-geojson").href = `/api/assess/${jobId}/parcels.geojson`;

  const rank = (feature) => CLASS_ORDER.indexOf(feature.properties.damage_class);
  const worst = state.results.features
    .filter((f) => f.properties.damage_class !== "none")
    .sort((a, b) => rank(a) - rank(b) ||
                    (b.properties.parcel_score || 0) - (a.properties.parcel_score || 0))
    .slice(0, 60);
  $("results-table").querySelector("tbody").innerHTML = worst.map((feature) => {
    const p = feature.properties;
    return `<tr data-id="${feature.id}"><td>${escapeHtml(p.label || p.pin || "")}</td>` +
      `<td style="text-align:right">${p.buildings_damaged}/${p.buildings_total}</td>` +
      `<td style="text-align:right"><span class="pill" ` +
      `style="background:${CLASS_COLORS[p.damage_class]}">${p.damage_class}</span></td></tr>`;
  }).join("") || `<tr><td class="muted">No damage detected in this area.</td></tr>`;

  $("results-table").querySelectorAll("tr[data-id]").forEach((row) => {
    row.addEventListener("click", () => showDetail(Number(row.dataset.id)));
  });
}

/* -------------------------------------------------------------- detail --- */
function showDetail(parcelId) {
  const feature = state.results &&
    state.results.features.find((f) => f.id === parcelId);
  if (!feature) return;
  const p = feature.properties;
  const buildings = state.buildings.get(parcelId) || [];
  const jobId = state.job.id;

  const rows = [
    ["Damage class", `<span class="pill" style="background:${CLASS_COLORS[p.damage_class]}">${p.damage_class}</span>`],
    ["PIN", p.pin || "—"],
    ["Owner", p.owner || "—"],
    ["Acres", p.acres != null ? p.acres.toFixed(2) : "—"],
    ["Structures", `${p.buildings_damaged} damaged / ${p.buildings_total} total`],
    ["Parcel change score", p.parcel_score != null ? p.parcel_score.toFixed(3) : "—"],
    ["Flooded", p.flooded_fraction != null ? `${(p.flooded_fraction * 100).toFixed(0)}%` : "—"],
    ["Vegetation loss", p.vegetation_loss_fraction != null
      ? `${(p.vegetation_loss_fraction * 100).toFixed(0)}%` : "—"],
    ["Est. structure loss", p.estimated_loss_usd ? money.format(p.estimated_loss_usd) : "—"],
    ["Usable pixels", p.coverage != null ? `${(p.coverage * 100).toFixed(0)}%` : "—"],
    ["Dwelling", p.dwel_desc ? `${p.dwel_desc}${p.dwel_yrblt ? `, ${p.dwel_yrblt}` : ""}` : "—"],
  ];

  const buildingRows = buildings.map((b) => {
    const bp = b.properties;
    return `<div class="bldg-row"><span>${escapeHtml(bp.description || bp.kind)}` +
      `<br><span class="muted">${Math.round(bp.area_m2)} m² · ${bp.confidence} confidence` +
      `${bp.approximate ? " · approximate" : ""}</span></span>` +
      `<span class="pill" style="background:${CLASS_COLORS[bp.damage_class]}">` +
      `${bp.damage_class}</span></div>`;
  }).join("") || `<p class="muted small">No structures recorded on this parcel.</p>`;

  $("detail-body").innerHTML =
    `<h3>${escapeHtml(p.label || `Parcel ${parcelId}`)}</h3>` +
    `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>` +
    `<img src="/api/assess/${jobId}/chip/${parcelId}.png?size=300" alt="before, after and damage">` +
    `<div class="chip-caption"><span>before</span><span>after</span><span>damage</span></div>` +
    `<h3 style="margin-top:14px">Structures</h3>${buildingRows}` +
    (p.note ? `<p class="muted small">${escapeHtml(p.note)}</p>` : "") +
    (p.tax_card ? `<p><a class="btn small" href="${p.tax_card}" target="_blank"
       rel="noopener">County tax card</a></p>` : "");
  $("detail").hidden = false;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

/* --------------------------------------------------------------- boot --- */
async function searchParcels(query) {
  const list = $("search-results");
  if (!query || query.length < 2) { list.hidden = true; return; }
  try {
    const hits = await api(`/api/parcels/search?q=${encodeURIComponent(query)}`);
    list.innerHTML = hits.map((hit) =>
      `<li data-id="${hit.id}" data-lat="${hit.lat}" data-lon="${hit.lon}">` +
      `${escapeHtml(hit.address || hit.pin)}<br>` +
      `<span class="muted">${escapeHtml(hit.owner || "")}</span></li>`).join("");
    list.hidden = hits.length === 0;
    list.querySelectorAll("li").forEach((item) => {
      item.addEventListener("click", () => {
        map.setView([Number(item.dataset.lat), Number(item.dataset.lon)], 17);
        list.hidden = true;
      });
    });
  } catch (error) {
    list.hidden = true;
  }
}

async function boot() {
  initMap();
  try {
    state.health = await api("/api/health");
  } catch (error) {
    $("dataset-line").textContent = `Cannot reach the API: ${error.message}`;
    return;
  }
  if (state.health.status === "no-parcels") {
    $("dataset-line").innerHTML =
      "No parcel data loaded. Run <code>scripts/ingest_parcels.py</code>.";
    return;
  }
  $("dataset-line").textContent =
    `${state.health.dataset} — ${fmt.format(state.health.parcels_loaded)} parcels`;
  if (state.health.extent) {
    const [w, s, e, n] = state.health.extent;
    map.fitBounds([[s, w], [n, e]]);
  }

  [state.providers] = await Promise.all([api("/api/imagery/providers")]);
  renderProviders();

  const sources = await api("/api/buildings/sources");
  $("building-source").innerHTML = sources.map((source) =>
    `<option value="${source.id}"${source.available ? "" : " disabled"}>` +
    `${source.label}${source.available ? "" : " — unavailable"}</option>`).join("");

  $("btn-draw").addEventListener("click", () => setMode("draw"));
  $("btn-pick").addEventListener("click", () => setMode("pick"));
  $("btn-clear").addEventListener("click", clearSelection);
  $("btn-search").addEventListener("click", findImagery);
  $("btn-run").addEventListener("click", runAssessment);
  $("provider-select").addEventListener("change", onProviderChange);
  $("pre-scene").addEventListener("change", renderAreaSummary);
  $("post-scene").addEventListener("change", renderAreaSummary);
  $("cloud-range").addEventListener("input", (event) => {
    $("cloud-value").textContent = event.target.value;
  });
  $("detail-close").addEventListener("click", () => { $("detail").hidden = true; });
  let searchTimer;
  $("search-input").addEventListener("input", (event) => {
    clearTimeout(searchTimer);
    const value = event.target.value;
    searchTimer = setTimeout(() => searchParcels(value), 250);
  });
  refreshParcels();
}

boot();
})();
