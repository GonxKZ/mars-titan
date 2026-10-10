// Trazas de aprendizaje: lo que ocurre dentro de Titans, del banco episódico y del
// optimizador paso a paso. Los paquetes los escribe el productor de #448 con
// traces.py y aquí solo se leen. Si no hay paquetes, la vista lo dice y no dibuja nada.

import * as fmt from "./format.mjs";
import { TRACE_GROUPS, validateIndex, validateManifest, decodeBundle } from "./traces.mjs";
import { readLimited } from "./sources.mjs";
import { LongSeriesChart, Heatmap, chartGroup, cividisGradient, tokens } from "./charts.mjs";
import { el, setOptions, replaceCharts } from "./ui.mjs";

const byId = id => document.getElementById(id);
const BASE = new URL("./data/traces/", import.meta.url);
const MAX_MANIFEST = 4 * 1024 * 1024;
const MAX_BLOB = 256 * 1024 * 1024;
// Se guardan como mucho dos paquetes decodificados. Uno de un millón de puntos ocupa
// unos 12 MB y la vista solo enseña uno cada vez.
const CACHE_LIMIT = 2;
const cache = new Map();
let index = { version: -1, bundles: null, error: "" };
let pending = null;

async function fetchBytes(url, maxBytes) {
  const response = await fetch(url, { cache: "no-store", credentials: "omit" });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`HTTP ${response.status} al leer ${url.pathname.split("/").pop()}`);
  return readLimited(response, maxBytes);
}

async function loadIndex(ctx) {
  if (index.version === ctx.store.tracesVersion && index.bundles) return index.bundles;
  const bytes = await fetchBytes(new URL("index.json", BASE), MAX_MANIFEST);
  index = { version: ctx.store.tracesVersion, bundles: bytes ? validateIndex(JSON.parse(new TextDecoder().decode(bytes))) : [], error: "" };
  return index.bundles;
}

async function loadBundle(entry) {
  const manifestBytes = await fetchBytes(new URL(entry.manifest, BASE), MAX_MANIFEST);
  if (!manifestBytes) throw new Error(`El manifiesto ${entry.manifest} ya no existe`);
  const manifest = validateManifest(JSON.parse(new TextDecoder().decode(manifestBytes)));
  const key = `${manifest.name}:${manifest.sha256}`;
  if (cache.has(key)) return cache.get(key);
  const started = performance.now();
  const blob = await fetchBytes(new URL(manifest.blob, BASE), Math.min(MAX_BLOB, manifest.bytes));
  if (!blob) throw new Error(`El bloque ${manifest.blob} ya no existe`);
  const buffer = blob.byteOffset === 0 && blob.byteLength === blob.buffer.byteLength ? blob.buffer : blob.slice().buffer;
  const bundle = await decodeBundle(manifest, buffer);
  bundle.loadMs = performance.now() - started;
  try { performance.measure("mt:trace-load", { start: started, end: performance.now() }); } catch { /* sin medidas */ }
  cache.set(key, bundle);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return bundle;
}

function bundleLabel(entry) {
  return [entry.name, entry.modelId, entry.provenance === "fixture" ? "prueba" : null].filter(Boolean).join(" · ");
}

function rampLegend(range, unit) {
  const ramp = el("span", { className: "ramp-bar", attrs: { "aria-hidden": "true" } });
  ramp.style.background = cividisGradient(tokens().dark);
  return el("div", { className: "ramp" }, el("span", { className: "mono", text: fmt.significant(range[0], 4) }), ramp, el("span", { className: "mono", text: `${fmt.significant(range[1], 4)} ${unit}` }));
}

function heatmapPanel(ctx, bundle, matrix, charts) {
  const plot = el("div", { className: "plot", attrs: { role: "img", "aria-label": `${matrix.label}. Mapa de calor de ${matrix.rows.length} filas por ${fmt.number(matrix.x.length)} posiciones.` } });
  const caption = el("p", { className: "caption" });
  let low = Infinity, high = -Infinity, missing = 0;
  for (const value of matrix.values) {
    if (Number.isNaN(value)) { missing++; continue; }
    if (value < low) low = value;
    if (value > high) high = value;
  }
  const range = matrix.range ?? [low, high];
  const node = el("figure", { className: "trace-panel trace-wide" },
    el("figcaption", {}, el("h3", { text: matrix.label }), rampLegend(range, matrix.unit)),
    plot, caption);
  requestAnimationFrame(() => {
    if (!plot.isConnected) return;
    charts.push(new Heatmap(plot, {
      x: matrix.x, rows: matrix.rows, values: matrix.values, range, unit: matrix.unit, label: matrix.label, xLabel: bundle.xLabel, live: ctx.live,
      onStats: ({ reduced }) => {
        caption.textContent = `${fmt.number(matrix.values.length)} valores. ${reduced ? `Cada columna de píxel muestra la media de unas ${fmt.number(matrix.x.length / Math.max(1, plot.clientWidth - 80), 0)} posiciones. El cursor lee el valor exacto.` : "Se dibujan todas las posiciones."}${missing ? ` ${fmt.number(missing)} celdas sin dato quedan en el color de la rejilla.` : ""}${matrix.range ? " Rango de color fijado por el productor." : " Rango de color del mínimo al máximo observados."}`;
      },
    }));
  });
  return node;
}

function seriesPanel(ctx, bundle, series, group, charts) {
  const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": `${series.label}. Usa las flechas para recorrer los puntos.` } });
  const stats = el("span", { className: "caption" });
  const toggle = el("input", { attrs: { type: "checkbox" } });
  const node = el("figure", { className: "trace-panel" },
    el("figcaption", {}, el("h3", { text: series.label }), el("span", { className: "caption", text: series.unit })),
    plot,
    el("div", { className: "trace-foot" }, stats, el("label", { className: "check" }, toggle, el("span", { text: "Dibujar todos" }))));
  requestAnimationFrame(() => {
    if (!plot.isConnected) return;
    const t = tokens();
    const chart = new LongSeriesChart(plot, {
      x: series.x, y: series.y, label: series.label, unit: series.unit, xLabel: bundle.xLabel, color: t.series[0], group, live: ctx.live, height: 170,
      onStats: ({ drawn, total, full }) => {
        stats.textContent = drawn < total
          ? `${fmt.number(drawn)} de ${fmt.number(total)} puntos dibujados por reducción M4 en el tramo visible. Los picos se conservan.`
          : `${fmt.number(total)} puntos${full ? ", sin reducir" : ", todos dibujados"}.`;
      },
    });
    toggle.addEventListener("change", () => chart.setFull(toggle.checked));
    charts.push(chart);
  });
  return node;
}

function renderBundle(ctx, bundle) {
  const charts = [];
  const group = chartGroup("trazas");
  const run = ctx.runs().find(item => item.run_id === bundle.runId && item.attempt_id === bundle.attemptId);
  const runLink = run
    ? el("button", { className: "link", text: bundle.runId, attrs: { type: "button" }, on: { click: () => ctx.openRun(run) } })
    : el("span", { className: "mono", text: bundle.runId });
  byId("trace-caption").replaceChildren(
    document.createTextNode(`${fmt.number(bundle.points)} valores de `), runLink,
    document.createTextNode(` · ${ctx.modelName(bundle.modelId)} · eje: ${bundle.xLabel.toLowerCase()}${bundle.cadence ? `, un registro cada ${fmt.number(bundle.cadence)}` : ""} · ${bundle.verified ? `huella SHA-256 ${fmt.shortHash(bundle.sha256)} comprobada` : "huella sin comprobar en un origen no seguro"} · leído en ${fmt.number(bundle.loadMs, 0)} ms.${bundle.truncatedAt != null ? ` El productor dejó de registrar en el paso ${fmt.number(bundle.truncatedAt)} al agotar su presupuesto de bytes, así que las series terminan ahí aunque el entrenamiento siguiera.` : ""}`),
  );
  byId("trace-fixture").hidden = bundle.provenance !== "fixture";
  const sections = Object.entries(TRACE_GROUPS).map(([id, label]) => {
    const matrices = bundle.matrices.filter(item => item.group === id);
    const series = bundle.series.filter(item => item.group === id);
    if (!matrices.length && !series.length) return null;
    return el("section", { className: "trace-group", attrs: { "aria-label": label } },
      el("h2", { text: label }),
      el("div", { className: "trace-grid" },
        ...matrices.map(matrix => heatmapPanel(ctx, bundle, matrix, charts)),
        ...series.map(item => seriesPanel(ctx, bundle, item, group, charts))));
  }).filter(Boolean);
  byId("trace-panels").replaceChildren(...sections);
  return charts;
}

function showEmpty(message) {
  byId("trace-filters").hidden = true;
  byId("trace-fixture").hidden = true;
  byId("trace-panels").replaceChildren(...(message ? [el("p", { className: "notice", text: message })] : []));
  byId("trace-empty").hidden = false;
}

// La carga es asíncrona. Si llega otra petición de dibujo mientras tanto, la anterior
// se descarta al terminar para no pintar un paquete que ya no está seleccionado.
export function renderTraces(ctx) {
  replaceCharts(ctx, "memoria", []);
  const ticket = {};
  pending = ticket;
  (async () => {
    let bundles;
    // En directo el servidor declara si tiene carpeta de trazas. Sin ella no se pide el
    // índice, para no llenar la consola de respuestas 404 esperadas.
    if (ctx.store.mode === "live" && !ctx.store.status?.sources?.traces) { showEmpty(""); return; }
    try {
      bundles = await loadIndex(ctx);
    } catch (error) {
      if (pending === ticket) showEmpty(`No se pudo leer el índice de trazas: ${error.message}.`);
      return;
    }
    if (pending !== ticket) return;
    if (!bundles.length) { showEmpty(""); return; }
    byId("trace-empty").hidden = true;
    byId("trace-filters").hidden = false;
    const select = byId("trace-select");
    const chosen = bundles.find(entry => entry.name === ctx.state.traza) ?? bundles.at(-1);
    setOptions(select, bundles.map(entry => [entry.name, bundleLabel(entry)]), chosen.name);
    select.onchange = event => ctx.set({ traza: event.target.value });
    byId("trace-caption").textContent = "Leyendo el paquete y comprobando su huella.";
    try {
      const bundle = await loadBundle(chosen);
      if (pending !== ticket) return;
      replaceCharts(ctx, "memoria", renderBundle(ctx, bundle));
    } catch (error) {
      if (pending !== ticket) return;
      byId("trace-panels").replaceChildren(el("p", { className: "notice", text: `No se muestra el paquete ${chosen.name}: ${error.message}` }));
      byId("trace-caption").textContent = "";
    }
  })();
}
