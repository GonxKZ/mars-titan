// Recursos: telemetría del equipo en directo y coste de cada ejecución según sus recibos.

import { ACTIVITY_LABELS, RAM_SCOPE_LABELS } from "./state.mjs";
import * as fmt from "./format.mjs";
import { resourcePoints, throughputPoints } from "./model.mjs";
import { LongSeriesChart, ScatterPlot, chartGroup, tokens } from "./charts.mjs";
import { el, legend, replaceCharts } from "./ui.mjs";

const byId = id => document.getElementById(id);
const RANGES = { "15m": 900, "1h": 3600, "6h": 21600, "24h": 86400 };
const TILES = [
  { field: "gpu_temp_c", label: "Temperatura de la GPU", unit: "°C" },
  { field: "gpu_util_pct", label: "Uso de la GPU", unit: "%", range: [0, 100] },
  { field: "gpu_mem_used_mib", label: "VRAM ocupada", unit: "GiB", scale: 1 / 1024, total: "gpu_mem_total_mib" },
  { field: "gpu_power_w", label: "Potencia de la GPU", unit: "W" },
  { field: "gpu_sm_clock_mhz", label: "Reloj de los SM", unit: "MHz" },
  { field: "cpu_util_pct", label: "CPU del equipo", unit: "%", range: [0, 100] },
  { field: "ram_used_mib", label: "RAM ocupada", unit: "GiB", scale: 1 / 1024, total: "ram_total_mib" },
  { field: "disk_free_gib", label: "Disco libre", unit: "GiB" },
  { field: "server_cpu_pct", label: "CPU de este servidor", unit: "% de un núcleo" },
];
// Motivos de reducción del reloj que declara NVML. Son lecturas del controlador y la
// página no actúa sobre ellos. La inactividad es normal y no se resalta como aviso.
const GPU_EVENTS = [
  [0x20, "Térmico, software", true], [0x40, "Térmico, hardware", true], [0x4, "Límite de potencia", true],
  [0x8, "Ralentización por hardware", true], [0x80, "Freno de potencia", true], [0x1, "Inactividad", false],
];
let tiles = [];

function windowSlice(ctx) {
  const telemetry = ctx.store.telemetry;
  if (!telemetry?.size) return null;
  const t = telemetry.columns.t.subarray(0, telemetry.size);
  const from = t[t.length - 1] - RANGES[ctx.state.rango];
  let start = 0;
  while (start < t.length - 1 && t[start] < from) start++;
  return { telemetry, start, end: t.length };
}

function series(slice, tile) {
  const column = slice.telemetry.columns[tile.field]?.subarray(slice.start, slice.end);
  if (!column) return null;
  return tile.scale ? column.map(value => value * tile.scale) : column;
}

function events(slice) {
  const column = slice.telemetry.columns.gpu_events?.subarray(slice.start, slice.end);
  const known = column ? [...column].filter(value => !Number.isNaN(value)) : [];
  if (!known.length || Number.isNaN(column[column.length - 1])) return [el("span", { text: "NVML no informa de los motivos de reducción del reloj." })];
  const latest = column[column.length - 1];
  return [el("strong", { text: "Reloj de la GPU reducido por" }), ...GPU_EVENTS.map(([bit, label, warn]) => {
    const share = known.filter(value => (value & bit) !== 0).length / known.length;
    const now = (latest & bit) !== 0;
    return el("span", { className: now && warn ? "event-on" : "", text: `${label}: ${now ? "sí" : "no"}, ${fmt.percent(share, 0)} del intervalo` });
  })];
}

function renderTelemetry(ctx) {
  const live = ctx.store.mode === "live";
  byId("telemetry").hidden = !live;
  byId("telemetry-empty").hidden = live;
  byId("resources-lede").textContent = live
    ? "Lecturas del equipo cada pocos segundos, sin cambiar perfiles, relojes ni ventiladores. Los máximos por ejecución proceden de los recibos."
    : "La telemetría del equipo solo existe en el modo local en directo. Los máximos por ejecución proceden de los recibos.";
  tiles = [];
  if (!live) return [];
  const select = byId("telemetry-range");
  select.value = ctx.state.rango;
  select.onchange = event => ctx.set({ rango: event.target.value });
  const slice = windowSlice(ctx);
  const container = byId("telemetry-tiles");
  if (!slice) { container.replaceChildren(el("p", { className: "caption", text: "Esperando la primera muestra del servidor." })); return []; }
  const status = ctx.store.status;
  byId("telemetry-caption").textContent = `${fmt.number(slice.end - slice.start)} muestras cada ${status?.limits?.telemetry_seconds ?? "?"} s mientras hay una pestaña abierta y cada 30 s sin ella. Fuentes: NVML, /proc y statvfs. GPU: ${status?.sources?.gpu ?? "no disponible"}.`;
  byId("gpu-events").replaceChildren(...events(slice));
  const group = chartGroup("telemetria");
  const t = tokens();
  const x = slice.telemetry.columns.t.subarray(slice.start, slice.end);
  const charts = [];
  container.replaceChildren(...TILES.map((tile, index) => {
    const y = series(slice, tile);
    const value = el("span", { className: "tile-value" });
    const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": `${tile.label}. Usa las flechas para recorrer las muestras.` } });
    const node = el("section", { className: "tile" }, el("div", { className: "tile-head" }, el("h3", { text: tile.label }), value), plot);
    tiles.push({ tile, value, plot });
    if (!y) return node;
    const total = tile.total ? slice.telemetry.columns[tile.total]?.[slice.end - 1] * (tile.scale ?? 1) : null;
    requestAnimationFrame(() => {
      if (!plot.isConnected) return;
      const chart = new LongSeriesChart(plot, {
        x, y, label: tile.label, unit: tile.unit, xLabel: "", color: t.series[0], group, live: ctx.live, time: true, height: 130,
        yRange: tile.range ?? (Number.isFinite(total) ? [0, total] : null),
      });
      tiles[index].chart = chart;
      charts.push(chart);
    });
    return node;
  }));
  updateValues(slice);
  return charts;
}

function updateValues(slice) {
  for (const { tile, value } of tiles) {
    const column = slice.telemetry.columns[tile.field];
    const latest = column?.[slice.end - 1];
    const total = tile.total ? slice.telemetry.columns[tile.total]?.[slice.end - 1] : null;
    const scaled = Number.isFinite(latest) ? latest * (tile.scale ?? 1) : null;
    value.replaceChildren(document.createTextNode(scaled === null ? "Sin dato" : fmt.number(scaled, tile.unit === "GiB" ? 1 : 0)), el("small", { text: ` ${tile.unit}${Number.isFinite(total) ? ` de ${fmt.number(total * (tile.scale ?? 1), 1)}` : ""}` }));
  }
}

// Una muestra nueva solo actualiza los datos de las gráficas visibles, sin rehacerlas.
export function appendTelemetry(ctx) {
  if (ctx.state.vista !== "recursos" || !tiles.length) return;
  ctx.scheduler.schedule("telemetry", () => {
    const slice = windowSlice(ctx);
    if (!slice) return;
    const x = slice.telemetry.columns.t.subarray(slice.start, slice.end);
    for (const entry of tiles) {
      if (!entry.chart?.chart) continue;
      entry.chart.x = x;
      entry.chart.y = series(slice, entry.tile);
      entry.chart.append();
    }
    updateValues(slice);
    byId("gpu-events").replaceChildren(...events(slice));
  });
}

function renderRuns(ctx) {
  const t = tokens();
  const groups = resourcePoints(ctx.runs());
  const colors = [t.series[0], t.series[1], t.series[2]];
  legend(byId("resource-legend"), groups.map((group, i) => ({ label: `${group.label} (${fmt.number(group.points.length)})`, color: `var(--series-${i + 1})` })));
  const plot = byId("resource-plot");
  plot.replaceChildren();
  const total = groups.reduce((sum, group) => sum + group.points.length, 0);
  byId("resource-caption").textContent = total ? `${fmt.number(total)} ejecuciones con duración y VRAM conocidas. Las demás no publican alguna de las dos medidas y no se dibujan.` : "Ninguna ejecución publica a la vez duración y VRAM.";
  const charts = [];
  if (total) {
    charts.push(new ScatterPlot(plot, {
      groups: groups.map((group, i) => ({ color: colors[i], points: group.points.map(point => ({ ...point, x: point.elapsed, y: point.vram })) })),
      xLabel: "Duración", xUnit: "s", yLabel: "VRAM asignada máxima", yUnit: "MiB", logX: true, logY: true, live: ctx.live,
      describe: point => [
        [`${fmt.number(point.vram, 0)} MiB`, "VRAM asignada máxima", colors[point.g]],
        [fmt.duration(point.elapsed), "Duración observada"],
        [point.ram === null ? "Sin dato" : `${fmt.number(point.ram, 0)} MiB`, RAM_SCOPE_LABELS[point.run.metadata?.ram_peak_scope] ?? "RAM máxima, alcance no informado"],
        [ctx.modelName(point.run.model_id), `${ACTIVITY_LABELS[point.run.activity] ?? ""} · ${point.run.run_id.slice(-20)}`],
      ],
      onPick: point => ctx.openRun(point.run),
    }));
  }
  const families = throughputPoints(ctx.runs());
  const strip = byId("throughput-plot");
  strip.replaceChildren();
  const count = families.reduce((sum, family) => sum + family.points.length, 0);
  byId("throughput-caption").textContent = count ? `${fmt.number(count)} épocas de ${fmt.number(families.length)} familias. Eje logarítmico. La dispersión vertical dentro de cada fila solo separa los puntos.` : "Los registros cargados no publican el caudal de entrenamiento por época. El recolector lo publica desde octubre de 2026.";
  if (count) {
    const categories = families.map(family => ctx.modelName(family.model));
    // Desplazamiento determinista por posición para que la misma vista no cambie al redibujar.
    const jitter = index => ((index * 0.6180339887) % 1 - .5) * .6;
    charts.push(new ScatterPlot(strip, {
      groups: [{ color: t.series[0], points: families.flatMap((family, row) => family.points.map((point, i) => ({ ...point, x: point.value, y: row + jitter(i) }))) }],
      categories, xLabel: "Muestras por segundo en entrenamiento", xUnit: "", yLabel: "", logX: true, live: ctx.live, height: Math.max(180, categories.length * 34 + 60),
      describe: point => [
        [`${fmt.number(point.value, 0)} muestras/s`, `Época ${point.epoch}`, t.series[0]],
        [point.seconds === null || point.seconds === undefined ? "Sin dato" : fmt.duration(point.seconds), "Duración del entrenamiento de la época"],
        [ctx.modelName(point.run.model_id), point.run.run_id.slice(-24)],
      ],
      onPick: point => ctx.openRun(point.run),
    }));
  }
  return charts;
}

export function renderResources(ctx) {
  // Las gráficas anteriores se destruyen antes de crear las nuevas, porque las de
  // telemetría se guardan en una lista del módulo que el nuevo dibujo sustituye.
  replaceCharts(ctx, "recursos", []);
  const charts = renderRuns(ctx);
  const telemetry = renderTelemetry(ctx);
  const current = tiles;
  replaceCharts(ctx, "recursos", [...charts, { resize: () => telemetry.forEach(chart => chart.resize()), destroy: () => current.forEach(entry => entry.chart?.destroy()) }]);
}
