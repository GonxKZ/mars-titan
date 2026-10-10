// Curvas por época en pequeños múltiplos, una gráfica por familia de modelo.

import * as fmt from "./format.mjs";
import { curveFacets, crossSection, curveSummary, CURVE_METRICS, SEEDS } from "./model.mjs";
import { LineChart, chartGroup, tokens } from "./charts.mjs";
import { el, legend, setOptions, seedColor, resolveColor, replaceCharts } from "./ui.mjs";
import { parseRange, rangeToken } from "./urlstate.mjs";

const byId = id => document.getElementById(id);

function controls(ctx) {
  const groups = [...ctx.campaignGroups().keys()].sort();
  const series = ctx.state.serie || ctx.selectedSeries() || "";
  setOptions(byId("curve-series"), [["*", "Todas las series"], ...groups.map(id => [id, id])], ctx.state.serie === "*" ? "*" : series);
  const windows = [...new Set(ctx.runs().filter(run => !series || series === "*" || ctx.seriesOf(run) === series).map(run => ctx.windowOf(run)).filter(Boolean))].sort();
  setOptions(byId("curve-window"), [["", "Todas las ventanas"], ...windows.map(w => [w, w])], ctx.state.ventana);
  setOptions(byId("curve-metric"), Object.entries(CURVE_METRICS).map(([key, value]) => [key, value.label]), ctx.state.medida);
  byId("curve-scale").value = ctx.state.escala;
  byId("curve-common").checked = ctx.state.comun !== "no";
  byId("curve-series").onchange = event => ctx.set({ serie: event.target.value, ventana: "", x: "" });
  byId("curve-window").onchange = event => ctx.set({ ventana: event.target.value, x: "" });
  byId("curve-metric").onchange = event => ctx.set({ medida: event.target.value });
  byId("curve-scale").onchange = event => ctx.set({ escala: event.target.value });
  byId("curve-common").onchange = event => ctx.set({ comun: event.target.checked ? "si" : "no" });
  return series === "*" ? null : series;
}

function table(ctx, facets, metric) {
  const head = el("thead", {}, el("tr", {}, ...["Familia", "Ejecución", "Semilla", "Ventana", "Épocas", "Primera", "Última", "Mínima", "Mejor época declarada"].map((label, i) => el("th", { text: label, className: i >= 4 ? "num" : "", attrs: { scope: "col" } }))));
  const rows = [];
  for (const facet of facets) {
    for (const line of facet.lines) {
      const values = line.y.filter(v => v !== null);
      rows.push(el("tr", {},
        el("td", { text: ctx.modelName(facet.model) }),
        el("td", {}, el("span", { text: line.run.run_id }), el("span", { className: "sub", text: line.run.attempt_id })),
        el("td", { text: line.run.seed ?? "Sin dato" }),
        el("td", { text: ctx.windowOf(line.run) ?? "Sin ventana" }),
        el("td", { className: "num", text: fmt.number(values.length) }),
        el("td", { className: "num", text: fmt.significant(values[0], 6) }),
        el("td", { className: "num", text: fmt.significant(values.at(-1), 6) }),
        el("td", { className: "num", text: fmt.significant(Math.min(...values), 6) }),
        el("td", { className: "num", text: line.best ?? "Sin dato" }),
      ));
    }
  }
  byId("curve-table").replaceChildren(el("caption", { className: "visually-hidden", text: `${CURVE_METRICS[metric].label} por ejecución` }), head, el("tbody", {}, ...rows));
}

export function renderCurves(ctx, container) {
  const series = controls(ctx);
  const metric = ctx.state.medida;
  const info = CURVE_METRICS[metric];
  const facets = curveFacets(ctx.runs(), { metric, series, window: ctx.state.ventana || null, seriesOf: run => ctx.seriesOf(run), windowOf: run => ctx.windowOf(run) });
  const t = tokens();
  legend(byId("curve-legend"), [
    ...SEEDS.map(seed => ({ label: `Semilla ${seed}`, color: seedColor(seed) })), { label: "Otra semilla o sin semilla", color: "var(--context)" },
    { label: "Mediana entre ejecuciones, derivada", color: "var(--ink)" }, { label: "Del primer al tercer cuartil", band: true },
  ]);
  const points = facets.reduce((sum, facet) => sum + facet.points, 0);
  const lines = facets.reduce((sum, facet) => sum + facet.lines.length, 0);
  const loading = ctx.store.loading.active ? ` Faltan ${ctx.store.loading.total - ctx.store.loading.done} páginas por cargar.` : "";
  byId("curve-caption").textContent = facets.length
    ? `${fmt.number(lines)} ejecuciones y ${fmt.number(points)} observaciones de ${info.phrase}, en ${info.unit}. Se dibujan todos los puntos y los huecos no se interpolan. La mediana y los cuartiles se calculan en cada época con tres ejecuciones o más. El círculo marca la mejor época declarada por el entrenador.${loading}`
    : `No hay observaciones de ${info.phrase} con estos filtros.${metric !== "mae" ? " Las medidas de entrenamiento y por sesión solo existen en los registros publicados desde octubre de 2026." : ""}${loading}`;
  const common = ctx.state.comun !== "no";
  const positive = facets.flatMap(f => [f.low, f.high]).filter(v => v > 0);
  const range = common && facets.length ? (() => {
    const low = Math.min(...facets.map(f => f.low)), high = Math.max(...facets.map(f => f.high));
    if (ctx.state.escala === "log") return positive.length ? [Math.min(...positive) * .95, high * 1.05] : null;
    const pad = (high - low) * .05 || Math.abs(high) * .05 || 1;
    return [low - pad, high + pad];
  })() : null;
  const group = chartGroup("curvas");
  const zoom = parseRange(ctx.state.x);
  const charts = [];
  const nodes = facets.map(facet => {
    const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": `${info.label} de ${ctx.modelName(facet.model)}. Usa las flechas para recorrer épocas y ejecuciones e Intro para abrir el detalle.` } });
    const node = el("section", { className: "facet" },
      el("div", { className: "facet-head" }, el("h2", { text: ctx.modelName(facet.model) }), el("p", { text: `${fmt.number(facet.lines.length)} ejec. · ${fmt.number(facet.points)} puntos` })),
      plot);
    return { facet, plot, node };
  });
  byId("facets").replaceChildren(...nodes.map(n => n.node));
  for (const { facet, plot } of nodes) {
    const lines = facet.lines.map(line => {
      const color = resolveColor(seedColor(line.run.seed), t);
      return { ...line, color, width: SEEDS.includes(line.run.seed) ? 1.3 : 1 };
    });
    const markers = lines.flatMap((line, lineIndex) => {
      if (line.best === null) return [];
      const index = facet.x.indexOf(line.best);
      return index >= 0 && line.y[index] !== null ? [{ lineIndex, x: line.best, y: line.y[index] }] : [];
    });
    const summary = curveSummary(lines, facet.x.length);
    const chart = new LineChart(plot, {
      x: facet.x, lines, xLabel: "Época", yLabel: info.unit, log: ctx.state.escala === "log", yRange: range, group, live: ctx.live, markers,
      summary: summary.median.some(value => value !== null) ? summary : null,
      describe(index, line) {
        const section = crossSection(lines, index);
        const rows = [];
        if (line && line.y[index] !== null) {
          rows.push([`${fmt.significant(line.y[index], 6)} ${info.unit}`, `${line.run.run_id.slice(-24)} · semilla ${line.run.seed ?? "sin dato"} · ${ctx.windowOf(line.run) ?? "sin ventana"}`, line.color]);
        }
        rows.push([`Época ${fmt.number(facet.x[index])}`, section ? `${section.n} con dato · mediana ${fmt.significant(section.median, 5)} · entre ${fmt.significant(section.min, 5)} y ${fmt.significant(section.max, 5)}` : "Sin observaciones en esta época"]);
        return rows;
      },
      onZoom: range => ctx.set({ x: range ? rangeToken(range.min, range.max) : "" }, { render: false }),
      onPick: line => ctx.openRun(line.run),
    });
    if (zoom) chart.zoom(zoom);
    charts.push(chart);
  }
  replaceCharts(ctx, "curvas", charts);
  table(ctx, facets, metric);
  return container;
}
