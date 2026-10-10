// Registro completo de ejecuciones con búsqueda, filtros, orden y exportación CSV. La
// tabla crece de cien en cien porque miles de filas en el DOM ralentizan el resto de la
// página, y el CSV siempre incluye todas las filas filtradas.

import { ACTIVITY_LABELS, displayStatus, publicMetrics, progressPercent, resultsProtected, toCSV } from "./state.mjs";
import * as fmt from "./format.mjs";
import { el, stateBadge, setOptions, STATE_LABELS } from "./ui.mjs";

const byId = id => document.getElementById(id);
const PAGE = 100;
let limit = PAGE;
let sort = { key: "updated", direction: -1 };
let lastFilter = "";

const COLUMNS = [
  { key: "status", label: "Estado", value: (run, ctx) => STATE_LABELS[displayStatus(run, ctx.now(), ctx.staleAfter())] },
  { key: "model", label: "Modelo", value: (run, ctx) => ctx.modelName(run.model_id) },
  { key: "run", label: "Ejecución", value: run => run.run_id },
  { key: "activity", label: "Actividad", value: run => ACTIVITY_LABELS[run.activity ?? "initial_training"] ?? run.activity },
  { key: "window", label: "Serie y ventana", value: (run, ctx) => `${ctx.seriesOf(run) ?? ""} ${ctx.windowOf(run) ?? ""}` },
  { key: "seed", label: "Semilla", value: run => run.seed, numeric: true },
  { key: "progress", label: "Progreso", value: run => progressPercent(run), numeric: true },
  { key: "mae", label: "MAE de validación", value: run => publicMetrics(run).mae, numeric: true },
  { key: "elapsed", label: "Duración", value: run => publicMetrics(run).elapsed_seconds, numeric: true },
  { key: "updated", label: "Actualizado", value: run => Date.parse(run.updated_at) || null, numeric: true },
];

function matches(run, ctx, query) {
  if (!query) return true;
  const text = [run.run_id, run.attempt_id, run.model_id, ctx.modelName(run.model_id), run.variant_id, run.metadata?.campaign, ctx.seriesOf(run), ctx.windowOf(run)].filter(Boolean).join(" ").toLowerCase();
  return query.split(/\s+/).every(term => text.includes(term));
}

function filtered(ctx) {
  const { buscar, estado, actividad } = ctx.state;
  const query = buscar.trim().toLowerCase();
  const runs = ctx.runs().filter(run =>
    (!estado || displayStatus(run, ctx.now(), ctx.staleAfter()) === estado)
    && (!actividad || (run.activity ?? "initial_training") === actividad)
    && matches(run, ctx, query));
  const column = COLUMNS.find(item => item.key === sort.key);
  const values = new Map(runs.map(run => [run, column.value(run, ctx)]));
  // Las ausencias van siempre al final, en cualquier sentido de orden.
  return runs.sort((a, b) => {
    const av = values.get(a), bv = values.get(b);
    if (av === null || av === undefined) return bv === null || bv === undefined ? 0 : 1;
    if (bv === null || bv === undefined) return -1;
    return (column.numeric ? av - bv : String(av).localeCompare(String(bv), "es")) * sort.direction;
  });
}

function controls(ctx) {
  const search = byId("records-search");
  if (document.activeElement !== search) search.value = ctx.state.buscar;
  search.oninput = () => {
    clearTimeout(search.timer);
    search.timer = setTimeout(() => ctx.set({ buscar: search.value.slice(0, 120) }), 180);
  };
  const totals = new Map();
  for (const run of ctx.runs()) {
    const state = displayStatus(run, ctx.now(), ctx.staleAfter());
    totals.set(state, (totals.get(state) ?? 0) + 1);
  }
  setOptions(byId("records-status"), [["", "Todos"], ...[...totals].sort((a, b) => b[1] - a[1]).map(([state, count]) => [state, `${STATE_LABELS[state] ?? state} (${fmt.number(count)})`])], ctx.state.estado);
  const activities = new Set(ctx.runs().map(run => run.activity ?? "initial_training"));
  setOptions(byId("records-activity"), [["", "Todas"], ...[...activities].sort().map(id => [id, ACTIVITY_LABELS[id] ?? id])], ctx.state.actividad);
  byId("records-status").onchange = event => ctx.set({ estado: event.target.value });
  byId("records-activity").onchange = event => ctx.set({ actividad: event.target.value });
}

function header(ctx) {
  return el("thead", {}, el("tr", {}, ...COLUMNS.map(column => {
    const active = sort.key === column.key;
    return el("th", { className: column.numeric ? "num" : "", attrs: { scope: "col", "aria-sort": active ? (sort.direction > 0 ? "ascending" : "descending") : null } },
      el("button", {
        text: `${column.label}${active ? (sort.direction > 0 ? " ↑" : " ↓") : ""}`, attrs: { type: "button" },
        on: { click: () => { sort = { key: column.key, direction: active ? -sort.direction : column.numeric ? -1 : 1 }; ctx.invalidate("registros"); } },
      }));
  })));
}

function row(ctx, run) {
  const state = displayStatus(run, ctx.now(), ctx.staleAfter());
  const metrics = publicMetrics(run);
  const updated = fmt.timestamp(run.updated_at);
  const progress = progressPercent(run);
  const cell = (text, numeric = true) => el("td", { className: `${numeric ? "num" : ""}${text === fmt.MISSING ? " missing" : ""}`, text });
  // La campaña ya aparece en su columna. Aquí basta con lo que distingue a la ejecución
  // dentro de ella y el identificador completo queda en el título y en el detalle.
  const campaign = run.metadata?.campaign;
  const short = campaign && run.run_id.startsWith(`${campaign}-`) ? `…${run.run_id.slice(campaign.length + 1)}` : run.run_id;
  const open = el("button", { className: "link mono", text: short, attrs: { type: "button", title: run.run_id, "aria-label": `Abrir el detalle de ${run.run_id}` }, on: { click: event => { event.stopPropagation(); ctx.openRun(run); } } });
  const node = el("tr", { on: { click: () => ctx.openRun(run) } },
    el("td", {}, stateBadge(state)),
    el("td", { text: ctx.modelName(run.model_id) }),
    el("td", {}, open, el("span", { className: "sub", text: run.attempt_id })),
    el("td", { text: ACTIVITY_LABELS[run.activity ?? "initial_training"] ?? run.activity }),
    el("td", {}, el("span", { text: ctx.seriesOf(run) ?? run.metadata?.campaign ?? "Sin campaña" }), el("span", { className: "sub", text: ctx.windowOf(run) ?? "" })),
    cell(run.seed === null || run.seed === undefined ? fmt.MISSING : String(run.seed)),
    cell(progress === null ? fmt.MISSING : `${fmt.number(progress, 0)} %`),
    resultsProtected(run) ? el("td", { className: "num missing", text: "Reservado" }) : cell(fmt.significant(metrics.mae, 5)),
    cell(fmt.duration(metrics.elapsed_seconds)),
    el("td", { className: "num", text: updated.text, attrs: { title: updated.utc } }),
  );
  return node;
}

export function renderRecords(ctx) {
  controls(ctx);
  const filterKey = `${ctx.state.buscar}|${ctx.state.estado}|${ctx.state.actividad}|${sort.key}|${sort.direction}`;
  if (filterKey !== lastFilter) { limit = PAGE; lastFilter = filterKey; }
  const runs = filtered(ctx);
  const total = ctx.runs().length;
  const loading = ctx.store.loading.active ? ` Faltan ${fmt.number(ctx.store.loading.total - ctx.store.loading.done)} páginas del historial por cargar.` : "";
  byId("records-count").textContent = `${fmt.number(runs.length)} de ${fmt.number(total)} ejecuciones cargadas${runs.length > limit ? `, se muestran las ${fmt.number(limit)} primeras` : ""}.${loading}`;
  byId("records").replaceChildren(
    el("caption", { className: "visually-hidden", text: "Ejecuciones registradas" }),
    header(ctx),
    el("tbody", {}, ...runs.slice(0, limit).map(run => row(ctx, run))));
  const more = byId("records-more");
  more.hidden = runs.length <= limit;
  more.textContent = `Mostrar ${fmt.number(Math.min(PAGE, runs.length - limit))} más`;
  more.onclick = () => { limit += PAGE; ctx.invalidate("registros"); };
  byId("records-csv").onclick = () => {
    const csv = toCSV(runs, ctx.store.index?.models ?? [], ctx.now(), ctx.staleAfter());
    const link = el("a", { attrs: { href: URL.createObjectURL(new Blob([`﻿${csv}`], { type: "text/csv;charset=utf-8" })), download: `mars-titan-ejecuciones-${new Date().toISOString().slice(0, 10)}.csv` } });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  };
}
