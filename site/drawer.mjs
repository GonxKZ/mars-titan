// Detalle de una ejecución en un panel lateral. Reúne todo lo que el recibo publica y
// enlaza a la página de datos de donde sale cada cifra. Las fases reservadas solo
// muestran identidad, estado y avance.

import {
  ACTIVITY_LABELS, PHASE_LABELS, CONDITION_LABELS, RAM_SCOPE_LABELS, FINANCIAL_REASON_LABELS,
  displayStatus, publicMetrics, publicHistory, progressPercent, resultsProtected, formatValue,
} from "./state.mjs";
import * as fmt from "./format.mjs";
import { LineChart, tokens } from "./charts.mjs";
import { el, facts, legend, stateBadge } from "./ui.mjs";

const byId = id => document.getElementById(id);
const DOMAIN_LABELS = { real: "Datos reales", synthetic: "Mundo sintético", technical: "Comprobación técnica" };
const CURVE_LINES = [
  ["mae", "MAE de validación por fila"],
  ["session_mae", "MAE de validación por sesión"],
  ["train_mae", "MAE de entrenamiento"],
];
let chart = null;
let bound = false;

function section(title, ...children) {
  return el("section", { className: "drawer-section" }, el("h3", { text: title }), ...children);
}

function when(value) {
  const stamp = fmt.timestamp(value, { seconds: true });
  return [stamp.text, { title: stamp.utc }];
}

function statusSection(ctx, run) {
  const state = displayStatus(run, ctx.now(), ctx.staleAfter());
  const progress = progressPercent(run);
  const bar = el("div", { className: "progress", attrs: { role: "progressbar", "aria-label": "Avance confirmado", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": progress === null ? null : Math.round(progress) } }, el("span"));
  bar.firstChild.style.width = `${progress ?? 0}%`;
  const heartbeat = Date.parse(run.heartbeat_at ?? "");
  return section("Estado y avance",
    el("div", { className: "drawer-status" }, stateBadge(state), el("span", { className: "mono", text: progress === null ? "Avance no informado" : `${fmt.number(run.completed_steps)} de ${fmt.number(run.total_steps)} pasos · ${fmt.number(progress, 0)} %` })),
    progress === null ? null : bar,
    facts([
      ["Época", run.epoch === null ? null : `${fmt.number(run.epoch)}${run.max_epochs === null ? "" : ` de ${fmt.number(run.max_epochs)}`}`],
      ["Fase", PHASE_LABELS[run.phase] ?? null],
      ["Inicio", ...when(run.started_at)],
      ["Última actualización", ...when(run.updated_at)],
      ["Último latido", run.heartbeat_at ? `${fmt.timestamp(run.heartbeat_at, { seconds: true }).text} (${fmt.age(ctx.now() - heartbeat)})` : null, { title: fmt.timestamp(run.heartbeat_at).utc }],
      ["Recuperación", run.checkpoint.resumable ? `Recuperable desde el paso ${fmt.number(run.checkpoint.step)}` : run.checkpoint.resumable === false ? "Sin punto de recuperación" : null],
      ["Punto guardado", run.checkpoint.saved_at ? fmt.timestamp(run.checkpoint.saved_at, { seconds: true }).text : null, { title: fmt.timestamp(run.checkpoint.saved_at).utc }],
      ["Tipo de error", run.metadata?.error_type ?? (run.status === "failed" ? "No informado" : "Ninguno")],
    ]));
}

function metricsSection(run) {
  const m = publicMetrics(run);
  const meta = run.metadata ?? {};
  const value = (key, digits = 5) => fmt.significant(m[key], digits);
  return section("Medidas del recibo",
    facts([
      ["MAE de validación por fila, retorno residual", value("mae")],
      ["MAE de validación por sesión, retorno residual", value("session_mae")],
      ["MSE de validación", value("mse")],
      ["Pérdida de validación", value("loss")],
      ["Rank IC", value("rank_ic", 4)],
      ["Cobertura del intervalo del 80 %", m.coverage_80 === null ? null : fmt.percent(m.coverage_80)],
      ["Cobertura del intervalo del 95 %", m.coverage_95 === null ? null : fmt.percent(m.coverage_95)],
      ["Latencia p50, p95 y p99, ms", [m.latency_p50_ms, m.latency_p95_ms, m.latency_p99_ms].every(v => v === null) ? null : [m.latency_p50_ms, m.latency_p95_ms, m.latency_p99_ms].map(v => fmt.number(v, 2)).join(" · ")],
      ["VRAM asignada máxima", m.vram_peak_mib === null ? null : `${fmt.number(m.vram_peak_mib, 0)} MiB`],
      [RAM_SCOPE_LABELS[meta.ram_peak_scope] ?? "RAM máxima, alcance no informado", m.ram_peak_mib === null ? null : `${fmt.number(m.ram_peak_mib, 0)} MiB`],
      ["Duración", m.elapsed_seconds === null ? null : fmt.duration(m.elapsed_seconds), { title: m.elapsed_seconds === null ? "" : `${fmt.number(m.elapsed_seconds, 1)} s` }],
      ["Caudal de la última validación", m.samples_per_second === null ? null : `${fmt.number(m.samples_per_second, 0)} muestras/s`],
      // La época 0 es el estado de partida, antes de ajustar. El entrenador la declara
      // cuando ninguna época posterior mejora el criterio de selección.
      ["Mejor época declarada", meta.best_epoch === null || meta.best_epoch === undefined ? null : meta.best_epoch === 0 ? "0, el estado de partida" : fmt.number(meta.best_epoch)],
      ["Parada temprana", meta.stopped_early === true ? "Sí, por paciencia" : meta.stopped_early === false ? "No" : null],
    ]));
}

function curveSection(ctx, run) {
  const history = publicHistory(run);
  if (!history.length) return null;
  const t = tokens();
  const lines = CURVE_LINES.map(([key, label], index) => ({ key, label, color: t.series[index], y: history.map(point => point[key] ?? null), run }))
    .filter(line => line.y.some(value => value !== null));
  if (!lines.length) return null;
  const x = history.map(point => point.step);
  const keys = el("ul", { className: "legend" });
  const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": "Curvas de error por época. Usa las flechas para recorrer las épocas." } });
  const node = section("Curvas por época", keys, plot, el("p", { className: "caption", text: "Valores del recibo sin interpolar. El círculo marca la mejor época declarada por el entrenador." }));
  legend(keys, lines.map((line, index) => ({ label: line.label, color: `var(--series-${index + 1})` })));
  const best = run.metadata?.best_epoch ?? null;
  const markers = best === null ? [] : lines.flatMap((line, lineIndex) => {
    const index = x.indexOf(best);
    return line.key === "mae" && index >= 0 && line.y[index] !== null ? [{ lineIndex, x: best, y: line.y[index] }] : [];
  });
  requestAnimationFrame(() => {
    if (!plot.isConnected) return;
    chart = new LineChart(plot, {
      x, lines, xLabel: "Época", yLabel: "retorno residual", live: ctx.live, markers, height: 220,
      describe(index, line) {
        const rows = lines.filter(item => item.y[index] !== null).map(item => [fmt.significant(item.y[index], 6), item.label, item.color]);
        rows.push([`Época ${fmt.number(x[index])}`, fmt.timestamp(history[index].recorded_at, { seconds: true }).text]);
        if (line && history[index].train_seconds !== null) rows.push([fmt.duration(history[index].train_seconds), "Entrenamiento de la época"]);
        return rows;
      },
    });
  });
  return node;
}

function contextSection(ctx, run) {
  const meta = run.metadata ?? {};
  return section("Contexto",
    facts([
      ["Actividad", ACTIVITY_LABELS[run.activity ?? "initial_training"] ?? run.activity],
      ["Dominio", DOMAIN_LABELS[meta.domain] ?? null],
      ["Condición de los datos", CONDITION_LABELS[meta.condition] ?? null],
      ["Campaña", meta.campaign ?? null],
      ["Serie y ventana", [ctx.seriesOf(run), ctx.windowOf(run)].filter(Boolean).join(" · ") || null],
      ["Variante", run.variant_id],
      ["Semilla", run.seed === null ? null : String(run.seed)],
      ["Grupo de comparación", run.comparison_group],
      ["Modelo padre", meta.parent_model ? `${ctx.modelName(meta.parent_model)}${meta.parent_frozen === true ? ", congelado" : meta.parent_frozen === false ? ", ajustable" : ""}` : null],
      ["Ejecución padre", meta.parent ?? null],
      ["Filas de entrenamiento", meta.train_rows === null || meta.train_rows === undefined ? null : fmt.number(meta.train_rows)],
      ["Filas de validación", meta.validation_rows === null || meta.validation_rows === undefined ? null : fmt.number(meta.validation_rows)],
      ["Moneda", meta.currency ?? null],
    ]));
}

function financialSection(run) {
  const f = run.financial_validation;
  if (!f) return null;
  return section("Validación financiera",
    facts([
      ["Retorno neto tras costes", formatValue(f.net_return, { digits: 2, style: "percent" })],
      ["Caída máxima", formatValue(f.max_drawdown, { digits: 2, style: "percent" })],
      ["Costes", formatValue(f.costs, { digits: 4 })],
      ["Rotación", formatValue(f.turnover, { digits: 2 })],
      ["Pasos del episodio", f.steps === null ? null : fmt.number(f.steps)],
      ["Episodio", f.completed === true ? "Completo" : f.completed === false ? FINANCIAL_REASON_LABELS[f.invalid_reason] ?? "Incompleto" : null],
    ]),
    el("p", { className: "caption", text: "Agregados del motor financiero sin deuda ni posiciones cortas. No son errores de predicción." }));
}

function provenanceSection(ctx, run) {
  const meta = run.metadata ?? {};
  const source = ctx.sourceOf(run);
  const link = source && ctx.store.mode !== "local"
    ? el("a", { className: "mono", text: `data/${source}`, attrs: { href: new URL(`./data/${source}`, document.baseURI).href, target: "_blank", rel: "noopener" } })
    : el("span", { className: "mono", text: ctx.store.mode === "local" ? `Archivo local ${ctx.store.localName}` : "Sin página" });
  const copy = el("button", { className: "button", text: "Copiar enlace a esta ejecución", attrs: { type: "button" } });
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(location.href); copy.textContent = "Enlace copiado"; }
    catch { copy.textContent = "No se pudo copiar. El enlace está en la barra de direcciones"; }
  });
  return section("Procedencia",
    el("dl", { className: "facts" },
      el("div", {}, el("dt", { text: "Página de datos" }), el("dd", {}, link)),
      el("div", {}, el("dt", { text: "Huella del recibo, SHA-256" }), el("dd", { className: meta.source_sha256 ? "" : "missing", text: meta.source_sha256 ?? "Sin dato", attrs: { title: meta.source_sha256 } })),
      el("div", {}, el("dt", { text: "Huella de la configuración, SHA-256" }), el("dd", { className: meta.configuration_sha256 ? "" : "missing", text: meta.configuration_sha256 ?? "Sin dato" })),
      el("div", {}, el("dt", { text: "Origen del eje temporal" }), el("dd", { text: meta.progress_time_source === "receipt_timestamp" ? "Fecha del recibo" : meta.progress_time_source ?? "Sin dato" }))),
    copy);
}

function close(ctx) {
  chart?.destroy();
  chart = null;
  if (ctx.state.ejecucion) ctx.set({ ejecucion: "" }, { render: false });
}

export function openRun(ctx, run) {
  const dialog = byId("drawer");
  if (!bound) {
    bound = true;
    byId("drawer-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => close(ctx));
    // Un clic en el fondo, fuera del panel, también lo cierra.
    dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
  }
  chart?.destroy();
  chart = null;
  byId("drawer-kind").textContent = [ACTIVITY_LABELS[run.activity ?? "initial_training"], PHASE_LABELS[run.phase]].filter(Boolean).join(" · ");
  byId("drawer-title").textContent = ctx.modelName(run.model_id);
  byId("drawer-id").textContent = `${run.run_id} · ${run.attempt_id}`;
  const reserved = resultsProtected(run)
    ? el("p", { className: "notice-reserved", text: "Fase reservada. Las medidas, curvas y recursos de esta ejecución no se publican mientras la prueba final siga sellada." })
    : null;
  byId("drawer-body").replaceChildren(...[
    reserved, statusSection(ctx, run),
    reserved ? null : curveSection(ctx, run),
    reserved ? null : metricsSection(run),
    financialSection(run), contextSection(ctx, run), provenanceSection(ctx, run),
  ].filter(Boolean));
  if (!dialog.open) dialog.showModal();
  byId("drawer-body").scrollTop = 0;
}
