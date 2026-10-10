// Vista de campaña: series, etapas y matriz de ventana por brazo y semilla.

import { STATUS_LABELS, ACTIVITY_LABELS, WINDOW_STAGES, publicMetrics } from "./state.mjs";
import * as fmt from "./format.mjs";
import { campaignMatrix, seriesSummary, windowCampaignMatrix, completionTimeline, quantile, relativeToGroup, STAGES } from "./model.mjs";
import { LineChart, LongSeriesChart, chartGroup, cividis, cividisGradient, diverging, divergingGradient, tokens } from "./charts.mjs";
import { el, stateMark, legend, setOptions, replaceCharts, STATE_LABELS } from "./ui.mjs";
import { runToken } from "./urlstate.mjs";

const byId = id => document.getElementById(id);
const DOMAINS = { real: "corpus real", synthetic: "mundo sintético", technical: "comprobación técnica" };
const LEGEND = ["completed", "running", "stale", "paused", "failed", "queued", "blocked", "cancelled"];
let previousMarks = null;

function seriesList(ctx, current) {
  const groups = ctx.campaignGroups();
  const now = Date.now(), stale = ctx.staleAfter();
  const runs = ctx.runs();
  const bySeries = new Map();
  for (const run of runs) {
    const series = ctx.seriesOf(run);
    if (!series) continue;
    if (!bySeries.has(series)) bySeries.set(series, []);
    bySeries.get(series).push(run);
  }
  const summaries = [...groups].map(([series, members]) => seriesSummary(series, members, bySeries.get(series) ?? [], now, stale));
  const weight = { running: 0, paused: 1, failed: 2, queued: 3, blocked: 4, completed: 5 };
  summaries.sort((a, b) => weight[a.status] - weight[b.status] || (b.latest ?? 0) - (a.latest ?? 0) || a.id.localeCompare(b.id));
  const items = summaries.map(summary => {
    const share = summary.planned ? summary.completed / summary.planned : 0;
    const meter = el("span", { className: summary.status === "running" ? "meter-active" : ["paused", "failed"].includes(summary.status) ? "meter-warn" : "" });
    meter.style.width = `${Math.round(share * 1000) / 10}%`;
    return el("li", {}, el("button", {
      attrs: { type: "button", "aria-current": String(summary.id === current) },
      on: { click: () => ctx.set({ serie: summary.id, celda: "" }) },
    },
    el("span", { className: "series-name", text: summary.id }),
    el("span", { className: "meter", attrs: { "aria-hidden": "true" } }, meter),
    el("span", { className: "series-meta" },
      el("span", { className: "state" }, stateMark(summary.status), el("span", { text: STATUS_LABELS[summary.status] ?? summary.status })),
      el("span", { className: "mono", text: `${fmt.number(summary.completed)}/${fmt.number(summary.planned)}` })),
    ));
  });
  byId("series-items").replaceChildren(...items);
  setOptions(byId("series-select"), summaries.map(s => [s.id, `${s.id} · ${fmt.number(s.completed)}/${fmt.number(s.planned)}`]), current);
  byId("series-select").onchange = event => ctx.set({ serie: event.target.value, celda: "" });
  return { summaries, bySeries };
}

function pipeline(ctx, members) {
  const stages = new Map();
  for (const member of members) {
    const key = member.stage ?? "otras";
    if (!stages.has(key)) stages.set(key, []);
    stages.get(key).push(member.campaign);
  }
  const order = ["neural", "tabular", "adaptation", "posttraining", "evaluation", "otras"];
  const rows = [...stages].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0])).map(([stage, campaigns]) => {
    const planned = campaigns.reduce((sum, c) => sum + (c.planned_runs ?? 0), 0);
    const done = campaigns.reduce((sum, c) => sum + (c.counts?.completed ?? 0), 0);
    const statuses = {};
    for (const c of campaigns) statuses[c.status] = (statuses[c.status] ?? 0) + 1;
    const paused = campaigns.reduce((sum, c) => sum + (c.counts?.paused ?? 0), 0);
    const blockedBy = [...new Set(campaigns.filter(c => c.status === "blocked").flatMap(c => c.dependencies ?? []))];
    const meter = el("span", { className: paused ? "meter-warn" : "" });
    meter.style.width = planned ? `${Math.round(done / planned * 1000) / 10}%` : "0";
    const state = Object.entries(statuses).map(([status, count]) => `${count} ${(STATUS_LABELS[status] ?? status).toLowerCase()}`).join(" · ");
    return el("li", {},
      el("div", { className: "stage-name", text: STAGES[stage] ?? "Sin etapa declarada" }),
      el("div", { className: "stage-count" }, el("span", { text: fmt.number(done) }), el("small", { text: ` / ${fmt.number(planned)}` })),
      el("div", { className: "meter", attrs: { role: "img", "aria-label": `${fmt.number(done)} completados de ${fmt.number(planned)} previstos` } }, meter),
      el("div", { className: "stage-state", text: campaigns.length > 1 ? `${campaigns.length} ventanas: ${state}` : state }),
      blockedBy.length ? el("div", { className: "stage-state", text: `Depende de ${blockedBy.length > 2 ? `${blockedBy.length} campañas` : blockedBy.join(" y ")}` }) : null,
    );
  });
  byId("pipeline").replaceChildren(...rows);
}

// MAE de validación que colorea una marca. Solo cuentan las ejecuciones completadas, para
// no mezclar el último valor de una ejecución en curso con resultados finales.
function finalMae(run) {
  return run.status === "completed" ? publicMetrics(run).mae : null;
}

// Escala de color de las marcas. En modo absoluto va del percentil 2 al 98 de la serie
// para que un valor extremo no aplaste a los demás. En modo relativo compara cada MAE
// con la mediana de su etapa y ventana. Los valores fuera del tramo toman el color del
// extremo y la cifra exacta sigue en el título de la marca y en el detalle.
function markScale(ctx, runs, mode) {
  if (mode === "estado") return null;
  const dark = tokens().dark;
  const values = runs.map(finalMae);
  const known = values.filter(value => value !== null).sort((a, b) => a - b);
  if (known.length < 2) return { empty: true };
  if (mode === "mae") {
    const low = quantile(known, .02), high = quantile(known, .98);
    const byRun = new Map(runs.map((run, i) => [run, values[i]]));
    return {
      color: run => byRun.get(run) === null ? null : `rgb(${cividis((byRun.get(run) - low) / (high - low || 1), dark).join(",")})`,
      text: run => byRun.get(run) === null ? "" : ` · MAE ${fmt.significant(byRun.get(run), 5)}`,
      cell: cell => {
        const list = cell.map(mark => byRun.get(mark.run)).filter(value => value !== null).sort((a, b) => a - b);
        return list.length ? `, MAE mediano ${fmt.significant(quantile(list, .5), 4)}` : ", sin MAE final";
      },
      gradient: cividisGradient(dark), low: fmt.significant(low, 4), high: fmt.significant(high, 4),
      caption: `Color según el MAE residual de validación por fila de ${fmt.number(known.length)} ejecuciones completadas, del percentil 2 al 98 de la serie. Un MAE menor es mejor. La cifra exacta aparece en el título de cada marca y en el detalle.`,
    };
  }
  const result = relativeToGroup(runs.map((run, i) => ({ group: `${ctx.stageOf(run) ?? ""}|${ctx.windowOf(run) ?? ""}`, value: values[i] })));
  if (result.limit === null) return { empty: true };
  const byRun = new Map(runs.map((run, i) => [run, result.relative[i]]));
  return {
    color: run => byRun.get(run) === null ? null : `rgb(${diverging(byRun.get(run) / result.limit, dark).join(",")})`,
    text: run => byRun.get(run) === null ? "" : ` · ${fmt.percent(byRun.get(run), 1)} frente a la mediana de su ventana`,
    cell: cell => {
      const list = cell.map(mark => byRun.get(mark.run)).filter(value => value !== null).sort((a, b) => a - b);
      return list.length ? `, diferencia mediana ${fmt.percent(quantile(list, .5), 1)} frente a su ventana` : ", sin MAE final";
    },
    gradient: divergingGradient(dark), low: `−${fmt.percent(result.limit, 1)}`, high: `+${fmt.percent(result.limit, 1)}`,
    caption: `Diferencia relativa del MAE final de ${fmt.number(result.n)} ejecuciones completadas con la mediana de su etapa y ventana. Azul indica menos error que esa mediana y bermellón más. El tramo es simétrico hasta el percentil 98 de las diferencias. Es una lectura descriptiva sin prueba de significación.`,
  };
}

function matrixTable(ctx, members, runs, selectedCell, scale) {
  const matrix = campaignMatrix(members, runs, Date.now(), ctx.staleAfter());
  const table = byId("matrix");
  const marks = new Map();
  const coloured = scale && !scale.empty;
  const head = el("thead", {}, el("tr", {}, el("th", { text: "Brazo", attrs: { scope: "col" } }), ...matrix.windows.map(window => el("th", { text: window === "—" ? "Sin ventana" : window, attrs: { scope: "col" } }))));
  const body = el("tbody");
  for (const stage of matrix.stages) {
    body.append(el("tr", { className: "stage-row" }, el("th", { text: STAGES[stage] ?? "Otras", attrs: { scope: "rowgroup", colspan: matrix.windows.length + 1 } })));
    for (const row of matrix.rows.filter(r => r.stage === stage)) {
      const total = [...row.cells.values()].reduce((sum, cell) => sum + cell.length, 0);
      const tr = el("tr", {}, el("th", { attrs: { scope: "row" } }, el("span", { text: ctx.modelName(row.model) }), el("small", { text: `${fmt.number(total)} registros` })));
      for (const window of matrix.windows) {
        const cell = row.cells.get(window) ?? [];
        const key = `${row.key}|${window}`;
        if (!cell.length) { tr.append(el("td", {}, el("span", { className: "cell cell-empty", text: "", attrs: { "aria-label": "Sin registros" } }))); continue; }
        const counts = {};
        for (const mark of cell) counts[mark.state] = (counts[mark.state] ?? 0) + 1;
        let summary = Object.entries(counts).map(([state, count]) => `${count} ${(STATE_LABELS[state] ?? state).toLowerCase()}`).join(", ");
        if (coloured) summary += scale.cell(cell);
        const button = el("button", {
          className: "cell",
          attrs: { type: "button", "aria-pressed": String(selectedCell === key), "aria-label": `${ctx.modelName(row.model)}, ${window}: ${cell.length} registros, ${summary}` },
          on: { click: () => ctx.set({ celda: selectedCell === key ? "" : key }) },
        });
        for (const mark of cell) {
          const token = runToken(mark.run);
          const color = coloured ? scale.color(mark.run) : null;
          const node = stateMark(mark.state, `${mark.run.run_id} · semilla ${mark.seed ?? "sin dato"} · ${STATE_LABELS[mark.state]}${coloured ? scale.text(mark.run) : ""}`);
          if (scale) {
            node.dataset.v = color === null ? "none" : "value";
            if (color !== null) node.style.background = color;
          }
          if (previousMarks && previousMarks.get(token) !== mark.state) node.classList.add("m-new");
          marks.set(token, mark.state);
          button.append(node);
        }
        tr.append(el("td", {}, button));
      }
      body.append(tr);
    }
    const pendingRow = matrix.windows.map(window => matrix.pending.get(`${stage ?? "otras"}|${window}`) ?? 0);
    if (pendingRow.some(Boolean)) {
      body.append(el("tr", {}, el("th", { attrs: { scope: "row" } }, el("span", { text: "Previstos sin registro" }), el("small", { text: "recuento del plan" })),
        ...pendingRow.map(count => el("td", {}, count ? el("span", { className: "pending-count", text: `+${fmt.number(count)}` }) : null))));
    }
  }
  table.replaceChildren(head, body);
  previousMarks = marks;
  return matrix;
}

function cellDetail(ctx, matrix, key) {
  const box = byId("cell-detail");
  if (!key) { box.hidden = true; return; }
  const [stage, model, window] = key.split("|");
  const row = matrix.rows.find(r => r.key === `${stage}|${model}`);
  const cell = row?.cells.get(window);
  if (!cell) { box.hidden = true; return; }
  box.hidden = false;
  byId("cell-title").textContent = `${ctx.modelName(model)} · ${STAGES[stage] ?? "Otras"} · ${window}`;
  byId("cell-runs").replaceChildren(...cell.map(mark => el("li", {}, el("button", {
    attrs: { type: "button" }, on: { click: () => ctx.openRun(mark.run) },
  },
  stateMark(mark.state),
  el("span", { text: `Semilla ${mark.seed ?? "sin dato"} · ${ACTIVITY_LABELS[mark.run.activity] ?? "Actividad no informada"}${mark.run.metadata?.method && mark.run.metadata.method !== mark.run.activity ? ` · ${mark.run.metadata.method}` : ""}` }),
  el("span", { className: "mono", text: fmt.significant(mark.run.metrics.mae) === fmt.MISSING ? STATE_LABELS[mark.state] : `MAE ${fmt.significant(mark.run.metrics.mae)}` }),
  ))));
}

const MODES = ["estado", "mae", "relativo"];

function marksControl(ctx, scale) {
  const mode = ctx.state.marcas;
  for (const button of document.querySelectorAll("[data-marks]")) {
    const active = button.dataset.marks === mode;
    button.setAttribute("aria-checked", String(active));
    button.tabIndex = active ? 0 : -1;
    button.onclick = () => ctx.set({ marcas: button.dataset.marks });
    button.onkeydown = event => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      const next = MODES[(MODES.indexOf(mode) + step + MODES.length) % MODES.length];
      ctx.set({ marcas: next });
      requestAnimationFrame(() => document.querySelector(`[data-marks="${next}"]`)?.focus());
    };
  }
  const ramp = byId("matrix-ramp");
  byId("matrix-legend").hidden = Boolean(scale);
  ramp.hidden = !scale;
  if (!scale) {
    byId("matrix-caption").textContent = "Cada marca es un registro. Dentro de cada celda, las semillas 42, 43 y 44 van en orden.";
    return;
  }
  const hollow = el("span", { className: "state" }, el("i", { className: "m", dataset: { s: "pending", v: "none" }, attrs: { "aria-hidden": "true" } }), el("span", { text: "Sin MAE final publicado" }));
  if (scale.empty) {
    ramp.replaceChildren(hollow);
    byId("matrix-caption").textContent = "Esta serie no tiene al menos dos ejecuciones completadas con MAE publicado, así que no hay escala que mostrar.";
    return;
  }
  const bar = el("span", { className: "ramp-bar", attrs: { "aria-hidden": "true" } });
  bar.style.background = scale.gradient;
  ramp.replaceChildren(el("span", { className: "ramp" }, el("span", { className: "mono", text: scale.low }), bar, el("span", { className: "mono", text: scale.high })), hollow);
  byId("matrix-caption").textContent = scale.caption;
}

function pace(ctx, runs, planned) {
  const figure = byId("pace");
  const timeline = completionTimeline(runs);
  if (timeline.x.length < 2) { figure.hidden = true; return []; }
  figure.hidden = false;
  const first = fmt.timestamp(timeline.x[0] * 1000), last = fmt.timestamp(timeline.x.at(-1) * 1000);
  byId("pace-caption").textContent = `${fmt.number(timeline.y.at(-1))} ejecuciones completadas con fecha de recibo, entre ${first.text} y ${last.text}. El techo del eje es el total previsto, ${fmt.number(planned)}. Un tramo plano es una pausa.`;
  const plot = byId("pace-plot");
  plot.replaceChildren();
  return [new LongSeriesChart(plot, {
    x: timeline.x, y: timeline.y, label: "Completadas acumuladas", unit: "ejecuciones", xLabel: "", color: tokens().series[0],
    live: ctx.live, time: true, height: 150, yRange: [0, Math.max(planned, timeline.y.at(-1))],
  })];
}

// Campañas por ventanas. Se dibuja una cada vez, porque la de adaptadores de A tiene más
// de 300 filas. En directo el estado llega por SSE y en Pages solo se descarga el
// documento de la campaña elegida.
function windowCampaigns(ctx) {
  const section = byId("window-campaigns");
  const options = ctx.windowCampaignList();
  section.hidden = !options.length;
  if (!options.length) return [];
  // Sin elección en la URL se abre la campaña en directo o, si no hay, la de resumen más reciente.
  const latest = [...options].sort((a, b) => Number(b.live) - Number(a.live) || (b.updated_at ?? "").localeCompare(a.updated_at ?? ""))[0];
  const selected = options.some(option => option.id === ctx.state.matriz) ? ctx.state.matriz : latest.id;
  setOptions(byId("window-select"), options.map(option => [option.id,
    `${option.id} · ${WINDOW_STAGES[option.stage] ?? "etapa no declarada"} · ${option.path === null && !option.live ? "sin resumen" : `${fmt.number(option.done)} de ${fmt.number(option.jobs)}`}`]), selected);
  byId("window-select").onchange = event => ctx.set({ matriz: event.target.value });
  const option = options.find(item => item.id === selected);
  const state = ctx.windowCampaign(option);
  const container = byId("window-campaign");
  const source = option.configuration ? ` Declaración: ${option.configuration}.` : "";
  if (!state.available) {
    const reason = state.loading ? "Cargando la matriz de trabajos."
      : state.error ? `No se pudo leer la matriz: ${state.error}. Se reintentará en la próxima consulta.`
      : "Sin resumen todavía: la etapa no ha empezado o su carpeta no está disponible para el recolector.";
    container.replaceChildren(el("p", { className: "caption", text: `${reason}${source}` }));
    return [];
  }
  const charts = [];
  const matrix = windowCampaignMatrix(state);
  const head = el("thead", {}, el("tr", {}, el("th", { text: "Ámbito · brazo", attrs: { scope: "col" } }), ...matrix.windows.map(w => el("th", { text: w, attrs: { scope: "col" } }))));
  const body = el("tbody", {}, ...matrix.rows.map(row => el("tr", {},
    el("th", { attrs: { scope: "row" } }, el("span", { text: row.arm }), el("small", { text: `${row.scope} · ${ctx.modelName(row.model ?? "unknown")}` })),
    ...matrix.windows.map(window => {
      const marks = row.cells.get(window) ?? [];
      return el("td", {}, el("span", { className: "cell", attrs: { role: "img", "aria-label": `${row.arm}, ${row.scope}, ${window}: ${marks.length} trabajos, ${marks.filter(m => m.state === "done").length} confirmados` } },
        ...marks.map(mark => stateMark(mark.state, `${mark.name} · ${STATE_LABELS[mark.state]}`))));
    }))));
  const estimate = matrix.estimate;
  const updated = state.updated_at ? ` Resumen escrito ${fmt.timestamp(state.updated_at).text}.` : "";
  container.replaceChildren(el("div", { className: "live-campaign" },
    el("p", { className: "eyebrow", text: `${state.id} · ${WINDOW_STAGES[state.stage] ?? "etapa no declarada"} · ${STATUS_LABELS[state.status] ?? state.status ?? "estado no declarado"}${option.live ? " · en directo" : ""}` }),
    el("p", { className: "caption", text: `${fmt.number(matrix.done)} de ${fmt.number(matrix.total)} trabajos confirmados. ${fmt.number(matrix.attempts)} con intento sin confirmar. ${estimate ? `Estimación: ${fmt.duration(estimate.seconds)} al ritmo mediano de ${estimate.basis} confirmaciones, según la fecha de modificación de cada recibo.` : "Sin estimación del tiempo restante."}${updated}${source}` }),
    el("div", { className: "matrix-scroll window-matrix", attrs: { tabindex: "0", role: "region", "aria-label": `Matriz de ${state.id}` } }, el("table", { className: "matrix" }, head, body)),
    activeAttempts(ctx, state.active, charts),
  ));
  return charts;
}

// Intentos abiertos de una campaña por ventanas, con sus curvas por época tal como las
// escribe run.json. Un intento sin confirmar no significa que el proceso siga vivo, y
// la fecha de modificación del archivo es la única señal de actividad que se muestra.
function activeAttempts(ctx, active, charts) {
  if (!active.length) return null;
  const t = tokens();
  const group = chartGroup("intentos");
  return el("div", { className: "attempts" }, ...active.map(attempt => {
    const lines = [["mae", "MAE de validación", t.series[0]], ["train_mae", "MAE de entrenamiento", t.series[2]]]
      .map(([key, label, color]) => ({ key, label, color, y: attempt.epochs.map(epoch => epoch[key]) }))
      .filter(line => line.y.some(value => value !== null));
    const x = attempt.epochs.map((epoch, i) => epoch.epoch ?? i + 1);
    const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": `Curvas del intento ${attempt.job}` } });
    const node = el("figure", { className: "attempt" },
      el("figcaption", {}, el("span", { className: "mono", text: attempt.job }),
        el("span", { className: "caption", text: `${attempt.attempt} · ${fmt.number(attempt.epochs.length)} épocas · ${attempt.updated_at ? `informe modificado ${fmt.timestamp(attempt.updated_at, { seconds: true }).text}` : "sin informe por épocas legible"}` })),
      lines.length ? plot : el("p", { className: "caption", text: "Todavía sin épocas con medidas." }));
    if (lines.length && x.every((value, i) => i === 0 || value > x[i - 1])) {
      requestAnimationFrame(() => {
        if (!plot.isConnected) return;
        charts.push(new LineChart(plot, {
          x, lines, xLabel: "", yLabel: "retorno residual", group, live: ctx.live, height: 140,
          describe: index => lines.filter(line => line.y[index] !== null).map(line => [fmt.significant(line.y[index], 5), line.label, line.color]).concat([[`Época ${x[index]}`, attempt.job]]),
        }));
      });
    }
    return node;
  }));
}

export function renderCampaign(ctx) {
  const index = ctx.store.index;
  if (!index) return;
  const series = ctx.selectedSeries();
  const { bySeries } = seriesList(ctx, series);
  legend(byId("matrix-legend"), LEGEND.map(state => ({ state, label: STATE_LABELS[state] })));
  replaceCharts(ctx, "campana", []);
  const liveCharts = windowCampaigns(ctx);
  const liveEntry = { resize: () => liveCharts.forEach(chart => chart.resize()), destroy: () => liveCharts.forEach(chart => chart.destroy()) };
  if (!series) {
    byId("series-title").textContent = "Sin campañas registradas";
    byId("pace").hidden = true;
    replaceCharts(ctx, "campana", [liveEntry]);
    return;
  }
  const members = ctx.campaignGroups().get(series) ?? [];
  const runs = bySeries.get(series) ?? [];
  const domains = [...new Set(members.map(m => DOMAINS[m.campaign.domain] ?? m.campaign.domain))];
  byId("series-kind").textContent = `Serie · ${domains.join(" y ")}`;
  byId("series-title").textContent = series;
  const windows = [...new Set(members.map(m => m.window).filter(Boolean))];
  const { summary, estimate, remaining } = ctx.seriesEstimate(series);
  byId("series-lede").textContent = [
    `${members.length} ${members.length === 1 ? "campaña" : "campañas"}${windows.length ? ` en ${windows.length} ventanas` : ""}.`,
    `${fmt.number(summary.completed)} de ${fmt.number(summary.planned)} ejecuciones previstas completadas, ${fmt.number(summary.registered)} registradas.`,
    estimate ? `Si las ${fmt.number(remaining)} restantes se ejecutan en serie al ritmo mediano reciente, faltarían unas ${fmt.duration(estimate.seconds)}. Es una estimación, no una medida.` : "",
  ].filter(Boolean).join(" ");
  pipeline(ctx, members);
  replaceCharts(ctx, "campana", [...pace(ctx, runs, summary.planned), liveEntry]);
  const scale = markScale(ctx, runs, ctx.state.marcas);
  marksControl(ctx, scale);
  const matrix = matrixTable(ctx, members, runs, ctx.state.celda, scale);
  const loading = ctx.store.loading;
  byId("matrix-loading").textContent = loading.active ? `Cargando el historial: ${loading.done} de ${loading.total} páginas. La matriz se completa a medida que llegan.` : loading.errors ? `${loading.errors} páginas sin leer. Las marcas que faltan no se dan por ausentes.` : "";
  cellDetail(ctx, matrix, ctx.state.celda);
}
