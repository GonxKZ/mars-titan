// Estructuras derivadas para las vistas. Todas son funciones puras sobre los registros
// validados por state.mjs, para poder probarlas sin navegador. Ninguna inventa valores:
// una medida ausente sigue siendo null o NaN y un recuento previsto sin registro se
// muestra como recuento, no como ejecuciones ficticias.

import { displayStatus, isPredictive, publicHistory, publicMetrics } from "./state.mjs";

export const STAGES = Object.freeze({
  neural: "Búsqueda neuronal", tabular: "Referencias tabulares", adaptation: "Adaptación predictiva",
  posttraining: "Posentrenamiento", evaluation: "Evaluación congelada",
});
const STAGE_ORDER = ["neural", "tabular", "adaptation", "posttraining", "evaluation"];
const STAGED = /^(neural|tabular|adaptation|posttraining|evaluation)-(.+?)(?:-fold-(\d{3}))?$/;
export const TERMINAL = new Set(["completed", "failed", "cancelled"]);
export const SEEDS = [42, 43, 44];

// Separa etapa, serie y ventana del identificador de campaña. Solo se agrupan sin ventana
// las campañas cuya serie comparten al menos dos etapas, para no unir archivos distintos
// que por casualidad empiezan igual.
export function groupCampaigns(campaigns) {
  const parsed = campaigns.map(campaign => {
    const match = STAGED.exec(campaign.id);
    return { campaign, stage: match?.[1] ?? null, series: match?.[2] ?? campaign.id, window: match?.[3] ? `fold-${match[3]}` : null };
  });
  const shared = new Map();
  for (const item of parsed) if (item.stage) shared.set(item.series, (shared.get(item.series) ?? new Set()).add(item.stage));
  const groups = new Map();
  for (const item of parsed) {
    const keep = item.stage && (item.window || shared.get(item.series).size > 1);
    const series = keep ? item.series : item.campaign.id;
    const entry = { ...item, stage: keep ? item.stage : null, series };
    if (!groups.has(series)) groups.set(series, []);
    groups.get(series).push(entry);
  }
  return groups;
}

function count(campaign, key) {
  return campaign.counts?.[key] ?? 0;
}

export function seriesSummary(series, members, runs, now, staleAfter) {
  const planned = members.reduce((sum, m) => sum + (m.campaign.planned_runs ?? 0), 0);
  const registered = members.reduce((sum, m) => sum + (m.campaign.registered_runs ?? 0), 0);
  const states = {};
  for (const run of runs) {
    const state = displayStatus(run, now, staleAfter);
    states[state] = (states[state] ?? 0) + 1;
  }
  const completed = members.reduce((sum, m) => sum + count(m.campaign, "completed"), 0);
  const notStarted = members.reduce((sum, m) => sum + count(m.campaign, "not_started"), 0);
  const statuses = new Set(members.map(m => m.campaign.status));
  const status = statuses.has("running") ? "running" : statuses.has("paused") ? "paused"
    : statuses.has("failed") ? "failed" : statuses.has("queued") ? "queued"
    : statuses.has("blocked") ? "blocked" : "completed";
  const latest = runs.reduce((max, run) => Math.max(max, Date.parse(run.updated_at) || 0), 0);
  return {
    id: series, planned, registered, completed, notStarted, states, status, latest: latest || null,
    domains: [...new Set(members.map(m => m.campaign.domain))],
    windows: [...new Set(members.map(m => m.window).filter(Boolean))].sort(),
    stages: STAGE_ORDER.filter(stage => members.some(m => m.stage === stage)),
  };
}

function rowKey(run, stage) {
  return `${stage ?? "otras"}|${run.model_id}`;
}

// Matriz de una serie: filas por etapa y modelo, columnas por ventana y, dentro de cada
// celda, una marca por ejecución ordenada por semilla. Los previstos sin registro se
// cuentan por etapa y ventana.
export function campaignMatrix(members, runs, now, staleAfter) {
  const byCampaign = new Map(members.map(m => [m.campaign.id, m]));
  const windows = [...new Set(members.map(m => m.window ?? "—"))].sort();
  const rows = new Map();
  for (const run of runs) {
    const member = byCampaign.get(run.metadata?.campaign);
    if (!member) continue;
    const key = rowKey(run, member.stage);
    if (!rows.has(key)) rows.set(key, { key, stage: member.stage, model: run.model_id, cells: new Map() });
    const window = member.window ?? "—";
    const row = rows.get(key);
    if (!row.cells.has(window)) row.cells.set(window, []);
    row.cells.get(window).push({ run, state: displayStatus(run, now, staleAfter), seed: run.seed });
  }
  for (const row of rows.values()) {
    for (const marks of row.cells.values()) {
      marks.sort((a, b) => (a.seed ?? Infinity) - (b.seed ?? Infinity) || String(a.run.variant_id).localeCompare(String(b.run.variant_id)) || a.run.run_id.localeCompare(b.run.run_id));
    }
  }
  const pending = new Map();
  for (const member of members) {
    const missing = count(member.campaign, "not_started");
    if (!missing) continue;
    const key = `${member.stage ?? "otras"}|${member.window ?? "—"}`;
    pending.set(key, (pending.get(key) ?? 0) + missing);
  }
  const order = [...STAGE_ORDER, null];
  const sorted = [...rows.values()].sort((a, b) => order.indexOf(a.stage) - order.indexOf(b.stage) || a.model.localeCompare(b.model));
  const stages = order.filter(stage => sorted.some(row => row.stage === stage) || members.some(m => m.stage === stage && count(m.campaign, "not_started")));
  return { windows, rows: sorted, pending, stages, maxMarks: Math.max(1, ...sorted.flatMap(row => [...row.cells.values()].map(cell => cell.length))) };
}

// Tiempo restante si los trabajos pendientes se ejecutan en serie al ritmo mediano de
// las últimas confirmaciones. La mediana de los intervalos resiste pausas largas mejor
// que el cociente entre trabajos y tiempo total. Es una estimación y se rotula como tal.
export function estimateRemaining(completedTimes, remaining, { recent = 200, minimum = 5 } = {}) {
  const times = completedTimes.filter(Number.isFinite).sort((a, b) => a - b).slice(-recent);
  if (!(remaining > 0) || times.length < minimum) return null;
  const gaps = [];
  for (let i = 1; i < times.length; i++) if (times[i] > times[i - 1]) gaps.push(times[i] - times[i - 1]);
  if (gaps.length < minimum - 1) return null;
  gaps.sort((a, b) => a - b);
  const middle = gaps.length >> 1;
  const median = gaps.length % 2 ? gaps[middle] : (gaps[middle - 1] + gaps[middle]) / 2;
  return { seconds: median * remaining / 1000, intervalSeconds: median / 1000, basis: times.length, from: times[0], to: times.at(-1), remaining };
}

export const CURVE_METRICS = Object.freeze({
  mae: { label: "MAE residual de validación por fila", phrase: "MAE residual de validación por fila", unit: "retorno residual", key: "mae" },
  session_mae: { label: "MAE residual de validación por sesión", phrase: "MAE residual de validación por sesión", unit: "retorno residual", key: "session_mae" },
  train_mae: { label: "MAE residual de entrenamiento", phrase: "MAE residual de entrenamiento", unit: "retorno residual", key: "train_mae" },
  loss: { label: "Pérdida del objetivo en validación", phrase: "pérdida del objetivo en validación", unit: "unidades de la pérdida", key: "loss" },
  train_samples_per_second: { label: "Caudal de entrenamiento", phrase: "caudal de entrenamiento", unit: "muestras/s", key: "train_samples_per_second" },
});

// Curvas alineadas por época para pequeños múltiplos. Cada familia comparte el eje x y
// cada ejecución aporta una serie con null donde no tiene observación.
export function curveFacets(runs, { metric = "mae", series = null, window = null, seriesOf = () => null, windowOf = () => null } = {}) {
  const key = CURVE_METRICS[metric]?.key ?? "mae";
  const families = new Map();
  for (const run of runs) {
    if (!isPredictive(run)) continue;
    if (series && seriesOf(run) !== series) continue;
    if (window && windowOf(run) !== window) continue;
    const history = publicHistory(run).filter(point => point[key] !== null && point[key] !== undefined);
    if (!history.length) continue;
    if (!families.has(run.model_id)) families.set(run.model_id, []);
    families.get(run.model_id).push({ run, history });
  }
  const facets = [];
  for (const [model, members] of families) {
    const steps = [...new Set(members.flatMap(m => m.history.map(p => p.step)))].sort((a, b) => a - b);
    const position = new Map(steps.map((step, index) => [step, index]));
    let low = Infinity, high = -Infinity, points = 0;
    const lines = members.map(({ run, history }) => {
      const y = new Array(steps.length).fill(null);
      for (const point of history) {
        y[position.get(point.step)] = point[key];
        low = Math.min(low, point[key]);
        high = Math.max(high, point[key]);
        points++;
      }
      return { run, y, best: run.metadata?.best_epoch ?? null };
    });
    lines.sort((a, b) => (a.run.seed ?? Infinity) - (b.run.seed ?? Infinity) || a.run.run_id.localeCompare(b.run.run_id));
    facets.push({ model, x: steps, lines, low, high, points });
  }
  facets.sort((a, b) => b.lines.length - a.lines.length || a.model.localeCompare(b.model));
  return facets;
}

// Mediana, mínimo y máximo entre ejecuciones en una época. Sirve para el resumen del
// cursor y no se dibuja como si fuera otra ejecución.
export function crossSection(lines, index) {
  const values = lines.map(line => line.y[index]).filter(value => value !== null && value !== undefined).sort((a, b) => a - b);
  if (!values.length) return null;
  const middle = values.length >> 1;
  return { n: values.length, min: values[0], max: values.at(-1), median: values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2 };
}

// Cuantil con interpolación lineal entre estadísticos de orden (tipo 7 de Hyndman y Fan,
// el de R y NumPy por defecto). Espera los valores ya ordenados.
export function quantile(sorted, q) {
  const position = (sorted.length - 1) * q, low = Math.floor(position), high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

// Mediana y cuartiles entre ejecuciones en cada época. Con menos de `minimum` ejecuciones
// con dato la época queda en null, porque una mediana de dos curvas no resume nada.
export function curveSummary(lines, length, minimum = 3) {
  const median = new Array(length).fill(null), low = new Array(length).fill(null), high = new Array(length).fill(null), count = new Array(length).fill(0);
  for (let i = 0; i < length; i++) {
    const values = [];
    for (const line of lines) if (line.y[i] !== null && line.y[i] !== undefined) values.push(line.y[i]);
    count[i] = values.length;
    if (values.length < minimum) continue;
    values.sort((a, b) => a - b);
    median[i] = quantile(values, .5);
    low[i] = quantile(values, .25);
    high[i] = quantile(values, .75);
  }
  return { median, low, high, count };
}

// Diferencia relativa de cada valor con la mediana de su grupo, por ejemplo el MAE de un
// brazo frente a la mediana de su etapa y ventana. Así se comparan brazos dentro de la
// misma ventana sin que la dificultad de cada ventana domine el color. El tramo es
// simétrico hasta el percentil 98 de las diferencias absolutas.
export function relativeToGroup(entries) {
  const groups = new Map();
  for (const { group, value } of entries) {
    if (value === null || value === undefined) continue;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(value);
  }
  const medians = new Map([...groups].map(([group, values]) => [group, quantile(values.sort((a, b) => a - b), .5)]));
  const relative = entries.map(({ group, value }) => {
    const median = medians.get(group);
    return value === null || value === undefined || !(median > 0) ? null : (value - median) / median;
  });
  const magnitudes = relative.filter(value => value !== null).map(Math.abs).sort((a, b) => a - b);
  return { medians, relative, limit: magnitudes.length >= 2 && magnitudes.at(-1) > 0 ? quantile(magnitudes, .98) || magnitudes.at(-1) : null, n: magnitudes.length };
}

// Recuento acumulado de ejecuciones completadas según la fecha de su recibo, para ver el
// ritmo real de una serie. Solo entran las ejecuciones cuya fecha procede del recibo.
export function completionTimeline(runs) {
  const times = runs.filter(run => run.status === "completed" && run.metadata?.progress_time_source === "receipt_timestamp")
    .map(run => Date.parse(run.updated_at) / 1000).filter(Number.isFinite).sort((a, b) => a - b);
  const x = [], y = [];
  for (const [i, time] of times.entries()) {
    // Dos recibos en el mismo milisegundo se funden en un punto para que x sea creciente.
    if (x.length && time <= x.at(-1)) { y[y.length - 1] = i + 1; continue; }
    x.push(time);
    y.push(i + 1);
  }
  return { x: Float64Array.from(x), y: Float64Array.from(y) };
}

export const RESOURCE_GROUPS = Object.freeze([
  { id: "training", label: "Entrenamiento y continuación", activities: ["initial_training", "supervised_continuation"] },
  { id: "adaptation", label: "Adaptación predictiva", activities: ["predictive_adaptation"] },
  { id: "policy", label: "RL, simulación y generación", activities: ["rl", "simulation", "synthetic_generation"] },
]);

export function resourcePoints(runs) {
  return RESOURCE_GROUPS.map(group => {
    const points = runs.filter(run => group.activities.includes(run.activity ?? "initial_training")).map(run => {
      const metrics = publicMetrics(run);
      return { run, elapsed: metrics.elapsed_seconds, vram: metrics.vram_peak_mib, ram: metrics.ram_peak_mib };
    }).filter(point => point.elapsed > 0 && point.vram > 0);
    return { ...group, points };
  });
}

export function throughputPoints(runs) {
  const families = new Map();
  for (const run of runs) {
    for (const point of publicHistory(run)) {
      if (!(point.train_samples_per_second > 0)) continue;
      if (!families.has(run.model_id)) families.set(run.model_id, []);
      families.get(run.model_id).push({ run, epoch: point.step, value: point.train_samples_per_second, seconds: point.train_seconds });
    }
  }
  return [...families].map(([model, points]) => ({ model, points })).sort((a, b) => a.model.localeCompare(b.model));
}

export const POLICY_GROUPS = Object.freeze([
  { id: "ppo", label: "PPO y variantes", test: model => model.startsWith("ppo") },
  { id: "dqn", label: "Double DQN", test: model => model === "double_dqn" },
  { id: "reference", label: "Referencias financieras", test: model => ["cash", "hold_initial", "rebalance_50"].includes(model) },
]);

export function financialPoints(runs) {
  return POLICY_GROUPS.map(group => ({
    ...group,
    points: runs.filter(run => run.financial_validation && group.test(run.model_id)).map(run => ({ run, ...run.financial_validation }))
      .filter(point => point.net_return !== null && point.max_drawdown !== null),
  }));
}

export function statusTotals(runs, now, staleAfter) {
  const totals = {};
  for (const run of runs) {
    const state = displayStatus(run, now, staleAfter);
    totals[state] = (totals[state] ?? 0) + 1;
  }
  return totals;
}

// Campaña por ventanas leída por el servidor local. Las celdas usan vocabularios para que
// el evento ocupe poco y aquí se convierten en filas de ámbito y brazo.
export function liveCampaignMatrix(state) {
  const { scopes, windows, arms, names } = state.vocabulary;
  const rows = new Map();
  const confirmed = [];
  for (const [scope, window, arm, name, status, confirmedAt] of state.cells) {
    const key = `${scopes[scope]}|${arms[arm]}`;
    if (!rows.has(key)) rows.set(key, { key, scope: scopes[scope], arm: arms[arm], cells: new Map() });
    const cell = rows.get(key).cells;
    if (!cell.has(windows[window])) cell.set(windows[window], []);
    const label = names[name];
    const seed = /-s(\d+)$/.exec(label)?.[1];
    cell.get(windows[window]).push({ name: label, seed: seed ? Number(seed) : label.startsWith("search-") ? 42 : null, state: status });
    if (status === "done" && confirmedAt) confirmed.push(Date.parse(confirmedAt));
  }
  for (const row of rows.values()) for (const marks of row.cells.values()) marks.sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0) || a.name.localeCompare(b.name));
  const total = state.cells.length, done = state.cells.filter(cell => cell[4] === "done").length;
  return {
    windows: [...windows].sort(), rows: [...rows.values()].sort((a, b) => a.key.localeCompare(b.key)),
    total, done, attempts: state.cells.filter(cell => cell[4] === "attempt").length,
    estimate: estimateRemaining(confirmed, total - done),
  };
}
