export const STATUS_LABELS = Object.freeze({
  blocked: "Bloqueada", queued: "En cola", running: "En curso", paused: "Pausada", completed: "Completada",
  failed: "Fallida", cancelled: "Cancelada", stale: "Sin actualización", unknown: "Estado no informado",
});

export const PHASE_LABELS = Object.freeze({
  prepare: "Preparación", train: "Entrenamiento", validation: "Validación",
  calibration: "Calibración", test: "Prueba final", evaluation: "Evaluación",
});

export const METRIC_KEYS = Object.freeze([
  "mae", "mse", "rank_ic", "coverage_80", "coverage_95", "loss",
  "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "vram_peak_mib",
  "ram_peak_mib", "elapsed_seconds", "samples_per_second",
]);

const RUN_STATUSES = ["blocked", "queued", "running", "paused", "completed", "failed", "cancelled"];
const EMPTY_METRICS = Object.freeze(Object.fromEntries(METRIC_KEYS.map(key => [key, null])));

function fail(path, detail = "valor no válido") {
  throw new TypeError(`${path}: ${detail}.`);
}

function record(value, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(path, "se esperaba un objeto");
  return value;
}

function list(value, path, maximum) {
  if (!Array.isArray(value) || value.length > maximum) fail(path, `se esperaba una lista de hasta ${maximum} elementos`);
  return value;
}

function text(value, path, maximum = 160, nullable = false) {
  if (nullable && value === null) return null;
  if (typeof value !== "string" || !value.trim() || value.length > maximum) fail(path, "texto ausente o demasiado largo");
  return value;
}

function number(value, path, { min = 0, max = Infinity, integer = false, nullable = true } = {}) {
  if (nullable && value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isSafeInteger(value))) {
    fail(path, "se esperaba un número dentro de los límites o null");
  }
  return value;
}

function timestamp(value, path, nullable = false) {
  if (nullable && value === null) return null;
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z$/.test(value)) {
    fail(path, "se esperaba una fecha UTC terminada en Z");
  }
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString().slice(0, 19) !== value.slice(0, 19)) fail(path, "fecha UTC imposible");
  return value;
}

function timestampKey(value) {
  return `${value.slice(0, 19)}.${value.slice(20, -1).padEnd(9, "0")}`;
}

function checkOrder(first, second, path) {
  if (first !== null && second !== null && timestampKey(first) > timestampKey(second)) fail(path, "fecha posterior al límite permitido");
}

function boolean(value, path, nullable = false) {
  if (nullable && value === null) return null;
  if (typeof value !== "boolean") fail(path, "se esperaba true o false");
  return value;
}

function metric(value, path, key) {
  if (key === "rank_ic") return number(value, path, { min: -1, max: 1 });
  if (key.startsWith("coverage_")) return number(value, path, { max: 1 });
  return number(value, path, { min: key === "loss" ? -Infinity : 0 });
}

function validateRun(input, index, modelIds, generatedAt, version = 1) {
  const path = `runs[${index}]`;
  record(input, path);
  const result = {};
  for (const key of ["run_id", "attempt_id", "model_id"]) result[key] = text(input[key], `${path}.${key}`);
  result.variant_id = text(input.variant_id, `${path}.variant_id`, 160, true);
  if (!modelIds.has(result.model_id)) fail(`${path}.model_id`, "modelo no incluido en el catálogo");
  if (input.status != null && !RUN_STATUSES.includes(input.status)) fail(`${path}.status`);
  if (input.phase != null && !Object.hasOwn(PHASE_LABELS, input.phase)) fail(`${path}.phase`);
  result.status = input.status ?? null;
  result.phase = input.phase ?? null;
  result.heartbeat_at = timestamp(input.heartbeat_at, `${path}.heartbeat_at`, true);
  result.started_at = timestamp(input.started_at, `${path}.started_at`, true);
  result.updated_at = timestamp(input.updated_at, `${path}.updated_at`, true);
  checkOrder(result.started_at, result.updated_at, `${path}.started_at`);
  checkOrder(result.started_at, result.heartbeat_at, `${path}.heartbeat_at`);
  for (const key of ["heartbeat_at", "started_at", "updated_at"]) checkOrder(result[key], generatedAt, `${path}.${key}`);
  for (const key of ["completed_steps", "total_steps", "epoch", "max_epochs", "seed"]) {
    result[key] = number(input[key], `${path}.${key}`, { integer: true });
  }
  if (result.total_steps !== null && result.completed_steps !== null && result.completed_steps > result.total_steps) fail(`${path}.completed_steps`, "supera total_steps");
  if (result.epoch !== null && result.max_epochs !== null && result.epoch > result.max_epochs) fail(`${path}.epoch`, "supera max_epochs");
  result.fold = typeof input.fold === "number" ? number(input.fold, `${path}.fold`, { integer: true }) : text(input.fold, `${path}.fold`, 120, true);
  result.comparison_group = text(input.comparison_group, `${path}.comparison_group`, 240, true);
  record(input.metrics, `${path}.metrics`);
  result.metrics = Object.fromEntries(METRIC_KEYS.map(key => [key, metric(input.metrics[key], `${path}.metrics.${key}`, key)]));
  let previousStep = -1;
  let previousTime = null;
  result.history = list(input.history, `${path}.history`, 500).map((entry, historyIndex) => {
    const field = `${path}.history[${historyIndex}]`;
    record(entry, field);
    if (Object.hasOwn(entry, "phase") && entry.phase !== result.phase) fail(field, "el punto pertenece a otra fase");
    if (Object.hasOwn(entry, "attempt_id") && entry.attempt_id !== result.attempt_id) fail(field, "el punto pertenece a otro intento");
    const step = number(entry.step, `${field}.step`, { integer: true, nullable: false });
    const recordedAt = timestamp(entry.recorded_at, `${field}.recorded_at`, version === 2);
    if (step <= previousStep) fail(field, "historia fuera de orden o paso duplicado");
    checkOrder(previousTime, recordedAt, field);
    checkOrder(result.started_at, recordedAt, field);
    checkOrder(recordedAt, result.updated_at, field);
    checkOrder(recordedAt, generatedAt, field);
    if (result.completed_steps !== null && step > result.completed_steps) fail(field, "paso posterior al avance confirmado");
    previousStep = step;
    previousTime = recordedAt;
    return { step, recorded_at: recordedAt, loss: metric(entry.loss, `${field}.loss`, "loss"), mae: metric(entry.mae, `${field}.mae`, "mae") };
  });
  const checkpoint = record(input.checkpoint, `${path}.checkpoint`);
  result.checkpoint = {
    step: number(checkpoint.step, `${path}.checkpoint.step`, { integer: true }),
    saved_at: timestamp(checkpoint.saved_at, `${path}.checkpoint.saved_at`, true),
    resumable: boolean(checkpoint.resumable, `${path}.checkpoint.resumable`, true),
  };
  if (result.completed_steps !== null && result.checkpoint.step !== null && result.checkpoint.step > result.completed_steps) fail(`${path}.checkpoint`, "paso posterior al avance confirmado");
  if (result.checkpoint.resumable && (result.checkpoint.step === null || result.checkpoint.saved_at === null)) fail(`${path}.checkpoint`, "recuperación sin punto guardado");
  checkOrder(result.checkpoint.saved_at, result.updated_at, `${path}.checkpoint.saved_at`);
  checkOrder(result.checkpoint.saved_at, generatedAt, `${path}.checkpoint.saved_at`);
  result.test_released = boolean(input.test_released, `${path}.test_released`);
  if (version === 2) {
    const meta = record(input.metadata, `${path}.metadata`);
    result.metadata = {};
    for (const key of ["campaign", "domain", "method", "history_axis", "progress_time_source"]) result.metadata[key] = text(meta[key], `${path}.${key}`, 96);
    if (!["real", "synthetic", "technical"].includes(meta.domain)) fail(path, "dominio desconocido");
    for (const key of ["train_rows", "validation_rows"]) result.metadata[key] = number(meta[key], `${path}.${key}`, {integer: true});
    for (const key of ["configuration_sha256", "source_sha256", "parent", "error_type"]) result.metadata[key] = text(meta[key], `${path}.${key}`, 96, true);
  }
  return result;
}

export function validateSnapshot(input) {
  record(input, "resumen");
  if (![1, 2].includes(input.schema_version)) fail("schema_version", "versión no admitida");
  if (input.project !== "MARS-TITAN") fail("project", "el resumen pertenece a otro proyecto");
  const generatedAt = timestamp(input.generated_at, "generated_at");
  if (!["no_runs_registered", "available"].includes(input.source_status)) fail("source_status");
  const poll = number(input.poll_interval_seconds, "poll_interval_seconds", { min: 60, max: 3600, integer: true, nullable: false });
  const stale = number(input.stale_after_seconds, "stale_after_seconds", { min: 60, max: 86400, integer: true, nullable: false });
  const modelIds = new Set();
  const models = list(input.models, "models", 64).map((model, index) => {
    const path = `models[${index}]`;
    record(model, path);
    const id = text(model.id, `${path}.id`, 80);
    if (modelIds.has(id)) fail(path, "identificador duplicado");
    modelIds.add(id);
    return { id, name: text(model.name, `${path}.name`, 180), kind: text(model.kind, `${path}.kind`, 80) };
  });
  const identities = new Set();
  const runs = list(input.runs, "runs", 128).map((entry, index) => {
    const validated = validateRun(entry, index, modelIds, generatedAt, input.schema_version);
    const key = JSON.stringify([validated.run_id, validated.attempt_id]);
    if (identities.has(key)) fail(`runs[${index}]`, "ejecución e intento duplicados");
    identities.add(key);
    return validated;
  });
  if ((runs.length === 0) !== (input.source_status === "no_runs_registered")) fail("source_status", "no coincide con el número de ejecuciones");
  return {
    schema_version: input.schema_version, project: "MARS-TITAN", generated_at: generatedAt,
    ...(input.schema_version === 2 ? validatePagination(input) : {}),
    source_status: input.source_status, poll_interval_seconds: poll, stale_after_seconds: stale,
    models, runs, notes: list(input.notes, "notes", 40).map((note, index) => text(note, `notes[${index}]`, 600)),
  };
}

export function displayStatus(run, now = Date.now(), staleAfterSeconds = 180) {
  if (run.status === null) return "unknown";
  if (run.status !== "running") return run.status;
  const age = now - Date.parse(run.heartbeat_at);
  return !Number.isFinite(age) || age < 0 || age > staleAfterSeconds * 1000 ? "stale" : "running";
}

export function progressPercent(run) {
  if (run.completed_steps === null || run.total_steps === null || run.total_steps <= 0) return null;
  return Math.min(100, Math.max(0, run.completed_steps / run.total_steps * 100));
}

export function resultsProtected(run) {
  return run.phase === null || (["test", "evaluation"].includes(run.phase) && (run.test_released !== true || !["completed", "failed", "cancelled"].includes(run.status)));
}

export function publicMetrics(run) {
  return resultsProtected(run) ? { ...EMPTY_METRICS } : { ...run.metrics };
}

export function publicHistory(run) {
  return resultsProtected(run) ? [] : run.history;
}

export function comparableRuns(runs, group, phase, sortMetric = "mae") {
  if (!group || !METRIC_KEYS.includes(sortMetric)) return [];
  return runs.filter(run => run.status === "completed" && run.comparison_group === group && run.phase === phase && !resultsProtected(run)).sort((a, b) => {
    const av = a.metrics[sortMetric];
    const bv = b.metrics[sortMetric];
    if (av === null && bv === null) return a.run_id.localeCompare(b.run_id);
    if (av === null) return 1;
    if (bv === null) return -1;
    return sortMetric === "rank_ic" ? bv - av : av - bv;
  });
}

export function formatValue(value, { digits = 4, style = "decimal" } = {}) {
  return value === null || value === undefined || !Number.isFinite(value) ? "Sin dato" : new Intl.NumberFormat("es-ES", { style, minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

function csvCell(value) {
  let content = value === null || value === undefined ? "" : String(value);
  if (typeof value === "string" && /^[\s\u0000-\u001f]*[=+\-@]/.test(content)) content = `'${content}`;
  return `"${content.replaceAll('"', '""')}"`;
}

export function toCSV(runs, models, now = Date.now(), staleAfterSeconds = 180) {
  const names = new Map(models.map(model => [model.id, model.name]));
  const headers = ["run_id", "attempt_id", "model_id", "model_name", "variant_id", "status", "display_status", "phase", "comparison_group", "fold", "seed", "updated_at", "test_released", ...METRIC_KEYS];
  const rows = runs.map(run => {
    const visible = publicMetrics(run);
    return [run.run_id, run.attempt_id, run.model_id, names.get(run.model_id) ?? run.model_id, run.variant_id,
      run.status, displayStatus(run, now, staleAfterSeconds), run.phase, run.comparison_group, run.fold, run.seed,
      run.updated_at, run.test_released, ...METRIC_KEYS.map(key => visible[key])];
  });
  return [headers, ...rows].map(row => row.map(csvCell).join(",")).join("\r\n");
}

function validatePagination(input) {
  const result = {campaigns: list(input.campaigns ?? [], "campaigns", 64).map(c => {
    record(c, "campaign");
    const counts = record(c.counts, "counts");
    return {id: text(c.id, "campaign.id", 96), domain: text(c.domain, "domain", 20),
      status: text(c.status, "status", 20), planned_runs: number(c.planned_runs, "planned_runs", {integer: true}),
      registered_runs: number(c.registered_runs, "registered_runs", {integer: true}),
      counts: Object.fromEntries(Object.entries(counts).map(([key, value]) => {
        if (![...RUN_STATUSES, "null", "not_started"].includes(key)) fail("counts");
        return [key, number(value, key, {integer: true, nullable: false})];
      }))};
  })};
  if (input.pagination) {
    const p = record(input.pagination, "pagination");
    result.pagination = {total_runs: number(p.total_runs, "total_runs", {integer: true, nullable: false}),
      page_size: number(p.page_size, "page_size", {min: 1, max: 128, integer: true, nullable: false}),
      pages: list(p.pages, "pages", 4096).map(path => {
        if (typeof path !== "string" || !/^pages\/[a-f0-9]{64}\.json$/.test(path)) fail("pages", "ruta no admitida");
        return path;
      })};
  }
  return result;
}
