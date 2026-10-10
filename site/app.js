// Núcleo del observatorio: origen de los datos, estado de la vista y planificación del
// dibujo. Las vistas viven en módulos propios y reciben un contexto común, de modo que
// todas leen los mismos registros validados y ninguna consulta la red por su cuenta.

import { validateSnapshot, displayStatus, STATUS_LABELS } from "./state.mjs";
import * as fmt from "./format.mjs";
import { groupCampaigns, statusTotals, seriesSummary, estimateRemaining, TERMINAL } from "./model.mjs";
import { ConditionalResource, PageStore, LiveStream } from "./sources.mjs";
import { FrameScheduler, throttle } from "./scheduler.mjs";
import { parseHash, serializeHash, runToken } from "./urlstate.mjs";
import { renderCampaign } from "./view-campaign.mjs";
import { renderCurves } from "./view-curves.mjs";
import { renderResources, appendTelemetry } from "./view-resources.mjs";
import { renderTraces } from "./view-traces.mjs";
import { renderPolicies } from "./view-policies.mjs";
import { renderRecords } from "./view-records.mjs";
import { openRun } from "./drawer.mjs";
import { playFlow } from "./view-method.mjs";

const byId = id => document.getElementById(id);
const VIEWS = ["campana", "curvas", "recursos", "memoria", "rl", "registros", "metodo"];
const LOOPBACK = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);
const TELEMETRY_CAPACITY = 17_280;

const store = {
  mode: "public",
  index: null,
  deployment: null,
  runs: new Map(),
  runSource: new Map(),
  pageRuns: new Map(),
  loading: { done: 0, total: 0, errors: 0, active: false },
  lastCheck: null,
  lastChange: null,
  nextCheck: null,
  error: "",
  liveState: "closed",
  lastEvent: null,
  status: null,
  telemetry: null,
  liveCampaigns: new Map(),
  tracesVersion: 0,
  localName: "",
  version: 0,
};

const scheduler = new FrameScheduler();
let state = parseHash(location.hash);
const indexResource = new ConditionalResource(new URL("./data/observatory.json", import.meta.url));
const deploymentResource = new ConditionalResource(new URL("./data/deployment.json", import.meta.url), { maxBytes: 4096 });
const pages = new PageStore(new URL("./data/", import.meta.url), validateSnapshot);
let pollTimer = null, ageTimer = null, healthTimer = null, stream = null, hiddenSince = null;
const rendered = new Map();

export const ctx = {
  store,
  get state() { return state; },
  scheduler,
  // `render: false` solo actualiza la URL. Lo usa el zoom, que ya ha movido las gráficas.
  set(patch, { push = false, render = true } = {}) {
    const next = { ...state, ...patch };
    const changedView = next.vista !== state.vista;
    state = next;
    writeHash(push || changedView);
    if (changedView) showView();
    else if (render) invalidate(state.vista);
  },
  now: () => Date.now(),
  staleAfter: () => store.index?.stale_after_seconds ?? 900,
  modelName(id) { return store.index?.models.find(model => model.id === id)?.name ?? id; },
  runs() { return [...store.runs.values()]; },
  run(token) { return store.runs.get(token) ?? null; },
  campaignGroups() { return memo("groups", () => groupCampaigns(store.index?.campaigns ?? [])); },
  campaignInfo() {
    return memo("campaignInfo", () => {
      const info = new Map();
      for (const [series, members] of ctx.campaignGroups()) for (const member of members) info.set(member.campaign.id, { series, stage: member.stage, window: member.window });
      return info;
    });
  },
  seriesOf(run) { return ctx.campaignInfo().get(run.metadata?.campaign)?.series ?? null; },
  windowOf(run) { return ctx.campaignInfo().get(run.metadata?.campaign)?.window ?? null; },
  stageOf(run) { return ctx.campaignInfo().get(run.metadata?.campaign)?.stage ?? null; },
  sourceOf(run) { return store.runSource.get(runToken(run)) ?? null; },
  openRun(run) {
    ctx.set({ ejecucion: runToken(run) });
    openRun(ctx, run);
  },
  live: byId("chart-live"),
  invalidate: view => invalidate(view),
};

const memos = new Map();
function memo(key, build) {
  const entry = memos.get(key);
  if (entry && entry.version === store.version) return entry.value;
  const value = build();
  memos.set(key, { version: store.version, value });
  return value;
}

const writeHash = throttle(push => {
  const hash = serializeHash(state);
  if (hash === (location.hash || "#")) return;
  if (push) history.pushState(null, "", hash);
  else history.replaceState(null, "", hash);
}, 250);

// Cada origen de aviso se guarda aparte. Así una consulta correcta del índice no borra
// el aviso de que la conexión en directo se ha cerrado, ni al revés.
const notices = new Map();
function showError(message, source = "index") {
  if (message) notices.set(source, message);
  else notices.delete(source);
  const text = [...notices.values()].join(" ");
  store.error = text;
  byId("error").textContent = text;
  byId("error").hidden = !text;
}

// Cambiar de vista solo dibuja la vista visible. Las demás quedan marcadas y se dibujan
// al mostrarse, para no gastar CPU en gráficas que nadie ve.
function showView() {
  if (!VIEWS.includes(state.vista)) state.vista = "campana";
  for (const view of VIEWS) {
    const tab = byId(`tab-${view}`), panel = byId(view);
    const active = view === state.vista;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.hidden = !active;
  }
  invalidate(state.vista);
}

function invalidate(view = state.vista) {
  rendered.delete(view);
  if (view === state.vista) scheduler.schedule(`view:${view}`, () => renderView(view));
}

function invalidateAll() {
  store.version++;
  rendered.clear();
  scheduler.schedule("pulse", renderPulse);
  scheduler.schedule(`view:${state.vista}`, () => renderView(state.vista));
}

const RENDERERS = {
  campana: renderCampaign, curvas: renderCurves, recursos: renderResources, memoria: renderTraces,
  rl: renderPolicies, registros: renderRecords, metodo: () => {},
};

function renderView(view) {
  if (rendered.get(view) === store.version || view !== state.vista) return;
  const started = performance.now();
  try {
    RENDERERS[view](ctx, byId(view));
    rendered.set(view, store.version);
    if (notices.has("view")) showError("", "view");
  } catch (error) {
    showError(`No se pudo dibujar la vista: ${error.message}`, "view");
    globalThis.reportError?.(error);
  }
  try { performance.measure(`mt:view:${view}`, { start: started, end: performance.now() }); } catch { /* sin medidas */ }
}

function setText(id, text, missing = false) {
  const node = byId(id);
  if (node.textContent !== text) node.textContent = text;
  node.classList.toggle("missing", missing);
}

function selectedSeries() {
  const groups = ctx.campaignGroups();
  if (state.serie && groups.has(state.serie)) return state.serie;
  return defaultSeries();
}

export function defaultSeries() {
  return memo("defaultSeries", () => {
    let best = null, latest = -1;
    for (const [series, members] of ctx.campaignGroups()) {
      const ids = new Set(members.map(member => member.campaign.id));
      let stamp = 0;
      for (const run of store.runs.values()) if (ids.has(run.metadata?.campaign)) stamp = Math.max(stamp, Date.parse(run.updated_at) || 0);
      const active = members.some(member => ["running", "paused"].includes(member.campaign.status));
      const score = stamp + (active ? 1e13 : 0);
      if (score > latest) { latest = score; best = series; }
    }
    return best;
  });
}
ctx.selectedSeries = selectedSeries;

export function seriesEstimate(series) {
  const members = ctx.campaignGroups().get(series) ?? [];
  const ids = new Set(members.map(member => member.campaign.id));
  const runs = ctx.runs().filter(run => ids.has(run.metadata?.campaign));
  const summary = seriesSummary(series, members, runs, Date.now(), ctx.staleAfter());
  const remaining = Math.max(0, summary.planned - summary.completed);
  const times = runs.filter(run => run.status === "completed" && run.metadata?.progress_time_source === "receipt_timestamp").map(run => Date.parse(run.updated_at));
  return { summary, estimate: estimateRemaining(times, remaining), remaining };
}
ctx.seriesEstimate = seriesEstimate;

function renderPulse() {
  const index = store.index;
  if (!index) return;
  const runs = ctx.runs();
  const total = index.pagination?.total_runs ?? index.runs.length;
  setText("pulse-runs", fmt.number(total));
  const pageCount = (index.pagination?.pages.length ?? 0) + 1;
  // pageRuns incluye el índice, que cuenta como la primera página.
  const loaded = store.mode === "local" ? 1 : store.pageRuns.size;
  setText("pulse-runs-note", `${fmt.number(index.campaigns?.length ?? 0)} campañas · ${loaded === pageCount ? `${pageCount} páginas cargadas` : `${loaded} de ${pageCount} páginas cargadas`}`);
  const totals = statusTotals(runs, Date.now(), ctx.staleAfter());
  const running = totals.running ?? 0, stale = totals.stale ?? 0, paused = totals.paused ?? 0;
  setText("pulse-active", `${fmt.number(running)} en curso`);
  const blocked = (index.campaigns ?? []).filter(c => c.status === "blocked").length;
  setText("pulse-active-note", [stale && `${fmt.number(stale)} sin actualización`, paused && `${fmt.number(paused)} en pausa`, blocked && `${fmt.number(blocked)} campañas bloqueadas`].filter(Boolean).join(" · ") || "Sin pausas ni bloqueos");
  const series = selectedSeries();
  if (series) {
    const { summary, estimate, remaining } = seriesEstimate(series);
    setText("pulse-series", summary.planned ? fmt.percent(summary.completed / summary.planned, 0) : fmt.MISSING, !summary.planned);
    setText("pulse-series-note", `${series} · ${fmt.number(summary.completed)} de ${fmt.number(summary.planned)} previstos`);
    if (estimate) {
      setText("pulse-eta", `≈ ${fmt.duration(estimate.seconds)}`);
      setText("pulse-eta-note", `${fmt.number(remaining)} pendientes al ritmo mediano de ${fmt.number(estimate.basis)} confirmaciones, en serie`);
    } else {
      setText("pulse-eta", remaining ? "Sin estimación" : "Sin pendientes", true);
      setText("pulse-eta-note", remaining ? "Faltan confirmaciones con fecha para estimar el ritmo" : "La serie no tiene trabajos previstos sin registrar");
    }
  }
  const released = runs.filter(run => run.test_released).length;
  setText("pulse-test", released ? `${fmt.number(released)} publicadas` : "Sellada");
  renderGpuPulse();
}

function renderGpuPulse() {
  const sample = store.telemetry?.latest;
  if (store.mode !== "live" || !sample) {
    setText("pulse-gpu", "Solo en directo", true);
    setText("pulse-gpu-note", store.mode === "live" ? "Esperando la primera muestra" : "Pages no lee el equipo");
    return;
  }
  const v = sample.values;
  setText("pulse-gpu", v.gpu_temp_c === null ? "Sin GPU" : `${fmt.number(v.gpu_temp_c)} °C · ${fmt.bytes(v.gpu_mem_used_mib)}`, v.gpu_temp_c === null);
  setText("pulse-gpu-note", v.gpu_temp_c === null ? "NVML no disponible" : `${sample.gpu ?? "GPU"} · ${fmt.number(v.gpu_util_pct)} % de uso`);
}

// Antigüedad de los datos. Se actualiza cada segundo durante el primer minuto y luego
// cada diez, y nunca con la pestaña oculta.
function renderFreshness() {
  clearTimeout(ageTimer);
  const dot = byId("source-dot");
  const generated = Date.parse(store.index?.generated_at ?? "");
  const age = Date.now() - generated;
  const stale = Number.isFinite(age) && age > ctx.staleAfter() * 1000;
  let text;
  if (store.mode === "local") {
    text = `Archivo ${store.localName} · recogido ${fmt.timestamp(generated).text}`;
    dot.dataset.state = "idle";
  } else if (!store.index) {
    text = store.error ? "Sin datos válidos" : "Leyendo el registro";
    dot.dataset.state = store.error ? "error" : "idle";
  } else {
    const parts = [`Recogido ${fmt.age(age)}`, fmt.timestamp(generated, { seconds: true }).text];
    if (store.mode === "live") parts.unshift(store.liveState === "open" ? `Último evento ${fmt.age(Date.now() - store.lastEvent)}` : store.liveState === "reconnecting" ? "Reconectando" : "Directo cerrado");
    else if (store.deployment) parts.push(`publicado ${fmt.timestamp(store.deployment.packaged_at).text}`);
    text = parts.join(" · ");
    dot.dataset.state = store.error ? "error" : stale ? "stale" : "fresh";
  }
  setText("source-mode", { public: "Pages", live: "Directo", local: "Archivo" }[store.mode]);
  setText("source-age", text);
  byId("source-age").title = [
    store.index ? `Recolección: ${new Date(generated).toISOString()}` : "",
    store.lastCheck ? `Última consulta: ${new Date(store.lastCheck).toISOString()}` : "",
    store.nextCheck ? `Próxima consulta: ${new Date(store.nextCheck).toISOString()}` : "",
    stale ? `Los datos superan ${ctx.staleAfter() / 60} min de antigüedad.` : "",
  ].filter(Boolean).join("\n");
  byId("refresh").disabled = store.mode === "local";
  if (document.hidden) return;
  const reference = store.mode === "live" && store.lastEvent ? Math.min(age, Date.now() - store.lastEvent) : age;
  ageTimer = setTimeout(() => scheduler.schedule("freshness", renderFreshness), reference < 60000 ? 1000 : 10000);
}

function flashDot() {
  const dot = byId("source-dot");
  dot.classList.add("pulse-once");
  setTimeout(() => dot.classList.remove("pulse-once"), 250);
}

function adoptIndex(next, path = "observatory.json") {
  store.index = next;
  const tokens = new Set();
  for (const run of next.runs) tokens.add(runToken(run));
  store.pageRuns.set(path, tokens);
  rebuildRuns();
}

// El conjunto de ejecuciones es la unión del índice y de las páginas que el índice
// actual enumera. Las páginas antiguas que ya no aparecen se descartan.
function rebuildRuns() {
  const current = new Set(["observatory.json", ...(store.mode === "local" ? [] : store.index?.pagination?.pages ?? [])]);
  for (const path of store.pageRuns.keys()) if (!current.has(path)) store.pageRuns.delete(path);
  store.runs.clear();
  store.runSource.clear();
  const add = (run, path) => {
    const token = runToken(run);
    store.runs.set(token, run);
    store.runSource.set(token, path);
  };
  for (const run of store.index?.runs ?? []) add(run, "observatory.json");
  for (const path of store.mode === "local" ? [] : store.index?.pagination?.pages ?? []) {
    const document = pages.pages.get(path);
    if (!document) continue;
    store.pageRuns.set(path, new Set(document.runs.map(runToken)));
    for (const run of document.runs) add(run, path);
  }
  invalidateAll();
  scheduleHealth();
}

async function loadAllPages() {
  const paths = (store.index?.pagination?.pages ?? []).filter(path => !pages.pages.has(path));
  if (!paths.length || store.mode === "local") { store.loading.active = false; return; }
  store.loading = { done: 0, total: paths.length, errors: 0, active: true };
  invalidate("campana");
  const relayout = throttle(rebuildRuns, 400);
  await pages.all(paths, (done, total, errors) => {
    store.loading = { done, total, errors, active: done < total };
    relayout();
  });
  store.loading.active = false;
  rebuildRuns();
  showError(store.loading.errors ? `No se pudieron leer ${store.loading.errors} páginas del historial. Se muestran las demás y se reintentará en la próxima consulta.` : "", "pages");
}

async function checkIndex({ manual = false } = {}) {
  if (store.mode === "local" || checkIndex.running) return;
  checkIndex.running = true;
  clearTimeout(pollTimer);
  try {
    const result = await indexResource.get();
    store.lastCheck = Date.now();
    if (result) {
      const next = validateSnapshot(result.data);
      store.lastChange = Date.now();
      adoptIndex(next);
      flashDot();
      if (store.mode === "public") {
        try {
          const deployment = await deploymentResource.get();
          if (deployment) store.deployment = /^[a-f0-9]{40}$/.test(deployment.data.data_sha ?? "") && Number.isFinite(Date.parse(deployment.data.packaged_at)) ? deployment.data : null;
        } catch { /* El recibo de despliegue solo existe en Pages. */ }
      }
      const saveData = navigator.connection?.saveData === true;
      if (!saveData || manual) loadAllPages();
    } else if (pages.failed.size) {
      loadAllPages();
    }
    showError("");
  } catch (error) {
    showError(`${error instanceof SyntaxError ? "El índice no contiene JSON válido" : error.message}. ${store.index ? "Se conserva el último registro válido." : "Prueba a consultar de nuevo o abre un JSON local."}`);
  } finally {
    checkIndex.running = false;
    scheduler.schedule("freshness", renderFreshness);
    schedulePoll();
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  store.nextCheck = null;
  // En directo el índice llega por eventos. El sondeo solo queda como respaldo si la
  // conexión se cierra, por ejemplo cuando el servidor rechaza clientes.
  if (store.mode === "local" || document.hidden || (store.mode === "live" && store.liveState !== "closed")) return;
  const interval = (store.index?.poll_interval_seconds ?? 60) * 1000;
  store.nextCheck = Date.now() + interval;
  pollTimer = setTimeout(() => checkIndex(), interval);
}

function telemetryStore() {
  if (store.telemetry) return store.telemetry;
  const columns = {};
  store.telemetry = { columns, size: 0, latest: null, fields: [] };
  return store.telemetry;
}

// La telemetría se guarda en columnas Float64Array del doble de capacidad. Al llenarse
// se desplaza la segunda mitad al principio, así añadir es O(1) amortizado y las vistas
// usan subarray sin copiar.
function pushTelemetry(t, values) {
  const telemetry = telemetryStore();
  for (const name of ["t", ...Object.keys(values)]) {
    if (!telemetry.columns[name]) telemetry.columns[name] = new Float64Array(TELEMETRY_CAPACITY * 2).fill(NaN);
  }
  if (telemetry.size === TELEMETRY_CAPACITY * 2) {
    for (const column of Object.values(telemetry.columns)) column.copyWithin(0, TELEMETRY_CAPACITY);
    telemetry.size = TELEMETRY_CAPACITY;
  }
  if (telemetry.size && t <= telemetry.columns.t[telemetry.size - 1]) return false;
  telemetry.columns.t[telemetry.size] = t;
  for (const [name, value] of Object.entries(values)) telemetry.columns[name][telemetry.size] = value ?? NaN;
  telemetry.size++;
  return true;
}

function startLive(status) {
  store.mode = "live";
  store.status = status;
  stream = new LiveStream(new URL("./api/events", import.meta.url), {
    state(next, failures) {
      store.liveState = next;
      if (next === "closed") showError("La conexión en directo se ha cerrado. Se consulta el índice cada minuto como respaldo.", "live");
      else if (next === "reconnecting" && failures > 2) showError("El servidor en directo no responde. Se reintenta la conexión.", "live");
      else if (next === "open") showError("", "live");
      scheduler.schedule("freshness", renderFreshness);
      schedulePoll();
    },
    sync(payload) {
      store.status = payload.status;
      store.lastEvent = Date.now();
      const backlog = payload.backlog ?? {};
      const fields = Object.keys(backlog).filter(name => name !== "t");
      for (let i = 0; i < (backlog.t?.length ?? 0); i++) {
        pushTelemetry(backlog.t[i], Object.fromEntries(fields.map(name => [name, backlog[name][i]])));
      }
      // El pulso de la GPU parte de la última muestra del historial, sin esperar al
      // siguiente evento.
      const last = (backlog.t?.length ?? 0) - 1;
      if (last >= 0) {
        telemetryStore().latest = { t: backlog.t[last], gpu: payload.status?.sources?.gpu ?? null, values: Object.fromEntries(fields.map(name => [name, backlog[name][last]])) };
      }
      for (const [name, topic] of Object.entries(payload.topics ?? {})) handleTopic(name, topic);
      invalidate("recursos");
      scheduler.schedule("pulse", renderPulse);
    },
    index: payload => handleTopic("index", payload),
    telemetry: payload => handleTopic("telemetry", payload),
    campaign: payload => handleTopic(`campaign:${payload.id}`, payload),
    traces: payload => handleTopic("traces", payload),
  });
  stream.open();
}

function handleTopic(name, payload) {
  store.lastEvent = Date.now();
  if (name === "index") {
    if (payload.available && payload.etag !== indexResource.etag) checkIndex();
  } else if (name === "telemetry") {
    if (pushTelemetry(payload.t, payload.values)) {
      telemetryStore().latest = payload;
      appendTelemetry(ctx);
      scheduler.schedule("pulse", renderGpuPulse);
    }
  } else if (name.startsWith("campaign:")) {
    store.liveCampaigns.set(name.slice(9), payload);
    invalidate("campana");
  } else if (name === "traces") {
    store.tracesVersion++;
    invalidate("memoria");
  }
  flashDot();
  scheduler.schedule("freshness", renderFreshness);
}

async function detectLive() {
  if (!LOOPBACK.has(location.hostname)) return null;
  try {
    const response = await fetch(new URL("./api/status", import.meta.url), { cache: "no-store", credentials: "omit" });
    if (!response.ok) return null;
    const status = await response.json();
    return status?.mode === "live" ? status : null;
  } catch {
    return null;
  }
}

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  const labels = { system: "Tema: según el sistema", light: "Tema: claro", dark: "Tema: oscuro" };
  byId("theme").setAttribute("aria-label", labels[theme] ?? labels.system);
  byId("theme").title = labels[theme] ?? labels.system;
}

function storedTheme() {
  try { return localStorage.getItem("mars-titan-theme") ?? "system"; } catch { return "system"; }
}

function bindControls() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  for (const tab of tabs) {
    tab.addEventListener("click", () => ctx.set({ vista: tab.dataset.view }, { push: true }));
    tab.addEventListener("keydown", event => {
      const at = tabs.indexOf(tab);
      const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[event.key];
      if (next === undefined) return;
      event.preventDefault();
      const target = tabs[(next + tabs.length) % tabs.length];
      target.focus();
      ctx.set({ vista: target.dataset.view }, { push: true });
    });
  }
  byId("refresh").addEventListener("click", () => checkIndex({ manual: true }));
  byId("theme").addEventListener("click", () => {
    const order = ["system", "light", "dark"];
    const next = order[(order.indexOf(storedTheme()) + 1) % order.length];
    try { localStorage.setItem("mars-titan-theme", next); } catch { /* sin almacenamiento */ }
    applyTheme(next);
    invalidateAll();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => invalidateAll());
  byId("import").addEventListener("click", () => byId("import-file").click());
  byId("import-file").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      if (file.size > 8 * 1024 * 1024) throw new Error("El archivo supera el límite de 8 MiB");
      const next = validateSnapshot(JSON.parse(await file.text()));
      stream?.close();
      clearTimeout(pollTimer);
      store.mode = "local";
      store.localName = file.name;
      store.pageRuns.clear();
      adoptIndex(next);
      byId("local-name").textContent = file.name;
      byId("local-notice").hidden = false;
      showError("", "import");
      renderFreshness();
    } catch (error) {
      showError(`${error instanceof SyntaxError ? "El archivo no contiene JSON válido" : error.message}. Se conserva el registro anterior.`, "import");
    }
  });
  byId("leave-local").addEventListener("click", async () => {
    byId("local-notice").hidden = true;
    store.mode = "public";
    store.pageRuns.clear();
    indexResource.etag = indexResource.lastModified = null;
    const status = await detectLive();
    if (status) startLive(status);
    checkIndex();
  });
  byId("flow-play").addEventListener("click", () => playFlow());
  window.addEventListener("hashchange", () => {
    const next = parseHash(location.hash);
    const viewChanged = next.vista !== state.vista;
    state = next;
    if (viewChanged) showView();
    else invalidateAll();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      hiddenSince = Date.now();
      clearTimeout(pollTimer);
      clearTimeout(ageTimer);
      clearTimeout(healthTimer);
      // En directo, una pestaña oculta más de 30 segundos cierra la conexión. Al volver
      // el servidor envía de nuevo el estado completo.
      setTimeout(() => { if (document.hidden && store.mode === "live") stream?.close(); }, 30000);
      return;
    }
    if (store.mode === "live" && !stream?.source) stream?.open();
    const interval = (store.index?.poll_interval_seconds ?? 60) * 1000;
    if (store.mode !== "local" && (!store.lastCheck || Date.now() - store.lastCheck >= interval || Date.now() - hiddenSince > interval)) checkIndex();
    else schedulePoll();
    renderFreshness();
    scheduleHealth();
  });
  const observer = new ResizeObserver(throttle(() => scheduler.schedule("resize", () => {
    for (const chart of ctx.chartsByView?.get(state.vista) ?? []) chart.resize?.();
  }), 150));
  observer.observe(byId("vistas"));
}

// Un registro «en curso» pasa a «sin actualización» con el tiempo aunque no lleguen datos.
// En lugar de redibujar cada pocos segundos, se programa un único aviso para el primer
// latido que vaya a superar el umbral.
function scheduleHealth() {
  clearTimeout(healthTimer);
  if (document.hidden) return;
  const now = Date.now(), limit = ctx.staleAfter() * 1000;
  let next = Infinity;
  for (const run of store.runs.values()) {
    if (run.status !== "running") continue;
    const due = Date.parse(run.heartbeat_at) + limit;
    if (due > now && due < next) next = due;
  }
  if (next === Infinity) return;
  healthTimer = setTimeout(() => { invalidateAll(); scheduleHealth(); }, next - now + 500);
}

async function main() {
  applyTheme(storedTheme());
  bindControls();
  showView();
  await document.fonts?.ready;
  const status = await detectLive();
  if (status) startLive(status);
  await checkIndex();
  const token = state.ejecucion;
  if (token) {
    const wait = setInterval(() => {
      const run = ctx.run(token);
      if (run) { clearInterval(wait); openRun(ctx, run); }
      else if (!store.loading.active && store.index) clearInterval(wait);
    }, 300);
  }
}

window.__observatory = { store, ctx, scheduler, indexResource, pages, STATUS_LABELS, displayStatus, TERMINAL };
main();
