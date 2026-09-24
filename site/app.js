import {
  validateSnapshot, displayStatus, progressPercent, publicMetrics, publicHistory,
  resultsProtected, comparableRuns, formatValue, toCSV, STATUS_LABELS, PHASE_LABELS,
  isPredictive, ACTIVITY_LABELS, FINANCIAL_REASON_LABELS, CONDITION_LABELS, RAM_SCOPE_LABELS,
} from "./state.mjs";

let pageIndex = 0, firstPage = null, deployment = null;
const MAX_BYTES = 8 * 1024 * 1024;
const REQUEST_TIMEOUT_MS = 10000;
const POLL_MS = 60000;
const SVG_NS = "http://www.w3.org/2000/svg";
const byId = id => document.getElementById(id);
let snapshot = null;
let sourceMode = "public";
let snapshotOrigin = "public";
let selectedRunKey = "";
let pollTimer = null;
let healthTimer = null;
let activeRequest = null;
let requestVersion = 0;
let pageActive = true;

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function svgElement(tag, attributes = {}, content) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (content !== undefined) node.textContent = content;
  return node;
}

function dateText(value) {
  if (!value) return "Sin fecha";
  return `${new Intl.DateTimeFormat("es-ES", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }).format(new Date(value))} UTC`;
}

function runKey(run) { return JSON.stringify([run.run_id, run.attempt_id]); }
function modelName(run) { return snapshot?.models.find(model => model.id === run.model_id)?.name ?? run.model_id; }
function count(value) { return value === null ? "Sin dato" : formatValue(value, { digits: 0 }); }
function selectedRun() { return snapshot?.runs.find(run => runKey(run) === selectedRunKey) ?? null; }

function setOptions(select, values, emptyLabel) {
  const previous = select.value;
  const next = values.length ? values : [{ value: "", label: emptyLabel }];
  const same = select.options.length === next.length && next.every((entry, index) => select.options[index].value === entry.value && select.options[index].textContent === entry.label);
  if (!same) select.replaceChildren(...next.map(entry => {
    const option = element("option", entry.label);
    option.value = entry.value;
    return option;
  }));
  if (next.some(entry => entry.value === previous)) select.value = previous;
  select.disabled = values.length === 0;
}

function announce(message, state = "ready") {
  byId("connection-status").textContent = message;
  byId("connection-status").dataset.state = state;
}

function showError(message) {
  byId("error-message").textContent = message;
  byId("error-message").hidden = !message;
}

function renderTracking() {
  const runs = snapshot.runs;
  setOptions(byId("run-select"), runs.map(run => ({ value: runKey(run), label: `${run.run_id} (${run.attempt_id})` })), "Sin ejecuciones");
  if (!runs.some(run => runKey(run) === selectedRunKey)) selectedRunKey = runs.length ? runKey(runs[0]) : "";
  byId("run-select").value = selectedRunKey;
  const run = selectedRun();
  byId("run-picker").hidden = !run;
  byId("run-content").hidden = !run;
  byId("empty-tracking").hidden = Boolean(run);
  const status = run ? displayStatus(run, Date.now(), snapshot.stale_after_seconds) : "empty";
  byId("run-status").className = `status status-${status}`;
  byId("run-status").textContent = run ? STATUS_LABELS[status] : "Registro vacío";
  byId("run-heading").textContent = run ? `${run.model_id} / ${modelName(run)}` : "Sin ejecuciones registradas";
  byId("run-description").textContent = run ? `${run.variant_id ?? "Variante no informada"} / ${run.run_id} / ${run.attempt_id}` : "Los modelos están planificados. Todavía no hay resultados de entrenamiento en este registro.";
  for (const item of byId("phase-list").children) {
    if (run?.phase === item.dataset.phase) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
  byId("phase-description").textContent = run?.metadata ? "Las métricas pertenecen a validación. El recibo no identifica la operación actual del proceso." : run ? `Fase declarada: ${PHASE_LABELS[run.phase] ?? "no informada"}. Las etapas anteriores del esquema no se dan por verificadas.` : "Secuencia de referencia. Ninguna fase está activa.";
  if (!run) return;
  const protectedResults = resultsProtected(run);
  byId("test-notice").hidden = !protectedResults;
  byId("test-notice").textContent = run.phase === null ? "Fase no informada. Las métricas y la curva permanecen ocultas hasta conocer su contexto." : "Resultados finales protegidos. Sus métricas y su curva no se muestran hasta su publicación explícita al terminar la ejecución.";
  const progress = progressPercent(run);
  byId("progress-label").textContent = progress === null ? "Avance no disponible" : `${count(run.completed_steps)} de ${count(run.total_steps)} pasos (${formatValue(progress, { digits: 1 })} %)`;
  byId("epoch-label").textContent = run.epoch === null ? "Época no informada" : `Época ${count(run.epoch)}${run.max_epochs === null ? "" : ` de ${count(run.max_epochs)}`}`;
  byId("run-progress").hidden = progress === null;
  if (progress !== null) byId("run-progress").value = progress;
  const metrics = publicMetrics(run);
  const predictive = isPredictive(run);
  byId("predictive-curve").hidden = !predictive;
  const primary = predictive ? [
    ["MAE residual", metrics.mae, { digits: 4 }],
    ["Rank IC", metrics.rank_ic, { digits: 4 }],
    ["Cobertura del intervalo 95 %", metrics.coverage_95, { digits: 1, style: "percent" }],
    ["Pico de VRAM, MiB", metrics.vram_peak_mib, { digits: 0 }],
  ] : [["Tiempo observado, segundos", metrics.elapsed_seconds, {digits: 1}],
    ["Pico de VRAM, MiB", metrics.vram_peak_mib, {digits: 0}]];
  byId("key-metrics").replaceChildren(...primary.map(([label, value, options]) => {
    const group = element("div");
    group.append(element("dt", label), element("dd", formatValue(value, options), value === null ? "missing" : undefined));
    return group;
  }));
  const details = [
    ["Actividad", ACTIVITY_LABELS[run.activity] ?? "No informada"],
    ["Última observación del proceso", dateText(run.heartbeat_at)],
    ["Último progreso registrado", dateText(run.updated_at)],
    ...(run.metadata ? [["Campaña y método", `${run.metadata.campaign} / ${run.metadata.method}`], ["Origen", {real: "Corpus real", synthetic: "Mundo sintético", technical: "Comprobación técnica"}[run.metadata.domain]], ["Filas de entrenamiento / validación", `${count(run.metadata.train_rows)} / ${count(run.metadata.validation_rows)}`], ...(predictive ? [["Eje de la curva", run.metadata.history_axis === "epoch" ? "Épocas sin fecha original" : "Pasos"]] : []), ["Error registrado", run.metadata.error_type ?? "Sin error informado"]] : []),
    ...(run.metadata?.condition ? [["Condición de entrenamiento", CONDITION_LABELS[run.metadata.condition]]] : []),
    ...(run.metadata?.parent_model ? [["Referencia de partida", snapshot.models.find(model => model.id === run.metadata.parent_model)?.name ?? run.metadata.parent_model]] : []),
    ["Último punto de control", run.checkpoint.step === null ? "Sin punto registrado" : `Paso ${count(run.checkpoint.step)} / ${dateText(run.checkpoint.saved_at)}`],
    ["Recuperación", run.checkpoint.resumable === null ? "No informada" : run.checkpoint.resumable ? "Disponible según el registro" : "No recuperable según el registro"],
    ["Semilla y partición", `${count(run.seed)} / ${run.fold ?? "Sin dato"}`],
    [RAM_SCOPE_LABELS[run.metadata?.ram_peak_scope] ?? "RAM máxima, MiB (alcance no informado)", formatValue(metrics.ram_peak_mib, { digits: 0 })],
    ...(run.metadata?.ram_peak_scope === "executable" ? [[RAM_SCOPE_LABELS.process_lifetime, formatValue(protectedResults ? null : run.metadata.process_lifetime_peak_rss_mib, {digits: 0})]] : []),
    ["Latencia p50 / p95 / p99, ms", [metrics.latency_p50_ms, metrics.latency_p95_ms, metrics.latency_p99_ms].map(value => formatValue(value, { digits: 1 })).join(" / ")],
    ["Tiempo observado, segundos", formatValue(metrics.elapsed_seconds, { digits: 0 })],
    ["Muestras por segundo", formatValue(metrics.samples_per_second, { digits: 1 })],
  ];
  byId("run-details").replaceChildren(...details.map(([label, value]) => {
    const group = element("div");
    group.append(element("dt", label), element("dd", value));
    return group;
  }));
  const financial = run.financial_validation;
  byId("financial-panel").hidden = !financial;
  if (financial) {
    const rows = [
      ["Retorno neto", formatValue(financial.net_return, {digits: 2, style: "percent"})],
      ["Caída máxima desde el pico", formatValue(financial.max_drawdown, {digits: 2, style: "percent"})],
      [`Costes${run.metadata.currency ? ` (${run.metadata.currency})` : ""}`, formatValue(financial.costs)], ["Rotación", formatValue(financial.turnover)],
      ["Pasos evaluados", count(financial.steps)],
      ["Episodio completo", financial.completed === null ? "No informado" : financial.completed ? "Sí" : "No"],
      ["Incidencia", financial.invalid_reason === null ? "No informada" : FINANCIAL_REASON_LABELS[financial.invalid_reason]],
    ];
    byId("financial-metrics").replaceChildren(...rows.map(([label, value]) => {
      const group = element("div");
      group.append(element("dt", label), element("dd", value));
      return group;
    }));
  }
  renderCurve(run);
}

function renderCurve(run) {
  const metric = byId("curve-metric").value;
  const history = publicHistory(run).slice(-240);
  const valid = history.filter(point => point[metric] !== null);
  const container = byId("curve-content");
  if (!valid.length) {
    container.replaceChildren(element("p", resultsProtected(run) ? "Curva protegida hasta publicar los resultados finales." : "No hay observaciones de esta medida en el registro."));
    return;
  }
  const width = 680, height = 242, left = 60, right = 18, top = 22, bottom = 34;
  const values = valid.map(point => point[metric]);
  const minimum = Math.min(...values), maximum = Math.max(...values);
  const padding = (maximum - minimum) * .1 || Math.abs(maximum) * .05 || .01;
  const low = minimum - padding, high = maximum + padding;
  const firstStep = history[0].step, lastStep = history.at(-1).step;
  const x = step => left + (step - firstStep) / (lastStep - firstStep || 1) * (width - left - right);
  const y = value => top + (high - value) / (high - low) * (height - top - bottom);
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": "curve-title curve-description" });
  const label = metric === "mae" ? "MAE residual" : "Pérdida";
  svg.append(svgElement("title", { id: "curve-title" }, `${label} por paso registrado`));
  svg.append(svgElement("desc", { id: "curve-description" }, `Últimos ${history.length} registros. ${valid.length} valores conocidos entre ${formatValue(minimum)} y ${formatValue(maximum)}. Los huecos permanecen separados.`));
  for (const value of [low, (low + high) / 2, high]) {
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: y(value), y2: y(value), class: "chart-axis" }));
    svg.append(svgElement("text", { x: left - 9, y: y(value) + 4, "text-anchor": "end", class: "chart-label" }, formatValue(value, { digits: 3 })));
  }
  for (const step of new Set([firstStep, lastStep])) svg.append(svgElement("text", { x: x(step), y: height - 11, "text-anchor": "middle", class: "chart-label" }, count(step)));
  let path = "", connected = false;
  for (const point of history) {
    if (point[metric] === null) { connected = false; continue; }
    path += `${connected ? "L" : "M"}${x(point.step).toFixed(2)},${y(point[metric]).toFixed(2)} `;
    connected = true;
  }
  svg.append(svgElement("path", { d: path, class: "chart-line" }));
  for (const point of valid) svg.append(svgElement("circle", { cx: x(point.step), cy: y(point[metric]), r: valid.length > 80 ? 1.4 : 2.5, class: "chart-point" }));
  container.replaceChildren(svg, element("p", `Últimos ${history.length} registros. Pasos ${count(firstStep)} a ${count(lastStep)}.`, "curve-summary"));
}

function renderComparison() {
  const phase = byId("comparison-phase").value;
  const eligible = snapshot.runs.filter(run => isPredictive(run) && run.status === "completed" && run.phase === phase && run.comparison_group && !resultsProtected(run));
  const groups = [...new Set(eligible.map(run => run.comparison_group))].sort();
  setOptions(byId("comparison-group"), groups.map(group => ({ value: group, label: `Grupo ${group.slice(0, 16)}${group.length > 16 ? "…" : ""}` })), "Sin grupos en esta fase");
  const group = byId("comparison-group").value;
  const rows = comparableRuns(snapshot.runs, group, phase, byId("comparison-metric").value);
  byId("comparison-empty").hidden = rows.length > 0;
  byId("comparison-table-wrap").hidden = rows.length === 0;
  byId("comparison-caption").textContent = `${rows.length} ejecuciones en ${PHASE_LABELS[phase]}. Grupo ${group}`;
  byId("comparison-body").replaceChildren(...rows.map(run => {
    const row = element("tr");
    const model = element("td");
    model.append(element("span", `${run.model_id} / ${modelName(run)}`), element("small", `${run.variant_id ?? "Variante no informada"} / ${run.attempt_id}`));
    row.append(model, element("td", `${count(run.seed)} / ${run.fold ?? "Sin dato"}`));
    for (const [key, options] of [["mae", { digits: 4 }], ["rank_ic", { digits: 4 }], ["coverage_95", { digits: 1, style: "percent" }], ["latency_p95_ms", { digits: 1 }], ["vram_peak_mib", { digits: 0 }]]) row.append(element("td", formatValue(run.metrics[key], options), "numeric"));
    return row;
  }));
}

function filteredHistory() {
  if (!snapshot) return [];
  const query = byId("history-search").value.trim().toLocaleLowerCase("es");
  const status = byId("history-status").value;
  return snapshot.runs.filter(run => {
    const haystack = [run.run_id, run.attempt_id, run.model_id, modelName(run), run.variant_id].join(" ").toLocaleLowerCase("es");
    return (!query || haystack.includes(query)) && (status === "all" || displayStatus(run, Date.now(), snapshot.stale_after_seconds) === status);
  }).sort((a, b) => (Date.parse(b.updated_at) || 0) - (Date.parse(a.updated_at) || 0));
}

function renderHistory() {
  const focusedRun = document.activeElement?.dataset?.runKey;
  const rows = filteredHistory();
  byId("history-count").textContent = snapshot.runs.length ? `${rows.length} de ${snapshot.runs.length} ejecuciones` : "Sin ejecuciones registradas";
  byId("history-empty").hidden = rows.length > 0;
  byId("history-table-wrap").hidden = rows.length === 0;
  byId("export-button").disabled = rows.length === 0;
  byId("history-empty").firstElementChild.textContent = snapshot.runs.length ? "No hay coincidencias con estos filtros" : "El registro aún no contiene ejecuciones";
  byId("history-empty").lastElementChild.textContent = snapshot.runs.length ? "Prueba otro identificador o selecciona todos los estados." : "No hay ejecuciones que mostrar. Los intentos aparecerán con su fecha, fase y estado declarado.";
  byId("history-body").replaceChildren(...rows.map(run => {
    const row = element("tr");
    row.dataset.runKey = runKey(run);
    const identity = element("td");
    identity.append(element("span", run.run_id), element("small", run.attempt_id));
    const state = displayStatus(run, Date.now(), snapshot.stale_after_seconds);
    const status = element("td");
    status.append(element("span", STATUS_LABELS[state], `status status-${state}`));
    const action = element("td");
    const button = element("button", "Ver", "text-button");
    button.type = "button";
    button.dataset.runKey = runKey(run);
    button.setAttribute("aria-label", `Ver ${run.run_id}, intento ${run.attempt_id}`);
    button.addEventListener("click", () => {
      selectedRunKey = runKey(run);
      renderTracking();
      location.hash = "seguimiento";
      byId("run-select").focus({ preventScroll: true });
    });
    action.append(button);
    row.append(identity, element("td", `${run.model_id} / ${run.variant_id ?? "Variante no informada"}`), element("td", PHASE_LABELS[run.phase] ?? "No informada"), status, element("td", dateText(run.updated_at)), action);
    return row;
  }));
  if (focusedRun) [...byId("history-body").querySelectorAll("button")].find(button => button.dataset.runKey === focusedRun)?.focus({ preventScroll: true });
}

function render() {
  if (!snapshot) return;
  byId("source-label").textContent = snapshotOrigin === "local" ? "Archivo local" : "Registro público";
  byId("snapshot-date").textContent = `Recolección: ${dateText(firstPage?.generated_at ?? snapshot.generated_at)}`;
  byId("publication-date").textContent = deployment && sourceMode === "public" ? `Publicación preparada: ${dateText(deployment.packaged_at)}. Datos ${deployment.data_sha.slice(0, 8)}.` : "Publicación no informada";
  byId("local-notice").hidden = snapshotOrigin !== "local";
  byId("public-source-button").disabled = Boolean(activeRequest);
  byId("refresh-button").disabled = sourceMode === "local" || Boolean(activeRequest);
  const campaigns = firstPage?.campaigns ?? snapshot.campaigns ?? [];
  byId("campaign-summary").hidden = !campaigns.length;
  byId("campaign-rows").replaceChildren(...campaigns.map(c => element("p", `${c.id}: ${c.counts.completed ?? 0} completadas de ${c.planned_runs} previstas, ${c.registered_runs} registradas. ${STATUS_LABELS[c.status] ?? c.status}.`)));
  const pages = sourceMode === "public" ? firstPage?.pagination?.pages ?? [] : [];
  byId("history-pages").hidden = !pages.length;
  byId("previous-page").disabled = pageIndex === 0 || Boolean(activeRequest);
  byId("next-page").disabled = pageIndex >= pages.length || Boolean(activeRequest);
  byId("page-position").textContent = `Página ${pageIndex + 1} de ${pages.length + 1}. ${firstPage?.pagination?.total_runs ?? snapshot.runs.length} registros.`;
  renderTracking();
  renderComparison();
  renderHistory();
  byId("model-catalog").replaceChildren(...snapshot.models.map(model => {
    const row = element("div", undefined, "model-item");
    row.dataset.kind = model.kind;
    row.append(element("span", model.id, "model-id"), element("span", model.name, "model-name"), element("span", snapshot.runs.some(run => run.model_id === model.id) ? "Con registro" : "Planificado", "model-state"));
    return row;
  }));
  byId("snapshot-notes").hidden = snapshot.notes.length === 0;
  byId("snapshot-notes").replaceChildren(...snapshot.notes.map(note => element("li", note)));
  scheduleHealth();
}

function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }
function stopHealth() { clearTimeout(healthTimer); healthTimer = null; }

function refreshHealth() {
  if (!snapshot || document.hidden || !pageActive) return;
  const current = selectedRun();
  if (current) {
    const status = displayStatus(current, Date.now(), snapshot.stale_after_seconds);
    if (byId("run-status").textContent !== STATUS_LABELS[status]) {
      byId("run-status").className = `status status-${status}`;
      byId("run-status").textContent = STATUS_LABELS[status];
    }
  }
  const expectedRows = filteredHistory();
  const shownRows = [...byId("history-body").rows];
  if (expectedRows.length !== shownRows.length || expectedRows.some((run, index) => runKey(run) !== shownRows[index].dataset.runKey)) {
    renderHistory();
  } else {
    expectedRows.forEach((run, index) => {
      const status = displayStatus(run, Date.now(), snapshot.stale_after_seconds);
      const indicator = shownRows[index].querySelector(".status");
      if (indicator.textContent !== STATUS_LABELS[status]) {
        indicator.textContent = STATUS_LABELS[status];
        indicator.className = `status status-${status}`;
      }
    });
  }
  if (!activeRequest && sourceMode === "public" && byId("connection-status").dataset.state !== "error") {
    const active = snapshotOrigin === "public" && snapshot.runs.some(run => displayStatus(run, Date.now(), snapshot.stale_after_seconds) === "running");
    byId("connection-status").dataset.state = active ? "ready" : "idle";
  }
}

function scheduleHealth() {
  stopHealth();
  if (snapshot && !document.hidden && pageActive && snapshot.runs.some(run => displayStatus(run, Date.now(), snapshot.stale_after_seconds) === "running")) {
    healthTimer = setTimeout(() => { refreshHealth(); scheduleHealth(); }, 15000);
  }
}

function schedulePoll() {
  stopPolling();
  if (sourceMode === "public" && !document.hidden && pageActive) pollTimer = setTimeout(refreshPublic, POLL_MS);
}

async function readResponse(response) {
  if (!response.ok) throw new Error(`El servidor respondió con HTTP ${response.status}`);
  const length = Number(response.headers.get("content-length"));
  if (Number.isFinite(length) && length > MAX_BYTES) throw new Error("El resumen supera el límite de 8 MiB");
  if (!response.body) throw new Error("El servidor no devolvió un resumen");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0, content = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MAX_BYTES) { await reader.cancel(); throw new Error("El resumen supera el límite de 8 MiB"); }
      content += decoder.decode(value, { stream: true });
    }
    return content + decoder.decode();
  } finally { reader.releaseLock(); }
}

async function refreshPublic() {
  if (sourceMode !== "public" || document.hidden || activeRequest || !pageActive) return;
  stopPolling();
  const version = requestVersion;
  const controller = new AbortController();
  activeRequest = controller;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, REQUEST_TIMEOUT_MS);
  byId("refresh-button").disabled = true;
  announce("Leyendo el registro", "loading");
  try {
    const response = await fetch(new URL("./data/observatory.json", import.meta.url), { cache: "no-store", credentials: "omit", signal: controller.signal, headers: { Accept: "application/json" } });
    const next = validateSnapshot(JSON.parse(await readResponse(response)));
    let publication = null;
    if (next.schema_version === 2) {
      try {
        const meta = JSON.parse(await readResponse(await fetch(new URL("./data/deployment.json", import.meta.url), {cache: "no-store", credentials: "omit", signal: controller.signal})));
        if (/^[a-f0-9]{40}$/.test(meta.data_sha) && /^[a-f0-9]{40}$/.test(meta.frontend_sha) && Number.isFinite(Date.parse(meta.packaged_at))) publication = meta;
      } catch (_error) { /* El recibo de despliegue puede faltar en una vista local. */ }
    }
    if (version !== requestVersion || sourceMode !== "public") return;
    snapshot = next;
    deployment = publication;
    firstPage = next; pageIndex = 0;
    snapshotOrigin = "public";
    showError("");
    const active = snapshot.runs.some(run => displayStatus(run, Date.now(), snapshot.stale_after_seconds) === "running");
    announce(snapshot.runs.length ? "Resumen leído. Próxima consulta en 60 s" : "Sin ejecuciones registradas", active ? "ready" : "idle");
    render();
  } catch (error) {
    if (version !== requestVersion || sourceMode !== "public" || (controller.signal.aborted && !timedOut)) return;
    const detail = timedOut ? "La petición ha superado los 10 segundos" : error instanceof SyntaxError ? "El archivo no contiene JSON válido" : error.message;
    showError(`${detail}. ${snapshot ? "Se conserva el último resumen válido." : "Puedes volver a actualizar o importar un JSON local."}`);
    announce("No se ha podido actualizar", "error");
    render();
  } finally {
    clearTimeout(timeout);
    if (activeRequest === controller) activeRequest = null;
    byId("refresh-button").disabled = sourceMode === "local";
    byId("public-source-button").disabled = false;
    byId("previous-page").disabled = pageIndex === 0;
    byId("next-page").disabled = pageIndex >= (firstPage?.pagination?.pages.length ?? 0);
    schedulePoll();
  }
}

async function loadPage(index) {
  if (activeRequest || sourceMode !== "public" || !firstPage || index < 0 || index > firstPage.pagination.pages.length) return;
  stopPolling();
  const version = requestVersion, source = firstPage;
  const controller = new AbortController();
  activeRequest = controller;
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const next = index === 0 ? source : validateSnapshot(JSON.parse(await readResponse(await fetch(new URL("./data/" + source.pagination.pages[index - 1], import.meta.url), {credentials: "omit", signal: controller.signal}))));
    if (version !== requestVersion || sourceMode !== "public" || source !== firstPage) return;
    snapshot = next; pageIndex = index; showError("");
  } catch (_error) {
    if (version === requestVersion) showError("No se pudo leer la página. Se conserva la página anterior.");
  } finally {
    clearTimeout(timeout);
    if (activeRequest === controller) activeRequest = null;
    render();
    if (pageIndex === 0) schedulePoll();
  }
}
byId("previous-page").addEventListener("click", () => loadPage(pageIndex - 1));
byId("next-page").addEventListener("click", () => loadPage(pageIndex + 1));

byId("refresh-button").addEventListener("click", refreshPublic);
byId("import-button").addEventListener("click", () => byId("import-file").click());
byId("import-file").addEventListener("change", async event => {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  try {
    if (file.size > MAX_BYTES) throw new Error("El archivo supera el límite de 8 MiB");
    const next = validateSnapshot(JSON.parse(await file.text()));
    requestVersion += 1;
    sourceMode = "local";
    snapshotOrigin = "local";
    stopPolling();
    activeRequest?.abort();
    snapshot = next; firstPage = null; pageIndex = 0;
    byId("local-file-name").textContent = file.name;
    showError("");
    announce("Archivo local. Consulta pública detenida", "idle");
    render();
  } catch (error) {
    showError(`${error instanceof SyntaxError ? "El archivo no contiene JSON válido" : error.message}. El registro anterior se conserva.`);
  }
});
byId("public-source-button").addEventListener("click", () => {
  requestVersion += 1;
  sourceMode = "public";
  refreshPublic();
});
byId("run-select").addEventListener("change", event => { selectedRunKey = event.target.value; renderTracking(); });
byId("curve-metric").addEventListener("change", () => { const run = selectedRun(); if (run) renderCurve(run); });
for (const id of ["comparison-group", "comparison-phase", "comparison-metric"]) byId(id).addEventListener("change", () => { if (snapshot) renderComparison(); });
byId("history-search").addEventListener("input", () => { if (snapshot) renderHistory(); });
byId("history-status").addEventListener("change", () => { if (snapshot) renderHistory(); });
byId("export-button").addEventListener("click", () => {
  if (!snapshot) return;
  const rows = filteredHistory();
  if (!rows.length) return;
  const blob = new Blob(["\ufeff", toCSV(rows, snapshot.models, Date.now(), snapshot.stale_after_seconds)], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = element("a");
  anchor.href = url;
  anchor.download = "mars-titan-ejecuciones.csv";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { stopPolling(); stopHealth(); activeRequest?.abort(); }
  else { refreshHealth(); scheduleHealth(); if (sourceMode === "public") refreshPublic(); }
});
window.addEventListener("pagehide", () => { pageActive = false; stopPolling(); stopHealth(); activeRequest?.abort(); });
window.addEventListener("pageshow", event => { pageActive = true; if (event.persisted) { refreshHealth(); scheduleHealth(); if (sourceMode === "public") refreshPublic(); } });

const navigationLinks = [...document.querySelectorAll(".section-links a")];
const sections = [...document.querySelectorAll("main > section")];
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const link of navigationLinks) {
      if (link.hash === `#${entry.target.id}`) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    }
  }
}, { rootMargin: "-10% 0px -70% 0px", threshold: 0 });
for (const section of sections) observer.observe(section);
window.addEventListener("pagehide", () => observer.disconnect());
window.addEventListener("pageshow", event => { if (event.persisted) for (const section of sections) observer.observe(section); });
refreshPublic();
