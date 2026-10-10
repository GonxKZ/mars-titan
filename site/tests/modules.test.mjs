// Pruebas de los módulos puros del observatorio. No necesitan navegador: formatos,
// reducción M4, paquetes de trazas, estado de la URL, estructuras derivadas, lectura
// condicional, flujo SSE y planificación de fotogramas.

import test from "node:test";
import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import * as fmt from "../format.mjs";
import { m4Indices, nearestIndex, lowerBound, pick } from "../decimate.mjs";
import { validateManifest, decodeBundle, validateIndex, sha256Hex } from "../traces.mjs";
import { parseHash, serializeHash, runToken, rangeToken, parseRange, DEFAULTS } from "../urlstate.mjs";
import {
  groupCampaigns, campaignMatrix, estimateRemaining, curveFacets, crossSection, curveSummary, quantile,
  relativeToGroup, completionTimeline, resourcePoints, throughputPoints, financialPoints, liveCampaignMatrix,
} from "../model.mjs";
import { ConditionalResource, PageStore, LiveStream, readLimited } from "../sources.mjs";
import { FrameScheduler, throttle } from "../scheduler.mjs";
import { stepDecimals, formatTicks, cividis, diverging, withAlpha } from "../charts.mjs";

const TIME = "2026-10-06T12:00:00Z";

function run(overrides = {}) {
  return {
    run_id: "neural-s-fold-000-a", attempt_id: "legacy", model_id: "gru", variant_id: "v", status: "completed",
    phase: "validation", seed: 42, test_released: false, activity: "initial_training", updated_at: TIME,
    heartbeat_at: TIME, started_at: TIME, completed_steps: 10, total_steps: 10,
    metrics: { mae: 0.015, loss: 0.1, elapsed_seconds: 60, vram_peak_mib: 200, ram_peak_mib: 900 },
    history: [], checkpoint: { step: null, saved_at: null, resumable: null },
    metadata: { campaign: "neural-s-fold-000", progress_time_source: "receipt_timestamp", best_epoch: null },
    ...overrides,
  };
}

test("los formatos usan coma decimal, unidades y ausencias explícitas", () => {
  assert.equal(fmt.number(1234.5, 1), "1234,5");
  assert.equal(fmt.number(12345), "12.345");
  assert.equal(fmt.number(null), fmt.MISSING);
  assert.equal(fmt.number(Number.NaN), fmt.MISSING);
  assert.equal(fmt.significant(0.014937, 3), "0,0149");
  assert.equal(fmt.significant(0, 3), "0,00");
  assert.match(fmt.percent(0.256, 0), /^26\s%$/);
  assert.equal(fmt.bytes(512), "512 MiB");
  assert.equal(fmt.bytes(2048), "2,0 GiB");
  assert.equal(fmt.duration(5.25), "5,3 s");
  assert.equal(fmt.duration(3600 * 3 + 60 * 5), "3 h 5 min");
  assert.equal(fmt.duration(86400 * 3), "3 d 0 h");
  assert.equal(fmt.duration(-1), fmt.MISSING);
  assert.equal(fmt.age(2000), "ahora");
  assert.equal(fmt.age(-5), "con fecha futura");
  const stamp = fmt.timestamp("2026-10-06T12:00:00Z", { timeZone: "Europe/Madrid" });
  assert.match(stamp.text, /14:00/);
  assert.match(stamp.text, /CEST|GMT\+2/);
  assert.equal(stamp.utc, "2026-10-06T12:00:00Z");
  assert.equal(fmt.timestamp("no es una fecha").text, "Sin fecha");
  assert.equal(fmt.shortHash("abcdef0123456789", 6), "abcdef");
});

test("las marcas del eje se distinguen por el paso y no por la magnitud", () => {
  assert.equal(stepDecimals(1), 0);
  assert.equal(stepDecimals(0.1), 1);
  assert.equal(stepDecimals(0.25), 2);
  assert.equal(stepDecimals(2.5), 1);
  assert.equal(stepDecimals(0.005), 3);
  assert.deepEqual(formatTicks([45.2, 45.3, 45.4]), ["45,2", "45,3", "45,4"]);
  assert.deepEqual(formatTicks([0, 0.05, 0.1]), ["0,00", "0,05", "0,10"]);
  assert.deepEqual(formatTicks([100, 1000, 10000], { log: true }), ["100", "1000", "10.000"]);
  assert.deepEqual(formatTicks([0.01, 0.1, 1], { log: true }), ["0,01", "0,1", "1"]);
  const ticks = formatTicks([-0.1, -0.05, 0, 0.05]);
  assert.equal(new Set(ticks).size, ticks.length);
});

test("las escalas de color son monótonas y la divergente es neutra en el centro", () => {
  const luminance = ([r, g, b]) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  for (const dark of [false, true]) {
    const values = [0, 0.25, 0.5, 0.75, 1].map(t => luminance(cividis(t, dark)));
    const direction = Math.sign(values.at(-1) - values[0]);
    assert.equal(direction, dark ? 1 : -1, "en claro más valor es más oscuro y en oscuro al revés");
    for (let i = 1; i < values.length; i++) assert.equal(Math.sign(values[i] - values[i - 1]), direction);
    const [r, g, b] = diverging(0, dark);
    assert.ok(Math.max(r, g, b) - Math.min(r, g, b) < 20, "el centro no tiene tono propio");
    assert.ok(diverging(-1, dark)[2] > diverging(-1, dark)[0], "por debajo es azul");
    assert.ok(diverging(1, dark)[0] > diverging(1, dark)[2], "por encima es bermellón");
    assert.deepEqual(diverging(5, dark), diverging(1, dark));
  }
  assert.equal(withAlpha("#0072b2", 0.5), "rgba(0,114,178,0.5)");
  assert.equal(withAlpha("rgb(1,2,3)", 0.5), "rgb(1,2,3)");
});

test("M4 conserva primero, último, mínimo y máximo de cada columna con índices reales", () => {
  const n = 100_000;
  const x = Float64Array.from({ length: n }, (_, i) => i);
  const y = Float32Array.from({ length: n }, (_, i) => Math.sin(i / 50));
  y[54321] = 9;
  y[12345] = -9;
  const indices = m4Indices(x, y, 0, n - 1, 200);
  assert.ok(indices.length <= 200 * 5 + 4);
  assert.ok(indices.includes(54321) && indices.includes(12345), "los picos aislados sobreviven");
  assert.ok(indices.includes(0) && indices.includes(n - 1));
  for (let i = 1; i < indices.length; i++) assert.ok(indices[i] > indices[i - 1], "índices crecientes y sin repetir");
  // Fuera del tramo visible solo entra un punto de cada lado para que la línea llegue al borde.
  const window = m4Indices(x, y, 1000, 2000, 50);
  assert.ok(window[0] >= 999 && window.at(-1) <= 2001);
});

test("M4 deja los huecos como huecos y no reduce series cortas", () => {
  const x = Float64Array.from({ length: 4000 }, (_, i) => i);
  const y = Float32Array.from({ length: 4000 }, () => 1);
  for (let i = 1000; i < 1500; i++) y[i] = Number.NaN;
  const indices = m4Indices(x, y, 0, 3999, 100);
  const [, picked] = pick(x, [y], indices);
  assert.ok(picked.includes(null), "la ausencia se transmite como null");
  assert.ok(!picked.some(value => Number.isNaN(value)));
  assert.equal(m4Indices(x, y, 0, 3999, 2000).length, 4000);
  assert.equal(lowerBound(x, 10.5), 11);
  assert.equal(nearestIndex(x, 10.4), 10);
  assert.equal(nearestIndex(new Float64Array(), 1), -1);
});

function bundleBytes() {
  const x = new Float64Array([1, 2, 3, 4]);
  const y = new Float32Array([0.5, Number.NaN, 0.25, 0.125]);
  const values = new Float32Array([0, 1, 0.5, 0.25, 1, 0, 0.5, 0.75]);
  const buffer = new ArrayBuffer(32 + 16 + 32);
  new Float64Array(buffer, 0, 4).set(x);
  new Float32Array(buffer, 32, 4).set(y);
  new Float32Array(buffer, 48, 8).set(values);
  return buffer;
}

function manifest(buffer, overrides = {}) {
  return {
    schema_version: 1, kind: "mars_titan_learning_traces", name: "prueba", blob: "prueba-0123456789abcdef.bin",
    blob_bytes: buffer.byteLength, blob_sha256: createHash("sha256").update(Buffer.from(buffer)).digest("hex"),
    provenance: "fixture", x_unit: "optimizer_step", run_id: "r", attempt_id: "a", model_id: "m",
    series: [{ id: "titans.surprise", group: "titans", label: "Sorpresa", unit: "pérdida", x: { offset: 0, length: 4, dtype: "f64" }, y: { offset: 32, length: 4, dtype: "f32" } }],
    matrices: [{ id: "titans.alpha", group: "titans", label: "α", unit: "fracción", rows: ["Capa 1", "Capa 2"], x: { offset: 0, length: 4, dtype: "f64" }, values: { offset: 48, length: 8, dtype: "f32" }, range: [0, 1] }],
    ...overrides,
  };
}

test("un paquete de trazas se lee sin copiar y comprueba tamaño y huella", async () => {
  const buffer = bundleBytes();
  const parsed = validateManifest(manifest(buffer));
  const bundle = await decodeBundle(parsed, buffer, { subtle: webcrypto.subtle });
  assert.equal(bundle.verified, true);
  assert.equal(bundle.points, 4 + 8);
  assert.ok(Number.isNaN(bundle.series[0].y[1]), "NaN llega como ausencia");
  assert.equal(bundle.series[0].y.buffer, buffer, "vista tipada sobre el mismo bloque");
  const tampered = bundleBytes();
  new Float32Array(tampered, 32, 1)[0] = 0.75;
  await assert.rejects(decodeBundle(parsed, tampered, { subtle: webcrypto.subtle }), /huella/);
  await assert.rejects(decodeBundle(parsed, new ArrayBuffer(8), { subtle: webcrypto.subtle }), /tamaño/);
  const unverified = await decodeBundle(parsed, buffer, { subtle: null });
  assert.equal(unverified.verified, false);
  assert.equal(await sha256Hex(buffer, null), null);
});

test("el manifiesto rechaza referencias fuera del bloque, grupos desconocidos y versiones ajenas", () => {
  const buffer = bundleBytes();
  const base = manifest(buffer);
  const broken = [
    { ...base, schema_version: 2 },
    { ...base, provenance: "inventada" },
    { ...base, blob: "../fuera.bin" },
    { ...base, series: [{ ...base.series[0], group: "otro" }] },
    { ...base, series: [{ ...base.series[0], y: { offset: 76, length: 4, dtype: "f32" } }] },
    { ...base, series: [{ ...base.series[0], y: { offset: 33, length: 1, dtype: "f32" } }] },
    { ...base, series: [base.series[0], base.series[0]] },
    { ...base, matrices: [{ ...base.matrices[0], rows: ["Capa 1"] }] },
    { ...base, matrices: [{ ...base.matrices[0], range: [1, 0] }] },
  ];
  for (const input of broken) assert.throws(() => validateManifest(input), TypeError);
  assert.deepEqual(validateIndex({ schema_version: 1, bundles: [{ name: "a", manifest: "a.json", provenance: "fixture" }] })[0].name, "a");
  assert.throws(() => validateIndex({ schema_version: 1, bundles: [{ name: "a", manifest: "../a.json" }] }));
});

test("el eje x de una traza debe crecer y los infinitos no son observaciones", async () => {
  const buffer = bundleBytes();
  new Float64Array(buffer, 0, 4).set([1, 3, 2, 4]);
  await assert.rejects(decodeBundle(validateManifest(manifest(buffer)), buffer, { subtle: null }), /creciente/);
  const infinite = bundleBytes();
  new Float32Array(infinite, 32, 4)[2] = Number.POSITIVE_INFINITY;
  await assert.rejects(decodeBundle(validateManifest(manifest(infinite)), infinite, { subtle: null }), /infinito/);
});

test("el estado de la URL es estable, descarta valores ajenos y admite enlaces antiguos", () => {
  const state = { ...DEFAULTS, vista: "curvas", medida: "session_mae", x: "3~17", marcas: "relativo", ejecucion: "r~legacy" };
  assert.deepEqual(parseHash(serializeHash(state)), state);
  assert.equal(serializeHash(DEFAULTS), "#");
  const hostile = parseHash("#vista=<script>&medida=drop&x=9~2&marcas=x&buscar=" + "a".repeat(200));
  assert.equal(hostile.vista, "campana");
  assert.equal(hostile.medida, "mae");
  assert.equal(hostile.x, "");
  assert.equal(hostile.marcas, "estado");
  assert.equal(hostile.buscar, "");
  assert.equal(parseHash("#historial").vista, "registros");
  assert.equal(runToken({ run_id: "a", attempt_id: "b" }), "a~b");
  assert.deepEqual(parseRange(rangeToken(1.5, 9.25)), { min: 1.5, max: 9.25 });
  assert.equal(parseRange("4~4"), null);
});

test("las campañas se agrupan en series por etapa y ventana sin unir archivos ajenos", () => {
  const campaigns = [
    "neural-real-x-fold-000", "neural-real-x-fold-001", "posttraining-real-x-fold-000", "tabular-solo", "otra-cosa",
  ].map(id => ({ id, status: "running", planned_runs: 2, registered_runs: 1, counts: { completed: 1, not_started: 1 } }));
  const groups = groupCampaigns(campaigns);
  assert.equal(groups.get("real-x").length, 3);
  assert.ok(groups.has("tabular-solo"), "una etapa sin pareja no crea una serie");
  assert.ok(groups.has("otra-cosa"));
  const members = groups.get("real-x");
  const runs = [
    run({ metadata: { campaign: "neural-real-x-fold-000" }, seed: 44 }),
    run({ run_id: "b", metadata: { campaign: "neural-real-x-fold-000" }, seed: 42 }),
    run({ run_id: "c", status: "failed", metadata: { campaign: "neural-real-x-fold-001" } }),
  ];
  const matrix = campaignMatrix(members, runs, Date.parse(TIME), 900);
  assert.deepEqual(matrix.windows, ["fold-000", "fold-001"]);
  assert.deepEqual(matrix.rows[0].cells.get("fold-000").map(mark => mark.seed), [42, 44]);
  assert.equal(matrix.pending.get("neural|fold-000"), 1);
  assert.equal(matrix.pending.get("posttraining|fold-000"), 1);
});

test("la estimación usa la mediana de intervalos y se niega con pocos datos", () => {
  const minute = 60_000;
  const times = [0, 1, 2, 3, 4, 50].map(i => i * minute);
  const estimate = estimateRemaining(times, 10);
  assert.equal(estimate.intervalSeconds, 60, "la pausa larga no mueve la mediana");
  assert.equal(estimate.seconds, 600);
  assert.equal(estimateRemaining(times.slice(0, 3), 10), null);
  assert.equal(estimateRemaining(times, 0), null);
});

function history(values, key = "mae") {
  return values.map((value, i) => ({ step: i + 1, recorded_at: TIME, loss: null, mae: null, [key]: value }));
}

test("las curvas se alinean por época, dejan huecos y ocultan fases reservadas", () => {
  const runs = [
    run({ run_id: "a", history: history([3, 2, 1]) }),
    run({ run_id: "b", seed: 43, history: history([4, null, 2]) }),
    run({ run_id: "c", phase: "test", history: history([9, 9, 9]) }),
    run({ run_id: "d", activity: "rl", history: history([7, 7, 7]) }),
  ];
  const facets = curveFacets(runs);
  assert.equal(facets.length, 1);
  assert.deepEqual(facets[0].x, [1, 2, 3]);
  assert.deepEqual(facets[0].lines.map(line => line.run.run_id), ["a", "b"]);
  assert.deepEqual(facets[0].lines[1].y, [4, null, 2]);
  assert.equal(facets[0].points, 5);
  assert.deepEqual(crossSection(facets[0].lines, 0), { n: 2, min: 3, max: 4, median: 3.5 });
  assert.equal(crossSection(facets[0].lines.map(line => ({ y: [null] })), 0), null);
});

test("la mediana y los cuartiles entre ejecuciones exigen un mínimo de curvas", () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5);
  assert.equal(quantile([1, 2, 3, 4], 0.25), 1.75);
  const lines = [{ y: [1, 5, null] }, { y: [2, 6, 1] }, { y: [3, 7, null] }, { y: [4, null, null] }];
  const summary = curveSummary(lines, 3);
  assert.deepEqual(summary.median, [2.5, 6, null]);
  assert.deepEqual(summary.low, [1.75, 5.5, null]);
  assert.deepEqual(summary.count, [4, 3, 1]);
});

test("la comparación con la mediana del grupo es simétrica y no inventa valores", () => {
  const result = relativeToGroup([
    { group: "w1", value: 1 }, { group: "w1", value: 2 }, { group: "w1", value: 3 },
    { group: "w2", value: 10 }, { group: "w2", value: null },
  ]);
  assert.deepEqual(result.relative, [-0.5, 0, 0.5, 0, null]);
  assert.ok(result.limit > 0 && result.limit <= 0.5);
  assert.equal(result.n, 4);
  assert.equal(relativeToGroup([{ group: "a", value: 1 }]).limit, null);
});

test("el ritmo solo usa fechas de recibo y funde recibos simultáneos", () => {
  const runs = [
    run({ updated_at: "2026-10-06T12:00:00Z" }), run({ updated_at: "2026-10-06T12:00:00Z" }),
    run({ updated_at: "2026-10-06T12:10:00Z" }), run({ status: "running", updated_at: "2026-10-06T12:20:00Z" }),
    run({ updated_at: "2026-10-06T12:30:00Z", metadata: { progress_time_source: "receipt_mtime" } }),
  ];
  const timeline = completionTimeline(runs);
  assert.deepEqual([...timeline.y], [2, 3]);
  assert.equal(timeline.x[1] - timeline.x[0], 600);
});

test("recursos, caudal y finanzas solo usan medidas publicadas", () => {
  const runs = [
    run({ run_id: "a" }), run({ run_id: "b", phase: "test" }), run({ run_id: "c", metrics: { elapsed_seconds: null, vram_peak_mib: 10 } }),
    run({ run_id: "d", history: [{ step: 1, recorded_at: TIME, mae: 1, loss: null, train_samples_per_second: 5000, train_seconds: 12 }] }),
    run({ run_id: "e", activity: "rl", model_id: "ppo_klpo", financial_validation: { net_return: 0.1, max_drawdown: 0.05, completed: true } }),
    run({ run_id: "f", activity: "rl", model_id: "double_dqn", financial_validation: { net_return: null, max_drawdown: 0.1 } }),
  ];
  const resources = resourcePoints(runs);
  assert.deepEqual(resources[0].points.map(point => point.run.run_id), ["a", "d"]);
  assert.deepEqual(throughputPoints(runs), [{ model: "gru", points: [{ run: runs[3], epoch: 1, value: 5000, seconds: 12 }] }]);
  const finance = financialPoints(runs);
  assert.equal(finance[0].points.length, 1);
  assert.equal(finance[1].points.length, 0, "un retorno ausente no se dibuja");
});

test("la campaña leída en directo decodifica el vocabulario y estima con recibos", () => {
  const state = {
    vocabulary: { scopes: ["us"], windows: ["fold-001", "fold-000"], arms: ["gru"], names: ["search-0", "finalist-s43", "carry-s44"] },
    cells: [[0, 0, 0, 0, "done", "2026-10-06T12:00:00Z"], [0, 1, 0, 1, "attempt", null], [0, 1, 0, 2, "pending", null]],
  };
  const matrix = liveCampaignMatrix(state);
  assert.deepEqual(matrix.windows, ["fold-000", "fold-001"]);
  assert.equal(matrix.done, 1);
  assert.equal(matrix.attempts, 1);
  assert.deepEqual(matrix.rows[0].cells.get("fold-000").map(mark => mark.seed), [43, 44]);
  assert.equal(matrix.estimate, null, "una sola confirmación no basta para estimar");
});

function response(status, body = "", headers = {}) {
  return new Response(status === 304 ? null : body, { status, headers });
}

test("la lectura condicional envía validadores y un 304 no descarga nada", async () => {
  const calls = [];
  const replies = [response(200, '{"a":1}', { etag: '"v1"', "last-modified": "Tue, 06 Oct 2026 12:00:00 GMT" }), response(304), response(200, '{"a":2}', { etag: '"v2"' })];
  const resource = new ConditionalResource("http://x/data.json", { fetchImpl: async (url, options) => { calls.push(options); return replies.shift(); } });
  assert.deepEqual((await resource.get()).data, { a: 1 });
  assert.equal(await resource.get(), null);
  assert.deepEqual((await resource.get()).data, { a: 2 });
  assert.equal(calls[0].headers["If-None-Match"], undefined);
  assert.equal(calls[1].headers["If-None-Match"], '"v1"');
  assert.equal(calls[1].cache, "no-store");
  assert.equal(calls[1].credentials, "omit");
  assert.equal(resource.notModified, 1);
  assert.equal(resource.requests, 3);
});

test("la lectura respeta el límite de bytes y el tiempo máximo", async () => {
  const big = new Response("x".repeat(2048));
  await assert.rejects(readLimited(big, 1024), /límite/);
  const declared = new Response("x", { headers: { "content-length": "999999" } });
  await assert.rejects(readLimited(declared, 1024), /límite/);
  const slow = new ConditionalResource("http://x", { timeoutMs: 20, fetchImpl: (url, { signal }) => new Promise((_, reject) => signal.addEventListener("abort", () => reject(new Error("abortada")))) });
  await assert.rejects(slow.get(), /superó/);
});

test("las páginas inmutables se piden una vez y una página fallida se puede reintentar", async () => {
  let failures = 1, requests = 0;
  const modes = [];
  const store = new PageStore("http://x/data/", value => value, {
    concurrency: 2,
    fetchImpl: async (url, options) => {
      requests++;
      if (url.pathname.endsWith("b.json")) modes.push(options.cache);
      if (url.pathname.endsWith("b.json") && failures-- > 0) return response(503);
      return response(200, JSON.stringify({ url: url.pathname }));
    },
  });
  const progress = [];
  const result = await store.all(["pages/a.json", "pages/b.json", "pages/c.json"], (done, total, errors) => progress.push([done, total, errors]));
  assert.deepEqual(result, { done: 3, errors: 1 });
  assert.ok(store.failed.has("pages/b.json"));
  await store.all(["pages/a.json", "pages/b.json"]);
  assert.ok(!store.failed.has("pages/b.json"));
  assert.equal(requests, 4, "la página ya leída no se vuelve a pedir");
  assert.deepEqual(modes, ["default", "reload"], "el reintento no reutiliza un 503 guardado en caché");
  assert.deepEqual(progress.at(-1), [3, 3, 1]);
});

class FakeEventSource {
  constructor(url) { this.url = url; this.listeners = new Map(); this.readyState = 0; FakeEventSource.last = this; }
  addEventListener(name, listener) { this.listeners.set(name, listener); }
  emit(name, data) { this.listeners.get(name)?.({ data }); }
  close() { this.readyState = 2; this.closed = true; }
}

test("el flujo SSE reparte eventos, ignora JSON roto y distingue reconexión de cierre", () => {
  const seen = [], states = [];
  const stream = new LiveStream("http://x/api/events", {
    state: (state, failures) => states.push([state, failures]),
    telemetry: payload => seen.push(payload),
  }, { EventSourceImpl: FakeEventSource });
  stream.open();
  const source = FakeEventSource.last;
  source.emit("telemetry", '{"t":1}');
  source.emit("telemetry", "{roto");
  assert.deepEqual(seen, [{ t: 1 }]);
  assert.equal(stream.state, "open");
  source.readyState = 0;
  source.onerror();
  assert.equal(stream.state, "reconnecting");
  source.readyState = 2;
  source.onerror();
  assert.equal(stream.state, "closed");
  assert.equal(stream.source, null);
  assert.deepEqual(states.at(-1), ["closed", 2]);
  stream.open();
  assert.notEqual(FakeEventSource.last, source, "se puede volver a abrir");
  stream.close();
  assert.ok(FakeEventSource.last.closed);
});

test("el planificador agrupa tareas por fotograma y espera con la pestaña oculta", () => {
  const frames = [];
  const doc = { hidden: false, listeners: [], addEventListener(name, fn) { this.listeners.push(fn); } };
  const scheduler = new FrameScheduler({ raf: fn => frames.push(fn), doc });
  let a = 0, b = 0;
  scheduler.schedule("a", () => a++);
  scheduler.schedule("a", () => a++);
  scheduler.schedule("b", () => b++);
  assert.equal(frames.length, 1);
  frames.shift()();
  assert.deepEqual([a, b], [1, 1]);
  doc.hidden = true;
  scheduler.schedule("a", () => a++);
  assert.equal(frames.length, 0, "oculta no pide fotogramas");
  doc.hidden = false;
  doc.listeners[0]();
  frames.shift()();
  assert.equal(a, 2);
  assert.equal(scheduler.frames, 2);
});

test("throttle ejecuta al instante, agrupa ráfagas y conserva la última llamada", () => {
  let clock = 0;
  const timers = [];
  const calls = [];
  const limited = throttle(value => calls.push(value), 100, { now: () => clock, setTimer: (fn, wait) => timers.push([fn, wait]) });
  limited(1);
  limited(2);
  limited(3);
  assert.deepEqual(calls, [1]);
  assert.equal(timers.length, 1);
  clock = 100;
  timers.shift()[0]();
  assert.deepEqual(calls, [1, 3]);
});
