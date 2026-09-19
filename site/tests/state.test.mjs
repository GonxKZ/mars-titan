import test from "node:test";
import assert from "node:assert/strict";
import {
  validateSnapshot,
  displayStatus,
  progressPercent,
  publicMetrics,
  publicHistory,
  comparableRuns,
  formatValue,
  toCSV,
} from "../state.mjs";

const NOW = Date.parse("2026-09-18T20:00:00Z");

function metrics(overrides = {}) {
  return {
    mae: null, mse: null, rank_ic: null, coverage_80: null, coverage_95: null,
    loss: null, latency_p50_ms: null, latency_p95_ms: null, latency_p99_ms: null,
    vram_peak_mib: null, ram_peak_mib: null, elapsed_seconds: null,
    samples_per_second: null, ...overrides,
  };
}

function run(overrides = {}) {
  return {
    run_id: "run-01", attempt_id: "attempt-01", model_id: "B1", variant_id: "base",
    status: "running", phase: "train", heartbeat_at: "2026-09-18T19:59:30Z",
    started_at: "2026-09-18T19:00:00Z", updated_at: "2026-09-18T19:59:30Z",
    completed_steps: 25, total_steps: 100, epoch: 1, max_epochs: 30, seed: 17,
    fold: "fold-01", comparison_group: "group-01", metrics: metrics(), history: [],
    checkpoint: { step: null, saved_at: null, resumable: false }, test_released: false,
    ...overrides,
  };
}

function snapshot(runs = []) {
  return {
    schema_version: 1, project: "MARS-TITAN", generated_at: "2026-09-18T20:00:00Z",
    source_status: runs.length ? "available" : "no_runs_registered",
    poll_interval_seconds: 60, stale_after_seconds: 180,
    models: [{ id: "B1", name: "Ridge", kind: "baseline" }], runs, notes: [],
  };
}

test("acepta un registro vacío sin inventar métricas ni modificar la entrada", () => {
  const input = snapshot();
  const output = validateSnapshot(input);
  assert.deepEqual(output, input);
  assert.notEqual(output, input);
  assert.deepEqual(output.runs, []);
});

test("preserva la diferencia entre una medida cero y una medida desconocida", () => {
  const output = validateSnapshot(snapshot([run({ metrics: metrics({ mae: 0 }) })]));
  assert.equal(output.runs[0].metrics.mae, 0);
  assert.equal(output.runs[0].metrics.loss, null);
  assert.equal(formatValue(null), "Sin dato");
  assert.equal(formatValue(0, { digits: 4 }), "0,0000");
});

test("rechaza versiones, estados y fechas sin UTC válidos", () => {
  assert.throws(() => validateSnapshot({ ...snapshot(), schema_version: 2 }), /versión|version/i);
  assert.throws(() => validateSnapshot(snapshot([run({ status: "active" })])), /status/);
  assert.throws(() => validateSnapshot(snapshot([run({ phase: "inference" })])), /phase/);
  assert.throws(() => validateSnapshot({ ...snapshot(), generated_at: "2026-09-18" }), /generated_at/);
});

test("un estado o fase desconocidos permanecen desconocidos", () => {
  const output = validateSnapshot(snapshot([run({ status: null, phase: null })]));
  assert.equal(output.runs[0].status, null);
  assert.equal(output.runs[0].phase, null);
  assert.equal(displayStatus(output.runs[0], NOW, 180), "unknown");
});

test("variante, fecha y recuperabilidad desconocidas permanecen null", () => {
  const output = validateSnapshot(snapshot([run({ variant_id: null, updated_at: null,
    checkpoint: { step: null, saved_at: null, resumable: null } })]));
  assert.equal(output.runs[0].variant_id, null);
  assert.equal(output.runs[0].updated_at, null);
  assert.equal(output.runs[0].checkpoint.resumable, null);
});

test("rechaza métricas imposibles y campos numéricos omitidos", () => {
  for (const invalid of [{ mae: "0.1" }, { loss: Infinity }, { rank_ic: 1.1 }, { coverage_95: -0.1 }]) {
    assert.throws(() => validateSnapshot(snapshot([run({ metrics: metrics(invalid) })])), /metrics/);
  }
  const missing = metrics();
  delete missing.mae;
  assert.throws(() => validateSnapshot(snapshot([run({ metrics: missing })])), /mae/);
});

test("rechaza identidades duplicadas, modelos ausentes y estados de fuente contradictorios", () => {
  assert.throws(() => validateSnapshot(snapshot([run(), run()])), /duplicad/);
  assert.throws(() => validateSnapshot(snapshot([run({ model_id: "missing" })])), /model_id/);
  assert.throws(() => validateSnapshot({ ...snapshot(), source_status: "available" }), /source_status/);
  assert.throws(() => validateSnapshot({ ...snapshot(), poll_interval_seconds: 0 }), /poll_interval_seconds/);
});

test("acota el registro a 128 ejecuciones y 500 puntos por curva", () => {
  const runs = Array.from({ length: 128 }, (_, index) => run({ run_id: `run-${index}` }));
  assert.equal(validateSnapshot(snapshot(runs)).runs.length, 128);
  assert.throws(() => validateSnapshot(snapshot([...runs, run({ run_id: "extra" })])), /runs/);
  const history = Array.from({ length: 501 }, (_, step) => ({ step, recorded_at: "2026-09-18T19:00:00Z", loss: 0, mae: null }));
  assert.throws(() => validateSnapshot(snapshot([run({ total_steps: 600, completed_steps: 600, history })])), /history/);
});

test("valida avance, punto de control y orden de historia para evitar progreso engañoso", () => {
  assert.throws(() => validateSnapshot(snapshot([run({ completed_steps: 101 })])), /completed_steps/);
  assert.throws(() => validateSnapshot(snapshot([run({ checkpoint: { step: 26, saved_at: "2026-09-18T19:30:00Z", resumable: true } })])), /checkpoint/);
  assert.throws(() => validateSnapshot(snapshot([run({ history: [
    { step: 2, recorded_at: "2026-09-18T19:02:00Z", loss: 0.2, mae: null },
    { step: 1, recorded_at: "2026-09-18T19:03:00Z", loss: 0.1, mae: null },
  ] })])), /history/);
});

test("rechaza fechas posteriores al resumen y cronologías imposibles", () => {
  for (const override of [
    { updated_at: "2026-09-18T20:00:01Z" },
    { updated_at: "2026-09-18T18:59:59Z" },
    { heartbeat_at: "2026-09-18T20:00:01Z" },
    { heartbeat_at: "2026-09-18T18:59:59Z" },
    { history: [{ step: 1, recorded_at: "2026-09-18T19:59:31Z", loss: 0.1, mae: null }] },
    { checkpoint: { step: 1, saved_at: "2026-09-18T20:00:01Z", resumable: true } },
  ]) assert.throws(() => validateSnapshot(snapshot([run(override)])), /fecha|orden|anterior|posterior/);
});

test("la comprobación temporal conserva precisión de microsegundos y admite fechas desconocidas", () => {
  const input = snapshot([run({ updated_at: "2026-09-18T20:00:00.000200Z" })]);
  input.generated_at = "2026-09-18T20:00:00.000100Z";
  assert.throws(() => validateSnapshot(input), /posterior/);
  assert.equal(validateSnapshot(snapshot([run({ started_at: null, updated_at: null, heartbeat_at: null })])).runs[0].updated_at, null);
});

test("rechaza puntos de historia que pertenezcan a otro intento o fase", () => {
  const point = { step: 1, recorded_at: "2026-09-18T19:00:00Z", loss: 0.1, mae: 0.02 };
  for (const extra of [{ phase: "test" }, { attempt_id: "old-attempt" }]) {
    assert.throws(() => validateSnapshot(snapshot([run({ history: [{ ...point, ...extra }] })])), /history/);
  }
});

test("una señal de actividad antigua, ausente o futura no equivale a una ejecución pausada", () => {
  assert.equal(displayStatus(run(), NOW, 180), "running");
  assert.equal(displayStatus(run({ heartbeat_at: "2026-09-18T19:50:00Z" }), NOW, 180), "stale");
  assert.equal(displayStatus(run({ heartbeat_at: null }), NOW, 180), "stale");
  assert.equal(displayStatus(run({ heartbeat_at: "2026-09-18T20:10:00Z" }), NOW, 180), "stale");
  assert.equal(displayStatus(run({ status: "paused", heartbeat_at: null }), NOW, 180), "paused");
  assert.equal(displayStatus(run({ status: "completed", heartbeat_at: null }), NOW, 180), "completed");
});

test("el progreso desconocido no se representa como cero ni divide por cero", () => {
  assert.equal(progressPercent(run()), 25);
  assert.equal(progressPercent(run({ total_steps: null })), null);
  assert.equal(progressPercent(run({ total_steps: 0, completed_steps: 0 })), null);
  assert.equal(progressPercent(run({ completed_steps: 0 })), 0);
});

test("la prueba final cerrada oculta métricas y curva aunque el archivo incluya valores", () => {
  const hidden = run({ phase: "test", metrics: metrics({ mae: 0.001, loss: 0.002 }),
    history: [{ step: 1, recorded_at: "2026-09-18T19:00:00Z", loss: 0.2, mae: 0.01 }] });
  assert.equal(publicMetrics(hidden).mae, null);
  assert.equal(publicMetrics(hidden).loss, null);
  assert.deepEqual(publicHistory(hidden), []);
  assert.equal(publicMetrics({ ...hidden, status: "completed", test_released: true }).mae, 0.001);
});

test("una evaluación final o una ejecución no terminal no revela resultados reservados", () => {
  const hidden = run({ phase: "evaluation", metrics: metrics({ mae: 0.003 }) });
  assert.equal(publicMetrics(hidden).mae, null);
  assert.equal(publicMetrics({ ...hidden, test_released: true }).mae, null);
  assert.equal(publicMetrics({ ...hidden, status: "completed", test_released: true }).mae, 0.003);
  assert.deepEqual(comparableRuns([hidden], "group-01", "evaluation"), []);
});

test("una fase desconocida no permite revelar una métrica sin contexto", () => {
  const unknown = run({ phase: null, metrics: metrics({ mae: 0.03 }), history: [
    { step: 1, recorded_at: "2026-09-18T19:00:00Z", loss: 0.1, mae: 0.03 },
  ] });
  assert.equal(publicMetrics(unknown).mae, null);
  assert.deepEqual(publicHistory(unknown), []);
});

test("compara solo ejecuciones completadas del mismo grupo y fase", () => {
  const a = run({ status: "completed", phase: "validation", metrics: metrics({ mae: 0.02 }) });
  const b = run({ run_id: "run-02", status: "completed", phase: "validation", metrics: metrics({ mae: 0.01 }) });
  const incompatible = [
    run({ run_id: "wrong-group", status: "completed", phase: "validation", comparison_group: "other" }),
    run({ run_id: "wrong-phase", status: "completed", phase: "train" }),
    run({ run_id: "not-finished", phase: "validation" }),
  ];
  assert.deepEqual(comparableRuns([a, ...incompatible, b], "group-01", "validation", "mae").map(r => r.run_id), ["run-02", "run-01"]);
  assert.deepEqual(comparableRuns([a], null, "validation"), []);
  assert.deepEqual(comparableRuns([run({ status: "completed", phase: "test" })], "group-01", "test"), []);
});

test("ordena Rank IC de mayor a menor y conserva los desconocidos al final", () => {
  const rows = [
    run({ run_id: "none", status: "completed", phase: "evaluation", test_released: true }),
    run({ run_id: "low", status: "completed", phase: "evaluation", test_released: true, metrics: metrics({ rank_ic: -0.1 }) }),
    run({ run_id: "high", status: "completed", phase: "evaluation", test_released: true, metrics: metrics({ rank_ic: 0.2 }) }),
  ];
  assert.deepEqual(comparableRuns(rows, "group-01", "evaluation", "rank_ic").map(r => r.run_id), ["high", "low", "none"]);
  assert.equal(rows[0].run_id, "none");
});

test("CSV neutraliza fórmulas y escapa comas, comillas y saltos de línea", () => {
  const rows = [run({ run_id: "=SUM(1,2)", variant_id: 'texto,"citado"\nsegunda línea', metrics: metrics({ rank_ic: -0.2 }) })];
  const csv = toCSV(rows, [{ id: "B1", name: "\t@danger" }], NOW, 180);
  assert.ok(csv.includes('"\'=SUM(1,2)"'));
  assert.ok(csv.includes('"\'\t@danger"'));
  assert.ok(csv.includes('"texto,""citado""\nsegunda línea"'));
  assert.ok(csv.includes('"-0.2"'));
  assert.ok(!csv.includes("undefined"));
  assert.ok(!csv.includes("null"));
});

test("CSV mantiene identificadores de intentos pero no exporta resultados protegidos", () => {
  const csv = toCSV([run({ attempt_id: "retry-02", phase: "test", metrics: metrics({ mae: 0.987654321 }) })], [], NOW, 180);
  assert.ok(csv.includes("retry-02"));
  assert.ok(!csv.includes("0.987654321"));
});
