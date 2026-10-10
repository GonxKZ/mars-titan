// Registro ficticio para las pruebas de navegador. Reproduce la forma del contrato v2 del
// recolector con valores inventados y rotulados como prueba. Nunca se publica ni se usa
// como resultado: solo ejercita estados, fases reservadas, actividades y paginación.

import { createHash } from "node:crypto";

export const SECRET_MAE = 0.987654321;

const MODELS = [
  { id: "gru", name: "GRU", kind: "baseline" }, { id: "lstm", name: "LSTM", kind: "baseline" },
  { id: "ridge", name: "Ridge", kind: "baseline" }, { id: "adaptation", name: "Adaptación predictiva", kind: "adaptation" },
  { id: "ppo_klpo", name: "PPO con KLPO", kind: "policy" }, { id: "double_dqn", name: "Double DQN", kind: "policy" },
  { id: "cash", name: "Efectivo", kind: "reference" }, { id: "generator", name: "Generador sintético", kind: "generator" },
];

const iso = time => new Date(time).toISOString().replace(".000Z", "Z");

function metrics(overrides = {}) {
  return {
    mae: null, mse: null, rank_ic: null, coverage_80: null, coverage_95: null, loss: null, latency_p50_ms: null,
    latency_p95_ms: null, latency_p99_ms: null, vram_peak_mib: null, ram_peak_mib: null, elapsed_seconds: null,
    samples_per_second: null, session_mae: null, ...overrides,
  };
}

function metadata(campaign, overrides = {}) {
  return {
    campaign, domain: "real", method: "initial_training", parent_frozen: null, currency: null, condition: null, parent_model: null,
    configuration_sha256: "c".repeat(64), source_sha256: "d".repeat(64), history_axis: "epoch", best_epoch: null, stopped_early: null,
    progress_time_source: "receipt_timestamp", train_rows: 1000, validation_rows: 200, parent: null, error_type: null,
    ram_peak_scope: "executable", executable_peak_rss_mib: 900, process_lifetime_peak_rss_mib: null, ...overrides,
  };
}

function history(start, epochs, base, slope, seed, now) {
  return Array.from({ length: epochs }, (_, i) => {
    const wobble = Math.sin((i + 1) * (seed % 7 + 1)) * 0.0004;
    return {
      step: i + 1, recorded_at: iso(start + (i + 1) * 60_000 > now ? now : start + (i + 1) * 60_000), loss: null,
      mae: base - slope * Math.log(i + 1) + wobble, train_mae: base * 0.9 - slope * Math.log(i + 2),
      train_samples_per_second: 4000 + 100 * (seed % 5) + i, train_seconds: 30 + i, session_mae: base * 1.01 - slope * Math.log(i + 1), validation_seconds: 4,
    };
  });
}

function trainingRun({ model, campaign, seed, window, status = "completed", epochs = 8, now, offset = 0, phase = "validation", extra = {} }) {
  const start = now - 8 * 3600_000 + offset * 60_000;
  const updated = Math.min(now, start + epochs * 60_000);
  const base = { gru: 0.0160, lstm: 0.0158, ridge: 0.0175 }[model] ?? 0.0165;
  const curve = history(start, epochs, base + window * 0.0002, 0.0003, seed, now);
  const done = status === "completed";
  return {
    run_id: `${campaign}-${model}-s${seed}-${String(offset).padStart(3, "0")}`, attempt_id: "attempt-0001", model_id: model,
    activity: "initial_training", variant_id: `${model}-variant`, status, phase, started_at: iso(start), updated_at: iso(updated),
    heartbeat_at: status === "running" ? iso(now - 20_000) : null, completed_steps: epochs, total_steps: done ? epochs : 12,
    epoch: epochs, max_epochs: done ? epochs : 12, seed, fold: null, comparison_group: "fixture-group",
    metrics: metrics({ mae: curve.at(-1).mae, session_mae: curve.at(-1).session_mae, mse: 0.0004, elapsed_seconds: epochs * 60, vram_peak_mib: 180 + seed, ram_peak_mib: 900 }),
    history: curve, checkpoint: { step: epochs, saved_at: iso(updated), resumable: !done },
    financial_validation: null, test_released: false,
    metadata: metadata(campaign, { best_epoch: Math.max(1, epochs - 2), stopped_early: false }), ...extra,
  };
}

// Construye el índice y sus páginas. `extraRuns` permite a una prueba añadir registros
// para comprobar que la página los recoge en la siguiente consulta.
export function buildSnapshot({ now = Date.parse("2026-10-09T12:00:00Z"), extraRuns = [], pageSize = 16 } = {}) {
  const runs = [];
  const campaigns = [];
  for (let window = 0; window < 3; window++) {
    const fold = `fold-${String(window).padStart(3, "0")}`;
    const neural = `neural-fixture-${fold}`;
    const tabular = `tabular-fixture-${fold}`;
    let offset = window * 40;
    for (const model of ["gru", "lstm"]) {
      for (const seed of [42, 43, 44]) {
        const running = window === 2 && model === "lstm" && seed === 44;
        runs.push(trainingRun({ model, campaign: neural, seed, window, status: running ? "running" : "completed", epochs: running ? 5 : 8, now, offset: offset++ }));
      }
    }
    runs.push(trainingRun({ model: "ridge", campaign: tabular, seed: 42, window, now, offset: offset++, epochs: 1 }));
    campaigns.push(
      { id: neural, domain: "real", planned_runs: 6, status: window === 2 ? "running" : "completed", dependencies: [], counts: { completed: window === 2 ? 5 : 6, running: window === 2 ? 1 : 0, not_started: 0 }, registered_runs: 6 },
      { id: tabular, domain: "real", planned_runs: 3, status: window === 0 ? "running" : "blocked", dependencies: window === 0 ? [] : [neural], counts: { completed: 1, not_started: 2 }, registered_runs: 1 },
    );
  }
  // Fase reservada: su MAE no debe aparecer en ningún sitio.
  runs.push(trainingRun({ model: "gru", campaign: "neural-fixture-fold-000", seed: 45, window: 0, now, offset: 200, phase: "test", extra: {
    run_id: "reserved-test-run", metrics: metrics({ mae: SECRET_MAE, elapsed_seconds: 10, vram_peak_mib: 50 }),
    metadata: metadata("neural-fixture-fold-000", { ram_peak_scope: null, executable_peak_rss_mib: null }),
  } }));
  // Texto hostil en un identificador: debe verse como texto.
  runs.push(trainingRun({ model: "lstm", campaign: "neural-fixture-fold-001", seed: 41, window: 1, now, offset: 201, extra: {
    run_id: "hostile-run", variant_id: "<img src=x onerror=alert(1)>", comparison_group: "=HYPERLINK(\"x\")",
    metadata: metadata("neural-fixture-fold-001", { ram_peak_scope: "process_lifetime", executable_peak_rss_mib: null, process_lifetime_peak_rss_mib: 1700 }),
  } }));
  // Actividades que no son predicción: no tienen curvas de MAE.
  for (const [i, model] of ["ppo_klpo", "double_dqn", "cash"].entries()) {
    runs.push({
      ...trainingRun({ model, campaign: "posttraining-fixture-rl", seed: 42 + i, window: 0, now, offset: 300 + i, epochs: 3 }),
      run_id: `rl-${model}`, activity: "rl", history: [], financial_validation: { net_return: 0.02 * (i - 1), max_drawdown: 0.03 + 0.01 * i, costs: 0.001, turnover: 1.5, steps: 250, completed: i !== 2, invalid_reason: i === 2 ? "incomplete" : null },
      metadata: metadata("posttraining-fixture-rl", { method: "rl", domain: "real" }),
    });
  }
  runs.push({
    ...trainingRun({ model: "generator", campaign: "synthetic-fixture", seed: 7, window: 0, now, offset: 400, epochs: 2 }),
    run_id: "synthetic-generation-run", activity: "synthetic_generation", history: [],
    metadata: metadata("synthetic-fixture", { domain: "synthetic", method: "synthetic_generation", condition: "real_synthetic" }),
  });
  runs.push(...extraRuns);
  campaigns.push(
    { id: "posttraining-fixture-rl", domain: "real", planned_runs: 3, status: "completed", dependencies: [], counts: { completed: 3, not_started: 0 }, registered_runs: 3 },
    { id: "synthetic-fixture", domain: "synthetic", planned_runs: 1, status: "completed", dependencies: [], counts: { completed: 1, not_started: 0 }, registered_runs: 1 },
  );
  const sorted = [...runs].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const head = sorted.slice(0, pageSize);
  const rest = sorted.slice(pageSize);
  const pages = new Map();
  for (let i = 0; i < rest.length; i += pageSize) {
    const document = { schema_version: 2, project: "MARS-TITAN", generated_at: iso(now), source_status: "available", poll_interval_seconds: 60, stale_after_seconds: 900, models: MODELS, notes: [], runs: rest.slice(i, i + pageSize) };
    const body = JSON.stringify(document);
    pages.set(`pages/${createHash("sha256").update(body).digest("hex")}.json`, body);
  }
  const windows = new Map();
  const windowEntries = windowCampaigns(now).map(({ entry, state }) => {
    if (!state) return entry;
    const body = JSON.stringify(state);
    const path = `windows/${createHash("sha256").update(body).digest("hex")}.json`;
    windows.set(path, body);
    return { ...entry, path };
  });
  const index = {
    schema_version: 2, project: "MARS-TITAN", generated_at: iso(now), source_status: "available", poll_interval_seconds: 60, stale_after_seconds: 900,
    models: MODELS, notes: ["Registro ficticio de las pruebas de interfaz."], campaigns, runs: head,
    pagination: { total_runs: runs.length, page_size: pageSize, pages: [...pages.keys()] },
    window_campaigns: windowEntries,
  };
  return { index: JSON.stringify(index), pages, windows, runs };
}

// Dos campañas por ventanas ficticias: una etapa base con su matriz y una etapa de
// adaptadores declarada que todavía no tiene resumen.
function windowCampaigns(now) {
  const cells = [[0, 0, 0, 0, "done", iso(now - 3 * 3600_000)], [0, 0, 0, 1, "done", iso(now - 2 * 3600_000)],
    [0, 1, 1, 0, "done", iso(now - 3600_000)], [0, 1, 1, 1, "attempt", null]];
  const state = {
    id: "fixture-base", kind: "historical_masked_campaign_run", stage: "base", status: "running", updated_at: iso(now - 600_000),
    summary_modified_at: iso(now - 600_000), planned: { training_jobs: 4, prediction_jobs: 0 }, completed: { training_jobs: 3, prediction_jobs: 0 },
    final_test_opened: false, vocabulary: { scopes: ["US+CN"], windows: ["fold-000", "fold-001"], arms: ["gru", "lstm"], names: ["finalist-s42", "finalist-s43"] },
    models: { gru: "gru", lstm: "lstm" }, cells,
    active: [{ job: "US+CN/fold-001/lstm/finalist-s43", attempt: "attempt-0001", updated_at: iso(now - 60_000), global_step: 20,
      epochs: [1, 2].map(epoch => ({ epoch, train_mae: 0.02 - epoch * 0.001, mae: 0.021 - epoch * 0.0008, session_mae: null, train_samples_per_second: 4000, train_seconds: 30 })) }],
  };
  const declared = { domain: "real", configuration: "configs/fixture.json", status: null, updated_at: null };
  return [
    { entry: { ...declared, id: "fixture-base", stage: "base", status: "running", updated_at: state.updated_at, jobs: 4, done: 3, attempts: 1 }, state },
    { entry: { ...declared, id: "fixture-adapters", stage: "adapters", path: null, jobs: 0, done: 0, attempts: 0 }, state: null },
  ];
}

export { trainingRun };
