import assert from "node:assert/strict";
import test from "node:test";
import {createHash} from "node:crypto";
import {readFile, writeFile} from "node:fs/promises";
import {createServer} from "node:http";
import path from "node:path";
import {pathToFileURL} from "node:url";

const {chromium} = await import(pathToFileURL(process.argv[2]).href);
const now = "2026-09-24T13:00:00Z";
const metrics = Object.fromEntries(["mae", "mse", "rank_ic", "coverage_80", "coverage_95", "loss",
  "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "vram_peak_mib", "ram_peak_mib",
  "elapsed_seconds", "samples_per_second"].map(key => [key, null]));
const run = {
  run_id: "fixture-rnn", attempt_id: "attempt-01", model_id: "rnn", variant_id: "base",
  activity: "initial_training", status: "completed", phase: "validation", seed: 42, fold: "fixture",
  started_at: now, updated_at: now, heartbeat_at: null, completed_steps: 2, total_steps: 2,
  epoch: 2, max_epochs: 5, comparison_group: "fixture", metrics: {...metrics, mae: .1},
  history: [1, 2].map(step => ({step, recorded_at: null, loss: null, mae: .3 / step})),
  checkpoint: {step: null, saved_at: null, resumable: null}, test_released: false,
  financial_validation: null,
  metadata: {campaign: "fixture", domain: "technical", method: "initial_training", history_axis: "epoch",
    progress_time_source: "receipt_mtime", train_rows: 100, validation_rows: 20,
    configuration_sha256: null, source_sha256: null, parent: null, error_type: null},
};
const simulation = {...run, run_id: "fixture-simulation", model_id: "financial_comparison", activity: "simulation",
  metrics: {...metrics, elapsed_seconds: 3}, history: [], metadata: {...run.metadata, method: "simulation"}};
const synthetic = {...simulation, run_id: "fixture-generation", model_id: "factor_world", activity: "synthetic_generation",
  phase: "prepare", metadata: {...run.metadata, method: "synthetic_generation"}};
const base = {
  schema_version: 2, project: "MARS-TITAN", source_status: "available", generated_at: now,
  poll_interval_seconds: 60, stale_after_seconds: 900, campaigns: [], notes: [],
  models: [{id: "rnn", name: "RNN", kind: "baseline"},
    {id: "financial_comparison", name: "Resumen de comparación financiera", kind: "summary"},
    {id: "factor_world", name: "Generador de mundos sintéticos", kind: "generator"},
    {id: "M0", name: "Memoria congelada", kind: "ablation"}],
};
const second = JSON.stringify({...base, runs: [{...run, run_id: "fixture-second", model_id: "factor_world"}]});
const secondPath = `data/pages/${createHash("sha256").update(second).digest("hex")}.json`;
const first = {...base, runs: [run, simulation, synthetic], pagination: {total_runs: 4, page_size: 3, pages: [secondPath.slice(5)]}};
const documents = new Map([["data/observatory.json", JSON.stringify(first)], [secondPath, second],
  ["data/deployment.json", JSON.stringify({data_sha: "a".repeat(40), frontend_sha: "b".repeat(40), packaged_at: now})]]);
const site = new URL("../", import.meta.url);
const files = new Set(["index.html", "styles.css", "app.js", "state.mjs"]);
const mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"};
let failPage = false;
const server = createServer(async (request, response) => {
  const name = new URL(request.url, "http://127.0.0.1").pathname.slice(1) || "index.html";
  if (failPage && name === secondPath) return response.writeHead(503).end();
  if (!files.has(name) && !documents.has(name)) return response.writeHead(404).end();
  response.setHeader("content-type", mime[path.extname(name)]);
  response.end(files.has(name) ? await readFile(new URL(name, site)) : documents.get(name));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu"]});
const errors = [];
const coverage = [];
try {
  async function open() {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}, reducedMotion: "reduce"});
    page.on("pageerror", error => errors.push(error.message));
    if (process.argv[3]) await page.coverage.startJSCoverage();
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.locator("#run-content").waitFor({state: "visible"});
    return page;
  }
  async function close(page) {
    if (process.argv[3]) coverage.push(...await page.coverage.stopJSCoverage());
    await page.close();
  }
  await test("la pantalla y los textos del catálogo caben desde 320 px", async () => {
    const page = await open();
    try {
      const failures = [];
      for (const width of [1440, 1024, 768, 390, 320]) {
        await page.setViewportSize({width, height: 1000});
        const result = await page.evaluate(() => {
          const bounds = node => { const range = document.createRange(); range.selectNodeContents(node); return range.getBoundingClientRect(); };
          const overlaps = [...document.querySelectorAll(".model-item")].filter(row => {
            const spans = [...row.children];
            return spans.some((span, index) => index && bounds(spans[index - 1]).right > bounds(span).left + 1);
          }).map(row => row.textContent);
          return {overflow: document.documentElement.scrollWidth > innerWidth, overlaps};
        });
        if (result.overflow || result.overlaps.length) failures.push({width, ...result});
      }
      assert.deepEqual(failures, []);
    } finally { await close(page); }
  });
  await test("el catálogo es estable al paginar y el nombre del modelo no se repite", async () => {
    const page = await open();
    try {
      assert.equal(await page.locator("#run-heading").textContent(), "RNN");
      const catalog = await page.locator("#model-catalog").innerText();
      assert.doesNotMatch(catalog, /Planificado|Con registro/);
      await page.locator("#next-page").click();
      await page.waitForFunction(() => document.querySelector("#page-position").textContent.startsWith("Página 2 "));
      assert.equal(await page.locator("#model-catalog").innerText(), catalog);
      assert.equal(await page.locator("#history-body tr").count(), 1);
    } finally { await close(page); }
  });
  await test("la curva elige MAE cuando no hay pérdida y conserva su eje de épocas", async () => {
    const page = await open();
    try {
      assert.equal(await page.locator("#curve-metric").inputValue(), "mae");
      assert.equal(await page.locator(".chart-point").count(), 2);
      assert.match(await page.locator("#curve-title").textContent(), /época/);
      assert.match(await page.locator(".curve-summary").textContent(), /Épocas 1 a 2/);
      assert.doesNotMatch(await page.locator(".curve-summary").textContent(), /Pasos/);
    } finally { await close(page); }
  });
  await test("la medida se elige dentro de las últimas 240 observaciones visibles", async () => {
    const page = await open();
    try {
      const previous = {...base, runs: [{...run, history: run.history.map(point => ({...point, loss: .1}))}]};
      await page.locator("#import-file").setInputFiles({name: "previous-fixture.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(previous))});
      await page.locator("#local-notice").waitFor({state: "visible"});
      await page.locator("#curve-metric").selectOption("loss");
      const history = Array.from({length: 500}, (_, index) => ({step: index + 1, recorded_at: null,
        loss: index < 260 ? .1 : null, mae: index < 260 ? null : .2}));
      const data = {...base, runs: [{...run, history, completed_steps: 500, total_steps: 500, epoch: 500, max_epochs: 500}]};
      await page.locator("#import-file").setInputFiles({name: "curve-fixture.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(data))});
      await page.waitForFunction(() => document.querySelector("#epoch-label").textContent.includes("500"));
      assert.equal(await page.locator("#curve-metric").inputValue(), "mae");
      assert.equal(await page.locator("#curve-metric option[value=loss]").isDisabled(), true);
      assert.equal(await page.locator(".chart-point").count(), 240);
    } finally { await close(page); }
  });
  await test("el catálogo distingue una ablación de una referencia", async () => {
    const page = await open();
    try {
      assert.equal(await page.locator('.model-item[data-kind="ablation"] .model-state').textContent(), "Ablación");
    } finally { await close(page); }
  });
  await test("el eje distingue errores próximos y acota las etiquetas de magnitudes extremas", async () => {
    const page = await open();
    try {
      for (const values of [[.015054, .015055], [1.2e-10, 1.3e-10], [1.2e8, 1.3e8]]) {
        const history = values.map((mae, index) => ({step: index + 1, recorded_at: null, loss: null, mae}));
        await page.locator("#import-file").setInputFiles({name: "precision-fixture.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({...base, runs: [{...run, history, metrics: {...metrics, mae: values.at(-1)}}]}))});
        await page.waitForFunction(mae => document.querySelector("#key-metrics dd").textContent === new Intl.NumberFormat("es-ES", {minimumFractionDigits: 4, maximumFractionDigits: 4}).format(mae), values.at(-1));
        const labels = await page.locator(".chart-label").allTextContents();
        assert.equal(new Set(labels.slice(0, 3)).size, 3);
        assert.ok(labels.slice(0, 3).every(label => label.length <= 26));
      }
    } finally { await close(page); }
  });
  await test("la simulación no repite el tiempo ni reserva una columna para una curva oculta", async () => {
    const page = await open();
    try {
      await page.locator("#run-select").selectOption(JSON.stringify([simulation.run_id, simulation.attempt_id]));
      assert.equal(await page.locator("#run-content dt").filter({hasText: /^Tiempo observado, segundos$/}).count(), 1);
      assert.equal(await page.locator("#predictive-curve").isVisible(), false);
      const details = await page.locator(".run-details").boundingBox();
      const area = await page.locator(".observation-layout").boundingBox();
      assert.ok(details.width >= area.width * .9);
      await page.locator("#run-select").selectOption(JSON.stringify([synthetic.run_id, synthetic.attempt_id]));
      assert.match(await page.locator("#phase-description").textContent(), /Preparación/);
      assert.doesNotMatch(await page.locator("#phase-description").textContent(), /pertenecen a validación/);
    } finally { await close(page); }
  });
  await test("un fallo al paginar conserva la selección y permite reintentar sin duplicar filas", async () => {
    const page = await open();
    try {
      const ids = await page.locator("#history-body tr").evaluateAll(rows => rows.map(row => row.dataset.runKey));
      failPage = true;
      await page.locator("#next-page").click();
      await page.locator("#error-message").waitFor({state: "visible"});
      assert.deepEqual(await page.locator("#history-body tr").evaluateAll(rows => rows.map(row => row.dataset.runKey)), ids);
      failPage = false;
      await page.locator("#next-page").click();
      await page.waitForFunction(() => document.querySelector("#page-position").textContent.startsWith("Página 2 "));
      assert.equal(await page.locator("#error-message").isVisible(), false);
      await page.locator("#previous-page").click();
      await page.waitForFunction(() => document.querySelector("#page-position").textContent.startsWith("Página 1 "));
      assert.deepEqual(await page.locator("#history-body tr").evaluateAll(rows => rows.map(row => row.dataset.runKey)), ids);
    } finally { failPage = false; await close(page); }
  });
  assert.deepEqual(errors, []);
  if (process.argv[3]) await writeFile(process.argv[3], JSON.stringify(coverage));
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
