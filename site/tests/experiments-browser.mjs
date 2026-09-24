import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import {createServer} from "node:http";
import {pathToFileURL} from "node:url";
import path from "node:path";

const {chromium} = await import(pathToFileURL(process.argv[2]).href);
const data = process.argv[3];
const source = JSON.parse(await readFile(path.join(data, "observatory.json")));
const runs = [...source.runs];
for (const name of source.pagination.pages) {
  assert.match(name, /^pages\/[a-f0-9]{64}\.json$/);
  runs.push(...JSON.parse(await readFile(path.join(data, name))).runs);
}
const native = runs.filter(run => run.metadata?.campaign.startsWith("native-financial-check-") && run.metadata.ram_peak_scope === "executable")
  .sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0];
assert.ok(native);
const paired = {
  ...native, run_id: "fixture-posttraining", attempt_id: "fixture-01", model_id: "gru",
  activity: "supervised_continuation", variant_id: "fixture-neural-mae", status: "completed",
  phase: "validation", started_at: source.generated_at, updated_at: source.generated_at,
  heartbeat_at: null, completed_steps: 4, total_steps: 4, epoch: 1, max_epochs: 1, seed: 42,
  fold: "fixture", comparison_group: null,
  metrics: {...Object.fromEntries(Object.keys(native.metrics).map(key => [key, null])), mae: 0.2},
  history: [{step: 1, recorded_at: null, loss: null, mae: 0.2}],
  checkpoint: {step: null, saved_at: null, resumable: null}, financial_validation: null,
  metadata: {...native.metadata, campaign: "fixture-posttraining", domain: "technical",
    method: "neural_mae", condition: "real_synthetic", parent_model: "gru", parent_frozen: null,
    parent: null, source_sha256: null, configuration_sha256: null, currency: null,
    train_rows: null, validation_rows: null, history_axis: "epoch", error_type: null,
    ram_peak_scope: null, executable_peak_rss_mib: null, process_lifetime_peak_rss_mib: null},
};
const base = {...source, campaigns: [], notes: ["Comprobación de presentación con un recibo nativo y una fixture técnica de postentrenamiento."]};
delete base.pagination;
const second = JSON.stringify({...base, runs: [paired]});
const secondName = `pages/${createHash("sha256").update(second).digest("hex")}.json`;
const first = {...base, runs: [native], pagination: {total_runs: 2, page_size: 1, pages: [secondName]}};
const documents = new Map([["data/observatory.json", JSON.stringify(first)], [`data/${secondName}`, second]]);
const site = new URL("../", import.meta.url);
const files = new Set(["index.html", "styles.css", "app.js", "state.mjs"]);
const mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"};
const server = createServer(async (request, response) => {
  const name = new URL(request.url, "http://127.0.0.1").pathname.slice(1) || "index.html";
  if (!files.has(name) && !documents.has(name)) return response.writeHead(404).end();
  response.setHeader("content-type", mime[path.extname(name)]);
  response.end(files.has(name) ? await readFile(new URL(name, site)) : documents.get(name));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu"]});
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const waitForPage = number => page.waitForFunction(
    expected => document.getElementById("page-position")?.textContent === expected,
    `Página ${number} de 2. 2 registros.`,
  );
  const detail = label => page.locator("#run-details dt").filter({hasText: label})
    .evaluate(node => node.nextElementSibling.textContent);
  await waitForPage(1);
  await page.locator("#run-select").selectOption(JSON.stringify([native.run_id, native.attempt_id]));
  const number = value => new Intl.NumberFormat("es-ES", {maximumFractionDigits: 0}).format(value);
  const executable = number(native.metadata.executable_peak_rss_mib);
  const lifetime = number(native.metadata.process_lifetime_peak_rss_mib);
  assert.notEqual(executable, lifetime);
  assert.equal(await detail("VmHWM"), executable);
  assert.equal(await detail("vida del proceso"), lifetime);
  assert.match(await detail("Último punto de control"), new RegExp(`Paso ${native.checkpoint.step} `));
  assert.equal(await detail("Recuperación"), "Disponible según el registro");
  assert.equal(await page.locator("#financial-panel").isVisible(), true);
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: path.join(data, "native-mobile.png"), fullPage: true});
  await page.locator("#next-page").click();
  await waitForPage(2);
  await page.locator("#run-select").selectOption(JSON.stringify([paired.run_id, paired.attempt_id]));
  assert.equal(await detail("Condición de entrenamiento"), "Reales y bloques sintéticos");
  assert.equal(await detail("Referencia de partida"), "GRU");
  assert.match(await detail("Actividad"), /Continuación supervisada/);
  assert.equal(await page.locator("#financial-panel").isVisible(), false);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: path.join(data, "paired-mobile.png"), fullPage: true});
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({browser: browser.version(), pages: 2, native_source_sha256: native.metadata.source_sha256, executable_rss: executable, process_lifetime_rss: lifetime, condition: "real_synthetic", parent: "gru", page_errors: errors.length}));
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
