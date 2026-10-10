// Prueba de navegador del modo local en directo contra el servidor de Python real.
//
// Prepara en una carpeta temporal un registro ficticio, una campaña por ventanas ficticia
// y un paquete de trazas de prueba, arranca scripts/serve_observatory.py con límites
// pequeños y comprueba que la página recibe eventos SSE sin sondear, que la campaña y el
// índice se actualizan sin recargar, que el límite de clientes devuelve 503 y que una
// pestaña cerrada libera su plaza.
//
// Uso: OBSERVATORY_PYTHON="uv run --locked python" \
//   node site/tests/live-browser.mjs RUTA/playwright/index.mjs

import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile, rename } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildSnapshot, trainingRun } from "./fixtures/snapshot.mjs";

if (!process.argv[2]) throw new Error("Indica la ruta local de playwright/index.mjs");
const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const repository = fileURLToPath(new URL("../../", import.meta.url));
const python = (process.env.OBSERVATORY_PYTHON ?? "python3").split(" ");
const env = { ...process.env, PYTHONPATH: join(repository, "src"), CUDA_VISIBLE_DEVICES: "-1" };
const folder = await mkdtemp(join(tmpdir(), "observatorio-directo-"));

async function json(path, value) {
  await mkdir(join(path, ".."), { recursive: true });
  // Escritura atómica, como el recolector, para que el servidor nunca lea medio archivo.
  await writeFile(`${path}.tmp`, JSON.stringify(value));
  await rename(`${path}.tmp`, path);
}

async function publish(snapshot) {
  for (const [path, body] of [...snapshot.pages, ...snapshot.windows]) await writeFile(join(folder, "public/data", path), body);
  await writeFile(join(folder, "public/data/observatory.tmp"), snapshot.index);
  await rename(join(folder, "public/data/observatory.tmp"), join(folder, "public/data/observatory.json"));
}

const now = Date.now();
await mkdir(join(folder, "public/data/pages"), { recursive: true });
await mkdir(join(folder, "public/data/windows"), { recursive: true });
await publish(buildSnapshot({ now }));
const jobs = {};
for (const window of ["fold-000", "fold-001", "fold-002"]) {
  for (const arm of ["gru", "transformer_compact"]) {
    for (const name of ["search-00", "finalist-s43", "carry-s44"]) jobs[`US/${window}/${arm}/${name}`] = false;
  }
}
const confirmed = Object.keys(jobs).slice(0, 7);
for (const job of confirmed) {
  jobs[job] = true;
  await json(join(folder, "campaign/jobs", job, "receipt.json"), { status: "completed", fixture: true });
}
const open = Object.keys(jobs)[7];
await json(join(folder, "campaign/jobs", open, "attempt-0001/run.json"), {
  global_step: 30, epochs: [1, 2, 3].map(epoch => ({ epoch, train: { mae: 0.02 - epoch * 0.001, samples_per_second: 4000 }, validation: { mae: 0.021 - epoch * 0.0008, session_mae: 0.0205 } })),
});
await json(join(folder, "campaign/summary.json"), { kind: "fixture_masked_campaign_run", status: "running", jobs });
const traces = spawnSync(python[0], [...python.slice(1), join(repository, "site/tests/fixtures/make_traces.py"), join(folder, "traces"), "--points", "5000", "--cadence", "4", "--truncated-at", "5000"], { env, encoding: "utf8" });
assert.equal(traces.status, 0, traces.stderr);

const port = await new Promise(resolve => {
  const probe = createServer().listen(0, "127.0.0.1", () => { const { port } = probe.address(); probe.close(() => resolve(port)); });
});
const server = spawn(python[0], [...python.slice(1), join(repository, "scripts/serve_observatory.py"),
  "--site", join(repository, "site"), "--public-dir", join(folder, "public/data"), "--campaign", `A=${join(folder, "campaign")}`,
  "--traces-dir", join(folder, "traces"), "--port", String(port), "--max-clients", "2", "--telemetry-seconds", "1"], { env, stdio: ["ignore", "pipe", "pipe"] });
let serverLog = "";
server.stdout.on("data", chunk => { serverLog += chunk; });
server.stderr.on("data", chunk => { serverLog += chunk; });
const origin = `http://127.0.0.1:${port}/`;

// Espera dos fotogramas para que el planificador haya dibujado lo pendiente.
async function settle(page) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function waitUntil(condition, timeout = 15_000, label = "condición") {
  const start = Date.now();
  while (!(await condition())) {
    if (Date.now() - start > timeout) throw new Error(`Tiempo agotado esperando ${label}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
}

const status = async () => (await fetch(`${origin}api/status`)).json();
const results = [];
async function check(name, body) {
  try { await body(); results.push(true); console.log(`✔ ${name}`); }
  catch (error) { results.push(false); console.log(`✘ ${name}\n  ${error.stack?.split("\n").slice(0, 4).join("\n  ")}`); }
}

let browser;
try {
  await waitUntil(() => /Observatorio en directo/.test(serverLog), 20_000, "el arranque del servidor");
  browser = await chromium.launch({ executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu", "--disable-dev-shm-usage"] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, timezoneId: "Europe/Madrid", locale: "es-ES" });
  const page = await context.newPage();
  const problems = [];
  page.on("pageerror", error => problems.push(error.message));
  page.on("console", message => { if (message.type() === "error" && !/status of 503/.test(message.text())) problems.push(message.text()); });
  await page.goto(origin);
  await page.waitForFunction(() => window.__observatory?.store.mode === "live" && window.__observatory.store.liveState === "open" && window.__observatory.store.runs.size > 0);
  await settle(page);

  await check("la página entra en modo directo y no sondea el índice", async () => {
    assert.equal(await page.locator("#source-mode").textContent(), "Directo");
    const before = await page.evaluate(() => window.__observatory.indexResource.requests);
    await page.waitForTimeout(3000);
    assert.equal(await page.evaluate(() => window.__observatory.indexResource.requests), before);
    assert.match(await page.locator("#source-age").textContent(), /Último evento/);
    assert.equal((await status()).clients, 1);
  });

  await check("la telemetría llega por eventos y llena las gráficas de recursos", async () => {
    await page.evaluate(() => { location.hash = "#vista=recursos"; });
    await page.waitForFunction(() => window.__observatory.store.telemetry?.size >= 3);
    await settle(page);
    assert.equal(await page.locator("#telemetry").isVisible(), true);
    assert.equal(await page.locator("#telemetry-tiles .tile").count(), 9);
    const size = await page.evaluate(() => window.__observatory.store.telemetry.size);
    await page.waitForFunction(count => window.__observatory.store.telemetry.size > count, size);
  });

  await check("la campaña por ventanas se lee y se actualiza sin recargar", async () => {
    await page.evaluate(() => { location.hash = "#vista=campana"; });
    await page.locator("#window-campaigns").waitFor({ state: "visible" });
    const caption = () => page.locator("#window-campaign .caption").first().textContent();
    // El índice puede llegar antes que el estado en directo: se espera a que lo sustituya.
    await page.waitForFunction(() => /7 de 18 trabajos confirmados/.test(document.querySelector("#window-campaign .caption")?.textContent ?? ""), null, { timeout: 15_000 });
    assert.match(await caption(), /7 de 18 trabajos confirmados\. 1 con intento sin confirmar/);
    // La campaña en directo se abre por defecto y las publicadas siguen en el selector.
    assert.deepEqual(await page.locator("#window-select option").evaluateAll(options => options.map(option => option.value)), ["fixture-base", "fixture-adapters", "A"]);
    assert.match(await page.locator("#window-campaign .eyebrow").textContent(), /en directo/);
    assert.equal(await page.locator("#window-campaign .attempt").count(), 1);
    jobs[open] = true;
    await json(join(folder, "campaign/jobs", open, "receipt.json"), { status: "completed", fixture: true });
    await json(join(folder, "campaign/summary.json"), { kind: "fixture_masked_campaign_run", status: "running", jobs });
    await page.waitForFunction(() => /8 de 18 trabajos confirmados/.test(document.querySelector("#window-campaign .caption")?.textContent ?? ""), null, { timeout: 15_000 });
    assert.equal(await page.locator("#window-campaign .attempt").count(), 0);
  });

  await check("un índice nuevo llega por evento y se descarga una sola vez", async () => {
    const runs = await page.evaluate(() => window.__observatory.store.runs.size);
    const requests = await page.evaluate(() => window.__observatory.indexResource.requests);
    const extra = trainingRun({ model: "gru", campaign: "neural-fixture-fold-002", seed: 46, window: 2, now, offset: 150 });
    await publish(buildSnapshot({ now, extraRuns: [extra] }));
    await page.waitForFunction(count => window.__observatory.store.runs.size === count + 1, runs, { timeout: 15_000 });
    assert.ok(await page.evaluate(() => window.__observatory.indexResource.requests) - requests <= 2);
  });

  await check("el paquete de trazas de prueba se dibuja con su aviso de procedencia", async () => {
    await page.evaluate(() => { location.hash = "#vista=memoria"; });
    await page.locator("#trace-panels .trace-panel").first().waitFor();
    assert.equal(await page.locator("#trace-fixture").isVisible(), true);
    const caption = await page.locator("#trace-caption").textContent();
    assert.match(caption, /SHA-256 [a-f0-9]{12} comprobada/);
    assert.match(caption, /un registro cada 4/);
    assert.match(caption, /dejó de registrar en el paso 5\.?000 al agotar/);
    assert.equal(await page.locator("#trace-panels .trace-panel").count(), 4);
  });

  await check("el tercer cliente recibe 503 y la página pasa a consultar el índice", async () => {
    const second = await context.newPage();
    await second.goto(origin);
    await second.waitForFunction(() => window.__observatory?.store.liveState === "open");
    const third = await context.newPage();
    await third.goto(origin);
    await third.waitForFunction(() => window.__observatory?.store.liveState === "closed", null, { timeout: 15_000 });
    await third.waitForFunction(() => window.__observatory.store.index !== null);
    await settle(third);
    assert.match(await third.locator("#error").textContent(), /conexión en directo se ha cerrado/);
    assert.match(await third.locator("#source-age").textContent(), /Directo cerrado/);
    assert.ok(await third.evaluate(() => window.__observatory.store.nextCheck !== null), "queda programada la consulta de respaldo");
    assert.equal((await status()).clients, 2);
    await third.close();
    await second.close();
    await waitUntil(async () => (await status()).clients === 1, 15_000, "la liberación de la plaza");
  });

  await check("cerrar la última pestaña libera el servidor", async () => {
    await page.close();
    await waitUntil(async () => (await status()).clients === 0, 15_000, "cero clientes");
  });

  await check("sin errores de página ni de consola en modo directo", async () => {
    assert.deepEqual(problems, []);
  });
} finally {
  await browser?.close();
  server.kill("SIGTERM");
  await new Promise(resolve => server.once("exit", resolve));
  await rm(folder, { recursive: true, force: true });
}

const failed = results.filter(ok => !ok).length;
console.log(`\n${results.length - failed} de ${results.length} comprobaciones superadas`);
if (failed) console.log(serverLog);
process.exitCode = failed ? 1 : 0;
