// Medidas de coste del observatorio en el portátil. No es una prueba de aprobado o
// suspenso: imprime un JSON con tiempos de dibujo y uso de CPU para documentarlos.
//
// 1. Serie de un millón de puntos de un paquete de prueba: lectura y comprobación de la
//    huella, dibujo con M4, ampliaciones y cambio a todos los puntos.
// 2. CPU de la pestaña (TaskDuration de Chrome) y del servidor (/proc) durante un minuto
//    en reposo, con la vista de campaña y con la de recursos, que recibe telemetría.
// 3. CPU del servidor sin clientes.
// 4. CPU de la pestaña en modo público, con un servidor estático que responde 304.
//
// Uso: OBSERVATORY_PYTHON="uv run --locked python" node site/tests/benchmark-browser.mjs \
//   RUTA/playwright/index.mjs [CARPETA_PUBLICA] [SEGUNDOS]

import { spawn, spawnSync } from "node:child_process";
import { mkdtemp, mkdir, rm, writeFile, readFile, readdir } from "node:fs/promises";
import { createServer } from "node:net";
import { createServer as createHttpServer } from "node:http";
import { stat } from "node:fs/promises";
import { tmpdir, cpus } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { buildSnapshot } from "./fixtures/snapshot.mjs";

const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const repository = fileURLToPath(new URL("../../", import.meta.url));
const python = (process.env.OBSERVATORY_PYTHON ?? "python3").split(" ");
const env = { ...process.env, PYTHONPATH: join(repository, "src"), CUDA_VISIBLE_DEVICES: "-1" };
const seconds = Number(process.argv[4] ?? 60);
const folder = await mkdtemp(join(tmpdir(), "observatorio-medidas-"));
let publicDir = process.argv[3];
if (!publicDir) {
  publicDir = join(folder, "public");
  const snapshot = buildSnapshot({ now: Date.now() });
  await mkdir(join(publicDir, "pages"), { recursive: true });
  for (const [path, body] of snapshot.pages) await writeFile(join(publicDir, path), body);
  await writeFile(join(publicDir, "observatory.json"), snapshot.index);
}
const made = spawnSync(python[0], [...python.slice(1), join(repository, "site/tests/fixtures/make_traces.py"), join(folder, "traces"), "--points", "1000000"], { env, encoding: "utf8" });
if (made.status !== 0) throw new Error(made.stderr);

const port = await new Promise(resolve => { const probe = createServer().listen(0, "127.0.0.1", () => { const { port } = probe.address(); probe.close(() => resolve(port)); }); });
const server = spawn(python[0], [...python.slice(1), join(repository, "scripts/serve_observatory.py"), "--site", join(repository, "site"),
  "--public-dir", publicDir, "--traces-dir", join(folder, "traces"), "--port", String(port)], { env, stdio: ["ignore", "pipe", "pipe"] });
let log = "";
server.stdout.on("data", chunk => { log += chunk; });
server.stderr.on("data", chunk => { log += chunk; });
const origin = `http://127.0.0.1:${port}/`;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
while (!/Observatorio en directo/.test(log)) { if (server.exitCode !== null) throw new Error(log); await sleep(100); }

// CPU acumulada del servidor y de sus descendientes, en segundos, leída de /proc.
async function processCpu(pid) {
  let total = 0;
  const stack = [pid];
  while (stack.length) {
    const current = stack.pop();
    try {
      const stat = (await readFile(`/proc/${current}/stat`, "utf8")).split(") ")[1].split(" ");
      total += (Number(stat[11]) + Number(stat[12])) / 100;
      for (const task of await readdir(`/proc/${current}/task`)) {
        const children = (await readFile(`/proc/${current}/task/${task}/children`, "utf8")).trim();
        if (children) stack.push(...children.split(" ").map(Number));
      }
    } catch { /* el proceso terminó */ }
  }
  return total;
}

async function cpuWindow(page, label) {
  const client = await page.context().newCDPSession(page);
  await client.send("Performance.enable");
  const metric = async () => Object.fromEntries((await client.send("Performance.getMetrics")).metrics.map(m => [m.name, m.value]));
  const before = await metric(), serverBefore = await processCpu(server.pid), start = performance.now();
  await sleep(seconds * 1000);
  const after = await metric(), serverAfter = await processCpu(server.pid), wall = (performance.now() - start) / 1000;
  await client.detach();
  return {
    label, seconds: Number(wall.toFixed(1)),
    page_cpu_percent: Number(((after.TaskDuration - before.TaskDuration) / wall * 100).toFixed(3)),
    page_script_ms: Number(((after.ScriptDuration - before.ScriptDuration) * 1000).toFixed(1)),
    page_layout_count: after.LayoutCount - before.LayoutCount,
    server_cpu_percent: Number(((serverAfter - serverBefore) / wall * 100).toFixed(3)),
  };
}

const report = { cpus: cpus().length, cpu_model: cpus()[0]?.model, window_seconds: seconds, public_dir_runs: null };
const browser = await chromium.launch({ executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu", "--disable-dev-shm-usage"] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(`${origin}#vista=memoria`);
  await page.waitForFunction(() => window.__observatory?.store.mode === "live" && !window.__observatory.store.loading.active && window.__observatory.store.runs.size > 0, null, { timeout: 120_000 });
  report.public_dir_runs = await page.evaluate(() => window.__observatory.store.runs.size);
  await page.locator("#trace-panels .trace-panel .uplot").first().waitFor({ timeout: 120_000 });
  await page.waitForTimeout(500);
  const measures = names => page.evaluate(list => performance.getEntriesByType("measure").filter(m => list.includes(m.name)).map(m => [m.name, Number(m.duration.toFixed(2))]), names);
  report.trace = {
    points_per_series: 1_000_000,
    load_and_verify_ms: (await measures(["mt:trace-load"]))[0]?.[1] ?? null,
    first_draw_m4_ms: (await measures(["mt:long-m4"])).map(([, value]) => value),
    heatmap_ms: (await measures(["mt:heatmap"])).map(([, value]) => value),
  };
  // Ampliaciones sucesivas sobre la primera serie, con M4 y después con todos los puntos.
  const zoom = full => page.evaluate(async useFull => {
    const chart = window.__observatory.ctx.chartsByView.get("memoria").find(item => item.x?.length === 1_000_000);
    if (useFull) chart.setFull(true);
    const spans = [1_000_000, 250_000, 50_000, 5_000, 1_000_000];
    const drawn = [];
    for (const span of spans) {
      chart.chart.setScale("x", { min: 1, max: span });
      drawn.push(chart.drawn);
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    if (useFull) chart.setFull(false);
    return drawn;
  }, full);
  report.trace.m4_drawn_points = await zoom(false);
  report.trace.m4_zoom_ms = (await measures(["mt:long-m4-zoom"])).map(([, value]) => value).slice(-5);
  report.trace.full_drawn_points = await zoom(true);
  report.trace.full_switch_ms = (await measures(["mt:long-full-switch"])).map(([, value]) => value);
  report.trace.full_zoom_ms = (await measures(["mt:long-full-zoom"])).map(([, value]) => value).slice(-5);
  report.trace.heap_mib = Number(((await page.evaluate(() => performance.memory?.usedJSHeapSize ?? 0)) / 1048576).toFixed(1));

  await page.evaluate(() => { location.hash = "#vista=campana"; });
  await page.waitForTimeout(2000);
  report.idle_campaign = await cpuWindow(page, "directo, vista de campaña");
  await page.evaluate(() => { location.hash = "#vista=recursos"; });
  await page.waitForTimeout(2000);
  report.idle_resources = await cpuWindow(page, "directo, vista de recursos con telemetría cada 5 s");
  await page.close();
  // Modo público: servidor estático con ETag, sin /api, como Pages.
  const site = /^(index\.html|styles\.css|[a-z][a-z0-9-]*\.m?js|vendor\/uplot\/[\w.]+|fonts\/[\w.-]+)$/;
  const types = { html: "text/html", css: "text/css", js: "text/javascript", mjs: "text/javascript", json: "application/json", woff2: "font/woff2" };
  const statics = createHttpServer(async (request, response) => {
    const path = new URL(request.url, "http://x").pathname.slice(1) || "index.html";
    const file = path.startsWith("data/") ? join(publicDir, path.slice(5)) : site.test(path) ? join(repository, "site", path) : null;
    try {
      const info = await stat(file);
      const etag = `"${info.size}-${info.mtimeMs}"`;
      if (request.headers["if-none-match"] === etag) { response.writeHead(304, { ETag: etag }).end(); return; }
      response.writeHead(200, { "Content-Type": types[path.split(".").pop()] ?? "application/octet-stream", ETag: etag });
      response.end(await readFile(file));
    } catch { response.writeHead(404).end(); }
  });
  await new Promise(resolve => statics.listen(0, "127.0.0.1", resolve));
  const publicPage = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await publicPage.goto(`http://127.0.0.1:${statics.address().port}/`);
  await publicPage.waitForFunction(() => window.__observatory?.store.mode === "public" && window.__observatory.store.index && !window.__observatory.store.loading.active && window.__observatory.store.runs.size > 0, null, { timeout: 120_000 });
  await publicPage.waitForTimeout(2000);
  report.idle_public = await cpuWindow(publicPage, "Pages, vista de campaña, consulta condicional cada 60 s");
  report.idle_public.index_requests = await publicPage.evaluate(() => window.__observatory.indexResource.requests);
  report.idle_public.not_modified = await publicPage.evaluate(() => window.__observatory.indexResource.notModified);
  delete report.idle_public.server_cpu_percent;
  await publicPage.close();
  statics.close();
  await sleep(3000);
  const serverBefore = await processCpu(server.pid), start = performance.now();
  await sleep(seconds * 1000);
  report.server_without_clients_percent = Number(((await processCpu(server.pid) - serverBefore) / ((performance.now() - start) / 1000) * 100).toFixed(3));
} finally {
  await browser.close();
  server.kill("SIGTERM");
  await new Promise(resolve => server.once("exit", resolve));
  await rm(folder, { recursive: true, force: true });
}
console.log(JSON.stringify(report, null, 2));
