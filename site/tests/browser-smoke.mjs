import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { pathToFileURL } from "node:url";

// Herramienta opcional de verificación. Playwright no es una dependencia del sitio.
if (!process.argv[2]) throw new Error("Indica la ruta local de playwright/index.mjs");
const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const siteRoot = new URL("../", import.meta.url);
const catalog = JSON.parse(await readFile(new URL("data/observatory.json", siteRoot), "utf8"));
const mime = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json" };
const paths = new Set(["index.html", "styles.css", "app.js", "state.mjs", "data/observatory.json"]);
let invalidResponse = false;
let oversizedResponse = false;
let dataRequests = 0;
const server = createServer(async (request, response) => {
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  const file = pathname === "/" ? "index.html" : pathname.slice(1);
  if (!paths.has(file)) { response.writeHead(404).end(); return; }
  if (file.endsWith(".json")) dataRequests += 1;
  response.setHeader("Content-Type", `${mime[file.slice(file.lastIndexOf("."))]}; charset=utf-8`);
  response.setHeader("Cache-Control", "no-store");
  if (oversizedResponse && file.endsWith(".json")) {
    response.write(" ".repeat(4 * 1024 * 1024));
    response.write(" ".repeat(4 * 1024 * 1024));
    response.end(" ");
    return;
  }
  response.end(invalidResponse && file.endsWith(".json") ? "invalid-json" : await readFile(new URL(file, siteRoot)));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({ executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu", "--disable-dev-shm-usage"] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1050 }, reducedMotion: "reduce" });
  await page.clock.install({ time: new Date("2026-09-18T20:00:00Z") });
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(origin);
  await page.locator(".model-item").first().waitFor();
  assert.equal(await page.locator(".model-item").count(), 9);
  assert.equal(await page.locator("#run-heading").textContent(), "Sin ejecuciones registradas");
  assert.equal(await page.locator("#export-button").isDisabled(), true);
  assert.equal(await page.locator("#connection-status").getAttribute("data-state"), "idle");
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior), "auto");
  await page.keyboard.press("Tab");
  assert.equal(await page.locator(".skip-link").evaluate(node => node === document.activeElement), true);
  await page.keyboard.press("Enter");
  assert.equal(await page.locator("#main").evaluate(node => node === document.activeElement), true);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.screenshot({ path: "/tmp/mars-titan-observatory-full.png", fullPage: true });
  const now = "2026-09-18T20:00:00Z";
  const metrics = {
    mae: .012, mse: .0002, rank_ic: .15, coverage_80: .8, coverage_95: .94, loss: .02,
    latency_p50_ms: 2, latency_p95_ms: 4, latency_p99_ms: 7, vram_peak_mib: 800,
    ram_peak_mib: 400, elapsed_seconds: 60, samples_per_second: 12,
  };
  const base = {
    run_id: "fixture-only", attempt_id: "attempt-01", model_id: "B1", variant_id: "<img src=x onerror=alert(1)>",
    status: "running", phase: "train", heartbeat_at: "2020-01-01T00:00:00Z", started_at: "2020-01-01T00:00:00Z",
    updated_at: now, completed_steps: 3, total_steps: 10, epoch: 1, max_epochs: 3, seed: 17,
    fold: "fold-01", comparison_group: "fixture-validation", metrics,
    history: [{ step: 1, recorded_at: now, loss: .03, mae: .02 }, { step: 2, recorded_at: now, loss: null, mae: null }, { step: 3, recorded_at: now, loss: .02, mae: .012 }],
    checkpoint: { step: 2, saved_at: now, resumable: true }, test_released: false,
  };
  const local = {
    ...catalog, generated_at: now, source_status: "available", notes: ["Datos sintéticos usados solo en la prueba de interfaz."],
    runs: [base,
      { ...base, run_id: "validation-a", phase: "validation", status: "completed", variant_id: "base" },
      { ...base, run_id: "validation-b", phase: "validation", status: "completed", variant_id: "base", metrics: { ...metrics, mae: .01 } },
      { ...base, run_id: "private-evaluation", phase: "evaluation", status: "completed", variant_id: "base", metrics: { ...metrics, mae: .987654321 } },
      { ...base, run_id: "fresh-run", variant_id: "base", heartbeat_at: now },
    ],
  };
  await page.locator("#import-file").setInputFiles({ name: "fixture-local.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(local)) });
  await page.locator("#local-notice").waitFor({ state: "visible" });
  assert.equal(await page.locator("#run-status").textContent(), "Sin actualización");
  assert.equal(await page.locator("#run-description img").count(), 0);
  assert.match(await page.locator("#run-description").textContent(), /<img/);
  assert.equal(await page.locator("#comparison-body tr").count(), 2);
  assert.equal(await page.locator("#run-progress").getAttribute("value"), "30");
  assert.equal(await page.locator(".chart-point").count(), 2);
  assert.ok((await page.locator(".curve-summary").boundingBox()).height < 60, "El pie de curva no debe ocupar el espacio de un estado vacío");
  const before = dataRequests;
  await page.locator("#run-select").selectOption(JSON.stringify(["fresh-run", "attempt-01"]));
  assert.equal(await page.locator("#run-status").textContent(), "En curso");
  await page.locator("#history-body").evaluate(node => { node.firstElementChild.dataset.healthProbe = "kept"; });
  await page.clock.fastForward(181000);
  assert.equal(await page.locator("#run-status").textContent(), "Sin actualización", "El reloj de salud también debe avanzar sin peticiones después de importar");
  assert.equal(await page.locator("[data-health-probe=kept]").count(), 1, "Actualizar la salud no reconstruye filas sin cambios de filtro");
  assert.equal(dataRequests, before, "La importación local debe detener la consulta pública");
  await page.locator("#history-search").fill("private-evaluation");
  assert.equal(await page.locator("#history-body tr").count(), 1);
  await page.locator("#history-body button").click();
  assert.equal(await page.locator("#test-notice").isVisible(), true);
  assert.ok(!(await page.locator("#run-content").textContent()).includes("0,9876"));
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#export-button").click();
  const download = await downloadPromise;
  const csv = await readFile(await download.path(), "utf8");
  assert.ok(!csv.includes("0.987654321"));
  await page.locator("#import-file").setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from('{"schema_version":99}') });
  await page.locator("#error-message").waitFor({ state: "visible" });
  assert.equal(await page.locator("#history-body tr").count(), 1);
  invalidResponse = true;
  await page.locator("#public-source-button").click();
  await page.waitForFunction(() => document.querySelector("#error-message").textContent.includes("JSON válido"));
  assert.equal(await page.locator("#source-label").textContent(), "Archivo local", "Una lectura pública fallida no cambia el origen del último resumen válido");
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  if (process.argv[3]) {
    const exported = await readFile(process.argv[3]);
    assert.ok(exported.byteLength > 1024 * 1024 && exported.byteLength < 8 * 1024 * 1024);
    await page.locator("#import-file").setInputFiles({ name: "exporter-fixture.json", mimeType: "application/json", buffer: exported });
    await page.waitForFunction(() => document.querySelector("#run-select").options.length === 32);
    assert.equal(await page.locator("#error-message").isVisible(), false);
    console.log(`Importación real de la CLI al navegador: ${exported.byteLength} bytes y 32 ejecuciones.`);
  }
  await page.locator("#import-file").setInputFiles({ name: "too-large.json", mimeType: "application/json", buffer: Buffer.alloc(8 * 1024 * 1024 + 1, " ") });
  await page.waitForFunction(() => document.querySelector("#error-message").textContent.includes("8 MiB"));
  invalidResponse = false;
  oversizedResponse = true;
  await page.locator("#public-source-button").click();
  await page.waitForFunction(() => document.querySelector("#error-message").textContent.includes("El resumen supera"));
  assert.equal(await page.locator("#source-label").textContent(), "Archivo local");
  oversizedResponse = false;
  invalidResponse = false;
  await page.locator("#refresh-button").click();
  await page.waitForFunction(() => document.querySelector("#source-label").textContent === "Registro público");
  await page.screenshot({ path: "/tmp/mars-titan-observatory-mobile-full.png", fullPage: true });
  await page.emulateMedia({ media: "print" });
  assert.equal(await page.locator(".navigation-wrap").isVisible(), false);
  assert.deepEqual(errors, []);
  console.log("Verificación de navegador: estado vacío, teclado, movimiento reducido, importación local, caducidad sin red, XSS, curva, comparación, filtros, CSV protegido, límites de bytes, error recuperable, móvil e impresión correctos.");
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
