// Prueba de navegador del modo público (Pages) con un registro ficticio.
//
// Un servidor de Node imita Pages: sirve el sitio y los datos con ETag y responde 304
// cuando el índice no cambia. La prueba comprueba la consulta condicional, la conservación
// de páginas tras un 503, el estado en la URL, el teclado, la fase reservada, el CSV, el
// escape de texto hostil, la importación local, los temas y el ancho en móvil. Guarda
// capturas en la carpeta indicada para revisarlas a mano.
//
// Uso: node site/tests/observatory-browser.mjs RUTA/playwright/index.mjs [CARPETA_DE_CAPTURAS]

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, mkdir } from "node:fs/promises";
import { createServer } from "node:http";
import { pathToFileURL } from "node:url";
import { buildSnapshot, trainingRun, SECRET_MAE } from "./fixtures/snapshot.mjs";

if (!process.argv[2]) throw new Error("Indica la ruta local de playwright/index.mjs");
const { chromium } = await import(pathToFileURL(process.argv[2]).href);
const shots = process.argv[3] ?? null;
if (shots) await mkdir(shots, { recursive: true });
const siteRoot = new URL("../", import.meta.url);
const NOW = Date.parse("2026-10-09T12:00:00Z");
const SITE = /^(index\.html|styles\.css|[a-z][a-z0-9-]*\.m?js|vendor\/uplot\/[\w.]+|fonts\/[\w.-]+)$/;
const MIME = { html: "text/html", css: "text/css", js: "text/javascript", mjs: "text/javascript", json: "application/json", woff2: "font/woff2", txt: "text/plain" };

const data = { snapshot: buildSnapshot({ now: NOW }), failOnce: new Set(), indexRequests: 0, notModified: 0, conditional: 0 };
const tag = body => `"${createHash("sha256").update(body).digest("hex").slice(0, 16)}"`;

const server = createServer(async (request, response) => {
  const path = new URL(request.url, "http://127.0.0.1").pathname.slice(1) || "index.html";
  const send = (status, body, type, headers = {}) => {
    response.writeHead(status, { "Content-Type": `${type}; charset=utf-8`, ...headers });
    response.end(body);
  };
  if (path === "data/observatory.json") {
    data.indexRequests++;
    const etag = tag(data.snapshot.index);
    if (request.headers["if-none-match"]) data.conditional++;
    if (request.headers["if-none-match"] === etag) { data.notModified++; response.writeHead(304, { ETag: etag }).end(); return; }
    send(200, data.snapshot.index, MIME.json, { ETag: etag, "Cache-Control": "no-cache" });
    return;
  }
  if (path === "data/deployment.json") {
    send(200, JSON.stringify({ frontend_sha: "a".repeat(40), data_sha: "b".repeat(40), packaged_at: "2026-10-09T11:58:00Z" }), MIME.json);
    return;
  }
  if (path.startsWith("data/windows/")) {
    data.windowRequests = (data.windowRequests ?? 0) + 1;
    const body = data.snapshot.windows.get(path.slice(5));
    if (body) send(200, body, MIME.json, { "Cache-Control": "public, max-age=31536000, immutable" });
    else send(404, "", MIME.json);
    return;
  }
  if (path.startsWith("data/pages/")) {
    const key = path.slice(5);
    if (data.failOnce.delete(key)) { send(503, "", MIME.json); return; }
    const body = data.snapshot.pages.get(key);
    if (body) send(200, body, MIME.json, { "Cache-Control": "public, max-age=31536000, immutable" });
    else send(404, "", MIME.json);
    return;
  }
  if (!SITE.test(path)) { send(404, "", MIME.txt); return; }
  try { send(200, await readFile(new URL(path, siteRoot)), MIME[path.split(".").pop()] ?? "application/octet-stream"); } catch { send(404, "", MIME.txt); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const origin = `http://127.0.0.1:${server.address().port}/`;

const results = [];
async function check(name, body) {
  try { await body(); results.push([true, name]); console.log(`✔ ${name}`); }
  catch (error) { results.push([false, name]); console.log(`✘ ${name}\n  ${error.stack?.split("\n").slice(0, 4).join("\n  ")}`); }
}

const browser = await chromium.launch({ executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu", "--disable-dev-shm-usage"] });

async function open({ width = 1440, height = 1000, scheme = "light", hash = "", reducedMotion = "no-preference", clock = true } = {}) {
  const context = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, timezoneId: "Europe/Madrid", locale: "es-ES", reducedMotion, acceptDownloads: true });
  const page = await context.newPage();
  const problems = [];
  page.on("pageerror", error => problems.push(`pageerror ${error.message}`));
  page.on("console", message => {
    // El 404 del índice de trazas es esperado mientras Pages no publique trazas.
    // El 503 lo provoca la propia prueba de reintento de páginas.
    if (message.type() === "error" && !/status of (404|503)/.test(message.text())) problems.push(`console ${message.text()}`);
  });
  await page.addInitScript(() => document.addEventListener("securitypolicyviolation", event => console.error(`CSP ${event.violatedDirective} ${event.blockedURI}`)));
  if (clock) await page.clock.install({ time: new Date(NOW + 30_000) });
  await page.goto(origin + hash);
  await page.waitForFunction(() => window.__observatory?.store.index && !window.__observatory.store.loading.active && window.__observatory.store.runs.size > 0);
  await settle(page);
  return { page, context, problems };
}

async function waitUntil(condition, timeout = 10_000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeout) throw new Error("tiempo de espera agotado");
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}

async function settle(page) {
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

const total = () => buildSnapshot({ now: NOW }).runs.length;

try {
  const { page, context, problems } = await open();

  await check("carga el índice y todas las páginas sin errores ni violaciones de CSP", async () => {
    assert.equal(await page.locator("#pulse-runs").textContent(), String(total()));
    assert.equal(await page.evaluate(() => window.__observatory.store.runs.size), total());
    assert.equal(await page.locator("#source-mode").textContent(), "Pages");
    assert.match(await page.locator("#source-age").textContent(), /Recogido .*CEST/);
    assert.deepEqual(problems, []);
  });

  await check("la matriz de campaña muestra una marca por registro y los previstos sin registro", async () => {
    assert.equal(await page.locator("#series-title").textContent(), "fixture");
    assert.equal(await page.locator("#matrix .cell .m").count(), 23);
    assert.equal(await page.locator("#matrix .pending-count").count(), 3);
    assert.match(await page.locator("#matrix").innerText(), /\+2/);
  });

  await check("la campaña por ventanas se elige con su selector y su matriz se descarga una vez", async () => {
    await page.locator("#window-campaigns").waitFor({ state: "visible" });
    await page.waitForFunction(() => /3 de 4 trabajos confirmados/.test(document.querySelector("#window-campaign .caption")?.textContent ?? ""));
    assert.equal(await page.locator("#window-select option").count(), 2);
    assert.equal(await page.locator("#window-select").inputValue(), "fixture-base");
    assert.match(await page.locator("#window-campaign .eyebrow").textContent(), /Campaña base/);
    assert.equal(await page.locator("#window-campaign tbody tr").count(), 2);
    assert.match(await page.locator("#window-campaign tbody th").first().textContent(), /US\+CN · GRU/);
    assert.equal(await page.locator("#window-campaign .attempt").count(), 1);
    await page.selectOption("#window-select", "fixture-adapters");
    await page.waitForFunction(() => /Sin resumen todavía/.test(document.querySelector("#window-campaign .caption")?.textContent ?? ""));
    assert.match(page.url(), /matriz=fixture-adapters/);
    const requests = data.windowRequests;
    await page.selectOption("#window-select", "fixture-base");
    await page.waitForFunction(() => /3 de 4/.test(document.querySelector("#window-campaign .caption")?.textContent ?? ""));
    assert.equal(data.windowRequests, requests, "el documento inmutable no se vuelve a pedir");
    assert.deepEqual(problems, []);
  });

  await check("un índice sin cambios responde 304 y no vuelve a dibujar", async () => {
    const version = await page.evaluate(() => window.__observatory.store.version);
    const views = await page.evaluate(() => performance.getEntriesByName("mt:view:campana").length);
    const before = data.notModified;
    await page.clock.fastForward(61_000);
    await waitUntil(() => data.notModified > before);
    await settle(page);
    assert.ok(data.notModified > before, "el servidor respondió 304");
    assert.ok(data.conditional >= 1, "la petición llevó If-None-Match");
    assert.equal(await page.evaluate(() => window.__observatory.store.version), version);
    assert.equal(await page.evaluate(() => performance.getEntriesByName("mt:view:campana").length), views);
  });

  await check("un índice nuevo añade la marca nueva y la resalta", async () => {
    const extra = trainingRun({ model: "gru", campaign: "neural-fixture-fold-002", seed: 46, window: 2, now: NOW, offset: 150 });
    data.snapshot = buildSnapshot({ now: NOW, extraRuns: [extra] });
    await page.clock.fastForward(61_000);
    await page.waitForFunction(count => window.__observatory.store.runs.size === count, total() + 1);
    await settle(page);
    assert.equal(await page.locator("#matrix .cell .m").count(), 24);
    assert.ok(await page.locator("#matrix .m.m-new").count() >= 1);
  });

  await check("una página que falla con 503 conserva lo demás y se reintenta", async () => {
    data.snapshot = buildSnapshot({ now: NOW, pageSize: 8 });
    for (const path of data.snapshot.pages.keys()) { data.failOnce.add(path); break; }
    await page.clock.fastForward(61_000);
    await page.locator("#error").waitFor({ state: "visible" });
    assert.match(await page.locator("#error").textContent(), /No se pudieron leer 1 páginas/);
    assert.equal(await page.evaluate(() => window.__observatory.store.runs.size), total() - 8);
    await page.clock.fastForward(61_000);
    await page.waitForFunction(count => window.__observatory.store.runs.size === count, total());
    assert.equal(await page.locator("#error").isHidden(), true);
  });

  await check("las pestañas se recorren con el teclado y la vista queda en la URL", async () => {
    await page.locator("#tab-campana").focus();
    await page.keyboard.press("ArrowRight");
    await page.waitForFunction(() => location.hash.includes("vista=curvas"));
    assert.equal(await page.locator("#tab-curvas").getAttribute("aria-selected"), "true");
    assert.equal(await page.locator("#curvas").isVisible(), true);
    await page.keyboard.press("End");
    await page.waitForFunction(() => location.hash.includes("vista=metodo"));
    await page.keyboard.press("Home");
    await page.waitForFunction(() => !location.hash.includes("vista="));
  });

  await check("las curvas excluyen RL, generación y fases reservadas", async () => {
    await page.evaluate(() => { location.hash = "#vista=curvas"; });
    await page.locator("#facets .facet").first().waitFor();
    const names = await page.locator("#facets .facet h2").allTextContents();
    assert.deepEqual(names.sort(), ["GRU", "LSTM", "Ridge"]);
    const rows = await page.locator("#curve-table tbody tr").count();
    assert.equal(rows, 21 + 1, "18 neuronales, 3 Ridge y el registro con texto hostil");
    assert.match(await page.locator("#curve-caption").textContent(), /Se dibujan todos los puntos/);
  });

  await check("el estado de la URL fija medida, escala y ventana al recargar", async () => {
    await page.evaluate(() => { location.hash = "#vista=curvas&medida=session_mae&escala=log&ventana=fold-001"; });
    await settle(page);
    await page.waitForFunction(() => document.getElementById("curve-metric").value === "session_mae");
    assert.equal(await page.locator("#curve-scale").inputValue(), "log");
    assert.equal(await page.locator("#curve-window").inputValue(), "fold-001");
    await page.reload();
    await page.waitForFunction(() => window.__observatory?.store.runs.size > 20);
    await settle(page);
    assert.equal(await page.locator("#curve-metric").inputValue(), "session_mae");
  });

  await check("los modos de marca cambian la codificación sin perder la leyenda", async () => {
    await page.evaluate(() => { location.hash = "#vista=campana&marcas=relativo"; });
    await page.locator("#matrix-ramp").waitFor({ state: "visible" });
    assert.equal(await page.locator("#matrix-legend").isHidden(), true);
    assert.ok(await page.locator('#matrix .m[data-v="value"]').count() >= 18);
    assert.ok(await page.locator('#matrix .m[data-v="none"]').count() >= 1, "la ejecución en curso no tiene MAE final");
    await page.locator("#marks-relative").focus();
    await page.keyboard.press("ArrowRight");
    await page.waitForFunction(() => !location.hash.includes("marcas="));
    assert.equal(await page.locator("#matrix-legend").isVisible(), true);
  });

  await check("una celda se abre con el teclado y lista sus ejecuciones", async () => {
    await page.locator("#matrix button.cell").first().focus();
    await page.keyboard.press("Enter");
    await page.locator("#cell-detail").waitFor({ state: "visible" });
    assert.ok(await page.locator("#cell-runs li").count() >= 3);
    // La URL se escribe con un retardo máximo de 250 ms para agrupar cambios seguidos.
    await page.waitForFunction(() => /celda=/.test(location.hash));
  });

  await check("el detalle escapa texto hostil y cierra con Escape", async () => {
    await page.evaluate(() => { location.hash = "#vista=registros&buscar=hostile"; });
    await page.waitForFunction(() => document.querySelectorAll("#records tbody tr").length === 1);
    await page.locator("#records tbody .link").first().press("Enter");
    await page.locator("#drawer").waitFor({ state: "visible" });
    assert.equal(await page.locator("#drawer-body img").count(), 0);
    assert.match(await page.locator("#drawer-body").innerText(), /<img src=x onerror=alert\(1\)>/);
    assert.match(await page.locator("#drawer-body").innerText(), /RSS máximo durante la vida del proceso/);
    await page.waitForFunction(() => /ejecucion=hostile-run/.test(location.hash));
    await page.keyboard.press("Escape");
    await page.waitForFunction(() => !document.getElementById("drawer").open);
    await page.waitForFunction(() => !location.hash.includes("ejecucion="));
  });

  await check("una fase reservada no deja ver su MAE en ninguna vista, detalle ni CSV", async () => {
    const secret = /0[,.]98765/;
    for (const view of ["campana", "curvas", "recursos", "rl", "registros"]) {
      await page.evaluate(v => { location.hash = `#vista=${v}`; }, view);
      await settle(page);
      assert.doesNotMatch(await page.locator("body").innerText(), secret, view);
    }
    await page.evaluate(() => { location.hash = "#vista=registros&buscar=reserved"; });
    await page.waitForFunction(() => document.querySelectorAll("#records tbody tr").length === 1);
    assert.match(await page.locator("#records tbody").innerText(), /Reservado/);
    await page.locator("#records tbody .link").first().click();
    await page.locator("#drawer").waitFor({ state: "visible" });
    assert.match(await page.locator("#drawer-body").innerText(), /Fase reservada/);
    assert.doesNotMatch(await page.locator("#drawer-body").innerText(), secret);
    await page.locator("#drawer-close").click();
    await page.evaluate(() => { location.hash = "#vista=registros"; });
    await settle(page);
    const [download] = await Promise.all([page.waitForEvent("download"), page.locator("#records-csv").click()]);
    const csv = await readFile(await download.path(), "utf8");
    assert.ok(!csv.includes(String(SECRET_MAE)));
    assert.match(csv, /"'=HYPERLINK\(""x""\)"/);
    assert.equal(csv.trim().split("\r\n").length, total() + 1);
  });

  await check("las actividades no predictivas se rotulan y van a su vista", async () => {
    await page.evaluate(() => { location.hash = "#vista=registros&buscar=synthetic-generation"; });
    await page.waitForFunction(() => document.querySelectorAll("#records tbody tr").length === 1);
    await page.locator("#records tbody .link").first().click();
    await page.locator("#drawer").waitFor({ state: "visible" });
    assert.match(await page.locator("#drawer-kind").textContent(), /Generación sintética/);
    const text = await page.locator("#drawer-body").innerText();
    assert.match(text, /Mundo sintético/);
    assert.match(text, /Reales y bloques sintéticos/);
    assert.doesNotMatch(text, /Curvas por época/);
    await page.keyboard.press("Escape");
    await page.evaluate(() => { location.hash = "#vista=rl"; });
    await page.locator("#rl-facets .facet").first().waitFor();
    assert.equal(await page.locator("#rl-facets .facet").count(), 3);
    assert.equal(await page.locator("#rl-table tbody tr").count(), 3);
    assert.match(await page.locator("#rl-table").textContent(), /Episodio incompleto/);
  });

  await check("la importación local rechaza JSON roto y archivos grandes y se puede abandonar", async () => {
    await page.locator("#import-file").setInputFiles({ name: "roto.json", mimeType: "application/json", buffer: Buffer.from("{roto") });
    await page.locator("#error").waitFor({ state: "visible" });
    assert.match(await page.locator("#error").textContent(), /no contiene JSON válido/);
    await page.locator("#import-file").setInputFiles({ name: "grande.json", mimeType: "application/json", buffer: Buffer.alloc(8 * 1024 * 1024 + 1, 32) });
    await page.waitForFunction(() => /8 MiB/.test(document.getElementById("error").textContent));
    await page.locator("#import-file").setInputFiles({ name: "local.json", mimeType: "application/json", buffer: Buffer.from(data.snapshot.index) });
    await page.locator("#local-notice").waitFor({ state: "visible" });
    assert.equal(await page.locator("#source-mode").textContent(), "Archivo");
    assert.equal(await page.locator("#refresh").isDisabled(), true);
    await page.locator("#leave-local").click();
    await page.waitForFunction(count => window.__observatory.store.mode === "public" && window.__observatory.store.runs.size === count, total());
  });

  await check("el tema alterna entre sistema, claro y oscuro con su propio fondo", async () => {
    const background = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const light = await background();
    await page.locator("#theme").click();
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), "light");
    await page.locator("#theme").click();
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), "dark");
    assert.notEqual(await background(), light);
    await page.locator("#theme").click();
    assert.equal(await page.evaluate(() => document.documentElement.dataset.theme), undefined);
  });

  await check("ninguna vista desborda en horizontal entre 320 y 1440 píxeles", async () => {
    for (const width of [320, 390, 768, 1024, 1440]) {
      await page.setViewportSize({ width, height: 900 });
      for (const view of ["campana", "curvas", "recursos", "memoria", "rl", "registros", "metodo"]) {
        await page.evaluate(v => { location.hash = `#vista=${v}`; }, view);
        await settle(page);
        await page.waitForTimeout(50);
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
        assert.ok(overflow <= 0, `${view} a ${width}px desborda ${overflow}px`);
      }
    }
  });

  await check("no hay errores de página, de consola ni de CSP tras recorrer todo", async () => {
    assert.deepEqual(problems, []);
  });
  await context.close();

  await check("con movimiento reducido el recorrido del método no desplaza el punto", async () => {
    const { page: still, context: quiet } = await open({ hash: "#vista=metodo", reducedMotion: "reduce" });
    await still.locator("#flow-play").click();
    await still.clock.runFor(4300);
    const steps = await still.locator("#flow-diagram [data-step]").count();
    assert.equal(await still.locator("#flow-diagram [data-step].lit").count(), steps);
    assert.equal(await still.locator("#flow-token").evaluate(node => getComputedStyle(node).opacity), "0");
    await quiet.close();
  });

  if (shots) {
    await check("capturas de escritorio y móvil en claro y oscuro", async () => {
      for (const [width, height, label] of [[1440, 1000, "escritorio"], [390, 844, "movil"]]) {
        for (const scheme of ["light", "dark"]) {
          for (const view of ["campana", "curvas", "recursos"]) {
            const { page: shot, context: frame } = await open({ width, height, scheme, hash: `#vista=${view}` });
            await shot.waitForTimeout(300);
            await shot.screenshot({ path: `${shots}/${label}-${scheme === "light" ? "claro" : "oscuro"}-${view}.png`, fullPage: true });
            await frame.close();
          }
        }
      }
    });
  }
} finally {
  await browser.close();
  server.close();
}

const failed = results.filter(([ok]) => !ok).length;
console.log(`\n${results.length - failed} de ${results.length} comprobaciones superadas`);
process.exitCode = failed ? 1 : 0;
