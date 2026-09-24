import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {createServer} from "node:http";
import {pathToFileURL} from "node:url";
import path from "node:path";

const {chromium} = await import(pathToFileURL(process.argv[2]).href);
const data = process.argv[3];
const site = new URL("../", import.meta.url);
const allowed = new Set(["index.html", "styles.css", "app.js", "state.mjs"]);
const mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"};
let pageRequests = 0, failPages = false;
const server = createServer(async (request, response) => {
  const name = new URL(request.url, "http://127.0.0.1").pathname.slice(1) || "index.html";
  const isPage = /^data\/pages\/[a-f0-9]{64}\.json$/.test(name);
  if (!allowed.has(name) && name !== "data/observatory.json" && !isPage) return response.writeHead(404).end();
  if (isPage) pageRequests++;
  if (isPage && failPages) return response.writeHead(503).end();
  response.setHeader("content-type", mime[path.extname(name)]);
  response.end(await readFile(allowed.has(name) ? new URL(name, site) : path.join(data, name.slice(5))));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu"]});
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator("#history-pages").waitFor();
  assert.equal(pageRequests, 0);
  assert.match(await page.locator("#page-position").textContent(), /Página 1 de 9/);
  await page.locator("#next-page").click();
  await page.waitForFunction(() => document.getElementById("page-position").textContent.startsWith("Página 2"));
  assert.equal(pageRequests, 1);
  failPages = true;
  await page.locator("#next-page").click();
  await page.locator("#error-message").waitFor();
  assert.match(await page.locator("#page-position").textContent(), /Página 2/);
  failPages = false;
  for (let n = 3; n <= 9; n++) {
    await page.locator("#next-page").click();
    await page.waitForFunction(n => document.getElementById("page-position").textContent.startsWith(`Página ${n} `), n);
  }
  assert.equal(await page.locator("#next-page").isDisabled(), true);
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: "/tmp/mars-titan-campaigns-mobile.png", fullPage: true});
  assert.deepEqual(errors, []);
  console.log("Paginación comprobada: carga bajo demanda, nueve páginas, fallo recuperable y móvil.");
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
