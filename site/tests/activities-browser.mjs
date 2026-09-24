import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {createServer} from "node:http";
import {pathToFileURL} from "node:url";
import path from "node:path";

const {chromium} = await import(pathToFileURL(process.argv[2]).href);
const data = process.argv[3];
const snapshot = JSON.parse(await readFile(path.join(data, "observatory.json")));
const site = new URL("../", import.meta.url);
const files = new Set(["index.html", "styles.css", "app.js", "state.mjs"]);
const mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json"};
const server = createServer(async (request, response) => {
  const name = new URL(request.url, "http://localhost").pathname.slice(1) || "index.html";
  if (!files.has(name) && name !== "data/observatory.json") return response.writeHead(404).end();
  response.setHeader("content-type", mime[path.extname(name)]);
  response.end(await readFile(files.has(name) ? new URL(name, site) : path.join(data, "observatory.json")));
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({executablePath: "/opt/google/chrome/chrome", headless: true, args: ["--disable-gpu"]});
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(3000);
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.locator("#run-select").waitFor();
  for (const run of snapshot.runs) {
    await page.locator("#run-select").selectOption(JSON.stringify([run.run_id, run.attempt_id]));
    assert.equal(await page.locator("#predictive-curve").isVisible(), false);
    assert.equal(await page.locator("#financial-panel").isVisible(), run.financial_validation !== null);
    assert.doesNotMatch(await page.locator("#key-metrics").textContent(), /MAE/);
    if (run.activity === "synthetic_generation") assert.match(await page.locator("#run-details").textContent(), /Generación sintética/);
    if (run.financial_validation?.invalid_reason === "missing_close") assert.match(await page.locator("#financial-metrics").textContent(), /Falta un cierre/);
  }
  assert.equal(await page.locator("#comparison-group").isDisabled(), true);
  await page.locator("#history-status").selectOption("blocked");
  assert.equal(await page.locator("#history-body tr").count(), 1);
  const financial = snapshot.runs.find(run => run.model_id === "ppo");
  await page.locator("#run-select").selectOption(JSON.stringify([financial.run_id, financial.attempt_id]));
  await page.setViewportSize({width: 390, height: 844});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({path: path.join(data, "activities-mobile.png"), fullPage: true});
  assert.deepEqual(errors, []);
  console.log("Actividades comprobadas: generación, PPO, simulación, bloqueo y separación de medidas.");
} finally {
  await browser.close();
  await new Promise(resolve => server.close(resolve));
}
