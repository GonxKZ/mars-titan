// Exporta los HTML de archify a PNG con fondo blanco.
//
// Usa el estilo de impresión del visor, que fuerza la paleta clara y quita la rejilla, y
// oculta los controles. Cada imagen incluye el diagrama, la leyenda y las tarjetas a escala 2.
//
// Uso: ARCHIFY_HOME=<carpeta de archify> node scripts/export_diagrams.mjs <html> <png> [...]
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const home = process.env.ARCHIFY_HOME;
if (!home) throw new Error('Indica la carpeta de archify en ARCHIFY_HOME');
const { ChromeVisualBrowser, findChrome } = await import(
  pathToFileURL(path.join(home, 'bin/visual-check.mjs')).href
);

const pairs = process.argv.slice(2);
if (pairs.length === 0 || pairs.length % 2) throw new Error('Se esperan parejas <html> <png>');

const STABLE = `(function () {
  document.documentElement.setAttribute('data-motion', 'still');
  var panel = document.querySelector('.diagram-container');
  if (panel) panel.setAttribute('data-detail-level', 'read');
  var ready = document.fonts && document.fonts.ready ? document.fonts.ready.catch(function () {}) : Promise.resolve();
  return ready.then(function () {
    if (window.Archify && Archify.readerLayout && Archify.readerLayout.whenStable) return Archify.readerLayout.whenStable();
  }).then(function () {
    return new Promise(function (resolve) { requestAnimationFrame(function () { requestAnimationFrame(resolve); }); });
  });
})()`;

// El título de la leyenda es del visor y sale en inglés. Se traduce solo en la imagen.
const CLEAN = `(function () {
  var style = document.createElement('style');
  var cards = Math.min(3, document.querySelectorAll('.card').length || 1);
  style.textContent = '.no-print, .toolbar, .diagram-nav, .export-wrap { display: none !important; }' +
    'html, body { background: #ffffff !important; } body { padding: 28px !important; }' +
    '.diagram-container { min-height: 0 !important; height: auto !important; }' +
    '.cards { grid-template-columns: repeat(' + cards + ', minmax(0, 1fr)) !important; }';
  document.head.appendChild(style);
  document.querySelectorAll('svg text').forEach(function (text) {
    if (text.textContent.trim() === 'Legend') text.textContent = 'Leyenda';
  });
  var container = document.querySelector('.container');
  var box = container.getBoundingClientRect();
  return { x: box.left, y: box.top + window.scrollY, width: box.width, height: container.scrollHeight };
})()`;

const chrome = findChrome();
if (!chrome) throw new Error('No se encuentra Chrome o Chromium. Indica su ruta en ARCHIFY_CHROME');
const browser = new ChromeVisualBrowser(chrome);
const cdp = browser.cdp;

async function evaluate(session, expression, awaitPromise = false) {
  const response = await cdp.send('Runtime.evaluate', { expression, awaitPromise, returnByValue: true }, session);
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result?.value;
}

try {
  const session = await browser.sessionPromise;
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1000, deviceScaleFactor: 2, mobile: false }, session);
  await cdp.send('Emulation.setEmulatedMedia', { media: 'print' }, session);
  for (let index = 0; index < pairs.length; index += 2) {
    const [page, target] = [pairs[index], pairs[index + 1]];
    const url = new URL(pathToFileURL(path.resolve(page)).href);
    url.searchParams.set('theme', 'light');
    const loaded = cdp.waitFor('Page.loadEventFired', session);
    await cdp.send('Page.navigate', { url: url.href }, session);
    await loaded;
    await evaluate(session, STABLE, true);
    const box = await evaluate(session, CLEAN);
    await evaluate(session, STABLE, true);
    const margin = 28;
    const clip = {
      x: Math.max(0, box.x - margin),
      y: Math.max(0, box.y - margin),
      width: box.width + 2 * margin,
      height: box.height + 2 * margin,
      scale: 1,
    };
    const shot = await cdp.send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: true, fromSurface: true }, session, 30000);
    fs.writeFileSync(target, Buffer.from(shot.data, 'base64'));
    console.log(`${target} ${Math.round(clip.width * 2)}x${Math.round(clip.height * 2)}`);
  }
} finally {
  await browser.close();
}
