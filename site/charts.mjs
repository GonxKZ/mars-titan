// Gráficas del observatorio sobre canvas.
//
// Las líneas usan uPlot 1.6.32, que dibuja series largas en canvas y agrega por columna
// de píxel al trazar. El diagrama de dispersión y el mapa de calor son propios porque
// necesitan escalas logarítmicas por grupo y una escala de color cividis. Todas las
// gráficas comparten la misma lectura del cursor: el valor exacto con su unidad, leído
// de la serie completa aunque se haya reducido el dibujo.

import { m4Indices, nearestIndex, pick } from "./decimate.mjs";
import { number, significant, timestamp } from "./format.mjs";

const SANS = '"Atkinson Hyperlegible Next", system-ui, sans-serif';
const MONO = '"Atkinson Hyperlegible Mono", ui-monospace, monospace';

// Paleta cividis de Nuñez, Anderton y Renslow (2018), muestreada de matplotlib 3.11 en 17
// puntos. Su luminosidad crece de forma monótona y se distingue con daltonismo.
const CIVIDIS = ["#00224e", "#002e6a", "#1a386f", "#32436d", "#434e6c", "#535a6d", "#61656f", "#6f7073",
  "#7d7c78", "#8c8878", "#9b9476", "#aba072", "#bcae6c", "#cdbb63", "#dec958", "#f0d846", "#fee838"];

function hexToRgb(hex) {
  const value = parseInt(hex.slice(1), 16);
  return [value >> 16, (value >> 8) & 255, value & 255];
}

const CIVIDIS_LUT = (() => {
  const stops = CIVIDIS.map(hexToRgb);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const position = i / 255 * (stops.length - 1), low = Math.floor(position), high = Math.min(stops.length - 1, low + 1), t = position - low;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[low][c] + (stops[high][c] - stops[low][c]) * t;
  }
  return lut;
})();

export function tokens() {
  const style = getComputedStyle(document.documentElement);
  const get = name => style.getPropertyValue(name).trim();
  return {
    ink: get("--ink"), ink2: get("--ink-2"), ink3: get("--ink-3"), rule: get("--rule"), grid: get("--grid"),
    surface: get("--surface"), accent: get("--accent"), context: get("--context"), dark: get("--scheme") === "dark",
    series: [1, 2, 3, 4, 5, 6].map(i => get(`--series-${i}`)),
  };
}

// En modo claro, más valor es más oscuro. En modo oscuro se invierte para que el valor
// alto destaque sobre el fondo, como pide la convención de las escalas secuenciales.
export function cividis(t, dark) {
  const index = Math.round(Math.min(1, Math.max(0, dark ? t : 1 - t)) * 255) * 3;
  return [CIVIDIS_LUT[index], CIVIDIS_LUT[index + 1], CIVIDIS_LUT[index + 2]];
}

// Escala divergente de Okabe e Ito: azul para valores por debajo de la referencia,
// bermellón por encima y un gris neutro en el centro, sin tono propio.
const DIVERGING = {
  light: { low: [0, 114, 178], mid: [201, 198, 189], high: [213, 94, 0] },
  dark: { low: [43, 142, 186], mid: [74, 79, 87], high: [211, 87, 29] },
};

export function diverging(t, dark) {
  const { low, mid, high } = DIVERGING[dark ? "dark" : "light"];
  const clamped = Math.max(-1, Math.min(1, t));
  const end = clamped < 0 ? low : high, f = Math.abs(clamped);
  return mid.map((value, i) => Math.round(value + (end[i] - value) * f));
}

export function divergingGradient(dark) {
  const steps = [-1, -.5, 0, .5, 1].map((t, i) => `rgb(${diverging(t, dark).join(",")}) ${i * 25}%`);
  return `linear-gradient(90deg, ${steps.join(", ")})`;
}

export function cividisGradient(dark) {
  const steps = [0, .25, .5, .75, 1].map(t => `rgb(${cividis(t, dark).join(",")}) ${t * 100}%`);
  return `linear-gradient(90deg, ${steps.join(", ")})`;
}

function mark(name, start) {
  try { performance.measure(`mt:${name}`, { start, end: performance.now() }); } catch { /* sin API de medidas */ }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Lectura del cursor. Un único elemento por gráfica, con el valor primero y la etiqueta
// después, porque quien mira ya sabe qué serie sigue y busca la cifra.
class Readout {
  constructor(host, live) {
    this.node = element("div", "readout");
    this.node.setAttribute("aria-hidden", "true");
    this.node.hidden = true;
    host.append(this.node);
    this.live = live;
  }

  show(rows, left, top, width) {
    this.node.replaceChildren(...rows.map(([value, label, color]) => {
      const row = element("div", "readout-row");
      if (color) {
        const key = element("span", "readout-key");
        key.style.background = color;
        row.append(key);
      }
      row.append(element("strong", "", value), element("span", "", label));
      return row;
    }));
    this.node.hidden = false;
    const box = this.node.getBoundingClientRect();
    const x = left + 14 + box.width > width ? left - box.width - 14 : left + 14;
    this.node.style.transform = `translate(${Math.max(0, x)}px, ${Math.max(0, top - box.height / 2)}px)`;
    if (this.live) this.live.textContent = rows.map(([value, label]) => `${label}: ${value}`).join(". ");
  }

  hide() { this.node.hidden = true; }
}

function axis(t, label, values, size, space) {
  return {
    stroke: t.ink3, font: `12px ${MONO}`, labelFont: `600 12px ${SANS}`, label, labelGap: 4, labelSize: label ? 22 : 0,
    grid: { stroke: t.grid, width: 1 }, ticks: { stroke: t.rule, width: 1, size: 4 }, gap: 4, values, ...(size ? { size } : {}), ...(space ? { space } : {}),
  };
}

// Separación mínima entre marcas de un eje de pasos. Los rótulos crecen con la magnitud
// («1.000.000» ocupa unos 65 píxeles en la fuente monoespaciada de 12 px), así que una
// separación fija hace que se pisen en cuanto la serie pasa del millón de pasos.
export function stepSpace(_u, _axis, min, max) {
  const digits = String(Math.round(Math.max(Math.abs(min), Math.abs(max)))).length;
  return Math.max(50, (digits + Math.floor((digits - 1) / 3)) * 7.5 + 18);
}

// Decimales necesarios para que dos marcas consecutivas del eje no se lean iguales. Se
// deciden por el paso entre marcas y no por su magnitud, así 45,2 y 45,3 no salen como
// 45 y 45 en un eje estrecho.
export function stepDecimals(step) {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  let decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  while (decimals < 8 && Math.abs(step * 10 ** decimals - Math.round(step * 10 ** decimals)) > 1e-6 * 10 ** decimals) decimals++;
  return decimals;
}

export function formatTicks(values, { step = null, log = false } = {}) {
  const finite = values.filter(Number.isFinite);
  if (log) return values.map(value => !Number.isFinite(value) ? "" : value >= 1e7 ? significant(value, 2) : number(value, value >= 1 ? 0 : stepDecimals(value)));
  const gaps = finite.slice(1).map((value, i) => value - finite[i]).filter(gap => gap > 0);
  const increment = step ?? (gaps.length ? Math.min(...gaps) : Math.abs(finite[0] ?? 1));
  const largest = Math.max(0, ...finite.map(Math.abs));
  if (largest >= 1e7 || (largest > 0 && increment < 1e-6)) return values.map(value => Number.isFinite(value) ? significant(value, 3) : "");
  const decimals = stepDecimals(increment);
  return values.map(value => Number.isFinite(value) ? number(value, decimals) : "");
}

function tickFormat(u, values, axisIndex, _space, increment) {
  const scale = u.axes[axisIndex]?.scale ?? (axisIndex === 0 ? "x" : "y");
  return formatTicks(values, { step: increment, log: u.scales[scale]?.distr === 3 });
}

const clockFormats = new Map();
function clock(withSeconds, withDate) {
  const key = `${withSeconds}${withDate}`;
  if (!clockFormats.has(key)) {
    clockFormats.set(key, new Intl.DateTimeFormat("es-ES", {
      hour: "2-digit", minute: "2-digit", hourCycle: "h23", ...(withSeconds ? { second: "2-digit" } : {}), ...(withDate ? { day: "numeric", month: "short" } : {}),
    }));
  }
  return clockFormats.get(key);
}

// Marcas de un eje de tiempo en hora local de 24 horas. El paso decide si hacen falta
// segundos. La fecha solo acompaña a la primera marca y a las que cambian de día, para
// que las etiquetas no se monten. La zona horaria aparece en la lectura del cursor.
function timeTicks(_u, values, _axis, _space, increment) {
  let previous = null;
  return values.map(value => {
    if (value === null) return "";
    const date = new Date(value * 1000);
    const day = date.toDateString();
    const text = increment >= 86400 ? clock(false, true).format(date).split(",")[0] : clock(increment < 60, day !== previous).format(date);
    previous = day;
    return text;
  });
}

export function withAlpha(hex, alpha) {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) return hex;
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// Gráfica de líneas con muchas series alineadas en x, pensada para pequeños múltiplos.
// El cursor y el zoom horizontal se enlazan con las demás gráficas del mismo grupo.
// `summary` añade una mediana y un rango intercuartílico entre ejecuciones. Son valores
// derivados y la vista que los pide debe rotularlos así.
export class LineChart {
  constructor(host, { x, lines, xLabel, yLabel, unit, log = false, yRange = null, group = null, live, describe, onZoom, onPick, markers = [], height = 220, summary = null }) {
    this.host = host;
    this.lines = lines;
    this.unit = unit;
    this.group = group;
    this.describe = describe;
    this.onZoom = onZoom;
    this.onPick = onPick;
    this.focused = -1;
    this.readout = new Readout(host, live);
    const t = tokens();
    const started = performance.now();
    const options = {
      width: Math.max(200, host.clientWidth), height,
      padding: [10, 30, 0, 0],
      legend: { show: false },
      focus: { alpha: lines.length > 1 ? 0.18 : 1 },
      cursor: {
        sync: group ? { key: group.key, setSeries: false } : undefined,
        focus: { prox: 24 }, points: { show: false },
        drag: { x: true, y: false, setScale: true },
        bind: { dblclick: () => () => { this.resetZoom(true); return null; } },
      },
      scales: { x: { time: false }, y: { distr: log ? 3 : 1, ...(yRange ? { range: () => yRange } : {}) } },
      series: [{}, ...lines.map(line => ({ stroke: summary ? withAlpha(line.color, .5) : line.color, width: line.width ?? 1.25, points: { show: false }, spanGaps: false }))],
      // Las épocas son enteras, así que el eje x solo usa pasos enteros.
      axes: [{ ...axis(t, xLabel, tickFormat, 26), incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000] }, axis(t, yLabel, tickFormat)],
      hooks: {
        setCursor: [u => this.cursorMoved(u)],
        setSeries: [(u, index, options) => { if (options?.focus) this.focused = index - 1; }],
        setScale: [(u, key) => { if (key === "x") this.scaled(u); }],
        drawAxes: [u => this.drawBand(u, summary, t)],
        draw: [u => { this.drawMedian(u, summary, t); this.drawMarkers(u, markers, t); }],
      },
    };
    this.chart = new globalThis.uPlot(options, [x, ...lines.map(line => line.y)], host);
    this.x = x;
    this.range = [x[0], x.at(-1)];
    group?.members.add(this);
    mark("lines", started);
    host.tabIndex = 0;
    host.addEventListener("keydown", event => this.key(event));
    host.addEventListener("click", () => { if (this.focused >= 0) this.onPick?.(this.lines[this.focused]); });
    host.addEventListener("mouseleave", () => this.readout.hide());
  }

  // La banda se dibuja tras la rejilla y antes de las series, para que las ejecuciones
  // individuales sigan visibles por encima.
  drawBand(u, summary, t) {
    if (!summary) return;
    const ctx = u.ctx, x = this.x ?? u.data[0];
    ctx.save();
    ctx.fillStyle = withAlpha(t.ink.startsWith("#") ? t.ink : "#16181d", t.dark ? .16 : .1);
    let open = false;
    const close = (from, to) => {
      for (let j = to; j >= from; j--) ctx.lineTo(u.valToPos(x[j], "x", true), u.valToPos(summary.low[j], "y", true));
      ctx.closePath();
      ctx.fill();
    };
    let start = 0;
    for (let i = 0; i <= x.length; i++) {
      const ok = i < x.length && summary.low[i] !== null && summary.high[i] !== null;
      if (ok && !open) { ctx.beginPath(); ctx.moveTo(u.valToPos(x[i], "x", true), u.valToPos(summary.high[i], "y", true)); open = true; start = i; }
      else if (ok) ctx.lineTo(u.valToPos(x[i], "x", true), u.valToPos(summary.high[i], "y", true));
      else if (open) { close(start, i - 1); open = false; }
    }
    ctx.restore();
  }

  drawMedian(u, summary, t) {
    if (!summary) return;
    const ctx = u.ctx, x = this.x ?? u.data[0];
    ctx.save();
    ctx.strokeStyle = t.ink;
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let drawing = false;
    for (let i = 0; i < x.length; i++) {
      const value = summary.median[i];
      if (value === null) { drawing = false; continue; }
      const left = u.valToPos(x[i], "x", true), top = u.valToPos(value, "y", true);
      if (drawing) ctx.lineTo(left, top); else ctx.moveTo(left, top);
      drawing = true;
    }
    ctx.stroke();
    ctx.restore();
  }

  drawMarkers(u, markers, t) {
    if (!markers.length) return;
    const ctx = u.ctx;
    ctx.save();
    ctx.fillStyle = t.surface;
    ctx.lineWidth = 1.5 * devicePixelRatio;
    for (const { lineIndex, x, y } of markers) {
      if (y === null || x < u.scales.x.min || x > u.scales.x.max) continue;
      const left = u.valToPos(x, "x", true), top = u.valToPos(y, "y", true);
      ctx.strokeStyle = this.lines[lineIndex].color;
      ctx.beginPath();
      ctx.arc(left, top, 3 * devicePixelRatio, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  cursorMoved(u) {
    const index = u.cursor.idx;
    if (index === null || index === undefined || u.cursor.left < 0) { this.readout.hide(); return; }
    const line = this.lines[this.focused] ?? null;
    const rows = this.describe ? this.describe(index, line) : [];
    if (!rows.length) { this.readout.hide(); return; }
    const top = line ? u.valToPos(line.y[index] ?? 0, "y") : u.cursor.top;
    this.readout.show(rows, u.cursor.left, Number.isFinite(top) ? top : u.cursor.top, u.over.clientWidth);
  }

  scaled(u) {
    if (this.syncing) return;
    const { min, max } = u.scales.x;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return;
    const full = min <= this.range[0] && max >= this.range[1];
    this.onZoom?.(full ? null : { min, max });
    if (!this.group) return;
    for (const other of this.group.members) {
      if (other === this || !other.chart) continue;
      other.syncing = true;
      other.chart.setScale("x", { min, max });
      other.syncing = false;
    }
  }

  zoom(range) {
    if (range) this.chart.setScale("x", range);
  }

  resetZoom(propagate) {
    this.chart.setScale("x", { min: this.range[0], max: this.range[1] });
    if (!propagate) return;
    this.onZoom?.(null);
  }

  key(event) {
    const u = this.chart, length = this.x.length;
    let index = u.cursor.idx ?? -1;
    if (event.key === "ArrowRight") index = Math.min(length - 1, index + 1);
    else if (event.key === "ArrowLeft") index = Math.max(0, index < 0 ? 0 : index - 1);
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = length - 1;
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      // Recorre las ejecuciones visibles en el punto actual, de arriba abajo.
      const visible = this.lines.map((line, i) => [i, line.y[index]]).filter(([, value]) => value !== null && value !== undefined).sort((a, b) => b[1] - a[1]);
      if (!visible.length) return;
      const at = visible.findIndex(([i]) => i === this.focused);
      const next = visible[(at + (event.key === "ArrowDown" ? 1 : -1) + visible.length) % visible.length][0];
      this.focused = next;
      u.setSeries(next + 1, { focus: true });
    } else if (event.key === "Enter" && this.focused >= 0) { this.onPick?.(this.lines[this.focused]); return; }
    else if (event.key === "Escape") { this.readout.hide(); return; }
    else return;
    event.preventDefault();
    if (index < 0) index = 0;
    const left = u.valToPos(this.x[index], "x");
    u.setCursor({ left, top: u.cursor.top >= 0 ? u.cursor.top : 20 });
  }

  resize() {
    const width = Math.max(200, this.host.clientWidth);
    if (width !== this.chart.width) this.chart.setSize({ width, height: this.chart.height });
  }

  destroy() {
    this.group?.members.delete(this);
    this.chart.destroy();
    this.chart = null;
  }
}

export function chartGroup(key) {
  return { key, members: new Set() };
}

// Serie larga con reducción M4 declarada. El dibujo usa como mucho cuatro puntos por
// columna de píxel del tramo visible. Con `full` se pasan todos los puntos a uPlot.
export class LongSeriesChart {
  constructor(host, { x, y, label, unit, xLabel, color, group, live, full = false, time = false, onStats, height = 200, yRange = null }) {
    this.host = host;
    this.x = x;
    this.y = y;
    this.unit = unit;
    this.label = label;
    this.time = time;
    this.full = full;
    this.onStats = onStats;
    this.group = group;
    this.readout = new Readout(host, live);
    this.range = [x[0], x.at(-1)];
    const t = tokens();
    const started = performance.now();
    const data = this.prepare(this.range[0], this.range[1], Math.max(200, host.clientWidth));
    this.chart = new globalThis.uPlot({
      // El margen derecho deja sitio a la mitad del último rótulo, que puede ser largo.
      width: Math.max(200, host.clientWidth), height, padding: [10, time ? 30 : 40, 0, 0], legend: { show: false },
      cursor: {
        sync: group ? { key: group.key, setSeries: false } : undefined, points: { show: false },
        drag: { x: true, y: false, setScale: true },
        bind: { dblclick: () => () => { this.reset(); return null; } },
      },
      scales: { x: { time }, y: yRange ? { range: () => yRange } : {} },
      series: [{}, { stroke: color, width: 1.25, points: { show: false }, spanGaps: false }],
      axes: [time ? axis(t, xLabel, timeTicks, 26, 90) : axis(t, xLabel, tickFormat, 26, stepSpace), axis(t, unit, tickFormat)],
      hooks: {
        setScale: [(u, key) => { if (key === "x") this.scaled(u); }],
        setCursor: [u => this.cursorMoved(u)],
      },
    }, data, host);
    mark(full ? "long-full" : "long-m4", started);
    group?.members.add(this);
    host.tabIndex = 0;
    host.addEventListener("mouseleave", () => this.readout.hide());
    host.addEventListener("keydown", event => this.key(event));
  }

  prepare(min, max, width) {
    let indices;
    if (this.full) {
      const start = Math.max(0, nearestIndex(this.x, min) - 1), end = Math.min(this.x.length, nearestIndex(this.x, max) + 2);
      indices = Uint32Array.from({ length: end - start }, (_, i) => start + i);
    } else {
      indices = m4Indices(this.x, this.y, min, max, Math.max(1, Math.round(width)));
    }
    this.drawn = indices.length;
    this.onStats?.({ drawn: indices.length, total: this.x.length, full: this.full });
    return pick(this.x, [this.y], indices);
  }

  scaled(u) {
    if (this.busy) return;
    const { min, max } = u.scales.x;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return;
    this.busy = true;
    const started = performance.now();
    u.setData(this.prepare(min, max, u.over.clientWidth || this.host.clientWidth), false);
    u.setScale("x", { min, max });
    mark(this.full ? "long-full-zoom" : "long-m4-zoom", started);
    this.busy = false;
    if (this.group && !this.syncing) {
      for (const other of this.group.members) {
        if (other === this || !other.chart) continue;
        other.syncing = true;
        other.chart.setScale("x", { min, max });
        other.syncing = false;
      }
    }
  }

  reset() {
    this.chart.setScale("x", { min: this.range[0], max: this.range[1] });
  }

  setFull(full) {
    this.full = full;
    const { min, max } = this.chart.scales.x;
    this.busy = true;
    const started = performance.now();
    this.chart.setData(this.prepare(min, max, this.chart.over.clientWidth), false);
    this.chart.setScale("x", { min, max });
    mark(full ? "long-full-switch" : "long-m4-switch", started);
    this.busy = false;
  }

  // El cursor busca el punto más cercano en la serie completa, no en la reducida.
  cursorMoved(u) {
    if (u.cursor.left === null || u.cursor.left < 0) { this.readout.hide(); return; }
    const value = u.posToVal(u.cursor.left, "x");
    const index = nearestIndex(this.x, value);
    if (index < 0) return;
    const y = this.y[index];
    const xText = this.time ? timestamp(this.x[index] * 1000, { seconds: true }).text : number(this.x[index]);
    const rows = [[Number.isNaN(y) ? "Sin dato" : `${significant(y, 6)} ${this.unit}`, this.label], [xText, this.time ? "Momento de la muestra" : "Posición en el eje"]];
    const top = Number.isNaN(y) ? u.cursor.top : u.valToPos(y, "y");
    this.readout.show(rows, u.valToPos(this.x[index], "x"), top, u.over.clientWidth);
  }

  key(event) {
    const u = this.chart;
    const value = u.cursor.left >= 0 ? u.posToVal(u.cursor.left, "x") : this.x[0];
    let index = nearestIndex(this.x, value);
    const step = Math.max(1, Math.round(this.x.length / 200));
    if (event.key === "ArrowRight") index = Math.min(this.x.length - 1, index + step);
    else if (event.key === "ArrowLeft") index = Math.max(0, index - step);
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = this.x.length - 1;
    else return;
    event.preventDefault();
    u.setCursor({ left: u.valToPos(this.x[index], "x"), top: 20 });
  }

  append() {
    // La telemetría crece por la derecha. Si no hay zoom, el tramo visible sigue al final.
    const atEnd = !this.chart || this.chart.scales.x.max >= this.range[1];
    this.range = [this.x[0], this.x.at(-1)];
    if (!this.chart) return;
    this.busy = true;
    const min = atEnd ? this.range[0] : this.chart.scales.x.min, max = atEnd ? this.range[1] : this.chart.scales.x.max;
    this.chart.setData(this.prepare(min, max, this.chart.over.clientWidth), false);
    this.chart.setScale("x", { min, max });
    this.busy = false;
  }

  resize() {
    const width = Math.max(200, this.host.clientWidth);
    if (width !== this.chart.width) this.chart.setSize({ width, height: this.chart.height });
  }

  destroy() {
    this.group?.members.delete(this);
    this.chart.destroy();
    this.chart = null;
  }
}

function niceLog(min, max) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) ticks.push(10 ** e);
  return ticks;
}

function niceLinear(min, max, count = 5) {
  const span = max - min || Math.abs(max) || 1;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const unit = [1, 2, 2.5, 5, 10].map(f => f * step).find(s => span / s <= count) ?? step * 10;
  const ticks = [];
  for (let v = Math.ceil(min / unit) * unit; v <= max + unit * 1e-9; v += unit) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}

// Dispersión en canvas con hasta tres grupos de color. Para más grupos se usan pequeños
// múltiplos, porque más de tres colores juntos dejan de distinguirse con daltonismo.
//
// El fondo (ejes y puntos) se pinta una vez en un lienzo aparte. Pasar el cursor solo
// copia ese lienzo y dibuja el punto activo, así decenas de miles de puntos no se
// vuelven a trazar en cada movimiento del ratón.
const DENSE = 3000;

function extent(values, log, extra) {
  let min = Infinity, max = -Infinity;
  for (const value of values) { if (value < min) min = value; if (value > max) max = value; }
  if (extra !== null) { min = Math.min(min, extra); max = Math.max(max, extra); }
  if (log) return [10 ** Math.floor(Math.log10(min)), 10 ** Math.ceil(Math.log10(max))];
  const pad = (max - min) * .06 || Math.abs(max) * .1 || 1;
  return [min - pad, max + pad];
}

export class ScatterPlot {
  constructor(host, { groups, xLabel, yLabel, xUnit, yUnit, logX = false, logY = false, live, describe, onPick, height = 260, zero = null, categories = null }) {
    this.host = host;
    this.groups = groups;
    this.options = { xLabel, yLabel, xUnit, yUnit, logX, logY, describe, onPick, height, zero, categories };
    this.canvas = element("canvas", "scatter-canvas");
    this.layer = document.createElement("canvas");
    host.append(this.canvas);
    this.readout = new Readout(host, live);
    this.active = null;
    this.points = groups.flatMap((group, g) => group.points.map(point => ({ ...point, g })));
    this.order = [...this.points.keys()].sort((a, b) => this.points[a].x - this.points[b].x);
    host.tabIndex = 0;
    this.canvas.addEventListener("pointermove", event => this.hover(event));
    this.canvas.addEventListener("pointerleave", () => this.select(null));
    this.canvas.addEventListener("click", () => { if (this.active !== null) onPick?.(this.points[this.active]); });
    host.addEventListener("keydown", event => this.key(event));
    this.draw();
  }

  scales(width, height) {
    const { logX, logY, categories } = this.options;
    const [x0, x1] = extent(this.points.map(p => p.x), logX, null);
    const [y0, y1] = categories ? [-0.6, categories.length - 0.4] : extent(this.points.map(p => p.y), logY, this.options.zero);
    const left = categories ? 16 + Math.min(170, Math.max(...categories.map(label => label.length)) * 7.2) : 76, right = 16, top = 12, bottom = 44;
    const fx = logX ? v => (Math.log10(v) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0)) : v => (v - x0) / (x1 - x0);
    const fy = logY ? v => (Math.log10(v) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) : v => (v - y0) / (y1 - y0);
    return {
      x: v => left + fx(v) * (width - left - right), y: v => height - bottom - fy(v) * (height - top - bottom),
      xTicks: logX ? niceLog(x0, x1) : niceLinear(x0, x1), yTicks: categories ? categories.map((_, i) => i) : logY ? niceLog(y0, y1) : niceLinear(y0, y1),
      x0, x1, y0, y1, left, right, top, bottom,
    };
  }

  draw() {
    const started = performance.now();
    const t = tokens();
    this.t = t;
    const width = Math.max(240, this.host.clientWidth), height = this.options.height, ratio = devicePixelRatio || 1;
    this.size = { width, height, ratio };
    for (const canvas of [this.canvas, this.layer]) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    const ctx = this.layer.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!this.points.length) { this.paint(); return; }
    const s = this.scales(width, height);
    this.scale = s;
    this.px = Float32Array.from(this.points, point => s.x(point.x));
    this.py = Float32Array.from(this.points, point => s.y(point.y));
    ctx.font = `12px ${MONO}`;
    ctx.fillStyle = t.ink3;
    ctx.strokeStyle = t.grid;
    ctx.lineWidth = 1;
    const yTicks = s.yTicks.filter(tick => tick >= s.y0 && tick <= s.y1), xTicks = s.xTicks.filter(tick => tick >= s.x0 && tick <= s.x1);
    const yText = this.options.categories ? yTicks.map(tick => this.options.categories[tick]) : formatTicks(yTicks, { log: this.options.logY });
    const xText = formatTicks(xTicks, { log: this.options.logX });
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    yTicks.forEach((tick, i) => {
      const y = Math.round(s.y(tick)) + .5;
      ctx.beginPath(); ctx.moveTo(s.left, y); ctx.lineTo(width - s.right, y); ctx.stroke();
      ctx.fillText(yText[i], s.left - 6, y);
    });
    ctx.textBaseline = "top";
    xTicks.forEach((tick, i) => {
      const x = Math.round(s.x(tick)) + .5;
      ctx.beginPath(); ctx.moveTo(x, s.top); ctx.lineTo(x, height - s.bottom); ctx.stroke();
      // Las marcas de los extremos se alinean hacia dentro para no salirse del lienzo.
      const half = ctx.measureText(xText[i]).width / 2;
      ctx.textAlign = x + half > width - 2 ? "right" : x - half < 2 ? "left" : "center";
      ctx.fillText(xText[i], ctx.textAlign === "right" ? width - 2 : ctx.textAlign === "left" ? 2 : x, height - s.bottom + 6);
    });
    ctx.textAlign = "center";
    if (this.options.zero !== null) {
      ctx.strokeStyle = t.ink3;
      const y = Math.round(s.y(this.options.zero)) + .5;
      ctx.beginPath(); ctx.moveTo(s.left, y); ctx.lineTo(width - s.right, y); ctx.stroke();
    }
    ctx.font = `600 12px ${SANS}`;
    ctx.fillStyle = t.ink2;
    ctx.fillText(`${this.options.xLabel}${this.options.xUnit ? `, ${this.options.xUnit}` : ""}`, (s.left + width - s.right) / 2, height - 18);
    if (!this.options.categories) {
      ctx.save();
      ctx.translate(14, (s.top + height - s.bottom) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`${this.options.yLabel}${this.options.yUnit ? `, ${this.options.yUnit}` : ""}`, 0, -6);
      ctx.restore();
    }
    if (this.points.length <= DENSE) {
      // Cada punto lleva un anillo del color de la superficie que separa las marcas
      // superpuestas.
      ctx.strokeStyle = t.surface;
      ctx.lineWidth = 1.25;
      for (let i = 0; i < this.points.length; i++) {
        ctx.beginPath();
        ctx.arc(this.px[i], this.py[i], 3.5, 0, Math.PI * 2);
        ctx.fillStyle = this.groups[this.points[i].g].color;
        ctx.fill();
        ctx.stroke();
      }
    } else {
      // Con muchos puntos se traza un solo camino por grupo con transparencia, de modo
      // que la densidad se lee por acumulación de color. No se descarta ningún punto.
      ctx.globalAlpha = .5;
      this.groups.forEach((group, g) => {
        ctx.beginPath();
        for (let i = 0; i < this.points.length; i++) {
          if (this.points[i].g !== g) continue;
          ctx.moveTo(this.px[i] + 2.5, this.py[i]);
          ctx.arc(this.px[i], this.py[i], 2.5, 0, Math.PI * 2);
        }
        ctx.fillStyle = group.color;
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    }
    this.paint();
    mark("scatter", started);
  }

  paint() {
    const ctx = this.canvas.getContext("2d");
    const { width, height, ratio } = this.size;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width * ratio, height * ratio);
    ctx.drawImage(this.layer, 0, 0);
    if (this.active === null) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.beginPath();
    ctx.arc(this.px[this.active], this.py[this.active], 6, 0, Math.PI * 2);
    ctx.fillStyle = this.groups[this.points[this.active].g].color;
    ctx.strokeStyle = this.t.ink;
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  }

  hover(event) {
    if (!this.scale) return;
    const box = this.canvas.getBoundingClientRect();
    const px = event.clientX - box.left, py = event.clientY - box.top;
    let best = null, distance = 24 ** 2;
    for (let i = 0; i < this.points.length; i++) {
      const d = (this.px[i] - px) ** 2 + (this.py[i] - py) ** 2;
      if (d < distance) { distance = d; best = i; }
    }
    this.select(best);
  }

  select(index) {
    if (index === this.active) return;
    this.active = index;
    this.paint();
    if (index === null) { this.readout.hide(); return; }
    const point = this.points[index];
    this.readout.show(this.options.describe(point), this.px[index], this.py[index], this.canvas.clientWidth);
  }

  key(event) {
    const at = this.active === null ? -1 : this.order.indexOf(this.active);
    let next = at;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = Math.min(this.order.length - 1, at + 1);
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = Math.max(0, at - 1);
    else if (event.key === "Enter" && this.active !== null) { this.options.onPick?.(this.points[this.active]); return; }
    else if (event.key === "Escape") { this.select(null); return; }
    else return;
    event.preventDefault();
    this.select(this.order[next]);
  }

  resize() { if (Math.max(240, this.host.clientWidth) !== this.size?.width) this.draw(); }
  destroy() { this.canvas.remove(); this.readout.node.remove(); }
}

// Mapa de calor de filas por posición, por ejemplo capas de Titans por paso. Si hay más
// columnas que píxeles se dibuja la media de cada columna de píxel y se declara. La
// lectura del cursor da siempre el valor exacto de la posición más cercana.
export class Heatmap {
  constructor(host, { x, rows, values, range, unit, label, xLabel, live, height = null, onStats }) {
    this.host = host;
    Object.assign(this, { x, rows, values, unit, label, xLabel, onStats });
    let low = Infinity, high = -Infinity;
    for (const value of values) if (!Number.isNaN(value)) { low = Math.min(low, value); high = Math.max(high, value); }
    this.range = range ?? [low, high];
    this.height = height ?? Math.max(120, Math.min(360, rows.length * 22 + 40));
    this.canvas = element("canvas", "heatmap-canvas");
    host.append(this.canvas);
    this.readout = new Readout(host, live);
    host.tabIndex = 0;
    this.canvas.addEventListener("pointermove", event => this.hover(event));
    this.canvas.addEventListener("pointerleave", () => this.readout.hide());
    this.draw();
  }

  draw() {
    const started = performance.now();
    const t = tokens();
    const width = Math.max(240, this.host.clientWidth), height = this.height, ratio = devicePixelRatio || 1;
    const left = 72, right = 8, top = 6, bottom = 30;
    this.box = { left, right, top, bottom, width, height };
    this.canvas.width = width * ratio;
    this.canvas.height = height * ratio;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const plotWidth = Math.floor(width - left - right), plotHeight = height - top - bottom;
    const columns = this.x.length, rows = this.rows.length;
    const pixels = Math.min(columns, plotWidth);
    const image = ctx.createImageData(pixels, rows);
    const [low, high] = this.range;
    for (let r = 0; r < rows; r++) {
      for (let p = 0; p < pixels; p++) {
        const start = Math.floor(p * columns / pixels), end = Math.max(start + 1, Math.floor((p + 1) * columns / pixels));
        let sum = 0, n = 0;
        for (let c = start; c < end; c++) {
          const value = this.values[r * columns + c];
          if (!Number.isNaN(value)) { sum += value; n++; }
        }
        const offset = (r * pixels + p) * 4;
        if (!n) { image.data[offset + 3] = 0; continue; }
        const [cr, cg, cb] = cividis((sum / n - low) / (high - low || 1), t.dark);
        image.data[offset] = cr; image.data[offset + 1] = cg; image.data[offset + 2] = cb; image.data[offset + 3] = 255;
      }
    }
    const buffer = document.createElement("canvas");
    buffer.width = pixels;
    buffer.height = rows;
    buffer.getContext("2d").putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = t.grid;
    ctx.fillRect(left, top, plotWidth, plotHeight);
    ctx.drawImage(buffer, left, top, plotWidth, plotHeight);
    ctx.font = `12px ${MONO}`;
    ctx.fillStyle = t.ink3;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const rowHeight = plotHeight / rows;
    const every = Math.ceil(14 / rowHeight);
    this.rows.forEach((row, r) => { if (r % every === 0) ctx.fillText(row, left - 6, top + (r + .5) * rowHeight); });
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const ticks = niceLinear(this.x[0], this.x.at(-1), 6).filter(tick => tick >= this.x[0] && tick <= this.x.at(-1));
    formatTicks(ticks).forEach((label, i) => {
      const px = left + (ticks[i] - this.x[0]) / (this.x.at(-1) - this.x[0] || 1) * plotWidth;
      const half = ctx.measureText(label).width / 2;
      ctx.textAlign = px + half > width - 2 ? "right" : "center";
      ctx.fillText(label, ctx.textAlign === "right" ? width - 2 : px, height - bottom + 6);
    });
    this.onStats?.({ drawn: pixels * rows, total: columns * rows, reduced: pixels < columns });
    mark("heatmap", started);
  }

  hover(event) {
    const box = this.canvas.getBoundingClientRect(), b = this.box;
    const px = event.clientX - box.left, py = event.clientY - box.top;
    const plotWidth = b.width - b.left - b.right, plotHeight = b.height - b.top - b.bottom;
    if (px < b.left || px > b.left + plotWidth || py < b.top || py > b.top + plotHeight) { this.readout.hide(); return; }
    const xValue = this.x[0] + (px - b.left) / plotWidth * (this.x.at(-1) - this.x[0]);
    const column = nearestIndex(this.x, xValue), row = Math.min(this.rows.length - 1, Math.floor((py - b.top) / plotHeight * this.rows.length));
    const value = this.values[row * this.x.length + column];
    this.readout.show([[Number.isNaN(value) ? "Sin dato" : `${significant(value, 5)} ${this.unit}`, this.label], [this.rows[row], "Fila"], [number(this.x[column]), this.xLabel]], px, py, b.width);
  }

  resize() { if (Math.max(240, this.host.clientWidth) !== this.box?.width) this.draw(); }
  destroy() { this.canvas.remove(); this.readout.node.remove(); }
}
