// Piezas de interfaz compartidas por las vistas. Todo el texto procedente de los datos se
// inserta con textContent, nunca como HTML, porque los identificadores vienen de recibos.

import { STATUS_LABELS } from "./state.mjs";
import { SEEDS } from "./model.mjs";

export const STATE_LABELS = Object.freeze({
  ...STATUS_LABELS, done: "Confirmado", attempt: "Intento sin confirmar", pending: "Pendiente",
});

export function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  const { className, text, attrs, dataset, on } = options;
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  for (const [name, value] of Object.entries(attrs ?? {})) if (value !== undefined && value !== null && value !== false) node.setAttribute(name, value === true ? "" : String(value));
  Object.assign(node.dataset, dataset ?? {});
  for (const [name, handler] of Object.entries(on ?? {})) node.addEventListener(name, handler);
  node.append(...children.filter(child => child !== null && child !== undefined && child !== false));
  return node;
}

export function stateMark(state, label) {
  return el("i", { className: "m", dataset: { s: state }, attrs: { "aria-hidden": "true", title: label } });
}

export function stateBadge(state) {
  return el("span", { className: "state" }, stateMark(state), el("span", { text: STATE_LABELS[state] ?? state }));
}

export function seedColor(seed) {
  const index = SEEDS.indexOf(seed);
  return index >= 0 ? `var(--series-${index + 1})` : "var(--context)";
}

// Los colores de canvas no entienden var(), así que se resuelven con los tokens activos.
export function resolveColor(token, tokens) {
  const match = /var\(--series-(\d)\)/.exec(token);
  if (match) return tokens.series[Number(match[1]) - 1];
  return token === "var(--context)" ? tokens.context : token;
}

export function setOptions(select, options, value) {
  const current = [...select.options].map(option => `${option.value}\u0000${option.textContent}`).join("\u0001");
  const next = options.map(([v, label]) => `${v}\u0000${label}`).join("\u0001");
  if (current !== next) select.replaceChildren(...options.map(([v, label]) => el("option", { text: label, attrs: { value: v } })));
  if (options.some(([v]) => v === value)) select.value = value;
}

export function legend(container, items) {
  container.replaceChildren(...items.map(item => el("li", {},
    item.state ? stateMark(item.state) : el("span", { className: item.band ? "swatch-band" : "swatch-line", attrs: { "aria-hidden": "true" } }),
    el("span", { text: item.label }),
  )));
  for (const [index, item] of items.entries()) {
    if (item.color) container.children[index].firstChild.style.background = item.color;
  }
}

export function facts(rows) {
  return el("dl", { className: "facts" }, ...rows.map(([label, value, options = {}]) => {
    const missing = value === null || value === undefined || value === "Sin dato" || value === "Sin fecha";
    // Cifras, fechas y huellas van en monoespaciada para alinearse. El texto corriente
    // se queda en la tipografía de lectura.
    const numeric = !missing && /\d/.test(String(value));
    const dd = el("dd", { className: missing ? "missing" : numeric ? "mono" : "", text: missing ? "Sin dato" : value });
    if (options.title) dd.title = options.title;
    return el("div", {}, el("dt", { text: label }), dd);
  }));
}

// Cada vista registra sus gráficas para poder destruirlas antes de volver a dibujar y
// para que el observador de tamaño solo ajuste las de la vista visible.
export function replaceCharts(ctx, view, charts) {
  ctx.chartsByView ??= new Map();
  for (const chart of ctx.chartsByView.get(view) ?? []) chart.destroy?.();
  ctx.chartsByView.set(view, charts);
}
