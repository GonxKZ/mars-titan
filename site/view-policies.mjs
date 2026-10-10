// Validación financiera de las políticas: retorno neto frente a caída máxima, en
// pequeños múltiplos por familia para no mezclar más de dos colores en una gráfica.

import { FINANCIAL_REASON_LABELS, formatValue } from "./state.mjs";
import * as fmt from "./format.mjs";
import { financialPoints } from "./model.mjs";
import { ScatterPlot, tokens } from "./charts.mjs";
import { el, legend, replaceCharts } from "./ui.mjs";

const byId = id => document.getElementById(id);

function percentText(value) {
  return formatValue(value, { digits: 2, style: "percent" });
}

function table(ctx, groups) {
  const columns = ["Familia", "Modelo", "Ejecución", "Semilla", "Retorno neto", "Caída máxima", "Costes", "Rotación", "Pasos", "Episodio"];
  const rows = groups.flatMap(group => group.points.map(point => el("tr", {},
    el("td", { text: group.label }),
    el("td", { text: ctx.modelName(point.run.model_id) }),
    el("td", {}, el("button", { className: "link mono", text: point.run.run_id, attrs: { type: "button" }, on: { click: () => ctx.openRun(point.run) } })),
    el("td", { className: "num", text: point.run.seed ?? "Sin dato" }),
    el("td", { className: "num", text: percentText(point.net_return) }),
    el("td", { className: "num", text: percentText(point.max_drawdown) }),
    el("td", { className: "num", text: formatValue(point.costs, { digits: 4 }) }),
    el("td", { className: "num", text: formatValue(point.turnover, { digits: 2 }) }),
    el("td", { className: "num", text: point.steps === null ? "Sin dato" : fmt.number(point.steps) }),
    el("td", { text: point.completed === false ? FINANCIAL_REASON_LABELS[point.invalid_reason] ?? "Incompleto" : point.completed ? "Completo" : "Sin dato" }),
  )));
  byId("rl-table").replaceChildren(
    el("caption", { className: "visually-hidden", text: "Validación financiera por ejecución" }),
    el("thead", {}, el("tr", {}, ...columns.map((label, i) => el("th", { text: label, className: i >= 3 && i <= 8 ? "num" : "", attrs: { scope: "col" } })))),
    el("tbody", {}, ...rows));
}

export function renderPolicies(ctx) {
  const t = tokens();
  const groups = financialPoints(ctx.runs());
  const charts = [];
  const nodes = groups.map(group => {
    const complete = group.points.filter(point => point.completed !== false);
    const incomplete = group.points.filter(point => point.completed === false);
    const plot = el("div", { className: "plot", attrs: { role: "group", "aria-label": `${group.label}. Retorno neto frente a caída máxima. Usa las flechas para recorrer los episodios e Intro para abrir el detalle.` } });
    const keys = el("ul", { className: "legend" });
    const node = el("section", { className: "facet" },
      el("div", { className: "facet-head" }, el("h2", { text: group.label }), el("p", { text: `${fmt.number(group.points.length)} episodios` })),
      keys, plot);
    legend(keys, [{ label: `Completos (${fmt.number(complete.length)})`, color: "var(--series-1)" }, { label: `Incompletos o inválidos (${fmt.number(incomplete.length)})`, color: "var(--context)" }]);
    if (!group.points.length) {
      plot.append(el("p", { className: "caption", text: "Sin episodios de validación con retorno y caída publicados." }));
      return node;
    }
    const colors = [t.series[0], t.context];
    requestAnimationFrame(() => {
      if (!plot.isConnected) return;
      charts.push(new ScatterPlot(plot, {
        groups: [
          { color: colors[0], points: complete.map(point => ({ ...point, x: point.max_drawdown, y: point.net_return })) },
          { color: colors[1], points: incomplete.map(point => ({ ...point, x: point.max_drawdown, y: point.net_return })) },
        ],
        xLabel: "Caída máxima", xUnit: "fracción", yLabel: "Retorno neto", yUnit: "fracción", zero: 0, live: ctx.live, height: 240,
        describe: point => [
          [percentText(point.net_return), "Retorno neto tras costes", colors[point.g]],
          [percentText(point.max_drawdown), "Caída máxima"],
          [`${formatValue(point.costs, { digits: 4 })} · ${formatValue(point.turnover, { digits: 2 })}`, "Costes y rotación"],
          [ctx.modelName(point.run.model_id), `${point.run.run_id.slice(-24)} · semilla ${point.run.seed ?? "sin dato"}`],
        ],
        onPick: point => ctx.openRun(point.run),
      }));
    });
    return node;
  });
  byId("rl-facets").replaceChildren(...nodes);
  replaceCharts(ctx, "rl", charts);
  table(ctx, groups);
}
