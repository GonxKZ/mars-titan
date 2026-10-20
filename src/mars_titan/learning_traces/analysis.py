"""Análisis reproducible de las trazas de aprendizaje (#448).

Lee carpetas del registrador con `read_traces`, que valida versión, esquema, numeración,
nombres y presupuesto, y resume cada ejecución con reglas fijas declaradas aquí:

- Validación: puntuación por paso, mejor paso y última puntuación.
- Optimización: pérdida al principio y al final, fracción de pasos registrados en los que
  la norma total del gradiente supera el umbral de recorte, y por grupo de parámetros la
  mediana de la norma del gradiente y de la razón de actualización. Un grupo cuya razón
  de actualización es siempre cero no se ha movido en ningún paso registrado.
- Memoria de Titans: por memoria, la media de α, η y θ en el último paso registrado, las
  fracciones medias de puertas saturadas y la pérdida asociativa al principio y al final.
- Adaptadores: en la última validación, la norma de ΔW y su norma relativa por tensor.
- Corrección B6 y políticas: el último valor de cada métrica propia. Las políticas RL no
  usan `validation.score`, porque su criterio (crecimiento logarítmico) se maximiza.
- Integridad: valores no finitos y estadísticas con `finite_fraction` menor que uno.

`align` reúne una misma métrica de varias ejecuciones sobre la unión de sus pasos, sin
interpolar, para comparar modelos con el mismo eje. Las reglas solo describen lo que se ha
registrado. No seleccionan modelos ni sustituyen a la evaluación, y una traza recortada
por presupuesto se declara como tal.
"""

import math
from pathlib import Path

from .recorder import read_traces

SCHEMA_VERSION = 1
KIND = "learning_traces_analysis"
GATES = ("alpha", "eta", "theta")
# Nombre de la unidad del paso en el informe legible.
UNITS = {
    "optimizer_step": "pasos del optimizador",
    "write_event": "escrituras de la memoria asociativa",
    "boosting_round": "rondas de boosting",
    "fit": "ajustes",
    "transition": "transiciones",
    "wave": "oleadas",
}


def _median(values):
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _finite(value):
    return value if value is not None and math.isfinite(value) else None


class Run:
    """Trazas de una ejecución en series ordenadas por paso.

    Si un paso aparece dos veces en una serie se conserva la última fila, la de la parte
    escrita después, igual que en el observatorio.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest, table = read_traces(self.folder)
        columns = table.to_pydict()
        self.rows = len(columns["value"])
        collected = {}
        for step, phase, metric, group, stat, value in zip(
            columns["step"],
            columns["phase"],
            columns["metric"],
            columns["group"],
            columns["stat"],
            columns["value"],
            strict=True,
        ):
            collected.setdefault((phase, metric, group, stat), {})[step] = value
        self.series = {key: sorted(points.items()) for key, points in sorted(collected.items())}
        self.parts = sorted((self.folder / "parts").glob("part-*.parquet"))

    @property
    def family(self):
        return self.manifest["identity"].get("family", "unknown")

    def find(self, metric, *, phase=None, stat="value"):
        """Series de `metric` por grupo, con su fase, filtradas por fase si se indica."""
        return {
            (key_phase, group): points
            for (key_phase, key_metric, group, key_stat), points in self.series.items()
            if key_metric == metric and key_stat == stat and (phase is None or key_phase == phase)
        }

    def metrics(self):
        return sorted({metric for _, metric, _, _ in self.series})


def _validation(run):
    scores = run.find("validation.score", phase="validation")
    if not scores:
        return None
    points = [point for series in scores.values() for point in series]
    finite = [(step, value) for step, value in points if math.isfinite(value)]
    best = min(finite, key=lambda item: (item[1], item[0])) if finite else None
    return dict(
        points=[[step, _finite(value)] for step, value in points],
        best_step=None if best is None else best[0],
        best_score=None if best is None else best[1],
        last_score=_finite(points[-1][1]),
    )


def _optimization(run):
    losses = [point for series in run.find("loss").values() for point in series]
    norms = {step: value for series in run.find("grad_l2_total").values() for step, value in series}
    thresholds = {
        step: value for series in run.find("clip_threshold").values() for step, value in series
    }
    clipped = [norms[step] > thresholds[step] for step in norms if step in thresholds]
    groups = {}
    for name, label in (("grad_l2", "grad_l2_median"), ("update_ratio", "update_ratio_median")):
        for (_, group), points in run.find(name, phase="train").items():
            groups.setdefault(group, {})[label] = _median([value for _, value in points])
    for (_, group), points in run.find("update_ratio", phase="train").items():
        groups[group]["moved"] = any(value > 0 for _, value in points)
    for (_, group), points in run.find("weight_l2", phase="validation").items():
        first, last = points[0][1], points[-1][1]
        groups.setdefault(group, {})["weight_l2_change"] = (
            (last - first) / first if first and math.isfinite(first) else None
        )
    if not (losses or norms or groups):
        return None
    return dict(
        traced_steps=len({step for step, _ in losses} | set(norms)),
        loss_first=_finite(losses[0][1]) if losses else None,
        loss_last=_finite(losses[-1][1]) if losses else None,
        clip_fraction=sum(clipped) / len(clipped) if clipped else None,
        groups=dict(sorted(groups.items())),
    )


def _titans(run):
    memories = {}
    for (_, metric, group, stat), points in run.series.items():
        if not metric.startswith("memory.") or stat != "value":
            continue
        name = metric.removeprefix("memory.")
        prefix = ""
        if name.startswith("shared."):
            prefix, name = "shared_", name.removeprefix("shared.")
        entry = memories.setdefault(group, {})
        values = [value for _, value in points]
        if name in (f"{gate}.mean" for gate in GATES):
            entry[f"{prefix}{name.split('.')[0]}_last_mean"] = _finite(values[-1])
        elif name.endswith((".low_fraction", ".high_fraction")):
            entry[f"{prefix}{name.replace('.', '_')}_mean"] = _median(values)
        elif name == "associative_loss.mean":
            entry[f"{prefix}associative_loss_first"] = _finite(values[0])
            entry[f"{prefix}associative_loss_last"] = _finite(values[-1])
        elif name == "tokens":
            entry[f"{prefix}traced_tokens"] = sum(values)
        elif prefix and name in ("writes", "rows"):
            entry[f"{prefix}traced_{name}"] = _finite(sum(values))
        elif name.startswith("plasticity.") and name.endswith(".mean"):
            label = name.removesuffix(".mean").replace(".", "_")
            entry[f"{prefix}{label}_last_mean"] = _finite(values[-1])
        elif prefix and name == "rows_per_write.mean":
            entry["shared_rows_per_write_last_mean"] = _finite(values[-1])
    return dict(sorted(memories.items())) or None


def _adapters(run):
    latest = {}
    for metric in ("adapter.delta_l2", "adapter.delta_relative", "adapter.module_l2"):
        for (_, group), points in run.find(metric, phase="validation").items():
            latest.setdefault(group, {})[metric.removeprefix("adapter.")] = _finite(points[-1][1])
    return dict(sorted(latest.items())) or None


def _own(run, prefixes):
    values = {}
    for (phase, metric, group, stat), points in run.series.items():
        if metric.startswith(prefixes) and stat == "value":
            values[f"{phase}/{metric}/{group}"] = _finite(points[-1][1])
    return values or None


def _integrity(run):
    nonfinite = sum(
        not math.isfinite(value) for points in run.series.values() for _, value in points
    )
    partial = sum(
        value < 1
        for (_, _, _, stat), points in run.series.items()
        if stat == "finite_fraction"
        for _, value in points
    )
    return dict(nonfinite_values=nonfinite, partially_finite_statistics=partial)


def diagnose(run):
    """Resumen de una ejecución con las reglas del módulo."""
    config = run.manifest["config"]
    size = sum(path.stat().st_size for path in run.parts)
    # Pasos distintos con alguna fila, en todas las fases y en la unidad del manifiesto.
    steps = sorted({step for points in run.series.values() for step, _ in points})
    return dict(
        folder=str(run.folder),
        family=run.family,
        run_identity_sha256=run.manifest["identity"].get("run_identity_sha256"),
        every=config["every"],
        step_unit=run.manifest["step_unit"],
        rows=run.rows,
        parts=len(run.parts),
        bytes=size,
        budget_fraction=size / config["max_bytes"],
        exhausted_at_step=run.manifest["exhausted_at_step"],
        phases=sorted({phase for phase, _, _, _ in run.series}),
        steps=[steps[0], steps[-1]] if steps else None,
        recorded_steps=len(steps),
        metrics=run.metrics(),
        validation=_validation(run),
        optimization=_optimization(run),
        titans=_titans(run),
        adapters=_adapters(run),
        b6=_own(run, ("associative.",)),
        rl=_own(run, ("rl.", "policy.", "critic.")),
        integrity=_integrity(run),
    )


def align(runs, metric, *, phase="validation", group="all", stat="value"):
    """Una métrica de varias ejecuciones sobre la unión de sus pasos, sin interpolar."""
    curves = {}
    for label, run in runs.items():
        points = dict(run.series.get((phase, metric, group, stat), []))
        if points:
            curves[label] = points
    steps = sorted({step for points in curves.values() for step in points})
    return dict(
        metric=metric,
        phase=phase,
        group=group,
        stat=stat,
        steps=steps,
        runs={
            label: [_finite(points.get(step, math.nan)) for step in steps]
            for label, points in sorted(curves.items())
        },
    )


def discover(root, *, limit=4096):
    """Carpetas de trazas bajo `root`, en orden, con un límite explícito."""
    folders = sorted(path.parent for path in Path(root).rglob("traces/manifest.json"))
    if len(folders) > limit:
        raise ValueError(f"Hay {len(folders)} carpetas de trazas y el límite es {limit}")
    return folders


def analyze(folders, *, root=None):
    """Informe de varias carpetas, con sus curvas de validación alineadas por familia."""
    runs = {}
    for folder in folders:
        folder = Path(folder)
        label = str(folder.relative_to(root)) if root is not None else str(folder)
        runs[label] = Run(folder)
    by_family = {}
    for label, run in runs.items():
        by_family.setdefault(run.family, {})[label] = run
    return dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        runs={label: diagnose(run) for label, run in sorted(runs.items())},
        validation_by_family={
            family: align(members, "validation.score")
            for family, members in sorted(by_family.items())
        },
    )


def _number(value, digits=4):
    if value is None:
        return "sin dato"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def markdown(report, *, title="Análisis de trazas de aprendizaje", note=None):
    """Informe legible con una sección por ejecución."""
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines += [
        "| Ejecución | Familia | Unidad | Pasos con filas | Filas | Bytes | Presupuesto usado "
        "| Agotado |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, run in report["runs"].items():
        exhausted = run["exhausted_at_step"]
        exhausted = "no" if exhausted is None else f"desde el paso {exhausted}"
        lines.append(
            f"| `{label}` | {run['family']} | {UNITS[run['step_unit']]} | {run['recorded_steps']} "
            f"| {run['rows']} | {run['bytes']} | {run['budget_fraction']:.2%} | {exhausted} |"
        )
    for label, run in report["runs"].items():
        lines += ["", f"## `{label}`", ""]
        unit = UNITS[run["step_unit"]]
        lines.append(f"Fases: {', '.join(run['phases'])}. Cadencia: cada {run['every']} {unit}.")
        validation = run["validation"]
        if validation:
            lines.append(
                f"Validación: mejor {_number(validation['best_score'])} en el paso "
                f"{_number(validation['best_step'])}, última {_number(validation['last_score'])}."
            )
        optimization = run["optimization"]
        if optimization:
            lines.append(
                f"Optimización: pérdida {_number(optimization['loss_first'])} → "
                f"{_number(optimization['loss_last'])}, fracción recortada "
                f"{_number(optimization['clip_fraction'])}."
            )
            if optimization["groups"]:
                lines += [
                    "",
                    "| Grupo | Mediana de la norma del gradiente | Mediana de la razón de "
                    "actualización | Se mueve |",
                    "| --- | ---: | ---: | --- |",
                ]
                for group, values in optimization["groups"].items():
                    lines.append(
                        f"| `{group}` | {_number(values.get('grad_l2_median'))} | "
                        f"{_number(values.get('update_ratio_median'))} | "
                        f"{_number(values.get('moved'))} |"
                    )
        for section, heading in (("titans", "Memoria de Titans"), ("adapters", "Adaptadores")):
            if run[section]:
                lines += ["", f"{heading}:", ""]
                for group, values in run[section].items():
                    detail = ", ".join(f"{k} {_number(v)}" for k, v in sorted(values.items()))
                    lines.append(f"- `{group}`: {detail}.")
        for section, heading in (("b6", "Corrección B6"), ("rl", "Políticas")):
            if run[section]:
                detail = ", ".join(f"`{k}` {_number(v)}" for k, v in run[section].items())
                lines += ["", f"{heading}: {detail}."]
        integrity = run["integrity"]
        lines += [
            "",
            f"Integridad: {integrity['nonfinite_values']} valores no finitos y "
            f"{integrity['partially_finite_statistics']} estadísticas con valores no finitos "
            "en su tensor.",
        ]
    return "\n".join(lines) + "\n"
