"""Abstención publicada con su cobertura en cada comparación y en la matriz (#442).

Las reglas están en `evaluation.abstention_coverage`. Esta comprobación las repite desde la
tabla de sesiones publicada, sin el código de la comparación:

1. el informe publica las reglas declaradas y la cobertura de llamadas de cada modelo en
   cada vista (todas las sesiones y cada mercado), que se recalcula sumando llamadas y filas
   con objetivo no nulo de todas sus semillas;
2. cada contraste de una métrica condicionada a las llamadas lleva la marca de elegibilidad
   que da la cobertura mínima, en la comparación y en cada vista de la matriz;
3. la curva riesgo-cobertura de cada semilla y vista publica las filas retenidas y los
   aciertos que suman sus sesiones, con ceil(c · n) filas de las n con objetivo no nulo.

Solo lee informes y tablas publicados. No ajusta nada ni abre la reserva de 2024.
"""

from fractions import Fraction

import numpy as np

from mars_titan.evaluation import abstention_coverage as declared

KIND = "abstention_marks"


def _views(markets):
    views = ["+".join(markets)]
    return views + list(markets) if len(markets) > 1 else views


def _selected(raw, view, markets):
    return raw if view == "+".join(markets) else raw[raw["market"] == view]


def coverage(raw, markets):
    """Cobertura de llamadas de cada modelo en cada vista, desde las sesiones en bruto."""
    found = {}
    for view in _views(markets):
        part = _selected(raw, view, markets)
        sums = part.groupby("arm")[["direction_calls", "direction_eligible"]].sum()
        found[view] = {
            arm: None
            if int(row.direction_eligible) == 0
            else int(row.direction_calls) / int(row.direction_eligible)
            for arm, row in sums.iterrows()
        }
    return found


def _allowed(value, minimum):
    return value is not None and Fraction(value) >= Fraction(minimum)


def _check_rows(entries, coverage, minimum, label, failures):
    """Marca de cada contraste de las métricas condicionadas de una vista."""
    checked = 0
    for metric in declared.CALL_CONDITIONED:
        for family, entry in (entries.get(metric) or {}).items():
            for row in entry.get("contrasts", []):
                arms = sorted(arm for arm, value in row["coefficients"].items() if value != 0)
                missing = [arm for arm in arms if arm not in coverage]
                if missing:
                    failures.append(f"{label} {metric} {family}: sin cobertura de {missing}")
                    continue
                expected = all(_allowed(coverage[arm], minimum) for arm in arms)
                checked += 1
                if row.get("eligible") is not expected:
                    failures.append(
                        f"{label} {metric} {family} {row['name']}: marcado "
                        f"{row.get('eligible')} y su cobertura da {expected}"
                    )
                elif row.get("call_coverage") != {arm: coverage[arm] for arm in arms}:
                    failures.append(f"{label} {metric} {family} {row['name']}: otra cobertura")
    return checked


def _retained(eligible, coverage):
    fraction = Fraction(coverage).limit_denominator(1_000_000)
    return [
        0 if n == 0 else -(-(fraction.numerator * n) // fraction.denominator)
        for n in (int(value) for value in eligible)
    ]


def _check_curves(report, raw, markets, failures):
    """Filas retenidas y aciertos de la curva de cada semilla frente a sus sesiones."""
    checked = 0
    for arm, record in report["arms"].items():
        for seed, entry in record["seeds"].items():
            pooled = raw[raw["arm"] == arm]
            if seed != "deterministic":
                pooled = pooled[pooled["seed"] == int(seed)]
            for view, overall in entry["overall"].items():
                part = _selected(pooled, view, markets)
                selective = overall["summary"].get("selective_direction")
                if selective is None or "selective_hits" not in part:
                    failures.append(f"{arm} semilla {seed} {view}: sin curva riesgo-cobertura")
                    continue
                hits = np.vstack(
                    [np.asarray(item, dtype=np.int64) for item in part["selective_hits"]]
                )
                curve = selective["curve"]
                if [point["coverage"] for point in curve] != declared.declaration()["coverages"]:
                    failures.append(f"{arm} semilla {seed} {view}: otras coberturas")
                    continue
                for index, point in enumerate(curve):
                    retained = sum(_retained(part["direction_eligible"], point["coverage"]))
                    if point["retained_rows"] != retained or point["hits"] != int(
                        hits[:, index].sum()
                    ):
                        failures.append(
                            f"{arm} semilla {seed} {view} cobertura {point['coverage']:g}: "
                            "filas o aciertos distintos de sus sesiones"
                        )
                checked += 1
    return checked


def check_report(report, raw):
    """Reglas, coberturas, marcas y curvas de una comparación walk-forward publicada."""
    markets, failures = list(report["markets"]), []
    rules = declared.declaration()
    published = report.get("abstention") or {}
    if any(published.get(key) != value for key, value in rules.items()):
        failures.append("El informe no publica las reglas de abstención declaradas")
    found = coverage(raw, markets)
    if published.get("call_coverage") != found:
        failures.append("La cobertura de llamadas publicada no es la de sus sesiones")
    rows = sum(
        _check_rows(
            families_by_metric(report["contrasts"][view]),
            found[view],
            rules["min_call_coverage"],
            view,
            failures,
        )
        for view in found
    )
    curves = _check_curves(report, raw, markets, failures)
    return dict(
        kind=KIND,
        coverage=found,
        contrasts=rows,
        curves=curves,
        failures=failures,
        passed=not failures,
    )


def families_by_metric(view):
    """La comparación agrupa por familia y métrica, y la matriz por métrica y familia."""
    grouped = {}
    for family, metrics in view.items():
        for metric, entry in metrics.items():
            grouped.setdefault(metric, {})[family] = entry
    return grouped


def check_matrix(matrix, found):
    """Marcas de las vistas de predicción de la matriz con la cobertura recalculada."""
    failures, rows = [], 0
    minimum = declared.MIN_CALL_COVERAGE
    if matrix.get("abstention") != declared.declaration():
        failures.append("La matriz no publica las reglas de abstención declaradas")
    for name, view in matrix["views"].items():
        for label, metrics in view.items():
            if not isinstance(metrics, dict) or label not in found:
                continue
            rows += _check_rows(metrics, found[label], minimum, f"{name} {label}", failures)
    return dict(contrasts=rows, failures=failures, passed=not failures)
