"""Registro previo de cada cifra que publica la comparación walk-forward (#442).

`configs/evaluation/metric-registry.json` fija antes de evaluar la definición, la unidad, el
sentido, la regla de agregación entre sesiones y la sección de `docs/research/metrics.md` de
cada cifra del resumen de una vista (`forecast_scores.SessionScores.summary`). Los huecos
siguen reglas comunes: una sesión sin denominador queda fuera y se cuenta, una vista sin
sesiones definidas publica una cifra nula y una fila que falta bloquea la publicación.

El registro se comprueba de dos formas:

- `load_registry` exige que sus convenciones, coberturas, estados y series sean las del
  código, que cada unidad y regla sea conocida y que cada sección exista en la documentación;
- `check_summary` reconstruye el resumen de una vista desde las sesiones en bruto publicadas,
  aplicando solo las reglas del registro y la media de `independent_scores`, sin el código de
  la comparación. Una cifra publicada sin registrar, una registrada que no se publica o un
  valor distinto hacen fallar la comprobación. Las cifras que la tabla de sesiones no permite
  recalcular se registran con su motivo y solo se exige que existan.

`score_recheck.check_summaries` lo aplica a cada modelo, semilla y vista al publicar. Nada de
este módulo ajusta modelos ni abre la reserva de 2024.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from mars_titan.data.storage import sha256
from mars_titan.evaluation import forecast_scores
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import WEIGHTINGS

from .independent_scores import RANK_IC_STATUS, session_mean

KIND = "metric_registry"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT = ROOT / "configs" / "evaluation" / "metric-registry.json"
BETTER = ("lower", "higher", "closer_to_zero", "none")
# Parámetros de cada regla. Las de `of` derivan de otra cifra registrada de la misma vista.
RULES = {
    "descriptor": set(),
    "weighting": set(),
    "key": set(),
    "family_keys": {"of"},
    "central_level": {"side"},
    "bin_edge": {"side"},
    "not_recomputed": {"reason"},
    "count": set(),
    "sum": {"column"},
    "session_mean": {"column"},
    "session_ratio": {"numerator", "denominator"},
    "row_weighted_mean": {"column"},
    "pooled_ratio": {"numerator", "denominator"},
    "session_std": {"column", "status"},
    "positive_fraction": {"column", "status"},
    "sqrt": {"of"},
    "percent": {"of"},
    "minus_key": {"of"},
    "mean_over": {"of"},
    "mean_abs_over": {"of"},
    "max_abs_over": {"of"},
    "undercovered": {"of"},
    "selective_rows": set(),
    "selective_hits": set(),
    "selective_ratio": set(),
    "bin_total": set(),
    "bin_rows": set(),
    "bin_mean": {"column"},
    "ece": set(),
}
_OPTIONAL = {"value", "status", "zero", "positive", "minus"}
# Diccionarios cuyas claves son niveles o mercados y listas cuyos elementos se identifican
# por un campo (o por su posición con None).
FAMILIES = ("by_market", "quantiles.pinball", "quantiles.level_frequency")
FAMILIES += ("quantiles.level_frequency_gap",)
LISTS = {
    "quantiles.intervals": "nominal",
    "selective_direction.curve": "coverage",
    "quantiles.sign_probability.reliability": None,
}
SKIP = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _headings(path):
    return {
        line.lstrip("#").strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("## ") or line.startswith("### ")
    }


def _check_entry(name, entry, units, headings):
    _require(
        isinstance(entry, dict)
        and set(entry) <= {"definition", "unit", "better", "rule", "doc"}
        and isinstance(entry.get("definition"), str)
        and entry["definition"]
        and entry.get("unit") in units
        and entry.get("better") in BETTER,
        f"La cifra {name} necesita definición, unidad y sentido registrados",
    )
    rule = entry.get("rule")
    _require(
        isinstance(rule, dict) and rule.get("kind") in RULES,
        f"La regla de {name} no es conocida",
    )
    params = set(rule) - {"kind"}
    _require(
        RULES[rule["kind"]] <= params <= RULES[rule["kind"]] | _OPTIONAL
        and rule.get("side", "lower") in ("lower", "upper"),
        f"Los parámetros de la regla de {name} no son los suyos",
    )
    _require(
        "doc" not in entry or entry["doc"] in headings,
        f"La sección {entry.get('doc')} de {name} no está en la documentación",
    )


def _value(metrics, name):
    rule = metrics.get(name, {}).get("rule", {})
    return rule.get("value")


def _check_code(registry):
    """Las convenciones registradas deben ser las que aplica el código."""
    metrics = registry["metrics"]
    statuses = [
        name.removeprefix("rank_ic.sessions.")
        for name in metrics
        if name.startswith("rank_ic.sessions.")
    ]
    coverages = [f"{c:g}" for c in forecast_scores.SELECTIVE_COVERAGES]
    checks = (
        (registry["aggregation"]["weightings"] == list(WEIGHTINGS), "las ponderaciones"),
        (registry["aggregation"]["seeds"] == walk.SEED_AGGREGATION, "la agregación de semillas"),
        (registry["session"]["ineligible_rows"] == walk.INELIGIBLE_ROWS, "las filas excluidas"),
        (statuses == list(forecast_scores.RANK_IC_STATUS) == list(RANK_IC_STATUS), "los estados"),
        (registry["selective_coverages"] == coverages, "las coberturas de la abstención"),
        (registry["sign_bins"] == forecast_scores.SIGN_BINS, "los intervalos de probabilidad"),
        (
            _value(metrics, "direction.convention") == forecast_scores.DIRECTION_CONVENTION,
            "la convención de dirección",
        ),
        (
            _value(metrics, "selective_direction.rule") == forecast_scores.SELECTIVE_RULE,
            "la regla de la abstención",
        ),
        (
            _value(metrics, "quantiles.sign_probability.rule")
            == forecast_scores.SIGN_PROBABILITY_RULE,
            "la probabilidad de subida",
        ),
    )
    for passed, what in checks:
        _require(passed, f"El registro no recoge {what} del código")


def _check_series(registry):
    """Cada serie de los contrastes apunta a una cifra registrada con su mismo sentido."""
    series, metrics = registry["series"], registry["metrics"]
    prefixes = (
        forecast_scores.INTERVAL_SCORE,
        forecast_scores.COVERAGE_ERROR,
        forecast_scores.SELECTIVE_ACCURACY,
    )
    for name in walk.SERIES_METRICS:
        key = next((p for p in prefixes if name.startswith(p)), name)
        _require(key in series, f"La serie {name} de la comparación no está registrada")
    for name, target in series.items():
        _require(target in metrics, f"La serie {name} apunta a una cifra sin registrar")
        loss = forecast_scores.is_loss_series(name)
        _require(
            loss == (metrics[target]["better"] == "lower"),
            f"El sentido de la serie {name} no es el registrado",
        )


def load_registry(path=DEFAULT, *, root=ROOT):
    """Leer el registro y exigir que coincida con el código y la documentación de `root`."""
    path = Path(path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    _require(
        isinstance(registry, dict)
        and registry.get("schema_version") == 1
        and registry.get("kind") == KIND
        and registry.get("status") == walk.DECLARED
        and registry.get("final_test_opened") is False,
        "El registro de métricas no es válido o abre la reserva",
    )
    units, metrics = registry.get("units"), registry.get("metrics")
    _require(
        isinstance(units, dict) and units and isinstance(metrics, dict) and metrics,
        "El registro necesita unidades y cifras",
    )
    headings = _headings(Path(root) / registry["documentation"])
    for name, entry in metrics.items():
        _check_entry(name, entry, units, headings)
    for name, entry in metrics.items():
        of = entry["rule"].get("of")
        _require(of is None or of in metrics, f"La cifra {name} deriva de una sin registrar")
    _check_code(registry)
    _check_series(registry)
    return dict(registry, path=str(path.resolve()), sha256=sha256(path))


# Publicado: hojas del resumen con su patrón de registro y sus claves.


def _key(value):
    return f"{value:g}" if isinstance(value, float) else str(value)


def flatten(summary):
    """Hojas de un resumen como {(patrón, claves): valor}.

    El patrón sustituye las claves de las familias por `*` y los elementos de las listas
    por `[]`. Una lista de números es una sola hoja.
    """
    leaves = {}

    def walk_node(node, pattern, keys):
        dotted = ".".join(pattern)
        if isinstance(node, dict):
            family = dotted in FAMILIES
            for name, child in node.items():
                if family:
                    walk_node(child, (*pattern, "*"), (*keys, str(name)))
                else:
                    walk_node(child, (*pattern, name), keys)
        elif isinstance(node, list) and dotted in LISTS:
            field = LISTS[dotted]
            *head, last = pattern
            for index, item in enumerate(node):
                key = str(index) if field is None else _key(item[field])
                walk_node(item, (*head, f"{last}[]"), (*keys, key))
        else:
            leaves[".".join(pattern), keys] = node

    walk_node(summary, (), ())
    return leaves


# Esperado: el mismo resumen reconstruido desde la tabla de sesiones con las reglas.


def _column(frame, name, key):
    return frame[name.format(key=key)].to_numpy()


def _lists(frame, name):
    return np.vstack([np.asarray(item, dtype=np.float64) for item in frame[name]])


def _defined(frame, rule):
    defined = np.ones(len(frame), dtype=bool)
    if "status" in rule:
        defined &= frame["rank_ic_status"].to_numpy() == rule["status"]
    if "zero" in rule:
        defined &= frame[rule["zero"]].to_numpy() == 0
    if "positive" in rule:
        defined &= frame[rule["positive"]].to_numpy() > 0
    return defined


def _retained(frame, key):
    coverage = Fraction(key)
    eligible = frame["direction_eligible"].to_numpy()
    return -(-(coverage.numerator * eligible) // coverage.denominator)


def _ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator)
    return numerator / np.maximum(denominator, 1), denominator > 0


class _View:
    """Reglas del registro sobre las sesiones en bruto de un modelo, semilla y vista."""

    def __init__(self, registry, frame, weighting, quantiles):
        self.metrics, self.frame = registry["metrics"], frame
        self.weighting, self.quantiles = weighting, quantiles
        self.coverages = registry["selective_coverages"]
        self.bins = registry["sign_bins"]
        self.memo, self.matrices = {}, {}

    def keys(self, pattern):
        """Claves de las familias y listas de un patrón, desde la tabla y el registro."""
        if pattern.startswith("quantiles.intervals[]"):
            prefix = "coverage_"
        elif pattern.startswith(("quantiles.pinball", "quantiles.level_frequency")):
            prefix = "pinball_"
        elif pattern.startswith("selective_direction.curve[]"):
            return [(key,) for key in self.coverages]
        elif pattern.startswith("quantiles.sign_probability.reliability[]"):
            return [(str(index),) for index in range(self.bins)]
        else:
            return [()]
        found = [name.removeprefix(prefix) for name in self.frame if name.startswith(prefix)]
        return [(key,) for key in sorted(found, key=float)]

    def matrix(self, name):
        """Columna de listas como matriz, convertida una sola vez por vista."""
        if name not in self.matrices:
            self.matrices[name] = _lists(self.frame, name)
        return self.matrices[name]

    def mean(self, values, defined):
        markets = self.frame["market"].to_numpy()
        return session_mean(values, defined, markets, self.weighting)

    def value(self, name, keys):
        if (name, keys) not in self.memo:
            self.memo[name, keys] = self._compute(self.metrics[name]["rule"], name, keys)
        return self.memo[name, keys]

    def _over(self, of):
        values = [self.value(of, keys) for keys in self.keys(of)]
        return [value for value in values if value is not None]

    def _compute(self, rule, name, keys):
        kind, frame, key = rule["kind"], self.frame, keys[-1] if keys else None
        if kind == "descriptor":
            return rule.get("value", SKIP)
        if kind == "not_recomputed":
            return SKIP
        if kind == "weighting":
            return self.weighting
        if kind == "key":
            return float(key)
        if kind == "family_keys":
            return [float(found[0]) for found in self.keys(rule["of"])]
        if kind == "central_level":
            return (1 - float(key)) / 2 if rule["side"] == "lower" else (1 + float(key)) / 2
        if kind == "bin_edge":
            return (int(key) + (rule["side"] == "upper")) / self.bins
        if kind == "count":
            return int(np.sum(_defined(frame, rule)))
        if kind == "sum":
            total = frame[rule["column"]].sum()
            return int(total - sum(frame[column].sum() for column in rule.get("minus", ())))
        if kind == "session_mean":
            values = _column(frame, rule["column"], key)
            return self.mean(values, _defined(frame, rule))
        if kind == "session_ratio":
            return self.mean(*_ratio(frame[rule["numerator"]], frame[rule["denominator"]]))
        if kind == "row_weighted_mean":
            samples = frame["samples"].to_numpy()
            return float(np.dot(samples, frame[rule["column"]]) / samples.sum())
        if kind == "pooled_ratio":
            denominator = frame[rule["denominator"]].sum()
            return float(frame[rule["numerator"]].sum() / denominator) if denominator else None
        if kind in ("session_std", "positive_fraction"):
            values = frame[rule["column"]].to_numpy()[_defined(frame, rule)]
            if kind == "positive_fraction":
                return float(np.mean(values > 0)) if len(values) else None
            return float(np.std(values, ddof=1)) if len(values) > 1 else None
        if kind in ("sqrt", "percent", "minus_key"):
            base = self.value(rule["of"], keys)
            if base is None:
                return None
            if kind == "sqrt":
                return math.sqrt(base)
            return 100 * base if kind == "percent" else base - float(key)
        if kind in ("mean_over", "mean_abs_over", "max_abs_over"):
            values = self._over(rule["of"])
            if kind != "mean_over":
                values = [abs(value) for value in values]
            if not values:
                return None
            return max(values) if kind == "max_abs_over" else math.fsum(values) / len(values)
        if kind == "undercovered":
            covered = {found[0]: self.value(rule["of"], found) for found in self.keys(rule["of"])}
            return [
                float(key)
                for key, value in covered.items()
                if value is not None and value < float(key)
            ]
        if kind.startswith("selective_"):
            hits = self.matrix("selective_hits")[:, self.coverages.index(key)]
            retained = _retained(frame, key)
            if kind == "selective_rows":
                return int(retained.sum())
            if kind == "selective_hits":
                return int(hits.sum())
            return self.mean(*_ratio(hits, retained))
        rows = self.matrix("sign_bin_rows").sum(axis=0)
        if kind == "bin_total":
            return int(rows.sum())
        if kind == "bin_rows":
            return int(rows[int(key)])
        if kind == "bin_mean":
            index = int(key)
            total = self.matrix(rule["column"]).sum(axis=0)[index]
            return float(total / rows[index]) if rows[index] else None
        probability = self.matrix("sign_bin_probability").sum(axis=0)
        up = self.matrix("sign_bin_up").sum(axis=0)
        return float(np.abs(up - probability).sum() / rows.sum()) if rows.sum() else None


def expected(registry, frame, *, weighting, quantiles):
    """Resumen esperado de una vista como {(patrón, claves): valor} desde sus sesiones."""
    view = _View(registry, frame, weighting, quantiles)
    stored = [name for name in frame if name.startswith(("pinball_", "sign_bin_"))]
    _require(
        not quantiles or stored and frame[stored].notna().all(axis=None),
        "Las sesiones de un modelo con cuantiles no guardan sus cuantiles",
    )
    leaves = {}
    for name in registry["metrics"]:
        inside = name.startswith("quantiles.")
        if inside and not quantiles or name == "quantiles" and quantiles:
            continue
        for keys in view.keys(name):
            leaves[name, keys] = None if name == "quantiles" else view.value(name, keys)
    points = [name for name in registry["metrics"] if name.startswith("point.")]
    for market in sorted(frame["market"].unique()):
        part = _View(registry, frame[frame["market"] == market], weighting, quantiles)
        for name in points:
            alias = "by_market.*." + name.removeprefix("point.")
            leaves[alias, (market,)] = part.value(name, ())
    return leaves


def _close(left, right, rtol, atol):
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_close(a, b, rtol, atol) for a, b in zip(left, right, strict=True))
        )
    if left is None or right is None or isinstance(left, (str, bool)):
        return left == right
    return abs(left - right) <= atol + rtol * abs(right)


def registered(registry, summary):
    """Hojas de un resumen cuyo patrón no está registrado, como cadenas legibles."""
    names = set(registry["metrics"])
    points = {name.removeprefix("point.") for name in names if name.startswith("point.")}
    unknown = []
    for pattern, keys in flatten(summary):
        if pattern.startswith("by_market.*."):
            known = pattern.removeprefix("by_market.*.") in points
        else:
            known = pattern in names
        if not known:
            unknown.append(_label(pattern, keys))
    return sorted(unknown)


def _label(pattern, keys):
    return f"{pattern} {list(keys)}" if keys else pattern


def check_summary(registry, summary, frame, *, weighting, quantiles, rtol, atol):
    """Comparar el resumen publicado de una vista con el reconstruido desde sus sesiones."""
    published, wanted = (
        flatten(summary),
        expected(registry, frame, weighting=weighting, quantiles=quantiles),
    )
    missing = sorted(_label(*leaf) for leaf in wanted.keys() - published.keys())
    unregistered = sorted(_label(*leaf) for leaf in published.keys() - wanted.keys())
    mismatched, checked, skipped = [], 0, 0
    for leaf in sorted(wanted.keys() & published.keys(), key=str):
        value = wanted[leaf]
        if value is SKIP:
            skipped += 1
            continue
        checked += 1
        if not _close(value, published[leaf], rtol, atol):
            mismatched.append(
                dict(metric=_label(*leaf), recomputed=value, published=published[leaf])
            )
    failures = len(missing) + len(unregistered) + len(mismatched)
    return dict(
        checked=checked,
        not_recomputed=skipped,
        missing=missing,
        unregistered=unregistered,
        mismatched=mismatched,
        failures=failures,
        passed=failures == 0,
    )
