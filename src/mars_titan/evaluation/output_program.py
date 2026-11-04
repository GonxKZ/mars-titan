"""Gramática cerrada de salida de la autoedición `posthoc_self_edit_v1` (G1, #748).

Un programa de salida reescribe las predicciones guardadas de las semillas de un mismo ajuste
sin tocar pesos. Tiene cuatro campos con dominios finitos y tipados:

    q̄ = w·q_s + (1 − w)·(1/S)·Σ_k q_k
    A_i = 1{l_i > 0 o u_i < 0}
    T_i = 1{|m̄_i| ≥ Q_τ(|m̄| en G del mercado)}
    ŷ_i = λ_g·m̄_i si A_i y T_i, 0 en otro caso

- `seed_weight` (w) combina nivel a nivel los cuantiles de la propia semilla s con la media de
  las S semillas. m̄ es la mediana de q̄.
- `coverage` (c) es la abstención de `neutral_abstention`, con el intervalo [l, u] de
  cobertura c calibrado con la CQR de q̄ en G. Sin cobertura, A vale uno en todas las filas.
- `threshold` (τ) es el cuantil de |m̄| en G de cada mercado bajo el que no se emite. Con
  τ = 0, T vale uno en todas las filas.
- `shrinkage` (λ) es un factor fijo o `fit_g`, el λ por mercado de A1 elegido en G con el MAE
  por sesión (`scale_diagnostics`).

Una combinación convexa de cuantiles ordenados sigue ordenada también en coma flotante,
porque el redondeo es monótono. El programa de referencia (w = 1, sin cobertura, λ = 1 y
τ = 0) emite la mediana de la propia semilla bit a bit. Todo lo que el programa ajusta (CQR, λ
y umbrales) usa solo filas de G.
"""

import dataclasses
import hashlib
import itertools
from dataclasses import dataclass

import numpy as np

from mars_titan.evaluation import neutral_abstention, scale_diagnostics
from mars_titan.evaluation.forecast_panel import ForecastPanel

IDENTITY = "posthoc_self_edit_v1"
SCHEMA_VERSION = 1
# Dominios de cada campo, en el orden en que se enumeran los programas. La fase 0 encontró
# λ entre 0 y 0,019, así que los valores fijos bajan en potencias de cuatro hasta 1/64.
FIELDS = dict(
    seed_weight=(1.0, 0.75, 0.5, 0.25, 0.0),
    coverage=(None, 0.8, 0.95),
    shrinkage=(1.0, 0.25, 0.0625, 0.015625, "fit_g"),
    threshold=(0.0, 0.25, 0.5, 0.75),
)
REFERENCE = dict(seed_weight=1.0, coverage=None, shrinkage=1.0, threshold=0.0)
# Regla declarada de los cuantiles del umbral, la de NumPy por defecto.
QUANTILE_METHOD = "linear"
MIN_ROWS = neutral_abstention.DEFAULT["min_rows"]
_RECORD = (
    "schema_version",
    "identity",
    "program",
    "seeds",
    "own",
    "fitted_until",
    "abstention",
    "shrinkage",
    "thresholds",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _member(value, domain):
    # `True == 1.0` en Python. El tipo también debe coincidir con el del dominio.
    return any(type(value) is type(item) and value == item for item in domain)


def checked_program(program):
    """Validar un programa de la gramática y devolver una copia."""
    _require(
        isinstance(program, dict) and set(program) == set(FIELDS),
        f"Un programa de salida declara exactamente {sorted(FIELDS)}",
    )
    for name, domain in FIELDS.items():
        _require(_member(program[name], domain), f"{name} no pertenece a su dominio {domain}")
    return dict(program)


def programs():
    """Todos los programas de la gramática, en el orden canónico de los dominios."""
    return [
        dict(zip(FIELDS, values, strict=True)) for values in itertools.product(*FIELDS.values())
    ]


@dataclass(frozen=True, eq=False)
class SeedPanels:
    """Paneles de todas las semillas de un ajuste sobre la misma población, y la madurez.

    `own` es la posición de la semilla cuya salida se reescribe y `matures_at` la madurez de
    cada etiqueta en el orden canónico de los paneles.
    """

    panels: tuple
    own: int
    matures_at: np.ndarray

    def __post_init__(self):
        _require(
            isinstance(self.panels, tuple)
            and self.panels
            and all(isinstance(panel, ForecastPanel) for panel in self.panels),
            "Se necesitan los paneles de las semillas",
        )
        first = self.panels[0]
        _require(
            all(panel.cohort_sha256 == first.cohort_sha256 for panel in self.panels)
            and all(panel.levels == first.levels for panel in self.panels),
            "Las semillas no evalúan las mismas filas con los mismos niveles",
        )
        _require(
            type(self.own) is int and 0 <= self.own < len(self.panels),
            "La semilla propia no pertenece a los paneles",
        )
        matures_at = np.asarray(self.matures_at)
        _require(
            matures_at.shape == (first.rows,) and matures_at.dtype.kind in "iu",
            "La madurez necesita una fila por predicción",
        )
        object.__setattr__(self, "matures_at", matures_at.astype(np.int64))

    @property
    def reference(self):
        return self.panels[self.own]

    def sha256(self):
        """Huella de la población, la semilla propia, los cuantiles de todas y la madurez."""
        digest = hashlib.sha256(f"{self.reference.cohort_sha256}/{self.own}".encode())
        for panel in self.panels:
            digest.update(np.ascontiguousarray(panel.quantiles, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(panel.prediction, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(self.matures_at, dtype="<i8").tobytes())
        return digest.hexdigest()

    def bounds(self):
        """Primera emisión y mayor madurez del tramo, para ordenar los bloques."""
        return int(self.reference.prediction_at.min()), int(self.matures_at.max())


def combine(seeds, weight):
    """Panel de la propia semilla con los cuantiles combinados y su mediana como predicción."""
    _require(_member(weight, FIELDS["seed_weight"]), "Peso de combinación fuera de su dominio")
    own = seeds.reference
    if weight == 1.0:
        return own
    _require(len(seeds.panels) >= 2, "Combinar semillas necesita al menos dos")
    mean = np.mean([panel.quantiles for panel in seeds.panels], axis=0)
    quantiles = weight * own.quantiles + (1.0 - weight) * mean
    quantiles.setflags(write=False)
    median = quantiles[:, own.levels.index(0.5)]
    return dataclasses.replace(own, quantiles=quantiles, prediction=median)


def fit(program, seeds, *, min_rows=MIN_ROWS):
    """Ajustar en G la calibración, λ y los umbrales del programa. No ve tramos posteriores."""
    program = checked_program(program)
    combined = combine(seeds, program["seed_weight"])
    record = dict(
        schema_version=SCHEMA_VERSION,
        identity=IDENTITY,
        program=program,
        seeds=len(seeds.panels),
        own=seeds.own,
        fitted_until=int(seeds.matures_at.max()),
        abstention=None,
        shrinkage=None,
        thresholds=None,
    )
    if program["coverage"] is not None:
        config = neutral_abstention.DEFAULT | dict(nominal=program["coverage"], min_rows=min_rows)
        record["abstention"] = neutral_abstention.fit(combined, seeds.matures_at, config)
    if program["shrinkage"] == "fit_g":
        record["shrinkage"] = scale_diagnostics.market_shrinkage(combined, combined.prediction)
    if program["threshold"] > 0:
        magnitude = np.abs(combined.prediction)
        record["thresholds"] = {
            market: float(
                np.quantile(
                    magnitude[combined.market == code],
                    program["threshold"],
                    method=QUANTILE_METHOD,
                )
            )
            for code, market in enumerate(combined.markets)
            if np.any(combined.market == code)
        }
    return record


def apply(record, seeds):
    """Predicción del programa ajustado en un tramo posterior, sin leer sus objetivos."""
    _require(
        isinstance(record, dict)
        and set(record) == set(_RECORD)
        and record["schema_version"] == SCHEMA_VERSION
        and record["identity"] == IDENTITY,
        "El registro del programa de salida no conserva su contrato",
    )
    program = checked_program(record["program"])
    _require(
        record["seeds"] == len(seeds.panels) and record["own"] == seeds.own,
        "El programa se ajustó con otras semillas",
    )
    combined = combine(seeds, program["seed_weight"])
    _require(
        int(combined.prediction_at.min()) >= record["fitted_until"],
        "El programa usa etiquetas que maduran después de una emisión",
    )
    emitted = np.ones(combined.rows, dtype=bool)
    if record["abstention"] is not None:
        emitted &= ~neutral_abstention.apply(record["abstention"], combined)[1]
    if record["thresholds"] is not None:
        present = [combined.markets[code] for code in np.unique(combined.market)]
        _require(
            all(name in record["thresholds"] for name in present),
            "Hay mercados sin umbral ajustado en G",
        )
        bound = np.array([record["thresholds"].get(name, np.inf) for name in combined.markets])
        emitted &= np.abs(combined.prediction) >= bound[combined.market]
    if program["shrinkage"] == "fit_g":
        scaled = scale_diagnostics.apply_shrinkage(
            combined, combined.prediction, record["shrinkage"]
        )
    else:
        scaled = program["shrinkage"] * combined.prediction
    return np.where(emitted, scaled, 0.0)
