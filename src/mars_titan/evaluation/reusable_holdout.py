"""Escalera sin parámetros de Blum y Hardt (2015) para reutilizar S en la segunda ola (#794).

La figura 2 de *The Ladder: A Reliable Leaderboard for Machine Learning Competitions* publica
la pérdida de un envío solo si baja de la del mejor anterior en más de un error típico de la
diferencia emparejada, y entonces la redondea a múltiplos de 1/n. Si no, vuelve a publicar la
anterior y conserva el mejor. Así cada envío revela poco más que si mejoró, y el tablero sigue
siendo fiable aunque los envíos se elijan viendo los anteriores.

Aquí cada envío es la serie por sesión de una puntuación de S en la que mayor es mejor, como
el IC por sesión de la criba. La pérdida es la puntuación cambiada de signo. Dos desviaciones
respecto al artículo quedan declaradas en `configs/experiments/reusable-holdout-v1.json`:

- El error típico de la diferencia media se estima con el bootstrap circular por bloques de
  días UTC de la criba, porque las sesiones consecutivas no son independientes. El artículo
  usa std(ℓ_t − ℓ_mejor)/√n, que supone filas independientes.
- La diferencia se mide en las sesiones que definen a la vez el envío y el mejor. El valor
  publicado es la media del envío en sus propias sesiones definidas, redondeada a la
  resolución declarada.

La parte pública solo lleva el valor publicado y si el envío mejoró. La puntuación exacta, la
diferencia y su error típico van a la parte sellada, que no se lee hasta cerrar la ola.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest
from mars_titan.evaluation.paired_comparisons import circular_block_counts

DECLARATION_KIND = "reusable_holdout_declaration"
DECLARED = dict(
    schema_version=1,
    kind=DECLARATION_KIND,
    identity="parameter_free_ladder_block_bootstrap_v1",
    score="session_ic",
    direction="greater",
    threshold="one_bootstrap_standard_error_of_the_paired_mean_difference",
    resampling="circular_block_bootstrap_of_utc_days",
    public=["identity", "released", "improved"],
    sealed=[
        "score",
        "defined_sessions",
        "common_sessions",
        "paired_mean_difference",
        "standard_error",
        "best_before",
    ],
    final_test_access=False,
)
DAY_MICROSECONDS = 86_400_000_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def released_value(score, resolution):
    """Puntuación redondeada al múltiplo más cercano de la resolución, [x]_η del artículo."""
    return round(score / resolution) * resolution


def standard_error(differences, day, *, block_days, replicates, seed):
    """Error típico de la media de una serie por sesión con bloques circulares de días UTC."""
    _, period = np.unique(day, return_inverse=True)
    periods = int(period.max()) + 1
    _require(block_days < periods, "Se necesitan más días que la longitud del bloque")
    counts = circular_block_counts(np.random.default_rng(seed), replicates, periods, block_days)
    weights = counts[:, period].astype(np.float64)
    means = (weights @ differences) / weights.sum(axis=1)
    return float(means.std(ddof=1))


@dataclass(frozen=True)
class Submission:
    """Serie por sesión de un envío y las sesiones en que está definida."""

    identity: str
    values: np.ndarray
    defined: np.ndarray

    def __post_init__(self):
        _require(isinstance(self.identity, str) and self.identity, "El envío necesita un nombre")
        values = np.asarray(self.values, dtype=np.float64)
        defined = np.asarray(self.defined)
        _require(
            values.ndim == 1 and defined.shape == values.shape and defined.dtype == np.bool_,
            f"{self.identity} necesita una máscara booleana con la forma de su serie",
        )
        _require(
            bool(defined.any()) and bool(np.all(np.isfinite(values[defined]))),
            f"{self.identity} necesita sesiones definidas con valores finitos",
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "defined", defined)

    @property
    def score(self):
        return float(self.values[self.defined].mean())


class Ladder:
    """Tablero de S que solo publica una puntuación nueva cuando el envío mejora al mejor.

    `best` es el envío que ya está publicado al empezar, por ejemplo el mejor modelo de la
    primera ola. Sin él, el primer envío se publica siempre, como con R₀ = ∞ en el artículo.
    """

    def __init__(self, day, *, resolution, block_days, replicates, seed, best=None):
        self.day = np.asarray(day, dtype=np.int64)
        _require(
            self.day.ndim == 1 and bool(np.all(np.diff(self.day) >= 0)),
            "Los días de S deben seguir el orden temporal",
        )
        _require(
            type(resolution) is float and math.isfinite(resolution) and resolution > 0,
            "La resolución debe ser un número positivo",
        )
        _require(
            type(block_days) is int and block_days >= 1,
            "La longitud del bloque es un entero positivo",
        )
        _require(type(replicates) is int and replicates >= 2, "Hacen falta al menos dos réplicas")
        _require(type(seed) is int, "La semilla del bootstrap es un entero")
        self.resolution, self.block_days = resolution, block_days
        self.replicates, self.seed = replicates, seed
        self.seen = set()
        self.best = None
        self.released = None
        if best is not None:
            self._check(best)
            self.best, self.released = best, released_value(best.score, resolution)

    def _check(self, submission):
        _require(isinstance(submission, Submission), "Cada envío debe ser una Submission")
        _require(submission.identity not in self.seen, f"{submission.identity} ya se envió")
        _require(
            submission.values.shape == self.day.shape,
            f"{submission.identity} no tiene una sesión por cada día de S",
        )
        self.seen.add(submission.identity)

    def submit(self, submission):
        """Puntuar un envío y devolver su parte pública y su parte sellada."""
        self._check(submission)
        sealed = dict(
            identity=submission.identity,
            score=submission.score,
            defined_sessions=int(submission.defined.sum()),
            best_before=None if self.best is None else self.best.identity,
        )
        if self.best is None:
            improved = True
        else:
            common = submission.defined & self.best.defined
            _require(
                int(common.sum()) >= 2,
                f"{submission.identity} y el mejor anterior no comparten sesiones",
            )
            differences = (submission.values - self.best.values)[common]
            error = standard_error(
                differences,
                self.day[common],
                block_days=self.block_days,
                replicates=self.replicates,
                seed=self.seed,
            )
            difference = float(differences.mean())
            # En pérdidas, ℓ_t − ℓ_mejor < −s. Con puntuaciones donde mayor es mejor, la
            # diferencia de puntuaciones tiene que superar un error típico.
            improved = difference > error
            sealed.update(
                common_sessions=int(common.sum()),
                paired_mean_difference=difference,
                standard_error=error,
            )
        if improved:
            self.best, self.released = submission, released_value(submission.score, self.resolution)
        public = dict(identity=submission.identity, released=self.released, improved=improved)
        return public, dict(sealed, improved=improved)


def load_declaration(path):
    """Leer la declaración de la escalera y exigir que describa este mecanismo."""
    declaration, digest = read_manifest(Path(path), 1024**2)
    _require(isinstance(declaration, dict), "La declaración debe ser un objeto JSON")
    _require(
        {key: declaration.get(key) for key in DECLARED} == DECLARED,
        "La declaración no coincide con la escalera que se aplica",
    )
    return declaration, digest


def run_ladder(declaration, session_time, submissions, *, best=None):
    """Aplicar la escalera declarada a los envíos en su orden y separar lo público de lo sellado.

    `session_time` son los instantes de decisión de S en microsegundos UTC, en orden.
    """
    days = np.asarray(session_time, dtype=np.int64) // DAY_MICROSECONDS
    ladder = Ladder(
        days,
        resolution=declaration["resolution"],
        block_days=declaration["block_days"],
        replicates=declaration["replicates"],
        seed=declaration["seed"],
        best=best,
    )
    initial = None if best is None else dict(identity=best.identity, released=ladder.released)
    public, sealed = [], []
    for submission in submissions:
        shown, hidden = ladder.submit(submission)
        public.append(shown)
        sealed.append(hidden)
    return dict(public=dict(initial=initial, board=public), sealed=sealed)
