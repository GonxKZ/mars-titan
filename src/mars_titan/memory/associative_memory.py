"""Memoria asociativa lineal escrita con resultados maduros, por regla delta o proximal.

Es el comparador B6 de la matriz de experimentos: una matriz A de clave por valor que
solo cambia al aplicar resultados ya maduros. No sustituye la memoria neuronal de
Titans-MAC, cuya sorpresa asociativa usa entradas disponibles, ni el banco episódico,
que conserva episodios recuperables. Las fórmulas, cotas y contraejemplos proceden de
`docs/research/memory-mathematics.md`.

Cada escritura devuelve una memoria nueva y deja intacta la anterior. La lectura no
modifica estado. El orden de aplicación es canónico por disponibilidad de la etiqueta,
decisión e identificador, e independiente del orden de carga.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

RULES = ("delta", "proximal")
# Las claves vienen normalizadas por el codec en FP32. Este margen admite su redondeo.
KEY_NORM_TOLERANCE = 1e-6
WEIGHT_TOLERANCE = 1e-12
MAX_RATE = 1e6
MAX_SIZE = 512
MAX_COHORT = 8192


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value, name, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} debe pertenecer a [{minimum}, {maximum}]")
    return float(value)


def _size(value, name):
    if type(value) is not int or not 1 <= value <= MAX_SIZE:
        raise ValueError(f"{name} debe ser un entero entre 1 y {MAX_SIZE}")


def _matrix_digest(matrix):
    payload = matrix.detach().cpu().contiguous().numpy().astype("<f8", copy=False).tobytes()
    digest = hashlib.sha256(str(tuple(matrix.shape)).encode())
    digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssociativeMemoryConfig:
    """`rate` es η y `forgetting` es λ. En la regla proximal la retención es ρ = 1 − λ."""

    rule: str
    key_size: int = 64
    value_size: int = 1
    rate: float = 0.1
    forgetting: float = 0.0

    def __post_init__(self):
        if self.rule not in RULES:
            raise ValueError("La regla asociativa debe ser delta o proximal")
        _size(self.key_size, "La dimensión de clave")
        _size(self.value_size, "La dimensión de valor")
        forgetting = _finite(self.forgetting, "El olvido", 0.0, 1.0)
        rate = _finite(self.rate, "La tasa", 0.0, MAX_RATE)
        # Con ‖k‖ ≤ 1, η ≤ 2 − λ basta para no expandir diferencias en ninguna escritura.
        if self.rule == "delta" and rate > 2.0 - forgetting:
            raise ValueError("La regla delta exige η ≤ 2 − λ para no expandir el estado")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "forgetting", forgetting)

    @property
    def retention(self):
        return 1.0 - self.forgetting

    def contraction_bound(self):
        """Cota de ‖D⁺‖/‖D‖ para dos estados con las mismas entradas y ‖k‖ ≤ 1.

        Delta: máximo de |1 − λ − η‖k‖²| sobre ‖k‖² ∈ [0, 1], en norma espectral.
        Proximal: ρ/(1 + η λ_min(G)) ≤ ρ en Frobenius. No acota el estado absoluto.
        """
        if self.rule == "delta":
            return max(abs(self.retention), abs(self.retention - self.rate))
        return self.retention

    def identity(self):
        return dict(
            asdict(self),
            schema_version=1,
            kind="mature_associative_memory",
            dtype="float64",
            device="cpu",
            key_constraint="l2_norm_at_most_1",
            key_norm_tolerance=KEY_NORM_TOLERANCE,
            order="label_available_at_then_decision_at_then_id",
            write_cursor="strictly_increasing_canonical_key",
            empty_cohort="unchanged_without_forgetting",
            update="sequential_rank_one"
            if self.rule == "delta"
            else "cohort_cholesky_lapack_without_inverse",
            weights=None if self.rule == "delta" else "declared_nonnegative_sum_one",
        )

    def fingerprint(self):
        return hashlib.sha256(_canonical(self.identity()).encode()).hexdigest()


@dataclass(frozen=True)
class MatureFeedback:
    """Resultados maduros de un evento: claves estables y valores disponibles.

    `available_at` es el instante desde el que la etiqueta puede usarse. Las claves
    deben proceder del mismo espacio estable con el que se leerá la memoria.
    """

    ids: torch.Tensor
    decision_at: torch.Tensor
    available_at: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor | None = None


def _check_vector(value, name, size, dtype):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or value.layout != torch.strided
        or value.shape != (size,)
    ):
        raise ValueError(f"{name} necesita un vector CPU de {size} elementos y tipo {dtype}")


def _check_matrix(value, name, rows, columns):
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.layout != torch.strided
        or value.shape != (rows, columns)
    ):
        raise ValueError(f"{name} necesita una matriz CPU FP64 de forma {(rows, columns)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contiene valores no finitos")


def _check_keys(keys, rows, size):
    _check_matrix(keys, "Las claves", rows, size)
    if rows and torch.linalg.vector_norm(keys, dim=1).max() > 1.0 + KEY_NORM_TOLERANCE:
        raise ValueError("Las claves deben tener norma euclídea no mayor que uno")


class AssociativeMemory:
    """Matriz A de clave por valor en FP64 y CPU, con escrituras que devuelven copias."""

    def __init__(self, config, matrix=None, *, writes=0, cursor=None):
        if not isinstance(config, AssociativeMemoryConfig):
            raise ValueError("La memoria asociativa necesita su configuración identificada")
        shape = (config.key_size, config.value_size)
        if matrix is None:
            matrix = torch.zeros(shape, dtype=torch.float64)
        _check_matrix(matrix, "El estado asociativo", *shape)
        if type(writes) is not int or writes < 0:
            raise ValueError("El contador de escrituras debe ser un entero no negativo")
        if cursor is not None and (
            not isinstance(cursor, tuple)
            or len(cursor) != 3
            or any(type(item) is not int for item in cursor)
        ):
            raise ValueError("El cursor necesita disponibilidad, decisión e identificador")
        if (cursor is None) != (writes == 0):
            raise ValueError("El cursor y el contador de escrituras no son coherentes")
        self.config = config
        self._matrix = matrix.detach().clone()
        self.writes = writes
        self.cursor = cursor

    @property
    def matrix(self):
        return self._matrix.clone()

    def read(self, keys):
        """Devolver `keys @ A`, es decir Aᵀk por fila, de forma [b, valor] y sin efectos."""
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"La lectura necesita entre 1 y {MAX_COHORT} claves")
        _check_keys(keys, rows, self.config.key_size)
        return keys @ self._matrix

    def write(self, feedback, *, cutoff):
        """Aplicar los resultados maduros de un evento y devolver la memoria siguiente."""
        rows, order = self._validate(feedback, cutoff)
        if rows == 0:
            return self
        index = torch.tensor(order, dtype=torch.int64)
        keys = feedback.keys.index_select(0, index)
        values = feedback.values.index_select(0, index)
        if self.config.rule == "delta":
            matrix = self._delta(keys, values)
        else:
            matrix = self._proximal(keys, values, feedback.weights.index_select(0, index))
        if not torch.isfinite(matrix).all():
            raise ValueError("La escritura produjo un estado no finito")
        last = order[-1]
        cursor = (
            int(feedback.available_at[last]),
            int(feedback.decision_at[last]),
            int(feedback.ids[last]),
        )
        return AssociativeMemory(self.config, matrix, writes=self.writes + rows, cursor=cursor)

    def _validate(self, feedback, cutoff):
        if not isinstance(feedback, MatureFeedback) or type(cutoff) is not int:
            raise ValueError("La escritura necesita resultados maduros y un corte entero")
        rows = feedback.ids.shape[0] if isinstance(feedback.ids, torch.Tensor) else -1
        if not 0 <= rows <= MAX_COHORT:
            raise ValueError(f"Una escritura admite como máximo {MAX_COHORT} resultados")
        for name in ("ids", "decision_at", "available_at"):
            _check_vector(getattr(feedback, name), name, rows, torch.int64)
        _check_keys(feedback.keys, rows, self.config.key_size)
        _check_matrix(feedback.values, "Los valores", rows, self.config.value_size)
        proximal = self.config.rule == "proximal"
        if (feedback.weights is not None) != (proximal and rows > 0):
            raise ValueError("Solo la regla proximal declara pesos y solo con resultados")
        if rows == 0:
            return 0, []
        ids = feedback.ids.tolist()
        decisions = feedback.decision_at.tolist()
        available = feedback.available_at.tolist()
        if min(ids) < 1 or len(set(ids)) != rows:
            raise ValueError("Los identificadores deben ser positivos y únicos en la escritura")
        if any(not d < a <= cutoff for d, a in zip(decisions, available, strict=True)):
            raise ValueError("Cada etiqueta debe madurar después de su decisión y antes del corte")
        if proximal:
            _check_vector(feedback.weights, "Los pesos", rows, torch.float64)
            weights = feedback.weights
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Los pesos de la cohorte deben ser finitos y no negativos")
            if abs(float(weights.sum()) - 1.0) > WEIGHT_TOLERANCE:
                raise ValueError("Los pesos de la cohorte deben sumar uno")
        order = sorted(range(rows), key=lambda i: (available[i], decisions[i], ids[i]))
        first = order[0]
        if self.cursor is not None and (available[first], decisions[first], ids[first]) <= (
            self.cursor
        ):
            raise ValueError("El resultado ya se aplicó o llega fuera del orden canónico")
        return rows, order

    def _delta(self, keys, values):
        retention, rate = self.config.retention, self.config.rate
        if rate * float(torch.linalg.vector_norm(keys, dim=1).max()) ** 2 > 2.0 - (
            self.config.forgetting
        ):
            raise ValueError("Una clave incumple η‖k‖² ≤ 2 − λ")
        matrix = self._matrix.clone()
        for key, value in zip(keys, values, strict=True):
            residual = value - key @ matrix
            matrix = retention * matrix + rate * torch.outer(key, residual)
        return matrix

    def _proximal(self, keys, values, weights):
        weighted = keys * weights.unsqueeze(1)
        gram = weighted.T @ keys
        cross = weighted.T @ values
        system = torch.eye(self.config.key_size, dtype=torch.float64) + self.config.rate * gram
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise ValueError("El sistema proximal no es definido positivo en FP64")
        right = self.config.retention * self._matrix + self.config.rate * cross
        return torch.cholesky_solve(right, factor)

    def export(self):
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            matrix=self.matrix,
            matrix_sha256=_matrix_digest(self._matrix),
            writes=self.writes,
            cursor=list(self.cursor) if self.cursor is not None else None,
        )

    @classmethod
    def restore(cls, config, payload):
        if not isinstance(config, AssociativeMemoryConfig) or not isinstance(payload, dict):
            raise ValueError("La recuperación necesita configuración y estado exportado")
        expected = {"schema_version", "configuration", "matrix", "matrix_sha256", "writes"}
        if set(payload) != expected | {"cursor"} or payload["schema_version"] != 1:
            raise ValueError("El estado exportado no conserva todos sus campos")
        if _canonical(payload["configuration"]) != _canonical(config.identity()):
            raise ValueError("El estado pertenece a otra configuración asociativa")
        matrix = payload["matrix"]
        _check_matrix(matrix, "El estado exportado", config.key_size, config.value_size)
        if _matrix_digest(matrix) != payload["matrix_sha256"]:
            raise ValueError("La matriz no coincide con su huella")
        cursor = payload["cursor"]
        return cls(
            config,
            matrix,
            writes=payload["writes"],
            cursor=tuple(cursor) if isinstance(cursor, list) else cursor,
        )
