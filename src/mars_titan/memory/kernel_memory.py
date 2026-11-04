"""Regresión de Nadaraya-Watson sobre resultados maduros, el control no paramétrico de B1.

Guarda los mismos pares que escribe la memoria por áreas (clave del codec normalizada y
etiqueta madura menos predicción del núcleo) y lee la media de sus valores ponderada por un
núcleo exponencial del coseno y por la misma retención por escritura que la regla proximal:

    w_i(q) ∝ exp(qᵀk_i / τ) · ρ^(c − c_i),    ŷ(q) = Σ_i w_i(q) v_i,

donde c es el número de escrituras (cohortes) de la memoria, c_i el de la escritura del
episodio i y ρ = 1 − λ. El contador `writes` cuenta resultados, como en B6, para numerarlos.
Sin episodios la lectura es cero. La capacidad acota los episodios guardados: al superarla se
descartan los más antiguos del orden canónico. Es la referencia de
Nadaraya (1964) y Watson (1964) con la ponderación temporal declarada, no una variante de
Titans ni del banco episódico, que guarda episodios por otra política.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from .associative_memory import (
    MAX_COHORT,
    _check_keys,
    _check_matrix,
    _check_vector,
    canonical_order,
)

KERNEL_RULE = "nadaraya_watson"
MAX_CAPACITY = 65536
# Elementos de la matriz de pesos que se calculan a la vez al leer: unos 16 MiB en FP64.
READ_BLOCK = 2**21


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def declaration():
    """Valores admitidos del control, que la declaración de MARS-TITAN debe repetir."""
    return dict(
        rule=KERNEL_RULE,
        key=["codec"],
        bandwidth="tau",
        forgetting="lambda_below_one",
        capacity=[1, MAX_CAPACITY],
    )


@dataclass(frozen=True)
class KernelMemoryConfig:
    """`bandwidth` es τ, `forgetting` es λ < 1 y `capacity` el máximo de episodios guardados."""

    bandwidth: float
    forgetting: float = 0.0
    capacity: int = MAX_CAPACITY
    key_size: int = 64
    value_size: int = 1

    def __post_init__(self):
        for name in ("bandwidth", "forgetting"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} debe ser un número finito")
            object.__setattr__(self, name, float(value))
        if not 1e-4 <= self.bandwidth <= 100.0 or not 0.0 <= self.forgetting < 1.0:
            raise ValueError("Nadaraya-Watson necesita τ en [1e-4, 100] y λ en [0, 1)")
        if type(self.capacity) is not int or not 1 <= self.capacity <= MAX_CAPACITY:
            raise ValueError(f"La capacidad debe ser un entero entre 1 y {MAX_CAPACITY}")
        if (
            type(self.key_size) is not int
            or self.key_size != 64
            or type(self.value_size) is not int
            or self.value_size != 1
        ):
            raise ValueError("El control lee la clave de 64 coordenadas del codec y un valor")

    @property
    def rule(self):
        return KERNEL_RULE

    def identity(self):
        return dict(
            asdict(self),
            schema_version=1,
            kind="mature_kernel_regression",
            rule=KERNEL_RULE,
            dtype="float64",
            device="cpu",
            kernel="exp_cosine_over_bandwidth",
            temporal_weight="retention_power_writes_since_the_episode",
            empty_read="zero",
            eviction="oldest_in_canonical_order_beyond_capacity",
            order="label_available_at_then_decision_at_then_id",
        )


class KernelMemory:
    """Episodios maduros guardados en FP64 y CPU, con escrituras que devuelven copias."""

    def __init__(
        self, config, keys=None, values=None, written=None, *, writes=0, cohorts=0, cursor=None
    ):
        if type(config) is not KernelMemoryConfig:
            raise ValueError("Nadaraya-Watson necesita su configuración identificada")
        if keys is None:
            keys = torch.empty((0, config.key_size), dtype=torch.float64)
            values = torch.empty((0, 1), dtype=torch.float64)
            written = torch.empty(0, dtype=torch.int64)
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 0 <= rows <= config.capacity:
            raise ValueError("Los episodios guardados superan la capacidad declarada")
        _check_keys(keys, rows, config.key_size)
        _check_matrix(values, "Los valores guardados", rows, 1)
        _check_vector(written, "Las escrituras guardadas", rows, torch.int64)
        if (
            type(writes) is not int
            or type(cohorts) is not int
            or not 0 <= cohorts <= writes <= torch.iinfo(torch.int64).max
            or writes > cohorts * MAX_COHORT
            or rows != min(writes, config.capacity)
            or (cursor is None) != (writes == 0)
            or (cohorts == 0) != (writes == 0)
            or (
                cursor is not None
                and (
                    not isinstance(cursor, tuple)
                    or len(cursor) != 3
                    or any(type(item) is not int for item in cursor)
                    or not cursor[1] < cursor[0]
                    or cursor[2] < 1
                )
            )
        ):
            raise ValueError("El cursor y el contador de escrituras no son coherentes")
        if rows and (
            bool((written < 1).any())
            or int(written.max()) > cohorts
            or int(written[-1]) != cohorts
            or bool((written[1:] < written[:-1]).any())
        ):
            raise ValueError("Cada episodio pertenece a una escritura ya aplicada")
        self.config, self.writes, self.cohorts, self.cursor = config, writes, cohorts, cursor
        self._keys, self._values, self._written = (
            keys.detach().clone(),
            values.detach().clone(),
            written.detach().clone(),
        )

    @property
    def size(self):
        return self._keys.shape[0]

    @property
    def matrix(self):
        """Valores guardados, [episodios, 1]. Sirve para la norma de las trazas."""
        return self._values.clone()

    def read(self, keys):
        """Media ponderada de los valores guardados, de forma [filas, 1], sin efectos."""
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"La lectura necesita entre 1 y {MAX_COHORT} claves")
        _check_keys(keys, rows, self.config.key_size)
        if self.size == 0:
            return torch.zeros((rows, 1), dtype=torch.float64)
        retention = math.log1p(-self.config.forgetting)
        age = (self.cohorts - self._written).to(torch.float64) * retention
        block = max(1, READ_BLOCK // self.size)
        parts = []
        for start in range(0, rows, block):
            logits = keys[start : start + block] @ self._keys.T / self.config.bandwidth + age
            parts.append(torch.softmax(logits, dim=1) @ self._values)
        return torch.cat(parts)

    def write(self, feedback, *, cutoff):
        """Guardar los resultados maduros de un evento como una escritura más."""
        rows, order = canonical_order(
            feedback,
            cutoff,
            key_size=self.config.key_size,
            value_size=self.config.value_size,
            weighted=False,
            cursor=self.cursor,
        )
        if rows == 0:
            return self
        if (
            self.writes > torch.iinfo(torch.int64).max - rows
            or self.cohorts == torch.iinfo(torch.int64).max
        ):
            raise ValueError("Los contadores exceden el rango entero de las escrituras")
        index = torch.tensor(order, dtype=torch.int64)
        cohorts = self.cohorts + 1
        keys = torch.cat([self._keys, feedback.keys.index_select(0, index)])
        values = torch.cat([self._values, feedback.values.index_select(0, index)])
        written = torch.cat([self._written, torch.full((rows,), cohorts, dtype=torch.int64)])
        keep = slice(max(0, keys.shape[0] - self.config.capacity), None)
        last = order[-1]
        cursor = (
            int(feedback.available_at[last]),
            int(feedback.decision_at[last]),
            int(feedback.ids[last]),
        )
        return KernelMemory(
            self.config,
            keys[keep],
            values[keep],
            written[keep],
            writes=self.writes + rows,
            cohorts=cohorts,
            cursor=cursor,
        )

    def _digest(self):
        digest = hashlib.sha256()
        for tensor in (self._keys, self._values, self._written):
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
        return digest.hexdigest()

    def export(self):
        return dict(
            schema_version=1,
            configuration=self.config.identity(),
            keys=self._keys.clone(),
            values=self._values.clone(),
            written=self._written.clone(),
            matrix_sha256=self._digest(),
            writes=self.writes,
            cohorts=self.cohorts,
            cursor=list(self.cursor) if self.cursor is not None else None,
        )

    @classmethod
    def restore(cls, config, payload):
        fields = {"schema_version", "configuration", "keys", "values", "written"}
        fields |= {"matrix_sha256", "writes", "cohorts", "cursor"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("El estado exportado de Nadaraya-Watson no conserva sus campos")
        if payload["schema_version"] != 1 or _canonical(payload["configuration"]) != _canonical(
            config.identity()
        ):
            raise ValueError("El estado pertenece a otra configuración de Nadaraya-Watson")
        cursor = payload["cursor"]
        memory = cls(
            config,
            payload["keys"],
            payload["values"],
            payload["written"],
            writes=payload["writes"],
            cohorts=payload["cohorts"],
            cursor=tuple(cursor) if isinstance(cursor, list) else cursor,
        )
        if memory._digest() != payload["matrix_sha256"]:
            raise ValueError("Los episodios no coinciden con su huella")
        return memory
