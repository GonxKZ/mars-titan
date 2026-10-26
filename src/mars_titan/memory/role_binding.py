"""Claves ligadas a un rol observable mediante signos bipolares (B3).

El producto por signos y su inversa corresponden al caso bipolar descrito por Kanerva
(2009), páginas 147 y 156. Su uso sobre el codec financiero es una adaptación propia.
La inversa recupera la clave ligada, nunca una asociación perdida en DPFP o en la memoria.
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from .associative_memory import MAX_COHORT
from .regimes import CALENDAR_RULE, REGIME_RULE, SLOTS

WIDTH = 64
SOURCES = {"regime": REGIME_RULE, "calendar": CALENDAR_RULE}
MODES = ("bipolar", "constant_role", "projected_concat")


def declaration():
    return dict(source=list(SOURCES), mode=list(MODES), seed="uint32", key=["codec"])


@dataclass(frozen=True)
class RoleBindingConfig:
    """Fuente del rol, operador y semilla propia, sin parámetros ajustados."""

    source: str = "regime"
    mode: str = "bipolar"
    seed: int = 0

    def __post_init__(self):
        if (
            type(self.source) is not str
            or type(self.mode) is not str
            or self.source not in SOURCES
            or self.mode not in MODES
        ):
            raise ValueError("La fuente o el operador de ligadura de roles no están admitidos")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de roles debe ser un entero uint32")

    @cached_property
    def _codes(self):
        generator = np.random.Generator(np.random.PCG64([0xB3, 0, self.seed]))
        return torch.from_numpy(2 * generator.integers(0, 2, (SLOTS, WIDTH)) - 1).to(torch.float64)

    @cached_property
    def _projection(self):
        generator = np.random.Generator(np.random.PCG64([0xB3, 1, self.seed]))
        values = 2 * generator.integers(0, 2, (2 * WIDTH, WIDTH)) - 1
        return torch.from_numpy(values).to(torch.float64) / WIDTH**0.5

    @property
    def numeric_state_bytes(self):
        """Tablas fijas residentes, sin temporales ni objetos Python."""
        return self._codes.nbytes + (
            self._projection.nbytes if self.mode == "projected_concat" else 0
        )

    def identity(self):
        def digest(tensor):
            return hashlib.sha256(tensor.numpy().astype("<f8").tobytes()).hexdigest()

        return dict(
            kind="role_bound_keys_v1",
            source=self.source,
            mode=self.mode,
            seed=self.seed,
            width=WIDTH,
            slots=SLOTS,
            generator="numpy_pcg64_b3_0_seed_for_codes_b3_1_seed_for_projection",
            codes_sha256=digest(self._codes),
            projection_sha256=digest(self._projection) if self.mode == "projected_concat" else None,
            insertion="normalized_codec_before_optional_area_dpfp",
            inverse="same_sign_product" if self.mode != "projected_concat" else None,
        )

    def _roles(self, values, roles):
        if (
            not isinstance(values, torch.Tensor)
            or values.layout != torch.strided
            or values.device.type != "cpu"
            or values.dtype != torch.float64
            or values.ndim != 2
            or values.shape[1] != WIDTH
            or not 0 <= len(values) <= MAX_COHORT
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Las claves de roles deben ser una matriz CPU FP64 finita de ancho 64")
        if (
            not isinstance(roles, torch.Tensor)
            or roles.layout != torch.strided
            or roles.device.type != "cpu"
            or roles.dtype != torch.int64
            or roles.shape != (len(values),)
            or (roles < 0).any()
            or (roles >= SLOTS).any()
        ):
            raise ValueError("Cada clave necesita una ruta CPU int64 entre cero y cuatro")
        return torch.zeros_like(roles) if self.mode == "constant_role" else roles

    def apply(self, values, roles):
        """Producto por el código del rol, o concatenación proyectada y normalizada."""
        roles = self._roles(values, roles)
        codes = self._codes[roles]
        if self.mode != "projected_concat":
            return values * codes
        joined = torch.cat((values, codes / WIDTH**0.5), dim=1)
        projected = joined @ self._projection
        if not torch.isfinite(projected).all():
            raise ValueError("La proyección de la concatenación contiene valores no finitos")
        scale = projected.abs().amax(dim=1, keepdim=True)
        scaled = projected / torch.where(scale > 0, scale, 1.0)
        norm = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
        return scaled / torch.where(norm > 0, norm, 1.0)

    def unbind(self, values, roles):
        """Inversa exacta del producto bipolar para claves finitas, antes de DPFP."""
        if self.mode == "projected_concat":
            raise ValueError("La concatenación proyectada no tiene inversa declarada")
        return self.apply(values, roles)


class RoleBoundMemory:
    """Backend de B3 con contrato de roles incluido en la recuperación.

    Las claves ya llegan ligadas por MatureCorrection. El estado numérico conserva el backend
    de B6/B1/B2. La envoltura impide cargarlo con otra fuente, semilla o regla de roles.
    """

    def __init__(self, correction, backend):
        self.correction, self._backend = correction, backend

    @property
    def config(self):
        return self._backend.config

    @property
    def writes(self):
        return self._backend.writes

    @property
    def cursor(self):
        return self._backend.cursor

    @property
    def matrix(self):
        return self._backend.matrix

    @property
    def numeric_state_bytes(self):
        backend = self._backend
        resident = (
            backend.numeric_state_bytes
            if hasattr(backend, "numeric_state_bytes")
            else backend.matrix.nbytes
        )
        return resident + self.correction.role_binding.numeric_state_bytes

    def read(self, keys):
        return self._backend.read(keys)

    def write(self, feedback, *, cutoff):
        updated = self._backend.write(feedback, cutoff=cutoff)
        return self if updated is self._backend else RoleBoundMemory(self.correction, updated)

    def export(self):
        backend = self._backend.export()
        return dict(
            schema_version=1,
            configuration=self.correction.identity(),
            memory=backend,
            writes=self.writes,
            cursor=backend["cursor"],
            matrix_sha256=backend["matrix_sha256"],
        )

    @classmethod
    def restore(cls, correction, payload):
        def canonical(value):
            return json.dumps(value, sort_keys=True, allow_nan=False)

        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"schema_version", "configuration", "memory", "writes", "cursor", "matrix_sha256"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or canonical(payload["configuration"]) != canonical(correction.identity())
        ):
            raise ValueError("El estado no conserva la configuración de roles declarada")
        memory = cls(correction, correction._unbound().restore(payload["memory"]))
        expected = memory.export()
        if type(payload["writes"]) is not int or any(
            canonical(payload[field]) != canonical(expected[field])
            for field in ("writes", "cursor", "matrix_sha256")
        ):
            raise ValueError("Los roles no conservan el contador, cursor o huella de su memoria")
        return memory
