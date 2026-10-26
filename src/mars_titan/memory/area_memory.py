"""Memoria asociativa redundante por áreas escrita con resultados maduros (B1).

`redundant_area_memory_v1` reparte la corrección madura de B6 en A áreas. Cada área permuta
de forma fija las coordenadas de la clave del codec, la proyecta con DPFP-ν y guarda su propia
memoria asociativa lineal (`AssociativeMemory`) con la regla delta o proximal, la misma tasa y
la misma retención. La lectura es la media de las áreas activas, renormalizada por su número.

La redundancia procede de Danihelka et al. (2016), sección 3, ecuaciones 6 y 7: varias copias
de la traza con la clave permutada y la media de lo recuperado. La proyección DPFP-ν y la regla
delta proceden de Schlag et al. (2021), ecuaciones 24 y 37 y listado 1. La permutación sola no
basta con un estado matricial completo: si Q es ortogonal, escribir con Qk da Q por la memoria
escrita con k, en las dos reglas, y la lectura (Qq)ᵀ(QA) = qᵀA es la misma en todas las áreas.
DPFP multiplica coordenadas vecinas, así que cada permutación elige otros pares y las áreas
dejan de ser equivalentes. La derivación y sus controles están en
`docs/experiments/redundant-area-memory-v1.md`.

Las escrituras devuelven memorias nuevas y la lectura no modifica estado. No hay parámetros
entrenables ni gradientes: todo se calcula en FP64 y CPU, igual que B6.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .area_query_bridge import AreaQueryBridgeConfig, AreaQueryBridgeState, active_areas
from .associative_memory import (
    MAX_COHORT,
    MAX_SIZE,
    AssociativeMemory,
    AssociativeMemoryConfig,
    _check_keys,
    canonical_order,
)

KIND = "redundant_area_memory_v1"
PERMUTATIONS = ("seeded_random", "identity")
# `shuffled_within_cohort` es el control de etiquetas barajadas: conserva el conjunto de valores
# de cada escritura y rompe su asociación con la clave.
LABELS = ("mature", "shuffled_within_cohort")
AREA_RULES = ("delta", "proximal")
MAX_AREAS = 8
MAX_SHIFTS = 4


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().astype("<f8").tobytes())
    return digest.hexdigest()


def dpfp(values, shifts):
    """DPFP-ν de Schlag et al. (2021), ecuación 37, con el orden del listado 1.

    r = [ReLU(x), ReLU(−x)] y el bloque j multiplica r por su desplazamiento circular de j
    posiciones, de modo que la coordenada i del bloque j vale r_i · r_{(i − j) mod 2d}. La salida
    tiene 2dν coordenadas no negativas.
    """
    rectified = torch.cat([torch.relu(values), torch.relu(-values)], dim=-1)
    rolled = torch.cat([rectified.roll(shifts=j, dims=-1) for j in range(1, shifts + 1)], dim=-1)
    return torch.cat([rectified] * shifts, dim=-1) * rolled


def _scaled_dpfp(values, shifts):
    """DPFP salvo una potencia de dos común por fila, sin perder productos muy pequeños."""
    mantissas, exponents = torch.frexp(values)
    products = dpfp(mantissas, shifts)
    exponents = torch.cat([exponents, exponents], dim=-1)
    pair_exponents = torch.cat(
        [exponents + exponents.roll(shifts=j, dims=-1) for j in range(1, shifts + 1)],
        dim=-1,
    )
    present = products > 0
    maximum = torch.where(present, pair_exponents, torch.iinfo(exponents.dtype).min).amax(
        dim=-1, keepdim=True
    )
    maximum = torch.where(present.any(dim=-1, keepdim=True), maximum, 0)
    # Al restar el máximo antes de multiplicar por 2^e, ningún producto puede desbordarse.
    relative = torch.where(present, pair_exponents - maximum, 0)
    return torch.ldexp(products, relative)


def declaration():
    """Valores admitidos de la variante, que la declaración de MARS-TITAN debe repetir."""
    return dict(
        kind=KIND,
        rule=list(AREA_RULES),
        key=["codec"],
        rate="eta",
        forgetting="lambda",
        areas=dict(
            areas=[1, MAX_AREAS],
            shifts=[1, MAX_SHIFTS],
            permutation=list(PERMUTATIONS),
            seed="uint32",
            labels=list(LABELS),
        ),
    )


@dataclass(frozen=True)
class AreaLayout:
    """Número de áreas, desplazamientos de DPFP, permutaciones y etiquetas de B1.

    Con `permutation="identity"` todas las áreas usan la clave sin permutar, el control de
    áreas idénticas. Con una sola área y ν = A desplazamientos se obtiene el control de un área
    única con el mismo número de coordenadas de estado que A áreas con ν = 1. La semilla fija
    las permutaciones y el barajado de etiquetas del control.
    """

    areas: int = 4
    shifts: int = 1
    permutation: str = "seeded_random"
    seed: int = 0
    labels: str = "mature"

    def __post_init__(self):
        for name, value, maximum in (
            ("areas", self.areas, MAX_AREAS),
            ("shifts", self.shifts, MAX_SHIFTS),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} debe ser un entero entre 1 y {maximum}")
        if self.permutation not in PERMUTATIONS or self.labels not in LABELS:
            raise ValueError("La permutación o el tratamiento de etiquetas no están admitidos")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla de las áreas debe ser un entero uint32")

    def feature_size(self, key_size):
        return 2 * key_size * self.shifts

    def permutations(self, key_size):
        """Índices de cada área: la coordenada i del área s es la π_s(i) de la clave."""
        if self.permutation == "identity":
            return np.tile(np.arange(key_size, dtype=np.int64), (self.areas, 1))
        generator = np.random.Generator(np.random.PCG64(self.seed))
        return np.stack([generator.permutation(key_size) for _ in range(self.areas)])

    def identity(self):
        return dict(
            asdict(self),
            kind=KIND,
            feature_map="dpfp_of_permuted_key_then_l2_or_zero",
            permutation_generator="numpy_pcg64_seed_permutation_per_area_in_order",
            read="mean_over_active_areas",
            label_shuffle=None
            if self.labels == "mature"
            else "numpy_pcg64_seed_and_first_canonical_id_permutation_of_values",
        )


class AreaMemory:
    """A memorias asociativas sobre las proyecciones de la clave de cada área.

    `config` es la regla común declarada sobre la clave de entrada. Cada área usa la misma
    regla con `key_size` igual a 2dν. Todas reciben los mismos resultados en la misma escritura,
    así que comparten contador y cursor.
    """

    def __init__(self, config, layout, areas=None, *, query_bridge=None, bridge_state=None):
        if (
            type(config) is not AssociativeMemoryConfig
            or config.rule not in AREA_RULES
            or type(layout) is not AreaLayout
        ):
            raise ValueError("Las áreas necesitan una regla delta o proximal y su reparto")
        size = layout.feature_size(config.key_size)
        if size > MAX_SIZE:
            raise ValueError(f"Cada área admite como máximo {MAX_SIZE} coordenadas de DPFP")
        self.config, self.layout = config, layout
        self.area_config = replace(config, key_size=size)
        if areas is None:
            areas = tuple(AssociativeMemory(self.area_config) for _ in range(layout.areas))
        if len(areas) != layout.areas or any(
            type(area) is not AssociativeMemory or area.config != self.area_config for area in areas
        ):
            raise ValueError("Las áreas no corresponden al reparto declarado")
        if len({(area.writes, area.cursor) for area in areas}) != 1:
            raise ValueError("Las áreas no conservan el mismo contador y cursor")
        self._areas = tuple(areas)
        self._index = torch.from_numpy(layout.permutations(config.key_size))
        if query_bridge is not None and type(query_bridge) is not AreaQueryBridgeConfig:
            raise ValueError("El puente de consultas necesita su configuración identificada")
        if query_bridge is None and bridge_state is not None:
            raise ValueError("El estado del puente necesita una configuración activa")
        self.query_bridge = query_bridge
        self._bridge = None
        if query_bridge is not None:
            rule = replace(self.area_config, value_size=size)
            if bridge_state is None:
                bridge_state = AreaQueryBridgeState(rule, layout.areas, query_bridge)
            if (
                type(bridge_state) is not AreaQueryBridgeState
                or bridge_state.rule != rule
                or bridge_state.areas != layout.areas
                or bridge_state.config != query_bridge
                or (bridge_state.writes, bridge_state.cursor) != (self.writes, self.cursor)
            ):
                raise ValueError("Los mapas del puente no corresponden al estado de las áreas")
            self._bridge = bridge_state

    @property
    def writes(self):
        return self._areas[0].writes

    @property
    def cursor(self):
        return self._areas[0].cursor

    @property
    def matrix(self):
        """Estados apilados de B1, o columna con todos los coeficientes si hay puente B2."""
        if self._bridge is not None:
            return torch.cat([value.reshape(-1, 1) for value in self._matrices()])
        return torch.cat([area.matrix for area in self._areas])

    def _matrices(self):
        matrices = [area.matrix for area in self._areas]
        if self._bridge is not None:
            matrices.extend(self._bridge.matrices.unbind())
        return matrices

    @property
    def numeric_state_bytes(self):
        """Bytes de coeficientes e índices residentes, sin objetos Python ni temporales."""
        return (
            self._index.nbytes
            + sum(area.config.key_size * area.config.value_size * 8 for area in self._areas)
            + (0 if self._bridge is None else self._bridge.numeric_state_bytes)
        )

    def features(self, keys):
        """Proyección de cada área, [A, filas, 2dν], con norma uno o cero."""
        permuted = keys[:, self._index].permute(1, 0, 2)
        # El factor común de cada fila se cancela al normalizar.
        projected = _scaled_dpfp(permuted, self.layout.shifts)
        norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
        # Una clave sin pares vecinos del mismo signo no tiene proyección y no lee ni escribe.
        return torch.where(norms > 0, projected / torch.where(norms > 0, norms, 1.0), 0.0)

    def read_areas(self, keys, *, active=None):
        """Lectura de cada área, [A, filas, valor], sin efectos."""
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT:
            raise ValueError(f"La lectura necesita entre 1 y {MAX_COHORT} claves")
        _check_keys(keys, rows, self.config.key_size)
        mask = active_areas(active, self.layout.areas)
        features = self.features(keys)
        if self._bridge is not None:
            features = self._bridge.apply(features, active=mask)
        values = torch.stack(
            [area.read(part) for area, part in zip(self._areas, features, strict=True)]
        )
        return values if mask is None else torch.where(mask[:, None, None], values, 0.0)

    def read(self, keys, active=None):
        """Media de las áreas activas, de forma [filas, valor]. Sin `active` lee todas."""
        mask = active_areas(active, self.layout.areas)
        values = self.read_areas(keys, active=mask)
        if mask is None:
            return values.mean(dim=0)
        return values[mask].mean(dim=0)

    def _values(self, feedback, rows, order):
        """Valores que se escriben: los maduros o, en el control, barajados en la cohorte."""
        values = feedback.values
        if self.layout.labels == "mature":
            return values
        first = int(feedback.ids[order[0]])
        shuffle = np.random.Generator(np.random.PCG64([self.layout.seed, first])).permutation(rows)
        # El barajado se aplica en el orden canónico, así que no depende del orden de entrega.
        canonical = torch.tensor(order, dtype=torch.int64)
        shuffled = torch.empty_like(values)
        shuffled[canonical] = values[canonical[torch.from_numpy(shuffle)]]
        return shuffled

    def write(self, feedback, *, cutoff):
        """Escribir los resultados maduros de un evento en todas las áreas."""
        rows, order = canonical_order(
            feedback,
            cutoff,
            key_size=self.config.key_size,
            value_size=self.config.value_size,
            weighted=self.config.rule == "proximal",
            cursor=self.cursor,
        )
        if rows == 0:
            return self
        values = self._values(feedback, rows, order)
        features = self.features(feedback.keys)
        areas = tuple(
            area.write(replace(feedback, keys=part, values=values), cutoff=cutoff)
            for area, part in zip(self._areas, features, strict=True)
        )
        bridge = (
            None if self._bridge is None else self._bridge.write(features, feedback, cutoff=cutoff)
        )
        return AreaMemory(
            self.config, self.layout, areas, query_bridge=self.query_bridge, bridge_state=bridge
        )

    def export(self):
        payload = dict(
            schema_version=1,
            configuration=self.identity(),
            areas=[area.export() for area in self._areas],
            matrix_sha256=_digest(self._matrices()),
            writes=self.writes,
            cursor=list(self.cursor) if self.cursor is not None else None,
        )
        if self._bridge is not None:
            payload.update(schema_version=2, query_bridge=self._bridge.export())
        return payload

    def identity(self):
        identity = dict(rule=self.config.identity(), layout=self.layout.identity())
        if self.query_bridge is not None:
            identity["query_bridge"] = self.query_bridge.identity()
        return identity

    @classmethod
    def restore(cls, config, layout, payload, *, query_bridge=None):
        expected = {"schema_version", "configuration", "areas", "matrix_sha256", "writes", "cursor"}
        if query_bridge is not None:
            expected.add("query_bridge")
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("El estado exportado de las áreas no conserva todos sus campos")
        empty = cls(config, layout, query_bridge=query_bridge)
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != (1 if query_bridge is None else 2)
            or _canonical(payload["configuration"]) != _canonical(empty.identity())
        ):
            raise ValueError("El estado pertenece a otra memoria por áreas")
        if not isinstance(payload["areas"], list):
            raise ValueError("El estado exportado no conserva sus áreas")
        areas = tuple(
            AssociativeMemory.restore(empty.area_config, area) for area in payload["areas"]
        )
        bridge = None
        if query_bridge is not None:
            bridge = AreaQueryBridgeState.restore(
                replace(empty.area_config, value_size=empty.area_config.key_size),
                layout.areas,
                query_bridge,
                payload["query_bridge"],
            )
        memory = cls(config, layout, areas, query_bridge=query_bridge, bridge_state=bridge)
        cursor = payload["cursor"]
        if (
            memory.writes != payload["writes"]
            or memory.cursor != (tuple(cursor) if isinstance(cursor, list) else cursor)
            or memory.export()["matrix_sha256"] != payload["matrix_sha256"]
        ):
            raise ValueError("Las áreas no coinciden con su contador, cursor o huella")
        return memory
