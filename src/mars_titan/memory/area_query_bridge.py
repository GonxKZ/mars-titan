"""Mapas asociativos de consultas entre áreas, escritos solo con cohortes maduras (B2).

Cada arista del anillo a→(a+1) guarda la asociación entre los rasgos de la misma observación
en las dos áreas. No guarda otra copia del episodio ni usa su etiqueta financiera. Las
reglas de escritura son las de B1. La consulta se traduce antes de leer su corrección.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .associative_memory import (
    MAX_COHORT,
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
    _canonical,
    _check_keys,
    _finite,
    canonical_order,
)

KIND = "area_query_bridge_v1"
TARGETS = ("aligned", "shuffled_within_cohort")
# Dominio fijo de B2 para separar su flujo del barajado de etiquetas de B1.
SHUFFLE_DOMAIN = 0xB2


def unit_rows(values):
    """Norma uno o cero por fila, sin elevar al cuadrado valores extremos sin reescalar."""
    if not torch.isfinite(values).all():
        raise ValueError("Las consultas del puente contienen valores no finitos")
    maximum = values.abs().amax(dim=-1, keepdim=True)
    scaled = values / torch.where(maximum > 0, maximum, 1.0)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / torch.where(norm > 0, norm, 1.0)


def active_areas(value, count):
    """Máscara CPU común de áreas presentes, con al menos una disponible."""
    if value is None:
        return None
    mask = torch.as_tensor(value)
    if (
        mask.dtype != torch.bool
        or mask.shape != (count,)
        or mask.device.type != "cpu"
        or mask.layout != torch.strided
        or not mask.any()
    ):
        raise ValueError("Las áreas activas son una máscara booleana CPU con al menos una")
    return mask


def declaration():
    return dict(
        kind=KIND,
        requires="areas",
        gain=[0.0, 1.0],
        steps=[1, 4],
        targets=list(TARGETS),
        seed="uint32",
        topology="directed_ring",
        write_rule="same_as_area_memory",
    )


@dataclass(frozen=True)
class AreaQueryBridgeConfig:
    """Ganancia y pasos de traducción, distintos de las K lecturas del predictor."""

    gain: float = 0.25
    steps: int = 1
    targets: str = "aligned"
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "gain", _finite(self.gain, "La ganancia del puente", 0.0, 1.0))
        if type(self.steps) is not int or not 1 <= self.steps <= 4:
            raise ValueError("El puente necesita entre uno y cuatro pasos enteros")
        if self.targets not in TARGETS:
            raise ValueError("La correspondencia del puente debe ser alineada o barajada")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla del puente debe ser un entero uint32")

    def identity(self):
        return dict(
            asdict(self),
            kind=KIND,
            topology="source_a_to_target_a_plus_one_modulo_areas",
            update="synchronous_unit_convex_mix_of_current_query_and_translated_source",
            target="same_mature_observation_feature_in_the_next_area",
            label_values="not_used_by_the_query_map",
            shuffle="numpy_pcg64_domain_178_seed_first_canonical_id_and_source_area",
        )


class AreaQueryBridgeState:
    """Un mapa D×D por arista y un snapshot común durante todos los pasos de consulta."""

    def __init__(self, rule, areas, config, *, maps=None):
        if (
            type(rule) is not AssociativeMemoryConfig
            or rule.rule not in ("delta", "proximal")
            or rule.key_size != rule.value_size
            or type(areas) is not int
            or not 2 <= areas <= 8
            or type(config) is not AreaQueryBridgeConfig
        ):
            raise ValueError(
                "El puente necesita de dos a ocho áreas y mapas cuadrados identificados"
            )
        if maps is None:
            maps = tuple(AssociativeMemory(rule) for _ in range(areas))
        if (
            type(maps) is not tuple
            or len(maps) != areas
            or any(type(item) is not AssociativeMemory or item.config != rule for item in maps)
            or len({(item.writes, item.cursor) for item in maps}) != 1
        ):
            raise ValueError("Los mapas no comparten configuración, contador y cursor")
        self.rule, self.areas, self.config, self._maps = rule, areas, config, maps

    @property
    def writes(self):
        return self._maps[0].writes

    @property
    def cursor(self):
        return self._maps[0].cursor

    @property
    def matrices(self):
        """Copias de los mapas [áreas, D, D], ordenados por el área emisora."""
        return torch.stack([item.matrix for item in self._maps])

    @property
    def numeric_state_bytes(self):
        return self.areas * self.rule.key_size**2 * 8

    def _features(self, values, *, allow_empty=False):
        if (
            not isinstance(values, torch.Tensor)
            or values.layout != torch.strided
            or values.ndim != 3
            or values.shape[0] != self.areas
            or values.shape[2] != self.rule.key_size
            or not int(not allow_empty) <= values.shape[1] <= MAX_COHORT
        ):
            raise ValueError("Los rasgos del puente necesitan [áreas, filas, dimensión]")
        _check_keys(values.flatten(0, 1), self.areas * values.shape[1], self.rule.key_size)
        return values.shape[1]

    def apply(self, queries, *, active=None):
        """Traducir consultas con los mapas confirmados, sin modificar ningún estado."""
        self._features(queries)
        mask = active_areas(active, self.areas)
        current = queries if mask is None else torch.where(mask[:, None, None], queries, 0.0)
        if self.config.gain == 0:
            return current
        for _ in range(self.config.steps):
            messages = [
                item.read(current[a])
                if mask is None or (mask[a] and mask[(a + 1) % self.areas])
                else torch.zeros_like(current[a])
                for a, item in enumerate(self._maps)
            ]
            translated = torch.stack(messages).roll(shifts=1, dims=0)
            # Detectar overflow antes de mezclar o normalizar, incluso con ganancia pequeña.
            if not torch.isfinite(translated).all():
                raise ValueError("La traducción del puente produjo valores no finitos")
            current = unit_rows((1 - self.config.gain) * current + self.config.gain * translated)
        return current

    def write(self, features, feedback, *, cutoff):
        """Escribir correspondencias de rasgos de la misma cohorte madura, sin sus etiquetas."""
        rows = self._features(features, allow_empty=True)
        if not isinstance(feedback, MatureFeedback):
            raise ValueError("El puente necesita la procedencia de los resultados maduros")
        prototype = replace(feedback, keys=features[0], values=features[1])
        count, order = canonical_order(
            prototype,
            cutoff,
            key_size=self.rule.key_size,
            value_size=self.rule.value_size,
            weighted=self.rule.rule == "proximal",
            cursor=self.cursor,
        )
        if count != rows:
            raise ValueError("Los rasgos no corresponden a la misma cohorte madura")
        if rows == 0:
            return self
        canonical = torch.tensor(order, dtype=torch.int64)
        maps = []
        for source, previous in enumerate(self._maps):
            values = features[(source + 1) % self.areas]
            if self.config.targets == "shuffled_within_cohort":
                generator = np.random.Generator(
                    np.random.PCG64(
                        [SHUFFLE_DOMAIN, self.config.seed, int(feedback.ids[order[0]]), source]
                    )
                )
                permutation = torch.from_numpy(generator.permutation(rows))
                shuffled = torch.empty_like(values)
                shuffled[canonical] = values[canonical[permutation]]
                values = shuffled
            maps.append(
                previous.write(
                    replace(feedback, keys=features[source], values=values), cutoff=cutoff
                )
            )
        return AreaQueryBridgeState(self.rule, self.areas, self.config, maps=tuple(maps))

    def identity(self):
        return dict(
            configuration=self.config.identity(), areas=self.areas, rule=self.rule.identity()
        )

    def export(self):
        return dict(
            schema_version=1,
            configuration=self.identity(),
            maps=[item.export() for item in self._maps],
        )

    @classmethod
    def restore(cls, rule, areas, config, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "configuration",
            "maps",
        }:
            raise ValueError("El estado del puente no conserva sus campos")
        empty = cls(rule, areas, config)
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or _canonical(payload["configuration"]) != _canonical(empty.identity())
            or not isinstance(payload["maps"], list)
        ):
            raise ValueError("El estado pertenece a otro puente de consultas")
        maps = tuple(AssociativeMemory.restore(rule, item) for item in payload["maps"])
        return cls(rule, areas, config, maps=maps)
