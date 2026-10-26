"""Protocolo B4 de lesiones de lectura sobre una misma trayectoria de memoria."""

import hashlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch

from .area_memory import AreaMemory
from .associative_memory import MAX_COHORT, MatureCorrection, _check_keys, _check_matrix
from .multiscale_mixture import MultiscaleMemory, _clock
from .role_binding import RoleBoundMemory

KIND = "lesion_protocol_v1"
MAX_CASES = 24


@dataclass(frozen=True)
class LesionProtocol:
    """Fracciones y semillas fijadas antes de observar resultados del tramo."""

    seeds: tuple[int, ...] = (0, 1)
    fractions: tuple[tuple[int, int], ...] = ((1, 4), (1, 2), (3, 4))

    def __post_init__(self):
        if (
            not isinstance(self.seeds, (list, tuple))
            or not 1 <= len(self.seeds) <= 4
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
            or not isinstance(self.fractions, (list, tuple))
            or not 1 <= len(self.fractions) <= 4
            or 2 + len(self.seeds) * len(self.fractions) * 3 > MAX_CASES
        ):
            raise ValueError("Las semillas y fracciones de B4 deben ser distintas y acotadas")
        fractions = []
        for pair in self.fractions:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(type(value) is not int for value in pair)
                or not 0 < pair[0] < pair[1] <= 64
            ):
                raise ValueError("B4 declara fracciones racionales entre cero y uno")
            fractions.append(Fraction(*pair))
        if fractions != sorted(set(fractions)):
            raise ValueError("Las fracciones de B4 deben estar ordenadas y ser distintas")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(
            self, "fractions", tuple((f.numerator, f.denominator) for f in fractions)
        )

    def identity(self):
        return dict(
            kind=KIND,
            seeds=list(self.seeds),
            fractions=[list(pair) for pair in self.fractions],
            generator="numpy_pcg64_180_axis_seed_area_fixed_four_words",
            selection="nested_prefix_of_fixed_permutation_floor_fraction_times_dimension",
            scope="read_queries_and_bridge_endpoints_without_writes_or_model_adaptation",
        )

    def bind(self, correction):
        return LesionReadout(self, correction)


class LesionReadout:
    """Máscaras inmutables por configuración. Los tensores devueltos son copias."""

    def __init__(self, protocol, correction):
        if (
            type(protocol) is not LesionProtocol
            or type(correction) is not MatureCorrection
            or correction.areas is None
            or correction.memory.rule not in ("delta", "proximal")
        ):
            raise ValueError("B4 necesita la memoria por áreas, incluido su control monolítico")
        self.protocol, self.correction = protocol, correction
        areas = correction.areas.areas
        width = correction.areas.feature_size(correction.memory.key_size)
        self.areas, self.width = areas, width
        cases = [
            ("intact", None, None, "active_mean"),
            ("zero", (False,) * areas, None, "fixed_mean"),
        ]
        for seed in protocol.seeds:
            area_order = np.random.Generator(np.random.PCG64([0xB4, 0, seed, 0])).permutation(areas)
            orders = [
                np.random.Generator(np.random.PCG64([0xB4, 1, seed, area])).permutation(width)
                for area in range(areas)
            ]
            for numerator, denominator in protocol.fractions:
                suffix = f"s{seed}_{numerator}of{denominator}"
                active = np.ones(areas, dtype=np.bool_)
                active[area_order[: numerator * areas // denominator]] = False
                for reduction in ("active_mean", "fixed_mean"):
                    cases.append(
                        (f"area_{suffix}_{reduction}", tuple(active.tolist()), None, reduction)
                    )
                coordinates = np.ones((areas, width), dtype=np.bool_)
                for area, order in enumerate(orders):
                    coordinates[area, order[: numerator * width // denominator]] = False
                cases.append(
                    (
                        f"coordinate_{suffix}",
                        None,
                        tuple(tuple(row) for row in coordinates.tolist()),
                        "active_mean",
                    )
                )
        self._cases = tuple(cases)
        self.names = tuple(case[0] for case in cases)

    def masks(self):
        return tuple(
            dict(
                area=torch.tensor(active, dtype=torch.bool)
                if active is not None
                else torch.ones(self.areas, dtype=torch.bool),
                coordinate=torch.tensor(coordinates, dtype=torch.bool)
                if coordinates is not None
                else torch.ones(self.areas, self.width, dtype=torch.bool),
            )
            for _, active, coordinates, _ in self._cases
        )

    def identity(self):
        return dict(
            protocol=self.protocol.identity(),
            correction=self.correction.identity(),
            area_count=self.areas,
            coordinate_width=self.width,
            cases=[
                dict(
                    name=name,
                    reduction=reduction,
                    active_areas=int(masks["area"].sum()),
                    active_coordinates=masks["coordinate"].sum(dim=1).tolist(),
                    mask_sha256=hashlib.sha256(
                        masks["area"].numpy().tobytes() + masks["coordinate"].numpy().tobytes()
                    ).hexdigest(),
                )
                for (name, _, _, reduction), masks in zip(self._cases, self.masks(), strict=True)
            ],
        )

    def _matches(self, memory):
        correction = self.correction
        if correction.multiscale is not None:
            return (
                type(memory) is MultiscaleMemory
                and memory.correction == correction._unmixed()
                and memory.mixture == correction.multiscale
            )
        if correction.role_binding is not None:
            return type(memory) is RoleBoundMemory and memory.correction == correction
        return (
            type(memory) is AreaMemory
            and memory.config == correction.memory
            and memory.layout == correction.areas
            and memory.query_bridge == correction.query_bridge
        )

    def read(self, memory, keys, *, at, baseline=None):
        """Correcciones [filas, condiciones], calculadas antes de recibir la etiqueta."""
        _clock(at)
        rows = keys.shape[0] if isinstance(keys, torch.Tensor) and keys.ndim == 2 else -1
        if not 1 <= rows <= MAX_COHORT or not self._matches(memory):
            raise ValueError("La lectura B4 no corresponde a la configuración y cohorte declaradas")
        _check_keys(keys, rows, self.correction.memory.key_size)
        timed = dict(at=at) if self.correction.multiscale is not None else {}
        baseline = memory.read(keys, **timed) if baseline is None else baseline
        _check_matrix(baseline, "La corrección original de B4", rows, 1)
        values = [baseline, torch.zeros_like(baseline)]
        for _, active, coordinates, reduction in self._cases[2:]:
            values.append(
                memory.read_diagnostic(
                    keys, active=active, coordinates=coordinates, reduction=reduction, **timed
                )
            )
        result = torch.cat(values, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Una condición de B4 produjo valores no finitos")
        return result
