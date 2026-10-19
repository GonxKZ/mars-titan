"""Controles de umbral y asignación por hash para rutas de refinamiento."""

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

import torch
from torch import nn

from .config import bounded_integer, canonical, require_identity
from .financial_inputs import FLOW_ID
from .refinement_routes import ROUTES
from .state import check_finite

FEATURES = (
    "base_prediction",
    "first_prediction",
    "first_prediction_change",
    "first_interval_width",
    "attention_entropy",
    "attention_maximum",
    "read_rms",
    "state_change_rms",
    "memory_present",
)


class ThresholdRoutePolicy(nn.Module):
    """Continuar si un rasgo de K1 supera el umbral. No estima ganancias."""

    def __init__(self, *, feature_name, cutoff, route):
        super().__init__()
        if type(feature_name) is not str or feature_name not in FEATURES:
            raise ValueError("El rasgo del umbral no está declarado")
        if type(route) is not str or route not in ROUTES[1:]:
            raise ValueError("El umbral necesita una ruta posterior a K1")
        if type(cutoff) not in (int, float):
            raise ValueError("El umbral debe ser un número finito")
        try:
            cutoff = float(cutoff)
        except OverflowError as error:
            raise ValueError("El umbral debe ser finito en FP64") from error
        if not math.isfinite(cutoff):
            raise ValueError("El umbral debe ser finito en FP64")
        self.feature_name, self.cutoff, self.route = feature_name, cutoff, route
        self.eval()

    def get_extra_state(self):
        return dict(
            schema_version=1,
            policy="strict_feature_threshold_after_k1_v1",
            feature_name=self.feature_name,
            cutoff=self.cutoff,
            route=self.route,
            comparison="fp64_strict_greater_than",
            features=list(FEATURES),
            routes=list(ROUTES),
        )

    def set_extra_state(self, value):
        require_identity(value, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        supplied = {
            name.removeprefix(prefix): value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        if set(supplied) != {"_extra_state"}:
            raise ValueError("El checkpoint no conserva el contrato completo del umbral")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, features):
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape[1] != len(FEATURES)
            or not 1 <= len(features) <= 256
            or features.dtype not in (torch.float32, torch.float64)
            or features.layout != torch.strided
        ):
            raise ValueError("Los rasgos del umbral no corresponden al contrato")
        check_finite(features, "Los rasgos del umbral")
        selected = features[:, FEATURES.index(self.feature_name)].to(torch.float64) > self.cutoff
        return torch.where(selected, ROUTES.index(self.route), 0)


@dataclass(frozen=True)
class HashedRoutes:
    """Asignación pseudoaleatoria determinista por emisión, sin usar ningún RNG.

    Los recuentos se fijan en validación. Describen las proporciones de la CDF,
    no garantizan las frecuencias de un conjunto finito de emisiones.
    """

    counts: tuple[int, ...]
    seed: int

    def __post_init__(self):
        if (
            type(self.counts) is not tuple
            or len(self.counts) != len(ROUTES)
            or any(type(value) is not int or not 0 <= value < 2**63 for value in self.counts)
            or sum(self.counts) <= 0
        ):
            raise ValueError("Se necesitan seis recuentos int64 no negativos con suma positiva")
        bounded_integer(self.seed, "semilla", 0, 2**32 - 1)

    def identity(self):
        return dict(
            schema_version=1,
            schedule="hashed_validation_route_frequencies_v1",
            counts=list(self.counts),
            seed=self.seed,
            routes=list(ROUTES),
            hash="sha256_first64_big_endian_of_canonical_seed_flow_time",
            cdf="floor_cumulative_count_times_2pow64_over_total",
        )

    def choices(self, flow_ids, prediction_at):
        if (
            type(flow_ids) not in (tuple, list)
            or type(prediction_at) not in (tuple, list)
            or not 1 <= len(flow_ids) <= 256
            or len(prediction_at) != len(flow_ids)
        ):
            raise ValueError("La asignación necesita entre 1 y 256 flujos e instantes alineados")
        for flow, at in zip(flow_ids, prediction_at, strict=True):
            if type(flow) is not str or FLOW_ID.fullmatch(flow) is None:
                raise ValueError("El flujo debe identificar mercado y activo")
            bounded_integer(at, "instante UTC en microsegundos", 0, 2**63 - 1)
        total = sum(self.counts)
        boundaries = tuple(cumulative * 2**64 // total for cumulative in accumulate(self.counts))
        values = (
            int.from_bytes(
                hashlib.sha256(canonical([self.seed, flow, at]).encode()).digest()[:8], "big"
            )
            for flow, at in zip(flow_ids, prediction_at, strict=True)
        )
        return tuple(bisect_right(boundaries, value) for value in values)
