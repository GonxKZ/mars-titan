"""Consulta de tres bancos históricos y contrato explícito de sus gradientes de escritura."""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from .state import check_finite, deferred_checks

BANK_READOUT = "bank_readout_v1"


@dataclass(frozen=True)
class BankReadoutConfig:
    routing: str = "query"
    temperature: float = 1.0
    available: tuple[bool, bool, bool] = (True, True, True)
    write_gradient: str = "common_translation"

    def __post_init__(self):
        if type(self.routing) is not str or self.routing not in ("query", "uniform", "constant"):
            raise ValueError("La lectura de bancos admite query, uniform o constant")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 1e-3 <= self.temperature <= 100
        ):
            raise ValueError("La temperatura debe ser finita y pertenecer a [0.001, 100]")
        if (
            type(self.available) not in (list, tuple)
            or len(self.available) != 3
            or any(type(v) is not bool for v in self.available)
        ):
            raise ValueError("La disponibilidad necesita tres booleanos")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "available", tuple(self.available))
        if type(self.write_gradient) is not str or self.write_gradient not in (
            "common_translation",
            "mean_partial",
            "per_bank",
        ):
            raise ValueError("La escritura admite common_translation, mean_partial o per_bank")

    def identity(self):
        result = dict(
            schema_version=1,
            kind=BANK_READOUT,
            routing=self.routing,
            temperature=self.temperature,
            available=list(self.available),
            regions=["level_1", "level_2", "level_3"],
            mixture="complete_function_outputs",
            availability="read_only_not_erasure",
            associative_gradient="common_translation_sum_of_partials",
        )
        if self.write_gradient != "common_translation":
            result.update(
                schema_version=2,
                write_gradient=self.write_gradient,
                associative_gradient=(
                    "per_bank_partials_of_mixture_loss"
                    if self.write_gradient == "per_bank"
                    else "orthogonal_projection_to_equal_bank_gradients"
                ),
                momentum="three_states_bank_major_then_layer",
            )
        return result


def _unit(value):
    """Normalizar sin desbordar una suma de cuadrados finita evitable."""
    scaled = value / value.abs().amax(-1, keepdim=True).clamp_min(1)
    return scaled / scaled.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()


class BankReadout(nn.Module):
    """Rutas por consulta, sin comunicación entre flujos, tokens o sesiones."""

    def __init__(self, dim, config, *, seed, device="cpu", dtype=torch.float32):
        super().__init__()
        from .config import bounded_integer

        bounded_integer(dim, "dimensión de bancos", 1, 512)
        bounded_integer(seed, "semilla de bancos", 0, 2**63 - 1)
        if type(config) is not BankReadoutConfig or dtype not in (torch.float32, torch.float64):
            raise ValueError("La lectura necesita configuración y precisión explícitas")
        self.dim, self.config = dim, config
        derived = int.from_bytes(
            hashlib.sha256(f"{BANK_READOUT}/{seed}".encode()).digest()[:7], "little"
        )
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derived)
            self.addresses = nn.Parameter(torch.randn(3, dim, dtype=dtype))
            self.register_buffer("available", torch.tensor(config.available, dtype=torch.bool))
        self.to(device=device)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), dim=self.dim, max_queries=16384)

    def operation_counts(self, rows, *, mode, depth):
        """Trabajo de MAC con un token por observación, no FLOPs ni lanzamientos."""
        from .config import bounded_integer

        bounded_integer(rows, "filas del trabajo de bancos", 0, 16384)
        bounded_integer(depth, "capas de memoria", 1, 2)
        if mode not in ("disabled", "frozen", "online"):
            raise ValueError("El contador necesita un modo de memoria identificado")
        calls = 3 if mode == "online" else 2 if mode == "frozen" else 0
        result = dict(
            bank_function_evaluations=3 * rows * calls,
            bank_query_routes=rows * calls,
            bank_matrix_updates=3 * depth * rows if mode == "online" else 0,
        )
        if self.config.write_gradient != "common_translation":
            result.update(
                bank_gradient_matrices=3 * depth * rows if mode == "online" else 0,
                bank_momentum_updates=3 * depth * rows if mode == "online" else 0,
            )
        return result

    def set_extra_state(self, state):
        from .config import require_identity

        require_identity(state, self.get_extra_state())

    def validate_payload(self, state_dict, prefix):
        from .config import require_identity

        expected = self.state_dict()
        supplied = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if set(supplied) != set(expected):
            raise ValueError("El checkpoint no conserva el lector de bancos")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        for name in ("addresses", "available"):
            other, value = supplied[name], expected[name]
            if (
                not isinstance(other, torch.Tensor)
                or other.shape != value.shape
                or other.dtype != value.dtype
                or other.layout != torch.strided
            ):
                raise ValueError("El checkpoint no conserva formas y precisión del lector")
            check_finite(other, "El checkpoint de bancos")
        if not torch.equal(supplied["available"].cpu(), self.available.cpu()):
            raise ValueError("El checkpoint cambió los bancos admisibles")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.validate_payload(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def probabilities(self, query):
        if (
            not isinstance(query, torch.Tensor)
            or query.layout != torch.strided
            or query.ndim != 3
            or query.shape[-1] != self.dim
            or not 1 <= query.shape[0] * query.shape[1] <= 16384
            or query.dtype not in (torch.float32, torch.float64)
            or query.dtype != self.addresses.dtype
            or query.device != self.addresses.device
        ):
            raise ValueError("La consulta no corresponde al lector de bancos")
        if torch.is_autocast_enabled(query.device.type):
            raise ValueError("El lector exige FP32 o FP64 explícitos")
        with deferred_checks():
            check_finite(query, "La consulta de bancos")
            check_finite(self.addresses, "Las direcciones de bancos")
            direction = _unit(query)
            if self.config.routing == "constant":
                direction = torch.ones_like(direction) / math.sqrt(self.dim)
            logits = direction @ _unit(self.addresses).T / self.config.temperature
            check_finite(logits, "Los logits de bancos")
            if self.config.routing == "uniform":
                logits = logits * 0
            active = any(self.config.available)
            allowed = self.available if active else torch.ones_like(self.available)
            probabilities = torch.softmax(logits.masked_fill(~allowed, -torch.inf), -1)
            if not active:
                probabilities = probabilities * 0
            check_finite(probabilities, "Las probabilidades de bancos")
        return probabilities

    def forward(self, query, outputs):
        probabilities = self.probabilities(query)
        if (
            not isinstance(outputs, torch.Tensor)
            or outputs.layout != torch.strided
            or outputs.shape != (*query.shape[:2], 3, self.dim)
            or outputs.dtype != query.dtype
            or outputs.device != query.device
        ):
            raise ValueError("Las salidas no corresponden a los tres bancos")
        with deferred_checks():
            check_finite(outputs, "Las lecturas de todos los bancos")
            result = (outputs * probabilities.unsqueeze(-1)).sum(2)
            check_finite(result, "La mezcla de bancos")
        return result
