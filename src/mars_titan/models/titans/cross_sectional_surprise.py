"""Sorpresa relativa transversal ortogonalizada (CSOS, `cross_sectional_surprise_v1`).

En cada actualización de `NeuralMemory` sobre los N_t activos de un mercado m en la sesión t,
separa el gradiente asociativo G_{i,t}^{(l)} = ∂ℓ(M_{i,t-1}; k_{i,t}, v_{i,t}) / ∂W_l en choque
común de mercado y componente idiosincrásica, y modula la puerta de escritura θ con la sorpresa
logarítmica estandarizada transversalmente:

    Ḡ_{m,t}^{(l)} = (1 / N_t) Σ_{j ∈ S_{m,t}} G_{j,t}^{(l)}
    G̃_{i,t}^{(l)} = G_{i,t}^{(l)} − λ_cs · Ḡ_{m,t}^{(l)},      λ_cs = tanh(z_cs) ∈ (−1, 1)

    s_{i,t} = log(e_{i,t} + ε),      e_{i,t} = Σ_c elements_{i,t,c} ≥ 0
    μ_{m,t} = (1 / N_t) Σ_{j ∈ S_{m,t}} s_{j,t}
    σ²_{m,t} = (1 / N_t) Σ_{j ∈ S_{m,t}} (s_{j,t} − μ_{m,t})²
    u_{i,t} = (s_{i,t} − μ_{m,t}) / √(σ²_{m,t} + ε)
    θ_{i,t} = θ_max · σ(z_{θ,i,t} + b · tanh(w_rel · u_{i,t}))

La parametrización λ_cs = tanh(z_cs) con z_cs = 0 y w_rel = 0 arranca exactamente en cero
(idéntica al control no fusionado en el paso 0) sin anular la derivada inicial (λ'(0) = 1 y
∂δ_θ/∂w_rel|_{0} = b · u_{i,t}): cualquier función C¹ acotada en [0, 1] con f(0) = 0 tendría
derivada nula en cero por el teorema de Fermat, y clamp(z, 0, 1) moriría permanentemente si el
primer paso desplazase z < 0.

Solo actúa sobre las rutas no fusionadas (`analytic_v1` y `autograd_v1`): es incompatible con
`fused_v1` porque el kernel Triton calcula y escribe por flujo en registros sin sincronización
transversal.

Control placebo integrado (`mode="shuffled_session"`): conserva los mismos dos parámetros
(`lambda_logit`, `write_weight`), misma inicialización nula y mismas formas y normas de
perturbación, pero destruye la alineación transversal dentro de cada sesión (m, t) mediante un
desarreglo determinista (permutación sin puntos fijos) sobre u_{i,t} y signos de Rademacher
equilibrados r_{i,t} ∈ {−1, +1} sobre Ḡ_{m,t}^{(l)}, sin consumir ningún generador global.
"""

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .state import check_finite, require_true

CROSS_SECTIONAL_SURPRISE = "cross_sectional_surprise_v1"
MODES = ("same_session", "shuffled_session")


def _require_identity(actual: object, expected: dict) -> None:
    try:
        left = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False)
        right = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("La configuración no tiene un contrato JSON válido") from error
    if left != right:
        raise ValueError("La configuración no coincide con el contrato del módulo")


def _finite_float(
    value: object, low: float, high: float, message: str, *, strict_low: bool = False
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= low if strict_low else value < low)
        or value > high
    ):
        raise ValueError(message)
    return float(value)


@dataclass(frozen=True)
class CrossSectionalSurprise:
    """Contrato identificado de CSOS para `MemoryConfig` y `FinancialConfig`."""

    mode: str = "same_session"
    bound: float = 1.0
    eps: float = 1e-6
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("mode de CSOS debe ser same_session o shuffled_session")
        bound = _finite_float(
            self.bound,
            0.0,
            4.0,
            "bound de CSOS debe ser finito y pertenecer a (0, 4]",
            strict_low=True,
        )
        eps = _finite_float(
            self.eps,
            1e-8,
            1e-1,
            "eps de CSOS debe ser finito y pertenecer a [1e-8, 1e-1]",
        )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed de CSOS debe ser un entero en [0, 2^32 - 1]")
        object.__setattr__(self, "bound", bound)
        object.__setattr__(self, "eps", eps)

    def identity(self) -> dict:
        same = self.mode == "same_session"
        return {
            "schema_version": 1,
            "name": CROSS_SECTIONAL_SURPRISE,
            "mode": self.mode,
            "role": "cross_sectional_variant" if same else "shuffled_session_placebo",
            "groups": "same_market_and_prediction_instant_in_the_block",
            "orthogonalization": (
                "subtract_tanh_lambda_times_group_mean_gradient"
                if same
                else "subtract_tanh_lambda_times_balanced_random_sign_group_mean_gradient"
            ),
            "write_gate": (
                "bound_times_tanh_of_learned_weight_times_standardized_log_surprise"
                if same
                else "bound_times_tanh_of_learned_weight_times_deranged_standardized_log_surprise"
            ),
            "lambda_parametrization": "tanh_of_zero_initialized_scalar",
            "bound": self.bound,
            "eps": self.eps,
            "seed": self.seed,
            "single_row_group_policy": "identity_gradient_and_zero_shift",
            "initialization": "zero_lambda_and_zero_write_weight_without_rng",
            "state": "none_modifies_neural_memory_write",
            "shared_memory": "unsupported",
        }


def parse_cross_sectional_surprise(raw: object) -> CrossSectionalSurprise | None:
    """Aceptar CrossSectionalSurprise, su diccionario JSON o None."""
    if raw is None:
        return None
    if type(raw) is CrossSectionalSurprise:
        raw.__post_init__()
        return raw
    allowed = {"mode", "bound", "eps", "seed"}
    if isinstance(raw, dict) and "mode" in raw and set(raw) <= allowed:
        return CrossSectionalSurprise(**raw)
    raise ValueError(
        "memory_cross_sectional_surprise debe ser CrossSectionalSurprise, un diccionario con "
        "mode (y opcionalmente bound, eps, seed) o None"
    )


@dataclass(frozen=True)
class SessionGroups:
    """Agrupación causal de filas por (mercado, instante de predicción) dentro de un bloque."""

    markets: tuple[str, ...]
    prediction_at: tuple[int, ...]
    flow_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.markets)
        if (
            type(self.markets) is not tuple
            or type(self.prediction_at) is not tuple
            or type(self.flow_ids) is not tuple
            or size == 0
            or len(self.prediction_at) != size
            or len(self.flow_ids) != size
        ):
            raise ValueError("SessionGroups requiere tuplas no vacías de igual longitud")

    def grouped(self) -> list[tuple[tuple[str, int], list[int]]]:
        """Pares ((mercado, instante), filas) en orden de primera aparición."""
        sessions: dict[tuple[str, int], list[int]] = {}
        for row, key in enumerate(zip(self.markets, self.prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())


def session_groups(flow_ids: tuple[str, ...], prediction_at: tuple[int, ...]) -> SessionGroups:
    """Construir los grupos de sesión a partir de los flujos e instantes de un bloque."""
    markets = tuple(flow.split("/", 1)[0] for flow in flow_ids)
    return SessionGroups(
        markets=markets, prediction_at=tuple(prediction_at), flow_ids=tuple(flow_ids)
    )


def _index_tensor(values: list[int], device: torch.device) -> Tensor:
    if device.type == "cuda":
        return torch.tensor(values, dtype=torch.int64, pin_memory=True).to(
            device=device, non_blocking=True
        )
    return torch.tensor(values, dtype=torch.int64, device=device)


def _mix31(value: Tensor) -> Tensor:
    mask = 0x7FFF_FFFF
    mixed = value & mask
    mixed = ((mixed ^ (mixed >> 16)) * 0x45D9F3B) & mask
    mixed = ((mixed ^ (mixed >> 16)) * 0x45D9F3B) & mask
    return (mixed ^ (mixed >> 16)) & mask


def _placebo_plan(
    seed: int,
    group_key: tuple[str, int],
    members: list[int],
    groups: SessionGroups | None,
    token_index: int,
    steps: Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Desarreglo sin puntos fijos y signos de Rademacher equilibrados por grupo.

    Con `groups` presente se calcula de forma determinista con blake2b sobre los identificadores
    canónicos de cada fila (sin tocar ningún generador global), por lo que es equivariante al
    orden de las filas. Sin `groups`, se calcula íntegramente con operaciones tensoriales en
    `device` sin `.tolist()` ni sincronizaciones anfitrión-dispositivo.
    """
    size = len(members)
    if groups is None:
        rows = torch.arange(size, dtype=torch.int64, device=device)
        step_col = (
            torch.full((size,), token_index, dtype=torch.int64, device=device)
            if steps is None
            else steps.detach().to(dtype=torch.int64)
        )
        seed_mix = (seed & 0x7FFF_FFFF) ^ ((token_index * 0x9E3779B) & 0x7FFF_FFFF)
        key = _mix31(rows ^ _mix31((step_col & 0x7FFF_FFFF) ^ seed_mix))
        order = torch.argsort((key << 31) | rows, stable=True)
        ranks = torch.argsort(order, stable=True)
        perm = order.index_select(0, (ranks + 1) % size)
        signs = torch.where((ranks & 1) == 0, 1.0, -1.0).to(dtype=dtype)
        return perm, signs
    group_tag = f"{group_key[0]}@{group_key[1]}#{token_index}"
    row_tags = [
        f"{groups.flow_ids[row]}@{groups.prediction_at[row]}#{token_index}" for row in members
    ]
    entries = []
    for local_pos, row_tag in enumerate(row_tags):
        digest = hashlib.blake2b(f"{seed}/{group_tag}/{row_tag}".encode(), digest_size=8).digest()
        entries.append((int.from_bytes(digest, "little"), row_tag, local_pos))
    entries.sort(key=lambda item: (item[0], item[1]))
    order = [item[2] for item in entries]
    perm_list = [0] * size
    signs_list = [0.0] * size
    for rank, local_pos in enumerate(order):
        perm_list[local_pos] = order[(rank + 1) % size]
        signs_list[local_pos] = 1.0 if rank % 2 == 0 else -1.0
    perm = _index_tensor(perm_list, device)
    if device.type == "cuda":
        signs = torch.tensor(signs_list, dtype=dtype, pin_memory=True).to(
            device=device, non_blocking=True
        )
    else:
        signs = torch.tensor(signs_list, dtype=dtype, device=device)
    return perm, signs


class CrossSectionalSurpriseModule(nn.Module):
    """Módulo con los dos escalares aprendidos de CSOS, inicializados en cero sin RNG."""

    def __init__(
        self, config: CrossSectionalSurprise, *, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        if type(config) is not CrossSectionalSurprise:
            raise ValueError("CSOS necesita una configuración CrossSectionalSurprise")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("CSOS admite precisión FP32 o FP64")
        self.config = config
        self.lambda_logit = nn.Parameter(torch.zeros((), dtype=dtype))
        self.write_weight = nn.Parameter(torch.zeros((), dtype=dtype))

    def get_extra_state(self) -> dict:
        return {"configuration": self.config.identity()}

    def set_extra_state(self, state: object) -> None:
        _require_identity(state, self.get_extra_state())

    def _transform_group(
        self,
        group_grads: tuple[Tensor, ...],
        group_log_losses: Tensor,
        lam: Tensor,
        write_weight: Tensor,
        group_key: tuple[str, int],
        members: list[int],
        groups: SessionGroups | None,
        token_index: int,
        steps: Tensor | None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        centered = group_log_losses - group_log_losses.mean()
        variance = centered.square().mean()
        u = centered * torch.rsqrt(variance + self.config.eps)
        if self.config.mode == "same_session":
            out_grads = tuple(g - lam * g.mean(dim=0, keepdim=True) for g in group_grads)
            u_effective = u
        else:
            perm, signs = _placebo_plan(
                self.config.seed,
                group_key,
                members,
                groups,
                token_index,
                steps,
                group_log_losses.device,
                group_log_losses.dtype,
            )
            signs_view = signs.view(-1, 1, 1)
            out_grads = tuple(
                g - (lam * signs_view) * g.mean(dim=0, keepdim=True) for g in group_grads
            )
            u_effective = u.index_select(0, perm)
        shift = (self.config.bound * torch.tanh(write_weight * u_effective)).unsqueeze(-1)
        return out_grads, shift

    def _compute(
        self,
        gradients: tuple[Tensor, ...],
        elements: Tensor,
        groups: SessionGroups | None,
        lambda_logit: Tensor,
        write_weight: Tensor,
        *,
        token_index: int,
        steps: Tensor | None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        batch = elements.shape[0]
        lam = torch.tanh(lambda_logit)
        losses = elements.sum(dim=-1)
        log_losses = torch.log(losses + self.config.eps)
        buckets = [(("", 0), list(range(batch)))] if groups is None else groups.grouped()
        if len(buckets) == 1:
            key, members = buckets[0]
            if batch < 2:
                return gradients, torch.zeros(
                    batch, 1, dtype=elements.dtype, device=elements.device
                )
            return self._transform_group(
                gradients, log_losses, lam, write_weight, key, members, groups, token_index, steps
            )
        out_grads = list(gradients)
        write_shift = torch.zeros(batch, 1, dtype=elements.dtype, device=elements.device)
        for key, members in buckets:
            if len(members) < 2:
                continue
            index = _index_tensor(members, elements.device)
            sub_grads = tuple(g.index_select(0, index) for g in gradients)
            sub_log_losses = log_losses.index_select(0, index)
            transformed, sub_shift = self._transform_group(
                sub_grads,
                sub_log_losses,
                lam,
                write_weight,
                key,
                members,
                groups,
                token_index,
                steps,
            )
            for layer, value in enumerate(transformed):
                out_grads[layer] = out_grads[layer].index_copy(0, index, value)
            write_shift = write_shift.index_copy(0, index, sub_shift)
        return tuple(out_grads), write_shift

    def forward(
        self,
        gradients: tuple[Tensor, ...],
        elements: Tensor,
        groups: SessionGroups | None = None,
        *,
        token_index: int = 0,
        steps: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        reference = self.lambda_logit
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("CSOS admite precisión FP32 o FP64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("CSOS exige FP32 o FP64 explícitos, sin autocast")
        if (
            not isinstance(gradients, tuple)
            or not gradients
            or not isinstance(elements, Tensor)
            or elements.layout != torch.strided
            or elements.ndim != 2
            or elements.shape[0] == 0
            or elements.dtype != reference.dtype
            or elements.device != reference.device
        ):
            raise ValueError("CSOS necesita gradientes [B, D, D] y elementos [B, D] compatibles")
        batch = elements.shape[0]
        for grad in gradients:
            if (
                not isinstance(grad, Tensor)
                or grad.layout != torch.strided
                or grad.ndim != 3
                or grad.shape[0] != batch
                or grad.dtype != reference.dtype
                or grad.device != reference.device
            ):
                raise ValueError("Cada gradiente de CSOS debe tener forma [B, D, D] y mismo tipo")
            check_finite(grad, "El gradiente asociativo de CSOS")
        check_finite(elements, "La pérdida asociativa de CSOS")
        require_true((elements >= 0).all(), "La pérdida asociativa de CSOS debe ser no negativa")
        if groups is not None:
            if type(groups) is not SessionGroups or len(groups.markets) != batch:
                raise ValueError(
                    "session_groups debe corresponder exactamente a las filas del lote"
                )
        out_grads, write_shift = self._compute(
            gradients,
            elements,
            groups,
            self.lambda_logit,
            self.write_weight,
            token_index=token_index,
            steps=steps,
        )
        for grad in out_grads:
            check_finite(grad, "El gradiente ortogonalizado de CSOS")
        check_finite(write_shift, "El desplazamiento de escritura de CSOS")
        return out_grads, write_shift
