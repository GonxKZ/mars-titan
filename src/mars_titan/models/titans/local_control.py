"""Diagnóstico local de la transición rápida completa mediante una compresión fija.

El radio comprimido no certifica el Jacobiano completo, sus potencias ni la
estabilidad global. Las reevaluaciones no publican ni modifican el estado.
"""

import hashlib
import math
import re
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.cm.numerical_radius import (
    NumericalRadiusEstimates,
    numerical_radius_estimates,
    radius_penalty,
)

from .config import MACConfig, bounded_integer, canonical, require_identity
from .mac import TitansMAC
from .state import check_differentiable, check_finite


@dataclass(frozen=True)
class MACProjectionConfig:
    mode: str = "disabled"
    rank: int = 4
    frequency: int = 64
    seed: int = 91
    grid_size: int = 64
    threshold: float = 1.0
    weight: float = 0.0
    max_flows: int = 1
    max_estimated_bytes: int = 128 * 1024**2

    def __post_init__(self):
        if self.mode not in ("disabled", "diagnostic", "penalty"):
            raise ValueError("El modo del control proyectado no es válido")
        for name, minimum, maximum in (
            ("rank", 1, 4),
            ("frequency", 1, 2**31 - 1),
            ("seed", 0, 2**63 - 1),
            ("grid_size", 2, 128),
            ("max_flows", 1, 16),
            ("max_estimated_bytes", 1, 512 * 1024**2),
        ):
            bounded_integer(getattr(self, name), name, minimum, maximum)
        for name in ("threshold", "weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} debe ser finito y no negativo")
            object.__setattr__(self, name, float(value))
        if (self.mode == "penalty") != (self.weight > 0):
            raise ValueError("Solo penalty admite y exige un peso positivo")

    def identity(self):
        return dict(
            **asdict(self),
            schema_version=1,
            state_layout="weights_then_momentum_row_major",
            local_state="detached_snapshot",
            token_gradient="preserved_when_differentiable_penalty",
            coordinate_metric="unscaled_euclidean",
            derivative="autograd_functional_jvp",
            attention_backend="math",
            basis="cpu_float64_qr_positive_diagonal_then_cast",
            selection="logical_group_eligible_canonical_ids",
            max_group_flows=4096,
            schedule="next_observed_step_mod_frequency",
            measure="angular_corrected_estimate",
            reduction="sum_block_over_logical_selected_flows",
            budget_version=1,
        )


@dataclass(frozen=True)
class ProjectionSelection:
    control_id: str
    context_id: str
    flow_steps: tuple[tuple[str, int], ...]
    selected_flow_ids: tuple[str, ...]
    eligible_flows: int
    estimated_bytes: int

    def fingerprint(self):
        return hashlib.sha256(canonical(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class ProjectedMACResult:
    flow_ids: tuple[str, ...]
    indices: tuple[int, ...]
    operators: torch.Tensor
    estimates: NumericalRadiusEstimates
    penalty: torch.Tensor | None
    eligible_flows: int
    reevaluations: int
    control_id: str
    selection_id: str


def _digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class MACProjectionControl(nn.Module):
    """Base fija por contrato, sin contador oculto ni estado temporal propio."""

    def __init__(self, config, mac_config, *, device="cpu", dtype=torch.float32):
        super().__init__()
        if not isinstance(config, MACProjectionConfig) or not isinstance(mac_config, MACConfig):
            raise ValueError("Se necesitan los contratos del control y de MAC")
        if mac_config.memory_mode != "online" or mac_config.memory.dim > 64:
            raise ValueError("El control local admite MAC online con dimensión hasta 64")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("El control local admite float32 y float64")
        self.config, self.mac_config = config, mac_config
        self.state_dimension = 2 * mac_config.memory.depth * mac_config.memory.dim**2
        if config.rank > self.state_dimension:
            raise ValueError("El rango supera la dimensión del estado")
        self._check_budget(1, dtype)
        self.register_buffer("basis", self._make_basis(dtype, device))
        self._seal_basis()
        self.register_load_state_dict_post_hook(self._after_load)

    def _estimate(self, flows, dtype):
        size = 8 if dtype == torch.float64 else 4
        rank, grid = self.config.rank, self.config.grid_size
        state_bytes = self.state_dimension * size
        packed = self.mac_config.persistent_tokens + 2
        attention = self.mac_config.heads * packed**2 * size
        tokens = packed * self.mac_config.memory.dim * size
        # Incluye generación QR, base, grafos JVP y temporales de atención.
        # Es una estimación conservadora contrastada, no una cota del RSS.
        basis = self.state_dimension * rank * (4 * 8 + size)
        graphs = flows * rank * (128 * state_bytes + 256 * (attention + tokens) + 65536)
        radius = 16 * flows * rank**2 * (32 + 8 * grid) + 128 * grid * (flows + 1)
        return basis + graphs + radius + flows * (8 * state_bytes + 16 * tokens)

    def _check_budget(self, flows, dtype):
        estimated = self._estimate(flows, dtype)
        if estimated > self.config.max_estimated_bytes:
            raise ValueError("El control local supera el presupuesto estimado de base y grafos")
        return estimated

    def estimated_bytes(self, flows=1):
        bounded_integer(flows, "flujos medidos", 1, self.config.max_flows)
        return self._estimate(flows, self.basis.dtype)

    def _make_basis(self, dtype, device):
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        with torch.device("cpu"):
            matrix = torch.randn(
                self.state_dimension,
                self.config.rank,
                dtype=torch.float64,
                device="cpu",
                generator=generator,
            )
            basis, triangular = torch.linalg.qr(matrix, mode="reduced")
            basis = basis * torch.where(triangular.diag() < 0, -1.0, 1.0)
        return basis.to(device=device, dtype=dtype).contiguous()

    def _signature(self):
        return (id(self.basis), self.basis._version, self.basis.dtype, self.basis.device)

    def _seal_basis(self):
        self._basis_id = _digest(self.basis)
        self._basis_signature = self._signature()

    def _apply(self, function, recurse=True):
        # Reconstruir desde FP64 evita depender de conversiones intermedias de dtype.
        target = function(torch.empty(0, device=self.basis.device, dtype=self.basis.dtype))
        if target.dtype not in (torch.float32, torch.float64):
            raise ValueError("El control local admite float32 y float64")
        self._check_budget(1, target.dtype)
        self.basis = self._make_basis(target.dtype, target.device)
        self._seal_basis()
        return self

    def verify_basis(self):
        """Comprobar bytes en guardado, carga o una frontera explícita de sesión."""
        if self._signature() != self._basis_signature or _digest(self.basis) != self._basis_id:
            raise ValueError("La base fue modificada fuera de su contrato")

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            mac_contract=self.mac_config.identity(),
            dtype=str(self.basis.dtype),
            basis_sha256=self._basis_id,
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def fingerprint(self):
        return hashlib.sha256(canonical(self.get_extra_state()).encode()).hexdigest()

    def validate_payload(self, payload, prefix=""):
        require_identity(payload.get(prefix + "_extra_state"), self.get_extra_state())
        basis = payload.get(prefix + "basis")
        if (
            not isinstance(basis, torch.Tensor)
            or basis.layout != torch.strided
            or basis.shape != self.basis.shape
            or basis.dtype != self.basis.dtype
            or not basis.is_contiguous()
            or basis.untyped_storage().nbytes() != self.basis.numel() * self.basis.element_size()
            or _digest(basis) != self._basis_id
        ):
            raise ValueError("La base guardada no coincide con su identidad")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.validate_payload(state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _after_load(module, incompatible_keys):
        module._seal_basis()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.verify_basis()
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def _selection(self, flow_ids, observed_steps, batch):
        bounded_integer(batch, "flujos del grupo lógico", 1, 4096)
        if (
            type(flow_ids) is not tuple
            or len(flow_ids) != batch
            or any(
                not isinstance(value, str) or not value.isascii() or not 1 <= len(value) <= 128
                for value in flow_ids
            )
            or len(set(flow_ids)) != batch
            or not isinstance(observed_steps, torch.Tensor)
            or observed_steps.device.type != "cpu"
            or observed_steps.dtype != torch.int64
            or observed_steps.shape != (batch,)
            or observed_steps.layout != torch.strided
        ):
            raise ValueError("Los IDs y contadores necesitan un flujo explícito por fila")
        counters = observed_steps.tolist()
        if any(not 0 <= value < torch.iinfo(torch.int64).max for value in counters):
            raise ValueError("El contador de observaciones no admite otra decisión")
        eligible = [
            index
            for index, value in enumerate(counters)
            if (value + 1) % self.config.frequency == 0
        ]
        selected = sorted(eligible, key=lambda index: flow_ids[index])[: self.config.max_flows]
        return tuple(selected), len(eligible)

    def select_flows(self, flow_ids, observed_steps, *, context_id):
        """Fijar una selección para todo el snapshot/grupo, antes de dividirlo."""
        if not isinstance(context_id, str) or re.fullmatch(r"[0-9a-f]{64}", context_id) is None:
            raise ValueError("El contexto necesita la huella del snapshot y del grupo lógico")
        if type(flow_ids) is not tuple:
            raise ValueError("La selección necesita una tupla de IDs canónicos")
        selected, eligible = self._selection(flow_ids, observed_steps, len(flow_ids))
        estimated = self._check_budget(len(selected), self.basis.dtype) if selected else 0
        return ProjectionSelection(
            self.fingerprint(),
            context_id,
            tuple(sorted(zip(flow_ids, observed_steps.tolist(), strict=True))),
            tuple(flow_ids[index] for index in selected),
            eligible,
            estimated,
        )

    def _resolve_selection(self, selection, context_id, flow_ids, observed_steps):
        if (
            not isinstance(selection, ProjectionSelection)
            or selection.control_id != self.fingerprint()
            or selection.context_id != context_id
            or type(selection.eligible_flows) is not int
            or type(selection.estimated_bytes) is not int
            or type(selection.selected_flow_ids) is not tuple
            or any(not isinstance(value, str) for value in selection.selected_flow_ids)
            or type(selection.flow_steps) is not tuple
            or not 1 <= len(selection.flow_steps) <= 4096
            or any(
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[1]) is not int
                or not 0 <= pair[1] < torch.iinfo(torch.int64).max
                for pair in selection.flow_steps
            )
        ):
            raise ValueError("Falta una selección del control y contexto actuales")
        ids, counts = zip(*selection.flow_steps, strict=True)
        expected = self.select_flows(
            ids, torch.tensor(counts, dtype=torch.int64, device="cpu"), context_id=context_id
        )
        if selection != expected:
            raise ValueError("La selección lógica no coincide con su contenido canónico")
        counters = dict(selection.flow_steps)
        if any(
            counters.get(flow) != count
            for flow, count in zip(flow_ids, observed_steps.tolist(), strict=True)
        ):
            raise ValueError("El bloque no pertenece al snapshot y grupo de la selección")
        indices = {flow: index for index, flow in enumerate(flow_ids)}
        return tuple(indices[flow] for flow in selection.selected_flow_ids if flow in indices)

    def _operator(self, mac, token, state, *, differentiable):
        memory = state.memory
        point = torch.cat([value.flatten() for value in (*memory.weights, *memory.momentum)])
        point = point.detach().requires_grad_(True)
        dim, depth = self.mac_config.memory.dim, self.mac_config.memory.depth

        def transition(value):
            pieces = tuple(piece.reshape(1, dim, dim).clone() for piece in value.split(dim**2))
            local = replace(memory, weights=pieces[:depth], momentum=pieces[depth:])
            _, following = mac(token, replace(state, memory=local), differentiable=True)
            return torch.cat(
                [
                    tensor.flatten()
                    for tensor in (*following.memory.weights, *following.memory.momentum)
                ]
            )

        columns = []
        for direction in self.basis.T:
            _, product = torch.autograd.functional.jvp(
                transition, point, direction, create_graph=differentiable
            )
            columns.append(self.basis.T @ product)
        return torch.stack(columns, dim=1)

    def forward(
        self,
        mac,
        segment,
        state,
        *,
        flow_ids,
        observed_steps,
        selection=None,
        context_id=None,
        differentiable=False,
    ):
        check_differentiable(differentiable)
        if torch.is_inference_mode_enabled():
            raise ValueError("El control local necesita autograd interno, no inference_mode")
        if not isinstance(mac, TitansMAC) or mac.config != self.mac_config:
            raise ValueError("El MAC no corresponde al contrato del control local")
        if self._signature() != self._basis_signature:
            raise ValueError("La base fue modificada fuera de una carga identificada")
        mac._validate_state(state)
        mac.memory.validate_input(segment, state.memory)
        if (
            segment.shape[1] != 1
            or segment.dtype != self.basis.dtype
            or segment.device != self.basis.device
        ):
            raise ValueError("El control necesita C=1 y la precisión y dispositivo de su base")
        self._selection(flow_ids, observed_steps, segment.shape[0])
        if not torch.equal(observed_steps.to(state.memory.steps.device), state.memory.steps):
            raise ValueError("Las observaciones no coinciden con las escrituras de MAC")
        if self.config.mode == "disabled":
            return None
        selected = self._resolve_selection(selection, context_id, flow_ids, observed_steps)
        if not selected:
            return None
        graph = self.config.mode == "penalty" and differentiable
        operators = []
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            for index in selected:
                memory = replace(
                    state.memory,
                    weights=tuple(
                        value[index : index + 1].detach().clone() for value in state.memory.weights
                    ),
                    momentum=tuple(
                        value[index : index + 1].detach().clone() for value in state.memory.momentum
                    ),
                    steps=state.memory.steps[index : index + 1].clone(),
                )
                token = segment[index : index + 1]
                if not graph:
                    token = token.detach()
                operators.append(
                    self._operator(mac, token, replace(state, memory=memory), differentiable=graph)
                )
            matrices = torch.stack(operators)
            estimates = numerical_radius_estimates(
                matrices,
                grid_size=self.config.grid_size,
                max_estimated_bytes=self.config.max_estimated_bytes,
            )
            penalty = None
            if self.config.mode == "penalty":
                penalty = (
                    self.config.weight
                    * radius_penalty(estimates, self.config.threshold).sum()
                    / len(selection.selected_flow_ids)
                )
                check_finite(penalty, "La penalización local")
        return ProjectedMACResult(
            tuple(flow_ids[index] for index in selected),
            selected,
            matrices,
            estimates,
            penalty,
            selection.eligible_flows,
            len(selected) * self.config.rank,
            self.fingerprint(),
            selection.fingerprint(),
        )
