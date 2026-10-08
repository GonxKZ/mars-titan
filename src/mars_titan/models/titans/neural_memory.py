"""Memoria asociativa funcional con momentum y olvido dependientes de la entrada."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import MemoryConfig, bounded_integer, require_identity
from .state import (
    NeuralMemoryState,
    check_differentiable,
    check_finite,
    copy_state,
    require_payload,
)


class NeuralMemory(nn.Module):
    """Los parámetros lentos son compartidos y el estado rápido pertenece al llamante."""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not isinstance(config, MemoryConfig):
            raise ValueError("Se necesita MemoryConfig")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La memoria admite float32 y float64")
        self.config = config
        # Inicializar en CPU aísla la semilla de los flujos científicos existentes.
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.parameter_seed)
            self.initial_weights = nn.ParameterList(
                nn.Parameter(torch.empty(config.dim, config.dim, dtype=dtype))
                for _ in range(config.depth)
            )
            for weight in self.initial_weights:
                nn.init.xavier_uniform_(weight)
            self.key_projection = nn.Linear(config.dim, config.dim, bias=False, dtype=dtype)
            self.value_projection = nn.Linear(config.dim, config.dim, bias=False, dtype=dtype)
            self.alpha_projection = nn.Linear(config.dim, config.dim, bias=False, dtype=dtype)
            self.eta_projection = nn.Linear(config.dim, 1, bias=False, dtype=dtype)
            self.theta_projection = nn.Linear(config.dim, 1, bias=False, dtype=dtype)
        self.to(device=device)

    def _reference(self) -> Tensor:
        reference = self.initial_weights[0]
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("La memoria admite float32 y float64")
        if torch.is_autocast_enabled(reference.device.type):
            raise ValueError("El núcleo exige precisión explícita, sin autocast")
        return reference

    def _check_size(self, batch_size: int) -> None:
        bounded_integer(batch_size, "lote", 1, self.config.max_batch)
        element_size = self._reference().element_size()
        required = batch_size * (2 * self.config.depth * self.config.dim**2 * element_size + 8)
        if required > self.config.max_state_bytes:
            raise ValueError("El estado supera el presupuesto de bytes")

    def initial_state(self, batch_size: int, *, differentiable: bool = False) -> NeuralMemoryState:
        check_differentiable(differentiable)
        self._check_size(batch_size)
        reference = self._reference()
        with torch.set_grad_enabled(differentiable):
            weights = tuple(w.unsqueeze(0).repeat(batch_size, 1, 1) for w in self.initial_weights)
        state = NeuralMemoryState(
            weights,
            tuple(torch.zeros_like(w) for w in weights),
            torch.zeros(batch_size, dtype=torch.int64, device=reference.device),
            self.config.fingerprint(),
        )
        self.validate_state(state)
        return state

    def validate_state(self, state: NeuralMemoryState) -> None:
        if not isinstance(state, NeuralMemoryState) or state.config_id != self.config.fingerprint():
            raise ValueError("El estado pertenece a otro contrato de configuración")
        if (
            type(state.weights) is not tuple
            or type(state.momentum) is not tuple
            or len(state.weights) != self.config.depth
            or len(state.momentum) != self.config.depth
            or not isinstance(state.steps, Tensor)
            or state.steps.ndim != 1
        ):
            raise ValueError("Las capas o los pasos del estado tienen una forma incompatible")
        batch = state.steps.shape[0]
        self._check_size(batch)
        reference = self._reference()
        expected = (batch, self.config.dim, self.config.dim)
        storage_ids: set[tuple[torch.device, int]] = set()
        storage_bytes = 0
        tensors = (*state.weights, *state.momentum, state.steps)
        for index, value in enumerate(tensors):
            is_steps = index == len(tensors) - 1
            shape = (batch,) if is_steps else expected
            dtype = torch.int64 if is_steps else reference.dtype
            if (
                not isinstance(value, Tensor)
                or value.layout != torch.strided
                or value.shape != shape
                or value.dtype != dtype
                or value.device != reference.device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    "El estado tiene forma, tipo, dispositivo o contigüidad incompatible"
                )
            storage = value.untyped_storage()
            storage_id = (value.device, storage.data_ptr())
            if storage_id in storage_ids:
                raise ValueError(
                    "Los tensores del estado no pueden compartir almacenamiento mutable"
                )
            storage_ids.add(storage_id)
            storage_bytes += storage.nbytes()
            if storage_bytes > self.config.max_state_bytes:
                raise ValueError("El almacenamiento del estado supera el presupuesto de bytes")
            if is_steps:
                if (value < 0).any():
                    raise ValueError("Los pasos del estado no pueden ser negativos")
            else:
                check_finite(value, "El estado")

    def validate_input(self, values: Tensor, state: NeuralMemoryState) -> None:
        self.validate_state(state)
        reference = self._reference()
        if (
            not isinstance(values, Tensor)
            or values.layout != torch.strided
            or values.ndim != 3
            or values.shape[0] != state.steps.shape[0]
            or values.shape[2] != self.config.dim
            or values.dtype != reference.dtype
            or values.device != reference.device
        ):
            raise ValueError(
                "La entrada debe tener forma [lote, secuencia, dimensión] y tipo del módulo"
            )
        if not 1 <= values.shape[1] <= self.config.max_tokens:
            raise ValueError("La secuencia supera el presupuesto de tokens")
        check_finite(values, "La entrada")

    @staticmethod
    def _apply_memory(values: Tensor, weights: tuple[Tensor, ...]) -> Tensor:
        for index, weight in enumerate(weights):
            values = torch.bmm(values, weight.transpose(1, 2))
            if index + 1 < len(weights):
                values = F.gelu(values)
        return values

    def read(self, query: Tensor, state: NeuralMemoryState) -> Tensor:
        """Lee sin normalizar ni escribir. La proyección de queries corresponde a MAC."""
        self.validate_input(query, state)
        result = self._apply_memory(query, state.weights)
        check_finite(result, "La lectura de memoria")
        return result

    def update(
        self, observed: Tensor, state: NeuralMemoryState, *, differentiable: bool = False
    ) -> NeuralMemoryState:
        """Actualiza por token usando la derivada parcial con la observación fija."""
        check_differentiable(differentiable)
        if torch.is_inference_mode_enabled():
            raise ValueError("inference_mode impide el gradiente interno, utilice no_grad")
        self.validate_input(observed, state)
        if (state.steps > torch.iinfo(torch.int64).max - observed.shape[1]).any():
            raise ValueError("El contador de pasos desbordaría int64")
        weights, momentum = state.weights, state.momentum
        with torch.enable_grad():
            for token in observed.unbind(1):
                if not differentiable:
                    token = token.detach()
                keys = self.key_projection(token)
                values = self.value_projection(token)
                alpha_logits = self.alpha_projection(token)
                eta_logits = self.eta_projection(token)
                theta_logits = self.theta_projection(token)
                for projected in (keys, values, alpha_logits, eta_logits, theta_logits):
                    check_finite(projected, "Las proyecciones asociativas")
                if self.config.normalize_qk:
                    keys = F.normalize(keys, dim=-1, eps=1e-12)
                alpha = alpha_logits.sigmoid().unsqueeze(-1)
                eta = eta_logits.sigmoid().unsqueeze(-1)
                theta = self.config.theta_max * theta_logits.sigmoid().unsqueeze(-1)
                # La copia separa la variable de derivación interna de y(M_prev).
                local = tuple(
                    (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
                    for w in weights
                )
                residual = self._apply_memory(keys.unsqueeze(1), local).squeeze(1) - values
                loss = residual.square().sum()
                check_finite(loss, "La pérdida asociativa")
                gradients = torch.autograd.grad(loss, local, create_graph=differentiable)
                momentum = tuple(
                    eta * previous - theta * gradient
                    for previous, gradient in zip(momentum, gradients, strict=True)
                )
                weights = tuple(
                    (1 - alpha) * weight + surprise
                    for weight, surprise in zip(weights, momentum, strict=True)
                )
                for value in (*weights, *momentum):
                    check_finite(value, "La actualización de memoria")
                if not differentiable:
                    weights = tuple(w.detach() for w in weights)
                    momentum = tuple(m.detach() for m in momentum)
        return NeuralMemoryState(
            weights, momentum, state.steps + observed.shape[1], state.config_id
        )

    def get_extra_state(self) -> dict:
        return {"configuration": self.config.identity(), "dtype": str(self._reference().dtype)}

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        require_identity(state_dict.get(prefix + "_extra_state"), self.get_extra_state())
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def export_state(self, state: NeuralMemoryState) -> dict:
        self.validate_state(state)
        copied = copy_state(state, differentiable=False)
        return {
            "schema_version": 1,
            "configuration": self.config.identity(),
            "weights": copied.weights,
            "momentum": copied.momentum,
            "steps": copied.steps,
        }

    def restore_state(self, payload: object) -> NeuralMemoryState:
        value = require_payload(
            payload, {"schema_version", "configuration", "weights", "momentum", "steps"}
        )
        require_identity(value["configuration"], self.config.identity())
        state = NeuralMemoryState(
            value["weights"], value["momentum"], value["steps"], self.config.fingerprint()
        )
        self.validate_state(state)
        return copy_state(state, differentiable=False)
