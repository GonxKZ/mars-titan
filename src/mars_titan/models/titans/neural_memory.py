"""Memoria asociativa funcional con momentum y olvido dependientes de la entrada."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .causal_convolution import CausalDepthwiseConvolution
from .config import LAYER_NORM_EPS, MemoryConfig, bounded_integer, require_identity
from .state import (
    NeuralMemoryState,
    check_differentiable,
    check_finite,
    copy_state,
    require_payload,
    require_true,
)


def clip_rows(gradient: Tensor, limit: float) -> Tensor:
    """Reescala cada fila del gradiente para que su norma euclídea no supere `limit` (PT1).

    Las filas que ya cumplen la cota se multiplican exactamente por 1. La rama que `where`
    descarta usa la norma al cuadrado acotada por debajo, de modo que su derivada es finita
    también en una fila nula y el grafo de segundo orden no produce NaN.
    """
    squared = gradient.square().sum(dim=-1, keepdim=True)
    bound = limit * limit
    scale = torch.where(squared > bound, limit * squared.clamp(min=bound).rsqrt(), 1.0)
    return gradient * scale


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
            if config.gate_bias is not None:
                # Los bias constantes no consumen RNG: los pesos coinciden con los de v1.
                gates = (self.alpha_projection, self.eta_projection, self.theta_projection)
                logits = config.gate_bias.logits(config.theta_max, config.stability)
                for gate, value in zip(gates, logits, strict=True):
                    gate.bias = nn.Parameter(torch.full((gate.out_features,), value, dtype=dtype))
            if config.qkv_convolution:
                # Se sortean después de los parámetros anteriores para que estos conserven
                # sus valores con la misma semilla.
                self.key_convolution = CausalDepthwiseConvolution(
                    config.dim, config.qkv_convolution, dtype=dtype
                )
                self.value_convolution = CausalDepthwiseConvolution(
                    config.dim, config.qkv_convolution, dtype=dtype
                )
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
        # Se cuentan las tres ventanas de q, k y v porque dependen de esta configuración,
        # aunque la de q viaje en el estado de MAC.
        required += batch_size * 3 * self.config.window * self.config.dim * element_size
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
            self.initial_windows(batch_size, 2),
        )
        self.validate_state(state)
        return state

    def initial_windows(self, batch_size: int, count: int) -> tuple[Tensor, ...]:
        """Ventanas a cero, equivalentes al relleno por la izquierda. Vacío sin convolución."""
        if not self.config.window:
            return ()
        reference = self._reference()
        shape = (batch_size, self.config.window, self.config.dim)
        return tuple(
            torch.zeros(shape, dtype=reference.dtype, device=reference.device) for _ in range(count)
        )

    def validate_windows(
        self,
        windows: object,
        count: int,
        batch: int,
        *,
        device=None,
        storage_ids: set | None = None,
    ) -> int:
        """Comprobar las ventanas declaradas y devolver los bytes de su almacenamiento."""
        expected_count = count if self.config.window else 0
        if type(windows) is not tuple or len(windows) != expected_count:
            raise ValueError("Las ventanas de convolución no corresponden al contrato")
        reference = self._reference()
        expected_device = reference.device if device is None else torch.device(device)
        storage_ids = set() if storage_ids is None else storage_ids
        storage_bytes = 0
        for value in windows:
            if (
                not isinstance(value, Tensor)
                or value.layout != torch.strided
                or value.shape != (batch, self.config.window, self.config.dim)
                or value.dtype != reference.dtype
                or value.device != expected_device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    "La ventana tiene forma, tipo, dispositivo o contigüidad incompatible"
                )
            storage = value.untyped_storage()
            storage_id = (value.device, storage.data_ptr())
            if storage_id in storage_ids:
                raise ValueError(
                    "Los tensores del estado no pueden compartir almacenamiento mutable"
                )
            storage_ids.add(storage_id)
            storage_bytes += storage.nbytes()
            check_finite(value, "La ventana de convolución")
        return storage_bytes

    def validate_state(self, state: NeuralMemoryState, *, device=None) -> None:
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
        expected_device = reference.device if device is None else torch.device(device)
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
                or value.device != expected_device
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
                require_true((value >= 0).all(), "Los pasos del estado no pueden ser negativos")
            else:
                check_finite(value, "El estado")
        storage_bytes += self.validate_windows(
            state.convolution, 2, batch, device=expected_device, storage_ids=storage_ids
        )
        if storage_bytes > self.config.max_state_bytes:
            raise ValueError("El almacenamiento del estado supera el presupuesto de bytes")

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

    def _apply_memory(self, values: Tensor, weights: tuple[Tensor, ...]) -> Tensor:
        """Aplicar la red de memoria. Con residual_layer_norm, M(x) = x + LN(MLP(x))."""
        result = values
        for index, weight in enumerate(weights):
            result = torch.bmm(result, weight.transpose(1, 2))
            if index + 1 < len(weights):
                result = F.gelu(result)
        if self.config.residual_layer_norm:
            result = values + F.layer_norm(result, (self.config.dim,), eps=LAYER_NORM_EPS)
        return result

    def project(self, values: Tensor, convolution, window: Tensor | None):
        """Convolución causal opcional y SiLU opcional sobre una proyección lineal."""
        if window is not None:
            values, window = convolution(values, window)
        if self.config.qkv_silu:
            values = F.silu(values)
        return values, window

    def _gates(
        self, alpha_logits: Tensor, eta_logits: Tensor, theta_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convierte los logits de un token en las tasas de olvido, momentum y paso.

        α tiene una tasa por fila de salida, con forma [B, D, 1], y η y θ son escalares por
        flujo, con forma [B, 1, 1]. Con la caja de PT1 las tasas quedan dentro de la región
        del certificado para cualquier entrada.
        """
        stability = self.config.stability
        if stability is not None and stability.gate_box:
            # Ninguna entrada puede llevar las puertas fuera de α ∈ [α_lo, 1] y η ∈ [0, η_hi].
            floor = stability.alpha_floor
            alpha = (floor + (1 - floor) * alpha_logits.sigmoid()).unsqueeze(-1)
            eta = (stability.eta_ceiling * eta_logits.sigmoid()).unsqueeze(-1)
        else:
            alpha = alpha_logits.sigmoid().unsqueeze(-1)
            eta = eta_logits.sigmoid().unsqueeze(-1)
        theta = self.config.theta_max * theta_logits.sigmoid().unsqueeze(-1)
        return alpha, eta, theta

    def read(self, query: Tensor, state: NeuralMemoryState) -> Tensor:
        """Lee sin escribir. La proyección y normalización de queries corresponden a MAC."""
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
        require_true(
            (state.steps <= torch.iinfo(torch.int64).max - observed.shape[1]).all(),
            "El contador de pasos desbordaría int64",
        )
        weights, momentum = state.weights, state.momentum
        key_window, value_window = state.convolution or (None, None)
        stability = self.config.stability
        clip = None if stability is None else stability.gradient_clip
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
                if key_window is not None or self.config.qkv_silu:
                    # La convolución avanza token a token, igual que la proyección, y por eso
                    # cada posición recibe los mismos bits en cualquier partición de la secuencia.
                    keys, key_window = self.project(
                        keys.unsqueeze(1), getattr(self, "key_convolution", None), key_window
                    )
                    values, value_window = self.project(
                        values.unsqueeze(1), getattr(self, "value_convolution", None), value_window
                    )
                    keys, values = keys.squeeze(1), values.squeeze(1)
                if self.config.normalize_qk:
                    keys = F.normalize(keys, dim=-1, eps=1e-12)
                alpha, eta, theta = self._gates(alpha_logits, eta_logits, theta_logits)
                # La copia separa la variable de derivación interna de y(M_prev).
                local = tuple(
                    (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
                    for w in weights
                )
                residual = self._apply_memory(keys.unsqueeze(1), local).squeeze(1) - values
                loss = residual.square().sum()
                check_finite(loss, "La pérdida asociativa")
                gradients = torch.autograd.grad(loss, local, create_graph=differentiable)
                if clip is not None:
                    # PT1 recorta antes del momentum para que cada escritura quede acotada.
                    gradients = tuple(clip_rows(gradient, clip) for gradient in gradients)
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
        windows = () if key_window is None else (key_window, value_window)
        if not differentiable:
            windows = tuple(w.detach() for w in windows)
        return NeuralMemoryState(
            weights, momentum, state.steps + observed.shape[1], state.config_id, windows
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
        payload = {
            "schema_version": 1,
            "configuration": self.config.identity(),
            "weights": copied.weights,
            "momentum": copied.momentum,
            "steps": copied.steps,
        }
        if self.config.window:
            payload["convolution"] = copied.convolution
        return payload

    def payload_fields(self) -> set[str]:
        fields = {"schema_version", "configuration", "weights", "momentum", "steps"}
        return fields | {"convolution"} if self.config.window else fields

    def restore_state(self, payload: object) -> NeuralMemoryState:
        value = require_payload(payload, self.payload_fields())
        require_identity(value["configuration"], self.config.identity())
        state = NeuralMemoryState(
            value["weights"],
            value["momentum"],
            value["steps"],
            self.config.fingerprint(),
            value.get("convolution", ()),
        )
        self.validate_state(state)
        return copy_state(state, differentiable=False)
