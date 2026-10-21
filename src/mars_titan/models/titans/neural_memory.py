"""Memoria asociativa funcional con momentum y olvido dependientes de la entrada."""

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bank_readout import BankReadout
from .causal_convolution import CausalDepthwiseConvolution
from .config import LAYER_NORM_EPS, MemoryConfig, bounded_integer, require_identity
from .key_competition import competitive_normalize
from .plasticity import _advance_validated as advance_plasticity
from .plasticity import initial_history, validate_history
from .state import (
    NeuralMemoryState,
    check_differentiable,
    check_finite,
    copy_state,
    require_payload,
    require_true,
)

# Filas que puede escribir o leer de una vez la memoria compartida de PT4: todas las
# decisiones de una sesión de un mercado.
MAX_SHARED_ROWS = 16384


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

    # Observador de las trazas de #448 (`learning_traces.memory`). Solo se activa en los pasos
    # que tocan registro, lee copias separadas del grafo y no forma parte del estado.
    observer = None

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
        if config.plasticity is not None and not (
            torch.finfo(dtype).tiny <= config.plasticity.epsilon <= torch.finfo(dtype).max
        ):
            raise ValueError("epsilon no es representable como número normal en esta precisión")
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
        self.bank_readout = (
            BankReadout(config.dim, config.bank_readout, seed=config.parameter_seed, dtype=dtype)
            if config.bank_readout is not None
            else None
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
        required = batch_size * (self.config.matrix_elements * element_size + 8)
        required += batch_size * self.config.modulation_elements * element_size
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
            # PT5 parte de copias de los pesos iniciales en cada nivel oculto, así que al
            # empezar la lectura coincide con la de la memoria sin niveles.
            levels = tuple(
                w.unsqueeze(0).repeat(batch_size, 1, 1)
                for w in self.initial_weights
                for _ in range(self.config.hidden_levels)
            )
        state = NeuralMemoryState(
            weights,
            tuple(torch.zeros_like(w) for _ in range(self.config.momentum_copies) for w in weights),
            torch.zeros(batch_size, dtype=torch.int64, device=reference.device),
            self.config.fingerprint(),
            self.initial_windows(batch_size, 2),
            levels,
            tuple(value for weight in weights for value in initial_history(weight))
            if self.config.plasticity is not None
            else (),
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
            or type(state.levels) is not tuple
            or type(state.modulation) is not tuple
            or len(state.weights) != self.config.depth
            or len(state.momentum) != self.config.depth * self.config.momentum_copies
            or len(state.levels) != self.config.depth * self.config.hidden_levels
            or len(state.modulation)
            != (0 if self.config.plasticity is None else 3 * self.config.depth)
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
        tensors = (*state.weights, *state.momentum, *state.levels, state.steps)
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
        for index in range(self.config.depth if self.config.plasticity is not None else 0):
            history = state.modulation[3 * index : 3 * index + 3]
            validate_history(history, state.weights[index])
            for value in history:
                storage = value.untyped_storage()
                storage_id = (value.device, storage.data_ptr())
                if storage_id in storage_ids:
                    raise ValueError("La plasticidad no puede compartir almacenamiento mutable")
                storage_ids.add(storage_id)
                storage_bytes += storage.nbytes()
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

    def _read_memory(self, query, state, translation=None):
        if self.bank_readout is None:
            return self._apply_memory(query, self.readable_weights(state))
        banks = self._banks(state)
        outputs = []
        for bank in banks:
            weights = (
                bank
                if translation is None
                else tuple(w + d for w, d in zip(bank, translation, strict=True))
            )
            outputs.append(self._apply_memory(query, weights))
        return self.bank_readout(query, torch.stack(outputs, dim=2))

    def _banks(self, state):
        return (state.weights,) + tuple(
            tuple(state.levels[2 * layer + region] for layer in range(self.config.depth))
            for region in range(2)
        )

    def _associative_problem(self, keys, state, differentiable):
        if self.bank_readout is None:
            local = tuple(
                (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
                for w in self.readable_weights(state)
            )
            return local, self._apply_memory(keys, local)
        if self.config.momentum_copies == 3:
            local = tuple(
                (w.clone() if differentiable else w.detach().clone()).requires_grad_(True)
                for bank in self._banks(state)
                for w in bank
            )
            depth = self.config.depth
            outputs = torch.stack(
                [self._apply_memory(keys, local[r * depth : (r + 1) * depth]) for r in range(3)],
                dim=2,
            )
            return local, self.bank_readout(keys, outputs)
        # La variable de derivación es una traslación compartida por los bancos. Su
        # gradiente suma las parciales de cada región sin fingir que se lee una media de pesos.
        local = tuple(torch.zeros_like(w, requires_grad=True) for w in state.weights)
        if not differentiable:
            state = replace(
                state,
                weights=tuple(w.detach() for w in state.weights),
                levels=tuple(w.detach() for w in state.levels),
            )
        return local, self._read_memory(keys, state, local)

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

    def readable_weights(self, state: NeuralMemoryState) -> tuple[Tensor, ...]:
        """Pesos que lee la memoria: el primer nivel, salvo en el banco de PT5, su media.

        La media se escribe como w + ((z2 - w) + (z3 - w)) / 3. Es la misma cantidad que
        (w + z2 + z3) / 3, pero con tres copias iguales devuelve w sin redondeo, así que el
        banco arranca leyendo exactamente los pesos del control.
        """
        if self.bank_readout is not None:
            raise ValueError("La lectura por consulta no tiene una red equivalente de pesos medios")
        multiscale = self.config.multiscale
        if multiscale is None or multiscale.form == "cascade":
            return state.weights
        hidden = self.config.hidden_levels
        return tuple(
            weight
            + (
                (state.levels[hidden * index] - weight)
                + (state.levels[hidden * index + 1] - weight)
            )
            / 3
            for index, weight in enumerate(state.weights)
        )

    def _advance_levels(self, weights, levels, momentum, alpha, capture_gains=None):
        """Olvido y escritura con PT5: Euler de paso 1 con los valores del token anterior."""
        multiscale = self.config.multiscale
        g12, g23, g34 = multiscale.conductances
        hidden = self.config.hidden_levels
        new_weights, new_levels = [], []
        for index, (first, surprise) in enumerate(zip(weights, momentum, strict=True)):
            second, third = levels[hidden * index], levels[hidden * index + 1]
            if multiscale.form == "cascade":
                # Capacidades 1, 2 y 4. El primer nivel conserva el olvido α de Titans.
                exchange = g12 if capture_gains is None else g12 * capture_gains[index]
                new_weights.append((1 - alpha) * first + surprise + exchange * (second - first))
                new_levels.append(
                    second + (exchange * (first - second) + g23 * (third - second)) / 2
                )
                new_levels.append(third + (g23 * (second - third) - g34 * third) / 4)
            else:
                decay_second, decay_third = (
                    (alpha, alpha) if multiscale.form == "replica_bank" else multiscale.bank_decays
                )
                new_weights.append((1 - alpha) * first + surprise)
                new_levels.append((1 - decay_second) * second + surprise)
                new_levels.append((1 - decay_third) * third + surprise)
        return tuple(new_weights), tuple(new_levels)

    def _write(
        self,
        weights,
        momentum,
        levels,
        gradients,
        gates,
        differentiable,
        *,
        modulation=(),
        steps=None,
        observation=None,
    ):
        """Momentum, olvido y escritura de un paso con el gradiente asociativo ya calculado."""
        if self.config.momentum_copies == 3:
            return self._write_banks(weights, momentum, levels, gradients, gates, differentiable)
        alpha, eta, theta = gates
        stability = self.config.stability
        if stability is not None and stability.gradient_clip is not None:
            # PT1 recorta antes del momentum para que cada escritura quede acotada.
            gradients = tuple(
                clip_rows(gradient, stability.gradient_clip) for gradient in gradients
            )
        capture_gains = None
        if self.config.plasticity is None:
            momentum = tuple(
                eta * previous - theta * gradient
                for previous, gradient in zip(momentum, gradients, strict=True)
            )
        else:
            plasticity = self.config.plasticity
            scalar_controls = plasticity.gradient_contract == "scalar_controls_v2"
            updates = tuple(
                advance_plasticity(plasticity, gradient, modulation[3 * i : 3 * i + 3], steps)
                for i, gradient in enumerate(gradients)
            )
            if observation is not None:
                observation.update(
                    plasticity=updates,
                    capture_applied=(
                        self.config.multiscale is not None
                        and self.config.multiscale.form == "cascade"
                    ),
                )
            if scalar_controls and plasticity.write != "coherence":
                # Expandir theta por filas antes de multiplicar g altera su reducción backward.
                if plasticity.write == "constant" and plasticity.write_constant != 1:
                    theta = theta * plasticity.write_constant
                momentum = tuple(
                    eta * previous - theta * gradient
                    for previous, gradient in zip(momentum, gradients, strict=True)
                )
            else:
                momentum = tuple(
                    eta * previous - (theta * update.write_gain) * gradient
                    for previous, gradient, update in zip(momentum, gradients, updates, strict=True)
                )
            modulation = tuple(value for update in updates for value in update.history)
            if scalar_controls and plasticity.capture == "constant":
                if plasticity.capture_constant != 1:
                    capture_gains = (plasticity.capture_constant,) * len(updates)
            elif plasticity.capture != "none":
                capture_gains = tuple(update.capture_gain for update in updates)
        if self.config.multiscale is None:
            weights = tuple(
                (1 - alpha) * weight + surprise
                for weight, surprise in zip(weights, momentum, strict=True)
            )
        else:
            weights, levels = self._advance_levels(weights, levels, momentum, alpha, capture_gains)
        for value in (*weights, *momentum, *levels, *modulation):
            check_finite(value, "La actualización de memoria")
        if not differentiable:
            weights = tuple(w.detach() for w in weights)
            momentum = tuple(m.detach() for m in momentum)
            levels = tuple(level.detach() for level in levels)
            modulation = tuple(value.detach() for value in modulation)
        return weights, momentum, levels, modulation

    def _write_banks(self, weights, momentum, levels, gradients, gates, differentiable):
        """Parciales de la mezcla, o su proyección media, con un momentum por banco."""
        alpha, eta, theta = gates
        depth = self.config.depth
        if self.config.bank_readout.write_gradient == "mean_partial":
            average = tuple(
                torch.stack([gradients[r * depth + i] for r in range(3)]).mean(0)
                for i in range(depth)
            )
            gradients = average * 3
        following_momentum = tuple(
            eta * old - theta * gradient for old, gradient in zip(momentum, gradients, strict=True)
        )
        decays = (
            (alpha, alpha, alpha)
            if self.config.multiscale.form == "replica_bank"
            else (alpha, *self.config.multiscale.bank_decays)
        )
        matrices = weights + tuple(levels[2 * i + r] for r in range(2) for i in range(depth))
        following = tuple(
            (1 - decays[r]) * matrices[r * depth + i] + following_momentum[r * depth + i]
            for r in range(3)
            for i in range(depth)
        )
        for value in (*following, *following_momentum):
            check_finite(value, "La actualización por banco")
        if not differentiable:
            following = tuple(value.detach() for value in following)
            following_momentum = tuple(value.detach() for value in following_momentum)
        return (
            following[:depth],
            following_momentum,
            tuple(following[(r + 1) * depth + i] for i in range(depth) for r in range(2)),
            (),
        )

    def read_shared(self, query: Tensor, state: NeuralMemoryState) -> Tensor:
        """Leer n consultas [n, dim] con un único estado compartido de lote 1 (PT4)."""
        self._check_shared(query, state)
        result = self._read_memory(query.unsqueeze(0), state).squeeze(0)
        check_finite(result, "La lectura de memoria")
        return result

    def update_shared(
        self, observed: Tensor, state: NeuralMemoryState, *, differentiable: bool = False
    ) -> NeuralMemoryState:
        """Una sola escritura con la media de los gradientes de n observaciones (PT4).

        Todas las filas [n, dim] se escriben a la vez en el estado compartido de lote 1. La
        pérdida asociativa es la media de las n pérdidas en los mismos pesos, así que su
        gradiente es la media de los gradientes internos. Las puertas salen de la media de
        los logits, que coincide con los logits de la observación media porque las
        proyecciones son lineales. El orden de las filas solo cambia el redondeo.
        """
        check_differentiable(differentiable)
        if torch.is_inference_mode_enabled():
            raise ValueError("inference_mode impide el gradiente interno, utilice no_grad")
        self._check_shared(observed, state)
        require_true(
            (state.steps < torch.iinfo(torch.int64).max).all(),
            "El contador de pasos desbordaría int64",
        )
        with torch.enable_grad():
            token = observed if differentiable else observed.detach()
            keys = self.key_projection(token)
            values = self.value_projection(token)
            logits = tuple(
                projection(token).mean(dim=0, keepdim=True)
                for projection in (
                    self.alpha_projection,
                    self.eta_projection,
                    self.theta_projection,
                )
            )
            for projected in (keys, values, *logits):
                check_finite(projected, "Las proyecciones asociativas")
            if self.config.key_competition is not None:
                keys = competitive_normalize(keys, self.config.key_competition)
            elif self.config.normalize_qk:
                keys = F.normalize(keys, dim=-1, eps=1e-12)
            gates = self._gates(*logits)
            local, prediction = self._associative_problem(keys.unsqueeze(0), state, differentiable)
            residual = prediction.squeeze(0) - values
            loss = residual.square().sum() / observed.shape[0]
            check_finite(loss, "La pérdida asociativa")
            gradients = torch.autograd.grad(loss, local, create_graph=differentiable)
            observation = {} if self.observer is not None else None
            weights, momentum, levels, modulation = self._write(
                state.weights,
                state.momentum,
                state.levels,
                gradients,
                gates,
                differentiable,
                modulation=state.modulation,
                steps=state.steps,
                observation=observation,
            )
            if self.observer is not None:
                self.observer(
                    alpha=gates[0],
                    eta=gates[1],
                    theta=gates[2],
                    loss=loss,
                    weights=weights,
                    momentum=momentum,
                    shared_rows=observed.shape[0],
                    **observation,
                )
        return NeuralMemoryState(
            weights, momentum, state.steps + 1, state.config_id, (), levels, modulation
        )

    def _check_shared(self, values: Tensor, state: NeuralMemoryState) -> None:
        self.validate_state(state)
        reference = self._reference()
        if self.config.window or self.config.qkv_silu:
            raise ValueError("La memoria compartida no admite ventanas causales ni SiLU por flujo")
        if (
            state.steps.shape != (1,)
            or not isinstance(values, Tensor)
            or values.layout != torch.strided
            or values.ndim != 2
            or not 1 <= values.shape[0] <= MAX_SHARED_ROWS
            or values.shape[1] != self.config.dim
            or values.dtype != reference.dtype
            or values.device != reference.device
        ):
            raise ValueError("La memoria compartida necesita [n, dim] y un estado de lote 1")
        check_finite(values, "La entrada")

    def read(self, query: Tensor, state: NeuralMemoryState) -> Tensor:
        """Lee sin escribir. La proyección y normalización de queries corresponden a MAC."""
        self.validate_input(query, state)
        result = self._read_memory(query, state)
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
        weights, momentum, levels = state.weights, state.momentum, state.levels
        modulation = state.modulation
        key_window, value_window = state.convolution or (None, None)
        with torch.enable_grad():
            for index, token in enumerate(observed.unbind(1)):
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
                if self.config.key_competition is not None:
                    keys = competitive_normalize(keys, self.config.key_competition)
                elif self.config.normalize_qk:
                    keys = F.normalize(keys, dim=-1, eps=1e-12)
                alpha, eta, theta = self._gates(alpha_logits, eta_logits, theta_logits)
                # La copia separa la variable de derivación interna de y(M_prev). Con el banco
                # de PT5 la memoria lee la media de sus niveles, y el gradiente se toma en ella.
                local, prediction = self._associative_problem(
                    keys.unsqueeze(1),
                    replace(state, weights=weights, levels=levels),
                    differentiable,
                )
                residual = prediction.squeeze(1) - values
                loss = residual.square().sum()
                check_finite(loss, "La pérdida asociativa")
                gradients = torch.autograd.grad(loss, local, create_graph=differentiable)
                observation = {} if self.observer is not None else None
                weights, momentum, levels, modulation = self._write(
                    weights,
                    momentum,
                    levels,
                    gradients,
                    (alpha, eta, theta),
                    differentiable,
                    modulation=modulation,
                    steps=state.steps + index if self.config.plasticity is not None else None,
                    observation=observation,
                )
                if self.observer is not None:
                    self.observer(
                        alpha=alpha,
                        eta=eta,
                        theta=theta,
                        loss=loss,
                        weights=weights,
                        momentum=momentum,
                        **observation,
                    )
        windows = () if key_window is None else (key_window, value_window)
        if not differentiable:
            windows = tuple(w.detach() for w in windows)
        return NeuralMemoryState(
            weights,
            momentum,
            state.steps + observed.shape[1],
            state.config_id,
            windows,
            levels,
            modulation,
        )

    def get_extra_state(self) -> dict:
        return {"configuration": self.config.identity(), "dtype": str(self._reference().dtype)}

    def set_extra_state(self, state: object) -> None:
        require_identity(state, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        require_identity(state_dict.get(prefix + "_extra_state"), self.get_extra_state())
        if self.bank_readout is not None:
            self.bank_readout.validate_payload(state_dict, prefix + "bank_readout.")
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
        if self.config.hidden_levels:
            payload["levels"] = copied.levels
        if self.config.plasticity is not None:
            payload["modulation"] = copied.modulation
        return payload

    def payload_fields(self) -> set[str]:
        fields = {"schema_version", "configuration", "weights", "momentum", "steps"}
        if self.config.window:
            fields |= {"convolution"}
        if self.config.hidden_levels:
            fields |= {"levels"}
        if self.config.plasticity is not None:
            fields |= {"modulation"}
        return fields

    def restore_state(self, payload: object) -> NeuralMemoryState:
        value = require_payload(payload, self.payload_fields())
        require_identity(value["configuration"], self.config.identity())
        state = NeuralMemoryState(
            value["weights"],
            value["momentum"],
            value["steps"],
            self.config.fingerprint(),
            value.get("convolution", ()),
            value.get("levels", ()),
            value.get("modulation", ()),
        )
        self.validate_state(state)
        return copy_state(state, differentiable=False)
