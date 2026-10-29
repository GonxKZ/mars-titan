"""Lectura episódica posterior a MAC, sin admisión ni estado persistente propio.

R3 de #786 (`familiarity_gate_v1`) adapta las señales de novedad del hipocampo. La
familiaridad de una lectura es el coseno medio entre la consulta y los episodios que la
forman, con los pesos de su atención y sin gradiente. La puerta 2σ(a·f + b) escala el cambio
de cada refinamiento, de modo que la memoria pesa menos si los episodios se parecen poco al
presente. Su control usa la misma puerta con la familiaridad sustituida por una constante
aprendida, 2σ(b). Los dos parámetros empiezan en cero y la puerta en uno, así que el lector
empieza siendo el de referencia.
"""

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..quantile_head import LEVELS, QuantileHead, median
from .config import MAX_BLOCK_ROWS, MAX_STATE_BYTES, bounded_integer, require_identity
from .episodic_snapshot import EpisodeSnapshot, _device, _digest_id
from .financial import PreparedDecisions
from .state import check_differentiable, check_finite, deferred_checks, require_true

if TYPE_CHECKING:
    from .evidence_bridge import EvidenceBridgeConfig
    from .regional_episode_readout import BridgedReadoutResult

# `per_step` vuelve a seleccionar episodios en cada refinamiento. `first_read` fija en los
# pasos siguientes los episodios de la primera lectura y solo recalcula sus pesos.
EPISODE_SELECTIONS = ("per_step", "first_read")
REFINEMENT_INITIALIZATIONS = ("pytorch_default", "zero_refinement")
# R3 de #786: `familiarity` es la puerta que depende de la familiaridad de cada fila y
# `constant` su control, con la misma puerta sin ese término.
FAMILIARITY_GATES = ("familiarity", "constant")


@dataclass(frozen=True)
class EpisodicReadoutConfig:
    codec_id: str
    hidden_size: int = 64
    refinements: int = 1
    neighbors: int = 8
    temperature: float = 1.0
    mode: str = "bank"
    seed: int = 42
    max_batch: int = 256
    max_working_bytes: int = 32 * 1024**2
    episode_selection: str = "per_step"
    refinement_initialization: str = "pytorch_default"
    evidence_bridge: "EvidenceBridgeConfig | None" = None
    familiarity_gate: str | None = None

    def __post_init__(self):
        _digest_id(self.codec_id, "El codec")
        bounded_integer(self.hidden_size, "dimensión", 1, 128)
        bounded_integer(self.neighbors, "vecinos", 1, 8)
        bounded_integer(self.max_batch, "lote", 1, MAX_BLOCK_ROWS)
        bounded_integer(self.seed, "semilla", 0, 2**32 - 1)
        bounded_integer(self.max_working_bytes, "presupuesto de lectura", 1024, MAX_STATE_BYTES)
        if type(self.refinements) is not int or self.refinements not in (1, 2, 4):
            raise ValueError("K debe ser 1, 2 o 4")
        if self.mode not in ("bank", "no_bank"):
            raise ValueError("El modo de lectura debe ser bank o no_bank")
        if self.episode_selection not in EPISODE_SELECTIONS:
            raise ValueError("La selección de episodios debe ser per_step o first_read")
        if (
            type(self.refinement_initialization) is not str
            or self.refinement_initialization not in REFINEMENT_INITIALIZATIONS
        ):
            raise ValueError("La inicialización del refinador no está admitida")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 1e-4 <= self.temperature <= 100
        ):
            raise ValueError("La temperatura debe ser finita y pertenecer a [1e-4, 100]")
        object.__setattr__(self, "temperature", float(self.temperature))
        # Sin banco no hay episodios que pesar, y las lecturas regionales no tienen una sola
        # familiaridad por fila.
        if self.familiarity_gate is not None and (
            type(self.familiarity_gate) is not str
            or self.familiarity_gate not in FAMILIARITY_GATES
            or self.mode != "bank"
            or self.evidence_bridge is not None
        ):
            raise ValueError(
                "R3 declara la puerta familiarity o constant sobre el banco y sin puente regional"
            )
        if self.evidence_bridge is not None:
            from .evidence_bridge import EvidenceBridgeConfig

            if type(self.evidence_bridge) is not EvidenceBridgeConfig:
                raise ValueError("El puente necesita EvidenceBridgeConfig explícito")
            self.evidence_bridge.__post_init__()
            for name in (
                "codec_id",
                "hidden_size",
                "neighbors",
                "temperature",
                "max_batch",
                "max_working_bytes",
            ):
                if getattr(self.evidence_bridge, name) != getattr(self, name):
                    raise ValueError(
                        "El puente y lector deben compartir formas, codec y presupuestos"
                    )

    def identity(self):
        fields = asdict(self)
        # El modo anterior conserva literalmente su identidad y su huella.
        episodes = fields.pop("episode_selection")
        initialization = fields.pop("refinement_initialization")
        gate = fields.pop("familiarity_gate")
        fields.pop("evidence_bridge")
        result = dict(
            **fields,
            schema_version=1,
            query_normalization="scaled_l2_eps_1e-12",
            selection="global_each_step_fp64_stable_low_id",
            selection_gradient=False,
            attention="model_dtype_softmax",
            values="fixed64_plus_mature_label",
            refinement="residual_sigmoid_scalar_tanh_concat_state_base_read_presence",
            bias=True,
            initial_step=0.1,
            dropout=0.0,
        )
        if episodes == "first_read":
            result.update(
                episode_selection=episodes,
                selection="global_first_read_fp64_stable_low_id_then_fixed_positions",
            )
        if initialization != "pytorch_default":
            result["refinement_initialization"] = initialization
        if gate is not None:
            result.update(
                familiarity_gate=gate,
                familiarity="attention_weighted_cosine_of_query_and_read_episodes_no_gradient",
                gate="two_sigmoid_of_weight_times_familiarity_plus_bias"
                if gate == "familiarity"
                else "two_sigmoid_of_bias",
                gate_initialization="zero_parameters_gate_one_reference_reader",
                gated="tanh_change_of_each_refinement_before_its_step",
                without_episodes="reference_step",
            )
        if self.evidence_bridge is not None:
            result.update(
                schema_version=2,
                evidence_bridge=self.evidence_bridge.identity(),
                attention="regional_weights_in_separate_typed_result",
                selection="common_bridge_query_fp64_stable_low_id_" + episodes,
                projection_initialization="paired_legacy_query_value_and_refiner",
                regional_availability="all_configured_views",
            )
            if self.evidence_bridge.intervention is not None:
                result["regional_availability"] = "fixed_declared_active_views"
        return result


@dataclass(frozen=True)
class EpisodeRead:
    values: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    presence: torch.Tensor
    # Solo con la puerta de R3: [filas, 1], el coseno medio de la lectura sin gradiente.
    familiarity: torch.Tensor | None = None


@dataclass(frozen=True)
class _SelectedEpisodes:
    keys: torch.Tensor
    projected: torch.Tensor
    ids: torch.Tensor


@dataclass(frozen=True)
class ReadoutResult:
    state: torch.Tensor
    reads: tuple[EpisodeRead, ...]
    # El estado tras cada refinamiento, en orden. El último es `state`.
    states: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class EpisodicPrediction:
    point_predictions: torch.Tensor
    readout: "ReadoutResult | BridgedReadoutResult | None"
    # Solo con `quantile_head_v1`: [flujos, 5]. La predicción puntual es su mediana.
    quantiles: torch.Tensor | None = None


def _normalize(value):
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / scale.clamp_min(1e-12)
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / norm.clamp_min(1e-12 / scale.clamp_min(1e-12))


class EpisodicReadout(nn.Module):
    def __init__(self, config, *, device="cpu", dtype=torch.float32):
        super().__init__()
        if not isinstance(config, EpisodicReadoutConfig) or dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("La configuración y precisión del lector no son válidas")
        self.config = config
        device = _device(device)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.query_projection = nn.Linear(config.hidden_size, 64, dtype=dtype)
            self.value_projection = nn.Linear(65, config.hidden_size, dtype=dtype)
            self.refinement = nn.Linear(3 * config.hidden_size + 1, config.hidden_size, dtype=dtype)
            self.step_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=dtype))
            if config.refinement_initialization == "zero_refinement":
                # Se conservan los sorteos de las proyecciones y la ruta anterior del RNG.
                nn.init.zeros_(self.refinement.weight)
                nn.init.zeros_(self.refinement.bias)
            if config.familiarity_gate is not None:
                # Con a = b = 0 la puerta vale 2σ(0) = 1. Los ceros no consumen el generador.
                self.familiarity_bias = nn.Parameter(torch.zeros((), dtype=dtype))
                if config.familiarity_gate == "familiarity":
                    self.familiarity_weight = nn.Parameter(torch.zeros((), dtype=dtype))
        self.evidence_bridge = None
        if config.evidence_bridge is not None:
            from .evidence_bridge import SharedEvidenceBridge

            self.evidence_bridge = SharedEvidenceBridge(config.evidence_bridge, dtype=dtype)
            self.evidence_bridge.query.load_state_dict(self.query_projection.state_dict())
            self.evidence_bridge.value.load_state_dict(self.value_projection.state_dict())
            del self.query_projection, self.value_projection
        self.to(device=device, dtype=dtype)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), dtype=str(self.step_logit.dtype))

    def set_extra_state(self, value):
        require_identity(value, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        expected = self.state_dict()
        supplied = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if set(supplied) != set(expected):
            raise ValueError("El archivo no conserva todos los campos del lector")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        if self.evidence_bridge is not None:
            self.evidence_bridge.validate_payload(state_dict, prefix + "evidence_bridge.")
        for name, value in self.named_parameters():
            other = supplied[name]
            if (
                not isinstance(other, torch.Tensor)
                or other.shape != value.shape
                or other.dtype != value.dtype
                or other.layout != torch.strided
            ):
                raise ValueError("Los parámetros del lector no conservan forma o precisión")
            check_finite(other, "Los parámetros del lector")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _state(self, value):
        if self.step_logit.dtype not in (torch.float32, torch.float64):
            raise ValueError("El lector admite FP32 o FP64")
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.layout != torch.strided
            or value.shape[1] != self.config.hidden_size
            or value.dtype != self.step_logit.dtype
            or value.device != self.step_logit.device
        ):
            raise ValueError("El estado de trabajo no conserva forma, precisión o dispositivo")
        bounded_integer(len(value), "lote", 1, self.config.max_batch)
        check_finite(value, "El estado de trabajo")

    def estimated_bytes(self, batch, episodes, *, differentiable=False, refinements=None):
        check_differentiable(differentiable)
        refinements = self.config.refinements if refinements is None else refinements
        bounded_integer(refinements, "refinamientos estimados", 1, 9)
        bounded_integer(batch, "lote", 1, self.config.max_batch)
        bounded_integer(episodes, "episodios", 0, 1024)
        width, neighbors, size = (
            self.config.hidden_size,
            min(episodes, self.config.neighbors),
            self.step_logit.element_size(),
        )
        selection = batch * episodes * 32
        per_step = batch * (
            neighbors * (64 * (8 + size) + (65 + width) * size + 32) + (12 * width + 128) * size
        )
        if self.config.familiarity_gate is not None:
            # Cosenos, su producto por los pesos, familiaridad, puerta y cambio escalado.
            per_step += batch * (2 * neighbors + width + 4) * size
        total = 65536 + selection + per_step * refinements * (4 if differentiable else 1)
        if self.evidence_bridge is not None:
            total += refinements * self.evidence_bridge.estimated_bytes(
                batch, episodes, differentiable=differentiable
            )
        return total

    def read(self, state, snapshot=None, *, context_id=None, cutoff=None):
        return self._read(state, snapshot, context_id, cutoff)[0]

    def node_read_limit(self, episodes):
        """Límite por fila del evento completo, sin renovar cuotas por ruta o bloque."""
        bridge = self.config.evidence_bridge
        if bridge is None or episodes == 0:
            return 38
        recurrence = bridge.hierarchy if bridge.hierarchy is not None else bridge.recurrence
        if recurrence is None:
            return 38
        return recurrence.max_node_updates_per_row // recurrence.node_updates(
            len(bridge.regions), 1
        )

    def preflight_node_updates(self, reads, episodes):
        """Admitir todas las lecturas previstas antes de ejecutar la primera MAC."""
        bounded_integer(reads, "lecturas del evento", 0, 38)
        bounded_integer(episodes, "episodios", 0, 1024)
        if reads > self.node_read_limit(episodes):
            raise ValueError("El presupuesto regional del evento no cabe antes de MAC")

    def recurrence_work(self, value_projections, *, selected_steps=None, evaluated_steps=None):
        """Contar R0 y actualizaciones nuevas J, sin atribuirles todos los FLOPs."""
        bridge = self.config.evidence_bridge
        adaptive = (
            bridge is not None
            and bridge.recurrence is not None
            and bridge.recurrence.stopping is not None
        )
        if not adaptive and (selected_steps is not None or evaluated_steps is not None):
            raise ValueError("El trabajo observado necesita una parada adaptativa explícita")
        if bridge is None:
            return {}
        if bridge.hierarchy is not None:
            bounded_integer(value_projections, "proyecciones del evento", 0, 38 * 8192)
            return dict(
                bridge_initial_views=4 * value_projections,
                bridge_local_updates=8 * bridge.hierarchy.local_steps * value_projections,
                bridge_group_updates=8 * value_projections,
                bridge_root_updates=2 * value_projections,
                bridge_node_updates=bridge.hierarchy.node_updates(4, 1) * value_projections,
            )
        if bridge.recurrence is None:
            return {}
        bounded_integer(value_projections, "proyecciones del evento", 0, 38 * 8192)
        initial = len(bridge.regions) * value_projections
        stopping = bridge.recurrence.stopping
        if stopping is not None:
            if value_projections == 0:
                for value in (selected_steps, evaluated_steps):
                    if value is not None and (type(value) is not int or value != 0):
                        raise ValueError("Una lectura vacía no declara vueltas adaptativas")
                selected_steps = evaluated_steps = 0
            else:
                maximum = value_projections * bridge.recurrence.steps
                bounded_integer(selected_steps, "vueltas seleccionadas", value_projections, maximum)
                bounded_integer(evaluated_steps, "vueltas evaluadas", value_projections, maximum)
                if bridge.recurrence.steps == 4 and selected_steps == maximum - 1:
                    raise ValueError("La suma seleccionada no corresponde a prefijos J1, J2 o J4")
                expected = (
                    maximum if stopping.backend == "adaptive_reference_full" else selected_steps
                )
                if evaluated_steps != expected:
                    raise ValueError("Las vueltas evaluadas no corresponden al backend declarado")
            return dict(
                bridge_initial_views=initial,
                bridge_selected_node_updates=len(bridge.regions) * selected_steps,
                bridge_node_updates=len(bridge.regions) * evaluated_steps,
            )
        return dict(
            bridge_initial_views=initial, bridge_node_updates=initial * bridge.recurrence.steps
        )

    def _read(self, state, snapshot, context_id, cutoff, positions=None, *, repeated=False):
        """Leer y devolver también las posiciones elegidas en la instantánea.

        Con `positions` se atienden esos episodios con la consulta actual, sin seleccionar.
        Con `repeated`, el primer paso de la misma llamada ya validó la instantánea y el
        presupuesto, que no cambian entre pasos, y el estado sale del refinamiento anterior,
        que conserva su forma, precisión y dispositivo y ya se comprobó finito.
        """
        read, indices, _ = self._read_parts(
            state, snapshot, context_id, cutoff, positions, repeated=repeated
        )
        return read, indices

    def _read_parts(
        self, state, snapshot, context_id, cutoff, positions=None, *, repeated=False, cached=None
    ):
        """Conservar proyecciones solo durante una decisión con pesos e instantánea fijos."""
        if not repeated:
            self._validate_read(state, snapshot, context_id, cutoff)
        if self.evidence_bridge is not None:
            from .regional_episode_readout import regional_read

            if cached is not None:
                raise ValueError("Las rutas con caché anterior no admiten vistas regionales")
            return regional_read(self, state, snapshot, context_id, cutoff, positions)
        count = 0 if snapshot is None else min(snapshot.count, self.config.neighbors)
        gated = self.config.familiarity_gate == "familiarity"
        if count == 0:
            return (
                EpisodeRead(
                    torch.zeros_like(state),
                    state.new_empty((len(state), 0)),
                    torch.empty((len(state), 0), dtype=torch.int64, device=state.device),
                    torch.zeros((len(state), 1), dtype=torch.bool, device=state.device),
                    # Sin episodios la puerta no actúa y la familiaridad queda en cero.
                    state.new_zeros((len(state), 1)) if gated else None,
                ),
                None,
                None,
            )
        raw = self.query_projection(state)
        check_finite(raw, "La consulta episódica")
        query = _normalize(raw)
        if cached is not None:
            similarity = (cached.keys * query.unsqueeze(1)).sum(-1)
            logits = similarity / self.config.temperature
            weights = logits.softmax(-1)
            read = (cached.projected * weights.unsqueeze(-1)).sum(1)
            check_finite(read, "La lectura episódica")
            return (
                EpisodeRead(
                    read,
                    weights,
                    cached.ids,
                    torch.ones((len(state), 1), dtype=torch.bool, device=state.device),
                    _familiarity(similarity, weights) if gated else None,
                ),
                None,
                cached,
            )
        keys, values, labels, ids = snapshot._values[:4]
        if positions is None:
            with torch.no_grad():
                scores = query.double() @ keys.T
                indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count]
        elif (
            not isinstance(positions, torch.Tensor)
            or positions.dtype != torch.int64
            or positions.shape != (len(state), count)
            or positions.device != state.device
        ):
            raise ValueError("Las posiciones fijadas no corresponden a la lectura")
        else:
            indices = positions
        flat = indices.flatten()
        selected_keys = (
            keys.index_select(0, flat).reshape(len(state), count, 64).to(dtype=state.dtype)
        )
        similarity = (selected_keys * query.unsqueeze(1)).sum(-1)
        logits = similarity / self.config.temperature
        weights = logits.softmax(-1)
        selected_values = torch.cat(
            (
                values.index_select(0, flat),
                labels.index_select(0, flat).to(dtype=state.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        check_finite(selected_values, "Los valores y etiquetas seleccionados")
        projected = self.value_projection(selected_values).reshape(len(state), count, -1)
        check_finite(projected, "Los valores proyectados")
        read = (projected * weights.unsqueeze(-1)).sum(1)
        check_finite(read, "La lectura episódica")
        selected_ids = ids.index_select(0, flat).reshape(len(state), count)
        return (
            EpisodeRead(
                read,
                weights,
                selected_ids,
                torch.ones((len(state), 1), dtype=torch.bool, device=state.device),
                _familiarity(similarity, weights) if gated else None,
            ),
            indices,
            _SelectedEpisodes(selected_keys, projected, selected_ids),
        )

    def _validate_read(self, state, snapshot, context_id, cutoff, *, refinements=None):
        self._state(state)
        episodes = snapshot.count if isinstance(snapshot, EpisodeSnapshot) else 0
        self.preflight_node_updates(
            self.config.refinements if refinements is None else refinements, episodes
        )
        if (
            self.estimated_bytes(
                len(state),
                episodes,
                differentiable=torch.is_grad_enabled(),
                refinements=refinements,
            )
            > self.config.max_working_bytes
        ):
            raise ValueError("La lectura supera el presupuesto de trabajo estimado")
        if self.config.mode == "no_bank":
            if snapshot is not None:
                raise ValueError("El control sin banco no acepta episodios implícitos")
        elif not isinstance(snapshot, EpisodeSnapshot):
            raise ValueError("La lectura requiere una instantánea explícita")
        if snapshot is not None:
            snapshot.check(
                codec_id=self.config.codec_id,
                context_id=context_id,
                cutoff=cutoff,
                dtype=state.dtype,
                device=state.device,
            )

    def refine(self, state, base, read):
        check_finite(self.step_logit, "La puerta de refinamiento")
        self._state(state)
        self._state(base)
        from .regional_episode_readout import RegionalEpisodeRead

        if (
            not isinstance(read, (EpisodeRead, RegionalEpisodeRead))
            or read.values.shape != state.shape
            or base.shape != state.shape
        ):
            raise ValueError("La lectura no corresponde al estado de trabajo")
        self._state(read.values)
        if (
            read.presence.shape != (len(state), 1)
            or read.presence.dtype != torch.bool
            or read.presence.device != state.device
        ):
            raise ValueError("La presencia de memoria no corresponde al lote")
        # Sin indexar con la máscara, que obligaría a esperar a la GPU para contar las filas.
        require_true(
            ~((read.values != 0) & ~read.presence).any(),
            "Una lectura ausente necesita relleno cero",
        )
        if self.config.familiarity_gate == "familiarity":
            familiarity = getattr(read, "familiarity", None)
            if (
                not isinstance(familiarity, torch.Tensor)
                or familiarity.shape != (len(state), 1)
                or familiarity.dtype != state.dtype
                or familiarity.device != state.device
            ):
                raise ValueError("R3 necesita la familiaridad de cada fila de la lectura")
            check_finite(familiarity, "La familiaridad de la lectura")
        return self._step(state, base, read)

    def _step(self, state, base, read):
        """Refinar con una lectura ya validada, sin repetir sus comprobaciones."""
        joined = torch.cat((state, base, read.values, read.presence.to(dtype=state.dtype)), dim=-1)
        update = self.refinement(joined)
        check_finite(update, "La actualización de trabajo")
        change = update.tanh()
        if self.config.familiarity_gate is not None:
            # La puerta escala el cambio antes del paso. Con la puerta en uno el producto es
            # exacto y el estado y los gradientes compartidos coinciden con la referencia.
            change = self._familiarity_gate(read) * change
        result = state + self.step_logit.sigmoid() * change
        check_finite(result, "El estado refinado")
        return result

    def _familiarity_gate(self, read):
        """2σ(a·f + b) por fila, 2σ(b) en el control y 1 en las filas sin episodios."""
        logit = self.familiarity_bias
        if self.config.familiarity_gate == "familiarity":
            familiarity = getattr(read, "familiarity", None)
            if familiarity is None:
                raise ValueError("R3 necesita la familiaridad de cada fila de la lectura")
            logit = self.familiarity_weight * familiarity + logit
        gate = 2 * logit.sigmoid()
        return torch.where(read.presence, gate, torch.ones_like(gate))

    def forward(self, z_base, snapshot=None, *, context_id=None, cutoff=None, differentiable=False):
        """Refinar el estado de trabajo con K lecturas del banco.

        Las comprobaciones de datos en CUDA de todos los refinamientos se resuelven juntas al
        final, con el mismo error y mensaje que en su línea, como en `prepare`.
        """
        with deferred_checks():
            return self._forward(z_base, snapshot, context_id, cutoff, differentiable)

    def _forward(self, z_base, snapshot, context_id, cutoff, differentiable):
        check_differentiable(differentiable)
        if differentiable and torch.is_inference_mode_enabled():
            raise ValueError("inference_mode no permite la diferenciación solicitada")
        self._state(z_base)
        episodes = snapshot.count if isinstance(snapshot, EpisodeSnapshot) else 0
        self.preflight_node_updates(self.config.refinements, episodes)
        if (
            self.estimated_bytes(len(z_base), episodes, differentiable=differentiable)
            > self.config.max_working_bytes
        ):
            raise ValueError("La lectura supera el presupuesto de trabajo estimado")
        fixed = self.config.episode_selection == "first_read"
        with torch.set_grad_enabled(differentiable):
            base = z_base if differentiable else z_base.detach()
            state, reads, states, positions = base, [], [], None
            for step in range(self.config.refinements):
                read, chosen = self._read(
                    state, snapshot, context_id, cutoff, positions, repeated=step > 0
                )
                if fixed:
                    positions = chosen
                if step == 0:
                    # Los parámetros no cambian dentro de la llamada. Se comprueban una vez, en
                    # el mismo orden que `refine`, después de la primera lectura.
                    check_finite(self.step_logit, "La puerta de refinamiento")
                # `_read` construye cada lectura con la forma, el tipo y el dispositivo del
                # estado, la deja comprobada finita y rellena con ceros las filas ausentes, así
                # que el refinamiento no vuelve a validarla.
                state = self._step(state, base, read)
                reads.append(read)
                states.append(state)
        if self.evidence_bridge is not None:
            from .regional_episode_readout import BridgedReadoutResult

            return BridgedReadoutResult(state, tuple(reads), tuple(states))
        return ReadoutResult(state, tuple(reads), tuple(states))


def _familiarity(similarity, weights):
    """Coseno medio entre la consulta y los episodios de la lectura, con sus pesos.

    No lleva gradiente: la puerta aprende a usar la familiaridad, pero la consulta no se
    ajusta para fabricarla.
    """
    with torch.no_grad():
        familiarity = (weights * similarity).sum(-1, keepdim=True)
    check_finite(familiarity, "La familiaridad de la lectura")
    return familiarity


def apply_episodic_readout(
    prepared,
    head,
    extension=None,
    snapshot=None,
    *,
    context_id=None,
    cutoff=None,
    differentiable=False,
):
    """Aplicar la cabeza del núcleo al estado refinado por la lectura episódica.

    Una cabeza escalar debe devolver un valor por flujo. `QuantileHead` devuelve
    [flujos, 5] cuantiles ordenados y la predicción puntual es su mediana. Las
    comprobaciones de la lectura y de la cabeza se resuelven juntas antes de devolver.
    """
    with deferred_checks():
        return _apply_episodic_readout(
            prepared, head, extension, snapshot, context_id, cutoff, differentiable
        )


def _apply_episodic_readout(
    prepared, head, extension, snapshot, context_id, cutoff, differentiable
):
    check_differentiable(differentiable)
    if not isinstance(prepared, PreparedDecisions):
        raise ValueError("Falta la preparación previa del núcleo")
    if isinstance(head, QuantileHead) != (prepared.quantiles is not None):
        raise ValueError("La cabeza no corresponde a la salida preparada por el núcleo")
    if extension is None:
        if snapshot is not None:
            raise ValueError("Una ampliación ausente no consume un banco")
        return EpisodicPrediction(prepared.point_predictions, None, prepared.quantiles)
    if not isinstance(extension, EpisodicReadout):
        raise ValueError("La ampliación no es un lector identificado")
    result = extension(
        prepared.working_state,
        snapshot,
        context_id=context_id,
        cutoff=cutoff,
        differentiable=differentiable,
    )
    with torch.set_grad_enabled(differentiable):
        predictions, quantiles = head_predictions(head, result.state)
    return EpisodicPrediction(predictions, result, quantiles)


def head_predictions(head, state):
    """Predicción puntual y, con `QuantileHead`, cuantiles de un estado de trabajo."""
    quantiles = None
    output = head(state)
    if isinstance(head, QuantileHead):
        if output.shape != (len(state), len(LEVELS)):
            raise ValueError("La cabeza de cuantiles debe devolver cinco niveles por flujo")
        check_finite(output, "Los cuantiles refinados")
        quantiles, predictions = output, median(output)
    else:
        predictions = output.squeeze(-1)
    if predictions.shape != (len(state),):
        raise ValueError("La cabeza debe devolver una salida escalar por flujo")
    check_finite(predictions, "La predicción refinada")
    return predictions, quantiles


def copy_readout_parameters(source, target):
    """Emparejar controles por copia explícita sin relajar la carga de contratos."""
    if not isinstance(source, EpisodicReadout) or not isinstance(target, EpisodicReadout):
        raise ValueError("La copia requiere dos lectores episódicos")

    def identity(model):
        return {
            key: value
            for key, value in model.config.identity().items()
            if key not in {"mode", "seed", "episode_selection", "selection"}
        }

    require_identity(identity(source), identity(target))
    if source.step_logit.dtype != target.step_logit.dtype:
        raise ValueError("La copia no convierte precisión")
    original, destination = dict(source.named_parameters()), dict(target.named_parameters())
    for value in original.values():
        check_finite(value, "Los parámetros de origen")
    with torch.no_grad():
        for name, value in destination.items():
            value.copy_(original[name])
    fingerprints = {}
    for name, value in destination.items():
        actual = value.detach().cpu().contiguous().numpy().tobytes()
        expected = original[name].detach().cpu().contiguous().numpy().tobytes()
        if actual != expected:
            raise ValueError("La copia emparejada cambió los parámetros")
        fingerprints[name] = hashlib.sha256(actual).hexdigest()
    return dict(
        source=source.get_extra_state(),
        target=target.get_extra_state(),
        parameters_sha256=fingerprints,
        snapshot_transferred=False,
    )
