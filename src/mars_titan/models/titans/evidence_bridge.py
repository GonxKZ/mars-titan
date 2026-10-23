"""Vistas regionales de un único snapshot episódico, sin otra memoria persistente."""

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .bridge_hierarchy import (
    BridgeHierarchyConfig,
    hierarchy_graph,
    hierarchy_mix,
    hierarchy_options,
)
from .bridge_intervention import BridgeInterventionConfig, intervention_options
from .bridge_recurrence import BridgeRecurrenceConfig, nonlinear_mix
from .config import MAX_BLOCK_ROWS, MAX_STATE_BYTES, bounded_integer, require_identity
from .episodic_readout import _normalize
from .episodic_snapshot import EpisodeSnapshot, _device, _digest_id
from .state import check_finite, deferred_checks

KIND = "shared_evidence_bridge_v1"
BRIDGE_VIEW_FIELDS = ("regions", "topology", "cycles", "bridge_gain", "seed")


def bridge_view_options(value):
    """Validar las vistas y devolver sus opciones propias, sin codec ni formas del lector."""
    if type(value) is not dict or set(value) - {"recurrence", "intervention", "hierarchy"} != set(
        BRIDGE_VIEW_FIELDS
    ):
        raise ValueError("El puente declara regiones, topología, ciclos, ganancia y semilla")
    regions = value["regions"]
    if type(regions) not in (list, tuple):
        raise ValueError("Las regiones del puente necesitan una lista o tupla")
    regions = tuple(regions)
    if (
        not 1 <= len(regions) <= 8
        or any(type(name) is not str or not name or len(name) > 64 for name in regions)
        or len(set(regions)) != len(regions)
    ):
        raise ValueError("Se necesitan entre una y ocho regiones distintas identificadas")
    topology = value["topology"]
    if type(topology) is not str or topology not in ("self", "ring", "complete"):
        raise ValueError("La topología de puentes debe ser self, ring o complete")
    bounded_integer(value["cycles"], "ciclos de puente", 0, 4)
    bounded_integer(value["seed"], "semilla de puente", 0, 2**32 - 1)
    gain = value["bridge_gain"]
    if type(gain) not in (int, float) or not math.isfinite(gain) or not 0 <= gain <= 1:
        raise ValueError("La ganancia del puente debe ser finita y pertenecer a [0, 1]")
    result = {name: value[name] for name in BRIDGE_VIEW_FIELDS}
    result.update(regions=regions, bridge_gain=float(gain))
    recurrence = value.get("recurrence")
    if recurrence is not None:
        if type(recurrence) is not dict or set(recurrence) != {"steps", "max_node_updates_per_row"}:
            raise ValueError("La recurrencia declara J y su presupuesto explícitos")
        checked = BridgeRecurrenceConfig(**recurrence)
        if value["cycles"] != 0:
            raise ValueError("La recurrencia J requiere cycles=0")
        result["recurrence"] = checked.options()
    intervention = value.get("intervention")
    if intervention is not None:
        checked = intervention_options(intervention)
        checked.edges(len(regions), topology)
        result["intervention"] = asdict(checked)
    hierarchy = value.get("hierarchy")
    if hierarchy is not None:
        checked = hierarchy_options(hierarchy)
        _check_hierarchy(len(regions), topology, value["cycles"], recurrence, intervention)
        result["hierarchy"] = asdict(checked)
    return result


def _check_hierarchy(regions, topology, cycles, recurrence, intervention):
    if regions != 4 or topology != "self" or cycles != 0:
        raise ValueError("La jerarquía requiere cuatro vistas, topology=self y cycles=0")
    if recurrence is not None or intervention is not None:
        raise ValueError("La jerarquía no compone recurrencia plana ni lesiones fijas")


@dataclass(frozen=True)
class EvidenceBridgeConfig:
    codec_id: str
    hidden_size: int = 64
    regions: tuple[str, ...] = ("view_1", "view_2", "view_3")
    neighbors: int = 8
    topology: str = "ring"
    cycles: int = 1
    bridge_gain: float = 0.25
    temperature: float = 1.0
    seed: int = 42
    max_batch: int = 256
    max_working_bytes: int = 32 * 1024**2
    recurrence: BridgeRecurrenceConfig | None = None
    intervention: BridgeInterventionConfig | None = None
    hierarchy: BridgeHierarchyConfig | None = None

    def __post_init__(self):
        _digest_id(self.codec_id, "El codec de evidencia")
        bounded_integer(self.hidden_size, "dimensión regional", 1, 128)
        bounded_integer(self.neighbors, "vecinos comunes", 1, 8)
        bounded_integer(self.max_batch, "lote de puente", 1, MAX_BLOCK_ROWS)
        bounded_integer(self.max_working_bytes, "presupuesto de puente", 1024, MAX_STATE_BYTES)
        if type(self.regions) is not tuple:
            raise ValueError("Se necesitan entre una y ocho regiones distintas identificadas")
        bridge_view_options({name: getattr(self, name) for name in BRIDGE_VIEW_FIELDS})
        if self.hierarchy is not None:
            if type(self.hierarchy) is not BridgeHierarchyConfig:
                raise ValueError("La jerarquía necesita BridgeHierarchyConfig explícito")
            self.hierarchy.__post_init__()
            _check_hierarchy(
                len(self.regions), self.topology, self.cycles, self.recurrence, self.intervention
            )
        if self.intervention is not None:
            if type(self.intervention) is not BridgeInterventionConfig:
                raise ValueError("La intervención necesita BridgeInterventionConfig explícito")
            self.intervention.__post_init__()
            self.intervention.edges(len(self.regions), self.topology)
        if self.recurrence is not None:
            if type(self.recurrence) is not BridgeRecurrenceConfig:
                raise ValueError("La recurrencia necesita BridgeRecurrenceConfig explícito")
            self.recurrence.__post_init__()
            if self.recurrence.stopping is not None:
                raise ValueError("La parada residual requiere la lectura adaptativa explícita")
            if self.cycles != 0:
                raise ValueError("La recurrencia J requiere cycles=0, sin composición implícita")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or not 1e-4 <= self.temperature <= 100.0
        ):
            raise ValueError("La ganancia o temperatura del puente no es admisible")

    def identity(self):
        fields = asdict(self)
        fields.pop("recurrence")
        fields.pop("intervention")
        fields.pop("hierarchy")
        result = dict(
            schema_version=1,
            kind=KIND,
            **fields,
            evidence="one_mature_snapshot_one_common_candidate_selection",
            regional_views="own_query_weights_and_value_transform",
            bridge="convex_messages_between_derived_views",
            storage="no_persistent_episode_copy_or_state",
            semantics="region_names_do_not_attest_specialization_or_historical_sectors",
        )
        if self.recurrence is not None:
            result.update(
                schema_version=2,
                kind="shared_evidence_bridge_nonlinear_v1",
                bridge="synchronous_receiver_weights_reused_mixture_tanh",
                recurrence=self.recurrence.identity(),
            )
        if self.intervention is not None:
            result.update(
                schema_version=3,
                base_kind=result["kind"],
                kind="shared_evidence_bridge_intervention_v1",
                intervention=asdict(self.intervention),
            )
        if self.hierarchy is not None:
            result.update(
                schema_version=4,
                kind="shared_evidence_bridge_hierarchy_v1",
                bridge="fixed_group_root_group_local_two_rounds",
                hierarchy=self.hierarchy.identity(),
            )
        return result

    def active_views(self):
        """Distinguir lectores activos de la existencia física de episodios."""
        return (
            (True,) * len(self.regions)
            if self.intervention is None
            else self.intervention.active_regions
        )


@dataclass(frozen=True)
class BridgeRead:
    values: torch.Tensor
    region_values: torch.Tensor
    evidence_ids: torch.Tensor
    evidence_weights: torch.Tensor
    bridge_weights: torch.Tensor
    snapshot_sha256: str
    selected_positions: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True)
class HierarchyBridgeRead(BridgeRead):
    group_values: torch.Tensor
    root_values: torch.Tensor
    hierarchy_weights: torch.Tensor


class SharedEvidenceBridge(nn.Module):
    """Leer [B,H] desde un pool común y comunicar sus vistas mediante un grafo pequeño.

    Los valores seleccionados se proyectan una vez por fila, sin eje regional.
    Cada región atiende ese pool y transforma su resumen con parámetros propios.
    Los IDs identifican evidencia, no una explicación causal de la predicción.
    """

    def __init__(self, config, *, device="cpu", dtype=torch.float32):
        super().__init__()
        if type(config) is not EvidenceBridgeConfig or dtype not in (torch.float32, torch.float64):
            raise ValueError("El puente necesita configuración y precisión explícitas")
        config.__post_init__()
        device = _device(device)
        self.config = config
        width, regions = config.hidden_size, len(config.regions)
        derived = int.from_bytes(
            hashlib.sha256(f"{KIND}/{config.seed}".encode()).digest()[:7], "little"
        )
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(derived)
            self.query = nn.Linear(width, 64, dtype=dtype)
            self.value = nn.Linear(65, width, dtype=dtype)
            self.region_query = nn.Parameter(
                torch.randn(regions, width, 64, dtype=dtype) / math.sqrt(width)
            )
            self.region_value = nn.Parameter(
                torch.randn(regions, width, width, dtype=dtype) / math.sqrt(width)
            )
            self.output = nn.Linear(regions * width, width, bias=False, dtype=dtype)
            edges = torch.eye(regions, dtype=torch.bool)
            if config.topology == "complete":
                edges[:] = True
            elif config.topology == "ring":
                positions = torch.arange(regions)
                edges[positions, (positions + 1) % regions] = True
            self.register_buffer("edges", edges)
            if config.intervention is not None:
                self.edges.copy_(torch.tensor(config.intervention.edges(regions, config.topology)))
                self.register_buffer("active_regions", torch.tensor(config.active_views()))
            if config.hierarchy is not None:
                self.register_buffer("hierarchy_edges", torch.tensor(config.hierarchy.edges()))
        self.to(device=device)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), dtype=str(self.region_query.dtype))

    def set_extra_state(self, value):
        require_identity(value, self.get_extra_state())

    def validate_payload(self, state_dict, prefix=""):
        expected = self.state_dict()
        supplied = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        if set(supplied) != set(expected):
            raise ValueError("El checkpoint no conserva todos los campos del puente")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        for name, value in expected.items():
            if name == "_extra_state":
                continue
            other = supplied[name]
            if (
                not isinstance(other, torch.Tensor)
                or other.shape != value.shape
                or other.dtype != value.dtype
                or other.layout != torch.strided
            ):
                raise ValueError("El checkpoint del puente no conserva formas y precisión")
            check_finite(other, "El checkpoint del puente")
        declared_edges = expected["edges"].cpu()
        if self.config.intervention is not None:
            if any(supplied[name].dtype != torch.bool for name in ("edges", "active_regions")):
                raise ValueError("Los puentes y regiones declarados necesitan buffers booleanos")
            declared_edges = torch.tensor(
                self.config.intervention.edges(len(self.config.regions), self.config.topology)
            )
        if not torch.equal(supplied["edges"].cpu(), declared_edges):
            raise ValueError("El checkpoint cambió los puentes declarados")
        if self.config.intervention is not None and not torch.equal(
            supplied["active_regions"].cpu(), torch.tensor(self.config.active_views())
        ):
            raise ValueError("El checkpoint cambió las regiones activas declaradas")
        if self.config.hierarchy is not None:
            if (
                supplied["edges"].dtype != torch.bool
                or supplied["hierarchy_edges"].dtype != torch.bool
                or not torch.equal(supplied["edges"].cpu(), torch.eye(4, dtype=torch.bool))
                or not torch.equal(
                    supplied["hierarchy_edges"].cpu(), torch.tensor(self.config.hierarchy.edges())
                )
            ):
                raise ValueError("El checkpoint cambió la geometría jerárquica declarada")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        self.validate_payload(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def estimated_bytes(self, batch, episodes, *, differentiable=False):
        bounded_integer(batch, "lote estimado de puente", 1, self.config.max_batch)
        bounded_integer(episodes, "episodios de puente", 0, 1024)
        if type(differentiable) is not bool:
            raise ValueError("La estimación necesita un booleano de diferenciación")
        r, h, k = (
            len(self.config.regions),
            self.config.hidden_size,
            min(episodes, self.config.neighbors),
        )
        size = self.region_query.element_size()
        steps = (
            self.config.cycles
            if self.config.recurrence is None
            else 3 * self.config.recurrence.steps
        )
        working = batch * (
            episodes * 32
            + k * (64 * (8 + size) + (65 + h) * size + 32)
            + size * (r * (128 + k + h * (steps + 4)) + r * r)
        )
        if self.config.hierarchy is not None:
            # Estados completos retenidos y mensajes de todas las fases, además de G.
            phases = 2 * (3 + self.config.hierarchy.local_steps)
            working += batch * size * (49 * 3 + 7 * h * (4 + 3 * phases))
        return 131072 + working * (4 if differentiable else 1)

    def forward(self, state, snapshot, *, context_id, cutoff, available=None, positions=None):
        reference = self.region_query
        if (
            not isinstance(state, torch.Tensor)
            or state.layout != torch.strided
            or state.ndim != 2
            or state.shape[1] != self.config.hidden_size
            or state.dtype != reference.dtype
            or state.dtype not in (torch.float32, torch.float64)
            or state.device != reference.device
            or torch.is_autocast_enabled(state.device.type)
        ):
            raise ValueError(
                "El estado no corresponde a las formas, precisión o dispositivo del puente"
            )
        bounded_integer(len(state), "filas de puente", 1, self.config.max_batch)
        if type(snapshot) is not EpisodeSnapshot:
            raise ValueError("El puente requiere una instantánea episódica común")
        recurrence = self.config.recurrence
        hierarchy = self.config.hierarchy
        if (
            hierarchy is not None
            and snapshot.count > 0
            and hierarchy.node_updates(4, 1) > hierarchy.max_node_updates_per_row
        ):
            raise ValueError("La lectura supera su presupuesto de actualizaciones jerárquicas")
        if (
            recurrence is not None
            and snapshot.count > 0
            and recurrence.node_updates(len(self.config.regions), 1)
            > recurrence.max_node_updates_per_row
        ):
            raise ValueError("La lectura supera su presupuesto de actualizaciones regionales")
        if (
            self.estimated_bytes(len(state), snapshot.count, differentiable=torch.is_grad_enabled())
            > self.config.max_working_bytes
        ):
            raise ValueError("El puente supera su presupuesto de trabajo estimado")
        snapshot.check(
            codec_id=self.config.codec_id,
            context_id=context_id,
            cutoff=cutoff,
            dtype=state.dtype,
            device=state.device,
        )
        r, b, h = len(self.config.regions), len(state), self.config.hidden_size
        if available is None:
            available = torch.ones((b, r), dtype=torch.bool, device=state.device)
        if (
            not isinstance(available, torch.Tensor)
            or available.layout != torch.strided
            or available.shape != (b, r)
            or available.dtype != torch.bool
            or available.device != state.device
        ):
            raise ValueError("La máscara no corresponde a las regiones del puente")
        if self.config.intervention is not None:
            available = available & self.active_regions[None]
        count = min(snapshot.count, self.config.neighbors)
        if positions is not None:
            if (
                not isinstance(positions, torch.Tensor)
                or positions.layout != torch.strided
                or positions.shape != (b, count)
                or positions.dtype != torch.int64
                or positions.device != state.device
            ):
                raise ValueError("Las posiciones comunes no corresponden al snapshot")
            ordered = positions.sort(-1).values
            valid = ((positions >= 0) & (positions < snapshot.count)).all() & (
                ordered[:, 1:] != ordered[:, :-1]
            ).all()
            # Esta validación precede al gather para impedir un acceso inválido en CUDA.
            if not valid.item():
                raise ValueError("Las posiciones comunes se repiten o quedan fuera del snapshot")
        with deferred_checks():
            check_finite(state, "El estado del puente")
            for value in self.parameters():
                check_finite(value, "Los parámetros del puente")
            allowed = self.edges[None] & available[:, :, None] & available[:, None, :]
            graph = allowed.to(state.dtype) / allowed.sum(-1, keepdim=True).clamp_min(1)
            hierarchy_weights = None
            if hierarchy is not None:
                hierarchy_weights, _ = hierarchy_graph(
                    available, self.hierarchy_edges, feedback=hierarchy.feedback
                )
                hierarchy_weights = hierarchy_weights.to(state.dtype)
                graph = hierarchy_weights[:, :4, :4]
            if count == 0:
                if hierarchy is not None:
                    return HierarchyBridgeRead(
                        torch.zeros_like(state),
                        state.new_zeros((b, r, h)),
                        torch.empty((b, 0), dtype=torch.int64, device=state.device),
                        state.new_empty((b, r, 0)),
                        graph,
                        snapshot._digest,
                        group_values=state.new_zeros((b, 2, h)),
                        root_values=state.new_zeros((b, 1, h)),
                        hierarchy_weights=hierarchy_weights,
                    )
                return BridgeRead(
                    torch.zeros_like(state),
                    state.new_zeros((b, r, h)),
                    torch.empty((b, 0), dtype=torch.int64, device=state.device),
                    state.new_empty((b, r, 0)),
                    graph,
                    snapshot._digest,
                )
            raw = self.query(state)
            check_finite(raw, "La consulta común del puente")
            common = _normalize(raw)
            keys, values, labels, ids = snapshot._values[:4]
            if positions is None:
                with torch.no_grad():
                    positions = torch.argsort(
                        common.double() @ keys.T, dim=-1, descending=True, stable=True
                    )[:, :count]
            flat = positions.flatten()
            selected_keys = keys.index_select(0, flat).reshape(b, count, 64).to(state.dtype)
            selected = torch.cat(
                (
                    values.index_select(0, flat),
                    labels.index_select(0, flat).to(state.dtype).unsqueeze(-1),
                ),
                -1,
            )
            pool = self.value(selected).reshape(b, count, h)
            check_finite(pool, "La proyección común de evidencia")
            queries = _normalize(
                raw[:, None] + torch.einsum("bh,rhd->brd", state, self.region_query)
            )
            logits = torch.bmm(queries, selected_keys.transpose(1, 2)) / self.config.temperature
            check_finite(logits, "Los logits regionales")
            weights = logits.softmax(-1) * available[:, :, None]
            mixed = torch.bmm(weights, pool)
            check_finite(mixed, "La lectura regional de evidencia")
            transformed = torch.einsum("brh,rhd->brd", mixed, self.region_value)
            check_finite(transformed, "La transformación regional antes de tanh")
            local = torch.tanh(transformed)
            if hierarchy is not None:
                full = hierarchy_mix(
                    local,
                    hierarchy_weights,
                    self.region_value,
                    available,
                    local_steps=hierarchy.local_steps,
                    gain=self.config.bridge_gain,
                )
                local = full[:, :4]
            elif recurrence is None:
                for _ in range(self.config.cycles):
                    local = (
                        1 - self.config.bridge_gain
                    ) * local + self.config.bridge_gain * torch.bmm(graph, local)
            else:
                local = nonlinear_mix(
                    local,
                    graph,
                    self.region_value,
                    available,
                    steps=recurrence.steps,
                    gain=self.config.bridge_gain,
                )
            result = self.output(local.reshape(b, r * h))
            check_finite(local, "Las vistas regionales")
            check_finite(result, "La salida del puente")
            if hierarchy is not None:
                return HierarchyBridgeRead(
                    result,
                    local,
                    ids.index_select(0, flat).reshape(b, count),
                    weights,
                    graph,
                    snapshot._digest,
                    positions,
                    group_values=full[:, 4:6],
                    root_values=full[:, 6:7],
                    hierarchy_weights=hierarchy_weights,
                )
            return BridgeRead(
                result,
                local,
                ids.index_select(0, flat).reshape(b, count),
                weights,
                graph,
                snapshot._digest,
                positions,
            )
