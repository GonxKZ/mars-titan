"""Vistas regionales de un único snapshot episódico, sin otra memoria persistente."""

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .config import MAX_BLOCK_ROWS, MAX_STATE_BYTES, bounded_integer, require_identity
from .episodic_readout import _normalize
from .episodic_snapshot import EpisodeSnapshot, _device, _digest_id
from .state import check_finite, deferred_checks

KIND = "shared_evidence_bridge_v1"


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

    def __post_init__(self):
        _digest_id(self.codec_id, "El codec de evidencia")
        bounded_integer(self.hidden_size, "dimensión regional", 1, 128)
        bounded_integer(self.neighbors, "vecinos comunes", 1, 8)
        bounded_integer(self.cycles, "ciclos de puente", 0, 4)
        bounded_integer(self.seed, "semilla de puente", 0, 2**32 - 1)
        bounded_integer(self.max_batch, "lote de puente", 1, MAX_BLOCK_ROWS)
        bounded_integer(self.max_working_bytes, "presupuesto de puente", 1024, MAX_STATE_BYTES)
        if (
            type(self.regions) is not tuple
            or not 1 <= len(self.regions) <= 8
            or any(type(name) is not str or not name or len(name) > 64 for name in self.regions)
            or len(set(self.regions)) != len(self.regions)
        ):
            raise ValueError("Se necesitan entre una y ocho regiones distintas identificadas")
        if type(self.topology) is not str or self.topology not in ("self", "ring", "complete"):
            raise ValueError("La topología de puentes debe ser self, ring o complete")
        for name, lower, upper in (("bridge_gain", 0.0, 1.0), ("temperature", 1e-4, 100.0)):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("La ganancia o temperatura del puente no es admisible")

    def identity(self):
        return dict(
            schema_version=1,
            kind=KIND,
            **asdict(self),
            evidence="one_mature_snapshot_one_common_candidate_selection",
            regional_views="own_query_weights_and_value_transform",
            bridge="convex_messages_between_derived_views",
            storage="no_persistent_episode_copy_or_state",
            semantics="region_names_do_not_attest_specialization_or_historical_sectors",
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
        if not torch.equal(supplied["edges"].cpu(), expected["edges"].cpu()):
            raise ValueError("El checkpoint cambió los puentes declarados")

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
        working = batch * (
            episodes * 32
            + k * (64 * (8 + size) + (65 + h) * size + 32)
            + size * (r * (128 + k + h * (self.config.cycles + 4)) + r * r)
        )
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
            if count == 0:
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
            for _ in range(self.config.cycles):
                local = (1 - self.config.bridge_gain) * local + self.config.bridge_gain * torch.bmm(
                    graph, local
                )
            result = self.output(local.reshape(b, r * h))
            check_finite(local, "Las vistas regionales")
            check_finite(result, "La salida del puente")
            return BridgeRead(
                result,
                local,
                ids.index_select(0, flat).reshape(b, count),
                weights,
                graph,
                snapshot._digest,
                positions,
            )
