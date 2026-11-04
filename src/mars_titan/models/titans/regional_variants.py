"""Variantes del operador regional de #751 sobre los cinco bloques de fuente.

- C1 `region_axis_mixing_v1`: cinco cabezas de atención sobre los nodos, tantas como regiones,
  y mezcla 5×5 de sus logits antes del softmax y de sus pesos después, al modo de
  talking-heads (Shazeer et al., 2020). Cada fuente llega como un único vector, así que
  ninguna cabeza queda atada a una región. Su control (`mixing="none"`) conserva las mismas
  cabezas y parámetros, salvo las dos matrices de mezcla.
- C6 `shared_workspace_v1`: pocas ranuras compartidas sustituyen al resumen. Las regiones
  compiten por escribir en ellas y después todas leen lo escrito (Goyal et al., 2022). Su
  control (`write="dense"`) escribe con atención cruzada sobre todas las regiones presentes.

Las dos sustituyen a `RegionalFusion` como modulación de la fusión y repiten su calendario
de pasos locales y resumen por ciclo, sus máscaras y su lectura final. No modifican
`regional_fusion.py`, porque su código forma parte de la identidad de los consumidores
congelados y de los estudios ya preparados. El estado de trabajo se reinicia en cada
llamada y el forward no actualiza parámetros.
"""

import hashlib
import math
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from ..baselines.multimodal import MODALITIES
from .config import MAX_BLOCK_ROWS, bounded_integer, require_identity
from .regional_fusion import (
    RegionalFusionConfig,
    _topology_edges,
    validate_assignment,
    validate_assignment_request,
)
from .state import check_finite, deferred_checks

REGION_AXIS_MIXING = "region_axis_mixing_v1"
SHARED_WORKSPACE = "shared_workspace_v1"
REGIONS = len(MODALITIES)


def _operator(value):
    """El calendario común del operador, completo y con la ruta de atención."""
    if isinstance(value, dict):
        if set(value) != {field.name for field in fields(RegionalFusionConfig)}:
            raise ValueError("El operador de la variante debe declarar toda su configuración")
        value = RegionalFusionConfig(**value)
    if type(value) is not RegionalFusionConfig:
        raise ValueError("La variante regional necesita RegionalFusionConfig o sus valores")
    value.__post_init__()
    if value.routing != "attention":
        raise ValueError("Las variantes regionales solo admiten la ruta attention")
    return value


@dataclass(frozen=True)
class RegionAxisMixingConfig:
    operator: RegionalFusionConfig
    mixing: str = "talking_heads"

    def __post_init__(self):
        object.__setattr__(self, "operator", _operator(self.operator))
        if type(self.mixing) is not str or self.mixing not in ("talking_heads", "none"):
            raise ValueError("La mezcla entre regiones debe ser talking_heads o none")

    def identity(self):
        return dict(
            schema_version=1,
            kind=REGION_AXIS_MIXING,
            operator=asdict(self.operator),
            regions=list(MODALITIES),
            mixing=self.mixing,
            heads="as_many_as_regions_not_bound_to_sources",
            head_dim="hidden_size_floor_divided_by_regions",
            pre_softmax="masked_logits_set_to_zero_mixed_across_heads_then_masked",
            post_softmax="weights_mixed_across_heads_without_renormalization",
            initialization="identity_mixing_equal_to_the_unmixed_control",
            state="working_only_reset_per_prediction",
            parameter_updates="none_in_forward",
        )


@dataclass(frozen=True)
class SharedWorkspaceConfig:
    operator: RegionalFusionConfig
    slots: int = 2
    write: str = "competition"
    write_top_k: int | None = 2

    def __post_init__(self):
        object.__setattr__(self, "operator", _operator(self.operator))
        if not self.operator.summary:
            raise ValueError("El espacio de trabajo sustituye al resumen y necesita summary")
        bounded_integer(self.slots, "ranuras del espacio de trabajo", 1, 4)
        if type(self.write) is not str or self.write not in ("competition", "dense"):
            raise ValueError("La escritura debe ser competition o dense")
        if self.write == "dense":
            if self.write_top_k is not None:
                raise ValueError("La escritura densa no usa write_top_k")
        else:
            bounded_integer(self.write_top_k, "regiones que escriben", 1, REGIONS - 1)

    def identity(self):
        return dict(
            schema_version=1,
            kind=SHARED_WORKSPACE,
            operator=asdict(self.operator),
            regions=list(MODALITIES),
            slots=self.slots,
            write=self.write,
            write_top_k=self.write_top_k,
            selection="top_k_present_regions_per_slot_not_differentiated",
            read="each_region_attends_over_the_slots",
            initial_slots="learned_shared_by_every_row",
            state="working_only_reset_per_prediction",
            parameter_updates="none_in_forward",
        )


def parse_regional_variant(raw):
    """Aceptar una de las dos configuraciones completas sin mezclar sus campos."""
    if raw is None:
        return None
    if type(raw) in (RegionAxisMixingConfig, SharedWorkspaceConfig):
        raw.__post_init__()
        return raw
    if isinstance(raw, dict):
        for kind in (RegionAxisMixingConfig, SharedWorkspaceConfig):
            if set(raw) == {field.name for field in fields(kind)}:
                return kind(**raw)
    raise ValueError("regional_variant debe declarar una configuración completa de C1 o C6")


class _RegionalVariant(nn.Module):
    """Calendario común: pasos locales con máscara y un resumen por ciclo.

    Las subclases construyen sus parámetros y definen los mensajes locales y el resumen.
    La semilla deriva de la familia, así que una variante y su control empiezan con los
    mismos pesos comunes y ninguna construcción consume el generador global.
    """

    kind = None

    def __init__(self, hidden_size, config, *, seed, max_batch=256):
        super().__init__()
        bounded_integer(hidden_size, "dimensión regional", REGIONS, 512)
        bounded_integer(seed, "semilla regional", 0, 2**32 - 1)
        bounded_integer(max_batch, "lote regional", 1, MAX_BLOCK_ROWS)
        config.__post_init__()
        self.hidden_size, self.max_batch, self.config = hidden_size, max_batch, config
        derived = hashlib.sha256(f"{self.kind}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.local_norm = nn.LayerNorm(hidden_size)
            self._build(hidden_size)
            self.readout = nn.Linear(hidden_size, hidden_size, bias=False)
            self.register_buffer("edges", _topology_edges(config.operator.topology, device="cpu"))

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            max_batch=self.max_batch,
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        validate_assignment_request(state_dict, assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        validate_assignment(state_dict, prefix, local_metadata)
        edges = state_dict.get(prefix + "edges")
        expected = _topology_edges(self.config.operator.topology, device="cpu")
        if (
            not isinstance(edges, torch.Tensor)
            or edges.dtype != torch.bool
            or not torch.equal(edges.cpu(), expected)
        ):
            raise ValueError("El checkpoint no conserva las aristas de su configuración")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)

    def operation_counts(self, rows):
        """Evaluaciones de nodos y resúmenes, incluidas las filas enmascaradas."""
        bounded_integer(rows, "filas del trabajo regional", 0, MAX_BLOCK_ROWS)
        operator = self.config.operator
        return dict(
            regional_local_node_evaluations=rows * REGIONS * operator.cycles * operator.local_steps,
            regional_summary_evaluations=rows * operator.cycles if operator.summary else 0,
        )

    def _checked_presence(self, blocks, presence):
        reference = self.readout.weight
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.layout != torch.strided
            or blocks.ndim != 3
            or blocks.shape[1:] != (REGIONS, self.hidden_size)
            or not 1 <= len(blocks) <= self.max_batch
            or blocks.dtype not in (torch.float32, torch.float64)
            or blocks.dtype != reference.dtype
            or blocks.device != reference.device
        ):
            raise ValueError("Los bloques regionales no corresponden al contrato")
        if presence is None:
            return torch.ones(blocks.shape[:2], dtype=torch.bool, device=blocks.device)
        if (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.shape != blocks.shape[:2]
            or presence.device != blocks.device
            or presence.layout != torch.strided
        ):
            raise ValueError("Los bloques y máscaras regionales no corresponden al contrato")
        return presence

    def forward(self, blocks, presence):
        presence = self._checked_presence(blocks, presence)
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("La variante regional exige FP32 o FP64 explícitos, sin autocast")
        operator = self.config.operator
        with deferred_checks():
            check_finite(blocks, "Los bloques regionales")
            mask = presence.unsqueeze(-1)
            initial = torch.where(mask, blocks, torch.zeros_like(blocks))
            local = initial
            summary = self._initial_summary(blocks)
            context = blocks.new_zeros(len(blocks), 1, self.hidden_size)
            count = presence.sum(1, keepdim=True).clamp_min(1).to(blocks.dtype)
            active = presence.any(1, keepdim=True)
            allowed = self.edges.unsqueeze(0) & presence.unsqueeze(1)
            # Las consultas ausentes solo ven su propio cero y se anulan después.
            diagonal = torch.eye(REGIONS, dtype=torch.bool, device=blocks.device)
            allowed = allowed | (diagonal.unsqueeze(0) & ~presence.unsqueeze(-1))
            for _ in range(operator.cycles):
                for _ in range(operator.local_steps):
                    normalized = self.local_norm(local + context)
                    check_finite(normalized, "La normalización regional")
                    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
                    messages = self._messages(normalized, allowed)
                    updated = local + operator.local_gain * self.local_out(messages)
                    check_finite(updated, "La actualización regional local")
                    local = torch.where(mask, updated, torch.zeros_like(updated))
                if operator.summary:
                    summary, context = self._summary(summary, local, presence, count, active)
                    check_finite(context, "El contexto regional")
            projected = self.readout(local - initial + context)
            check_finite(projected, "La proyección regional")
            result = torch.where(mask, blocks + projected, blocks)
            check_finite(result, "La salida regional")
        return result


class RegionAxisMixing(_RegionalVariant):
    """C1: cinco cabezas sobre los nodos con mezcla de logits y de pesos entre ellas.

    Cada cabeza tiene sus propias proyecciones de dimensión `hidden_size // 5` sobre los cinco
    nodos. Con mezcla, la cabeza g usa L'_g = Σ_h P_gh L_h antes del softmax y
    A'_g = Σ_h Q_gh A_h después. Las posiciones prohibidas valen cero antes de mezclar y
    vuelven a prohibirse después, así que las aristas y la presencia se conservan. P y Q
    empiezan en la identidad, de modo que al construirla calcula lo mismo que su control.
    """

    kind = REGION_AXIS_MIXING

    def _build(self, hidden_size):
        if type(self.config) is not RegionAxisMixingConfig:
            raise ValueError("C1 necesita RegionAxisMixingConfig")
        self.head_dim = hidden_size // REGIONS
        inner = REGIONS * self.head_dim
        self.qkv = nn.Linear(hidden_size, 3 * inner, bias=False)
        self.local_out = nn.Linear(inner, hidden_size, bias=False)
        if self.config.operator.summary:
            self.summary_to_local = nn.Linear(hidden_size, hidden_size, bias=False)
            self.summary_norm = nn.LayerNorm(hidden_size)
            self.summary_update = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size, bias=False),
            )
        if self.config.mixing == "talking_heads":
            self.mix_logits = nn.Parameter(torch.eye(REGIONS))
            self.mix_weights = nn.Parameter(torch.eye(REGIONS))

    def get_extra_state(self):
        return dict(super().get_extra_state(), head_dim=self.head_dim)

    def _messages(self, normalized, allowed):
        rows = len(normalized)
        q, k, v = (
            self.qkv(normalized)
            .view(rows, REGIONS, 3, REGIONS, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        allowed = allowed.unsqueeze(1)
        mixing = self.config.mixing == "talking_heads"
        if mixing:
            zeroed = torch.where(allowed, logits, torch.zeros_like(logits))
            logits = torch.einsum("gh,bhij->bgij", self.mix_logits, zeroed)
        weights = torch.softmax(logits.masked_fill(~allowed, -math.inf), dim=-1)
        if mixing:
            weights = torch.einsum("gh,bhij->bgij", self.mix_weights, weights)
        messages = weights @ v
        return messages.transpose(1, 2).reshape(rows, REGIONS, REGIONS * self.head_dim)

    def _initial_summary(self, blocks):
        return blocks.new_zeros(len(blocks), self.hidden_size)

    def _summary(self, summary, local, presence, count, active):
        pooled = local.sum(1) / count
        update = self.summary_update(self.summary_norm(summary) + pooled)
        summary = torch.where(active, summary + self.config.operator.summary_gain * update, summary)
        return summary, self.summary_to_local(summary).unsqueeze(1)


class SharedWorkspace(_RegionalVariant):
    """C6: ranuras compartidas con escritura por competición y lectura por región.

    En cada resumen, cada ranura puntúa las regiones presentes. Con competición solo las
    `write_top_k` mejores de cada ranura escriben, con softmax entre ellas. La elección es
    discreta y no recibe gradiente, que sí reciben las puntuaciones elegidas. Después cada
    región atiende sobre las ranuras y recibe su propio contexto, en lugar del resumen común
    del operador original. Una fila sin regiones presentes conserva sus ranuras.
    """

    kind = SHARED_WORKSPACE

    def _build(self, hidden_size):
        if type(self.config) is not SharedWorkspaceConfig:
            raise ValueError("C6 necesita SharedWorkspaceConfig")
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.local_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.initial_slots = nn.Parameter(
            torch.randn(self.config.slots, hidden_size) / math.sqrt(hidden_size)
        )
        self.slot_norm = nn.LayerNorm(hidden_size)
        self.write_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.write_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slot_update = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=False),
        )
        self.read_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.read_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def _messages(self, normalized, allowed):
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        return F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=allowed.unsqueeze(1)
        ).squeeze(1)

    def _initial_summary(self, blocks):
        return self.initial_slots.unsqueeze(0).expand(len(blocks), -1, -1)

    def _write_weights(self, slots, local, presence, active):
        """Pesos [filas, ranuras, regiones] de la escritura, cero fuera de las elegidas."""
        scale = math.sqrt(self.hidden_size)
        scores = self.write_query(self.slot_norm(slots)) @ self.write_key(local).transpose(1, 2)
        scores = scores / scale
        eligible = presence.unsqueeze(1).expand_as(scores)
        if self.config.write == "competition":
            ranked = scores.detach().masked_fill(~eligible, -math.inf)
            chosen = ranked.topk(self.config.write_top_k, dim=-1).indices
            selected = torch.zeros_like(eligible).scatter(-1, chosen, True) & eligible
        else:
            selected = eligible
        # Una fila vacía puntúa todo para no dividir por cero y después se descarta.
        selected = selected | ~active.unsqueeze(-1)
        return torch.softmax(scores.masked_fill(~selected, -math.inf), dim=-1)

    def _summary(self, slots, local, presence, count, active):
        weights = self._write_weights(slots, local, presence, active)
        written = weights @ self.write_value(local)
        update = self.slot_update(self.slot_norm(slots) + written)
        slots = torch.where(
            active.unsqueeze(-1), slots + self.config.operator.summary_gain * update, slots
        )
        scale = math.sqrt(self.hidden_size)
        reading = torch.softmax(
            self.read_query(local) @ self.read_key(slots).transpose(1, 2) / scale, dim=-1
        )
        return slots, self.read_out(reading @ slots)


def build_regional_variant(hidden_size, config, *, seed, max_batch):
    """Construir el módulo de la configuración identificada."""
    if type(config) is RegionAxisMixingConfig:
        return RegionAxisMixing(hidden_size, config, seed=seed, max_batch=max_batch)
    if type(config) is SharedWorkspaceConfig:
        return SharedWorkspace(hidden_size, config, seed=seed, max_batch=max_batch)
    raise ValueError("La variante regional necesita una configuración identificada")
