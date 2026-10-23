"""Comunicación regional con dos ritmos de memoria de trabajo por predicción.

Los cinco nodos reciben las representaciones ya proyectadas de las fuentes.
El resumen se actualiza después de cada grupo de pasos locales. No hay aquí
memoria histórica, etiquetas financieras ni actualización de parámetros.
"""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..baselines.multimodal import MODALITIES
from .config import MAX_BLOCK_ROWS, bounded_integer, require_identity
from .state import check_finite, deferred_checks, finite_indicator, is_finite

REGIONAL_FUSION = "regional_fusion_v1"


def _topology_edges(topology, *, device):
    indices = torch.arange(len(MODALITIES), device=device)
    difference = (indices[:, None] - indices[None, :]) % len(MODALITIES)
    if topology == "complete":
        return torch.ones_like(difference, dtype=torch.bool)
    if topology == "self":
        return difference == 0
    return (difference == 0) | (difference == 1) | (difference == len(MODALITIES) - 1)


def validate_assignment(state_dict, prefix, local_metadata):
    """Rechazar asignaciones con dispositivos mezclados antes de sustituir tensores.

    La copia ordinaria admite orígenes distintos. La asignación exige un dispositivo
    común y no admite activación parcial mediante metadatos de un descendiente.
    """
    assign = local_metadata.get("assign_to_params_buffers", False)
    if type(assign) is not bool:
        raise ValueError("La asignación del checkpoint necesita un booleano explícito")
    metadata = getattr(state_dict, "_metadata", {})
    if not assign and any(
        item.get("assign_to_params_buffers", False)
        for name, item in metadata.items()
        if name.startswith(prefix) and name != prefix.removesuffix(".")
    ):
        raise ValueError("El checkpoint no admite asignaciones parciales por módulo")
    if assign:
        devices = {
            value.device
            for name, value in state_dict.items()
            if name.startswith(prefix) and isinstance(value, torch.Tensor)
        }
        if len(devices) > 1:
            raise ValueError("La asignación del checkpoint necesita un dispositivo común")


def validate_assignment_request(state_dict, assign):
    """Exigir que los metadatos no contradigan la asignación pedida explícitamente."""
    if type(assign) is not bool:
        raise ValueError("La asignación del checkpoint necesita un booleano explícito")
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is None:
        return
    if not isinstance(metadata, Mapping):
        raise ValueError("Los metadatos del checkpoint deben ser un mapa")
    for item in metadata.values():
        if not isinstance(item, Mapping):
            raise ValueError("Los metadatos de cada módulo deben ser un mapa")
        if "assign_to_params_buffers" in item:
            flag = item["assign_to_params_buffers"]
            if type(flag) is not bool or flag != assign:
                raise ValueError("Los metadatos contradicen la asignación solicitada")


@dataclass(frozen=True)
class RegionalFusionConfig:
    topology: str = "complete"
    routing: str = "attention"
    cycles: int = 2
    local_steps: int = 2
    summary: bool = True
    local_gain: float = 0.25
    summary_gain: float = 0.25

    def __post_init__(self):
        if type(self.topology) is not str or self.topology not in ("complete", "ring", "self"):
            raise ValueError("La topología regional debe ser complete, ring o self")
        if type(self.routing) is not str or self.routing not in ("attention", "uniform"):
            raise ValueError("La ruta regional debe ser attention o uniform")
        bounded_integer(self.cycles, "ciclos regionales", 1, 4)
        bounded_integer(self.local_steps, "pasos locales", 1, 4)
        if type(self.summary) is not bool:
            raise ValueError("El resumen regional necesita un booleano explícito")
        if self.cycles * (len(MODALITIES) * self.local_steps + int(self.summary)) > 64:
            raise ValueError("El grafo supera 64 actualizaciones de nodo por predicción")
        for name in ("local_gain", "summary_gain"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Las ganancias regionales deben ser finitas y estar entre 0 y 1")

    def identity(self):
        return dict(
            schema_version=1,
            kind=REGIONAL_FUSION,
            regions=list(MODALITIES),
            **asdict(self),
            state="working_only_reset_per_prediction",
            order="local_steps_then_summary_per_cycle",
            global_paths="summary_hub_connects_present_nodes_when_enabled",
            absent_nodes="no_messages_original_block_unchanged",
            parameter_updates="none_in_forward",
        )


class RegionalFusion(nn.Module):
    """Transformar [B,5,H] sin compartir estados de trabajo entre filas o llamadas.

    La atención usa una matriz densa pequeña con máscara de aristas. El resumen
    crea conexiones indirectas incluso si las aristas locales son diagonales.
    La ruta uniforme conserva los parámetros y evalúa QKV, pero anula Q y K.
    Sus gradientes solo usan V. No se declara igualdad de capacidad efectiva.
    """

    def __init__(self, hidden_size, config, *, seed, max_batch=256):
        super().__init__()
        bounded_integer(hidden_size, "dimensión regional", 1, 512)
        bounded_integer(seed, "semilla regional", 0, 2**32 - 1)
        bounded_integer(max_batch, "lote regional", 1, MAX_BLOCK_ROWS)
        if type(config) is not RegionalFusionConfig:
            raise ValueError("La fusión regional necesita su configuración identificada")
        config.__post_init__()
        self.hidden_size, self.max_batch, self.config = hidden_size, max_batch, config
        derived = hashlib.sha256(f"{REGIONAL_FUSION}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.local_norm = nn.LayerNorm(hidden_size)
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            self.local_out = nn.Linear(hidden_size, hidden_size, bias=False)
            self.summary_to_local = nn.Linear(hidden_size, hidden_size, bias=False)
            self.summary_norm = nn.LayerNorm(hidden_size)
            self.summary_update = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size, bias=False),
            )
            self.readout = nn.Linear(hidden_size, hidden_size, bias=False)
            self.register_buffer("edges", _topology_edges(config.topology, device="cpu"))

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            max_batch=self.max_batch,
        )

    def operation_counts(self, rows):
        """Evaluaciones de nodos ejecutadas, incluidas las filas enmascaradas.

        La planificación puede reunir varios bloques físicos del mismo evento.
        Los contadores no son FLOPs, bytes ni actualizaciones de parámetros.
        """
        bounded_integer(rows, "filas del trabajo regional", 0, MAX_BLOCK_ROWS)
        return dict(
            regional_local_node_evaluations=rows
            * len(MODALITIES)
            * self.config.cycles
            * self.config.local_steps,
            regional_summary_evaluations=rows * self.config.cycles if self.config.summary else 0,
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        validate_assignment_request(state_dict, assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def validate_structure(self):
        """Validar el buffer físico sin impedir que una carga repare sus valores."""
        if type(self.config) is not RegionalFusionConfig:
            raise ValueError("La fusión regional necesita su configuración identificada")
        self.config.__post_init__()
        bounded_integer(self.hidden_size, "dimensión regional", 1, 512)
        bounded_integer(self.max_batch, "lote regional", 1, MAX_BLOCK_ROWS)
        dim = self.hidden_size
        shapes = {
            "local_norm.weight": (dim,),
            "local_norm.bias": (dim,),
            "qkv.weight": (3 * dim, dim),
            "local_out.weight": (dim, dim),
            "summary_to_local.weight": (dim, dim),
            "summary_norm.weight": (dim,),
            "summary_norm.bias": (dim,),
            "summary_update.0.weight": (2 * dim, dim),
            "summary_update.2.weight": (dim, 2 * dim),
            "readout.weight": (dim, dim),
        }
        parameters = dict(self.named_parameters())
        if set(parameters) != set(shapes):
            raise ValueError("Los parámetros regionales no conservan su estructura declarada")
        reference = parameters["qkv.weight"]
        if any(
            value.shape != shapes[name]
            or value.layout != torch.strided
            or value.device != reference.device
            or value.dtype != reference.dtype
            for name, value in parameters.items()
        ):
            raise ValueError("Los parámetros regionales no conservan sus formas y precisión")
        edges = self._buffers.get("edges")
        if (
            not isinstance(edges, torch.Tensor)
            or edges.dtype != torch.bool
            or edges.layout != torch.strided
            or edges.shape != (len(MODALITIES), len(MODALITIES))
            or edges.device != self.qkv.weight.device
            or "edges" in self._non_persistent_buffers_set
            or set(dict(self.named_buffers())) != {"edges"}
        ):
            raise ValueError("El buffer regional no conserva su estructura declarada")

    def validate_geometry(self):
        """Comprobar las aristas instaladas antes de confirmar una identidad."""
        self.validate_structure()
        expected = _topology_edges(self.config.topology, device=self.edges.device)
        if not torch.equal(self.edges, expected):
            raise ValueError("Las aristas regionales no corresponden a su configuración")

    def validate_payload(self, state_dict, prefix=""):
        self.validate_structure()
        supplied = {
            key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
        }
        expected = self.state_dict()
        if set(supplied) != set(expected):
            raise ValueError("El checkpoint no conserva todo el estado regional")
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
                raise ValueError("El checkpoint regional no conserva formas y precisión")
            # Una carga no puede aplazar el rechazo hasta después de copiar parámetros.
            if not is_finite(finite_indicator(other)):
                raise ValueError("El checkpoint regional contiene NaN o infinito")
        edges = supplied["edges"]
        if not torch.equal(edges, _topology_edges(self.config.topology, device=edges.device)):
            raise ValueError("El checkpoint no conserva las aristas de su configuración")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, *args, **kwargs):
        validate_assignment(state_dict, prefix, local_metadata)
        self.validate_payload(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, *args, **kwargs)

    def _checked_presence(self, blocks, presence):
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.layout != torch.strided
            or blocks.ndim != 3
            or blocks.shape[1:] != (len(MODALITIES), self.hidden_size)
            or not 1 <= len(blocks) <= self.max_batch
            or blocks.dtype not in (torch.float32, torch.float64)
            or blocks.dtype != self.qkv.weight.dtype
            or blocks.device != self.qkv.weight.device
        ):
            raise ValueError("Los bloques regionales no corresponden al contrato")
        if presence is None:
            # La fusión estricta admite solo entradas completas y no pasa bits de ausencia.
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
        return self._execute(
            blocks,
            presence,
            cycles=self.config.cycles,
            local_steps=self.config.local_steps,
        )

    def _execute(self, blocks, presence, *, cycles, local_steps, context_seed=None, observe=None):
        presence = self._checked_presence(blocks, presence)
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("La fusión regional exige FP32 o FP64 explícitos, sin autocast")
        with deferred_checks():
            check_finite(blocks, "Los bloques regionales")
            mask = presence.unsqueeze(-1)
            initial = torch.where(mask, blocks, torch.zeros_like(blocks))
            local = initial
            summary = blocks.new_zeros(len(blocks), self.hidden_size)
            context = summary.unsqueeze(1) if context_seed is None else context_seed.unsqueeze(1)
            count = presence.sum(1, keepdim=True).clamp_min(1).to(blocks.dtype)
            active = presence.any(1, keepdim=True)
            allowed = self.edges.unsqueeze(0) & presence.unsqueeze(1)
            # Las consultas ausentes solo ven su propio cero y se anulan después.
            diagonal = torch.eye(len(MODALITIES), dtype=torch.bool, device=blocks.device)
            allowed = allowed | (diagonal.unsqueeze(0) & ~presence.unsqueeze(-1))
            for _ in range(cycles):
                for _ in range(local_steps):
                    if observe is not None:
                        observe("local")
                    normalized = self.local_norm(local + context)
                    check_finite(normalized, "La normalización regional")
                    normalized = torch.where(mask, normalized, torch.zeros_like(normalized))
                    q, k, v = self.qkv(normalized).chunk(3, dim=-1)
                    if self.config.routing == "uniform":
                        q, k = torch.zeros_like(q), torch.zeros_like(k)
                    messages = F.scaled_dot_product_attention(
                        q.unsqueeze(1),
                        k.unsqueeze(1),
                        v.unsqueeze(1),
                        attn_mask=allowed.unsqueeze(1),
                    ).squeeze(1)
                    updated = local + self.config.local_gain * self.local_out(messages)
                    check_finite(updated, "La actualización regional local")
                    local = torch.where(mask, updated, torch.zeros_like(updated))
                    check_finite(local, "El estado regional local")
                if self.config.summary:
                    if observe is not None:
                        observe("summary")
                    pooled = local.sum(1) / count
                    update = self.summary_update(self.summary_norm(summary) + pooled)
                    check_finite(update, "La actualización del resumen regional")
                    summary = torch.where(
                        active, summary + self.config.summary_gain * update, summary
                    )
                    context = self.summary_to_local(summary).unsqueeze(1)
                    if context_seed is not None:
                        context = context_seed.unsqueeze(1) + context
                    check_finite(summary, "El resumen regional")
            projected = self.readout(local - initial + context)
            check_finite(projected, "La proyección regional")
            result = torch.where(mask, blocks + projected, blocks)
            check_finite(result, "La salida regional")
        return result
