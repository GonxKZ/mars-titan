"""Atención axial en el tiempo y entre regiones `axial_time_region_v1` (C2 de #751).

En `mac_online` cada flujo emite un token por instante y el tiempo solo vive en la memoria de
MAC. C2 guarda en el estado de cada flujo los bloques de fuente de sus W − 1 emisiones
anteriores, con su presencia, y atiende primero a lo largo del tiempo y después entre las
regiones de la emisión actual. Con z_r los cinco bloques de la emisión (precios, noticias,
gráficos, fundamentales y macro), S_r la secuencia de W bloques de la región r con la actual
al final y m su presencia,

    x_{w,r} = S_{w,r} + ρ_r + ℓ_{W−1−w}
    u_r = z_r + O_1 [Σ_w softmax_w(q_r · k_{w,r} / √d) v_{w,r}]_h,  w con m_{w,r} o actual
    y_r = LayerNorm_2(u_r + ρ_r)
    salida_r = u_r + O_2 [Σ_s softmax_s(q'_r · k'_s / √d) v'_s]_h,  s presente o s = r

con q_r = W_q LayerNorm_1(x_{W−1,r}), k y v de LayerNorm_1(x) en todas las posiciones, y
q', k', v' de y. Solo se calcula la salida de la posición actual, que es la fila W − 1 de la
atención con máscara causal de Ho et al. (2019, sección 3.1, página 4,
https://arxiv.org/abs/1912.12180v1): el resultado en una posición solo depende de las
anteriores y de ella misma. Dos capas axiales cubren todo el tensor tiempo × región, porque
la segunda reparte entre regiones lo que la primera leyó del tiempo de cada una. MASTER
(Li et al., 2024, páginas 164 y 165, https://ojs.aaai.org/index.php/AAAI/article/view/27767)
alterna de la misma forma la agregación dentro de cada acción a lo largo de su ventana y la
agregación entre acciones en cada instante, con una ventana de τ = 8 sesiones.

El control `time_only_attention_v1` tiene exactamente los mismos parámetros y la misma
inicialización. Su segunda capa vuelve a leer el tiempo de cada región, con la consulta desde
u_r y la posición actual sustituida por u_r, en lugar de mezclar regiones. Así separa la
mezcla entre regiones del efecto de leer el pasado con dos capas.

O_1 y O_2 empiezan en cero, así que al construirse la salida es z y el predictor emite lo
mismo que su referencia. Una región ausente en la emisión conserva su bloque y no envía nada.
Las posiciones ausentes del pasado no entran en ninguna atención. La posición actual siempre
admite su propia consulta, lo que evita un softmax vacío, y su salida se descarta después si
la región falta. El historial guarda los bloques de entrada, no las salidas, y avanza una
posición por emisión del flujo: el desfase ℓ cuenta emisiones y no días de calendario. Su
grafo se conserva hasta que el recorrido corta el del estado de MAC.
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from ..baselines.multimodal import MODALITIES
from .config import bounded_integer, require_identity
from .state import check_finite

AXIAL_TIME_REGION = "axial_time_region_v1"
TIME_ONLY = "time_only_attention_v1"
KINDS = (AXIAL_TIME_REGION, TIME_ONLY)
HEADS = (1, 2, 4, 8)
REGIONS = len(MODALITIES)
# Desviación de los embeddings de región y de desfase al construirse.
EMBEDDING_STD = 0.02


@dataclass(frozen=True)
class AxialTimeRegionConfig:
    kind: str = AXIAL_TIME_REGION
    window: int = 8
    heads: int = 4

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError(f"C2 admite los tipos {KINDS}")
        bounded_integer(self.window, "ventana de C2", 2, 32)
        if type(self.heads) is not int or self.heads not in HEADS:
            raise ValueError(f"C2 admite {HEADS} cabezas")

    def identity(self):
        return dict(
            schema_version=1,
            kind=self.kind,
            regions=list(MODALITIES),
            window=self.window,
            heads=self.heads,
            history="source_blocks_of_the_previous_window_minus_one_emissions_of_the_flow",
            lag="counted_in_emissions_of_the_flow_not_in_calendar_days",
            time_axis="causal_attention_of_the_current_emission_over_present_positions",
            second_layer="attention_across_present_regions_of_the_current_emission"
            if self.kind == AXIAL_TIME_REGION
            else "second_causal_time_read_with_the_updated_current_position",
            embeddings="learned_region_and_lag_shared_by_both_kinds",
            output="residuals_with_zero_initialized_projections",
            absent_regions="keep_their_block_and_send_nothing",
            history_graph="kept_until_the_flow_state_is_detached",
            parameter_updates="none_in_forward",
        )


def parse_axial_time_region(raw):
    """Aceptar la configuración, sus valores completos o None."""
    if raw is None:
        return None
    if type(raw) is AxialTimeRegionConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"kind", "window", "heads"}:
        return AxialTimeRegionConfig(**raw)
    raise ValueError("axial_time_region debe declarar kind, window y heads, o ser None")


@dataclass(frozen=True)
class RegionHistory:
    """Bloques [flujos, W − 1, 5, H] de las emisiones anteriores y su presencia booleana."""

    blocks: torch.Tensor
    present: torch.Tensor


def history_tensors(history):
    return (history.blocks, history.present)


def map_history_rows(history, function):
    """Aplicar la misma operación por filas a los dos tensores del historial."""
    return RegionHistory(function(history.blocks), function(history.present))


def join_history_rows(parts):
    """Concatenar por filas historiales de varios bloques, en el orden recibido."""
    return RegionHistory(
        torch.cat([part.blocks for part in parts], dim=0),
        torch.cat([part.present for part in parts], dim=0),
    )


class AxialTimeRegion(nn.Module):
    """Dos capas residuales sobre el tensor tiempo × región de cada flujo."""

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de C2", 1, 4096)
        bounded_integer(seed, "semilla de C2", 0, 2**32 - 1)
        if type(config) is not AxialTimeRegionConfig:
            raise ValueError("C2 necesita AxialTimeRegionConfig")
        config.__post_init__()
        if hidden_size % config.heads:
            raise ValueError("La dimensión de C2 debe ser múltiplo del número de cabezas")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("C2 necesita precisión FP32 o FP64")
        self.config, self.hidden_size = config, hidden_size
        self.head_dim = hidden_size // config.heads
        # La semilla deriva de la del predictor y no depende del tipo, así que la variante
        # y su control empiezan con los mismos pesos y ninguno consume el generador global.
        derived = hashlib.sha256(f"{AXIAL_TIME_REGION}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.region_embedding = nn.Parameter(
                torch.randn(REGIONS, hidden_size, dtype=dtype) * EMBEDDING_STD
            )
            self.lag_embedding = nn.Parameter(
                torch.randn(config.window, hidden_size, dtype=dtype) * EMBEDDING_STD
            )
            self.time_norm = nn.LayerNorm(hidden_size, dtype=dtype)
            self.time_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False, dtype=dtype)
            self.time_out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
            self.second_norm = nn.LayerNorm(hidden_size, dtype=dtype)
            self.second_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False, dtype=dtype)
            self.second_out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.time_out.weight)
        nn.init.zeros_(self.second_out.weight)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    @property
    def past(self):
        return self.config.window - 1

    def bytes_per_flow(self):
        """Bytes del historial de un flujo: bloques en la precisión del módulo y presencia."""
        positions = self.past * REGIONS
        return positions * self.hidden_size * self.time_out.weight.element_size() + positions

    def initial_history(self, rows, *, device=None):
        """Historial vacío: ninguna posición anterior presente."""
        reference = self.time_out.weight
        device = reference.device if device is None else torch.device(device)
        return RegionHistory(
            torch.zeros(
                rows, self.past, REGIONS, self.hidden_size, dtype=reference.dtype, device=device
            ),
            torch.zeros(rows, self.past, REGIONS, dtype=torch.bool, device=device),
        )

    def validate_history(self, history, rows, *, device=None):
        """Comprobar forma, tipo y dispositivo del historial sin esperar a la GPU."""
        reference = self.time_out.weight
        device = reference.device if device is None else torch.device(device)
        shape = (rows, self.past, REGIONS)
        if (
            type(history) is not RegionHistory
            or any(
                not isinstance(value, torch.Tensor)
                or value.layout != torch.strided
                or value.device != device
                for value in history_tensors(history)
            )
            or history.blocks.shape != (*shape, self.hidden_size)
            or history.blocks.dtype != reference.dtype
            or history.present.shape != shape
            or history.present.dtype != torch.bool
        ):
            raise ValueError("El historial de C2 no corresponde a sus flujos ni al módulo")

    def validate_counts(self, history, observed_steps):
        """Un flujo con s emisiones no puede tener presentes posiciones más antiguas que s.

        Se comprueba al decodificar un payload en CPU, donde no obliga a esperar a la GPU.
        """
        present = history.present.detach().cpu()
        steps = observed_steps.detach().cpu().clamp(max=self.past)
        # La posición p es la emisión p − (W − 1) respecto a la actual.
        oldest = self.past - steps
        positions = torch.arange(self.past).unsqueeze(0)
        if (present.any(-1) & (positions < oldest.unsqueeze(1))).any():
            raise ValueError("El historial de C2 tiene posiciones anteriores a la primera emisión")

    def operation_counts(self, rows):
        """Consultas y pares de atención de las dos capas, incluidas las filas ausentes."""
        bounded_integer(rows, "filas de C2", 0, 16384)
        window = self.config.window
        time_pairs = rows * REGIONS * window
        second = rows * REGIONS * REGIONS if self.config.kind == AXIAL_TIME_REGION else time_pairs
        return dict(axial_time_pairs=time_pairs, axial_second_pairs=second)

    def _read(self, query, keys, values, allowed):
        """Atención de una consulta por región sobre W posiciones, con varias cabezas."""
        rows, heads, width = query.shape[0], self.config.heads, self.head_dim
        q = query.view(rows, REGIONS, heads, width)
        k = keys.view(rows, self.config.window, REGIONS, heads, width)
        v = values.view(rows, self.config.window, REGIONS, heads, width)
        logits = torch.einsum("frhd,fwrhd->frhw", q, k) / math.sqrt(width)
        weights = torch.softmax(logits.masked_fill(~allowed, -math.inf), dim=-1)
        return torch.einsum("frhw,fwrhd->frhd", weights, v).reshape(rows, REGIONS, -1)

    def forward(self, blocks, presence, history):
        """Bloques modulados [flujos, 5, H] y el historial con la emisión actual añadida."""
        reference = self.time_out.weight
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.layout != torch.strided
            or blocks.ndim != 3
            or blocks.shape[1:] != (REGIONS, self.hidden_size)
            or len(blocks) == 0
            or blocks.dtype != reference.dtype
            or blocks.device != reference.device
        ):
            raise ValueError("Los bloques de C2 no corresponden al contrato")
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("C2 exige FP32 o FP64 explícitos, sin autocast")
        rows, size = len(blocks), self.hidden_size
        self.validate_history(history, rows)
        if presence is None:
            present = torch.ones(rows, REGIONS, dtype=torch.bool, device=blocks.device)
        elif (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.shape != (rows, REGIONS)
            or presence.device != blocks.device
        ):
            raise ValueError("La presencia de C2 debe ser booleana [flujos, 5]")
        else:
            present = presence
        sequence = torch.cat((history.blocks, blocks.unsqueeze(1)), dim=1)
        mask = torch.cat((history.present, present.unsqueeze(1)), dim=1)
        # La posición w tiene desfase W − 1 − w, así que se recorren los desfases al revés.
        tokens = sequence + self.region_embedding + self.lag_embedding.flip(0).unsqueeze(1)
        current = torch.zeros(self.config.window, dtype=torch.bool, device=blocks.device)
        current[-1] = True
        # [flujos, 5, 1, W]: posiciones presentes de cada región y siempre la actual.
        allowed = (mask.transpose(1, 2) | current).unsqueeze(2)
        query_weight, key_weight, value_weight = self.time_qkv.weight.split(size)
        normalized = self.time_norm(tokens)
        read = self._read(
            normalized[:, -1] @ query_weight.T,
            normalized @ key_weight.T,
            normalized @ value_weight.T,
            allowed,
        )
        updated = blocks + self.time_out(read)
        query_weight, key_weight, value_weight = self.second_qkv.weight.split(size)
        if self.config.kind == AXIAL_TIME_REGION:
            heads, width = self.config.heads, self.head_dim
            mixed = self.second_norm(updated + self.region_embedding)
            q, k, v = (
                (mixed @ weight.T).view(rows, REGIONS, heads, width)
                for weight in (query_weight, key_weight, value_weight)
            )
            logits = torch.einsum("frhd,fshd->fhrs", q, k) / math.sqrt(width)
            itself = torch.eye(REGIONS, dtype=torch.bool, device=blocks.device)
            # Cada región atiende a las presentes y a sí misma, en la misma emisión.
            allowed_regions = (present[:, None, None, :] | itself).expand(-1, heads, -1, -1)
            weights = torch.softmax(logits.masked_fill(~allowed_regions, -math.inf), dim=-1)
            second = torch.einsum("fhrs,fshd->frhd", weights, v).reshape(rows, REGIONS, size)
        else:
            refreshed = updated + self.region_embedding + self.lag_embedding[0]
            again = self.second_norm(torch.cat((tokens[:, :-1], refreshed.unsqueeze(1)), dim=1))
            second = self._read(
                again[:, -1] @ query_weight.T,
                again @ key_weight.T,
                again @ value_weight.T,
                allowed,
            )
        result = torch.where(present.unsqueeze(-1), updated + self.second_out(second), blocks)
        check_finite(result, "Los bloques con atención axial")
        following = RegionHistory(sequence[:, 1:].contiguous(), mask[:, 1:].contiguous())
        return result, following
