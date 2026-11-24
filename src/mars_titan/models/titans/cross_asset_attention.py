"""Atención entre los activos de una misma sesión `cross_asset_attention_v1` (C5 de #751).

Tras MAC, cada flujo i tiene la representación e_i de su emisión. C5 la combina con las de
los demás activos del mismo mercado y del mismo instante:

    n_i = LayerNorm(e_i)
    a^h_ij = softmax_{j ∈ G(i)}((W^h_q n_i) · (W^h_k n_j) / √d_h)
    e'_i = e_i + W_o [Σ_{j ∈ G(i)} a^h_ij W^h_v n_j]_h

G(i) son las filas del bloque con el mercado y el instante de i. Todas sus entradas están
disponibles en la emisión, así que la mezcla no lee nada posterior. W_o empieza en cero y el
predictor emite al construirse lo mismo que sin C5. Parte de la agregación entre acciones de
MASTER (Li et al., 2024, https://arxiv.org/abs/2312.15235). MASTER la aplica en cada instante
de su ventana, entre la codificación de cada acción y el resumen temporal. Aquí el estado
temporal de cada flujo es el de MAC y solo existe la representación de la emisión, así que C5
mezcla una vez, en el instante de decisión. Tampoco incorpora la puerta guiada por el estado
del mercado de MASTER.

El control `self_only` tiene los mismos parámetros y la misma inicialización, pero G(i) = {i}.
Cada activo solo se atiende a sí mismo y la mezcla se reduce a W_o W_v n_i. Así separa la
información de los demás activos del efecto de añadir una capa residual.

C5 necesita todas las filas de la sesión en el mismo bloque (`session_level`).
"""

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import bounded_integer, require_identity
from .state import check_finite

CROSS_ASSET_ATTENTION = "cross_asset_attention_v1"
MODES = ("same_session", "self_only")
HEADS = (1, 2, 4, 8)


@dataclass(frozen=True)
class CrossAssetAttentionConfig:
    heads: int = 4
    mode: str = "same_session"

    def __post_init__(self):
        if type(self.heads) is not int or self.heads not in HEADS:
            raise ValueError(f"C5 admite {HEADS} cabezas")
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("El modo de C5 debe ser same_session o self_only")

    def identity(self):
        return dict(
            schema_version=1,
            kind=CROSS_ASSET_ATTENTION,
            heads=self.heads,
            mode=self.mode,
            groups="same_market_and_prediction_instant_in_the_block"
            if self.mode == "same_session"
            else "each_row_with_itself",
            position="after_mac_before_head",
            normalization="layer_norm_before_projections",
            output="residual_with_zero_initialized_projection",
            state="none",
            parameter_updates="none_in_forward",
        )


def parse_cross_asset_attention(raw):
    """Aceptar la configuración, sus valores completos o None."""
    if raw is None:
        return None
    if type(raw) is CrossAssetAttentionConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"heads", "mode"}:
        return CrossAssetAttentionConfig(**raw)
    raise ValueError("cross_asset_attention debe declarar heads y mode, o ser None")


class CrossAssetAttention(nn.Module):
    """Mezcla residual de las representaciones de los activos de cada sesión."""

    session_level = True

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de C5", 1, 4096)
        bounded_integer(seed, "semilla de C5", 0, 2**32 - 1)
        if type(config) is not CrossAssetAttentionConfig:
            raise ValueError("C5 necesita CrossAssetAttentionConfig")
        config.__post_init__()
        if hidden_size % config.heads:
            raise ValueError("La dimensión de C5 debe ser múltiplo del número de cabezas")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("C5 necesita precisión FP32 o FP64")
        self.config, self.hidden_size = config, hidden_size
        self.head_dim = hidden_size // config.heads
        # La semilla deriva de la del predictor y del nombre, sin consumir el generador
        # global. La variante y su control empiezan con los mismos pesos.
        derived = hashlib.sha256(f"{CROSS_ASSET_ATTENTION}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False, dtype=dtype)
            self.out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def groups(self, markets, prediction_at):
        """Filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.values())

    def forward(self, encoded, markets, prediction_at):
        reference = self.out.weight
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError("C5 necesita una representación, un mercado y un instante por fila")
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("C5 exige FP32 o FP64 explícitos, sin autocast")
        rows = len(encoded)
        heads, width = self.config.heads, self.head_dim
        projected = self.qkv(self.norm(encoded))
        if self.config.mode == "self_only":
            # Con G(i) = {i} el peso softmax es exactamente 1 y la mezcla es V_i, con gradiente
            # nulo en Q y K. Evita construir matrices N×N y es idéntico bit a bit.
            mixed = projected[:, 2 * self.hidden_size :]
            result = encoded + self.out(mixed)
            check_finite(result, "La representación con atención entre activos")
            return result
        q, k, v = projected.view(rows, 3, heads, width).permute(1, 2, 0, 3)
        use_flash = (
            encoded.is_cuda
            and width >= 16
            and (width & (width - 1)) == 0
            and width <= 128
        )
        if use_flash:
            from .fused_cuda_kernels import fused_flash_cross_asset

        groups = self.groups(markets, prediction_at)
        if len(groups) == 1:
            if use_flash:
                part = fused_flash_cross_asset(q, k, v)
            else:
                logits = q @ k.transpose(-1, -2) / math.sqrt(width)
                part = torch.softmax(logits, dim=-1) @ v
            mixed = part.transpose(0, 1).reshape(rows, self.hidden_size)
        else:
            mixed = torch.zeros_like(encoded)
            for members in groups:
                size = len(members)
                contiguous = members[-1] - members[0] + 1 == size and members == list(
                    range(members[0], members[0] + size)
                )
                if contiguous:
                    start, stop = members[0], members[0] + size
                    query, key, value = q[:, start:stop], k[:, start:stop], v[:, start:stop]
                else:
                    if encoded.is_cuda:
                        pinned = torch.tensor(members, dtype=torch.int64, pin_memory=True)
                        index = pinned.to(device=encoded.device, non_blocking=True)
                    else:
                        index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
                    query, key, value = (part.index_select(1, index) for part in (q, k, v))
                if use_flash:
                    part = fused_flash_cross_asset(query, key, value)
                else:
                    logits = query @ key.transpose(-1, -2) / math.sqrt(width)
                    part = torch.softmax(logits, dim=-1) @ value
                part = part.transpose(0, 1).reshape(size, self.hidden_size)
                if contiguous:
                    mixed[start:stop] = part
                else:
                    mixed = mixed.index_copy(0, index, part)
        result = encoded + self.out(mixed)
        check_finite(result, "La representación con atención entre activos")
        return result
