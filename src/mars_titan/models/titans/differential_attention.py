"""Atención diferencial `differential_attention_v1` sobre la secuencia de MAC, U1 de #786.

El Transformer diferencial (Ye y otros, ICLR 2025, sección 2.1) resta dos mapas softmax para
cancelar el ruido que comparten: (softmax(Q₁K₁ᵀ/√d) − λ softmax(Q₂K₂ᵀ/√d))V. Aquí el primer
mapa es el de la atención multicabeza de MAC, sin cambios, y el segundo usa dos proyecciones
nuevas de consultas y claves con los valores de la propia atención:

    o = MHA(P)_S − W_O [λ_h softmax(Q₂,h K₂,hᵀ/√d_h + máscara) V_h]_h,

con P la secuencia [memoria persistente, lectura de memoria, segmento], S las posiciones del
segmento, V_h los valores de la cabeza h de la MHA y W_O su proyección de salida. λ_h es un
escalar por cabeza que empieza en cero, así que al empezar o coincide con MAC y λ recibe
gradiente desde el primer paso. Q₂ y K₂ se sortean con la cota de xavier del `in_proj` de la
MHA, con una semilla derivada de la de MAC y sin consumir el generador global.

Desviaciones respecto al artículo: λ es un escalar directo y no la reparametrización con
λ_init, no hay normalización por cabeza ni el factor (1 − λ_init), y V tiene la dimensión de
la cabeza y no el doble. Las tres cambiarían la salida al empezar. Solo se calculan las
consultas del segmento, que son las únicas posiciones que MAC usa.
"""

import hashlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

DIFFERENTIAL_ATTENTION = "differential_attention_v1"


def identity() -> dict:
    return dict(
        name=DIFFERENTIAL_ATTENTION,
        first_map="mac_multihead_attention_unchanged",
        second_map="new_query_and_key_projections_with_the_mha_values",
        weight="lambda_per_head_initialized_at_zero",
        head_normalization="none",
        output="subtracted_through_the_mha_output_projection",
        queries="segment_positions_only",
        initialization="xavier_bound_of_the_in_proj_from_a_derived_seed",
        source="arxiv_2410_05258_section_2_1",
    )


class DifferentialMap(nn.Module):
    """Segundo mapa de atención ponderado por λ_h y proyectado con W_O."""

    def __init__(self, dim: int, heads: int, *, seed: int, dtype: torch.dtype) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("heads debe dividir dim")
        derived = hashlib.sha256(f"{DIFFERENTIAL_ATTENTION}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            # La misma cota que xavier_uniform da al in_proj [3·dim, dim] de la MHA.
            bound = math.sqrt(6 / (4 * dim))
            drawn = torch.empty(2 * dim, dim, dtype=dtype).uniform_(-bound, bound)
        self.query = nn.Parameter(drawn[:dim].clone())
        self.key = nn.Parameter(drawn[dim:].clone())
        self.weight = nn.Parameter(torch.zeros(heads, dtype=dtype))

    def forward(
        self,
        packed: Tensor,
        values_weight: Tensor,
        output_projection: nn.Module,
        mask: Tensor,
        length: int,
    ) -> Tensor:
        """Devuelve W_O[λ_h A₂,h V_h]_h para las últimas `length` posiciones de `packed`.

        `mask` es la máscara booleana de la MHA, con True en las posiciones que no se atienden.
        `output_projection` es la W_O de la MHA de MAC, que no tiene bias, así que con λ nulo
        el resultado es exactamente cero.
        """
        batch, _, dim = packed.shape
        heads = self.weight.shape[0]
        width = dim // heads

        def split(projected: Tensor) -> Tensor:
            return projected.view(batch, -1, heads, width).transpose(1, 2)

        queries = split(F.linear(packed[:, -length:], self.query))
        keys = split(F.linear(packed, self.key))
        values = split(F.linear(packed, values_weight))
        mixed = F.scaled_dot_product_attention(queries, keys, values, attn_mask=~mask[-length:])
        mixed = mixed * self.weight.view(1, heads, 1, 1)
        return output_projection(mixed.transpose(1, 2).reshape(batch, length, dim))
