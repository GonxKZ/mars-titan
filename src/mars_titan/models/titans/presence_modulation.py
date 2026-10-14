"""Fusión condicionada por la presencia de modalidades (PT6) y su alternativa trivial.

La fusión común anula la proyección de cada modalidad ausente y concatena los cinco bits
de presencia. Eso desplaza la salida según el patrón de ausencias, pero no permite que la
ausencia de una modalidad reescale la contribución de otra. PT6 añade una modulación FiLM
(Perez et al., 2018) cuya escala y desplazamiento dependen solo del vector de ausencias
a = 1 − p:

    z'_m = p_m · ((1 + Γ_m a) ⊙ z_m + Β_m a)

Con todas las modalidades presentes, a = 0 y la fila conserva sus bloques sin pasar por la
modulación, así que la identidad es exacta. Una modalidad ausente sigue anulada. Γ y Β
empiezan en cero, de modo que el predictor emite al principio lo mismo que su control.

La alternativa trivial añade el mismo orden de parámetros sin interacción multiplicativa
con la presencia: k unidades ocultas más en la fusión, en una rama residual
D · SiLU(E x) cuya salida empieza en cero porque D empieza en cero.
"""

import hashlib
import math

import torch
from torch import nn

FILM = "film_v1"
WIDE_FUSION = "wide_fusion_v1"
PRESENCE_MODULATIONS = (FILM, WIDE_FUSION)
MODALITY_COUNT = 5


def film_parameters(hidden_size):
    """Escala y desplazamiento por modalidad y coordenada, lineales en las cinco ausencias."""
    return 2 * MODALITY_COUNT * hidden_size * MODALITY_COUNT


def wide_units(hidden_size):
    """Unidades de la rama ancha con el número de parámetros más cercano al de FiLM.

    La rama tiene k (5H + 5) pesos de entrada y H k de salida, sin sesgos. Un número entero
    de unidades no siempre iguala la cuenta, así que la identidad registra las dos.
    """
    return max(1, round(film_parameters(hidden_size) / wide_parameters_per_unit(hidden_size)))


def wide_parameters_per_unit(hidden_size):
    return (MODALITY_COUNT * hidden_size + MODALITY_COUNT) + hidden_size


class PresenceFiLM(nn.Module):
    """Modulación de los cinco bloques fusionados según el vector de ausencias."""

    def __init__(self, hidden_size):
        super().__init__()
        shape = (MODALITY_COUNT, hidden_size, MODALITY_COUNT)
        self.scale = nn.Parameter(torch.zeros(shape))
        self.shift = nn.Parameter(torch.zeros(shape))

    def forward(self, blocks, presence):
        """Modular [lote, 5, H] con la presencia [lote, 5] y devolver la misma forma."""
        absence = (~presence).to(dtype=blocks.dtype)
        scale = torch.einsum("mhk,bk->bmh", self.scale, absence)
        shift = torch.einsum("mhk,bk->bmh", self.shift, absence)
        modulated = presence.unsqueeze(-1).to(dtype=blocks.dtype) * (
            blocks + scale * blocks + shift
        )
        # Las filas completas no pasan por la modulación. Así la identidad no depende de
        # que la suma con ceros conserve el signo de un cero.
        complete = presence.all(dim=1)
        return torch.where(complete[:, None, None], blocks, modulated)


class WideFusionBranch(nn.Module):
    """Rama residual de la fusión con k unidades ocultas y salida inicial nula."""

    def __init__(self, hidden_size, *, seed):
        super().__init__()
        units = wide_units(hidden_size)
        width = MODALITY_COUNT * hidden_size + MODALITY_COUNT
        self.expand = nn.Parameter(torch.empty(units, width))
        self.project = nn.Parameter(torch.zeros(hidden_size, units))
        # Mismo reparto inicial que nn.Linear, con un generador propio derivado de la semilla.
        # Así la rama no consume el generador global ni cambia los pesos del resto del
        # predictor, y sus sorteos no repiten los primeros del codificador de precios.
        digest = hashlib.sha256(f"{WIDE_FUSION}/{seed}".encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:7], "little"))
        nn.init.kaiming_uniform_(self.expand, a=math.sqrt(5), generator=generator)

    def forward(self, fusion_input):
        return nn.functional.silu(fusion_input @ self.expand.T) @ self.project.T
