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

`macro_film_v1` condiciona los bloques de cada activo con el estado macroeconómico. El bloque
macro codifica los 140 indicadores de la cesta, 70 series y 70 fórmulas con sus máscaras y sus
edades, y es el mismo para todos los activos de una sesión: sumado a la fusión desplaza a la vez
todas las filas y no puede cambiar su orden. La modulación hace que la sensibilidad a cada rasgo
propio del activo dependa de ese estado:

    z'_m = p_m · ((1 + tanh(Γ_m z_macro)) ⊙ z_m + Β_m z_macro)

para m en precios, noticias, gráficos y fundamentales. La tangente acota la escala entre 0 y 2.
Γ y Β empiezan en cero, así que el predictor emite al principio lo mismo que su control, y el
gradiente llega desde todas las filas, también las completas. El bloque macro pasa sin cambios.
"""

import hashlib
import math

import torch
from torch import nn

FILM = "film_v1"
WIDE_FUSION = "wide_fusion_v1"
MACRO_FILM = "macro_film_v1"
PRESENCE_MODULATIONS = (FILM, WIDE_FUSION, MACRO_FILM)
PRESENCE_SOFTMAX = "presence_softmax_v1"
MODALITY_GATINGS = (PRESENCE_SOFTMAX,)
MODALITY_COUNT = 5
# Los cuatro primeros bloques son propios del activo y el quinto es el macro de la sesión.
ASSET_BLOCKS = 4


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


def macro_film_parameters(hidden_size):
    """Escala y desplazamiento de los cuatro bloques del activo, lineales en el bloque macro."""
    return 2 * ASSET_BLOCKS * hidden_size * hidden_size


class MacroFiLM(nn.Module):
    """Modulación de los bloques de cada activo por el bloque macro de su sesión."""

    def __init__(self, hidden_size):
        super().__init__()
        shape = (ASSET_BLOCKS, hidden_size, hidden_size)
        self.scale = nn.Parameter(torch.zeros(shape))
        self.shift = nn.Parameter(torch.zeros(shape))

    def forward(self, blocks, presence):
        """Modular [lote, 5, H] con su propio bloque macro y devolver la misma forma."""
        assets, macro = blocks[:, :ASSET_BLOCKS], blocks[:, ASSET_BLOCKS]
        scale = torch.tanh(torch.einsum("mhk,bk->bmh", self.scale, macro))
        shift = torch.einsum("mhk,bk->bmh", self.shift, macro)
        present = presence[:, :ASSET_BLOCKS, None].to(dtype=blocks.dtype)
        # Una modalidad ausente sigue anulada: ni la escala ni el desplazamiento la rellenan.
        modulated = present * (assets + scale * assets + shift)
        return torch.cat((modulated, blocks[:, ASSET_BLOCKS:]), dim=1)


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


def modality_gating_parameters(hidden_size):
    """Pesos y sesgos por modalidad de la compuerta enmascarada por presencia."""
    return MODALITY_COUNT * (hidden_size + 1)


class PresenceModalityGate(nn.Module):
    """Compuerta softmax enmascarada por presencia sobre los cinco bloques de la fusión.

    Cada modalidad m calcula su logit solo desde su propio bloque activo:

        g_{b,m} = w_m^T (p_{b,m} z_{b,m}) + c_m

    y softmax solo normaliza sobre las modalidades presentes de la fila b, dejando
    alpha_{b,m} = 0 en toda modalidad ausente. Cada bloque se escala por M * alpha_{b,m}
    (M = 5), conservando la suma de pesos (= M) sea cual sea el patrón de ausencias.
    """

    def __init__(self, hidden_size, *, dtype=torch.float32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.gate_weight = nn.Parameter(torch.zeros(MODALITY_COUNT, self.hidden_size, dtype=dtype))
        self.gate_bias = nn.Parameter(torch.zeros(MODALITY_COUNT, dtype=dtype))

    def weights(self, blocks, presence):
        """Pesos normalizados [lote, 5] en [0, 1], cero en modalidades ausentes."""
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.ndim != 3
            or blocks.shape[1:] != (MODALITY_COUNT, self.hidden_size)
            or blocks.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("La compuerta de modalidades necesita bloques [lote, 5, H] FP32/FP64")
        if (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.shape != blocks.shape[:2]
            or presence.device != blocks.device
        ):
            raise ValueError("La compuerta de modalidades necesita presencia booleana [lote, 5]")
        active = torch.where(presence.unsqueeze(-1), blocks, torch.zeros_like(blocks))
        logits = torch.einsum("bmh,mh->bm", active, self.gate_weight) + self.gate_bias
        neg_inf = torch.full((), float("-inf"), dtype=blocks.dtype, device=blocks.device)
        any_present = presence.any(dim=1, keepdim=True)
        # El máximo desplazado no altera el gradiente de softmax y evita -inf en backward.
        max_logit = torch.where(
            any_present,
            torch.where(presence, logits, neg_inf).amax(dim=1, keepdim=True),
            torch.zeros_like(logits[:, :1]),
        ).detach()
        zero_logits = torch.zeros_like(logits)
        centered = torch.where(presence, logits - max_logit, zero_logits)
        exp_weights = torch.where(presence, torch.exp(centered), zero_logits)
        denom = exp_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return exp_weights / denom

    def forward(self, blocks, presence):
        """Reponderar [lote, 5, H] con la presencia [lote, 5] y devolver la misma forma."""
        alpha = self.weights(blocks, presence)
        active = torch.where(presence.unsqueeze(-1), blocks, torch.zeros_like(blocks))
        return (MODALITY_COUNT * alpha).unsqueeze(-1) * active
