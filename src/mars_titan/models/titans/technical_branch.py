"""Rama de los rasgos técnicos `technical_branch_v1`.

Los rasgos técnicos (`data.technical_features`) llegan como las últimas columnas del bloque del
gráfico y entran en el token fusionado por la misma proyección lineal que las 512 dimensiones de
la incrustación. Un modelo de árboles ajustado solo con ellos alcanza en el origen 2013 de
EE. UU. un IC de rangos de 0,110 en validación, y el núcleo que los recibe por esa vía se queda
por debajo. Los árboles parten cada rasgo por umbrales y combinan los tramos, y una proyección
lineal seguida de la fusión no puede hacerlo con la misma facilidad. Esta rama lee los rasgos
aparte, tras MAC y la mezcla entre activos y antes de la cabeza:

    c_{f,b} = recorte((x_f − u_b) / (u_{b+1} − u_b), 0, 1)        codificación lineal a trozos
    z_f = c_f W_f + b_f                                           incrustación propia del rasgo
    r = W_2 SiLU(LayerNorm(W_1 [z_1 … z_F] + b_1)) + b_2
    e' = e + g · r

x_f es el rasgo f ya normalizado por `learning_scale` y acotado a ±clip. Los bordes u_b no se
estiman con datos: son los cuantiles de una normal estándar en probabilidades equiespaciadas,
con los extremos en ±clip, iguales para todos los rasgos. La puerta g empieza en cero, así que
al construirse el predictor emite lo mismo que sin la rama, y el resto de pesos sale de un
generador derivado de la semilla y del nombre, sin consumir el generador global.

La codificación lineal a trozos es la de Gorishniy, Rubachev y Babenko (2022,
https://arxiv.org/abs/2203.05556), que con ella acercan una red sencilla a los árboles en datos
tabulares. Desviaciones declaradas: allí los bordes son cuantiles de cada columna y la
incrustación alimenta toda la red; aquí los bordes son fijos sobre entradas ya normalizadas y la
rama se suma como un residuo a la representación de cada flujo, sin cambiar el resto.

La rama no tiene estado, no mezcla filas y solo lee entradas disponibles en la emisión. El
residuo queda en la representación de trabajo, así que un lector ajustado sobre el núcleo lo
recibe sin cambios.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .config import bounded_integer, require_identity
from .state import check_finite

TECHNICAL_BRANCH = "technical_branch_v1"


@dataclass(frozen=True)
class TechnicalBranchConfig:
    """Columnas técnicas al final del bloque del gráfico y tamaño de la rama."""

    columns: int
    bins: int = 16
    embedding: int = 8
    hidden: int = 128

    def __post_init__(self):
        bounded_integer(self.columns, "columnas de la rama técnica", 1, 512)
        bounded_integer(self.bins, "tramos de la rama técnica", 2, 64)
        bounded_integer(self.embedding, "incrustación de la rama técnica", 1, 64)
        bounded_integer(self.hidden, "capa oculta de la rama técnica", 1, 1024)

    def identity(self):
        return dict(
            schema_version=1,
            kind=TECHNICAL_BRANCH,
            columns=self.columns,
            bins=self.bins,
            embedding=self.embedding,
            hidden=self.hidden,
            source="last_columns_of_the_chart_block_after_learning_scale",
            edges="standard_normal_quantiles_at_equal_probabilities_with_ends_at_the_clip",
            encoding="piecewise_linear_per_feature",
            position="after_mac_and_session_mixers_before_head",
            output="residual_with_zero_initialized_gate",
            state="none",
            parameter_updates="none_in_forward",
        )


def parse_technical_branch(raw):
    """Aceptar la configuración, sus cuatro valores o None."""
    if raw is None:
        return None
    if type(raw) is TechnicalBranchConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"columns", "bins", "embedding", "hidden"}:
        return TechnicalBranchConfig(**raw)
    raise ValueError("technical_branch debe declarar columns, bins, embedding y hidden, o ser None")


def normal_edges(bins, clip, dtype=torch.float32):
    """Bordes crecientes: cuantiles de la normal estándar con los extremos en ±clip."""
    probabilities = torch.arange(1, bins, dtype=torch.float64) / bins
    inner = torch.special.ndtri(probabilities)
    if not -clip < inner[0] or not inner[-1] < clip:
        raise ValueError("El recorte no deja sitio a los tramos de la rama técnica")
    edges = torch.cat(
        [
            torch.tensor([-clip], dtype=torch.float64),
            inner,
            torch.tensor([clip], dtype=torch.float64),
        ]
    ).to(dtype)
    if not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("Los bordes de la rama técnica no son crecientes en su precisión")
    return edges


class TechnicalBranch(nn.Module):
    """Residuo de la representación de cada flujo calculado con sus rasgos técnicos."""

    def __init__(self, hidden_size, config, *, clip, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de la rama técnica", 1, 4096)
        bounded_integer(seed, "semilla de la rama técnica", 0, 2**32 - 1)
        if type(config) is not TechnicalBranchConfig:
            raise ValueError("La rama técnica necesita TechnicalBranchConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La rama técnica necesita precisión FP32 o FP64")
        if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not 1 <= clip <= 100:
            raise ValueError("La rama técnica necesita el recorte de learning_scale")
        self.config, self.hidden_size, self.clip = config, hidden_size, float(clip)
        self.register_buffer("edges", normal_edges(config.bins, self.clip, dtype))
        derived = hashlib.sha256(f"{TECHNICAL_BRANCH}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            # Cada rasgo tiene su propia proyección de sus tramos, con la escala de nn.Linear.
            bound = config.bins**-0.5
            shape = (config.columns, config.bins, config.embedding)
            self.embedding_weight = nn.Parameter(
                torch.empty(shape, dtype=dtype).uniform_(-bound, bound)
            )
            self.embedding_bias = nn.Parameter(
                torch.empty(shape[::2], dtype=dtype).uniform_(-bound, bound)
            )
            self.inner = nn.Linear(config.columns * config.embedding, config.hidden, dtype=dtype)
            self.norm = nn.LayerNorm(config.hidden, dtype=dtype)
            self.outer = nn.Linear(config.hidden, hidden_size, dtype=dtype)
        self.gate = nn.Parameter(torch.zeros((), dtype=dtype))

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(), hidden_size=self.hidden_size, clip=self.clip
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def encode(self, features):
        """Codificación lineal a trozos [filas, columnas, tramos], entre 0 y 1."""
        left, right = self.edges[:-1], self.edges[1:]
        position = features.unsqueeze(-1).clamp(min=left, max=right)
        return ((position - left) / (right - left)).clamp(0.0, 1.0)

    def forward(self, charts):
        """Residuo [filas, dimensión] a partir del bloque del gráfico ya normalizado."""
        config = self.config
        if (
            not isinstance(charts, torch.Tensor)
            or charts.ndim != 2
            or charts.shape[1] < config.columns
            or charts.dtype != self.gate.dtype
        ):
            raise ValueError("La rama técnica necesita el bloque del gráfico con sus columnas")
        encoded = self.encode(charts[:, charts.shape[1] - config.columns :])
        embedded = torch.einsum("rfb,fbe->rfe", encoded, self.embedding_weight)
        embedded = embedded + self.embedding_bias
        hidden = nn.functional.silu(self.norm(self.inner(embedded.flatten(1))))
        residual = self.gate * self.outer(hidden)
        check_finite(residual, "El residuo de la rama técnica")
        return residual
