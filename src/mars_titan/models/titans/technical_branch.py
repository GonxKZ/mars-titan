"""Ramas de los rasgos técnicos: `technical_branch_v1` y `technical_branch_v2`.

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

`technical_branch_v2` es la misma rama sin esa codificación: una red de capas densas sobre los
rasgos ya normalizados,

    h_0 = x,   h_k = SiLU(W_k h_{k−1} + b_k)  para k = 1 … depth,   e' = e + W_s h_depth + b_s

con W_s y b_s nulos al construirse, así que el predictor también empieza emitiendo lo mismo que
sin la rama, sin necesidad de puerta. Es la forma que mejor se ajustó en el origen 2013 de
EE. UU. con los 46 rasgos solos (IC de validación): con la pérdida de la mediana, 0,104 la red
densa y 0,101 con la codificación a trozos; con la pérdida de IC por sesión, 0,115 y 0,109. Un
modelo lineal con las mismas columnas se queda en 0,090.

Con `session_ranks` la red recibe además, junto a cada rasgo, la puntuación normal de su rango
entre las filas de su sesión, es decir, del mismo mercado y del mismo instante de decisión:

    h_0 = [x, g],   g_f = Φ⁻¹((r_f − 1/2) / n)

con r_f el rango de la fila en el rasgo f entre las n filas del bloque de su mercado, con los
empates promediados (`session_ranks.gaussian_ranks`). El valor normalizado conserva el nivel
del rasgo en la historia y el rango, su posición frente a los demás activos ese día. Con los
46 rasgos solos y la pérdida de IC, la red con ambos llega a 0,118 de IC de validación y 0,117
de evaluación, frente a 0,115 y 0,113 sin los rangos; solo con los rangos se queda en 0,111.
La transformación solo lee filas del bloque, que tiene un único instante y entradas disponibles
en él. Como los rangos de un subconjunto serían otros, el predictor la exige con la sesión
completa, igual que `session_rank_inputs_v1`.

Las ramas no tienen estado y solo leen entradas disponibles en la emisión. Sin `session_ranks`
no mezclan filas. El residuo queda en la representación de trabajo, así que un lector ajustado
sobre el núcleo lo recibe sin cambios.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from .config import LAYER_NORM_EPS, bounded_integer, require_identity
from .session_ranks import gaussian_ranks
from .state import check_finite

TECHNICAL_BRANCH = "technical_branch_v1"
TECHNICAL_DENSE = "technical_branch_v2"


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


@dataclass(frozen=True)
class TechnicalDenseConfig:
    """Columnas técnicas al final del bloque del gráfico y tamaño de la red densa de la rama."""

    columns: int
    hidden: int = 128
    depth: int = 2
    # Añadir a cada rasgo la puntuación normal de su rango entre las filas de su sesión.
    session_ranks: bool = False
    # Aplicar LayerNorm antes de SiLU en cada capa oculta (desactivado por defecto).
    layer_norm: bool = False
    # Multiplicar cada entrada de la red por una escala aprendida que empieza en uno.
    feature_scaling: bool = False

    def __post_init__(self):
        bounded_integer(self.columns, "columnas de la rama técnica", 1, 512)
        bounded_integer(self.hidden, "capa oculta de la rama técnica", 1, 1024)
        bounded_integer(self.depth, "capas ocultas de la rama técnica", 1, 8)
        if type(self.session_ranks) is not bool:
            raise ValueError("session_ranks de la rama técnica debe ser un booleano")
        if type(self.layer_norm) is not bool:
            raise ValueError("layer_norm de la rama técnica debe ser un booleano")
        if type(self.feature_scaling) is not bool:
            raise ValueError("feature_scaling de la rama técnica debe ser un booleano")
        if self.layer_norm and self.hidden < 2:
            raise ValueError("LayerNorm en la rama técnica necesita hidden >= 2")

    def identity(self):
        result = dict(
            schema_version=1,
            kind=TECHNICAL_DENSE,
            columns=self.columns,
            hidden=self.hidden,
            depth=self.depth,
            source="last_columns_of_the_chart_block_after_learning_scale",
            encoding="none_normalized_features",
            activation="silu_after_each_hidden_layer",
            position="after_mac_and_session_mixers_before_head",
            output="residual_with_zero_initialized_output_layer",
            state="none",
            parameter_updates="none_in_forward",
        )
        # Sin los rangos se conserva literalmente la identidad anterior.
        if self.session_ranks:
            result.update(
                encoding="normalized_features_then_their_session_normal_scores",
                session="rows_of_one_market_and_decision_instant_in_one_complete_block",
                normal_scores="inverse_normal_of_average_rank_minus_half_over_the_rows",
            )
        # Sin LayerNorm se conserva literalmente la identidad anterior.
        if self.layer_norm:
            result.update(
                layer_norm=True,
                normalization="layer_norm_before_silu_on_each_hidden_layer",
                layer_norm_eps=LAYER_NORM_EPS,
                layer_norm_affine=True,
            )
        # Sin la escala por entrada se conserva literalmente la identidad anterior.
        if self.feature_scaling:
            result["feature_scaling"] = "learned_scale_per_input_initialized_at_one_v1"
        return result


def parse_technical_branch(raw):
    """Aceptar una configuración, los valores de una de las dos ramas o None."""
    if raw is None:
        return None
    if type(raw) in (TechnicalBranchConfig, TechnicalDenseConfig):
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"columns", "bins", "embedding", "hidden"}:
        return TechnicalBranchConfig(**raw)
    if isinstance(raw, dict) and {"columns", "hidden", "depth"} <= set(raw) <= {
        "columns",
        "hidden",
        "depth",
        "session_ranks",
        "layer_norm",
        "feature_scaling",
    }:
        return TechnicalDenseConfig(**raw)
    raise ValueError(
        "technical_branch debe declarar columns, bins, embedding y hidden, o columns, hidden y "
        "depth (y opcionalmente session_ranks, layer_norm y feature_scaling), o ser None"
    )


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

    # No mezcla filas: no necesita la sesión completa en un bloque.
    session_level = False

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


class TechnicalDense(nn.Module):
    """Residuo de la representación de cada flujo: red densa sobre sus rasgos técnicos."""

    def __init__(self, hidden_size, config, *, clip, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de la rama técnica", 1, 4096)
        bounded_integer(seed, "semilla de la rama técnica", 0, 2**32 - 1)
        if type(config) is not TechnicalDenseConfig:
            raise ValueError("La rama técnica densa necesita TechnicalDenseConfig")
        config.__post_init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("La rama técnica necesita precisión FP32 o FP64")
        if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not 1 <= clip <= 100:
            raise ValueError("La rama técnica necesita el recorte de learning_scale")
        self.config, self.hidden_size, self.clip = config, hidden_size, float(clip)
        derived = hashlib.sha256(f"{TECHNICAL_DENSE}/{seed}".encode()).digest()
        first = config.columns * (2 if config.session_ranks else 1)
        widths = [first, *([config.hidden] * config.depth)]
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.layers = nn.ModuleList(
                nn.Linear(before, after, dtype=dtype)
                for before, after in zip(widths[:-1], widths[1:], strict=True)
            )
            # Se crea con el generador derivado para no consumir el global.
            self.outer = nn.Linear(config.hidden, hidden_size, dtype=dtype)
        # La salida empieza en cero: el predictor emite lo mismo que sin la rama.
        with torch.no_grad():
            self.outer.weight.zero_()
            self.outer.bias.zero_()
        # LayerNorm se construye fuera del generador derivado (pesos 1, sesgo 0, sin consumo RNG).
        self.norms = (
            nn.ModuleList(
                nn.LayerNorm(config.hidden, eps=LAYER_NORM_EPS, dtype=dtype)
                for _ in range(config.depth)
            )
            if config.layer_norm
            else None
        )
        # La escala por entrada empieza en uno: la rama calcula lo mismo hasta que se ajusta.
        self.input_scale = (
            nn.Parameter(torch.ones(first, dtype=dtype)) if config.feature_scaling else None
        )

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(), hidden_size=self.hidden_size, clip=self.clip
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    @property
    def session_level(self):
        """Si la rama necesita todas las filas de la sesión en el mismo bloque."""
        return self.config.session_ranks

    def ranks(self, features, markets):
        """Puntuaciones normales del rango de cada rasgo entre las filas de su mercado."""
        if markets is None or len(markets) != len(features):
            raise ValueError("Los rangos de la rama técnica necesitan el mercado de cada fila")
        result = torch.zeros_like(features)
        for market in sorted(set(markets)):
            rows = torch.tensor(
                [index for index, value in enumerate(markets) if value == market],
                dtype=torch.int64,
                device=features.device,
            )
            values = features.index_select(0, rows)
            present = torch.ones_like(values, dtype=torch.bool)
            result = result.index_copy(0, rows, gaussian_ranks(values, present))
        return result

    def forward(self, charts, markets=None):
        """Residuo [filas, dimensión] a partir del bloque del gráfico ya normalizado.

        `markets` es el mercado de cada fila del bloque y solo se usa con `session_ranks`.
        """
        config = self.config
        if (
            not isinstance(charts, torch.Tensor)
            or charts.ndim != 2
            or charts.shape[1] < config.columns
            or charts.dtype != self.outer.weight.dtype
        ):
            raise ValueError("La rama técnica necesita el bloque del gráfico con sus columnas")
        if bool(config.layer_norm) != (self.norms is not None):
            raise ValueError("La normalización de la rama técnica no coincide con su configuración")
        hidden = charts[:, charts.shape[1] - config.columns :]
        if config.session_ranks:
            # Los rangos no llevan gradiente hacia las entradas: son una función a trozos constante.
            hidden = torch.cat((hidden, self.ranks(hidden.detach(), markets)), dim=1)
        if self.input_scale is not None:
            hidden = hidden * self.input_scale
        if self.norms is not None:
            for layer, norm in zip(self.layers, self.norms, strict=True):
                hidden = nn.functional.silu(norm(layer(hidden)))
        else:
            for layer in self.layers:
                hidden = nn.functional.silu(layer(hidden))
        residual = self.outer(hidden)
        check_finite(residual, "El residuo de la rama técnica")
        return residual


def build_technical_branch(hidden_size, config, *, clip, seed, dtype=torch.float32):
    """El módulo de la rama que declara la configuración."""
    kind = TechnicalDense if type(config) is TechnicalDenseConfig else TechnicalBranch
    return kind(hidden_size, config, clip=clip, seed=seed, dtype=dtype)
