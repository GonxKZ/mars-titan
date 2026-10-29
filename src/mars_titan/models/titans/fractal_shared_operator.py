"""Operador compartido entre los niveles de la jerarquía de una sesión
`fractal_shared_operator_v1` (C3 de #751).

En cada sesión de un mercado la jerarquía tiene cuatro niveles: el mercado, sus sectores
estadísticos, sus activos y las cinco hojas de cada activo (precios, noticias, gráficos,
fundamentales y macro). Con z_{i,m} el bloque de la fuente m del activo i, P_i sus fuentes
presentes y g(i) su sector,

    a_i = media de z_{i,m} con m ∈ P_i
    s_g = media de a_i con g(i) = g
    c   = media de los s_g de la sesión

el mismo operador recorre la jerarquía de arriba abajo, una vez por cada nivel con hijos:

    s'_g     = s_g + O_mercado(s_g; {c} ∪ {s_h})
    a'_i     = a_i + O_sector(a_i; {s'_g(i)} ∪ {a_j : g(j) = g(i)})
    z'_{i,m} = z_{i,m} + O_activo(z_{i,m}; {a'_i} ∪ {z_{i,k} : k ∈ P_i})

O_ℓ(x; C) es una atención con normalización previa y varias cabezas, con la consulta desde x
y las claves y los valores desde C. Sus cuatro proyecciones valen W + B_ℓ A_ℓ: W es común a
los tres niveles y cada nivel añade una LoRA de rango r. Es la relajación del reparto de capas
de ALBERT (Lan et al., 2020, sección 3.1, página 4) que proponen Bae et al. (2025) con una LoRA
por profundidad. La LoRA sigue a Hu et al. (2022, sección 4.1, página 4), pero con A
inicializada como `nn.Linear` y B a cero, igual que `loralib` (commit c4593f0, `layers.py`,
líneas 119-125), y con escala α/r = 1.

Los sectores son estadísticos y en fecha. Los activos de la sesión se agrupan por la
correlación de sus rendimientos diarios en la ventana de precios recibida, con la regla de C4
(`data/correlation_regions.py`). La ventana termina en la sesión de la decisión, así que el
reparto no lee nada posterior a la emisión. Los activos sin `min_returns` rendimientos o con
la serie constante forman el sector 0.

Controles, todos con la misma semilla:

- `per_level`: un operador propio por nivel, con proyecciones U_ℓ V_ℓ de rango R y su propia
  normalización. R es el menor rango que no deja al control con menos parámetros que la
  variante.
- `all_shared`: el mismo operador en los tres niveles, sin LoRA.
- `random_partition`: la variante con los tamaños de sus sectores repartidos al azar entre los
  activos con historia de cada sesión.
- `dense_block`: los mismos parámetros y tres aplicaciones, pero cada hoja presente atiende a
  todas las hojas presentes de la sesión, sin nodos de activo, sector ni mercado.

Las proyecciones de salida empiezan en cero, así que al construirse la salida es z y el
predictor emite lo mismo que su referencia. Una hoja ausente conserva su bloque y no entra como
clave. C3 necesita todas las filas de la sesión en el mismo bloque y no guarda estado.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.correlation_regions import correlation_regions, permuted_regions
from mars_titan.data.price_windows import PRICE_FEATURES, PRICE_WINDOW_CHANNELS

from ..baselines.multimodal import MODALITIES
from .config import bounded_integer, require_identity
from .state import check_finite

FRACTAL_SHARED_OPERATOR = "fractal_shared_operator_v1"
SHARED_LORA = "shared_lora"
PER_LEVEL = "per_level"
ALL_SHARED = "all_shared"
RANDOM_PARTITION = "random_partition"
DENSE_BLOCK = "dense_block"
KINDS = (SHARED_LORA, PER_LEVEL, ALL_SHARED, RANDOM_PARTITION, DENSE_BLOCK)
LORA_KINDS = (SHARED_LORA, RANDOM_PARTITION, DENSE_BLOCK)
LEVELS = ("market", "sector", "asset")
HEADS = (1, 2, 4, 8)
LEAVES = len(MODALITIES)
CLOSE = PRICE_FEATURES.index("log_close_to_anchor_close")
# Consulta, clave, valor y salida, en este orden en la primera dimensión de los pesos.
PROJECTIONS = 4


@dataclass(frozen=True)
class FractalSharedOperatorConfig:
    kind: str = SHARED_LORA
    rank: int = 4
    heads: int = 4
    sectors: int = 8
    min_returns: int = 32

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError(f"C3 admite los tipos {KINDS}")
        bounded_integer(self.rank, "rango de las LoRA de C3", 1, 32)
        if type(self.heads) is not int or self.heads not in HEADS:
            raise ValueError(f"C3 admite {HEADS} cabezas")
        bounded_integer(self.sectors, "sectores de C3", 2, 32)
        bounded_integer(self.min_returns, "rendimientos mínimos de C3", 2, 1024)

    def identity(self):
        flat = self.kind == DENSE_BLOCK
        return dict(
            schema_version=1,
            kind=FRACTAL_SHARED_OPERATOR,
            operator=self.kind,
            rank=self.rank,
            heads=self.heads,
            sectors=self.sectors,
            min_returns=self.min_returns,
            leaves=list(MODALITIES),
            applications=list(LEVELS),
            hierarchy="every_present_leaf_of_the_session_in_each_application"
            if flat
            else "market_then_sectors_then_assets_then_leaves_of_each_session",
            partition="none"
            if flat
            else "statistical_sector_sizes_assigned_at_random_per_session"
            if self.kind == RANDOM_PARTITION
            else "statistical_sectors_of_the_session",
            returns="daily_differences_of_log_close_in_the_received_price_window",
            clustering="average_linkage_on_one_minus_correlation_as_in_c4",
            projections="independent_rank_R_factors_and_norm_per_application"
            if self.kind == PER_LEVEL
            else "shared_without_lora"
            if self.kind == ALL_SHARED
            else "shared_plus_lora_per_application_kaiming_A_zero_B_scale_one",
            output="residuals_with_zero_initialized_output_projections",
            absent_leaves="keep_their_block_and_are_not_keys",
            state="none",
            parameter_updates="none_in_forward",
        )


def parse_fractal_shared_operator(raw):
    """Aceptar la configuración, sus valores completos o None."""
    if raw is None:
        return None
    if type(raw) is FractalSharedOperatorConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == {"kind", "rank", "heads", "sectors", "min_returns"}:
        return FractalSharedOperatorConfig(**raw)
    raise ValueError(
        "fractal_shared_operator debe declarar kind, rank, heads, sectors y min_returns, o ser None"
    )


def control_rank(hidden_size, rank):
    """Menor R con el que el control por niveles no tiene menos parámetros que la variante.

    La variante tiene 2H + 4H² + 3·4·2rH parámetros y el control 3·(2H + 4·2RH).
    """
    return math.ceil((hidden_size + 6 * rank - 1) / 6)


class FractalSharedOperator(nn.Module):
    """Recorrido de arriba abajo de la jerarquía de cada sesión antes de la fusión."""

    session_level = True

    def __init__(self, hidden_size, config, *, seed, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de C3", 1, 4096)
        bounded_integer(seed, "semilla de C3", 0, 2**32 - 1)
        if type(config) is not FractalSharedOperatorConfig:
            raise ValueError("C3 necesita FractalSharedOperatorConfig")
        config.__post_init__()
        if hidden_size % config.heads:
            raise ValueError("La dimensión de C3 debe ser múltiplo del número de cabezas")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("C3 necesita precisión FP32 o FP64")
        self.config, self.hidden_size, self.seed = config, hidden_size, seed
        self.head_dim = hidden_size // config.heads
        size, levels = hidden_size, len(LEVELS)
        # La semilla deriva de la del predictor y no depende del tipo, así que los tipos con
        # operador compartido empiezan con los mismos pesos y ninguno consume el generador global.
        derived = hashlib.sha256(f"{FRACTAL_SHARED_OPERATOR}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            if config.kind == PER_LEVEL:
                self.control_rank = control_rank(hidden_size, config.rank)
                self.norms = nn.ModuleList(nn.LayerNorm(size, dtype=dtype) for _ in LEVELS)
                self.right = nn.Parameter(
                    torch.empty(levels, PROJECTIONS, self.control_rank, size, dtype=dtype)
                )
                self.left = nn.Parameter(
                    torch.empty(levels, PROJECTIONS, size, self.control_rank, dtype=dtype)
                )
                # V como `nn.Linear` y U con varianza 1/R, de modo que U V tiene por entrada la
                # varianza 1/(3H) de la proyección compartida de la variante.
                bound = math.sqrt(3 / self.control_rank)
                with torch.no_grad():
                    for right, left in zip(
                        self.right.flatten(0, 1), self.left.flatten(0, 1), strict=True
                    ):
                        nn.init.kaiming_uniform_(right, a=math.sqrt(5))
                        left.uniform_(-bound, bound)
                    self.left[:, -1].zero_()
            else:
                self.norm = nn.LayerNorm(size, dtype=dtype)
                self.weight = nn.Parameter(torch.empty(PROJECTIONS, size, size, dtype=dtype))
                with torch.no_grad():
                    for weight in self.weight:
                        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                    self.weight[-1].zero_()
                if config.kind in LORA_KINDS:
                    self.down = nn.Parameter(
                        torch.empty(levels, PROJECTIONS, config.rank, size, dtype=dtype)
                    )
                    self.up = nn.Parameter(
                        torch.zeros(levels, PROJECTIONS, size, config.rank, dtype=dtype)
                    )
                    with torch.no_grad():
                        for down in self.down.flatten(0, 1):
                            nn.init.kaiming_uniform_(down, a=math.sqrt(5))

    def get_extra_state(self):
        state = dict(configuration=self.config.identity(), hidden_size=self.hidden_size)
        # En los demás tipos la semilla solo fija los pesos iniciales. Aquí también decide el
        # reparto de cada sesión, así que unos pesos no se cargan con otra semilla.
        if self.config.kind == RANDOM_PARTITION:
            state["partition_seed"] = self.seed
        return state

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    @property
    def reference(self):
        return self.left if self.config.kind == PER_LEVEL else self.weight

    def _operator(self, level):
        """Normalización y proyecciones efectivas [4, H, H] de una aplicación."""
        if self.config.kind == PER_LEVEL:
            return self.norms[level], self.left[level] @ self.right[level]
        if self.config.kind == ALL_SHARED:
            return self.norm, self.weight
        return self.norm, self.weight + self.up[level] @ self.down[level]

    def _attend(self, level, queries, keys, allowed):
        """Atención residual de consultas [..., Q, H] sobre claves [..., K, H] con máscara."""
        norm, (query, key, value, out) = self._operator(level)
        heads, width = self.config.heads, self.head_dim

        def split(inputs, weight):
            return (inputs @ weight.T).unflatten(-1, (heads, width)).transpose(-2, -3)

        normalized = norm(keys)
        q, k, v = split(norm(queries), query), split(normalized, key), split(normalized, value)
        logits = q @ k.transpose(-1, -2) / math.sqrt(width)
        weights = torch.softmax(logits.masked_fill(~allowed.unsqueeze(-3), -math.inf), dim=-1)
        return queries + (weights @ v).transpose(-2, -3).flatten(-2) @ out.T

    def _partition_seed(self, market, prediction_at):
        text = f"{FRACTAL_SHARED_OPERATOR}/{self.seed}/{market}/{prediction_at}"
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "little")

    def sessions(self, flow_ids, prediction_at, prices):
        """Filas de cada sesión del bloque y el sector de cada una, numerado desde cero.

        Las sesiones siguen el orden de primera aparición y `dense_block` no tiene sectores.
        Para agruparlos, la ventana de precios se copia una vez a la CPU en FP64.
        """
        groups = {}
        for row, key in enumerate(zip(flow_ids, prediction_at, strict=True)):
            groups.setdefault((key[0].split("/", 1)[0], key[1]), []).append(row)
        if self.config.kind == DENSE_BLOCK:
            return [(rows, None) for rows in groups.values()]
        window = prices.detach().to(device="cpu", dtype=torch.float64).numpy()
        observed = (
            window[..., -1] > 0
            if window.shape[-1] == len(PRICE_WINDOW_CHANNELS)
            else np.ones(window.shape[:2], dtype=bool)
        )
        returns = np.diff(window[..., CLOSE], axis=1)
        valid = observed[:, 1:] & observed[:, :-1]
        result = []
        for (market, at), rows in groups.items():
            ordered = sorted(rows, key=flow_ids.__getitem__)
            regions = correlation_regions(
                [flow_ids[row] for row in ordered],
                returns[ordered],
                valid[ordered],
                clusters=self.config.sectors,
                min_observations=self.config.min_returns,
            )
            if self.config.kind == RANDOM_PARTITION:
                regions = permuted_regions(regions, self._partition_seed(market, at))
            number = {name: i for i, name in enumerate(sorted(set(regions.values()), key=int))}
            result.append((rows, [number[regions[flow_ids[row]]] for row in rows]))
        return result

    def _hierarchy(self, blocks, present, sessions):
        device = blocks.device
        share = present.unsqueeze(-1).to(blocks.dtype)
        assets = (blocks * share).sum(1) / share.sum(1).clamp_min(1)
        updated = assets
        for rows, sectors in sessions:
            index = torch.tensor(rows, dtype=torch.int64, device=device)
            members = assets.index_select(0, index)
            group = torch.tensor(sectors, dtype=torch.int64, device=device)
            membership = F.one_hot(group, max(sectors) + 1).to(blocks.dtype)
            nodes = membership.T @ members / membership.sum(0).unsqueeze(-1)
            market = nodes.mean(0, keepdim=True)
            # Cada sector atiende al mercado y a todos los sectores de la sesión.
            everything = torch.ones(len(nodes), len(nodes) + 1, dtype=torch.bool, device=device)
            nodes = self._attend(0, nodes, torch.cat((market, nodes)), everything)
            # Cada activo atiende a su sector ya actualizado y a los activos de su sector.
            allowed = torch.cat((membership.bool(), group.unsqueeze(1) == group), dim=1)
            part = self._attend(1, members, torch.cat((nodes, members)), allowed)
            updated = updated.index_copy(0, index, part)
        # Cada hoja atiende a su activo y a las hojas presentes del mismo activo. El activo
        # siempre es clave, así que tampoco la consulta de una hoja ausente queda vacía.
        context = torch.cat((updated.unsqueeze(1), blocks), dim=1)
        keys = torch.cat((torch.ones_like(present[:, :1]), present), dim=1)
        return self._attend(2, blocks, context, keys.unsqueeze(1).expand(-1, LEAVES, -1))

    def _dense(self, blocks, present, sessions):
        result = blocks
        for rows, _ in sessions:
            index = torch.tensor(rows, dtype=torch.int64, device=blocks.device)
            leaves = blocks.index_select(0, index).flatten(0, 1)
            itself = torch.eye(len(leaves), dtype=torch.bool, device=blocks.device)
            # Las hojas presentes son claves de todas. Una ausente solo se atiende a sí misma.
            allowed = present.index_select(0, index).flatten().unsqueeze(0) | itself
            for level in range(len(LEVELS)):
                leaves = self._attend(level, leaves, leaves, allowed)
            result = result.index_copy(0, index, leaves.view(len(rows), LEAVES, -1))
        return result

    def forward(self, blocks, presence, flow_ids, prediction_at, prices):
        """Bloques [filas, 5, H] tras recorrer la jerarquía de las sesiones del bloque."""
        reference = self.reference
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.layout != torch.strided
            or blocks.ndim != 3
            or blocks.shape[1:] != (LEAVES, self.hidden_size)
            or len(blocks) == 0
            or blocks.dtype != reference.dtype
            or blocks.device != reference.device
        ):
            raise ValueError("Los bloques de C3 no corresponden al contrato")
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("C3 exige FP32 o FP64 explícitos, sin autocast")
        rows = len(blocks)
        if presence is None:
            present = torch.ones(rows, LEAVES, dtype=torch.bool, device=blocks.device)
        elif (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.shape != (rows, LEAVES)
            or presence.device != blocks.device
        ):
            raise ValueError("La presencia de C3 debe ser booleana [filas, 5]")
        else:
            present = presence
        if (
            type(flow_ids) is not tuple
            or type(prediction_at) is not tuple
            or len(flow_ids) != rows
            or len(prediction_at) != rows
            or any(type(flow) is not str or "/" not in flow for flow in flow_ids)
            or len(set(flow_ids)) != rows
        ):
            raise ValueError("C3 necesita un flujo distinto y un instante por fila")
        if (
            not isinstance(prices, torch.Tensor)
            or prices.ndim != 3
            or len(prices) != rows
            or prices.shape[-1] not in (len(PRICE_FEATURES), len(PRICE_WINDOW_CHANNELS))
            or prices.shape[1] - 1 < self.config.min_returns
            or prices.device != blocks.device
        ):
            raise ValueError("C3 necesita la ventana de precios de cada fila y min_returns pasos")
        sessions = self.sessions(flow_ids, prediction_at, prices)
        if self.config.kind == DENSE_BLOCK:
            leaves = self._dense(blocks, present, sessions)
        else:
            leaves = self._hierarchy(blocks, present, sessions)
        result = torch.where(present.unsqueeze(-1), leaves, blocks)
        check_finite(result, "Los bloques con el operador fractal")
        return result
