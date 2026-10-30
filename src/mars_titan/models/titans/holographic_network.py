"""Red de redes holográfica de #783 en el nivel de la fila: H1 y sus ablaciones H2 y H3.

El mismo dato se lee con pesos distintos en varias áreas a la vez, sin copiarlo, y las áreas
se comunican por un puente. Es una combinación propia de piezas publicadas: la pertenencia
fraccionaria de MaGNet (https://arxiv.org/abs/2511.00085), las ranuras de Soft MoE
(Puigcerver y otros, ICLR 2024, https://arxiv.org/abs/2308.00951), ajustes de rango bajo
por área y el calendario del operador regional de `regional_fusion`.

Bloques. Con 5 áreas cada bloque es la representación de una modalidad. Con 19 o 38, los
fundamentales y el macro se parten por familias: el bloque de una familia es la salida del
codificador de su modalidad cuando solo ve las columnas de esa familia,

    z_f = SiLU(W[:, F] x[F] + b),

con los mismos W y b de la referencia, así que no añade parámetros. Las familias del macro
son las doce categorías del catálogo y las de los fundamentales, los conceptos contables y
los ratios de empresa. Precios, noticias y gráficos llegan como un único vector, porque sus
codificadores no separan canales. Con 38 áreas cada familia se parte en sus valores y en su
observación (máscara y edad), y cada vector único se desdobla en dos vistas.

Pertenencia. Cada bloque b se reparte entre las áreas a con

    p_ab = softmax_a(π_ab + Δ_ab),

donde π es el logaritmo del peso del catálogo, con 1 − `leakage` repartido entre las áreas
de la familia del bloque y `leakage` entre las demás, y Δ empieza en cero. H2
(`catalog_membership_only_v1`) fija Δ = 0.

Áreas. El estado inicial del área a es

    h_a = LayerNorm(m_a + U_a V_aᵀ m_a + e_a),   m_a = Σ_b p_ab z_b,

así que el área a lee cada bloque con (I + U_a V_aᵀ) z_b. El proyector compartido W_b de la
issue es el propio codificador de la referencia, que ya produce z_b, y el ajuste de rango bajo
se compone tras él en lugar de sumarse a sus pesos. U empieza en cero, como en LoRA, y e_a es
un embedding propio del área, sin identidad de activo.

Grupos ocultos. Cada ranura s mezcla los bloques presentes con
d_sb = softmax_b(k_s · LayerNorm(z_b) / √H) y su estado inicial es
LayerNorm(Σ_b d_sb z_b + f_s). La lectura de vuelta usa c_bs = softmax_s de los mismos
logits, como en Soft MoE. H3 (`hidden_groups_only_v1`) conserva solo las ranuras.

Puente. Las áreas y las ranuras son los nodos de J ciclos con pasos locales de atención
enmascarada y un resumen por ciclo, el calendario de `RegionalFusion`. La máscara une las
áreas en anillo, en anillo con cuerdas o de forma completa, y cada ranura con todos los
nodos. Como en `RegionalFusion`, un área cuyo bloque falta empieza en cero, no envía
mensajes y no entra en el resumen, que promedia los nodos presentes. El cambio de cada nodo,
Δ_n = local_n − inicial_n, vuelve a los bloques por la pertenencia y las ranuras, con el
contexto del resumen una sola vez,

    δ_b = Σ_a p_ab Δ_a + Σ_s c_bs Δ_s + contexto,

y cada modalidad m recibe la media de sus bloques. La salida es blocks_m + W_o media_{b∈m} δ_b
en las modalidades presentes. Con la media, la escala de la corrección no depende de en
cuántos bloques se parte la modalidad, así que 5, 19 y 38 áreas son comparables. W_o empieza
en cero, así que el predictor emite al construirse lo mismo que sin la red. El estado de
trabajo se reinicia en cada llamada, cada fila se calcula sola y el forward no actualiza
parámetros.

Pesos rápidos de P1 (#784). Con `edge_bias` [filas, aristas], el peso de cada par de áreas
distintas que une la topología se suma al logit de atención de la consulta i sobre la clave j
en todos los pasos del puente, y `traces` recibe la media de esas probabilidades de atención
en los ciclos y pasos, sin grafo. La traza de un área ausente vale cero, porque su consulta
solo evita una fila vacía en la máscara y su estado se descarta. Los pesos y su regla viven
fuera de la red, en
`mars_titan.memory.three_factor_bridge`. Con pesos nulos la máscara real de ceros y −∞ da la
misma atención que la booleana, así que la salida no cambia.
"""

import csv
import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.accounting_catalog import CAD_CONCEPTS, CNY_CONCEPTS
from mars_titan.data.company_factors import FACTOR_CONCEPTS
from mars_titan.data.samples import FUNDAMENTAL_CONCEPTS

from ..baselines.multimodal import MODALITIES
from .config import MAX_BLOCK_ROWS, bounded_integer, require_identity
from .state import check_finite, deferred_checks

HOLOGRAPHIC_NETWORK = "holographic_network_v1"
CATALOG_MEMBERSHIP_ONLY = "catalog_membership_only_v1"
HIDDEN_GROUPS_ONLY = "hidden_groups_only_v1"
KINDS = (HOLOGRAPHIC_NETWORK, CATALOG_MEMBERSHIP_ONLY, HIDDEN_GROUPS_ONLY)
AREA_COUNTS = (5, 19, 38)
TOPOLOGIES = ("ring", "small_world", "complete")
# Orden del informe de la segunda ola. Son las doce categorías del catálogo macro.
MACRO_FAMILIES = (
    "inflation",
    "labor",
    "activity",
    "housing",
    "national_accounts",
    "rates",
    "liquidity",
    "credit",
    "fx",
    "commodities",
    "financial_conditions",
    "supply_chains",
)
# El balance reúne los conceptos contables en USD, CAD y CNY de la representación conjunta
# de la edición y los ratios son los factores de empresa.
BALANCE_CONCEPTS = FUNDAMENTAL_CONCEPTS + CAD_CONCEPTS + CNY_CONCEPTS
FUNDAMENTAL_FAMILIES = (("balance", BALANCE_CONCEPTS), ("ratios", FACTOR_CONCEPTS))
PRICE_AREAS = ("price_trend", "price_range_volatility", "price_volume_liquidity")
# Canales de la disposición con máscaras: valores, observado y logaritmo de la edad.
CHANNELS = (("values", (0,)), ("observation", (1, 2)))
VIEWS = ("view_a", "view_b")
CATALOG = Path(__file__).parents[4] / "data/catalogs/macro-indicators.csv"
FAMILY_MODALITIES = ("fundamentals", "macro")
# Las ranuras, el puente y la lectura salen de la misma semilla en las tres identidades.
_SEED_FAMILY = "holographic_network"


def _unit_interval(value, message, *, open_interval):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(message)
    if not (0 < value < 1 if open_interval else 0 <= value <= 1):
        raise ValueError(message)


@dataclass(frozen=True)
class HolographicNetworkConfig:
    kind: str = HOLOGRAPHIC_NETWORK
    areas: int = 19
    hidden_groups: int = 4
    rank: int | None = 4
    leakage: float | None = 0.1
    topology: str = "ring"
    cycles: int = 2
    local_steps: int = 1
    summary: bool = True
    local_gain: float = 0.25
    summary_gain: float = 0.25
    levels: int = 1

    def __post_init__(self):
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError(f"La red holográfica admite {KINDS}")
        if type(self.areas) is not int or self.areas not in AREA_COUNTS:
            raise ValueError(f"La red holográfica admite {AREA_COUNTS} áreas")
        if type(self.topology) is not str or self.topology not in TOPOLOGIES:
            raise ValueError(f"La topología de las áreas debe ser una de {TOPOLOGIES}")
        if self.kind == HIDDEN_GROUPS_ONLY:
            bounded_integer(self.hidden_groups, "ranuras ocultas", 1, 46)
            if self.rank is not None or self.leakage is not None or self.topology != "complete":
                raise ValueError(
                    "H3 no tiene áreas del catálogo: sin rango ni fuga y con topología completa"
                )
        else:
            bounded_integer(self.hidden_groups, "ranuras ocultas", 0, 8)
            bounded_integer(self.rank, "rango de los ajustes por área", 1, 16)
            _unit_interval(
                self.leakage, "La fuga del catálogo debe estar entre 0 y 1", open_interval=True
            )
        bounded_integer(self.cycles, "ciclos del puente", 1, 4)
        bounded_integer(self.local_steps, "pasos locales del puente", 1, 2)
        if type(self.summary) is not bool:
            raise ValueError("El resumen del puente necesita un booleano explícito")
        for name in ("local_gain", "summary_gain"):
            _unit_interval(
                getattr(self, name),
                "Las ganancias del puente están entre 0 y 1",
                open_interval=False,
            )
        if type(self.levels) is not int or self.levels != 1:
            raise ValueError(
                "Los niveles 2 y 3 necesitan los grupos de activos de H4 y el estado de "
                "régimen, que todavía no existen"
            )

    @property
    def catalog_areas(self):
        return 0 if self.kind == HIDDEN_GROUPS_ONLY else self.areas

    @property
    def nodes(self):
        return self.catalog_areas + self.hidden_groups

    def identity(self):
        membership = "none"
        if self.kind != HIDDEN_GROUPS_ONLY:
            membership = "softmax_over_areas_of_catalog_log_weights"
            if self.kind == HOLOGRAPHIC_NETWORK:
                membership += "_plus_learned_delta_from_zero"
        return dict(
            schema_version=1,
            **asdict(self),
            seed_family=_SEED_FAMILY,
            family_blocks="modality_encoder_applied_to_the_family_columns_only",
            membership=membership,
            area_weights="shared_block_projection_plus_low_rank_adapter_up_from_zero",
            area_embedding="learned_per_area_without_asset_identity",
            hidden_groups_rule="soft_moe_dispatch_over_blocks_and_combine_over_slots",
            small_world="ring_plus_chords_at_a_quarter_of_the_areas_from_8_areas",
            slot_edges="each_slot_with_every_node",
            bridge="masked_attention_steps_then_summary_per_cycle",
            absent_nodes="areas_of_absent_blocks_start_at_zero_and_send_no_messages",
            summary_pool="mean_over_present_nodes",
            readback="node_changes_through_membership_and_slots_plus_context_once",
            modality_readback="mean_over_the_blocks_of_the_modality_then_zero_projection",
            state="working_only_reset_per_prediction",
            rows="each_row_alone",
            parameter_updates="none_in_forward",
        )


def parse_holographic_network(raw):
    """Aceptar la configuración, sus valores completos o None."""
    if raw is None:
        return None
    if type(raw) is HolographicNetworkConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == set(HolographicNetworkConfig.__dataclass_fields__):
        return HolographicNetworkConfig(**raw)
    raise ValueError("holographic_network debe declarar toda su configuración o ser None")


@cache
def macro_categories(catalog=CATALOG):
    """Categoría de cada indicador del catálogo macro versionado."""
    with Path(catalog).open(encoding="utf-8", newline="") as stream:
        return {row["id"]: row["category"] for row in csv.DictReader(stream)}


@dataclass(frozen=True)
class AreaLayout:
    """Áreas, bloques y pertenencia del catálogo de una configuración.

    `blocks[i]` es (nombre, modalidad, columnas), con `columns` None si el bloque es la
    representación entera de la modalidad. `members[i]` son las áreas de la familia del
    bloque i y `widths` la anchura de la entrada de cada modalidad con familias.
    """

    areas: tuple[str, ...]
    blocks: tuple[tuple[str, str, tuple[int, ...] | None], ...]
    members: tuple[tuple[int, ...], ...]
    families: tuple[tuple[str, tuple[str, ...]], ...]
    widths: tuple[tuple[str, int], ...]

    def identity(self):
        return dict(
            areas=list(self.areas),
            blocks=[
                dict(name=name, modality=modality, columns=None if c is None else list(c))
                for name, modality, c in self.blocks
            ],
            members=[list(items) for items in self.members],
            families={name: list(items) for name, items in self.families},
            widths=dict(self.widths),
        )


def _columns(concepts, chosen, channels):
    """Columnas de los conceptos elegidos en los canales indicados de la disposición."""
    count = len(concepts)
    return tuple(
        channel * count + index
        for channel in channels
        for index, name in enumerate(concepts)
        if name in chosen
    )


def area_layout(areas, representation=None, *, masked=True, catalog=CATALOG):
    """Disposición de 5, 19 o 38 áreas para la representación de la edición."""
    if type(areas) is not int or areas not in AREA_COUNTS:
        raise ValueError(f"La red holográfica admite {AREA_COUNTS} áreas")
    if areas == 5:
        return AreaLayout(
            areas=MODALITIES,
            blocks=tuple((name, name, None) for name in MODALITIES),
            members=tuple((index,) for index in range(len(MODALITIES))),
            families=(),
            widths=(),
        )
    if not masked or not isinstance(representation, dict):
        raise ValueError("Las áreas por familia necesitan la representación con máscaras")
    fundamentals = tuple(representation.get("fundamental_concepts") or ())
    indicators = tuple(representation.get("macro_indicators") or ())
    categories = macro_categories(catalog)
    families = [
        (f"fundamentals_{name}", tuple(c for c in fundamentals if c in concepts))
        for name, concepts in FUNDAMENTAL_FAMILIES
    ]
    if sum(len(items) for _, items in families) != len(fundamentals):
        raise ValueError("Cada concepto contable debe ser del balance o un ratio de empresa")
    if any(categories.get(i) not in MACRO_FAMILIES for i in indicators):
        raise ValueError("Cada indicador macro necesita una de las doce categorías del catálogo")
    families += [
        (f"macro_{name}", tuple(i for i in indicators if categories[i] == name))
        for name in MACRO_FAMILIES
    ]
    if any(not items for _, items in families):
        raise ValueError("Cada familia de las áreas necesita al menos un concepto o indicador")
    names, blocks, members = [], [], []

    def add(block, modality, columns, area_names):
        blocks.append((block, modality, columns))
        members.append(tuple(range(len(names), len(names) + len(area_names))))
        names.extend(area_names)

    split = areas == 38
    add(
        "prices",
        "prices",
        None,
        [f"{area}_{view}" for area in PRICE_AREAS for view in VIEWS] if split else PRICE_AREAS,
    )
    for modality in ("news", "charts"):
        add(modality, modality, None, [f"{modality}_{v}" for v in VIEWS] if split else [modality])
    for family, items in families:
        modality = "fundamentals" if family.startswith("fundamentals") else "macro"
        concepts = fundamentals if modality == "fundamentals" else indicators
        if split:
            for channel, indices in CHANNELS:
                name = f"{family}_{channel}"
                add(name, modality, _columns(concepts, set(items), indices), [name])
        else:
            add(family, modality, _columns(concepts, set(items), (0, 1, 2)), [family])
    widths = (("fundamentals", 3 * len(fundamentals)), ("macro", 3 * len(indicators)))
    return AreaLayout(tuple(names), tuple(blocks), tuple(members), tuple(families), widths)


def area_edges(count, topology):
    """Aristas entre áreas en el orden de la disposición, con lazos propios."""
    index = torch.arange(count)
    distance = (index[:, None] - index[None, :]) % count
    near = torch.minimum(distance, count - distance)
    if topology == "complete":
        return torch.ones(count, count, dtype=torch.bool)
    edges = near <= 1
    if topology == "small_world" and count >= 8:
        edges = edges | (near == count // 4)
    return edges


def plastic_edges(config):
    """Pares (i, j) de áreas distintas que une la topología, en orden de filas.

    Son las aristas en las que P1 de #784 guarda pesos rápidos y trazas. H3 no tiene áreas,
    así que tampoco tiene aristas.
    """
    areas = config.catalog_areas
    edges = area_edges(areas, config.topology).tolist()
    return tuple((i, j) for i in range(areas) for j in range(areas) if i != j and edges[i][j])


def node_edges(config):
    """Máscara [nodos, nodos]: las áreas según su topología y cada ranura con todos."""
    areas, nodes = config.catalog_areas, config.nodes
    edges = torch.ones(nodes, nodes, dtype=torch.bool)
    if areas:
        edges[:areas, :areas] = area_edges(areas, config.topology)
    return edges


class HolographicNetwork(nn.Module):
    """Transformar [B,5,H] leyendo cada bloque desde varias áreas a la vez."""

    def __init__(
        self, hidden_size, config, representation=None, *, seed, max_batch=256, masked=True
    ):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de la red holográfica", 8, 512)
        bounded_integer(seed, "semilla de la red holográfica", 0, 2**32 - 1)
        bounded_integer(max_batch, "lote de la red holográfica", 1, MAX_BLOCK_ROWS)
        if type(config) is not HolographicNetworkConfig:
            raise ValueError("La red holográfica necesita HolographicNetworkConfig")
        config.__post_init__()
        if config.rank is not None and config.rank >= hidden_size:
            raise ValueError("El rango de los ajustes debe ser menor que la dimensión")
        self.config, self.hidden_size, self.max_batch = config, hidden_size, max_batch
        self.layout = area_layout(config.areas, representation, masked=masked)
        # Las aristas de P1 no son buffers: no entran en la huella de parámetros de H1.
        self.plastic_edges, self._plastic_index = plastic_edges(config), {}
        areas, blocks = config.catalog_areas, len(self.layout.blocks)
        derived = hashlib.sha256(f"{_SEED_FAMILY}/{seed}".encode()).digest()
        scale = 1 / math.sqrt(hidden_size)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            if config.hidden_groups:
                slots = config.hidden_groups
                self.slot_keys = nn.Parameter(torch.randn(slots, hidden_size) * scale)
                self.slot_embedding = nn.Parameter(torch.randn(slots, hidden_size) * scale)
                self.block_norm = nn.LayerNorm(hidden_size)
                self.slot_norm = nn.LayerNorm(hidden_size)
            self.local_norm = nn.LayerNorm(hidden_size)
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
            self.local_out = nn.Linear(hidden_size, hidden_size, bias=False)
            if config.summary:
                self.summary_norm = nn.LayerNorm(hidden_size)
                self.summary_update = nn.Sequential(
                    nn.Linear(hidden_size, 2 * hidden_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(2 * hidden_size, hidden_size, bias=False),
                )
                self.summary_to_local = nn.Linear(hidden_size, hidden_size, bias=False)
            self.readout = nn.Linear(hidden_size, hidden_size, bias=False)
            # Las áreas van al final: H3 no las crea y no desplaza los pesos anteriores.
            if areas:
                self.area_down = nn.Parameter(torch.randn(areas, hidden_size, config.rank) * scale)
                self.area_up = nn.Parameter(torch.zeros(areas, config.rank, hidden_size))
                self.area_embedding = nn.Parameter(torch.randn(areas, hidden_size) * scale)
                self.area_norm = nn.LayerNorm(hidden_size)
                if config.kind == HOLOGRAPHIC_NETWORK:
                    self.membership_delta = nn.Parameter(torch.zeros(areas, blocks))
        nn.init.zeros_(self.readout.weight)
        # La estructura deriva de la configuración y de la disposición, que fija la identidad,
        # así que sus buffers no viajan en el checkpoint.
        self.register_buffer("edges", node_edges(config), persistent=False)
        modality = torch.zeros(blocks, len(MODALITIES))
        for index, (_, name, _) in enumerate(self.layout.blocks):
            modality[index, MODALITIES.index(name)] = 1
        self.register_buffer("block_modality", modality, persistent=False)
        if areas:
            owner = [b for b, members in enumerate(self.layout.members) for _ in members]
            self.register_buffer("area_owner", torch.tensor(owner), persistent=False)
            # Pesos del catálogo en FP64, para que el tipo de trabajo los redondee una sola vez.
            leakage, logits = config.leakage, torch.empty(areas, blocks, dtype=torch.float64)
            for index, members in enumerate(self.layout.members):
                logits[:, index] = math.log(leakage / (areas - len(members)))
                logits[list(members), index] = math.log((1 - leakage) / len(members))
            self.register_buffer("log_catalog", logits, persistent=False)
        widths = dict(self.layout.widths)
        for name in FAMILY_MODALITIES:
            chosen = [c for _, m, c in self.layout.blocks if m == name and c is not None]
            if chosen:
                mask = torch.zeros(len(chosen), widths[name])
                for row, columns in enumerate(chosen):
                    mask[row, list(columns)] = 1
                self.register_buffer(f"{name}_columns", mask, persistent=False)

    @property
    def family_blocks(self):
        return any(columns is not None for _, _, columns in self.layout.blocks)

    def get_extra_state(self):
        return dict(
            configuration=self.config.identity(),
            hidden_size=self.hidden_size,
            max_batch=self.max_batch,
            layout=self.layout.identity(),
        )

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def operation_counts(self, rows):
        """Evaluaciones de nodos y resúmenes del puente, incluidas las filas enmascaradas."""
        bounded_integer(rows, "filas del trabajo holográfico", 0, MAX_BLOCK_ROWS)
        config = self.config
        return dict(
            holographic_node_evaluations=rows * config.nodes * config.cycles * config.local_steps,
            holographic_summary_evaluations=rows * config.cycles if config.summary else 0,
        )

    def _checked(self, blocks, presence, inputs, encoders):
        reference = self.qkv.weight
        if (
            not isinstance(blocks, torch.Tensor)
            or blocks.layout != torch.strided
            or blocks.ndim != 3
            or blocks.shape[1:] != (len(MODALITIES), self.hidden_size)
            or not 1 <= len(blocks) <= self.max_batch
            or blocks.dtype != reference.dtype
            or blocks.device != reference.device
        ):
            raise ValueError("Los bloques de la red holográfica no corresponden al contrato")
        if presence is None:
            presence = torch.ones(blocks.shape[:2], dtype=torch.bool, device=blocks.device)
        elif (
            not isinstance(presence, torch.Tensor)
            or presence.dtype != torch.bool
            or presence.shape != blocks.shape[:2]
            or presence.device != blocks.device
        ):
            raise ValueError("La presencia de la red holográfica no corresponde a sus bloques")
        if self.family_blocks:
            widths = dict(self.layout.widths)
            for name in FAMILY_MODALITIES:
                value = inputs.get(name) if isinstance(inputs, Mapping) else None
                layers = encoders[name] if encoders is not None and name in encoders else None
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (len(blocks), widths[name])
                    or value.dtype != reference.dtype
                    or value.device != reference.device
                    or not isinstance(layers, nn.Sequential)
                    or len(layers) != 2
                    or type(layers[0]) is not nn.Linear
                    or type(layers[1]) is not nn.SiLU
                    or layers[0].in_features != widths[name]
                    or layers[0].out_features != self.hidden_size
                ):
                    raise ValueError(
                        "Los bloques por familia necesitan la entrada y el codificador lineal "
                        "de su modalidad"
                    )
        return presence

    def _plastic(self, device):
        """Índices de destino y origen de las aristas plásticas en el dispositivo."""
        index = self._plastic_index.get(device)
        if index is None:
            pairs = torch.tensor(self.plastic_edges, dtype=torch.int64)
            index = self._plastic_index[device] = (pairs[:, 0].to(device), pairs[:, 1].to(device))
        return index

    def _checked_bias(self, blocks, edge_bias, traces):
        if edge_bias is None and traces is None:
            return
        if (
            not self.plastic_edges
            or not isinstance(edge_bias, torch.Tensor)
            or edge_bias.layout != torch.strided
            or edge_bias.shape != (len(blocks), len(self.plastic_edges))
            or edge_bias.dtype != blocks.dtype
            or edge_bias.device != blocks.device
            or edge_bias.requires_grad
            or type(traces) is not list
            or traces
        ):
            raise ValueError(
                "Los pesos rápidos del puente necesitan [filas, aristas] sin grafo y una lista "
                "vacía para sus trazas"
            )

    def _source_blocks(self, blocks, presence, inputs, encoders):
        """Bloques [B, bloques, H] de la disposición, a cero en las modalidades ausentes."""
        if not self.family_blocks:
            return blocks
        parts = [blocks[:, : MODALITIES.index("fundamentals")]]
        for name in FAMILY_MODALITIES:
            linear = encoders[name][0]
            mask = getattr(self, f"{name}_columns")
            # W_f = W ⊙ máscara de la familia, en una sola multiplicación para todas ellas.
            weights = mask.unsqueeze(-1) * linear.weight.t().unsqueeze(0)
            projected = torch.einsum("rc,fch->rfh", inputs[name], weights) + linear.bias
            present = presence[:, MODALITIES.index(name)].to(blocks.dtype)
            parts.append(F.silu(projected) * present[:, None, None])
        return torch.cat(parts, dim=1)

    def forward(self, blocks, presence, inputs=None, encoders=None, edge_bias=None, traces=None):
        presence = self._checked(blocks, presence, inputs, encoders)
        self._checked_bias(blocks, edge_bias, traces)
        if torch.is_autocast_enabled(blocks.device.type):
            raise ValueError("La red holográfica exige FP32 o FP64 explícitos, sin autocast")
        config, areas = self.config, self.config.catalog_areas
        with deferred_checks():
            check_finite(blocks, "Los bloques de la red holográfica")
            sources = self._source_blocks(blocks, presence, inputs, encoders)
            check_finite(sources, "Los bloques por familia")
            present = presence.to(blocks.dtype) @ self.block_modality.t() > 0
            active = present.any(1)
            nodes, alive, membership, combine = [], [], None, None
            if areas:
                logits = self.log_catalog.to(blocks.dtype)
                if config.kind == HOLOGRAPHIC_NETWORK:
                    logits = logits + self.membership_delta
                membership = torch.softmax(logits, dim=0)
                mixed = torch.einsum("ab,rbh->rah", membership, sources)
                adapted = torch.einsum("rah,ahk->rak", mixed, self.area_down)
                adapted = mixed + torch.einsum("rak,akh->rah", adapted, self.area_up)
                nodes.append(self.area_norm(adapted + self.area_embedding))
                alive.append(present[:, self.area_owner])
            if config.hidden_groups:
                keys = self.block_norm(sources)
                scores = torch.einsum("sh,rbh->rsb", self.slot_keys, keys)
                scores = scores / math.sqrt(self.hidden_size)
                # Una fila sin bloques presentes reparte entre todos y se descarta al final.
                allowed = present | ~active.unsqueeze(-1)
                dispatch = torch.softmax(scores.masked_fill(~allowed.unsqueeze(1), -math.inf), -1)
                # La combinación es entre ranuras y no necesita la máscara de bloques.
                combine = torch.softmax(scores.transpose(1, 2), dim=-1)
                gathered = torch.einsum("rsb,rbh->rsh", dispatch, sources)
                nodes.append(self.slot_norm(gathered + self.slot_embedding))
                alive.append(active.unsqueeze(-1).expand(-1, config.hidden_groups))
            alive = torch.cat(alive, dim=1)
            mask = alive.unsqueeze(-1)
            initial = torch.where(mask, torch.cat(nodes, dim=1), 0)
            # Los nodos ausentes no envían mensajes. Su consulta conserva el lazo propio
            # para que ninguna fila de la máscara quede vacía y su estado se anula después.
            itself = torch.eye(config.nodes, dtype=torch.bool, device=blocks.device)
            edges = (self.edges & alive.unsqueeze(1)) | (itself & ~alive.unsqueeze(-1))
            edges = edges.unsqueeze(1)
            attention, trace = edges, None
            if edge_bias is not None:
                check_finite(edge_bias, "Los pesos rápidos del puente")
                target, source = self._plastic(blocks.device)
                bias = blocks.new_zeros(len(blocks), config.nodes, config.nodes)
                bias[:, target, source] = edge_bias
                # Misma atención que la máscara booleana con pesos nulos.
                attention = torch.where(edges, bias.unsqueeze(1), -math.inf)
                trace = blocks.new_zeros(len(blocks), len(self.plastic_edges))
            count = alive.sum(1, keepdim=True).clamp_min(1).to(blocks.dtype)
            local = initial
            summary = blocks.new_zeros(len(blocks), self.hidden_size)
            context = blocks.new_zeros(len(blocks), 1, self.hidden_size)
            for _ in range(config.cycles):
                for _ in range(config.local_steps):
                    normalized = self.local_norm(local + context)
                    q, k, v = self.qkv(normalized).chunk(3, dim=-1)
                    messages = F.scaled_dot_product_attention(
                        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=attention
                    ).squeeze(1)
                    if trace is not None:
                        # Probabilidades del paso, con la escala 1/√H de SDPA y sin grafo.
                        with torch.no_grad():
                            logits = q @ k.transpose(-1, -2)
                            logits = logits / math.sqrt(self.hidden_size) + attention.squeeze(1)
                            trace += torch.softmax(logits, dim=-1)[:, target, source]
                    local = torch.where(
                        mask, local + config.local_gain * self.local_out(messages), 0
                    )
                if config.summary:
                    pooled = local.sum(1) / count
                    update = self.summary_update(self.summary_norm(summary) + pooled)
                    summary = torch.where(
                        active.unsqueeze(-1), summary + config.summary_gain * update, summary
                    )
                    context = self.summary_to_local(summary).unsqueeze(1)
            if trace is not None:
                # La consulta de un área ausente no cuenta: su estado se anula en cada paso.
                trace = torch.where(alive[:, target], trace, 0)
                traces.append(trace / (config.cycles * config.local_steps))
            change = local - initial
            back = context.expand(-1, len(self.layout.blocks), -1)
            if areas:
                back = back + torch.einsum("ab,rah->rbh", membership, change[:, :areas])
            if config.hidden_groups:
                back = back + torch.einsum("rbs,rsh->rbh", combine, change[:, areas:])
            # La media de cada modalidad divide por su número exacto de bloques.
            gathered = torch.einsum("bm,rbh->rmh", self.block_modality, back)
            projected = self.readout(gathered / self.block_modality.sum(0).unsqueeze(-1))
            result = torch.where(
                presence.unsqueeze(-1) & active.view(-1, 1, 1), blocks + projected, blocks
            )
            # Un NaN o un infinito de un nodo presente llega a esta salida, así que no se
            # comprueba cada paso. En CPU, las ocho comprobaciones de dos ciclos eran el
            # 15 % del forward.
            check_finite(result, "La salida de la red holográfica")
        return result
