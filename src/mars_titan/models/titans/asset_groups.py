"""Grupos de activos de la sesión `asset_group_membership_v1` (H4 de #783).

Tras MAC, cada flujo i tiene la representación e_i de su emisión. H4 le suma lo que aportan los
grupos de activos de su sesión a los que pertenece:

    n_i = LayerNorm(e_i)
    p_g = Σ_i M_ig n_i / Σ_i M_ig
    s_g = W_2 SiLU(W_1 p_g + b_1) + b_2
    e'_i = e_i + W_o (Σ_g M_ig s_g) / F_i

M_ig es la pertenencia fraccionaria de la fila i al grupo g y F_i el número de familias en que
la fila tiene alguna, así que la lectura es la media de sus familias y su escala no depende de
cuántas tenga. Los grupos son de las filas del bloque con el mismo mercado y el mismo instante,
que son la sesión entera (`session_level`). Las pertenencias salen de la ventana de precios de
cada fila, que termina en la sesión de la decisión, así que no leen nada posterior. W_o empieza
en cero y el predictor emite al construirse lo mismo que sin H4.

Parte de la pertenencia fraccionaria de MaGNet (https://arxiv.org/abs/2511.00085) y de las
ranuras de Soft MoE (Puigcerver y otros, ICLR 2024), con una diferencia: aquí los grupos no se
aprenden, salen de medidas de la ventana pasada. `mars_titan.data.asset_groups` describe las
cinco familias.

Modos con los mismos parámetros y la misma inicialización:

- `past_window`: H4, con las cinco familias de la ventana pasada.
- `shuffled_rows`: control con las mismas pertenencias permutadas entre las filas de la sesión.
  La permutación sale de la semilla del predictor, del mercado y del instante. Conserva los
  tamaños de los grupos y el cálculo, pero rompe la relación entre cada fila y sus medidas.
- `current_sector`: control de fuga con una sola familia, el sector actual de cada activo según
  las fichas de FinMultiTime (`SectorMap`). No es una serie histórica, así que puede adelantar
  reclasificaciones posteriores a la decisión.
"""

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.asset_groups import FAMILIES, family_slices, session_memberships
from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_WINDOW_CHANNELS
from mars_titan.data.sector_map import SectorMap

from .config import bounded_integer, require_identity
from .state import check_finite

ASSET_GROUP_MEMBERSHIP = "asset_group_membership_v1"
PAST_WINDOW, SHUFFLED_ROWS, CURRENT_SECTOR = "past_window", "shuffled_rows", "current_sector"
MODES = (PAST_WINDOW, SHUFFLED_ROWS, CURRENT_SECTOR)
_WINDOW_FIELDS = ("buckets", "clusters", "min_observations", "iterations")


@dataclass(frozen=True)
class AssetGroupsConfig:
    mode: str = PAST_WINDOW
    buckets: int | None = 5
    clusters: int | None = 8
    min_observations: int | None = 36
    iterations: int | None = 32

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError(f"El modo de H4 debe ser uno de {MODES}")
        if self.mode == CURRENT_SECTOR:
            if any(getattr(self, name) is not None for name in _WINDOW_FIELDS):
                raise ValueError(
                    "El control sectorial de H4 no tiene cubos, grupos de correlación, mínimo "
                    "de observaciones ni iteraciones"
                )
            return
        bounded_integer(self.buckets, "cubos de H4", 2, 10)
        bounded_integer(self.clusters, "grupos de correlación de H4", 2, 16)
        bounded_integer(self.min_observations, "observaciones mínimas de H4", 2, 63)
        bounded_integer(self.iterations, "iteraciones de las c-medias difusas de H4", 1, 128)

    def identity(self):
        result = dict(
            schema_version=1,
            kind=ASSET_GROUP_MEMBERSHIP,
            mode=self.mode,
            groups="same_market_and_prediction_instant_in_the_block",
            position="after_mac_before_head",
            normalization="layer_norm_before_pooling",
            pooling="membership_weighted_mean_of_the_session_rows",
            group_state="shared_two_layer_silu_mlp",
            readback="membership_weighted_sum_divided_by_the_families_of_the_row",
            output="residual_with_zero_initialized_projection",
            state="none",
            parameter_updates="none_in_forward",
        )
        if self.mode == CURRENT_SECTOR:
            result.update(
                families=["current_sector"],
                membership="one_hot_of_the_current_finmultitime_sector",
                point_in_time=False,
            )
            return result
        result.update(
            {name: getattr(self, name) for name in _WINDOW_FIELDS},
            families=list(FAMILIES),
            window=f"{CONTEXT_SESSIONS}_sessions_ending_at_the_decision",
            measures=dict(
                volatility="sample_std_ddof_1_of_one_session_log_returns",
                momentum="log_close_of_last_minus_first_present_session",
                liquidity_spread="abdi_ranaldo_2017_chl_two_day_corrected",
                beta="ols_slope_on_equal_weight_mean_of_valid_rows",
                correlation=(
                    "fuzzy_c_means_m2_on_row_normalized_top_principal_components_of_the_"
                    "market_residuals"
                ),
            ),
            bucket_membership="ruspini_partition_of_session_midranks",
            initialization="farthest_first_from_the_mean",
            point_in_time=True,
        )
        if self.mode == SHUFFLED_ROWS:
            result.update(shuffle="rows_permuted_within_the_session_by_seed_market_and_instant")
        return result


def parse_asset_groups(raw):
    """Aceptar la configuración, sus valores completos o None."""
    if raw is None:
        return None
    if type(raw) is AssetGroupsConfig:
        raw.__post_init__()
        return raw
    if isinstance(raw, dict) and set(raw) == set(AssetGroupsConfig.__dataclass_fields__):
        return AssetGroupsConfig(**raw)
    raise ValueError("asset_groups debe declarar toda su configuración o ser None")


class AssetGroupMembership(nn.Module):
    """Suma residual de los grupos de activos de cada sesión a la representación de sus filas."""

    session_level = True

    def __init__(self, hidden_size, config, *, seed, sector_map=None, dtype=torch.float32):
        super().__init__()
        bounded_integer(hidden_size, "dimensión de H4", 1, 4096)
        bounded_integer(seed, "semilla de H4", 0, 2**32 - 1)
        if type(config) is not AssetGroupsConfig:
            raise ValueError("H4 necesita AssetGroupsConfig")
        config.__post_init__()
        if (config.mode == CURRENT_SECTOR) != (sector_map is not None) or (
            sector_map is not None and type(sector_map) is not SectorMap
        ):
            raise ValueError("Solo el control sectorial de H4 lee el reparto de sectores")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("H4 necesita precisión FP32 o FP64")
        self.config, self.hidden_size, self.seed = config, hidden_size, seed
        self.sector_map = sector_map
        # Los pesos salen de una semilla derivada de la del predictor, sin consumir el
        # generador global. H4 y sus dos controles empiezan con los mismos pesos.
        derived = hashlib.sha256(f"{ASSET_GROUP_MEMBERSHIP}/{seed}".encode()).digest()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int.from_bytes(derived[:7], "little"))
            self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
            self.inner = nn.Linear(hidden_size, hidden_size, dtype=dtype)
            self.outer = nn.Linear(hidden_size, hidden_size, dtype=dtype)
            self.out = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.out.weight)

    def get_extra_state(self):
        return dict(configuration=self.config.identity(), hidden_size=self.hidden_size)

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def sessions(self, markets, prediction_at):
        """Clave y filas de cada sesión del bloque, en el orden de primera aparición."""
        sessions = {}
        for row, key in enumerate(zip(markets, prediction_at, strict=True)):
            sessions.setdefault(key, []).append(row)
        return list(sessions.items())

    def _permutation(self, key, rows):
        market, moment = key
        text = f"{ASSET_GROUP_MEMBERSHIP}/{self.seed}/{market}/{moment}"
        digest = hashlib.sha256(text.encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:7], "little"))
        return torch.randperm(rows, generator=generator)

    def memberships(self, prices, flows, key):
        """Pertenencias de las filas de una sesión y el número de familias de cada fila."""
        if self.config.mode == CURRENT_SECTOR:
            groups = torch.tensor(
                [self.sector_map.group(flow) for flow in flows],
                dtype=torch.int64,
                device=prices.device,
            )
            membership = F.one_hot(groups, len(self.sector_map.groups)).to(prices.dtype)
            return membership, torch.ones(len(flows), dtype=prices.dtype, device=prices.device)
        config = self.config
        membership = session_memberships(
            prices,
            buckets=config.buckets,
            clusters=config.clusters,
            min_observations=config.min_observations,
            iterations=config.iterations,
        )
        if config.mode == SHUFFLED_ROWS:
            order = self._permutation(key, len(flows)).to(prices.device)
            membership = membership.index_select(0, order)
        families = sum(
            (membership[:, part].sum(1) > 0).to(prices.dtype)
            for part in family_slices(config.buckets, config.clusters)
        )
        return membership, families

    def forward(self, encoded, prices, flow_ids, markets, prediction_at):
        reference = self.out.weight
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.ndim != 2
            or encoded.shape[1] != self.hidden_size
            or len(encoded) == 0
            or encoded.dtype != reference.dtype
            or encoded.device != reference.device
            or not isinstance(prices, torch.Tensor)
            or prices.shape != (len(encoded), CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS))
            or prices.dtype != reference.dtype
            or prices.device != reference.device
            or len(flow_ids) != len(encoded)
            or len(markets) != len(encoded)
            or len(prediction_at) != len(encoded)
        ):
            raise ValueError(
                "H4 necesita una representación, una ventana con presencia, un flujo, un "
                "mercado y un instante por fila"
            )
        if torch.is_autocast_enabled(encoded.device.type):
            raise ValueError("H4 exige FP32 o FP64 explícitos, sin autocast")
        normalized = self.norm(encoded)
        mixed = torch.zeros_like(encoded)
        for key, members in self.sessions(markets, prediction_at):
            index = torch.tensor(members, dtype=torch.int64, device=encoded.device)
            membership, families = self.memberships(
                prices.index_select(0, index), [flow_ids[row] for row in members], key
            )
            total = membership.sum(0)
            # Un grupo sin miembros queda en cero y ninguna fila lo lee.
            pooled = membership.T @ normalized.index_select(0, index)
            pooled = pooled / torch.where(total > 0, total, 1.0)[:, None]
            state = self.outer(F.silu(self.inner(pooled)))
            part = (membership @ state) / families.clamp_min(1)[:, None]
            mixed = mixed.index_copy(0, index, part)
        result = encoded + self.out(mixed)
        check_finite(result, "La representación con los grupos de activos")
        return result
