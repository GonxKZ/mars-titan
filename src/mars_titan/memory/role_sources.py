"""Fuentes de rol de B3 calculadas para cada fila: mercado, modalidades y sector estadístico.

Amplían las reglas de `regimes.py` para `role_bound_keys_v1` (#750). Cada fila de un evento
recibe una ruta entre 0 y 4 con datos disponibles en la decisión y sin parámetros ajustados:

- `market_of_the_flow_v1`: ruta 1 para EE. UU. y 2 para China, según el prefijo del flujo.
  Con un solo mercado todas las filas comparten ruta y el rol equivale al rol constante con
  otro código.
- `news_and_fundamentals_presence_v1`: ruta 1 + noticias + 2·fundamentales, con los bits de
  presencia de la fila. Precios y gráficos son obligatorios y el macro está presente en todas
  las filas de la edición (`data/modality_ablation.py`), así que no distinguen filas. Con la
  política estricta todas las filas van a la ruta 4.
- `past_window_return_clusters_v1`: sector estadístico en fecha. Los activos de cada mercado
  se agrupan por la correlación de sus rendimientos diarios en la ventana recibida, con la
  regla común de C3 y C4 (`data/correlation_regions.py`) cortada en cuatro grupos. Los activos
  con menos de `min_returns` rendimientos o con la serie constante van a la ruta 0, sin
  clasificar. Los grupos se numeran por el menor activo que contienen. Una partición estable
  conserva sus rutas, pero un cambio de miembros puede renumerarlas, y el informe del recorrido
  cuenta esos cambios por activo.
- `permuted_past_window_return_clusters_v1`, control del sector: las mismas rutas y tamaños,
  repartidos al azar entre los activos con historia de cada mercado. El generador solo depende
  de la semilla y del mercado, así que con los mismos activos cada uno recibe siempre la ruta
  del mismo compañero. Conserva la ocupación de cada evento y retira la correlación.

El sector de las fichas actuales (`data/sector_map.py`) no se usa. Declara
`point_in_time=False` y tiene más sectores que los cuatro compartimentos clasificados de B3.
C4 y H4 ya lo miden como control de fuga.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from mars_titan.data.correlation_regions import (
    correlation_regions,
    permuted_regions,
    window_returns,
)
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRICE_FEATURES, PRICE_WINDOW_CHANNELS

from .regimes import MARKETS, SLOTS

MARKET_RULE = "market_of_the_flow_v1"
MODALITY_RULE = "news_and_fundamentals_presence_v1"
SECTOR_RULE = "past_window_return_clusters_v1"
PERMUTED_SECTOR_RULE = "permuted_past_window_return_clusters_v1"
ROW_RULES = (MARKET_RULE, MODALITY_RULE, SECTOR_RULE, PERMUTED_SECTOR_RULE)
SECTOR_RULES = (SECTOR_RULE, PERMUTED_SECTOR_RULE)
_SECTOR_LABELS = ("unclassified", "sector_1", "sector_2", "sector_3", "sector_4")
LABELS = {
    MARKET_RULE: ("unclassified", "US", "CN", "unused_3", "unused_4"),
    MODALITY_RULE: (
        "unclassified",
        "without_news_or_fundamentals",
        "news_only",
        "fundamentals_only",
        "news_and_fundamentals",
    ),
    SECTOR_RULE: _SECTOR_LABELS,
    PERMUTED_SECTOR_RULE: _SECTOR_LABELS,
}
_NEWS, _FUNDAMENTALS = MODALITIES.index("news"), MODALITIES.index("fundamentals")


@dataclass(frozen=True)
class MarketOccupancy:
    """Filas de un mercado en cada ruta de un evento, cuando cada fila tiene su propia ruta."""

    market: str
    occupancy: tuple[int, ...]

    def record(self, at, labels):
        return dict(
            at=at,
            market=self.market,
            assets=sum(self.occupancy),
            occupancy=dict(zip(labels, self.occupancy, strict=True)),
        )


@dataclass(frozen=True)
class RowRoleRule:
    """Fuente de rol por fila declarada antes de ejecutar.

    `min_returns` y `seed` solo intervienen en el sector y su control. Por defecto el sector
    pide 42 rendimientos, el mismo mínimo con el que el régimen clasifica una cohorte.
    """

    name: str
    min_returns: int = 42
    seed: int = 0
    per_row = True

    def __post_init__(self):
        if self.name not in ROW_RULES:
            raise ValueError("La fuente de rol no está declarada: " + ", ".join(ROW_RULES))
        if type(self.min_returns) is not int or not 2 <= self.min_returns < CONTEXT_SESSIONS:
            raise ValueError(
                f"El mínimo de rendimientos debe estar entre 2 y {CONTEXT_SESSIONS - 1}"
            )
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("La semilla del control de sector debe ser un entero uint32")

    @property
    def labels(self):
        return LABELS[self.name]

    def identity(self):
        identity = dict(
            schema_version=1,
            kind="row_role_source",
            rule=self.name,
            slots=SLOTS,
            labels=list(self.labels),
            grouping="own_route_for_each_row_of_the_event",
            fit="none_declared_before_execution",
        )
        if self.name == MARKET_RULE:
            return dict(identity, assignment="1_for_us_flows_2_for_cn_flows")
        if self.name == MODALITY_RULE:
            return dict(
                identity,
                assignment="1_plus_news_present_plus_2_times_fundamentals_present",
                ignored="prices_and_charts_required_macro_present_in_every_row",
            )
        identity.update(
            window="calendar_price_window_v1_of_64_sessions_ending_at_the_decision",
            series="daily_log_close_returns_between_consecutive_present_sessions",
            regions="correlation_regions_average_linkage_on_one_minus_rho_cut_in_4_per_market",
            min_returns=self.min_returns,
            unclassified="route_0_with_fewer_returns_than_declared_or_a_constant_series",
            numbering="from_1_by_the_smallest_asset_of_each_group",
        )
        if self.name == PERMUTED_SECTOR_RULE:
            identity.update(
                control="same_routes_and_sizes_permuted_among_assets_with_history",
                generator="numpy_seed_sequence_of_sha256_of_rule_seed_and_market",
                seed=self.seed,
            )
        return identity

    def routes(self, prices, flow_ids, at, presence=None):
        """Ruta de cada fila de un evento completo y la ocupación de cada mercado.

        `prices` son las ventanas del evento en el orden de `flow_ids`, `presence` sus cinco
        bits de modalidad y `at` el corte común en microsegundos UTC. Solo el sector lee los
        precios y solo las modalidades leen la presencia. Cada mercado se agrupa con todas sus
        filas, así que el resultado no depende de cómo se reparta el evento en bloques.
        """
        flow_ids = list(flow_ids)
        if type(at) is not int or not flow_ids:
            raise ValueError("La fuente de rol necesita los flujos y el corte entero del evento")
        markets = np.array([flow.split("/", 1)[0] for flow in flow_ids])
        if not set(markets.tolist()) <= set(MARKETS):
            raise ValueError("La fuente de rol solo admite los mercados US y CN")
        if self.name == MARKET_RULE:
            routes = 1 + (markets == "CN").astype(np.int64)
        elif self.name == MODALITY_RULE:
            presence = np.asarray(presence)
            if presence.dtype != np.bool_ or presence.shape != (len(flow_ids), len(MODALITIES)):
                raise ValueError("Las modalidades necesitan cinco bits de presencia por fila")
            news, fundamentals = presence[:, _NEWS], presence[:, _FUNDAMENTALS]
            routes = 1 + news.astype(np.int64) + 2 * fundamentals.astype(np.int64)
        else:
            routes = self._sectors(prices, flow_ids, markets)
        states = [
            MarketOccupancy(
                market, tuple(np.bincount(routes[markets == market], minlength=SLOTS).tolist())
            )
            for market in sorted(set(markets.tolist()))
        ]
        return routes, states

    def _sectors(self, prices, flow_ids, markets):
        prices = np.asarray(prices)
        if (
            prices.ndim != 3
            or prices.shape[:2] != (len(flow_ids), CONTEXT_SESSIONS)
            or prices.shape[2] not in (len(PRICE_FEATURES), len(PRICE_WINDOW_CHANNELS))
        ):
            raise ValueError("El sector necesita la ventana de 64 sesiones de cada fila")
        returns, valid = window_returns(prices)
        routes = np.empty(len(flow_ids), dtype=np.int64)
        for market in sorted(set(markets.tolist())):
            ordered = sorted(np.flatnonzero(markets == market).tolist(), key=flow_ids.__getitem__)
            regions = correlation_regions(
                [flow_ids[row] for row in ordered],
                returns[ordered],
                valid[ordered],
                clusters=SLOTS - 1,
                min_observations=self.min_returns,
            )
            if self.name == PERMUTED_SECTOR_RULE:
                text = f"{PERMUTED_SECTOR_RULE}/{self.seed}/{market}"
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:16], "little")
                regions = permuted_regions(regions, seed)
            routes[ordered] = [int(regions[flow_ids[row]]) for row in ordered]
        return routes
