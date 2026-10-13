"""Régimen de mercado observable con el que se enruta la corrección asociativa B6.

El régimen se calcula una vez por evento y mercado, solo con las ventanas de precios que la
cohorte ya recibe como entrada. No hay parámetros ajustados ni umbrales estimados con datos:
las longitudes y los mínimos se fijaron antes de ejecutar y forman parte de la identidad.

1. El mercado se resume en cada paso con la mediana transversal de los log-rendimientos de
   cierre entre sesiones presentes. Con el contrato de ventanas por sesión del calendario
   (edición v3.1) todos los activos de un mercado comparten las sesiones presentes, porque
   solo puede faltar una sesión sin filas en todo el mercado. El rendimiento que atraviesa
   esa sesión une los dos cierres que la rodean, igual que la anomalía de M3.
2. La volatilidad es turbulenta si la raíz de la media de los cuadrados de los 21 últimos
   rendimientos de mercado supera la de los anteriores. Mide si la volatilidad crece dentro
   de la ventana, no su nivel frente a la historia, que exigiría estimar una escala.
3. La tendencia es alcista si la mediana transversal del log-rendimiento de cada activo en
   la ventana (cierre de la decisión frente al primer cierre presente, unas 63 sesiones) es
   positiva. No se suman las medianas diarias: con rendimientos asimétricos la mediana de
   cada día queda por debajo de la media y la suma acumula ese sesgo. Con los precios reales
   de China en 2019 la suma de las medianas diarias fue -0,156 mientras la mediana del
   rendimiento anual por activo era +0,203, y casi todas las sesiones salían bajistas.

Las dos señales dan cuatro regímenes. Una cohorte con menos activos o rendimientos de los
declarados va a la ruta 0, sin clasificar, que tiene su propio compartimento y queda
registrada. La mediana no depende del orden de los activos ni del bloque en que lleguen, y
como solo se usan ventanas que terminan en el corte, cambiar sesiones posteriores no cambia
ninguna ruta anterior.

`calendar_month_v1` es el control de descarte. Usa los mismos cinco compartimentos y manda a
la ruta 0 exactamente las mismas cohortes, pero asigna las demás por el mes UTC de la
decisión módulo cuatro. Conserva la capacidad y cambia de compartimento cada mes, sin
información de mercado. Si el régimen no mejora a este control, la ganancia se explica por
la capacidad o por el reparto en compartimentos y no por el estado del mercado. La
persistencia de los dos repartos se registra en cada recorrido y no se supone igual.

No es un modelo de estados latentes ni el HMM filtrado de `native/src/markov_filter.cpp`, que
sigue sin conexión. Los nombres de los regímenes son descriptivos y no identifican crisis ni
causas económicas.
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from mars_titan.data.price_windows import CONTEXT_SESSIONS, PRESENCE_CHANNEL, PRICE_WINDOW_CHANNELS

REGIME_RULE = "observable_volatility_trend_v1"
CALENDAR_RULE = "calendar_month_v1"
RULES = (REGIME_RULE, CALENDAR_RULE)
UNCLASSIFIED = 0
LABELS = {
    REGIME_RULE: ("unclassified", "calm_up", "calm_down", "turbulent_up", "turbulent_down"),
    CALENDAR_RULE: ("unclassified", "month_0", "month_1", "month_2", "month_3"),
}
SLOTS = 5
MARKETS = ("US", "CN")
_CLOSE = PRICE_WINDOW_CHANNELS.index("log_close_to_anchor_close")
_PRESENT = PRICE_WINDOW_CHANNELS.index(PRESENCE_CHANNEL)
_MAX_STEPS = CONTEXT_SESSIONS - 1


def _count(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} debe ser un entero entre {minimum} y {maximum}")


@dataclass(frozen=True)
class MarketState:
    """Resumen de un mercado en un evento y la ruta que le corresponde."""

    market: str
    route: int
    assets: int
    returns: int
    recent_rms: float | None
    earlier_rms: float | None
    trend: float | None

    def record(self, at, labels):
        return dict(
            at=at,
            market=self.market,
            route=self.route,
            label=labels[self.route],
            assets=self.assets,
            returns=self.returns,
            recent_rms=self.recent_rms,
            earlier_rms=self.earlier_rms,
            trend=self.trend,
        )


@dataclass(frozen=True)
class RegimeRule:
    """Regla de enrutamiento declarada antes de ejecutar.

    `min_assets` y `min_returns` deciden cuándo una cohorte queda sin clasificar y
    `recent_returns` separa la parte reciente de la ventana. Los valores por defecto son los
    de la campaña. Las pruebas usan cohortes pequeñas y declaran otros mínimos, que cambian la
    identidad.
    """

    name: str
    min_assets: int = 20
    recent_returns: int = 21
    min_returns: int = 42

    def __post_init__(self):
        if self.name not in RULES:
            raise ValueError("La regla de régimen no está declarada: " + ", ".join(RULES))
        _count(self.min_assets, "El mínimo de activos", 1, 1_000_000)
        _count(self.recent_returns, "Los rendimientos recientes", 1, _MAX_STEPS - 1)
        _count(self.min_returns, "El mínimo de rendimientos", self.recent_returns + 1, _MAX_STEPS)

    @property
    def labels(self):
        return LABELS[self.name]

    def identity(self):
        common = dict(
            schema_version=1,
            kind="observable_market_regime",
            rule=self.name,
            slots=SLOTS,
            labels=list(self.labels),
            grouping="market_of_each_flow_once_per_event",
            window="calendar_price_window_v1_of_64_sessions_ending_at_the_decision",
            min_assets=self.min_assets,
            min_returns=self.min_returns,
            unclassified="route_0_when_fewer_assets_or_returns_than_declared",
            fit="none_declared_before_execution",
        )
        if self.name == CALENDAR_RULE:
            return dict(common, assignment="utc_month_index_of_the_decision_modulo_4_plus_1")
        return dict(
            common,
            market_proxy="cross_sectional_median_of_log_close_returns_between_present_sessions",
            volatility="rms_of_last_recent_returns_strictly_above_rms_of_earlier_returns",
            recent_returns=self.recent_returns,
            trend="median_over_assets_of_the_window_log_close_return_strictly_positive",
            routes="1_calm_up_2_calm_down_3_turbulent_up_4_turbulent_down",
        )

    def state(self, market, prices, at):
        """Ruta de un mercado a partir de sus ventanas [activos, 64, 6] en el corte `at`."""
        if market not in MARKETS:
            raise ValueError("El régimen solo admite los mercados US y CN")
        prices = np.asarray(prices)
        if (
            prices.ndim != 3
            or prices.shape[1:] != (CONTEXT_SESSIONS, len(PRICE_WINDOW_CHANNELS))
            or len(prices) == 0
        ):
            raise ValueError(
                "El régimen necesita ventanas por sesión del calendario con su canal de presencia"
            )
        present = prices[:, :, _PRESENT]
        if not np.isin(present, (0.0, 1.0)).all():
            raise ValueError("El canal de presencia solo puede valer cero o uno")
        if not (present == present[0]).all():
            raise ValueError("Las ventanas de un mercado no comparten las sesiones presentes")
        if present[0, -1] != 1.0:
            raise ValueError("La sesión de la decisión debe estar presente en la ventana")
        steps = np.flatnonzero(present[0] == 1.0)
        assets, returns = len(prices), len(steps) - 1
        if assets < self.min_assets or returns < self.min_returns:
            return MarketState(market, UNCLASSIFIED, assets, returns, None, None, None)
        closes = prices[:, steps, _CLOSE].astype(np.float64)
        if not np.isfinite(closes).all():
            raise ValueError("Las ventanas de precios contienen cierres no finitos")
        # La mediana no depende del orden de las filas: con un número par de activos promedia
        # los mismos dos valores centrales, y la suma de dos números es conmutativa en IEEE.
        market_returns = np.median(np.diff(closes, axis=1), axis=0)
        recent = market_returns[-self.recent_returns :]
        earlier = market_returns[: -self.recent_returns]
        recent_rms = math.sqrt(float(np.mean(recent * recent)))
        earlier_rms = math.sqrt(float(np.mean(earlier * earlier)))
        trend = float(np.median(closes[:, -1] - closes[:, 0]))
        if self.name == CALENDAR_RULE:
            moment = datetime.fromtimestamp(at / 1_000_000, tz=UTC)
            route = 1 + (moment.year * 12 + moment.month - 1) % 4
        else:
            route = 1 + (0 if trend > 0 else 1) + (2 if recent_rms > earlier_rms else 0)
        return MarketState(market, route, assets, returns, recent_rms, earlier_rms, trend)

    def routes(self, prices, flow_ids, at):
        """Ruta de cada fila de un evento completo y el estado de cada mercado.

        `prices` son todas las ventanas del evento en el orden de `flow_ids`, y `at` el corte
        común en microsegundos UTC. Cada mercado se resume con todas sus filas, así que el
        resultado no depende de cómo se reparta el evento en bloques.
        """
        prices = np.asarray(prices)
        if type(at) is not int or len(flow_ids) != len(prices) or not len(prices):
            raise ValueError("El régimen necesita las ventanas, los flujos y el corte del evento")
        markets = np.array([flow.split("/", 1)[0] for flow in flow_ids])
        routes = np.empty(len(prices), dtype=np.int64)
        states = []
        for market in sorted(set(markets.tolist())):
            rows = np.flatnonzero(markets == market)
            state = self.state(market, prices[rows], at)
            routes[rows] = state.route
            states.append(state)
        return routes, states
