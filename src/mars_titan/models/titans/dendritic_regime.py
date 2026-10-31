"""Contexto N2 de régimen observable, calculado solo con la ventana recibida.

El control shuffled_past_prefix permuta los prefijos anteriores elegibles y toma
el primero. Es un muestreo histórico causal, no una permutación global del tramo
ni un control que conserve la distribución del régimen contemporáneo.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from mars_titan.memory.regimes import REGIME_RULE, RegimeRule

from .config import bounded_integer
from .financial_inputs import DecisionBatch

MODES = ("observable", "shuffled_past_prefix", "constant")


@dataclass(frozen=True)
class DendriticRegime:
    mode: str = "observable"
    segments: int = 4
    min_assets: int = 20
    recent_returns: int = 21
    min_returns: int = 42

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("El contexto de N2 no declara un modo admitido")
        bounded_integer(self.segments, "segmentos de N2", 1, 64)
        self.rule()

    def rule(self):
        return RegimeRule(REGIME_RULE, self.min_assets, self.recent_returns, self.min_returns)

    def identity(self):
        return dict(
            kind="dendritic_observable_regime_v1",
            mode=self.mode,
            segments=self.segments,
            rule=self.rule().identity(),
            context="four_one_hot_routes_with_zero_for_unclassified",
            position="after_session_context_before_mac",
            seed="sha256_namespaces_of_the_predictor_seed",
            selection="first_of_a_seeded_permutation_of_eligible_earlier_prefix_slots",
            unavailable_history="zero_context",
            state="none_recomputed_from_the_received_window",
            differentiation="no_gradient_through_discrete_regime_assignment",
        )


def parse_dendritic_regime(value):
    if value is None:
        return None
    if type(value) is DendriticRegime:
        value.__post_init__()
        return value
    if type(value) is dict and set(value) == set(DendriticRegime.__dataclass_fields__):
        return DendriticRegime(**value)
    raise ValueError("active_dendrites debe declarar todos sus campos o ser None")


def parameter_seed(seed):
    bounded_integer(seed, "semilla de N2", 0, 2**32 - 1)
    digest = hashlib.sha256(f"active_dendrites_v1/{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class DendriticContext:
    values: torch.Tensor
    routes: tuple[int, ...]
    source_slots: tuple[int, ...]
    prediction_at: int
    input_available_at: tuple[int, ...]


def _route(rule, market, prices, at):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return rule.state(market, prices, at).route
    except FloatingPointError as error:
        raise ValueError("El régimen de N2 desborda su aritmética") from error


def regime_context(config, batch, *, seed):
    """Crear contexto y procedencia por fila tras verificar la frontera de decisión.

    La fuente usada en CUDA se copia explícitamente a CPU para reutilizar RegimeRule.
    Los slots de los prefijos no se presentan como fechas históricas reconstruidas.
    """
    if type(config) is not DendriticRegime or type(batch) is not DecisionBatch:
        raise ValueError("N2 necesita su configuración y un lote de decisión verificado")
    config.__post_init__()
    parameter_seed(seed)
    batch.verify()
    at = batch.prediction_at[0]
    if len(set(batch.prediction_at)) != 1 or max(batch.input_available_at) > at:
        raise ValueError("El contexto de N2 no puede consultar entradas posteriores al corte")
    prices = batch.inputs["prices"]
    windows = prices.detach().cpu().numpy()
    markets = np.array([flow.split("/", 1)[0] for flow in batch.flow_ids])
    routes = np.zeros(len(markets), dtype=np.int64)
    slots = np.full(len(markets), -1, dtype=np.int64)
    availability = [0] * len(markets)
    rule = config.rule()
    for market in sorted(set(markets.tolist())):
        rows = np.flatnonzero(markets == market)
        source = windows[rows]
        route = _route(rule, market, source, at)
        slot = 63 if route else -1
        if config.mode == "shuffled_past_prefix" and route:
            present = np.flatnonzero(source[0, :, 5] == 1)
            candidates = present[config.min_returns : -1]
            if not len(candidates):
                route, slot = 0, -1
            else:
                digest = hashlib.sha256(f"N2/prefix/{seed}/{market}/{at}".encode()).digest()
                generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                slot = int(generator.permutation(candidates)[0])
                prefix = np.zeros_like(source)
                prefix[:, -(slot + 1) :] = source[:, : slot + 1]
                route = _route(rule, market, prefix, at)
        elif config.mode == "constant" and route:
            route = 1
        routes[rows], slots[rows] = route, slot
        available = max(batch.input_available_at[int(i)] for i in rows)
        for index in rows:
            availability[int(index)] = available
    values = torch.as_tensor(np.eye(5)[routes, 1:], dtype=prices.dtype, device=prices.device)
    return DendriticContext(
        values, tuple(routes.tolist()), tuple(slots.tolist()), at, tuple(availability)
    )
