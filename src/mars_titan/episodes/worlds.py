"""Mundos factoriales controlados con precios y publicaciones coherentes."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mars_titan.data.samples import numeric_context
from mars_titan.data.storage import sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US, VALIDATION_START_US
from mars_titan.training.corpus_inputs import _price_contexts

DAY = 86_400_000_000
ENCODING = "synthetic-analytic-v1"


def recipe_fingerprints():
    root = Path(__file__).parents[1]
    names = (
        "data/charts.py",
        "data/samples.py",
        "data/company_factors.py",
        "data/macro_formulas.py",
        "training/corpus_inputs.py",
        "environments/cohorts.py",
    )
    return {name: sha256(root / name) for name in names}


@dataclass(frozen=True)
class WorldConfig:
    assets: int = 8
    sessions: int = 128
    context: int = 16
    sectors: int = 4
    seed: int = 42
    signal: float = 0.002
    volatility: float = 0.006
    jump_probability: float = 0.01
    regime_sessions: int = 32
    publication_lag: int = 2
    partition: str = "train"
    active_assets: tuple[int, ...] | None = None

    def validate(self):
        bounds = {
            "assets": (1, 4096),
            "sessions": (4, 8192),
            "context": (2, 512),
            "sectors": (1, 32),
            "seed": (0, 2**32 - 1),
            "regime_sessions": (4, 8192),
            "publication_lag": (1, 20),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("La configuración del mundo supera sus límites")
        if self.context >= self.sessions or self.assets * self.sessions > 1_048_576:
            raise ValueError("El contexto o el volumen del mundo supera el presupuesto")
        if self.active_assets is not None and (
            len(self.active_assets) != self.sessions
            or any(type(n) is not int or not 1 <= n <= self.assets for n in self.active_assets)
        ):
            raise ValueError("El calendario de empresas activas no corresponde al mundo")
        for value, low, high in (
            (self.signal, 0, 0.05),
            (self.volatility, 0.0001, 0.03),
            (self.jump_probability, 0, 0.1),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError("Los parámetros del mecanismo deben ser finitos y acotados")
        if self.partition not in {"train", "validation"}:
            raise ValueError("El mundo solo admite entrenamiento y validación")
        if self.partition == "validation" and self.sessions > 364:
            raise ValueError("El escenario de validación no puede cruzar el test sellado")


class SyntheticWorld:
    """Fuente de cohortes y de barras brutas, con empresas expresamente ficticias."""

    def __init__(self, config, prices, signal, market_returns, loading, accounts, times):
        self.config, self.prices, self.signal = config, prices, signal
        self.market_returns, self.market_loading = market_returns, loading
        self.assets, self.liabilities, self.equity = accounts
        self.times, self.partition = times, config.partition
        self.asset_ids = [f"US/FIC{i:06d}" for i in range(config.assets)]
        self.max_assets = config.assets
        periods = np.maximum(0, (np.arange(config.sessions) - config.publication_lag) // 20 * 20)
        self.periods = periods
        self.fundamental_period_end = times[periods]
        published = np.where(periods == 0, 0, periods + config.publication_lag)
        self.fundamental_available_at = times[published]
        self.shapes = {
            "prices": (config.context, 5),
            "news": (3,),
            "charts": (16 * 16,),
            "fundamentals": (9,),
            "macro": (6,),
        }
        self.identity = dict(
            schema_version=1,
            generator="factor-world-v1",
            config=asdict(config),
            generator_sha256=sha256(Path(__file__)),
            transformations=recipe_fingerprints(),
            numpy=np.__version__,
            encoding=ENCODING,
            target="known_market_residual_open_close",
            domain="synthetic",
            currency="USD",
            final_test_opened=False,
            units={
                "prices": "USD",
                "volume": "shares",
                "accounts": "USD",
                "macro": "decimal_return_and_loading",
            },
        )
        self.source_sha256 = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        self.manifest_sha256 = self.source_sha256
        self.index = [
            [int(times[i + config.context - 1]), self.count_at(i + config.context - 1)]
            for i in range(len(self))
        ]

    def count_at(self, at):
        return (
            self.config.active_assets[at]
            if self.config.active_assets is not None
            else self.config.assets
        )

    @property
    def raw_world(self):
        return self

    def content_sha256(self):
        digest = hashlib.sha256()
        for array in (
            self.prices,
            self.signal,
            self.market_returns,
            self.market_loading,
            self.assets,
            self.liabilities,
            self.equity,
            self.times,
        ):
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        return digest.hexdigest()

    def __len__(self):
        return self.config.sessions - self.config.context

    def __call__(self, position):
        if type(position) is not int or not 0 <= position <= len(self):
            raise ValueError("Cursor fuera del mundo")
        return None if position == len(self) else self.cohort(position + self.config.context - 1)

    def events(self, at):
        if type(at) is not int or not 0 <= at < self.config.sessions:
            raise ValueError("Fecha de evento fuera del mundo")
        words = {-1: "desfavorable", 0: "sin cambios", 1: "favorable"}
        n = self.count_at(at)
        return [
            f"La empresa ficticia {asset} publica una revisión {words[int(value)]}."
            for asset, value in zip(self.asset_ids[:n], self.signal[at, :n], strict=True)
        ]

    def chart(self, at):
        """Rasterizar únicamente los cierres presentes en la ventana de decisión."""
        if type(at) is not int or not self.config.context - 1 <= at < self.config.sessions:
            raise ValueError("El gráfico necesita una ventana disponible completa")
        closes = self.prices[at - self.config.context + 1 : at + 1, :, 3].T
        low, high = closes.min(1), closes.max(1)
        y = np.rint(15 * (closes - low[:, None]) / np.maximum(high - low, 1e-12)[:, None]).astype(
            int
        )
        x = np.rint(np.linspace(0, 15, closes.shape[1])).astype(int)
        pixels = np.zeros((len(closes), 16, 16), dtype=np.float32)
        for i in range(len(closes)):
            pixels[i, 15 - y[i], x] = 1
        return pixels.reshape(len(closes), -1)

    def cohort(self, at):
        if type(at) is not int or not self.config.context - 1 <= at < self.config.sessions - 1:
            raise ValueError("La decisión necesita contexto y una etiqueta posterior")
        n, context, period = self.count_at(at), self.config.context, self.periods[at]
        price_inputs = np.stack(
            [_price_contexts(self.prices[:, i], np.array([at]), context)[0] for i in range(n)]
        )
        news = np.stack((self.signal[at, :n], np.ones(n), np.zeros(n)), axis=1).astype(np.float32)
        age = (self.times[at] - self.fundamental_available_at[at]) / DAY
        fundamental = np.array(
            [
                numeric_context(
                    [self.assets[period, i], self.liabilities[period, i], self.equity[period, i]],
                    [age] * 3,
                )
                for i in range(n)
            ],
            dtype=np.float32,
        )
        macro = np.broadcast_to(
            np.array(
                numeric_context([self.market_returns[at], self.market_loading[at]], [0, 0]),
                dtype=np.float32,
            ),
            (n, 6),
        ).copy()
        target = self.prices[at + 1, :n, 3] / self.prices[at + 1, :n, 0] - 1
        target -= self.market_returns[at + 1] * self.market_loading[at + 1]
        return dict(
            prediction_at=int(self.times[at]),
            asset_ids=list(self.asset_ids[:n]),
            available_at=np.full(n, self.times[at], dtype=np.int64),
            target_available_at=np.full(n, self.times[at + 1], dtype=np.int64),
            target=target,
            inputs=dict(
                prices=price_inputs,
                news=news,
                charts=self.chart(at)[:n],
                fundamentals=fundamental,
                macro=macro,
            ),
        )


def generate_world(config):
    config.validate()
    rng = np.random.default_rng(config.seed)
    days, n = config.sessions, config.assets
    signal = rng.integers(-1, 2, (days, n))
    regimes = (np.arange(days) // config.regime_sessions) % 2
    loading = np.where(regimes == 0, 0.4, 1.2)
    market = np.zeros(days)
    prices = np.empty((days, n, 5), dtype=np.float64)
    assets = np.empty((days, n))
    liabilities = np.empty_like(assets)
    equity = np.empty_like(assets)
    sector_ids = np.arange(n) % config.sectors
    previous = np.full(n, 100.0)
    variance = config.volatility**2
    for at in range(days):
        # Los factores se comparten entre empresas dentro de la misma sesión.
        shock = rng.normal()
        variance = 0.85 * variance + 0.15 * config.volatility**2 * (0.5 + shock**2)
        market[at] = np.clip(math.sqrt(variance) * shock, -0.1, 0.1)
        sector = np.clip(
            rng.normal(0, config.volatility * 0.6, config.sectors),
            -1.8 * config.volatility,
            1.8 * config.volatility,
        )
        individual = np.clip(
            rng.normal(0, config.volatility, n), -3 * config.volatility, 3 * config.volatility
        )
        jumps = (rng.random(n) < config.jump_probability) * rng.choice([-1, 1], n) * 0.03
        expected = config.signal * signal[at - 1] if at else np.zeros(n)
        returns = np.clip(
            loading[at] * market[at] + sector[sector_ids] + individual + jumps + expected, -0.4, 0.4
        )
        opening = previous * (1 + rng.uniform(-0.003, 0.003, n))
        closing = opening * (1 + returns)
        high = np.maximum(opening, closing) * (1 + rng.uniform(0, 0.004, n))
        low = np.minimum(opening, closing) * (1 - rng.uniform(0, 0.004, n))
        volume = rng.integers(10_000, 100_000, n).astype(float)
        prices[at] = np.stack((opening, high, low, closing, volume), axis=1)
        equity[at] = np.maximum(10, (equity[at - 1] if at else np.full(n, 60.0)) + returns)
        liabilities[at] = 40 + sector_ids
        assets[at] = equity[at] + liabilities[at]
        previous = closing
    start = VALIDATION_START_US if config.partition == "validation" else 946_684_800_000_000
    times = start + np.arange(days, dtype=np.int64) * DAY + 16 * 3_600_000_000
    limit = VALIDATION_START_US if config.partition == "train" else FINAL_TEST_START_US
    if times[-1] >= limit:
        raise ValueError("El mundo cruza el límite de su partición")
    return SyntheticWorld(
        config, prices, signal, market, loading, (assets, liabilities, equity), times
    )
