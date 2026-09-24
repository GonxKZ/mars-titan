"""Seis decisiones experimentales comunes para comparadores financieros."""

import copy
import math

import gymnasium as gym
import numpy as np

from .portfolio import Instrument, Portfolio

ACTIONS = (None, 0.0, 0.25, 0.5, 0.75, 1.0)


class RecoverablePause(RuntimeError):
    """Pausa administrativa sin transición, recompensa ni final de episodio."""


class FinancialEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tape,
        *,
        capital=10000,
        cost_bps=10,
        participation=0.01,
        score_scale=0.01,
        ruin_penalty=-20,
        backend="python",
        native_library=None,
    ):
        if (
            not math.isfinite(score_scale)
            or score_scale <= 0
            or not math.isfinite(ruin_penalty)
            or ruin_penalty >= 0
        ):
            raise ValueError("La escala y la penalización de ruina deben fijarse antes de evaluar")
        if backend not in {"python", "native"} or (
            native_library is not None and backend != "native"
        ):
            raise ValueError("El motor debe ser python o native con una biblioteca explícita")
        self.backend, self.native_library = backend, None
        if backend == "native":
            from .native_runtime import load_library

            self.native_library = load_library(native_library)
        self.tape, self.capital, self.cost_bps, self.participation = (
            tape,
            capital,
            cost_bps,
            participation,
        )
        self.score_scale, self.ruin_penalty = score_scale, ruin_penalty
        self.identity = dict(
            tape_sha256=tape.sha256,
            capital=capital,
            cost_bps=cost_bps,
            participation=participation,
            score_scale=score_scale,
            ruin_penalty=ruin_penalty,
            allocation="positive_top_quartile_equal_weight",
            action_levels=list(ACTIONS),
            final_test_opened=False,
            accounting_backend=backend,
            native_library_sha256=self.native_library.sha256 if self.native_library else None,
        )
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(-10, 10, (6 * len(tape.assets) + 2,), np.float32)
        self.book, self.cursor, self.done, self.paused = None, 0, True, False

    def _new_book(self):
        instruments = {asset: Instrument(self.tape.currency) for asset in self.tape.assets}
        cash = {self.tape.currency: self.capital}
        options = dict(cost_bps=self.cost_bps, participation=self.participation)
        if self.backend == "native":
            from .native_portfolio import NativePortfolio

            return NativePortfolio(instruments, cash, library=self.native_library.path, **options)
        return Portfolio(instruments, cash, **options)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.book = self._new_book()
        self.book.start(int(self.tape.close_times[0]), self.tape.quotes(0))
        self.cursor, self.done, self.paused = 0, False, False
        return self._observation(), {"domain": self.tape.domain, "parent_frozen": True}

    def _observation(self):
        if self.backend == "native":
            return self.book.observation(
                self.tape.prices[self.cursor],
                self.tape.prices[max(0, self.cursor - 1)],
                self.tape.scores[self.cursor],
                self.score_scale,
            )
        nav = self.book.nav[self.tape.currency]
        if nav is None or nav <= 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        values, invested = [], 0.0
        positions, orders, retired = self.book.positions, self.book.orders, self.book.retired
        for i, asset in enumerate(self.tape.assets):
            close = self.tape.prices[self.cursor, i, 3]
            before = self.tape.prices[max(0, self.cursor - 1), i, 3]
            score = self.tape.scores[self.cursor, i]
            volume = self.tape.prices[self.cursor, i, 4]
            valid = np.isfinite(close) and np.isfinite(score) and asset not in retired
            weight = positions.get(asset, 0) * close / nav if np.isfinite(close) else 0
            invested += weight
            order = orders.get(asset)
            pending = (
                (order["target"] - positions.get(asset, 0)) * close / nav
                if order and np.isfinite(close)
                else 0
            )
            values.extend(
                (
                    score / self.score_scale if np.isfinite(score) else 0,
                    weight,
                    close / before - 1 if np.isfinite(close) and np.isfinite(before) else 0,
                    math.log1p(volume) / 20 if np.isfinite(volume) else 0,
                    pending,
                    float(valid),
                )
            )
        cash = self.book.cash[self.tape.currency] / nav
        values.extend((cash, max(0, 1 - cash - invested)))
        return np.clip(np.asarray(values, dtype=np.float64), -10, 10).astype(np.float32)

    def _targets(self, exposure):
        nav = self.book.nav[self.tape.currency]
        candidates = []
        for i, asset in enumerate(self.tape.assets):
            score, price = self.tape.scores[self.cursor, i], self.tape.prices[self.cursor, i, 3]
            if (
                np.isfinite(score)
                and score > 0
                and np.isfinite(price)
                and asset not in self.book.retired
            ):
                candidates.append((-score, asset, price))
        candidates.sort()
        selected = candidates[: max(1, math.ceil(len(candidates) / 4))] if candidates else []
        targets = {asset: 0 for asset in self.book.positions.keys() | self.book.orders.keys()}
        for _, asset, price in selected:
            targets[asset] = float(nav * exposure / len(selected) / price)
        return targets

    def step(self, action):
        if self.book is None or self.done:
            raise RuntimeError("Reinicia el episodio antes de decidir")
        if self.paused:
            raise RecoverablePause("La ejecución está pausada y conserva su estado")
        if (
            isinstance(action, bool)
            or not isinstance(action, (int, np.integer))
            or not self.action_space.contains(action)
        ):
            raise ValueError("La acción financiera debe estar entre cero y cinco")
        exposure = ACTIONS[int(action)]
        if exposure is not None:
            self.book.submit(self._targets(exposure), decision_at=self.book.clock)
        previous = self.book.nav[self.tape.currency]
        following = self.cursor + 1
        at = int(self.tape.open_times[following])
        advance = self.book.advance_frame if self.backend == "native" else self.book.advance
        quotes = (
            self.tape.prices[following] if self.backend == "native" else self.tape.quotes(following)
        )
        result = advance(
            at,
            int(self.tape.close_times[following]),
            quotes,
            actions=[a for a in self.tape.actions if a.effective_at == at],
        )
        self.cursor = following
        nav = result["nav"][self.tape.currency]
        valid, terminated = nav is not None, nav is not None and nav <= 0
        truncated = not valid or (following == len(self.tape) - 1 and not terminated)
        reason = (
            "missing_close"
            if not valid
            else "ruin"
            if terminated
            else "episode_limit"
            if truncated
            else None
        )
        reward = (
            0.0
            if not valid
            else self.ruin_penalty
            if terminated
            else math.log(nav) - math.log(previous)
        )
        self.done = terminated or truncated
        info = dict(
            result,
            reward_valid=valid,
            reason=reason,
            parent_frozen=True,
            reward_kind="ruin_penalty" if terminated else "log_net_worth_change",
        )
        return self._observation(), reward, terminated, truncated, info

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def snapshot(self):
        if self.book is None:
            raise RuntimeError("El episodio no está inicializado")
        return copy.deepcopy(
            dict(
                schema_version=1,
                identity=self.identity,
                cursor=self.cursor,
                done=self.done,
                paused=self.paused,
                book=self.book.snapshot(),
                rng=self.np_random.bit_generator.state,
                action_rng=self.action_space.np_random.bit_generator.state,
                rng_seed=self.np_random_seed,
            )
        )

    def restore(self, state):
        if (
            state.get("schema_version") != 1
            or state.get("identity") != self.identity
            or type(state.get("cursor")) is not int
            or not 0 <= state["cursor"] < len(self.tape)
            or type(state.get("done")) is not bool
            or type(state.get("paused")) is not bool
        ):
            raise ValueError("El estado pertenece a otro episodio o tiene un cursor inválido")
        book = self._new_book()
        book.restore(state["book"])
        if book.clock != int(self.tape.close_times[state["cursor"]]):
            raise ValueError("El cursor no corresponde al cierre de la cartera")
        nav = book.nav[self.tape.currency]
        expected_done = state["cursor"] == len(self.tape) - 1 or nav is None or nav <= 0
        if state["done"] != expected_done:
            raise ValueError("El final guardado no corresponde al cursor o a la valoración")
        rng, action_rng = np.random.default_rng(), np.random.default_rng()
        rng.bit_generator.state, action_rng.bit_generator.state = state["rng"], state["action_rng"]
        self.book, self.cursor, self.done, self.paused = (
            book,
            state["cursor"],
            state["done"],
            state["paused"],
        )
        self.np_random, self.action_space._np_random = rng, action_rng
        self._np_random_seed = state["rng_seed"]
