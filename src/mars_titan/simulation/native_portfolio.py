"""Cartera con arrays persistentes y cálculo C++, compatible con estados de referencia."""

import copy
import ctypes
import hashlib
import json
import math
import threading
from types import MappingProxyType

import numpy as np

from .native_runtime import ACCOUNT, POSITION, TRADE, load_library
from .portfolio import Portfolio, Quote, amount, instant

REASONS = {1: "missing_open", 2: "unknown_liquidity", 3: "cash_liquidity_or_lot_limit"}


class NativePortfolio:
    def __init__(self, instruments, cash, *, cost_bps=10, participation=0.01, library=None):
        self._schema = Portfolio(instruments, cash, cost_bps=cost_bps, participation=participation)
        if len(cash) > 32:
            raise ValueError("El núcleo nativo admite como máximo 32 cuentas")
        self.library = load_library(library)
        self.identity = copy.deepcopy(self._schema.identity)
        self.instruments = MappingProxyType(self._schema.instruments)
        self.assets, self.currencies = tuple(self.instruments), tuple(sorted(cash))
        self._asset_index = {asset: i for i, asset in enumerate(self.assets)}
        self._currency_ids = np.asarray(
            [self.currencies.index(self.instruments[a].currency) for a in self.assets],
            dtype=np.uint32,
        )
        self._lots = np.asarray([self.instruments[a].lot for a in self.assets], dtype=np.float64)
        self.rate, self.participation = self._schema.rate, participation
        self._lock = threading.RLock()
        self._error = ctypes.create_string_buffer(512)
        self._positions = np.zeros(len(self.assets), dtype=POSITION)
        self._next_positions = np.zeros_like(self._positions)
        self._accounts = np.zeros(len(self.currencies), dtype=ACCOUNT)
        self._next_accounts = np.zeros_like(self._accounts)
        self._trades = np.zeros(len(self.assets), dtype=TRADE)
        self._retired = np.zeros(len(self.assets), dtype=np.uint8)
        self._prices = np.full((len(self.assets), 5), np.nan)
        self._metadata = {}
        self._counts = dict(native_steps=0, reference_event_steps=0)
        self._load_state(self._schema.snapshot()["state"])

    def _load_state(self, state):
        positions = np.zeros_like(self._positions)
        positions["target"], positions["capacity"] = np.nan, np.nan
        accounts = np.zeros_like(self._accounts)
        retired = np.zeros_like(self._retired)
        for asset, quantity in state["positions"].items():
            positions[self._asset_index[asset]]["quantity"] = quantity
        for asset, order in state["orders"].items():
            row = positions[self._asset_index[asset]]
            row["target"], row["decision_at"] = order["target"], order["decision_at"]
            row["capacity"] = order["capacity"] if order["capacity"] is not None else np.nan
        for asset in state["retired"]:
            retired[self._asset_index[asset]] = 1
        for index, currency in enumerate(self.currencies):
            for field in ("cash", "costs", "turnover", "nav"):
                value = state[field][currency]
                accounts[index][field] = value if value is not None else np.nan
            accounts[index]["receivable"] = math.fsum(
                e["amount"] for e in state["receivables"] if e["currency"] == currency
            )
        quotes = {asset: Quote(**quote) for asset, quote in state["quotes"].items()}
        prices, quote_keys = self._frame(quotes), tuple(sorted(quotes))
        metadata = {
            key: copy.deepcopy(state[key])
            for key in ("receivables", "applied", "retired", "last_close", "last_open")
        }
        held_order = tuple(self._asset_index[asset] for asset in state["positions"])
        (
            self._positions,
            self._accounts,
            self._retired,
            self._prices,
            self._quote_keys,
            self._metadata,
            self._held_order,
        ) = (positions, accounts, retired, prices, quote_keys, metadata, held_order)

    def _frame(self, quotes):
        self._schema._quotes(quotes)
        frame = np.full((len(self.assets), 5), np.nan)
        for asset, quote in quotes.items():
            index = self._asset_index[asset]
            for column, value in ((0, quote.open), (3, quote.close), (4, quote.volume)):
                frame[index, column] = value if value is not None else np.nan
        return frame

    def _new_reference(self):
        return Portfolio(
            self.instruments,
            self.identity["initial_cash"],
            cost_bps=self.identity["cost_bps"],
            participation=self.participation,
        )

    def _account_values(self, field, accounts=None):
        accounts = self._accounts if accounts is None else accounts
        return {
            name: None if np.isnan(row[field]) else float(row[field])
            for name, row in zip(self.currencies, accounts, strict=True)
        }

    @property
    def cash(self):
        with self._lock:
            return self._account_values("cash")

    @property
    def nav(self):
        with self._lock:
            return self._account_values("nav")

    @property
    def clock(self):
        with self._lock:
            return self._metadata["last_close"]

    @property
    def positions(self):
        with self._lock:
            return {self.assets[i]: float(self._positions[i]["quantity"]) for i in self._held_order}

    @property
    def orders(self):
        with self._lock:
            return {
                self.assets[i]: dict(
                    target=float(self._positions[i]["target"]),
                    capacity=None
                    if np.isnan(self._positions[i]["capacity"])
                    else float(self._positions[i]["capacity"]),
                    decision_at=int(self._positions[i]["decision_at"]),
                )
                for i in np.flatnonzero(~np.isnan(self._positions["target"]))
            }

    @property
    def retired(self):
        with self._lock:
            return frozenset(self._metadata["retired"])

    @property
    def execution_counts(self):
        with self._lock:
            return dict(self._counts)

    def start(self, close_at, quotes):
        with self._lock:
            if self.clock is not None:
                raise ValueError("La cartera ya se ha iniciado")
            reference = self._new_reference()
            reference.start(close_at, quotes)
            self._load_state(reference.snapshot()["state"])

    def submit(self, targets, *, decision_at):
        with self._lock:
            if (
                instant(decision_at) != self.clock
                or not set(targets) <= set(self.assets)
                or any(asset in self.retired and value != 0 for asset, value in targets.items())
            ):
                raise ValueError("La orden necesita una decisión en el último cierre disponible")
            checked = [
                (self._asset_index[asset], amount(value)) for asset, value in targets.items()
            ]
            if any(value > 1e12 for _, value in checked):
                raise ValueError("La cantidad supera el presupuesto del simulador")
            (
                self._positions["target"],
                self._positions["capacity"],
                self._positions["decision_at"],
            ) = np.nan, np.nan, 0
            for index, target in checked:
                row = self._positions[index]
                if abs(target - row["quantity"]) < 1e-12:
                    continue
                row["target"], row["decision_at"] = target, decision_at
                row["capacity"] = self._prices[index, 4] * self.participation

    def advance(self, open_at, close_at, quotes, *, actions=()):
        with self._lock:
            return self._advance(
                open_at, close_at, self._frame(quotes), actions, tuple(sorted(quotes))
            )

    def advance_frame(self, open_at, close_at, frame, *, actions=()):
        with self._lock:
            return self._advance(
                open_at,
                close_at,
                self.library.frame(frame, len(self.assets)),
                actions,
                tuple(self.assets),
            )

    def _advance(self, open_at, close_at, frame, actions, quote_keys):
        if self.clock is None or not self.clock < instant(open_at) < instant(close_at):
            raise ValueError("La ejecución debe ser posterior al cierre de decisión")
        due = any(entry["pay_at"] <= close_at for entry in self._metadata["receivables"])
        counts = dict(self._counts)
        if actions or due:
            # Reutilizar la contabilidad de eventos infrecuentes y su auditoría.
            reference = self._new_reference()
            reference.restore(self.snapshot())
            quotes = {
                asset: Quote(
                    *(
                        None
                        if np.isnan(frame[self._asset_index[asset], i])
                        else float(frame[self._asset_index[asset], i])
                        for i in (0, 3, 4)
                    )
                )
                for asset in quote_keys
            }
            result = reference.advance(open_at, close_at, quotes, actions=actions)
            counts["reference_event_steps"] += 1
            self._load_state(reference.snapshot()["state"])
            self._counts = counts
            return result
        prices = frame.copy()
        metadata = dict(self._metadata, last_open=open_at, last_close=close_at)
        counts["native_steps"] += 1
        self.library.step(self, frame, open_at, close_at)
        result, held_order = self._result(self._next_positions, self._next_accounts, frame)
        (
            self._positions,
            self._next_positions,
            self._accounts,
            self._next_accounts,
            self._prices,
            self._quote_keys,
            self._metadata,
            self._counts,
            self._held_order,
        ) = (
            self._next_positions,
            self._positions,
            self._next_accounts,
            self._accounts,
            prices,
            quote_keys,
            metadata,
            counts,
            held_order,
        )
        return result

    def _result(self, positions, accounts, frame):
        trades = []
        for index in np.flatnonzero(self._trades["quantity"]):
            row = self._trades[index]
            trades.append(
                dict(
                    asset=self.assets[index],
                    quantity=float(row["quantity"]),
                    price=float(row["price"]),
                    cost=float(row["cost"]),
                    currency=self.instruments[self.assets[index]].currency,
                )
            )
        # El recibo conserva el orden de ventas y compras de la referencia.
        trades.sort(
            key=lambda trade: (
                trade["quantity"] > 0,
                trade["currency"] if trade["quantity"] > 0 else "",
                trade["asset"],
            )
        )
        unfilled = [
            dict(asset=self.assets[i], reason=REASONS[int(self._trades[i]["reason"])])
            for i in np.flatnonzero(self._trades["reason"])
        ]
        unfilled.sort(key=lambda row: (row["reason"] == REASONS[3], row["asset"]))
        held_order = [i for i in self._held_order if positions[i]["quantity"] != 0]
        retained = set(held_order)
        for trade in trades:
            index = self._asset_index[trade["asset"]]
            if trade["quantity"] > 0 and index not in retained:
                held_order.append(index)
                retained.add(index)
        result = dict(
            nav=self._account_values("nav", accounts),
            costs=self._account_values("costs", accounts),
            turnover=self._account_values("turnover", accounts),
            trades=trades,
            unfilled=unfilled,
            unvalued=[self.assets[i] for i in held_order if np.isnan(frame[i, 3])],
        )
        return result, tuple(held_order)

    def observation(self, prices, previous, scores, score_scale):
        with self._lock:
            if len(self.currencies) != 1:
                raise ValueError("La observación financiera necesita una moneda única")
            return self.library.observation(self, prices, previous, scores, score_scale)

    def snapshot(self):
        with self._lock:
            state = dict(
                copy.deepcopy(self._metadata),
                cash=self.cash,
                positions=self.positions,
                orders=self.orders,
                nav=self.nav,
                costs=self._account_values("costs"),
                turnover=self._account_values("turnover"),
                quotes={
                    asset: {
                        name: None
                        if np.isnan(self._prices[self._asset_index[asset], col])
                        else float(self._prices[self._asset_index[asset], col])
                        for name, col in (("open", 0), ("close", 3), ("volume", 4))
                    }
                    for asset in self._quote_keys
                },
            )
            checksum = hashlib.sha256(
                json.dumps(state, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            return dict(
                schema_version=1,
                identity=copy.deepcopy(self.identity),
                state=state,
                state_sha256=checksum,
                execution_counts=dict(self._counts),
            )

    def restore(self, snapshot):
        with self._lock:
            reference = self._new_reference()
            reference.restore(snapshot)
            counts = snapshot.get("execution_counts", dict(native_steps=0, reference_event_steps=0))
            if set(counts) != set(self._counts) or any(
                type(value) is not int or not 0 <= value < 2**53 for value in counts.values()
            ):
                raise ValueError("Los contadores nativos recuperados no son válidos")
            confirmed_counts = dict(counts)
            self._load_state(reference.snapshot()["state"])
            self._counts = confirmed_counts
