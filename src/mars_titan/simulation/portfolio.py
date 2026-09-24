"""Contabilidad de posiciones largas por moneda y ejecución posterior a la decisión."""

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


def amount(value, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError("El importe debe ser finito y tener un signo válido")
    return float(value)


def instant(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("El instante debe expresarse como entero UTC no negativo")
    return value


@dataclass(frozen=True)
class Instrument:
    currency: str
    lot: float = 1

    def __post_init__(self):
        if not re.fullmatch(r"[A-Z]{3}", self.currency):
            raise ValueError("La moneda necesita un identificador explícito")
        amount(self.lot, positive=True)


@dataclass(frozen=True)
class Quote:
    open: float | None
    close: float | None
    volume: float | None

    def __post_init__(self):
        for value in (self.open, self.close):
            if value is not None:
                amount(value, positive=True)
        if self.volume is not None:
            amount(self.volume)


@dataclass(frozen=True)
class CorporateAction:
    id: str
    asset: str
    kind: str
    effective_at: int
    value: float
    pay_at: int | None = None
    verified: bool = False

    def validate(self):
        if (
            not isinstance(self.id, str)
            or not 1 <= len(self.id) <= 128
            or self.kind not in {"split", "dividend", "writeoff"}
            or self.verified is not True
        ):
            raise ValueError("La acción corporativa necesita un tratamiento acreditado")
        instant(self.effective_at)
        amount(self.value, positive=self.kind == "split")
        if self.kind == "dividend":
            if self.pay_at is None or instant(self.pay_at) < self.effective_at:
                raise ValueError("El dividendo necesita una fecha de pago coherente")
        if self.kind == "writeoff" and self.value != 0:
            raise ValueError("La baja sin recuperación debe tener valor cero")


class Portfolio:
    def __init__(self, instruments, cash, *, cost_bps=10, participation=0.01):
        if (
            not 1 <= len(instruments) <= 4096
            or not cash
            or any(not isinstance(i, Instrument) for i in instruments.values())
            or any(not isinstance(name, str) or not 1 <= len(name) <= 96 for name in instruments)
            or any(i.currency not in cash for i in instruments.values())
            or not 0 < participation <= 1
            or not 0 <= cost_bps <= 1000
        ):
            raise ValueError("Los instrumentos, costes o presupuestos no son válidos")
        cash = {key: amount(value) for key, value in sorted(cash.items())}
        if not any(cash.values()):
            raise ValueError("Se necesita capital inicial positivo")
        self.instruments = dict(sorted(instruments.items()))
        self.rate, self.participation = cost_bps / 10000, participation
        self.identity = dict(
            instruments={k: asdict(v) for k, v in self.instruments.items()},
            initial_cash=cash,
            cost_bps=cost_bps,
            participation=participation,
        )
        self._state = dict(
            cash=dict(cash),
            positions={},
            orders={},
            receivables=[],
            applied=[],
            retired=[],
            quotes={},
            last_close=None,
            last_open=None,
            nav=dict(cash),
            costs={c: 0.0 for c in cash},
            turnover={c: 0.0 for c in cash},
        )

    @property
    def cash(self):
        return dict(self._state["cash"])

    @property
    def positions(self):
        return dict(self._state["positions"])

    @property
    def clock(self):
        return self._state["last_close"]

    @property
    def nav(self):
        return dict(self._state["nav"])

    @property
    def orders(self):
        return copy.deepcopy(self._state["orders"])

    @property
    def retired(self):
        return frozenset(self._state["retired"])

    def _quotes(self, quotes):
        if not set(quotes) <= set(self.instruments) or any(
            not isinstance(q, Quote) for q in quotes.values()
        ):
            raise ValueError("Las cotizaciones no corresponden a los instrumentos")

    def start(self, close_at, quotes):
        instant(close_at)
        self._quotes(quotes)
        if self.clock is not None:
            raise ValueError("La cartera ya se ha iniciado")
        self._mark(self._state, close_at, quotes)

    def submit(self, targets, *, decision_at):
        if (
            instant(decision_at) != self.clock
            or not set(targets) <= set(self.instruments)
            or any(asset in self.retired and value != 0 for asset, value in targets.items())
        ):
            raise ValueError("La orden necesita una decisión en el último cierre disponible")
        orders = {}
        for asset, value in sorted(targets.items()):
            quantity = amount(value)
            if quantity > 1e12:
                raise ValueError("La cantidad supera el presupuesto del simulador")
            if abs(quantity - self._state["positions"].get(asset, 0)) < 1e-12:
                continue
            volume = self._state["quotes"].get(asset, {}).get("volume")
            orders[asset] = dict(
                target=quantity,
                decision_at=decision_at,
                capacity=volume * self.participation if volume is not None else None,
            )
        self._state["orders"] = orders

    def _settle(self, state, at):
        outstanding = []
        for entry in state["receivables"]:
            if entry["pay_at"] <= at:
                state["cash"][entry["currency"]] += entry["amount"]
            else:
                outstanding.append(entry)
        state["receivables"] = outstanding

    def _actions(self, state, actions):
        for action in actions:
            asset, quantity = action.asset, state["positions"].get(action.asset, 0)
            currency = self.instruments[asset].currency
            if action.kind == "split":
                if quantity:
                    state["positions"][asset] = quantity * action.value
                if asset in state["orders"]:
                    order = state["orders"][asset]
                    order["target"] *= action.value
                    if order["capacity"] is not None:
                        order["capacity"] *= action.value
            elif action.kind == "dividend" and quantity:
                state["receivables"].append(
                    dict(
                        id=action.id,
                        currency=currency,
                        amount=quantity * action.value,
                        pay_at=action.pay_at,
                    )
                )
            elif action.kind == "writeoff":
                state["positions"].pop(asset, None)
                state["orders"].pop(asset, None)
                state["retired"].append(asset)
            state["applied"].append(action.id)

    def _fill(self, state, asset, quantity, price, trades):
        currency = self.instruments[asset].currency
        notional, cost = quantity * price, abs(quantity * price) * self.rate
        before = state["cash"][currency]
        remaining = math.fsum((before, -notional, -cost))
        # Corregir solo el redondeo de la resta, nunca financiar un descubierto.
        tolerance = 8 * math.ulp(max(before, abs(notional), cost, 1))
        state["cash"][currency] = 0.0 if -tolerance <= remaining < 0 else remaining
        held = state["positions"].get(asset, 0) + quantity
        if abs(held) < 1e-12:
            state["positions"].pop(asset, None)
        else:
            state["positions"][asset] = held
        state["costs"][currency] += cost
        state["turnover"][currency] += abs(notional)
        trades.append(
            dict(asset=asset, quantity=quantity, price=price, cost=cost, currency=currency)
        )

    def _execute(self, state, quotes):
        trades, unfilled, buys = [], [], []
        for asset, order in sorted(state["orders"].items()):
            quote = quotes.get(asset)
            if quote is None or quote.open is None:
                unfilled.append(dict(asset=asset, reason="missing_open"))
                continue
            if order["capacity"] is None:
                unfilled.append(dict(asset=asset, reason="unknown_liquidity"))
                continue
            delta = order["target"] - state["positions"].get(asset, 0)
            quantity = min(abs(delta), order["capacity"])
            lot = self.instruments[asset].lot
            quantity = math.floor(quantity / lot) * lot
            if delta < 0:
                quantity = min(state["positions"].get(asset, 0), quantity)
                if quantity:
                    self._fill(state, asset, -quantity, quote.open, trades)
            elif quantity:
                buys.append((asset, quantity, quote.open))
        for currency in state["cash"]:
            selected = [b for b in buys if self.instruments[b[0]].currency == currency]
            required = math.fsum(q * price * (1 + self.rate) for _, q, price in selected)
            scale = min(1.0, state["cash"][currency] / required) if required else 0
            for asset, quantity, price in selected:
                lot = self.instruments[asset].lot
                executable = math.floor(quantity * scale / lot) * lot
                if executable:
                    self._fill(state, asset, executable, price, trades)
        pending = {}
        reasons = {entry["asset"] for entry in unfilled}
        for asset, order in state["orders"].items():
            if abs(order["target"] - state["positions"].get(asset, 0)) >= 1e-12:
                pending[asset] = order
                if asset not in reasons:
                    unfilled.append(dict(asset=asset, reason="cash_liquidity_or_lot_limit"))
        state["orders"] = pending
        return trades, unfilled

    def _mark(self, state, close_at, quotes):
        self._settle(state, close_at)
        nav = {}
        for currency, cash in state["cash"].items():
            positions = [
                (asset, quantity)
                for asset, quantity in state["positions"].items()
                if self.instruments[asset].currency == currency
            ]
            if any(asset not in quotes or quotes[asset].close is None for asset, _ in positions):
                nav[currency] = None
            else:
                nav[currency] = cash + math.fsum(
                    quantity * quotes[asset].close for asset, quantity in positions
                )
                nav[currency] += math.fsum(
                    e["amount"] for e in state["receivables"] if e["currency"] == currency
                )
        state.update(
            nav=nav,
            last_close=close_at,
            quotes={asset: asdict(quote) for asset, quote in sorted(quotes.items())},
        )
        for asset, order in state["orders"].items():
            volume = quotes[asset].volume if asset in quotes else None
            order["capacity"] = volume * self.participation if volume is not None else None

    def advance(self, open_at, close_at, quotes, *, actions=()):
        self._quotes(quotes)
        if self.clock is None or not self.clock < instant(open_at) < instant(close_at):
            raise ValueError("La ejecución debe ser posterior al cierre de decisión")
        if len(actions) > 4096 or len(self._state["applied"]) + len(actions) > 65536:
            raise ValueError("Las acciones corporativas superan el presupuesto")
        ids = set(self._state["applied"])
        for action in actions:
            action.validate()
            if (
                action.asset not in self.instruments
                or action.effective_at != open_at
                or action.id in ids
            ):
                raise ValueError("La acción corporativa está duplicada o fuera de su sesión")
            ids.add(action.id)
        state = copy.deepcopy(self._state)
        self._actions(state, actions)
        self._settle(state, open_at)
        trades, unfilled = self._execute(state, quotes)
        self._mark(state, close_at, quotes)
        state["last_open"] = open_at
        self._validate(state)
        self._state = state
        return dict(
            nav=dict(state["nav"]),
            costs=dict(state["costs"]),
            turnover=dict(state["turnover"]),
            trades=trades,
            unfilled=unfilled,
            unvalued=[
                asset
                for asset in state["positions"]
                if asset not in quotes or quotes[asset].close is None
            ],
        )

    def _validate(self, state):
        if set(state) != set(self._state) or set(state["cash"]) != set(self._state["cash"]):
            raise ValueError("El estado no conserva sus cuentas")
        if any(set(state[key]) != set(state["cash"]) for key in ("nav", "costs", "turnover")):
            raise ValueError("La valoración no conserva las monedas de la cartera")
        close, opened = state["last_close"], state["last_open"]
        if close is not None:
            instant(close)
        if opened is not None and (close is None or not instant(opened) < close):
            raise ValueError("Los tiempos de la cartera no conservan su orden")
        quotes = {asset: Quote(**quote) for asset, quote in state["quotes"].items()}
        self._quotes(quotes)
        for key in ("cash", "positions", "costs", "turnover"):
            for value in state[key].values():
                amount(value)
        if not set(state["positions"]) <= set(self.instruments) or not set(state["orders"]) <= set(
            self.instruments
        ):
            raise ValueError("El estado contiene instrumentos desconocidos")
        if len(state["receivables"]) > 65536 or len(state["applied"]) > 65536:
            raise ValueError("El estado supera el presupuesto")
        if (
            len(set(state["applied"])) != len(state["applied"])
            or any(
                not isinstance(value, str) or not 1 <= len(value) <= 128
                for value in state["applied"]
            )
            or len(set(state["retired"])) != len(state["retired"])
            or not set(state["retired"]) <= set(self.instruments)
            or set(state["retired"]) & (set(state["positions"]) | set(state["orders"]))
        ):
            raise ValueError("Las acciones confirmadas o las bajas no son coherentes")
        for entry in state["receivables"]:
            amount(entry["amount"])
            instant(entry["pay_at"])
            if (
                entry["currency"] not in state["cash"]
                or close is None
                or entry["pay_at"] <= close
                or entry["id"] not in state["applied"]
            ):
                raise ValueError("El dividendo pertenece a otra moneda")
        for order in state["orders"].values():
            amount(order["target"])
            instant(order["decision_at"])
            if close is None or order["decision_at"] > close:
                raise ValueError("La orden guardada es posterior al cierre disponible")
            if order["capacity"] is not None:
                amount(order["capacity"])
        for value in state["nav"].values():
            if value is not None:
                amount(value)
        if close is None and (
            quotes or state["positions"] or state["orders"] or state["receivables"]
        ):
            raise ValueError("La cartera sin iniciar contiene operaciones")
        expected = copy.deepcopy(state)
        if close is not None:
            self._mark(expected, close, quotes)
        else:
            expected["nav"] = dict(state["cash"])
        if expected["nav"] != state["nav"]:
            raise ValueError("El patrimonio no corresponde al efectivo, posiciones y derechos")
        if expected["orders"] != state["orders"]:
            raise ValueError("La liquidez pendiente no corresponde al último cierre")

    def snapshot(self):
        state = copy.deepcopy(self._state)
        checksum = hashlib.sha256(
            json.dumps(state, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return dict(
            schema_version=1,
            identity=copy.deepcopy(self.identity),
            state=state,
            state_sha256=checksum,
        )

    def restore(self, snapshot):
        if snapshot.get("schema_version") != 1 or snapshot.get("identity") != self.identity:
            raise ValueError("La cartera pertenece a otra configuración")
        state = copy.deepcopy(snapshot["state"])
        checksum = hashlib.sha256(
            json.dumps(state, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if checksum != snapshot.get("state_sha256"):
            raise ValueError("La cartera guardada está corrupta")
        self._validate(state)
        self._state = state
