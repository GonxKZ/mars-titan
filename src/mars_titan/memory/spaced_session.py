"""Enlace cerrado E2 al coordinador financiero y sus generaciones durables."""

import hashlib
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

from mars_titan.models.titans.frozen_financial import FrozenPreparation

from . import (
    spaced_regional_readout,
    spaced_regional_session,
    spaced_retention,
    spaced_retention_state,
    spaced_utility,
    spaced_utility_readout,
    spaced_utility_state,
)
from .financial_consumers import TitansBinding, mature_record
from .retention_bank import RetentionConfig
from .spaced_retention import SpacedUtilityBank
from .spaced_utility import (
    SpacedUtilityConfig,
    SpacedUtilityLedger,
    _flow,
    _integer,
    _number,
    _require,
    _rows,
)
from .spaced_utility_readout import apply_spaced_utility_readout
from .spaced_utility_state import _fields


@dataclass(frozen=True)
class SpacedPreparation(FrozenPreparation):
    utility: object = None


class SpacedTitansBinding(TitansBinding):
    """Mismo recorrido por bloques, con lectura K1 y retención E2 explícitas."""

    def __init__(self, consumer, codec, retention, admission):
        _require(admission == "m1", "E2 solo admite M1 y conserva separados los índices M2/M3")
        reference = RetentionConfig(
            capacity=retention.capacity, seed=retention.seed, frontier=min(retention.capacity, 8)
        )
        super().__init__(consumer, codec, reference, admission)
        control = consumer.predictor.local_control
        _require(
            control is None or control.config.mode == "disabled",
            "La sesión E2 requiere C desactivado para fijar su contraste",
        )
        readout = consumer.readout
        _require(
            readout is not None
            and readout.config.mode == "bank"
            and readout.config.refinements == 1,
            "La sesión E2 necesita un lector bank K1",
        )
        self.spaced_config = retention
        self.evidence_config = SpacedUtilityConfig(
            capacity=retention.capacity,
            max_pending_sessions=40960,
            max_references=40960 * readout.config.neighbors,
        )

    def signature(self):
        base = asdict(self.spaced_config), self.evidence_config.identity()
        return (
            base + (spaced_regional_session.signature(self.consumer.readout),)
            if self.regional()
            else base
        )

    def regional(self):
        return self.consumer.readout.config.evidence_bridge is not None

    def code(self):
        modules = (
            inspect.getmodule(begin),
            spaced_retention,
            spaced_retention_state,
            spaced_utility,
            spaced_utility_state,
            spaced_utility_readout,
        )
        if self.regional():
            modules += (spaced_regional_session, spaced_regional_readout)
        return {
            m.__name__.rsplit(".", 1)[-1]: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in modules
        }

    def prepare_event(self, session, rows, fast, snapshot, *, context_id, warmup, batch_rows):
        work = None
        if self.regional():
            work = spaced_regional_session.preflight(
                self, rows, snapshot, context_id=context_id, warmup=warmup, block_rows=batch_rows
            )
        values, updated, control = super().prepare_event(
            session,
            rows,
            fast,
            snapshot,
            context_id=context_id,
            warmup=warmup,
            batch_rows=batch_rows,
        )
        if work is not None:
            control["spaced_regional"] = work
        return values, updated, control

    def bank(self, native, retention, **scope):
        return SpacedUtilityBank(
            native,
            retention,
            model_id=self.consumer.model_id,
            evidence_config=self.evidence_config,
            **scope,
        )

    def prepare_block(self, session, batch, previous, *, context_id, snapshot, warmup, selection):
        if warmup:
            return super().prepare_block(
                session,
                batch,
                previous,
                context_id=context_id,
                snapshot=snapshot,
                warmup=True,
                selection=selection,
            )
        self.consumer.verify(strong=False)
        model = self.consumer.predictor
        options = (
            {}
            if model.local_control is None
            else dict(control_selection=selection, control_context_id=context_id)
        )
        prepared = model.prepare(batch, previous, differentiable=False, **options)
        producer = (
            spaced_regional_readout.apply_regional_spaced_utility_readout
            if self.regional()
            else apply_spaced_utility_readout
        )
        prediction, trace = producer(
            prepared,
            model.head,
            self.consumer.readout,
            snapshot,
            enabled=True,
            context_id=context_id,
            cutoff=batch.prediction_at[0],
            allow_mixed_markets=True,
        )
        return SpacedPreparation(
            point_predictions=prediction.point_predictions,
            next_state=prepared.next_state,
            readout=prediction.readout,
            local_control=prepared.local_control,
            quantiles=prediction.quantiles,
            utility=trace,
        )

    def record_block(self, session, batch, result, *, warmup):
        if warmup:
            return
        _require(type(result) is SpacedPreparation, "El bloque E2 no contiene sus diagnósticos")
        dates = {
            flow: session.prefixes.expected_maturity(flow, at)
            for flow, at in zip(batch.flow_ids, batch.prediction_at, strict=True)
        }
        session._inflight["spaced"]["emissions"].extend(result.utility.ledger_rows(dates))


def enabled(session):
    return type(session._binding) is SpacedTitansBinding


def _restore(session, bank, bundle):
    if bundle is None:
        return SpacedUtilityLedger(
            session.codec.fingerprint(),
            session._binding.evidence_config,
            scope_id=bank.fingerprint(),
        ), dict(emissions=[], receipts=[])
    payload = session._read(bundle["spaced"], "spaced")
    _fields(payload, ("ledger", "event"), "El artefacto E2 no conserva sus campos")
    ledger = SpacedUtilityLedger.restore(
        session.codec.fingerprint(),
        session._binding.evidence_config,
        payload["ledger"],
        scope_id=bank.fingerprint(),
    )
    return ledger, payload["event"]


def begin(session, bank, bundle, cutoff, *, feedback_only=False):
    ledger, _ = _restore(session, bank, bundle)
    if bundle is not None:
        bank.validate_evidence(ledger)
    ledger.begin(cutoff, feedback_only=feedback_only)
    session._inflight["spaced"] = dict(ledger=ledger, emissions=[])


def issue(session, snapshot, values, rows, *, warmup):
    active = session._inflight["spaced"]
    if warmup:
        _require(not active["emissions"], "El calentamiento E2 no emite diagnósticos")
        return
    emissions = active["emissions"]
    _require(
        len(emissions) == len(rows)
        and all(
            e["flow_id"] == row.flow_id and e["point"] == value
            for e, row, value in zip(emissions, rows, values, strict=True)
        ),
        "Los diagnósticos E2 no corresponden a las predicciones preparadas",
    )
    groups = {}
    for row in emissions:
        groups.setdefault(_flow(row["flow_id"]), []).append(row)
    for market in sorted(groups):
        active["ledger"].issue(snapshot, groups[market])


def resolve(session, bank, pending, admitted, proposed, outcomes, excluded, finalized):
    active = session._inflight["spaced"]
    ledger = active["ledger"]
    targets = []
    for item in (*outcomes, *excluded, *finalized):
        prediction = item.prediction
        row = ledger._pending_row(prediction.asset, prediction.decision_at)
        _require(
            row["point"] == prediction.value, "El punto E2 no es la predicción realmente emitida"
        )
    for item in outcomes:
        targets.append(
            (
                item.prediction.asset,
                item.prediction.decision_at,
                item.label.available_at,
                item.label.value,
            )
        )
    ledger.receive(targets)
    for values, reason in ((excluded, "excluded"), (finalized, "finalized")):
        ledger.censor(
            [(i.prediction.asset, i.prediction.decision_at) for i in values], reason=reason
        )
    receipts = ledger.publish()
    if admitted:
        incoming = [
            mature_record(
                session.native,
                identifier=identifier,
                decision_at=item.prediction.decision_at,
                available_at=pending["rows"][index]["input_available_at"],
                maturity_at=item.label.available_at,
                key=pending["key_inputs"][index].tolist(),
                value=pending["values"][index].tolist(),
                label=item.label.value,
            )
            for identifier, index, item in admitted
        ]
        proposal = bank.propose(incoming, confirmed_at=proposed["cutoff"], evidence=ledger)
        bank, ledger = proposal.bank, proposal.evidence
    bank.validate_evidence(ledger)
    reference = session._stage(
        dict(ledger=ledger.export(), event=dict(emissions=active["emissions"], receipts=receipts)),
        "spaced",
    )
    return bank, reference


def _event(value, cutoff, config):
    _fields(value, ("emissions", "receipts"), "El evento E2 no conserva su esquema")
    rows, receipts = value["emissions"], value["receipts"]
    _require(
        type(rows) is list
        and len(rows) <= 8192
        and type(receipts) is list
        and len(receipts) <= config.max_pending_sessions,
        "El evento E2 excede sus límites",
    )
    groups, seen = {}, set()
    for row in rows:
        _fields(
            row,
            ("flow_id", "point", "deleted_points", "label_not_before"),
            "La emisión E2 no conserva sus campos",
        )
        _require(
            type(row["flow_id"]) is str
            and isinstance(row["deleted_points"], (tuple, list))
            and len(row["deleted_points"]) <= 8,
            "La emisión E2 necesita flujo y soporte acotados",
        )
        flow = row["flow_id"]
        _require(flow not in seen, "El evento E2 repite emisiones")
        seen.add(flow)
        groups.setdefault(_flow(flow), []).append(row)
    for group in groups.values():
        ids = {
            p[0]
            for r in group
            for p in r.get("deleted_points", [])
            if type(p) in (list, tuple) and len(p) == 2 and _integer(p[0], 1)
        }
        _rows(group, cutoff, ids)
    count = 0
    for receipt in receipts:
        _fields(
            receipt,
            ("market", "decision_at", "published_at", "gains", "resolved", "censored"),
            "El recibo de utilidad E2 no conserva su esquema",
        )
        _require(
            receipt["market"] in ("US", "CN")
            and _integer(receipt["decision_at"])
            and receipt["decision_at"] <= cutoff
            and type(receipt["published_at"]) is int
            and receipt["published_at"] == cutoff
            and all(_integer(receipt[k], 0, 8192) for k in ("resolved", "censored"))
            and type(receipt["gains"]) is dict
            and len(receipt["gains"]) <= config.capacity,
            "El recibo E2 no corresponde a su corte",
        )
        count += len(receipt["gains"])
        _require(count <= config.max_references, "Los recibos E2 exceden sus referencias")
        for key, item in receipt["gains"].items():
            _require(_integer(key, 1), "El recibo E2 necesita un episodio válido")
            _fields(item, ("mean", "rows"), "La ganancia E2 tiene otro esquema")
            _number(item["mean"])
            _require(_integer(item["rows"], 1, 8192), "La ganancia E2 necesita filas reales")


def verify(session, bundle, pending_keys):
    bank, _ = session._bank(bundle["bank"])
    ledger, event = _restore(session, bank, bundle)
    bank.validate_evidence(ledger)
    unresolved = {
        (flow, at)
        for (_, at), c in ledger._sessions.items()
        for flow, row in c["rows"].items()
        if row["status"] is None
    }
    _require(
        ledger.clock == bundle["cutoff"] and unresolved == set(pending_keys),
        "El ledger E2 no concilia con el corte y pendientes financieros",
    )
    _event(event, bundle["cutoff"], ledger.config)
    if session._binding.regional():
        spaced_regional_session.verify(session, bundle, event)


def verify_confirmed(session, bundle, record):
    bank, _ = session._bank(bundle["bank"])
    ledger, event = _restore(session, bank, bundle)
    if session._binding.regional():
        _require(type(record.get("event_kind")) is str, "El tipo nativo regional no es válido")
        spaced_regional_session.verify(session, bundle, event, event_kind=record["event_kind"])
    actual = sorted((p["asset"], p["value"]) for p in record["predictions"])
    emitted = sorted((r["flow_id"], r["point"]) for r in event["emissions"])
    _require(actual == emitted, "El evento E2 no coincide con las predicciones nativas")
    for prediction in session._executor.pending():
        row = ledger._pending_row(prediction.asset, prediction.decision_at)
        _require(row["point"] == prediction.value, "El pendiente E2 cambió su predicción emitida")


def inspect_state(session, *, event=False):
    _require(enabled(session), "La sesión no tiene E2 activado")
    bundle = session._bundle(session.snapshot()["state"])
    if bundle is None:
        bank = session._prototype
        ledger, info = _restore(session, bank, None)
    else:
        bank, _ = session._bank(bundle["bank"])
        ledger, info = _restore(session, bank, bundle)
    return info if event else ledger.export()
