"""E2 en una fase de ventana con padre y lector congelados, sin ajustes de parámetros."""

from dataclasses import replace

import torch

from mars_titan.memory import spaced_regional_readout
from mars_titan.memory.episodic_session import _digest
from mars_titan.memory.financial_consumers import mature_record
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.memory.spaced_session import SpacedTitansBinding
from mars_titan.memory.spaced_utility import SpacedUtilityLedger, _flow, _require
from mars_titan.memory.spaced_utility_readout import apply_spaced_utility_readout
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

from . import spaced_regional_phase
from .financial_run import Paused
from .learning_hold import require_learning_allowed
from .mars_titan_run import MarsTitanInference
from .prefix_eligibility import PrefixTargetVerifier


class SpacedInference(MarsTitanInference):
    """Reutilizar la inferencia del lector y los mismos componentes E2 de la sesión durable.

    Cada fase empieza con estados vacíos. Una interrupción obliga a repetirla desde su
    principio. La recuperación por evento pertenece a FinancialSession. Este contraste
    cambia la retención de un lector ya seleccionado, sin reentrenar cada variante.
    """

    def __init__(
        self,
        consumer,
        recipe,
        *,
        prefixes,
        retention,
        native,
        codec,
        world,
        fold,
        audit=False,
        reader_contract="standard",
    ):
        _require(
            type(consumer) is FrozenFinancialConsumer
            and type(retention) is SpacedRetentionConfig
            and type(prefixes) is PrefixTargetVerifier,
            "La ventana E2 necesita consumidor congelado, retención y prefijos identificados",
        )
        consumer.verify()
        _require(
            type(reader_contract) is str and reader_contract in ("standard", "regional"),
            "La ventana E2 necesita un contrato de lector identificado",
        )
        _require(
            consumer.readout is not None
            and (consumer.readout.config.evidence_bridge is not None)
            == (reader_contract == "regional"),
            "El contrato de la ventana E2 no corresponde al lector estándar o regional",
        )
        binding = SpacedTitansBinding(consumer, codec, retention, "m1")
        _require(
            prefixes.source_id == consumer.predictor.config.inputs.source_sha256,
            "Los prefijos E2 no pertenecen al corpus del padre",
        )
        super().__init__(
            consumer.predictor,
            consumer.readout,
            recipe,
            admission="m1",
            retention=RetentionConfig(
                capacity=retention.capacity,
                seed=retention.seed,
                frontier=min(retention.capacity, 8),
            ),
            native=native,
            codec=codec,
            world=world,
            fold=fold,
            audit=audit,
        )
        self.consumer, self.prefixes, self.retention = consumer, prefixes, retention
        self.binding, self._ledger, self._emissions = binding, None, []
        self._snapshot, self._last = None, None
        self._block_state = None
        self._last_bank = None
        self.reader_contract = reader_contract
        self._regional_work, self._regional_event_work = None, None
        self._regional_episodes = None

    def _event_plan(self, run, phase, event, batches, *, train):
        if self.reader_contract == "regional" and event.at >= phase.decision_start:
            _require(train is False, "La fase regional E2 no ajusta parámetros")
            episodes = self._ledger.usage()["episodes"]
            self._regional_event_work = spaced_regional_phase.plan(
                self.readout, batches, episodes, cutoff=event.at
            )
            self._regional_episodes = episodes
        return super()._event_plan(run, phase, event, batches, train=train)

    def _new_bank(self, partition):
        return self.binding.bank(
            self.native,
            self.retention,
            codec_id=self.codec.fingerprint(),
            world=_digest([self.world, "residual", 1]),
            partition=partition,
            fold=self.fold,
        )

    def _prepare_block(self, run, batch, plan, *, train):
        prepared = super()._prepare_block(run, batch, plan, train=train)
        # La ruta común elimina next_state antes del lector. E2 necesita sus identidades
        # durante la supresión. La referencia dura un bloque y no copia los pesos rápidos.
        self._block_state = prepared.next_state
        return prepared

    def _predict_block(self, prepared, batch, phase, snapshot, context):
        if self._snapshot is None:
            self._snapshot = snapshot
        _require(snapshot is self._snapshot, "Los bloques E2 deben compartir snapshot")
        regional = self.reader_contract == "regional"
        if regional:
            _require(
                self._regional_event_work is not None and snapshot.count == self._regional_episodes,
                "El snapshot regional no corresponde al presupuesto anterior a MAC",
            )
        producer = (
            spaced_regional_readout.apply_regional_spaced_utility_readout
            if regional
            else apply_spaced_utility_readout
        )
        result, trace = producer(
            replace(prepared, next_state=self._block_state),
            self.predictor.head,
            self.readout,
            snapshot,
            enabled=True,
            context_id=context,
            cutoff=batch.prediction_at[0],
            allow_mixed_markets=True,
        )
        dates = {
            flow: self.prefixes.expected_maturity(flow, at)
            for flow, at in zip(batch.flow_ids, batch.prediction_at, strict=True)
        }
        rows = trace.ledger_rows(dates)
        _require(len(self._emissions) + len(rows) <= 8192, "El evento E2 excede 8192 emisiones")
        self._emissions.extend(rows)
        self._block_state = None
        return result

    def _issue(self, run, at):
        groups = {}
        for row in self._emissions:
            entry = run.pending.get((row["flow_id"], at))
            _require(
                entry is not None and entry.issued == row["point"],
                "El diagnóstico E2 no corresponde al punto realmente emitido",
            )
            groups.setdefault(_flow(row["flow_id"]), []).append(row)
        for market in sorted(groups):
            self._ledger.issue(self._snapshot, groups[market])
        if self._regional_event_work is not None:
            spaced_regional_phase.accumulate(self._regional_work, self._regional_event_work)

    def _admit(self, run, phase, at):
        if not run.staged:
            return
        staged, run.staged = sorted(run.staged, key=lambda item: item[:2]), []
        records = [
            mature_record(
                self.native,
                identifier=run.bank.seen + 1 + offset,
                decision_at=decision_at,
                available_at=entry.available,
                maturity_at=at,
                key=entry.key.tolist(),
                value=entry.value.tolist(),
                label=target,
            )
            for offset, (decision_at, _, entry, target) in enumerate(staged)
        ]
        proposed = run.bank.propose(records, confirmed_at=at, evidence=self._ledger)
        run.bank, self._ledger = proposed.bank, proposed.evidence
        run.counters["admitted"] += len(records)
        if self.audit is not None:
            ids = tuple(record.id for record in records)
            self.audit.append(("admit", phase.partition, at, ids, run.bank.seen))

    def _finish_event(self, run, phase, event, rows):
        targets = []
        for flow, at, value in event.labels:
            entry = run.pending.get((flow, at))
            pending = self._ledger._pending_row(flow, at)
            _require(
                entry is not None and pending["point"] == entry.issued,
                "La etiqueta E2 no corresponde a su predicción emitida",
            )
            targets.append((flow, at, event.at, value))
        self._ledger.receive(targets)
        self._labels(run, phase, event, train=False, rows=rows)
        if event.close_phase:
            self._ledger.censor(list(run.pending), reason="finalized")
        self._ledger.publish()
        self._admit(run, phase, event.at)
        run.bank.validate_evidence(self._ledger)
        if event.close_phase:
            self._close(run)

    def _pass(self, phase, events, *, stop=None, rows=None):
        _require(type(phase) is FinancialPhase, "La ventana E2 necesita una fase identificada")
        self.consumer.verify()
        self._last = None
        self._last_bank = None
        self._regional_work = (
            spaced_regional_phase.empty_totals() if self.reader_contract == "regional" else None
        )
        run = self._new_pass(phase.partition)
        self._ledger = SpacedUtilityLedger(
            self.codec.fingerprint(), self.binding.evidence_config, scope_id=run.bank.fingerprint()
        )
        closed = False
        try:
            with torch.no_grad():
                for event in events:
                    if stop is not None and stop.requested:
                        raise Paused
                    _require(not closed, "La ventana E2 ya ha cerrado su fase")
                    self._ledger.begin(event.at, feedback_only=not event.inputs)
                    self._emissions, self._snapshot = [], None
                    self._regional_event_work, self._regional_episodes = None, None
                    if event.inputs:
                        self._observe(run, phase, event, train=False)
                    self._issue(run, event.at)
                    self._finish_event(run, phase, event, rows)
                    closed = event.close_phase
            _require(closed, "La ventana E2 necesita su cierre explícito")
            _require(run.counters["labels"] > 0, "La fase no contiene etiquetas maduras")
            result = self._metrics(run)
            result["spaced"] = dict(
                issued=self._ledger.issued,
                resolved=self._ledger.resolved,
                censored=self._ledger.censored,
                completed_sessions=self._ledger.completed_sessions,
                qualified=sum(score["qualified"] for score in self._ledger.scores().values()),
                **self._ledger.usage(),
            )
            if self._regional_work is not None:
                result["spaced_regional_work"] = dict(self._regional_work)
            self._last = self._ledger.export()
            self._last_bank = run.bank
            return result
        finally:
            self._ledger, self._emissions, self._snapshot = None, [], None
            self._block_state = None
            self._regional_work, self._regional_event_work, self._regional_episodes = (
                None,
                None,
                None,
            )

    def spaced_state(self):
        """Copiar la evidencia de la última fase completada. No es un checkpoint de reanudación."""
        from mars_titan.memory.spaced_utility import _copy

        _require(self._last is not None, "Todavía no hay una fase E2 completada")
        return _copy(self._last)

    def evaluate(self, source, *, stop=None, rows=None):
        require_learning_allowed("la evaluación de una ventana congelada E2")
        return super().evaluate(source, stop=stop, rows=rows)

    def spaced_snapshot(self):
        """Banco y evidencia finales para comparación. No incluye cursor ni pesos del padre."""
        ledger = self.spaced_state()
        return dict(
            schema_version=1, resumable=False, ledger=ledger, bank=self._last_bank.snapshot()
        )
