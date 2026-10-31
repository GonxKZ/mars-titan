"""Publicar el libro S2 en la misma generación que las predicciones financieras."""

import hashlib
import inspect
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.holographic_network import CATALOG

from . import failure_context, failure_ledger, failure_ledger_state, failure_producer
from .failure_ledger import FailureLedger, FailureLedgerConfig
from .failure_producer import FailureContextProducer


@dataclass(frozen=True)
class FailureSessionConfig:
    ledger: FailureLedgerConfig
    macro_catalog_sha256: str
    catalog: Path = CATALOG
    mode: str = "confidence_only"
    constant_confidence: float = 0.75

    def __post_init__(self):
        if type(self.ledger) is not FailureLedgerConfig or self.mode not in (
            "confidence_only",
            "shrink_point",
            "constant_shrink",
        ):
            raise ValueError("S2 necesita un libro identificado y un modo de consumo admitido")
        failure_context._digest_id(self.macro_catalog_sha256, "El catálogo S2")
        if (
            type(self.constant_confidence) not in (float, int)
            or not math.isfinite(self.constant_confidence)
            or not 0 <= self.constant_confidence <= 1
        ):
            raise ValueError("La confianza constante debe ser finita y estar entre cero y uno")
        object.__setattr__(self, "catalog", Path(self.catalog))

    def identity(self):
        return dict(
            schema_version=1,
            kind="financial_failure_memory_v1",
            ledger=self.ledger.identity(),
            macro_catalog_sha256=self.macro_catalog_sha256,
            mode=self.mode,
            constant_confidence=self.constant_confidence,
            action="fixed_predictor/no_external_tool",
            source_scope="frozen_consumer_identity_with_declared_fit_boundary",
        )


def configure(config, binding, phase, *, lesions, spaced):
    if config is None:
        return None
    if type(config) is not FailureSessionConfig:
        raise ValueError("La sesión S2 necesita FailureSessionConfig")
    if (
        config.ledger.source_manifest_id != binding.consumer.model_id
        or max(config.ledger.source_fitted_until, config.ledger.threshold_fitted_until)
        >= phase.decision_start
    ):
        raise ValueError(
            "La procedencia S2 no corresponde al consumidor o supera el pasado permitido"
        )
    if config.mode != "confidence_only" and (lesions or spaced):
        raise ValueError("La atenuación S2 con B4 o E2 necesita una composición propia")
    return FailureContextProducer(
        binding, macro_catalog_sha256=config.macro_catalog_sha256, catalog=config.catalog
    )


def signature(session):
    return canonical(
        session._failure_config.identity()
    ), session._failure_producer.layout.fingerprint()


def code():
    modules = (
        inspect.getmodule(code),
        failure_context,
        failure_ledger,
        failure_ledger_state,
        failure_producer,
        inspect.getmodule(failure_producer.HolographicNetwork),
    )
    return {
        m.__name__.rsplit(".", 1)[-1]: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in modules
    }


def restore(session, bundle):
    producer, config = session._failure_producer, session._failure_config.ledger
    if bundle is None:
        return FailureLedger(producer.layout, config)
    payload = session._read(bundle["failure_memory"], "failure_memory")
    if (
        type(payload) is not dict
        or set(payload) != {"encoding", "state"}
        or payload["encoding"] != "context_uint8_tensors_v1"
    ):
        raise ValueError("El artefacto S2 no conserva su codificación")
    state = payload["state"]
    if (
        type(state) is not dict
        or type(state.get("emissions")) is not list
        or len(state["emissions"]) > config.max_pending + 2 * config.capacity
    ):
        raise ValueError("El estado S2 no conserva una lista acotada de emisiones")
    sizes = dict(
        key_inputs=producer.layout.key_width * 8,
        macro=len(producer.layout.macro_indicators) * 12,
        presence=5,
        macro_available_at=len(producer.layout.macro_indicators) * 8,
        area_activity=len(producer.layout.area_names),
    )
    for emission in state["emissions"]:
        if (
            type(emission) is not dict
            or set(emission) != {"context", "prediction"}
            or type(emission["context"]) is not dict
            or not set(failure_context._BUFFERS) <= set(emission["context"])
        ):
            raise ValueError("La emisión S2 serializada no conserva el contexto esperado")
        context = emission["context"]
        for name in failure_context._BUFFERS:
            tensor = context[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.uint8
                or tensor.ndim != 1
                or tensor.layout != torch.strided
                or not tensor.is_contiguous()
                or tensor.numel() != sizes[name]
            ):
                raise ValueError("El buffer S2 no conserva su representación de bytes CPU")
            context[name] = tensor.numpy().tobytes()
    return FailureLedger.restore(producer.layout, config, state)


def stage(session, ledger):
    state = ledger.export()
    for emission in state["emissions"]:
        context = emission["context"]
        for name in failure_context._BUFFERS:
            context[name] = torch.from_numpy(np.frombuffer(context[name], dtype=np.uint8).copy())
    return session._stage(dict(encoding="context_uint8_tensors_v1", state=state), "failure_memory")


def prepare(session, bundle, rows, values, cutoff, *, warmup):
    ledger = restore(session, bundle)
    if warmup:
        return values, stage(session, ledger), dict(mode="warmup", emitted=0)
    ledger.begin(cutoff)
    contexts = [
        session._failure_producer.capture(row, action="fixed_predictor/no_external_tool")
        for row in rows
    ]
    scores = [ledger.query(context) for context in contexts]
    config = session._failure_config
    confidence = [
        config.constant_confidence if config.mode == "constant_shrink" else score.confidence
        for score in scores
    ]
    emitted = (
        values
        if config.mode == "confidence_only"
        else [point * weight for point, weight in zip(values, confidence, strict=True)]
    )
    ledger.issue(list(zip(contexts, emitted, strict=True)))
    control = dict(
        mode=config.mode,
        source_points=list(values),
        emitted=len(emitted),
        risks=[s.risk for s in scores],
        confidence=confidence,
        eligible=[s.eligible for s in scores],
        contexts=[c.context_id for c in contexts],
    )
    return emitted, stage(session, ledger), control


def settlement(session, bundle, cutoff):
    ledger = restore(session, bundle)
    if cutoff >= session.phase.decision_start:
        ledger.begin(cutoff, settlement=True)
    return stage(session, ledger)


def resolve(session, proposed, outcomes, excluded, finalized):
    ledger = restore(session, proposed)
    if ledger.clock is None:
        if outcomes or excluded or finalized:
            raise ValueError("El calentamiento S2 no puede resolver predicciones no emitidas")
        return stage(session, ledger)
    records = []
    for item in (*outcomes, *excluded, *finalized):
        key = item.prediction.asset, item.prediction.decision_at
        if key not in ledger._pending or ledger._pending[key].prediction != item.prediction.value:
            raise ValueError("El pendiente S2 no coincide con la predicción realmente emitida")
    for item in outcomes:
        records.append(
            (
                item.prediction.asset,
                item.prediction.decision_at,
                item.label.available_at,
                item.label.value,
            )
        )
    ledger.resolve(records)
    ledger.censor(
        [(item.prediction.asset, item.prediction.decision_at) for item in (*excluded, *finalized)]
    )
    return stage(session, ledger)


def verify(session, bundle, pending_keys, *, event_kind):
    ledger = restore(session, bundle)
    warming = event_kind == "warmup" and ledger.clock is None
    if (not warming and ledger.clock != bundle["cutoff"]) or set(ledger._pending) != set(
        pending_keys
    ):
        raise ValueError("El libro S2 no concilia con el corte y la cola financiera")


def confirmed_book(session):
    """Obtener una copia del libro confirmado, sin exponer el estado vivo del ejecutor."""
    from .financial_session import FinancialSession, _check_callbacks

    if type(session) is not FinancialSession or session._failure_config is None:
        raise ValueError("La lectura analítica necesita una sesión S2 activa")
    _check_callbacks(session)
    before = session.snapshot()
    session._verify_confirmed()
    ledger = restore(session, session._bundle(before["state"]))
    if session.snapshot() != before:
        raise ValueError("La sesión cambió durante la lectura del libro confirmado")
    return ledger
