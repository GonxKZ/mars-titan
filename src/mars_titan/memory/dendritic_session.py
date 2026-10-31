"""Asignación N2 común al evento y traza confirmada con la generación financiera."""

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.dendritic_context import DendriticPlan
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer


def enabled(session):
    return (
        type(session.consumer) in (FrozenFinancialConsumer, FrozenMechanismConsumer)
        and session.consumer.predictor.active_dendrites is not None
    )


def prepare(session, rows, context_id):
    model = session.consumer.predictor
    return DendriticPlan.create(
        model.config.active_dendrites,
        rows,
        seed=model.config.seed,
        input_contract=session._input_spec.fingerprint(),
        context_id=context_id,
    )


def verify(session, control, rows, cutoff):
    record = control.get("dendritic_context")
    if not enabled(session) or not rows:
        if "dendritic_context" in control and (not enabled(session) or record is not None):
            raise ValueError("Este evento no admite una traza de contexto N2")
        return
    if any(row.prediction_at != cutoff for row in rows):
        raise ValueError("La fuente N2 no corresponde al corte de la generación")
    expected = prepare(session, rows, control.get("context_id")).record()
    if canonical(record) != canonical(expected):
        raise ValueError("La traza N2 no coincide con el contexto del evento completo")
