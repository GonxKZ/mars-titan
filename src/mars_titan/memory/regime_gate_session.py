"""Publicar la puerta E4 junto a la memoria A en cada generación de `FinancialSession` (#750).

Cada generación nativa confirmada es una publicación E4, también la que no madura ningún
resultado. La cohorte son los resultados que el ejecutor acepta para predicciones pendientes
ya emitidas, comparados con el punto final emitido. La puerta de cada mercado regula la tasa
de las escrituras de A en esa misma generación y su estado se confirma con ella. La sesión
empieza sin estado, igual que cada fase del recorrido por ventanas.
"""

import hashlib
import inspect
from pathlib import Path

from . import regime_evidence_gate
from .regime_evidence_gate import KIND, RegimeEvidenceGate, check_correction


def configure(config, correction):
    return None if config is None else check_correction(config, correction)


def contract(config):
    return dict(
        kind=KIND,
        configuration=config.identity(),
        publication="one_confirmed_native_generation_including_empty_events",
        cohort="unique_mature_outcomes_of_native_pending_issued_predictions",
        error="target_minus_final_issued_prediction",
        target="rate_of_the_proximal_writes_of_a_in_the_same_generation",
        reset="empty_state_at_the_start_of_the_session",
    )


def code():
    return {
        module.__name__.rsplit(".", 1)[-1]: hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in (inspect.getmodule(code), regime_evidence_gate)
    }


def restore(session, bundle):
    reference = bundle["regime_gate"] if bundle is not None else None
    if reference is None:
        return RegimeEvidenceGate(session._regime_gate_config)
    return RegimeEvidenceGate.restore(
        session._regime_gate_config, session._read(reference, "regime_gate")
    )


def resolve(session, proposed, outcomes):
    """Referencia del estado siguiente y puerta de cada mercado de la generación propuesta."""
    cohort = tuple(
        (
            item.prediction.asset,
            item.prediction.decision_at,
            item.label.available_at,
            item.prediction.value,
            item.label.value,
        )
        for item in outcomes
    )
    decision, following = restore(session, proposed).step(cohort, cutoff=proposed["cutoff"])
    return session._stage(following.export(), "regime_gate"), decision.by_market()


def verify(session, bundle):
    state = restore(session, bundle)
    if state.publications != bundle["generation"] or state.cursor != bundle["cutoff"]:
        raise ValueError("La publicación E4 no concilia con la generación confirmada")
    return state


def verify_confirmed(session, bundle, snapshot):
    if restore(session, bundle).observed != snapshot["applied"]:
        raise ValueError("Los errores E4 no concilian con los resultados nativos aplicados")
