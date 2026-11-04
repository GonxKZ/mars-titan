"""Confirmar el cupo E3 junto a la memoria y el registro nativo de resultados."""

import hashlib
import inspect
from pathlib import Path

from . import salience_budget
from .associative_memory import MatureCorrection
from .salience_budget import CommittedWriteBudget, MatureWrite, SalienceBudgetConfig


def configure(config, correction):
    if config is not None and (
        type(config) is not SalienceBudgetConfig
        or type(correction) is not MatureCorrection
        or correction.memory.rule not in ("delta", "proximal")
    ):
        raise ValueError("E3 necesita su configuración y una corrección delta/proximal")
    return config


def contract(config):
    return dict(
        kind=CommittedWriteBudget.KIND,
        configuration=config.identity(),
        publication="one_confirmed_native_generation_including_empty_events",
        eligibility="unique_mature_outcomes_of_native_pending_issued_predictions",
        statistics="all_outcomes_after_scoring_with_the_prior",
        write_value="target_minus_core_prediction",
        priority_error="target_minus_final_issued_prediction",
        calendar_session_quota=False,
    )


def code():
    return {
        module.__name__.rsplit(".", 1)[-1]: hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in (inspect.getmodule(code), salience_budget)
    }


def restore(session, bundle):
    reference = bundle["salience"] if bundle is not None else None
    if reference is None:
        return CommittedWriteBudget(session._salience_config)
    return CommittedWriteBudget.restore(
        session._salience_config, session._read(reference, "salience")
    )


def resolve(session, proposed, outcomes):
    rows = tuple(
        MatureWrite(
            item.prediction.asset,
            item.prediction.decision_at,
            item.label.available_at,
            item.prediction.value,
            item.label.value,
        )
        for item in outcomes
    )
    selection, following = restore(session, proposed).select(
        rows, cutoff=proposed["cutoff"], generation=proposed["generation"]
    )
    chosen = set(selection.selected)
    indices = [index for index, row in enumerate(rows) if row.key in chosen]
    return session._stage(following.export(), "salience"), indices


def verify(session, bundle, memory):
    state = restore(session, bundle)
    if (
        state.generation != bundle["generation"]
        or state.cutoff != bundle["cutoff"]
        or state.statistics.selected != memory.writes
    ):
        raise ValueError("La publicación E3 no concilia con la generación y sus escrituras")
    return state


def verify_confirmed(session, bundle, snapshot, record):
    state = restore(session, bundle)
    keys = {
        (item["prediction"]["asset"], item["prediction"]["decision_at"])
        for item in record["feedback"]
    }
    if (
        state.statistics.observed != snapshot["applied"]
        or state.last_observed != len(keys)
        or not set(state.last_selected) <= keys
    ):
        raise ValueError("Las estadísticas E3 no concilian con el registro nativo de resultados")
