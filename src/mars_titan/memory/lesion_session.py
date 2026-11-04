"""Diagnósticos B4 confirmados junto al bundle financiero, sin cambiar sus emisiones."""

import hashlib
import inspect
import json
from pathlib import Path

from .lesion_ledger import LesionLedger
from .lesion_protocol import LesionProtocol


def configure(protocol, correction):
    if protocol is None:
        return None
    if type(protocol) is not LesionProtocol:
        raise ValueError("La sesión B4 necesita su protocolo identificado")
    return protocol.bind(correction)


def signature(plan):
    return (
        None
        if plan is None
        else (json.dumps(plan.identity(), sort_keys=True, allow_nan=False), plan.names)
    )


def code():
    modules = (
        inspect.getmodule(code),
        inspect.getmodule(LesionLedger),
        inspect.getmodule(LesionProtocol),
    )
    return {
        module.__name__.rsplit(".", 1)[-1]: hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
        for module in modules
    }


def restore(session, bundle):
    if bundle is None:
        return LesionLedger(session._lesion_plan)
    return LesionLedger.restore(session._lesion_plan, session._read(bundle["lesions"], "lesions"))


def begin(session, bundle, cutoff):
    ledger = restore(session, bundle)
    ledger.begin(cutoff)
    return ledger


def stage(session, ledger):
    return session._stage(ledger.export(), "lesions")


def resolve(session, proposed, outcomes, excluded, finalized):
    ledger = restore(session, proposed)
    records = []
    for item in outcomes:
        key = item.prediction.asset, item.prediction.decision_at
        if key not in ledger.pending or float(ledger.pending[key][0]) != item.prediction.value:
            raise ValueError("El control intacto B4 no coincide con la predicción emitida")
        records.append((*key, item.label.available_at, item.label.value))
    ledger.receive(records)
    for values, reason in ((excluded, "excluded"), (finalized, "finalized")):
        ledger.discard(
            [(item.prediction.asset, item.prediction.decision_at) for item in values], reason=reason
        )
    return stage(session, ledger)


def verify(session, bundle, pending_keys):
    ledger = restore(session, bundle)
    if ledger.clock != bundle["cutoff"] or set(ledger.pending) != set(pending_keys):
        raise ValueError("Los diagnósticos B4 no concilian con el corte y los pendientes")
