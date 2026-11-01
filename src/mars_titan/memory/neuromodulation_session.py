"""Publicación N1 con estados anterior y posterior y comprobación del registro nativo."""

import hashlib
from pathlib import Path

from . import neuromodulation_source
from .neuromodulation_source import NeuromodulationSource, NeuromodulationSourceConfig
from .salience_budget import MatureWrite, _canonical

FIELDS = ("neuromodulation", "neuromodulation_before")


def configure(consumer, binding):
    from .financial_consumers import TitansBinding

    config = getattr(consumer, "neuromodulation_source", None)
    if config is not None and (
        type(config) is not NeuromodulationSourceConfig or type(binding) is not TitansBinding
    ):
        raise ValueError("La fuente N1 necesita el enlace financiero registrado")
    return config


def code():
    return {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in (
            ("neuromodulation_session", __file__),
            ("neuromodulation_source", neuromodulation_source.__file__),
        )
    }


def before(session, bundle):
    reference = (
        bundle["neuromodulation"]
        if bundle
        else session._stage(
            NeuromodulationSource(session._neuromodulation_config).export(), "neuromodulation"
        )
    )
    source = NeuromodulationSource.restore(
        session._neuromodulation_config, session._read(reference, "neuromodulation")
    )
    return {"neuromodulation": reference, "neuromodulation_before": reference}, source


def state(session, bundle, *, previous=False):
    if bundle is None:
        return NeuromodulationSource(session._neuromodulation_config)
    name = "neuromodulation_before" if previous else "neuromodulation"
    return NeuromodulationSource.restore(
        session._neuromodulation_config, session._read(bundle[name], "neuromodulation")
    )


def resolve(session, bundle, outcomes):
    previous = state(session, bundle, previous=True)
    rows = tuple(
        MatureWrite(
            x.prediction.asset,
            x.prediction.decision_at,
            x.label.available_at,
            x.prediction.value,
            x.label.value,
        )
        for x in outcomes
    )
    following = previous.advance(
        rows,
        cutoff=bundle["cutoff"],
        generation=bundle["generation"],
        applied_total=previous.applied_total + len(rows),
    )
    return dict(bundle, neuromodulation=session._stage(following.export(), "neuromodulation"))


def verify(session, bundle, control, flows):
    previous, current = state(session, bundle, previous=True), state(session, bundle)
    if (
        previous.generation + 1 != current.generation
        or current.generation != bundle["generation"]
        or current.cutoff != bundle["cutoff"]
        or (previous.cutoff is not None and previous.cutoff > current.cutoff)
    ):
        raise ValueError("La fuente N1 no corresponde a la generación confirmada")
    trace = control.get("neuromodulation_snapshot")
    expected = (
        previous.snapshot(
            tuple(flows), prediction_at=bundle["cutoff"], context_id=control.get("context_id")
        ).record()
        if flows
        else None
    )
    if _canonical(trace) != _canonical(expected):
        raise ValueError("El snapshot N1 no coincide con la fuente anterior y el evento")


def verify_confirmed(session, bundle, snapshot, record):
    previous, current = state(session, bundle, previous=True), state(session, bundle)
    rows = tuple(
        MatureWrite(
            r["prediction"]["asset"],
            r["prediction"]["decision_at"],
            r["available_at"],
            r["prediction"]["value"],
            r["target"],
        )
        for r in record["feedback"]
    )
    expected = previous.advance(
        rows,
        cutoff=bundle["cutoff"],
        generation=bundle["generation"],
        applied_total=snapshot["applied"],
    )
    if _canonical(expected.export()) != _canonical(current.export()):
        raise ValueError(
            "La fuente N1 contradice los resultados realmente aplicados por el ejecutor"
        )
