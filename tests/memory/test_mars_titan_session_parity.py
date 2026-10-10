"""El recorrido cronológico del lector emite lo mismo que `FinancialSession` con el mismo banco.

Los seis eventos de cuatro flujos se entregan a `FinancialSession` y, como eventos de
observación equivalentes, a `MarsTitanInference`. Las predicciones dependen del banco
retenido, porque llegan 16 etiquetas a un banco de capacidad 4 y el lector elige dos
vecinos. Coincidir bit a bit acredita el mismo orden de evento, la misma admisión y los
mismos IDs que la sesión, también con M2, con K = 2 en sus dos modos de selección y con
la retención con centros fijos que usa M en el factorial CM-v1. M3 usa una variante de los
flujos con noticias, fundamentales y últimos precios distintos, así que sus tres componentes
cambian entre flujos e instantes.
"""

import math

import numpy as np
import pytest
import torch
from test_financial_session import moment
from test_financial_session_associative import EVENTS, label, options
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_financial_session_controls import FLOWS
from test_mars_titan_variant import session_run

from mars_titan.data.input_policy import MODALITIES
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.memory.write_scores import WriteScalers
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training import mars_titan_run as mt

# La paridad se omite sin enlace en la suite CPU y falla con MARS_TITAN_REQUIRE_NATIVE=1.
pytestmark = pytest.mark.native_binding

PHASE = FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201))
# Escalas manuales, como si procedieran del tramo de entrenamiento de la ventana.
SCALERS = WriteScalers(
    source_sha256="c" * 64,
    dataset_sha256="d" * 64,
    decision_start=1,
    decision_end=2,
    decisions=64,
    labels=64,
    filing_decisions=32,
    news_decisions=16,
    error_median=0.25,
    anomaly_median=1.0,
    filing_age_median=6.0,
    news_share=0.25,
)


def varied(source):
    """Flujos con noticias en 1 y 3, fundamentales en 2 y 3 y un último precio propio."""
    batches = {}
    for index, group in source["batches"].items():
        batches[index] = []
        for position, batch in enumerate(group):
            inputs = {name: np.array(values) for name, values in batch.inputs.items()}
            presence = np.array(batch.presence)
            inputs["prices"][:, -1, :4] += np.float32(0.004 * position)
            if position in (1, 3):
                inputs["news"][:] = (0.6, 0.8)
                presence[:, 1] = True
            if position in (2, 3):
                age = 2.0 * (index - 124) + position
                inputs["fundamentals"][:] = (0.5, 1.0, math.log1p(age))
                presence[:, 3] = True
            raw = dict(
                inputs=inputs,
                presence=presence,
                sample_ids=list(batch.sample_ids),
                prediction_at=np.array(batch.prediction_at, dtype="datetime64[us]"),
                input_available_at=np.array(batch.input_available_at, dtype="datetime64[us]"),
            )
            batches[index].append(validated_cpu_batch(raw, source["spec"]))
    return dict(source, batches=batches)


def instants(batches, name):
    values = [value for batch in batches for value in getattr(batch, name)]
    return np.array(values, dtype="datetime64[us]")


def raw(batches):
    """Bloque de los cuatro flujos de un instante con los campos de entrada validados."""
    return dict(
        inputs={name: np.concatenate([b.inputs[name] for b in batches]) for name in MODALITIES},
        presence=np.concatenate([batch.presence for batch in batches]),
        sample_ids=[sample for batch in batches for sample in batch.sample_ids],
        prediction_at=instants(batches, "prediction_at"),
        input_available_at=instants(batches, "input_available_at"),
    )


def events(source):
    """Los eventos de la sesión: etiquetas de la emisión anterior y entradas del instante."""
    result, previous = [], []
    for index in EVENTS:
        at = moment(index)
        inputs = () if index == 129 else (raw(source["batches"][index]),)
        labels = tuple((flow, decided, label(flow, index)) for flow, decided in previous)
        result.append(ObservationEvent(at, inputs, labels, False))
        previous = [(flow, at) for flow in FLOWS] if inputs else []
    return result


def models(source, refinements, episodes):
    config = FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42)
    predictor = FinancialPredictor(config, dtype=torch.float64, device="cpu")
    readout = EpisodicReadout(
        EpisodicReadoutConfig(
            source["codec"].fingerprint(),
            hidden_size=32,
            refinements=refinements,
            neighbors=2,
            seed=7,
            episode_selection=episodes,
        ),
        dtype=torch.float64,
        device="cpu",
    )
    return predictor.eval().requires_grad_(False), readout.eval().requires_grad_(False)


@pytest.mark.parametrize(
    ("admission", "refinements", "episodes", "policy"),
    [
        ("m1", 1, "per_step", "reservoir"),
        ("m2", 1, "per_step", "reservoir"),
        ("m1", 2, "first_read", "reservoir"),
        ("m1", 2, "per_step", "reservoir"),
        ("m1", 1, "per_step", "anchored"),
        ("m3", 1, "per_step", "reservoir"),
        ("m3", 2, "first_read", "reservoir"),
    ],
)
def test_chronological_pass_emits_exactly_what_the_session_emits(
    shared_native, four_flow_source, tmp_path, admission, refinements, episodes, policy
):
    if admission == "m3":
        four_flow_source = varied(four_flow_source)
    predictor, readout = models(four_flow_source, refinements, episodes)
    retention = (
        MatureErrorConfig(capacity=4)
        if admission == "m2"
        else CompositeScoreConfig(SCALERS, capacity=4)
        if admission == "m3"
        else RetentionConfig(policy=policy, capacity=4, seed=73, frontier=1)
    )
    settings = options(
        shared_native,
        four_flow_source,
        FrozenFinancialConsumer(predictor, readout=readout),
        admission=admission,
        retention=retention,
    )
    session = session_run(shared_native, four_flow_source, settings, tmp_path / "session")
    plan = mt.ReadoutRecipe(
        loss="mae", block_rows=4, neighbors=2, max_working_bytes=readout.config.max_working_bytes
    )
    inference = mt.MarsTitanInference(
        predictor,
        readout,
        plan,
        admission=admission,
        retention=retention,
        native=shared_native,
        codec=four_flow_source["codec"],
        world="parity",
        fold="0",
        audit=True,
    )
    metrics = inference._pass(PHASE, events(four_flow_source))
    emitted = [(a, t, v) for event in session["emitted"] for _, a, t, v in event]
    issued = [(e[2], e[3], e[4]) for e in inference.audit if e[0] == "prediction"]
    assert issued == emitted and len(issued) == 20
    assert metrics["labels"] == metrics["admitted"] == session["snapshot"]["applied"] == 16
    admitted = [e[3] for e in inference.audit if e[0] == "admit"]
    assert [i for ids in admitted for i in ids] == list(range(1, 17))
    retained, episodes_kept = session["bank"]
    assert 1 <= len(retained) <= 4 and set(retained) <= set(range(1, 17))
    # El error M2 y M3 de la sesión es el de la predicción emitida, como en el recorrido.
    if admission in ("m2", "m3"):
        for identifier in retained:
            episode = episodes_kept[identifier]
            assert episode["error"] == episode["label"] - episode["issued_prediction"]
    if admission == "m3":
        final = [e for e in inference.audit if e[0] == "admit"][-1]
        indices, components = final[5], final[6]
        assert {i for ids in indices.values() for i in ids} == set(retained)
        # Los tres componentes varían: la paridad no se reduce a ordenar por el error.
        rows = [row for e in inference.audit if e[0] == "admit" for row in e[6]]
        assert len({row[2] for row in rows}) > 2 and len({row[3] for row in rows}) > 2
        assert {row[5] for row in rows} == {0, 1, 2, 3} and components
