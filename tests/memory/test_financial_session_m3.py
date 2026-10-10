"""M3 en la sesión y en el recorrido del lector: causalidad, presupuesto y recuperación.

Los flujos de `varied` tienen noticias, fundamentales y últimos precios distintos. Las
etiquetas son manuales y los parámetros están congelados. No hay ajustes ni objetivos reales.
"""

import copy

import pytest
from test_financial_session import moment
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import label, options
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_session_parity import PHASE, SCALERS, events, models, varied

from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training import mars_titan_run as mt

M3 = CompositeScoreConfig(SCALERS, capacity=4)
RETENTION = dict(
    m1=RetentionConfig(policy="reservoir", capacity=4, seed=73, frontier=1),
    m2=MatureErrorConfig(capacity=4),
    m3=M3,
)


@pytest.fixture(scope="module")
def source(four_flow_source):
    return varied(four_flow_source)


def inference(source, native, admission):
    predictor, readout = models(source, 1, "per_step")
    plan = mt.ReadoutRecipe(
        loss="mae", block_rows=4, neighbors=2, max_working_bytes=readout.config.max_working_bytes
    )
    return mt.MarsTitanInference(
        predictor,
        readout,
        plan,
        admission=admission,
        retention=RETENTION[admission],
        native=native,
        codec=source["codec"],
        world="m3",
        fold="0",
        audit=True,
    )


def admissions(engine, until=None):
    return [e for e in engine.audit if e[0] == "admit" and (until is None or e[2] <= until)]


def test_immature_labels_and_later_inputs_do_not_change_earlier_admissions(shared_native, source):
    """Etiquetas que maduran después de 127 y entradas posteriores a 128, alteradas."""
    labels_after, inputs_after = moment(127), moment(128)
    base = inference(source, shared_native, "m3")
    base._pass(PHASE, events(source))
    changed = []
    for event in events(source):
        labels, inputs = event.labels, event.inputs
        if event.at > labels_after:
            labels = tuple((flow, at, value + 10.0) for flow, at, value in labels)
        if event.at > inputs_after:
            inputs = tuple(
                dict(raw, inputs=dict(raw["inputs"], charts=raw["inputs"]["charts"] + 1.0))
                for raw in inputs
            )
        changed.append(ObservationEvent(event.at, inputs, labels, event.close_phase))
    other = inference(source, shared_native, "m3")
    other._pass(PHASE, changed)
    assert admissions(base, labels_after) == admissions(other, labels_after)
    assert admissions(base, labels_after) and admissions(base) != admissions(other)
    # Las predicciones de 128 leen el banco anterior a las etiquetas alteradas de su evento.
    before = [e for e in base.audit if e[0] == "prediction" and e[3] <= inputs_after]
    assert before == [e for e in other.audit if e[0] == "prediction" and e[3] <= inputs_after]


def test_m3_examines_the_same_candidates_with_the_capacity_and_offers_of_m2(shared_native, source):
    passes = {}
    for admission in ("m1", "m2", "m3"):
        engine = inference(source, shared_native, admission)
        metrics = engine._pass(PHASE, events(source))
        passes[admission] = (engine, metrics)
    labels = {name: metrics["admitted"] for name, (_, metrics) in passes.items()}
    assert labels == dict(m1=16, m2=16, m3=16)
    m3 = passes["m3"][1]
    assert m3["selective_admitted"] + m3["selective_rejected"] == 16
    assert m3["selective_admitted"] - m3["selective_evicted"] == M3.quotas["selective"]
    assert "selective_admitted" not in passes["m2"][1]
    m2_admits, m3_admits = admissions(passes["m2"][0]), admissions(passes["m3"][0])
    assert [e[:5] for e in m2_admits] == [e[:5] for e in m3_admits]
    # Mismo reservorio y mismos recientes: solo cambia la puntuación del selectivo.
    for indices in (e[5] for e in m3_admits):
        assert sum(map(len, indices.values())) <= M3.capacity
    assert m3_admits[-1][5]["reservoir"] and m3_admits[-1][5]["recent"] == (16,)


def open_session(native, source, consumer, output, *, resume=False):
    settings = options(native, source, consumer, admission="m3", retention=M3)
    settings.update(world="manual_m3", resume=resume)
    return FinancialSession(output, **settings)


def trajectory(native, source, consumer, output, *, recover=False):
    """Seis eventos M3 con un corte opcional en 128. También lo usa la comprobación CUDA."""
    run = open_session(native, source, consumer, output)
    previous, receipts, points = [], [], {}
    try:
        for index in (125, 126, 127, 128, 129, 130):
            batches = [] if index == 129 else source["batches"][index]
            feedback = [
                native.Feedback(p.id, 0, moment(index), label(p.asset, index)) for p in previous
            ]
            kind = "settlement" if index == 129 else "decision"
            if recover and index == 128:
                before = run.snapshot()

                def fail(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte M3 antes de publicar")

                with pytest.raises(RuntimeError, match="Corte M3"):
                    run.step(batches, feedback, kind=kind, cutoff=moment(index), fault=fail)
                run.close()
                run = open_session(native, source, consumer, output, resume=True)
                assert run.snapshot() == before
            previous = run.step(batches, feedback, kind=kind, cutoff=moment(index)).predictions
            points.update({(p.asset, p.decision_at): p.value for p in previous})
            bundle = run._bundle(run.snapshot()["state"])
            bank, episodes = run._bank(bundle["bank"])
            receipts.append((bank.index_ids(), bank.receipt, bank.write_features))
            assert set(bank.write_features) == set(episodes)
        final = run.snapshot()
        diagnostics = run.diagnostics()
    finally:
        run.close()
    with open_session(native, source, consumer, output, resume=True) as restored:
        assert restored.snapshot() == final
    return receipts, final, diagnostics, points


def test_session_recovers_m3_atomically_without_repeating_offers(shared_native, source, tmp_path):
    predictor, readout = models(source, 1, "per_step")
    consumer = FrozenFinancialConsumer(predictor, readout=readout)
    reference = trajectory(shared_native, source, consumer, tmp_path / "reference")
    recovered = trajectory(shared_native, source, consumer, tmp_path / "recovered", recover=True)
    assert reference == recovered
    receipts, final, diagnostics, points = reference
    assert len(points) == 20
    assert (final["issued"], final["applied"]) == (20, 16)
    assert diagnostics["B_mem"] == 4 and diagnostics["physical_slots"] <= 4
    assert all(receipt is not None for _, receipt, _ in receipts[1:])


def test_session_rejects_a_bank_whose_m3_error_contradicts_the_emission(
    shared_native, source, tmp_path
):
    predictor, readout = models(source, 1, "per_step")
    consumer = FrozenFinancialConsumer(predictor, readout=readout)
    with open_session(shared_native, source, consumer, tmp_path / "m3") as session:
        first = session.step(source["batches"][125], [])
        feedback = [
            shared_native.Feedback(p.id, 0, moment(126), label(p.asset, 126))
            for p in first.predictions
        ]
        session.step(source["batches"][126], feedback)
        bundle = session._bundle(session.snapshot()["state"])
        payload = copy.deepcopy(session._read(bundle["bank"], "bank"))
        reference = session._stage(payload, "bank")
        session._bank(reference)
        # Una emisión registrada distinta, coherente con su etiqueta, deja el banco intacto.
        # Solo el contraste con los errores M3 del banco puede detectarla.
        episode = payload["episodes"][next(iter(payload["episodes"]))]
        episode["issued_prediction"] += 0.5
        episode["error"] = episode["label"] - episode["issued_prediction"]
        with pytest.raises(ValueError, match="errores M3"):
            session._bank(session._stage(payload, "bank"))
    assert all(parameter.grad is None for parameter in predictor.parameters())
