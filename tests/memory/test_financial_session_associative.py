"""Corrección asociativa B6 conectada a la sesión financiera de Titans-MAC sin banco.

La sesión emite la predicción del núcleo más la lectura de A de la generación anterior y
escribe A solo con etiquetas maduras, con el valor etiqueta menos predicción del núcleo. El
oráculo reproduce esa recurrencia por separado a partir de la sesión sin corrección.
"""

import os

import pytest
import torch
from test_financial_session import moment
from test_financial_session_controls import FLOWS
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_native_episode_backend import native as native

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    KalmanNoise,
    MatureCorrection,
)
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

EVENTS = (125, 126, 127, 128, 129, 130)
DELTA = MatureCorrection(AssociativeMemoryConfig("delta", rate=0.5, forgetting=0.25))
CONSTANT = MatureCorrection(
    AssociativeMemoryConfig("proximal", key_size=1, rate=2.0, forgetting=0.1), key="constant"
)
KALMAN = MatureCorrection(
    AssociativeMemoryConfig(
        "kalman",
        kalman=KalmanNoise(
            process_noise=1e-4, observation_noise=0.05, cohort_correlation=0.2, prior_variance=0.5
        ),
    )
)


def label(asset, index, shift=0.0):
    return 0.125 * (int(asset[-1]) + 1) + 0.03125 * (index - 125) + shift


@pytest.fixture(scope="module", autouse=True)
def module_backend():
    # Los recorridos se comparten en el módulo y el consumidor exige fastpath apagado.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared_native():
    path = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not path:
        pytest.skip("Falta el enlace nativo CPU compilado para esta comprobación")
    return load_native(path)


@pytest.fixture(scope="module")
def frozen_consumer(four_flow_source):
    config = FinancialConfig(
        four_flow_source["spec"], variant="mac_online", hidden_size=32, seed=42
    )
    model = FinancialPredictor(config, dtype=torch.float64, device="cpu")
    return FrozenFinancialConsumer(model.eval().requires_grad_(False))


def options(native, source, consumer, **extra):
    return (
        dict(
            native=native,
            consumer=consumer,
            codec=source["codec"],
            prefixes=source["prefixes"],
            retention=RetentionConfig(policy="reservoir", capacity=4, seed=73, frontier=1),
            phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
            world="four_flow_associative",
            fold="0",
            admission="m0",
            block_rows=4,
        )
        | extra
    )


def associative_state(run):
    """Matriz A y núcleo pendiente confirmados. Solo una clave enrutada guarda rutas pendientes."""
    if not run.associative:
        return None, None
    bundle = run._bundle(run.snapshot()["state"])
    memory, core, routes = run._associative_state(bundle["associative"])
    assert (routes is None) == (run.associative.routing is None)
    return memory, core


def trajectory(native, source, consumer, output, *, correction=None, shift=None, recover=False):
    """Recorre seis eventos y guarda emisiones, aplicaciones y estado de cada generación."""
    extra = {} if correction is None else dict(associative=correction)
    settings = options(native, source, consumer, **extra)
    run = FinancialSession(output, **settings)
    previous, events = [], []
    try:
        for index in EVENTS:
            at = moment(index)
            batches = [] if index == 129 else source["batches"][index]
            displaced = (shift or {}).get(index, 0.0)
            feedback = [
                native.Feedback(p.id, 0, at, label(p.asset, index, displaced)) for p in previous
            ]
            step = dict(kind="settlement" if index == 129 else "decision", cutoff=at)
            if recover and index == 128:
                before = run.snapshot()

                def fail(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte técnico antes de publicar")

                with pytest.raises(RuntimeError, match="Corte técnico"):
                    run.step(batches, feedback, fault=fail, **step)
                run.close()
                run = FinancialSession(output, **settings, resume=True)
                assert run.snapshot() == before
            commit = run.step(batches, feedback, **step)
            previous = commit.predictions
            memory, core = associative_state(run)
            events.append(
                dict(
                    index=index,
                    cutoff=at,
                    predictions=[(p.id, p.asset, p.decision_at, p.value) for p in previous],
                    applied=[
                        (o.prediction.id, o.prediction.asset, o.prediction.decision_at)
                        + (o.label.available_at, o.label.value)
                        for o in commit.applied
                    ],
                    memory=None if memory is None else memory.export(),
                    core=None if core is None else core.clone(),
                    applied_total=run.snapshot()["applied"],
                    fast=run._fast_store.gather(run.fast_state(), FLOWS).observed_steps.tolist(),
                )
            )
        snapshot = run.snapshot()
    finally:
        run.close()
    with FinancialSession(output, **settings, resume=True) as restored:
        assert restored.snapshot() == snapshot
    consumer.verify()
    assert all(parameter.grad is None for parameter in consumer.predictor.parameters())
    return dict(events=events, snapshot=snapshot)


def key_inputs(source, asset, decision_at):
    return torch.tensor(source["encoded"][(asset, decision_at)].key_inputs)


def oracle(source, baseline, correction, shift=None):
    """Emisiones y matrices esperadas a partir del núcleo sin corrección."""
    core = {
        (asset, at): value
        for event in baseline["events"]
        for _, asset, at, value in event["predictions"]
    }
    memory, emitted, matrices, written = AssociativeMemory(correction.memory), {}, [], 0
    for event in baseline["events"]:
        rows = [(asset, at) for _, asset, at, _ in event["predictions"]]
        if rows:
            keys = correction.keys(torch.cat([key_inputs(source, *row) for row in rows]))
            reads = memory.read(keys)[:, 0].tolist()
            emitted.update((row, core[row] + value) for row, value in zip(rows, reads, strict=True))
        applied = event["applied"]
        if applied:
            displaced = (shift or {}).get(event["index"], 0.0)
            memory = memory.write(
                correction.feedback(
                    ids=list(range(written + 1, written + len(applied) + 1)),
                    decision_at=[item[2] for item in applied],
                    available_at=[item[3] for item in applied],
                    keys=correction.keys(
                        torch.cat([key_inputs(source, item[1], item[2]) for item in applied])
                    ),
                    values=[item[4] + displaced - core[item[1], item[2]] for item in applied],
                ),
                cutoff=event["cutoff"],
            )
            written += len(applied)
        matrices.append(memory.matrix)
    return emitted, matrices


@pytest.fixture(scope="module")
def baseline(shared_native, four_flow_source, frozen_consumer, tmp_path_factory):
    root = tmp_path_factory.mktemp("associative_baseline")
    return trajectory(shared_native, four_flow_source, frozen_consumer, root / "core")


@pytest.fixture(scope="module")
def corrected(shared_native, four_flow_source, frozen_consumer, tmp_path_factory):
    root = tmp_path_factory.mktemp("associative_delta")
    return trajectory(
        shared_native, four_flow_source, frozen_consumer, root / "b6", correction=DELTA
    )


def emissions(result):
    return {
        (asset, at): value
        for event in result["events"]
        for _, asset, at, value in event["predictions"]
    }


def test_corrected_emissions_follow_the_previous_matrix_and_mature_writes(
    four_flow_source, baseline, corrected
):
    expected, matrices = oracle(four_flow_source, baseline, DELTA)
    observed = emissions(corrected)
    assert observed.keys() == expected.keys() and len(observed) == 20
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, rel=0, abs=1e-15)
    for event, matrix in zip(corrected["events"], matrices, strict=True):
        state = AssociativeMemory.restore(DELTA.memory, event["memory"])
        assert torch.equal(state.matrix, matrix)
        # Cada etiqueta aplicada escribe A una vez, sin depender de la admisión al banco.
        assert state.writes == event["applied_total"]
    assert corrected["events"][-1]["applied_total"] == 16
    first = [value for *_, value in baseline["events"][0]["predictions"]]
    assert [value for *_, value in corrected["events"][0]["predictions"]] == first
    # Desde el tercer evento A ya contiene etiquetas maduras y corrige la emisión.
    later = [observed[key] - emissions(baseline)[key] for key in observed if key[1] >= moment(127)]
    assert min(map(abs, later)) > 1e-6


def test_core_queue_and_fast_state_do_not_depend_on_the_correction(baseline, corrected):
    for left, right in zip(baseline["events"], corrected["events"], strict=True):
        assert left["fast"] == right["fast"]
        # Los identificadores nativos dependen del contrato. El resto del resultado coincide.
        assert [item[1:4] for item in left["applied"]] == [item[1:4] for item in right["applied"]]
        # Cada evento resuelve el anterior, así que la cola guarda el núcleo recién emitido.
        assert right["core"].tolist() == [value for *_, value in left["predictions"]]
        assert right["core"].dtype == torch.float64


def test_labels_of_an_event_only_change_later_emissions(
    native, four_flow_source, frozen_consumer, baseline, corrected, tmp_path
):
    shift = {127: 0.5}
    perturbed = trajectory(
        native, four_flow_source, frozen_consumer, tmp_path / "shift", correction=DELTA, shift=shift
    )
    before, after = emissions(corrected), emissions(perturbed)
    for key in before:
        if key[1] <= moment(127):
            assert after[key] == before[key]
        else:
            assert abs(after[key] - before[key]) > 1e-6
    expected, _ = oracle(four_flow_source, baseline, DELTA, shift=shift)
    for key, value in expected.items():
        assert after[key] == pytest.approx(value, rel=0, abs=1e-15)


def test_constant_key_control_applies_one_bias_per_event(
    native, four_flow_source, frozen_consumer, baseline, tmp_path
):
    result = trajectory(
        native, four_flow_source, frozen_consumer, tmp_path / "constant", correction=CONSTANT
    )
    expected, _ = oracle(four_flow_source, baseline, CONSTANT)
    core = emissions(baseline)
    for event in result["events"]:
        shifts = {
            round(value - core[asset, at], 15) for _, asset, at, value in event["predictions"]
        }
        assert len(shifts) <= 1
        for _, asset, at, value in event["predictions"]:
            assert value == pytest.approx(expected[asset, at], rel=0, abs=1e-15)


def test_kalman_rule_follows_the_same_session_contract_and_recovers_its_covariance(
    native, four_flow_source, frozen_consumer, baseline, tmp_path
):
    # La regla kalman (PT3) usa el mismo contrato de emisión, madurez y recuperación que
    # delta y proximal. El corte antes de publicar comprueba que la covarianza se conserva.
    result = trajectory(
        native,
        four_flow_source,
        frozen_consumer,
        tmp_path / "kalman",
        correction=KALMAN,
        recover=True,
    )
    expected, matrices = oracle(four_flow_source, baseline, KALMAN)
    observed = emissions(result)
    assert observed.keys() == expected.keys()
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, rel=0, abs=1e-15)
    for event, matrix in zip(result["events"], matrices, strict=True):
        state = AssociativeMemory.restore(KALMAN.memory, event["memory"])
        assert torch.equal(state.matrix, matrix) and state.writes == event["applied_total"]


def test_cut_before_commit_resumes_with_the_same_matrix(
    native, four_flow_source, frozen_consumer, corrected, tmp_path
):
    recovered = trajectory(
        native, four_flow_source, frozen_consumer, tmp_path / "cut", correction=DELTA, recover=True
    )
    assert recovered["snapshot"] == corrected["snapshot"]
    for left, right in zip(corrected["events"], recovered["events"], strict=True):
        assert left["predictions"] == right["predictions"]
        assert left["memory"]["matrix_sha256"] == right["memory"]["matrix_sha256"]


def test_contract_changes_only_when_the_correction_is_declared(
    native, four_flow_source, frozen_consumer, tmp_path
):
    plain = FinancialSession(
        tmp_path / "plain", **options(native, four_flow_source, frozen_consumer)
    )
    delta = FinancialSession(
        tmp_path / "delta", **options(native, four_flow_source, frozen_consumer, associative=DELTA)
    )
    constant = FinancialSession(
        tmp_path / "constant",
        **options(native, four_flow_source, frozen_consumer, associative=CONSTANT),
    )
    try:
        identities = {plain.contract_id, delta.contract_id, constant.contract_id}
        assert len(identities) == 3
        assert plain.associative is None
        assert plain.model_id != delta.model_id
    finally:
        for run in (plain, delta, constant):
            run.close()


@pytest.mark.parametrize(
    "change",
    [
        dict(admission="m1"),
        dict(associative=DELTA.memory),
        dict(associative="delta"),
    ],
)
def test_correction_requires_titans_without_bank_and_its_declaration(
    native, four_flow_source, frozen_consumer, tmp_path, change
):
    settings = options(native, four_flow_source, frozen_consumer)
    settings.update(dict(associative=DELTA) | change)
    with pytest.raises(ValueError, match="B6"):
        FinancialSession(tmp_path / "rejected", **settings)


def test_correction_keys_and_feedback_validate_their_contract():
    with pytest.raises(ValueError, match="una clave de codec, constant, regime"):
        MatureCorrection(DELTA.memory, key="label")
    with pytest.raises(ValueError, match="clave codec usa 64 coordenadas"):
        MatureCorrection(AssociativeMemoryConfig("delta", key_size=8))
    with pytest.raises(ValueError, match="FP32"):
        DELTA.keys(torch.ones((2, 64), dtype=torch.float64))
    with pytest.raises(ValueError, match="norma positiva"):
        DELTA.keys(torch.zeros((1, 64)))
    keys = DELTA.keys(torch.arange(1, 129, dtype=torch.float32).reshape(2, 64))
    torch.testing.assert_close(
        torch.linalg.vector_norm(keys, dim=1), torch.ones(2, dtype=torch.float64)
    )
    assert CONSTANT.keys(torch.ones((3, 64))).tolist() == [[1.0]] * 3
    feedback = CONSTANT.feedback(
        ids=[3, 4],
        decision_at=[1, 1],
        available_at=[2, 2],
        keys=torch.ones((2, 1), dtype=torch.float64),
        values=[0.5, -0.5],
    )
    assert feedback.weights.tolist() == [0.5, 0.5]
    assert (
        DELTA.feedback(
            ids=[3], decision_at=[1], available_at=[2], keys=keys[:1], values=[0.5]
        ).weights
        is None
    )
