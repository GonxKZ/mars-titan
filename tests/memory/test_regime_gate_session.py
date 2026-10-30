"""Publicación durable de la puerta E4 en cada generación de `FinancialSession` (#750).

La sesión se contrasta con el recorrido por ventanas de E4, comprobado aparte con sus
ecuaciones, y con una recurrencia independiente de `RegimeEvidenceGate` sobre los resultados
que aplica el ejecutor. Ninguna prueba ajusta parámetros ni recorre datos reales.
"""

from dataclasses import replace

import pytest
import torch
from test_financial_session import moment
from test_financial_session_associative import DELTA, associative_state, options
from test_financial_session_associative import four_flow_source as four_flow_source
from test_financial_session_associative import frozen_consumer as frozen_consumer
from test_financial_session_associative import module_backend as module_backend
from test_financial_session_associative import no_target_estimation as no_target_estimation
from test_financial_session_associative import shared_native as shared_native
from test_mars_titan_correction_parity import (
    PROXIMAL,
    RECIPE,
    SOURCE,
    core,
    corrected_pass,
    emitted,
)
from test_mars_titan_session_parity import events
from test_mars_titan_variant import session_run

from mars_titan.memory import regime_evidence_gate, regime_gate_session
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.regime_evidence_gate import RegimeEvidenceGate, RegimeGateConfig
from mars_titan.memory.role_binding import RoleBindingConfig
from mars_titan.memory.salience_budget import SalienceBudgetConfig
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.training.regime_gate_inference import RegimeGateCorrectionInference

# Puerta que reacciona desde el primer error, para que cambie entre generaciones.
CUSUM = RegimeGateConfig(threshold=1.0, drift=0.0, floor_gate=0.25, half_life=2.0, min_history=1)


def session(native, source, consumer, output, gate=None, correction=PROXIMAL, **extra):
    return FinancialSession(
        output,
        **(options(native, source, consumer, associative=correction) | extra),
        regime_gate=gate,
    )


def feedback(native, predictions, at, errors=None):
    errors = errors or [0.01 * (index + 1) * (at - 124) for index in range(len(predictions))]
    return [
        native.Feedback(p.id, 0, moment(at), p.value + error)
        for p, error in zip(predictions, errors, strict=True)
    ]


def values(commit):
    return [(p.asset, p.decision_at, p.value) for p in commit.predictions]


def walk(run, source, native, steps=(125, 126, 127, 128)):
    """Pasos con etiquetas de las predicciones anteriores y la cohorte de cada generación."""
    previous, commits, cohorts = [], [], []
    for at in steps:
        commit = run.step(source["batches"][at], feedback(native, previous, at))
        cohorts.append(
            (
                tuple(
                    (
                        o.prediction.asset,
                        o.prediction.decision_at,
                        o.label.available_at,
                        o.prediction.value,
                        o.label.value,
                    )
                    for o in commit.applied
                ),
                moment(at),
            )
        )
        commits.append(commit)
        previous = commit.predictions
    return commits, cohorts


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_a_constant_gate_of_one_keeps_b6_bit_for_bit(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    args = (shared_native, four_flow_source, frozen_consumer)
    one = RegimeGateConfig(mode="constant", constant=1.0)
    with (
        session(*args, tmp_path / "plain") as plain,
        session(*args, tmp_path / "one", one) as gated,
    ):
        left, _ = walk(plain, four_flow_source, shared_native)
        right, _ = walk(gated, four_flow_source, shared_native)
        assert [values(c) for c in left] == [values(c) for c in right]
        x, y = associative_state(plain)[0].export(), associative_state(gated)[0].export()
        for name in ("configuration", "writes", "cursor", "matrix_sha256"):
            assert x[name] == y[name]
        assert x["writes"] == 12
        assert plain.regime_gate_state() is None
        assert "regime_gate" not in plain._bundle(plain.snapshot()["state"])
        assert gated.regime_gate_state()["publications"] == gated.snapshot()["cursor"]


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_a_constant_gate_is_the_rate_times_the_gate_with_one_market(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    args = (shared_native, four_flow_source, frozen_consumer)
    half = RegimeGateConfig(mode="constant", constant=0.5)
    slower = MatureCorrection(replace(PROXIMAL.memory, rate=PROXIMAL.memory.rate * 0.5))
    with (
        session(*args, tmp_path / "half", half) as gated,
        session(*args, tmp_path / "slower", correction=slower) as plain,
    ):
        left, _ = walk(gated, four_flow_source, shared_native)
        right, _ = walk(plain, four_flow_source, shared_native)
        assert [values(c) for c in left] == [values(c) for c in right]
        x, y = associative_state(gated)[0], associative_state(plain)[0]
        assert torch.equal(x.matrix, y.matrix) and x.writes == y.writes == 12
        # A conserva su tasa declarada aunque cada escritura use η·g.
        assert x.config == PROXIMAL.memory


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_each_generation_publishes_the_gate_of_its_mature_errors(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    with session(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "cusum", CUSUM
    ) as run:
        _, cohorts = walk(run, four_flow_source, shared_native)
        expected, gates = RegimeEvidenceGate(CUSUM), []
        for cohort, cutoff in cohorts:
            decision, expected = expected.step(cohort, cutoff=cutoff)
            gates.append(decision.by_market())
        assert run.regime_gate_state() == expected.export()
        assert expected.publications == run.snapshot()["cursor"] == 4
        assert expected.observed == run.snapshot()["applied"] == 12
        # La puerta cambia entre generaciones, así que el contraste no es trivial.
        assert len({g["US"] for g in gates[1:]}) > 1
        assert all(len(cohort) == 4 for cohort, _ in cohorts[1:]) and cohorts[0][0] == ()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_the_session_emits_and_writes_what_the_e4_window_does(
    shared_native, four_flow_source, tmp_path
):
    consumer = FrozenFinancialConsumer(core(four_flow_source))
    settings = options(shared_native, four_flow_source, consumer, associative=PROXIMAL)
    result = session_run(
        shared_native, four_flow_source, dict(settings, regime_gate=CUSUM), tmp_path / "session"
    )
    window = RegimeGateCorrectionInference(
        core(four_flow_source),
        RECIPE,
        PROXIMAL,
        four_flow_source["codec"],
        regime_gate=CUSUM,
        audit=True,
    )
    window._pass(SOURCE, events(four_flow_source))
    expected = [(a, t, v) for event in result["emitted"] for _, a, t, v in event]
    assert emitted(window) == expected and len(expected) == 20
    # Sin la puerta, B6 escribe con η y las emisiones cambian desde el tercer evento.
    plain, _ = corrected_pass(four_flow_source, PROXIMAL)
    assert emitted(plain)[:8] == expected[:8] and emitted(plain)[8:] != expected[8:]
    reopened = FinancialSession(tmp_path / "session", **settings, regime_gate=CUSUM, resume=True)
    try:
        assert torch.equal(associative_state(reopened)[0].matrix, window.memory.matrix)
        session_gate = reopened.regime_gate_state()
    finally:
        reopened.close()
    window_gate = window.regime_gate.export()
    # La ventana no publica en los eventos sin etiquetas. Las estadísticas coinciden igual.
    assert session_gate["markets"] == window_gate["markets"]
    assert window_gate["markets"]["US"][0] == 16


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize("boundary", ("before_commit", "committed"))
def test_recovery_preserves_the_gate_and_the_next_prediction(
    shared_native, four_flow_source, frozen_consumer, tmp_path, boundary
):
    args = (shared_native, four_flow_source, frozen_consumer)
    reference = session(*args, tmp_path / "reference", CUSUM)
    failed = session(*args, tmp_path / "failed", CUSUM)
    try:
        a = reference.step(four_flow_source["batches"][125], [])
        b = failed.step(four_flow_source["batches"][125], [])
        reference.step(
            four_flow_source["batches"][126], feedback(shared_native, a.predictions, 126)
        )

        def fault(point):
            if point == getattr(shared_native.Boundary, boundary):
                raise RuntimeError("Corte del fixture E4")

        with pytest.raises(RuntimeError, match="Corte del fixture"):
            failed.step(
                four_flow_source["batches"][126],
                feedback(shared_native, b.predictions, 126),
                fault=fault,
            )
        failed.close()
        failed = session(*args, tmp_path / "failed", CUSUM, resume=True)
        if boundary == "before_commit":
            failed.step(
                four_flow_source["batches"][126], feedback(shared_native, b.predictions, 126)
            )
        assert failed.regime_gate_state() == reference.regime_gate_state()
        assert values(failed.step(four_flow_source["batches"][127], [])) == values(
            reference.step(four_flow_source["batches"][127], [])
        )
        assert failed.regime_gate_state() == reference.regime_gate_state()
    finally:
        reference.close()
        failed.close()


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_changed_runtime_or_resume_configuration_is_rejected(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    args = (shared_native, four_flow_source, frozen_consumer, tmp_path / "contract")
    with session(*args, CUSUM) as run:
        run.step(four_flow_source["batches"][125], [])
        before = run.snapshot()
        run._regime_gate_config = replace(CUSUM, threshold=2.0)
        with pytest.raises(ValueError, match="configuración"):
            run.step(four_flow_source["batches"][126], [])
        assert run.snapshot() == before
    with pytest.raises(ValueError):
        session(*args, replace(CUSUM, threshold=2.0), resume=True)


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    "module", [regime_gate_session, regime_evidence_gate], ids=["session", "gate"]
)
def test_the_session_pins_the_code_and_the_configuration_of_e4(
    shared_native, four_flow_source, frozen_consumer, tmp_path, monkeypatch, module
):
    args = (shared_native, four_flow_source, frozen_consumer)
    with (
        session(*args, tmp_path / "plain") as plain,
        session(*args, tmp_path / "first", CUSUM) as first,
        session(*args, tmp_path / "other", replace(CUSUM, threshold=2.0)) as other,
    ):
        assert first.model_id != plain.model_id
        # Otra configuración con el mismo código solo cambia el contrato.
        assert other.model_id == first.model_id and other.contract_id != first.contract_id
        before = first.model_id
    changed = tmp_path / "other.py"
    changed.write_text("# Puerta diferente.\n")
    monkeypatch.setattr(module, "__file__", str(changed))
    with session(*args, tmp_path / "second", CUSUM) as second:
        assert second.model_id != before


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
def test_a_published_state_that_does_not_match_its_generation_is_rejected(
    shared_native, four_flow_source, frozen_consumer, tmp_path
):
    with session(
        shared_native, four_flow_source, frozen_consumer, tmp_path / "audit", CUSUM
    ) as run:
        walk(run, four_flow_source, shared_native, steps=(125, 126))
        bundle = run._bundle(run.snapshot()["state"])
        state = regime_gate_session.restore(run, bundle)
        snapshot = run.snapshot()
        regime_gate_session.verify(run, bundle)
        regime_gate_session.verify_confirmed(run, bundle, snapshot)
        with pytest.raises(ValueError, match="generación"):
            regime_gate_session.verify(run, dict(bundle, generation=bundle["generation"] + 1))
        with pytest.raises(ValueError, match="generación"):
            regime_gate_session.verify(run, dict(bundle, cutoff=bundle["cutoff"] + 1))
        with pytest.raises(ValueError, match="aplicados"):
            regime_gate_session.verify_confirmed(
                run, bundle, dict(snapshot, applied=snapshot["applied"] + 1)
            )
        assert state.observed == snapshot["applied"] == 4


@pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
@pytest.mark.parametrize(
    ("gate", "correction", "extra"),
    [
        (CUSUM, DELTA, {}),
        (CUSUM, None, {}),
        (
            CUSUM,
            MatureCorrection(AssociativeMemoryConfig("proximal"), role_binding=RoleBindingConfig()),
            {},
        ),
        (CUSUM, PROXIMAL, dict(salience=SalienceBudgetConfig())),
        (dict(mode="cusum"), PROXIMAL, {}),
    ],
    ids=["delta", "without_a", "roles", "with_e3", "untyped"],
)
def test_invalid_session_configuration_fails_before_creating_output(
    shared_native, four_flow_source, frozen_consumer, tmp_path, gate, correction, extra
):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="E4|E3"):
        session(shared_native, four_flow_source, frozen_consumer, output, gate, correction, **extra)
    assert not output.exists()
