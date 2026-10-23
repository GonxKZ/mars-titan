"""Ajuste CUDA sobre episodios de prueba maduros, con referencia NumPy independiente."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch
from mechanism_fit_oracle import cohort, ridge
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_mars_titan_session_parity import SCALERS, varied
from test_mechanism_admission import PHASE, configured

from mars_titan.calibration.cohort_feedback import CohortFeedback
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import FEATURES, MechanismPolicy
from mars_titan.models.titans.mechanism_selection import digest, policy_signature
from mars_titan.training.mechanism_fit_contract import MechanismFitConfig
from mars_titan.training.mechanism_fit_window import MechanismFitWindow
from mars_titan.training.mechanism_policy_fit import _cohorts, fit_policy

pytestmark = [pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE"), pytest.mark.cuda]


@pytest.fixture(scope="module", autouse=True)
def no_optimizer_steps():
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    def forbidden(*args, **kwargs):
        pytest.fail("Esta prueba solo admite el solver ridge, no pasos de optimizador")

    handle = register_optimizer_step_pre_hook(forbidden)
    try:
        yield
    finally:
        handle.remove()


@pytest.fixture(scope="module")
def source_run(four_flow_source, tmp_path_factory, request):
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
        source = varied(four_flow_source)
        mode = getattr(request, "param", "regional")
        regional_enabled = mode != "limited"
        parent = engines(
            source, "cuda:0", online=regional_enabled, max_rows=4 if mode == "capacity" else 32768
        )[3]
        if regional_enabled:
            regional = configured(source, "cuda:0")[3]
            consumer = FrozenMechanismConsumer(
                parent.predictor,
                readout=regional.readout,
                calibration=CohortFeedback.restore(parent._initial_calibration),
                catalog=regional.catalog,
                action=MechanismAction("parent", "k4cr", "online"),
            )
        else:
            consumer = parent
        with FinancialSession(
            tmp_path_factory.mktemp("native-fit") / "source",
            native=native,
            consumer=consumer,
            codec=source["codec"],
            prefixes=source["prefixes"],
            retention=CompositeScoreConfig(SCALERS, capacity=4),
            phase=PHASE,
            world="mechanism_fit_numeric_fixture",
            fold="0",
            admission="m3",
            block_rows=4,
            mechanism_collection=MechanismCollectionConfig(),
        ) as run:
            run.step(source["batches"][125], [], kind="warmup")
            prior = []
            late = None
            for index in (126, 127, 128, 130):
                selected = prior[:2] if index == 127 and mode != "capacity" else prior
                labels = [
                    native.Feedback(point.id, 0, moment(index), 0.125 * (int(point.asset[-1]) + 1))
                    for point in selected
                ]
                if index == 130 and mode != "capacity":
                    labels.append(native.Feedback(late.id, 0, moment(127), 0.375))
                prior = run.step(source["batches"][index], labels).predictions
                if index == 126:
                    late = prior[2]
            run.step([], [], kind="settlement", cutoff=PHASE.close_at, close_phase=True)
            assert run.snapshot()["applied"] == (12 if mode == "capacity" else 11)
            yield run
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def protocol(run, *, mode="closed_phases", interval_weight=0.5):
    return MechanismFitConfig(
        parent_consumer_id=run.consumer.model_id,
        parent_selected_before=moment(125),
        fit_start=moment(126),
        fit_end=moment(133),
        ridge=0.01,
        interval_weight=interval_weight,
        min_gain=0.0,
        market_weights=(("US", 1.0),),
        costs=(0.0,) * 36,
        min_cohorts=2,
        catalog_id=run.consumer.catalog.fingerprint(),
        source_mode=mode,
    )


def raw_cohorts(run, *, early=False, full_first=False):
    result = []
    for generation in (2, 3) if early else (2, 3, 4):
        packet = CollectedMechanisms.restore(run.mechanism_collection(generation)["collection"])
        body = packet.payload()
        flows = [row["flow_id"] for row in body["rows"]]
        if generation == 2 and not full_first:
            flows = flows[:2] if early else flows[:3]
        labels = {flow: 0.125 * (int(flow[-1]) + 1) for flow in flows}
        result.append(cohort(body, labels))
    return result


def compare_oracle(policy, receipt, expected):
    assert not policy.training and policy.mean.device == torch.device("cuda:0")
    assert policy.mean.dtype == torch.float64
    assert receipt["objective_counts"] == expected["counts"]
    for name in (
        "mean",
        "scale",
        "support_min",
        "support_max",
        "point_coefficients",
        "interval_coefficients",
    ):
        actual = getattr(policy, name).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected[name], atol=1e-12, rtol=3e-11)
    np.testing.assert_array_equal(policy.action_mask.cpu().numpy(), expected["action_mask"])
    assert torch.count_nonzero(policy.point_coefficients[:, 0]).item() == 0
    assert torch.count_nonzero(policy.interval_coefficients[:, 0]).item() == 0
    residuals = [value for value in receipt["normalized_stationarity_norms"] if value is not None]
    solved = sum(
        count >= 2 for column, count in enumerate(expected["counts"]) if column not in (0, 12)
    )
    assert len(residuals) == solved
    assert all(value < 1e-10 for value in residuals)
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert digest(content) == receipt["receipt_sha256"] == policy.fit_receipt_sha256
    assert (
        receipt["policy_buffers_sha256"]
        == policy_signature(policy, strong=True)[1]["buffers_sha256"]
    )
    assert receipt["selection_performed"] is receipt["predictive_improvement_measured"] is False


@pytest.mark.parametrize("mode", ["closed_phases", "native_windows"])
@pytest.mark.parametrize("interval_weight", [0.0, 0.5])
def test_native_fit_matches_independent_losses_moments_and_normal_equations(
    source_run, mode, interval_weight
):
    config = protocol(source_run, mode=mode, interval_weight=interval_weight)
    sources = (
        (source_run,)
        if mode == "closed_phases"
        else (MechanismFitWindow(source_run, (2, 3, 4, 5), through_generation=6),)
    )
    before = source_run.snapshot()
    original = raw_cohorts(source_run)
    rows, weights, provenance = _cohorts(sources, config=config, device="cuda:0")
    assert [row["mature_rows"] for row in rows] == [3, 4, 4]
    assert [row["excluded_rows"] for row in rows] == [0, 0, 0]
    first = source_run.mechanism_collection(2)
    assert len(first["censored"]) == 1
    assert first["supervision"]["outcomes"][3] is None
    assert first["supervision"]["outcomes"][2]["available_at"] == moment(127)
    np.testing.assert_allclose(weights, [1 / 3] * 3, atol=0, rtol=0)
    assert provenance["observed_cohorts"] == 4 and provenance["no_label_cohorts"] == 1
    for actual, expected in zip(rows, original, strict=True):
        assert actual["features"] == expected["features"]
        for actual_target, expected_target in zip(
            actual["point"] + actual["interval"], expected["targets"], strict=True
        ):
            if expected_target is None:
                assert actual_target is None
            else:
                assert actual_target == pytest.approx(expected_target, abs=1e-12, rel=3e-11)
    expected = ridge(original, penalty=config.ridge, minimum=2, interval_weight=interval_weight)
    policy, receipt = fit_policy(sources, config=config, device="cuda:0")
    assert receipt["cohorts"] == receipt["decision_instants"] == 3
    compare_oracle(policy, receipt, expected)
    assert source_run.snapshot() == before


def test_closed_and_windowed_fit_share_buffers_and_do_not_advance_rng(source_run):
    config = protocol(source_run)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state("cuda:0").clone()
    closed, _ = fit_policy((source_run,), config=config, device="cuda:0")
    windowed, receipt = fit_policy(
        (MechanismFitWindow(source_run, (2, 3, 4, 5), through_generation=6),),
        config=replace(config, source_mode="native_windows"),
        device="cuda:0",
    )
    for name, value in closed.named_buffers():
        assert torch.equal(value, dict(windowed.named_buffers())[name])
    assert receipt["schema_version"] == 2
    assert receipt["kind"] == "native_window_cohort_mechanism_fit_receipt_v1"
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state("cuda:0"), cuda_rng)


def test_earlier_fit_ignores_later_received_labels_and_roundtrips_into_consumer(
    source_run, tmp_path
):
    config = replace(protocol(source_run, mode="native_windows"), fit_end=moment(129))
    request = MechanismFitWindow(source_run, (2, 3, 4), through_generation=4)
    policy, receipt = fit_policy((request,), config=config, device="cuda:0")
    expected = ridge(
        raw_cohorts(source_run, early=True), penalty=0.01, minimum=2, interval_weight=0.5
    )
    compare_oracle(policy, receipt, expected)
    assert receipt["cohorts"] == 2
    assert [row["decision_at"] for row in receipt["cohorts_used"]] == [moment(126), moment(127)]
    path = tmp_path / "fitted-policy.pt"
    torch.save(policy.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(receipt, sort_keys=True))
    state = torch.load(path, weights_only=True, map_location="cuda:0")
    restored = MechanismPolicy(
        **{name: state[name] for name, _ in policy.named_buffers()},
        interval_weight=policy.interval_weight,
        min_gain=policy.min_gain,
        catalog_id=policy.catalog_id,
        fit_receipt_sha256=policy.fit_receipt_sha256,
    )
    restored.load_state_dict(state)
    parent = source_run.consumer
    consumer = FrozenMechanismConsumer(
        parent.predictor,
        readout=parent.readout,
        calibration=CohortFeedback.restore(parent._initial_calibration),
        catalog=parent.catalog,
        policy=restored,
    )
    assert consumer.policy.fit_receipt_sha256 == receipt["receipt_sha256"]
    for name, value in policy.named_buffers():
        assert torch.equal(value, dict(consumer.policy.named_buffers())[name])
    saved_receipt = json.loads(path.with_suffix(".json").read_text())
    assert digest(saved_receipt) == digest(receipt)
    assert saved_receipt["protocol_sha256"] == config.fingerprint()


def test_fitted_support_can_force_reference_after_a_different_context_warmup(
    source_run, four_flow_source, tmp_path
):
    config = replace(protocol(source_run, mode="native_windows"), fit_end=moment(129))
    policy, _ = fit_policy(
        (MechanismFitWindow(source_run, (2, 3, 4), through_generation=4),),
        config=config,
        device="cuda:0",
    )
    index = FEATURES.index("previous_context_present")
    assert policy.support_min[index].item() == policy.support_max[index].item() == 1.0
    feature = policy.mean.clone()
    feature[index] = 0.0
    choice = policy(feature[None], torch.ones((1, 36), dtype=torch.bool, device="cuda:0"))
    assert not choice.supported.item() and choice.action_ids.item() == 0
    parent = source_run.consumer

    def consumer(*, fitted):
        return FrozenMechanismConsumer(
            parent.predictor,
            readout=parent.readout,
            calibration=CohortFeedback.restore(parent._initial_calibration),
            catalog=parent.catalog,
            **(dict(policy=policy) if fitted else dict(action=MechanismAction())),
        )

    source = varied(four_flow_source)
    phase = FinancialPhase("validation", moment(125), moment(130), moment(131), moment(132))
    assert config.fit_end <= phase.decision_start
    outputs = []
    for fitted in (False, True):
        with FinancialSession(
            tmp_path / str(fitted),
            native=source_run.native,
            consumer=consumer(fitted=fitted),
            codec=source["codec"],
            prefixes=source["prefixes"],
            retention=CompositeScoreConfig(SCALERS, capacity=4),
            phase=phase,
            world="fit_support_fixture",
            fold="0",
            admission="m3",
            block_rows=4,
        ) as run:
            for at in (125, 126, 127, 128):
                run.step(source["batches"][at], [], kind="warmup")
            emitted = run.step(source["batches"][130], [])
            outputs.append([(p.asset, p.decision_at, p.value) for p in emitted.predictions])
            event = run.mechanism_event(run.snapshot()["cursor"])["payload"]
            assert all(row["requested"] == "off/k1/none" for row in event["rows"])
            context = run.consumer.restore_state(run.mechanism_state()).context
            assert all(at < 0 for at in context.committed_at)
            assert all(at < 0 for at in context.pending_at)
            run.step([], [], kind="settlement", cutoff=phase.close_at, close_phase=True)
    assert outputs[0] == outputs[1]


def test_fit_rejects_cpu_before_opening_sources(source_run):
    with pytest.raises(ValueError, match="cuda:0"):
        fit_policy((), config=protocol(source_run), device="cpu")


def test_fit_rejects_autocast_before_opening_sources(source_run):
    with torch.autocast("cuda", dtype=torch.bfloat16), pytest.raises(ValueError, match="FP64"):
        fit_policy((), config=protocol(source_run), device="cuda:0")


def test_fit_rejects_excessive_condition_instead_of_publishing_coefficients(source_run):
    with pytest.raises(ValueError, match="condicionado"):
        fit_policy(
            (source_run,), config=replace(protocol(source_run), max_condition=1.0), device="cuda:0"
        )


@pytest.mark.parametrize("source_run", ["limited"], indirect=True)
@pytest.mark.parametrize("interval_weight", [0.0, 0.5])
def test_native_fit_preserves_missing_objectives_and_disables_unobserved_actions(
    source_run, interval_weight
):
    config = protocol(source_run, interval_weight=interval_weight)
    original = raw_cohorts(source_run)
    assert any(value is None for row in original for value in row["targets"])
    expected = ridge(original, penalty=0.01, minimum=2, interval_weight=interval_weight)
    policy, receipt = fit_policy((source_run,), config=config, device="cuda:0")
    compare_oracle(policy, receipt, expected)
    assert 0 in receipt["objective_counts"] and 3 in receipt["objective_counts"]
    for column, count in enumerate(receipt["objective_counts"]):
        if not count:
            assert receipt["augmented_system_conditions"][column] is None
            assert receipt["normalized_stationarity_norms"][column] is None
    assert not policy.action_mask.all().item()


@pytest.mark.parametrize("source_run", ["capacity"], indirect=True)
def test_native_fit_renormalizes_each_partially_observed_objective(source_run):
    config = protocol(source_run)
    original = raw_cohorts(source_run, full_first=True)
    expected = ridge(original, penalty=0.01, minimum=2, interval_weight=0.5)
    policy, receipt = fit_policy((source_run,), config=config, device="cuda:0")
    compare_oracle(policy, receipt, expected)
    assert {2, 3} <= set(receipt["objective_counts"])
    partial = [column for column, count in enumerate(receipt["objective_counts"]) if count == 2]
    assert partial
    for column in partial:
        assert [row["targets"][column] is not None for row in original] == [True, False, True]
        assert receipt["augmented_system_conditions"][column] is not None
