"""M2/M3 con la emisión del controlador y etiquetas manuales, sin ajustes."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_bridge_mechanism_session import engines
from test_financial_mechanisms import assert_same
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_mars_titan_session_parity import SCALERS, varied
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import MechanismPolicy

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))
PHASE = FinancialPhase("validation", moment(125), moment(126), moment(131), moment(132))


def configured(source, device="cpu", *, automatic=False, regional=True):
    data = engines(source, device, automatic=automatic)
    parent = data[3]
    reader = parent.readout
    catalog = parent.catalog if regional else MechanismCatalog()
    if not regional:
        reader = (
            EpisodicReadout(
                replace(reader.config, evidence_bridge=None), device=device, dtype=torch.float64
            )
            .eval()
            .requires_grad_(False)
        )
    options = dict(action=MechanismAction(route="k4cc"))
    if automatic:
        policy = parent.policy
        options = dict(
            policy=MechanismPolicy(
                **dict(policy.named_buffers()),
                interval_weight=policy.interval_weight,
                min_gain=policy.min_gain,
                catalog_id=catalog.fingerprint(),
            )
        )
    consumer = FrozenMechanismConsumer(parent.predictor, readout=reader, catalog=catalog, **options)
    return (*data[:3], consumer, data[4])


def retention(admission, *, scalers=SCALERS):
    return (
        MatureErrorConfig(capacity=4)
        if admission == "m2"
        else CompositeScoreConfig(scalers, capacity=4)
    )


def session(
    native,
    source,
    data,
    output,
    admission,
    *,
    resume=False,
    collect=False,
    phase=PHASE,
    block_rows=4,
):
    return FinancialSession(
        output,
        native=native,
        consumer=data[3],
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention(admission),
        phase=phase,
        world="mechanism_admission_fixture",
        fold="0",
        admission=admission,
        resume=resume,
        block_rows=block_rows,
        mechanism_collection=MechanismCollectionConfig() if collect else None,
    )


def feedback(native, issued, index):
    return [native.Feedback(p.id, 0, moment(index), (int(p.asset[-1]) + 1) * 0.125) for p in issued]


def assert_partition_close(actual, expected):
    """La recuperación exige bits iguales. Cambiar el lote conserva la tolerancia FP64 previa."""
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    elif type(expected) is dict:
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_partition_close(actual[key], expected[key])
    elif type(expected) in (list, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_partition_close(left, right)
    elif type(expected) is float:
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def inspect_bank(run, emitted):
    bundle = run._bundle(run.snapshot()["state"])
    bank, episodes = run._bank(bundle["bank"])
    assert bank.seen == run.snapshot()["applied"]
    for key, item in episodes.items():
        expected = emitted[item["flow_id"], item["prediction_at"]]
        assert item["issued_prediction"] == expected
        assert item["error"] == item["label"] - expected
        if run.admission == "m3":
            raw = bank.write_features[key]
            assert raw[0] == abs(item["label"] - expected)
            position = int(item["flow_id"][-1])
            index = next(i for i in (126, 127, 128, 130) if moment(i) == item["prediction_at"])
            if position in (2, 3):
                assert raw[2] == pytest.approx(2.0 * (index - 124) + position, rel=1e-6)
            else:
                assert raw[2] is None
            assert raw[3] == (position in (1, 3))
    if run.admission == "m2" and episodes:
        best = min(episodes, key=lambda key: (-abs(episodes[key]["error"]), key))
        assert bank.index_ids()["selective"] == (best,)
        assert bank.selective_scores[best] == abs(episodes[best]["error"])
    return bank, episodes


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("admission", ["m2", "m3"])
@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
def test_controller_reuses_mature_admission_and_matches_its_fixed_reader(
    native, four_flow_source, tmp_path, device, admission, regional, automatic
):
    source = varied(four_flow_source)
    data = configured(source, device, automatic=automatic, regional=regional)
    control_reader = (
        EpisodicReadout(
            replace(
                data[3].readout.config,
                refinements=2 if automatic else 4,
                episode_selection="first_read",
            ),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    fixed = FrozenFinancialConsumer(data[3].predictor, readout=control_reader)
    issued, expected, points = [], [], {}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device != "cpu" else None
    with (
        session(native, source, data, tmp_path / "controller", admission, collect=automatic) as run,
        FinancialSession(
            tmp_path / "fixed",
            native=native,
            consumer=fixed,
            codec=source["codec"],
            prefixes=source["prefixes"],
            retention=retention(admission),
            phase=PHASE,
            world="mechanism_admission_fixture",
            fold="0",
            admission=admission,
            block_rows=4,
        ) as reference,
    ):
        for target in (run, reference):
            target.step(source["batches"][125], [], kind="warmup")
        for index in (126, 127, 128, 130):
            prior_bundle = run._bundle(run.snapshot()["state"])
            prior_bank, _ = run._bank(prior_bundle["bank"])
            eligible = {key for values in prior_bank.index_ids().values() for key in values}
            event = run.step(source["batches"][index], feedback(native, issued, index))
            control = reference.step(source["batches"][index], feedback(native, expected, index))
            issued, expected = event.predictions, control.predictions
            assert [(p.asset, p.value) for p in issued] == [(p.asset, p.value) for p in expected]
            points.update({(p.asset, p.decision_at): p.value for p in issued})
            bank, _ = inspect_bank(run, points)
            other, _ = reference._bank(reference._bundle(reference.snapshot()["state"])["bank"])
            assert bank.snapshot() == other.snapshot()
            record = run.mechanism_event(run.snapshot()["cursor"])["payload"]
            assert all(
                r["executed_route"] == ("k2c" if automatic else "k4cc") for r in record["rows"]
            )
            assert (
                len({r.get("regional_readout", {}).get("snapshot_sha256") for r in record["rows"]})
                == 1
            )
            if regional:
                for row in record["rows"]:
                    trace = row["regional_readout"]
                    assert trace["snapshot_sha256"] == record["snapshot_sha256"]
                    for read in trace["reads"]:
                        assert set(read["evidence_ids"]) <= eligible
        assert run.diagnostics()["physical_slots"] == 4
        assert run.diagnostics()["retained"] <= 4
        assert run.snapshot()["issued"] == 16 and run.snapshot()["applied"] == 12
        modules = run._binding.code()
        for name in ("write_policy", "write_scores") if admission == "m3" else ("write_policy",):
            path = Path("src/mars_titan/memory") / f"{name}.py"
            assert modules[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
    assert all(parameter.grad is None for parameter in data[3].predictor.parameters())


@pytest.mark.parametrize("partition", ["validation", "calibration", "evaluation", "train"])
def test_m3_rejects_a_declared_scaling_window_after_its_allowed_boundary(
    native, four_flow_source, tmp_path, partition
):
    data = engines(four_flow_source)
    phase = replace(PHASE, partition=partition)
    end = phase.decision_end if partition == "train" else phase.decision_start
    config = CompositeScoreConfig(replace(SCALERS, decision_end=end + 1), capacity=4)
    with pytest.raises(ValueError, match="escala|escalas"):
        FinancialSession(
            tmp_path / "future",
            native=native,
            consumer=FrozenFinancialConsumer(data[3].predictor),
            codec=four_flow_source["codec"],
            prefixes=four_flow_source["prefixes"],
            retention=config,
            phase=phase,
            world="scalers_fixture",
            fold="0",
            admission="m3",
        )
    assert not (tmp_path / "future").exists()


@pytest.mark.parametrize("partition", ["validation", "calibration", "evaluation", "train"])
def test_m3_accepts_the_exclusive_end_boundary_without_executing_a_phase(
    native, four_flow_source, tmp_path, partition
):
    data = configured(four_flow_source)
    phase = replace(PHASE, partition=partition)
    end = phase.decision_end if partition == "train" else phase.decision_start
    config = CompositeScoreConfig(replace(SCALERS, decision_end=end), capacity=4)
    with FinancialSession(
        tmp_path / "boundary",
        native=native,
        consumer=data[3],
        codec=four_flow_source["codec"],
        prefixes=four_flow_source["prefixes"],
        retention=config,
        phase=phase,
        world="scalers_boundary_fixture",
        fold="0",
        admission="m3",
    ) as run:
        assert run.snapshot()["issued"] == run.snapshot()["applied"] == 0


def trajectory(
    native, source, data, output, admission, *, cut=None, reverse=False, changed_future=False
):
    run = session(
        native, source, data, output, admission, collect=True, block_rows=1 if reverse else 4
    )
    emitted, prior, points, banks = {}, [], {}, []
    try:
        run.step(source["batches"][125], [], kind="warmup")
        for index in (126, 127, 128, 129, 130):
            batches = [] if index == 129 else source["batches"][index]
            labels = feedback(native, prior, index)
            if changed_future and index >= 128:
                labels = [
                    native.Feedback(p.id, 0, moment(index), 10.0 + int(p.asset[-1])) for p in prior
                ]
            if reverse:
                batches, labels = list(reversed(batches)), list(reversed(labels))
            options = dict(kind="settlement" if index == 129 else "decision", cutoff=moment(index))
            if cut is not None and index == 128:
                before = run.snapshot()

                def fail(boundary):
                    if boundary == getattr(native.Boundary, cut):
                        raise RuntimeError("Corte de admisión del controlador")

                with pytest.raises(RuntimeError, match="Corte de admisión"):
                    run.step(batches, labels, fault=fail, **options)
                run.close()
                run = session(native, source, data, output, admission, resume=True, collect=True)
                if cut == "before_commit":
                    assert run.snapshot() == before
                    prior = run.step(batches, labels, **options).predictions
                else:
                    assert run.snapshot()["cursor"] == before["cursor"] + 1
                    prior = [p for p in run._executor.pending() if p.decision_at == moment(index)]
            else:
                prior = run.step(batches, labels, **options).predictions
            emitted.update({(p.asset, p.decision_at): p.value for p in prior})
            points.update(emitted)
            bank, _ = inspect_bank(run, emitted)
            banks.append(bank.snapshot())
        run.step([], [], kind="settlement", cutoff=PHASE.close_at, close_phase=True)
        assert run.snapshot()["applied"] == 12
        assert run.diagnostics()["pending"] == 0
        assert all(len(values) <= 4 for values in bank.index_ids().values())
        flows = tuple(sorted(run.fast_state()["references"]))
        state = data[3].predictor.export_state(run._fast_store.gather(run.fast_state(), flows))
        final = run.snapshot()
    finally:
        run.close()
    return dict(points=points, banks=banks, fast=state, final=final)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("admission", ["m2", "m3"])
@pytest.mark.parametrize("cut", ["before_commit", "committed"])
def test_controller_admission_recovers_without_repeating_offers(
    native, four_flow_source, tmp_path, device, admission, cut
):
    source = varied(four_flow_source)
    data = configured(source, device, automatic=True)
    reference = trajectory(native, source, data, tmp_path / "reference", admission)
    restored = trajectory(native, source, data, tmp_path / "restored", admission, cut=cut)
    assert_same(restored, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("admission", ["m2", "m3"])
def test_admission_uses_the_common_snapshot_before_feedback_and_is_order_invariant(
    native, four_flow_source, tmp_path, device, admission
):
    source = varied(four_flow_source)
    data = configured(source, device)
    reference = trajectory(native, source, data, tmp_path / "reference", admission)
    permuted = trajectory(native, source, data, tmp_path / "permuted", admission, reverse=True)
    assert reference["points"].keys() == permuted["points"].keys()
    for key, value in reference["points"].items():
        assert value == pytest.approx(permuted["points"][key], abs=1e-12, rel=1e-12)
    assert_partition_close(permuted["banks"], reference["banks"])
    assert_partition_close(permuted["fast"], reference["fast"])
    later = trajectory(native, source, data, tmp_path / "future", admission, changed_future=True)
    assert all(
        value == later["points"][key]
        for key, value in reference["points"].items()
        if key[1] <= moment(128)
    )
    assert reference["banks"][1] == later["banks"][1]
    assert reference["banks"][-1] != later["banks"][-1]


@pytest.mark.parametrize("admission", ["m2", "m3"])
def test_controller_bank_rejects_an_error_rebased_on_an_unissued_prediction(
    native, four_flow_source, tmp_path, admission
):
    source = varied(four_flow_source)
    data = configured(source)
    with session(native, source, data, tmp_path / "tampered", admission) as run:
        first = run.step(source["batches"][126], [])
        run.step(source["batches"][127], feedback(native, first.predictions, 127))
        bundle = run._bundle(run.snapshot()["state"])
        payload = copy.deepcopy(run._read(bundle["bank"], "bank"))
        row = payload["episodes"][next(iter(payload["episodes"]))]
        row["issued_prediction"] += 0.5
        row["error"] = row["label"] - row["issued_prediction"]
        before = run.snapshot()
        with pytest.raises(ValueError, match="error|errores"):
            run._bank(run._stage(payload, "bank"))
        assert run.snapshot() == before
