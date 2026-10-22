"""Alternativas y etiquetas confirmadas por la generación nativa, sin aprendizaje."""

import importlib
import json

import pytest
import torch
from test_financial_mechanisms import assert_same, engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def config(**options):
    try:
        api = importlib.import_module("mars_titan.memory.native_mechanism_collection")
    except ModuleNotFoundError:
        pytest.fail("Falta la recogida de alternativas ligada a la generación nativa")
    return api.MechanismCollectionConfig(**options)


def session(native, output, data, *, enabled=True, collection=None, resume=False, phase=None):
    options = {"mechanism_collection": collection or config()} if enabled else {}
    return FinancialSession(
        output,
        native=native,
        consumer=data[3],
        codec=data[2],
        prefixes=data[4],
        retention=RetentionConfig(capacity=4, frontier=2, new_candidates=4),
        phase=phase
        or FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        world="collection_fixture",
        fold="0",
        resume=resume,
        block_rows=2,
        **options,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_collection_is_optional_and_preserves_real_emissions_and_bank_errors(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    with (
        session(native, tmp_path / "off", data, enabled=False) as off,
        session(native, tmp_path / "on", data) as on,
    ):
        off.step(four_flow_source["batches"][125], [], kind="warmup")
        on.step(four_flow_source["batches"][125], [], kind="warmup")
        with pytest.raises(ValueError, match="recogida|colección|decisión"):
            on.mechanism_collection(1)
        a = off.step(four_flow_source["batches"][126], [])
        b = on.step(four_flow_source["batches"][126], [])
        assert [(p.asset, p.value) for p in a.predictions] == [
            (p.asset, p.value) for p in b.predictions
        ]
        assert_same(
            data[3].predictor.export_state(
                on._fast_store.gather(on.fast_state(), tuple(sorted(on.fast_state()["references"])))
            ),
            data[3].predictor.export_state(
                off._fast_store.gather(
                    off.fast_state(), tuple(sorted(off.fast_state()["references"]))
                )
            ),
        )
        packet = on.mechanism_collection(2)
        assert len(packet["collection"]["payload_utf8"]) > 0
        assert packet["supervision"]["outcomes"] == [None] * 4
        assert packet["censored"] == []
        losses = on.mechanism_collection_metrics(2, device=device)
        assert losses[0]["mature_rows"] == 0 and losses[0]["point_losses"] == [None] * 12
        labels_a = [native.Feedback(p.id, 0, moment(127), 0.125) for p in a.predictions[:2]]
        labels_b = [native.Feedback(p.id, 0, moment(127), 0.125) for p in b.predictions[:2]]
        off.step(four_flow_source["batches"][127], labels_a)
        on.step(four_flow_source["batches"][127], labels_b)
        losses = on.mechanism_collection_metrics(2, device=device)[0]
        assert losses["mature_rows"] == losses["pending_rows"] == 2
        assert losses["point_losses"][0] == pytest.approx(
            sum(abs(p.value - 0.125) for p in b.predictions[:2]) / 2, rel=1e-12, abs=1e-12
        )
        assert losses["point_gains"][0] == 0
        bundle = on._bundle(on.snapshot()["state"])
        _, episodes = on._bank(bundle["bank"])
        assert all(e["error"] == e["label"] - e["issued_prediction"] for e in episodes.values())
        assert_same(
            data[3].predictor.export_state(
                on._fast_store.gather(on.fast_state(), tuple(sorted(on.fast_state()["references"])))
            ),
            data[3].predictor.export_state(
                off._fast_store.gather(
                    off.fast_state(), tuple(sorted(off.fast_state()["references"]))
                )
            ),
        )
        assert on.diagnostics() == off.diagnostics()
        assert on.model_id != off.model_id
        assert "mechanism_collections" not in off._bundle(off.snapshot()["state"])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "boundary", ["record_written", "checkpoint_written", "before_commit", "committed"]
)
def test_cut_recovery_keeps_collection_partial_supervision_and_next_decision(
    native, four_flow_source, tmp_path, device, boundary
):
    data = engines(four_flow_source, device, online=True)
    with session(native, tmp_path / "reference", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions[:2]]
        run.step(four_flow_source["batches"][127], labels)
        before = run.snapshot(), run.mechanism_collection(2), run.mechanism_collection(3)
        run.step(four_flow_source["batches"][128], [])
        after = run.snapshot(), run.mechanism_collection(2), run.mechanism_collection(4)
    with session(native, tmp_path / "cut", data) as run:
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        run.step(four_flow_source["batches"][126], [])

        def fail(point):
            if point == getattr(native.Boundary, boundary):
                raise RuntimeError("Corte técnico de recogida")

        with pytest.raises(RuntimeError, match="Corte técnico de recogida"):
            run.step(four_flow_source["batches"][127], labels, fault=fail)
    with session(native, tmp_path / "cut", data, resume=True) as run:
        if boundary != "committed":
            run.step(four_flow_source["batches"][127], labels)
        assert_same(
            (run.snapshot(), run.mechanism_collection(2), run.mechanism_collection(3)), before
        )
        run.step(four_flow_source["batches"][128], [])
        assert_same(
            (run.snapshot(), run.mechanism_collection(2), run.mechanism_collection(4)), after
        )
        assert run.snapshot()["applied"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_pruning_retains_finished_collections_and_native_label_provenance(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    output = tmp_path / "retained"
    with session(native, output, data) as run:
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = run.mechanism_collection(1)
        assert all(t["value"] == 0.125 for t in expected["supervision"]["outcomes"])
        run.step([], [], kind="settlement", cutoff=moment(128))
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        assert not (output / "checkpoint-1.json").exists()
        assert run.mechanism_collection(1) == expected
    with session(native, output, data, resume=True) as run:
        assert run.mechanism_collection(1) == expected
        assert run.mechanism_collection_metrics(1, device=device)[0]["pending_rows"] == 0
        head = json.loads((output / "latest.json").read_text())
        assert head["generation"] == 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_labels", [False, True])
def test_administrative_boundary_censors_without_fake_label_timestamps(
    native, four_flow_source, tmp_path, device, with_labels
):
    from mars_titan.models.titans.financial_inputs import FINAL_TEST_US

    data = engines(four_flow_source, device, online=True)
    phase = FinancialPhase("validation", moment(125), moment(126), moment(130), FINAL_TEST_US)
    output = tmp_path / "closed"
    with session(native, output, data, phase=phase) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        labels = (
            [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
            if with_labels
            else []
        )
        result = run.step([], labels, kind="settlement", cutoff=FINAL_TEST_US, close_phase=True)
        receipt = run.mechanism_collection(1)
        if with_labels:
            assert len(result.applied) == 4 and receipt["censored"] == []
            assert {r["available_at"] for r in receipt["supervision"]["outcomes"]} == {moment(127)}
        else:
            assert len(result.finalized) == 4
            assert receipt["supervision"]["outcomes"] == [None] * 4
            assert {r["closed_at"] for r in receipt["censored"]} == {FINAL_TEST_US}
        metric = run.mechanism_collection_metrics(1, device=device)[0]
        assert metric["pending_rows"] == 0
        assert metric["censored_rows"] == (0 if with_labels else 4)
        assert metric["mature_rows"] == (4 if with_labels else 0)
        if not with_labels:
            assert metric["point_losses"] == [None] * 12
    with session(native, output, data, phase=phase, resume=True) as run:
        assert run.mechanism_collection(1) == receipt


@pytest.mark.parametrize("device", DEVICES)
def test_collection_receives_labels_omitted_by_calibration_capacity(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device, online=True, max_rows=1)
    with session(native, tmp_path / "capacity", data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        assert run.mechanism_event(2)["payload"]["received_prediction_ids"] == []
        metric = run.mechanism_collection_metrics(1, device=device)[0]
        assert metric["mature_rows"] == 4 and metric["pending_rows"] == 0
        assert metric["point_losses"][0] is not None
        assert run.snapshot()["applied"] == 4


def test_collection_capacity_failure_keeps_last_confirmed_generation(
    native, four_flow_source, tmp_path
):
    from mars_titan.memory.mechanism_history import HistoryCapacityError

    data = engines(four_flow_source, "cpu")
    limit = config(max_decisions=1)
    output = tmp_path / "limited"
    with session(native, output, data, collection=limit) as run:
        run.step(four_flow_source["batches"][126], [])
        before = (output / "latest.json").read_bytes()
        first = run.mechanism_collection(1)
        with pytest.raises(HistoryCapacityError, match="límite|presupuesto"):
            run.step(four_flow_source["batches"][127], [])
    assert (output / "latest.json").read_bytes() == before
    with session(native, output, data, collection=limit, resume=True) as run:
        assert run.snapshot()["cursor"] == 1
        assert run.mechanism_collection(1) == first


@pytest.mark.parametrize(
    "field", ["label", "native", "missing_resolution", "pending", "censor", "clock"]
)
def test_resealed_collection_receipt_cannot_fabricate_mature_evidence(
    native, four_flow_source, tmp_path, field
):
    import copy

    from mars_titan.models.titans.mechanism_selection import digest

    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / field, data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions[:2]]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        bundle = copy.deepcopy(run._bundle(run.snapshot()["state"]))
        entry = bundle["mechanism_collections"]["entries"][0]
        receipt = run._read(entry["receipt"], "mechanism_receipt")
        if field == "label":
            receipt["supervision"]["outcomes"][0]["value"] += 1
            body = {k: v for k, v in receipt["supervision"].items() if k != "state_sha256"}
            receipt["supervision"]["state_sha256"] = digest(body)
        elif field == "clock":
            receipt["supervision"]["clock"] = moment(128)
            body = {k: v for k, v in receipt["supervision"].items() if k != "state_sha256"}
            receipt["supervision"]["state_sha256"] = digest(body)
        elif field == "native":
            receipt["resolutions"][0]["native_record_sha256"] = "d" * 64
        elif field == "missing_resolution":
            receipt["resolutions"] = []
        elif field == "pending":
            entry["pending"] = []
        else:
            receipt["censored"] = [
                dict(
                    row_index=2, generation=2, closed_at=moment(131), native_record_sha256="d" * 64
                )
            ]
        entry["receipt"] = run._stage(receipt, "mechanism_receipt")
        with pytest.raises(ValueError):
            run._binding.collections.verify(run, bundle, strong=True)


@pytest.mark.parametrize("value", [0, 257, True, 1.5])
def test_invalid_collection_limits_fail_before_opening_a_session(value):
    with pytest.raises(ValueError):
        config(max_decisions=value)


@pytest.mark.parametrize("field", ["disabled", "method", "limit"])
def test_collection_runtime_cannot_change_after_session_identification(
    native, four_flow_source, tmp_path, field
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / field, data) as run:
        run.step(four_flow_source["batches"][126], [])
        before = (run.output / "latest.json").read_bytes()
        if field == "disabled":
            run._binding.collections = None
        elif field == "method":
            run._binding.collections.resolve = lambda *args, **kwargs: {}
        else:
            object.__setattr__(run._collection_config, "max_decisions", 256)
        with pytest.raises(ValueError, match="cambi|sustitui|configuraci"):
            run.step([], [], kind="settlement", cutoff=moment(127))
        assert (run.output / "latest.json").read_bytes() == before


@pytest.mark.parametrize("device", DEVICES)
def test_all_36_actions_use_the_populated_prefix_and_actual_selected_emission(
    native, four_flow_source, tmp_path, device
):
    from test_mechanism_policy import policy

    from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    data = engines(four_flow_source, device, online=True)
    parent = data[3]
    reader = (
        EpisodicReadout(
            EpisodicReadoutConfig(data[2].fingerprint(), hidden_size=32),
            device=device,
            dtype=torch.float64,
        )
        .eval()
        .requires_grad_(False)
    )
    selector = policy(device=device)
    selector.support_max.fill_(100)
    selector.point_coefficients.zero_()
    selector.interval_coefficients.zero_()
    selector.point_coefficients[-1, 10] = 2
    engine = FrozenMechanismConsumer(
        parent.predictor,
        readout=reader,
        calibration=parent.initial_state().calibration,
        policy=selector,
    )
    data = (*data[:3], engine, data[4])
    with session(native, tmp_path / "all", data) as run:
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions]
        run.step(four_flow_source["batches"][127], labels)
        before = run.mechanism_state()
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(0) if device != "cpu" else None
        issued = run.step(four_flow_source["batches"][128], [])
        packet = CollectedMechanisms.restore(run.mechanism_collection(3)["collection"]).payload()
        candidates = packet["groups"][0]["candidates"]
        assert len(candidates) == 36 and all(c["status"] == "observed" for c in candidates)
        assert packet["work"]["core_observations"] == 8
        assert packet["work"]["reader_refinements"] == 72
        assert packet["observation"]["observation"]["previous_cutoff"] == moment(127)
        assert packet["observation"]["observation"]["snapshot_sha256"] is not None
        assert all(
            row["requested"] == "parent/k4cr/none"
            for row in run.mechanism_event(3)["payload"]["rows"]
        )
        assert [c[2] for c in candidates[30]["quantiles"]] == pytest.approx(
            [p.value for p in issued.predictions], rel=1e-12, abs=1e-12
        )
        assert before["cutoff"] == moment(127)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if gpu_rng is not None:
            assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))


def test_empty_phase_and_changed_collection_contract_recover_without_opening_data(
    native, four_flow_source, tmp_path
):
    data = engines(four_flow_source, "cpu")
    output = tmp_path / "empty"
    with session(native, output, data) as run:
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        before = run.snapshot()
        with pytest.raises(ValueError, match="recogida|decisión"):
            run.mechanism_collection(1)
    with session(native, output, data, resume=True) as run:
        assert run.snapshot() == before
    with pytest.raises((ValueError, RuntimeError)):
        session(native, output, data, resume=True, collection=config(max_decisions=63))


@pytest.mark.parametrize("kind", ["packet", "receipt"])
def test_changed_collection_file_is_rejected_on_resume_without_changing_head(
    native, four_flow_source, tmp_path, kind
):
    data = engines(four_flow_source, "cpu")
    output = tmp_path / kind
    with session(native, output, data) as run:
        run.step(four_flow_source["batches"][126], [])
        entry = run._bundle(run.snapshot()["state"])["mechanism_collections"]["entries"][0]
        path = output / "artifacts" / entry[kind]["name"]
    head = (output / "latest.json").read_bytes()
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(ValueError):
        session(native, output, data, resume=True)
    assert (output / "latest.json").read_bytes() == head
