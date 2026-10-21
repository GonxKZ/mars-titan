"""Prefijos de recepción nativa con datos de prueba, sin ajuste de parámetros."""

import json

import numpy as np
import pytest
import torch
from test_financial_mechanisms import engines
from test_financial_session import moment
from test_financial_session_controls import four_flow_source as four_flow_source
from test_financial_session_controls import no_target_estimation as no_target_estimation
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native
from test_native_mechanism_collection import session

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def view(run, *, through=2, before=None, **limits):
    method = getattr(run, "mechanism_collection_at", None)
    assert method is not None, "Falta la vista histórica por recepción nativa"
    return method(
        1, through_generation=through, before=moment(128) if before is None else before, **limits
    )


@pytest.mark.parametrize("device", DEVICES)
def test_late_receipt_never_backdates_labels_and_prefix_survives_resume(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    output = tmp_path / "asof"
    with session(native, output, data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        early = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions[:2]]
        run.step([], early, kind="settlement", cutoff=moment(127))
        prefix = view(run)
        assert [t["value"] if t else None for t in prefix["supervision"]["outcomes"]] == [
            0.125,
            0.125,
            None,
            None,
        ]
        assert prefix["censored"] == []
        assert [r["generation"] for r in prefix["as_of"]["native_records"]] == [1, 2]
        assert prefix["as_of"]["native_records"][-1]["cutoff"] == moment(127)
        assert prefix["as_of"]["resolutions"] == [
            dict(
                generation=2,
                row_indices=[0, 1],
                native_record_sha256=prefix["as_of"]["native_records"][1]["sha256"],
            )
        ]
        run.step([], [], kind="settlement", cutoff=moment(128))
        late = [native.Feedback(p.id, 0, moment(127), 99.0) for p in issued.predictions[2:]]
        run.step([], late, kind="settlement", cutoff=moment(131), close_phase=True)
        before = run.snapshot()
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device != "cpu" else None
        assert view(run) == prefix
        assert run.snapshot() == before and torch.equal(torch.random.get_rng_state(), rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
        with pytest.raises(ValueError, match="corte|futuro|anterior"):
            view(run, through=4, before=moment(131))
        complete = view(run, through=4, before=moment(132))
        assert [t["value"] for t in complete["supervision"]["outcomes"]] == [
            0.125,
            0.125,
            99.0,
            99.0,
        ]
        assert complete["as_of"]["view_sha256"] != prefix["as_of"]["view_sha256"]
        assert not (output / "checkpoint-1.json").exists()
    with session(native, output, data, resume=True) as run:
        assert view(run) == prefix


def test_reader_ignores_latest_receipt_and_future_native_record(native, four_flow_source, tmp_path):
    data = engines(four_flow_source, "cpu")
    output = tmp_path / "future"
    with session(native, output, data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions[:2]]
        run.step([], labels, kind="settlement", cutoff=moment(127))
        expected = view(run)
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        bundle = run._bundle(run.snapshot()["state"])
        receipt = bundle["mechanism_collections"]["entries"][0]["receipt"]
        path = run.artifacts.directory / receipt["name"]
        saved_receipt, saved_record = path.read_bytes(), (output / "record-3.json").read_bytes()
        try:
            path.write_bytes(b"Recibo posterior no consultable")
            (output / "record-3.json").write_bytes(b"Registro posterior no consultable")
            assert view(run) == expected
        finally:
            path.write_bytes(saved_receipt)
            (output / "record-3.json").write_bytes(saved_record)


@pytest.mark.parametrize("device", DEVICES)
def test_censor_is_visible_only_after_real_closure(native, four_flow_source, tmp_path, device):
    data = engines(four_flow_source, device)
    with session(native, tmp_path / "censor", data) as run:
        run.step(four_flow_source["batches"][126], [])
        pending = view(run, through=1, before=moment(127))
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        assert view(run, through=1, before=moment(127)) == pending
        closed = view(run, through=2, before=moment(132))
        assert closed["supervision"]["outcomes"] == [None] * 4
        assert closed["supervision"]["clock"] is None
        assert {r["closed_at"] for r in closed["censored"]} == {moment(131)}
        assert {r["generation"] for r in closed["censored"]} == {2}


@pytest.mark.parametrize(
    "options", [{"max_records": 1}, {"max_bytes": 1}, {"through": 3}, {"before": moment(127)}]
)
def test_invalid_or_exhausted_view_keeps_confirmed_state(
    native, four_flow_source, tmp_path, options
):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "limits", data) as run:
        run.step(four_flow_source["batches"][126], [])
        run.step([], [], kind="settlement", cutoff=moment(127))
        before = run.snapshot()
        with pytest.raises(ValueError):
            view(run, **options)
        assert run.snapshot() == before


@pytest.mark.parametrize("field", ["target", "schema_version"])
def test_changed_native_feedback_is_rejected_by_trace_hash(
    native, four_flow_source, tmp_path, field
):
    data = engines(four_flow_source, "cpu")
    output = tmp_path / "changed"
    with session(native, output, data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        run.step(
            [],
            [native.Feedback(issued.predictions[0].id, 0, moment(127), 0.125)],
            kind="settlement",
            cutoff=moment(127),
        )
        path = output / "record-2.json"
        original = path.read_bytes()
        record = json.loads(original)
        if field == "target":
            record["feedback"][0]["target"] = 300.0
        else:
            record["schema_version"] = 3
        try:
            path.write_text(json.dumps(record))
            with pytest.raises(ValueError, match="huella|nativo|traza|registro"):
                view(run)
        finally:
            path.write_bytes(original)


@pytest.mark.parametrize("device", DEVICES)
def test_prefix_losses_ignore_later_decisions_and_match_mature_rows(
    native, four_flow_source, tmp_path, device
):
    from mars_titan.memory.mechanism_collection_record import CollectedMechanisms
    from mars_titan.memory.mechanism_supervision import MechanismSupervision

    data = engines(four_flow_source, device)
    with session(native, tmp_path / "losses", data) as run:
        first = run.step(four_flow_source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions[:2]]
        later = run.step(four_flow_source["batches"][127], labels)
        receipt = view(run)
        packet = CollectedMechanisms.restore(receipt["collection"])
        metric = MechanismSupervision.restore(packet, receipt["supervision"]).metrics(
            device=device
        )[0]
        assert metric["mature_rows"] == metric["pending_rows"] == 2
        assert metric["point_losses"][0] == pytest.approx(
            sum(abs(p.value - 0.125) for p in first.predictions[:2]) / 2, abs=1e-12, rel=1e-12
        )
        assert metric["point_gains"][0] == 0
        assert len(later.predictions) == 4
        assert len(receipt["supervision"]["outcomes"]) == 4
        entry = run._bundle(run.snapshot()["state"])["mechanism_collections"]["entries"][0]
        nbytes = entry["packet"]["bytes"] + sum(
            (run.output / f"record-{i}.json").stat().st_size for i in (1, 2)
        )
        assert receipt["as_of"]["source_bytes"] == nbytes
        assert view(run, max_bytes=nbytes) == receipt
        with pytest.raises(
            (ValueError, RuntimeError), match="bytes|size|tamaño|límite|limit|budget"
        ):
            view(run, max_bytes=nbytes - 1)


@pytest.mark.parametrize("device", DEVICES)
def test_native_prefix_exclusion_is_not_a_zero_label(native, four_flow_source, tmp_path, device):
    from mars_titan.memory.financial_session import FinancialPhase
    from mars_titan.models.titans.financial_inputs import validated_cpu_batch

    data = engines(four_flow_source, device)
    early = []
    for batch in four_flow_source["batches"][126]:
        early.append(
            validated_cpu_batch(
                dict(
                    inputs={name: np.zeros_like(value) for name, value in batch.inputs.items()},
                    presence=batch.presence.copy(),
                    sample_ids=[f"{flow}/{moment(64)}" for flow in batch.flow_ids],
                    prediction_at=np.array([moment(64)], dtype="datetime64[us]"),
                    input_available_at=np.array([moment(64)], dtype="datetime64[us]"),
                ),
                four_flow_source["spec"],
            )
        )
    phase = FinancialPhase("validation", moment(63), moment(64), moment(130), moment(131))
    with session(native, tmp_path / "excluded", data, phase=phase) as run:
        emitted = run.step(early, [])
        assert len(emitted.excluded) == 4
        receipt = view(run, through=1, before=moment(128))
        targets = receipt["supervision"]["outcomes"]
        assert [t["status"] for t in targets] == ["excluded"] * 4
        assert [t["value"] for t in targets] == [None] * 4
        assert {t["available_at"] for t in targets} == {moment(64)}
        assert receipt["censored"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"through": True},
        {"before": 0},
        {"max_records": True},
        {"max_records": 257},
        {"max_bytes": True},
        {"max_bytes": 1024**3 + 1},
    ],
)
def test_view_rejects_invalid_types_and_unbounded_work(native, four_flow_source, tmp_path, options):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "types", data) as run:
        run.step(four_flow_source["batches"][126], [])
        with pytest.raises(ValueError):
            view(run, **options)


def test_unconfirmed_and_nondecision_views_are_rejected(native, four_flow_source, tmp_path):
    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "empty", data) as run:
        with pytest.raises(ValueError, match="confirmada"):
            view(run, through=1)
        run.step(four_flow_source["batches"][125], [], kind="warmup")
        with pytest.raises(ValueError, match="decisión"):
            view(run, through=1)
    with session(native, tmp_path / "disabled", data, enabled=False) as run:
        with pytest.raises(ValueError, match="recogida"):
            view(run, through=1)


@pytest.mark.parametrize("device", DEVICES)
def test_receipt_watermark_can_differ_from_publication_order(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    with session(native, tmp_path / "watermark", data) as run:
        issued = run.step(four_flow_source["batches"][126], [])
        run.step(
            [],
            [native.Feedback(p.id, 0, moment(128), 1.0) for p in issued.predictions[:2]],
            kind="settlement",
            cutoff=moment(128),
        )
        run.step(
            [],
            [native.Feedback(p.id, 0, moment(127), 2.0) for p in issued.predictions[2:]],
            kind="settlement",
            cutoff=moment(129),
        )
        receipt = view(run, through=3, before=moment(130))
        assert receipt["supervision"]["clock"] == moment(128)
        assert receipt["as_of"]["native_records"][-1]["cutoff"] == moment(129)
        assert [t["value"] for t in receipt["supervision"]["outcomes"]] == [1.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize("device", DEVICES)
def test_same_assets_at_another_decision_do_not_share_results(
    native, four_flow_source, tmp_path, device
):
    data = engines(four_flow_source, device)
    with session(native, tmp_path / "identities", data) as run:
        first = run.step(four_flow_source["batches"][126], [])
        second = run.step(four_flow_source["batches"][127], [])
        run.step(
            [],
            [native.Feedback(p.id, 0, moment(128), 10.0) for p in first.predictions]
            + [native.Feedback(p.id, 0, moment(128), 0.125) for p in second.predictions[:2]],
            kind="settlement",
            cutoff=moment(128),
        )
        receipt = run.mechanism_collection_at(2, through_generation=3, before=moment(129))
        assert [t["value"] if t else None for t in receipt["supervision"]["outcomes"]] == [
            0.125,
            0.125,
            None,
            None,
        ]
        assert [r["generation"] for r in receipt["as_of"]["native_records"]] == [2, 3]
        assert receipt["as_of"]["resolutions"][0]["row_indices"] == [0, 1]


def test_changing_confirmed_head_during_a_view_rejects_the_result(
    native, four_flow_source, tmp_path, monkeypatch
):
    import mars_titan.memory.mechanism_collection_asof as api

    data = engines(four_flow_source, "cpu")
    with session(native, tmp_path / "concurrent", data) as run:
        run.step(four_flow_source["batches"][126], [])
        run.step([], [], kind="settlement", cutoff=moment(127))
        original = api.collection_at

        def advance(*args, **kwargs):
            result = original(*args, **kwargs)
            run.step([], [], kind="settlement", cutoff=moment(128))
            return result

        monkeypatch.setattr(api, "collection_at", advance)
        with pytest.raises(ValueError, match="cambió durante"):
            view(run)
        assert run.snapshot()["cursor"] == 3
