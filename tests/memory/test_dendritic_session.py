"""Contexto N2 común y publicación nativa con fixtures sin ajuste de etiquetas."""

import numpy as np
import pytest
import torch
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.data.price_windows import price_window_contract
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from tests.memory.test_financial_session import moment
from tests.memory.test_financial_session_controls import flow_source
from tests.memory.test_financial_session_controls import (
    no_target_estimation as no_target_estimation,
)
from tests.memory.test_mars_titan_session_parity import SCALERS

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    base = flow_source(tmp_path_factory.mktemp("dendritic-source"))
    old = base["spec"]
    representation = old.representation
    representation["price_window"] = price_window_contract(
        {"US": dict(start="2021-01-01", end="2023-12-31", decisions_sha256="f" * 64)},
        {"US": []},
    )
    spec = FinancialInputSpec(
        source_sha256=old.source_sha256,
        view_sha256=old.view_sha256,
        representation=representation,
        dimensions=old.dimensions | dict(prices=6),
        input_policy=old.input_policy,
    )
    batches = {}
    for at, blocks in base["batches"].items():
        batches[at] = []
        for block in blocks:
            inputs = {name: np.array(value, copy=True) for name, value in block.inputs.items()}
            inputs["prices"] = np.concatenate(
                [inputs["prices"], np.ones((1, 64, 1), np.float32)], axis=-1
            )
            batches[at].append(
                validated_cpu_batch(
                    dict(
                        inputs=inputs,
                        presence=block.presence,
                        sample_ids=block.sample_ids,
                        prediction_at=np.array(block.prediction_at, dtype="datetime64[us]"),
                        input_available_at=np.array(
                            block.input_available_at, dtype="datetime64[us]"
                        ),
                    ),
                    spec,
                )
            )
    return dict(base, spec=spec, codec=FrozenEpisodeCodec(spec), batches=batches)


def engine(source, mode, *, admission="m1", gain=1.0, device="cpu", dtype=torch.float64):
    config = FinancialConfig(
        source["spec"],
        variant="mac_online",
        hidden_size=32,
        active_dendrites=None
        if mode is None
        else dict(mode=mode, segments=4, min_assets=2, recent_returns=21, min_returns=42),
    )
    model = FinancialPredictor(config, dtype=dtype, device=device)
    if mode is not None:
        with torch.no_grad():
            model.active_dendrites.gain.fill_(gain)
    model._seal_parameters()
    readout = (
        EpisodicReadout(
            EpisodicReadoutConfig(
                source["codec"].fingerprint(),
                hidden_size=32,
                mode="no_bank" if admission == "m0" else "bank",
            ),
            dtype=dtype,
        )
        .to(device=device)
        .eval()
        .requires_grad_(False)
    )
    return FrozenFinancialConsumer(model.eval().requires_grad_(False), readout=readout)


def run(
    native,
    path,
    source,
    consumer,
    *,
    block_rows=1,
    resume=False,
    decision_start=125,
    admission="m1",
):
    retention = (
        MatureErrorConfig(capacity=8)
        if admission == "m2"
        else CompositeScoreConfig(SCALERS, capacity=8)
        if admission == "m3"
        else RetentionConfig(capacity=8, frontier=2, new_candidates=4)
    )
    return FinancialSession(
        path,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention,
        phase=FinancialPhase(
            "validation", moment(125), moment(decision_start), moment(200), moment(201)
        ),
        admission=admission,
        world="dendritic_fixture",
        fold="0",
        block_rows=block_rows,
        resume=resume,
    )


@pytest.mark.parametrize("mode", ["observable", "shuffled_past_prefix", "constant"])
def test_physical_blocks_and_recovery_keep_the_same_context_and_aligned_predictions(
    native, tmp_path, source, mode
):
    consumer = engine(source, mode)
    results = []
    for size in (1, 4):
        path = tmp_path / str(size)
        session = run(native, path, source, consumer, block_rows=size)
        first = session.step(source["batches"][125], [])
        context = session.dendritic_event()
        assert len(context["entries"]) == 4 and all(row[5] > 0 for row in context["entries"])
        altered = session.dendritic_event()
        altered["entries"][0][5] = 0
        assert session.dendritic_event() == context
        session.close()
        session = run(native, path, source, consumer, block_rows=size, resume=True)
        assert session.dendritic_event() == context
        second = session.step(
            source["batches"][126],
            [native.Feedback(p.id, 0, moment(126), 0.125) for p in first.predictions],
        )
        assert len(second.predictions) == 4
        third = session.step(source["batches"][127], [])
        control = session._read(session._bundle(session.snapshot()["state"])["control"], "control")
        assert control["snapshot_bytes"] > 0
        results.append(
            dict(
                points=[(p.asset, p.value) for p in third.predictions],
                context=session.dendritic_event(),
                diagnostics=session.diagnostics(),
            )
        )
        session.close()
    assert [r[0] for r in results[0]["points"]] == [r[0] for r in results[1]["points"]]
    torch.testing.assert_close(
        torch.tensor([r[1] for r in results[0]["points"]], dtype=torch.float64),
        torch.tensor([r[1] for r in results[1]["points"]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert results[0]["context"]["entries"] == results[1]["context"]["entries"]
    assert results[0]["diagnostics"]["admitted"] == results[1]["diagnostics"]["admitted"] == 4


def test_failed_publication_restores_the_confirmed_context_and_settlement_has_none(
    native, tmp_path, source
):
    consumer = engine(source, "observable")
    path = tmp_path / "rollback"
    session = run(native, path, source, consumer)
    issued = session.step(source["batches"][125], [])
    previous = session.snapshot()
    context = session.dendritic_event()

    def fail(boundary):
        if boundary == native.Boundary.before_commit:
            raise RuntimeError("fallo N2 de fixture")

    with pytest.raises(RuntimeError, match="fallo N2"):
        session.step(source["batches"][126], [], fault=fail)
    session = run(native, path, source, consumer, resume=True)
    assert session.snapshot() == previous and session.dendritic_event() == context
    # La transacción fallida debe repetirse con el mismo evento antes de avanzar.
    session.step(source["batches"][126], [])
    session.step(
        [],
        [native.Feedback(p.id, 0, moment(126), 0.125) for p in issued.predictions],
        kind="settlement",
        cutoff=moment(127),
    )
    assert session.dendritic_event() is None
    session.close()


def test_warmup_publishes_context_without_issuing_predictions(native, tmp_path, source):
    consumer = engine(source, "observable")
    session = run(native, tmp_path / "warmup", source, consumer, decision_start=126)
    result = session.step(source["batches"][125], [], kind="warmup")
    assert not result.predictions and session.dendritic_event() is not None
    assert session.diagnostics()["pending"] == 0
    session.step(source["batches"][126], [])
    session.close()


@pytest.mark.parametrize("admission", ["m0", "m2", "m3"])
def test_registered_admissions_keep_maturation_and_the_dendritic_context_separate(
    native, tmp_path, source, admission
):
    consumer = engine(source, "observable", admission=admission)
    path = tmp_path / admission
    session = run(native, path, source, consumer, admission=admission)
    issued = session.step(source["batches"][125], [])
    assert session.diagnostics()["admitted"] == 0
    first = session.dendritic_event()
    session.step(
        source["batches"][126],
        [native.Feedback(p.id, 0, moment(126), 0.125) for p in issued.predictions],
    )
    assert session.diagnostics()["admitted"] == (0 if admission == "m0" else 4)
    session.close()
    session = run(native, path, source, consumer, admission=admission, resume=True)
    assert session.dendritic_event()["configuration"] == first["configuration"]
    session.step(source["batches"][127], [])
    session.close()


def test_initial_gain_keeps_native_predictions_exactly_equal_to_the_existing_consumer(
    native, tmp_path, source
):
    outcomes = []
    for name, mode in (("base", None), ("n2", "observable")):
        consumer = engine(source, mode, gain=0.0)
        session = run(native, tmp_path / name, source, consumer)
        predictions = []
        pending = []
        for at in (125, 126, 127):
            result = session.step(
                source["batches"][at],
                [native.Feedback(p.id, 0, moment(at), 0.125) for p in pending],
            )
            predictions.extend((p.asset, p.value) for p in result.predictions)
            pending = result.predictions
        outcomes.append(predictions)
        if mode is None:
            with pytest.raises(ValueError):
                session.dendritic_event()
        session.close()
    assert outcomes[0] == outcomes[1]


def test_confirmed_trace_is_recomputed_from_the_actual_inputs_and_rejects_changes(
    native, tmp_path, source
):
    import copy

    from mars_titan.memory import dendritic_session

    consumer = engine(source, "observable")
    session = run(native, tmp_path / "trace", source, consumer)
    session.step(source["batches"][125], [])
    previous = session.snapshot()
    bundle = session._bundle(previous["state"])
    rows = tuple(row for ref in bundle["current_inputs"] for row in session._input_rows(ref))
    control = session._read(bundle["control"], "control")
    dendritic_session.verify(session, control, rows, bundle["cutoff"])
    with pytest.raises(ValueError):
        dendritic_session.verify(session, control, rows, bundle["cutoff"] + 1)
    with pytest.raises(ValueError):
        dendritic_session.verify(session, control, (), bundle["cutoff"])
    for field in ("configuration", "columns", "entries"):
        bad = copy.deepcopy(control)
        if field == "entries":
            bad["dendritic_context"][field][0][5] = 0
        else:
            bad["dendritic_context"][field] = None
        with pytest.raises(ValueError):
            dendritic_session.verify(session, bad, rows, bundle["cutoff"])
    bad = copy.deepcopy(control)
    bad["dendritic_context"]["entries"][0][5] = 0
    forged = dict(bundle, control=session._stage(bad, "control"))
    with pytest.raises(ValueError, match="traza N2"):
        session._verify_bundle(forged, event_kind="decision")
    assert session.snapshot() == previous
    session.close()


@pytest.mark.cuda
def test_cuda_native_context_snapshot_and_resume_match_fp32_cpu(native, tmp_path, source):
    prior = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        points = []
        for name, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
            consumer = engine(source, "observable", dtype=torch.float32, device=device)
            path = tmp_path / name
            session = run(native, path, source, consumer)
            session.step(source["batches"][125], [])
            previous = session.dendritic_event()
            session.close()
            session = run(native, path, source, consumer, resume=True)
            assert session.dendritic_event() == previous
            result = session.step(source["batches"][126], [])
            points.append(torch.tensor([p.value for p in result.predictions]))
            session.close()
        torch.testing.assert_close(points[0], points[1], rtol=2e-5, atol=2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prior
