"""N2 en propuestas y generaciones de mecanismos, con supervisión manual."""

from dataclasses import replace

import pytest
import torch
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialPredictor
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from tests.memory.dendritic_fixtures import engine
from tests.memory.test_dendritic_session import source as source
from tests.memory.test_financial_mechanisms import assert_same
from tests.memory.test_financial_session import moment
from tests.memory.test_financial_session_controls import (
    no_target_estimation as no_target_estimation,
)
from tests.memory.test_spaced_regional_readout import options
from tests.memory.test_spaced_session import feedback
from tests.models.titans.test_mechanism_policy import policy

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def consumers(source, mode, *, context=None, automatic=False, bridge=None):
    base = engine(source, mode)
    model = FinancialPredictor(
        replace(base.predictor.config, head=QUANTILE_HEAD, session_context=context),
        dtype=torch.float64,
    )
    with torch.no_grad():
        model.active_dendrites.gain.fill_(1)
        if context is not None:
            model.session_context.output.copy_(torch.eye(32, dtype=torch.float64) * 0.2)
    model._seal_parameters()
    model.eval().requires_grad_(False)
    reader = base.readout
    catalog = MechanismCatalog()
    if bridge is not None:
        config = reader.config
        reader = (
            EpisodicReadout(
                replace(
                    config,
                    evidence_bridge=EvidenceBridgeConfig(
                        codec_id=config.codec_id,
                        hidden_size=config.hidden_size,
                        neighbors=config.neighbors,
                        temperature=config.temperature,
                        max_batch=config.max_batch,
                        max_working_bytes=config.max_working_bytes,
                        **options(bridge),
                    ),
                ),
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
        catalog = MechanismCatalog(reader_contract=reader.config.evidence_bridge.identity()["kind"])
    selection = (
        dict(policy=policy())
        if automatic
        else dict(
            action=MechanismAction(
                "parent" if context is not None else "off", "k4cr" if context else "k1"
            )
        )
    )
    return (
        FrozenFinancialConsumer(model, readout=reader) if context is None else None,
        FrozenMechanismConsumer(model, readout=reader, catalog=catalog, **selection),
    )


def open_session(native, source, consumer, path, *, resume=False, collection=None, block_rows=1):
    return FinancialSession(
        path,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=8, frontier=2, new_candidates=4),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        world="dendritic_mechanisms_fixture",
        fold="0",
        block_rows=block_rows,
        resume=resume,
        mechanism_collection=collection,
    )


@pytest.mark.parametrize("mode", ["observable", "shuffled_past_prefix", "constant"])
def test_k1_mechanisms_preserve_actual_n2_predictions_and_context(native, source, tmp_path, mode):
    ordinary, mechanisms = consumers(source, mode)
    with (
        open_session(native, source, ordinary, tmp_path / "ordinary") as plain,
        open_session(native, source, mechanisms, tmp_path / "mechanisms") as wrapped,
    ):
        for session in (plain, wrapped):
            warmup = session.step(source["batches"][125], [], kind="warmup")
            assert not warmup.predictions
        previous = [], []
        for at in (126, 127, 128):
            left = plain.step(source["batches"][at], feedback(native, previous[0], at))
            right = wrapped.step(source["batches"][at], feedback(native, previous[1], at))
            assert [(p.asset, p.value) for p in left.predictions] == [
                (p.asset, p.value) for p in right.predictions
            ]
            assert plain.dendritic_event()["entries"] == wrapped.dendritic_event()["entries"]
            archived = wrapped.mechanism_event(right.generation)
            assert archived["payload"]["dendritic_context"] == wrapped.dendritic_event()
            assert all(len(row["quantiles"]) == 5 for row in archived["payload"]["rows"])
            previous = left.predictions, right.predictions


@pytest.mark.parametrize(
    "context,automatic", [(None, False), ("market_memory_v1", False), (None, True)]
)
def test_collection_and_recovery_preserve_the_emitted_state_and_both_contexts(
    native, source, tmp_path, context, automatic
):
    _, consumer = consumers(source, "observable", context=context, automatic=automatic)
    collection_trajectory(native, source, tmp_path, consumer)


@pytest.mark.parametrize("context,bridge", [(None, "ring"), ("market_memory_v1", "tree")])
def test_regional_bridges_combine_dendrites_context_collection_and_resume(
    native, source, tmp_path, context, bridge
):
    _, consumer = consumers(source, "observable", context=context, bridge=bridge)
    collection_trajectory(native, source, tmp_path, consumer)


def collection_trajectory(native, source, tmp_path, consumer):
    expected = []
    for collected, resume in ((False, False), (True, False), (True, True)):
        path = tmp_path / f"{collected}-{resume}"
        collection = MechanismCollectionConfig() if collected else None
        run = open_session(native, source, consumer, path, collection=collection)
        run.step(source["batches"][125], [], kind="warmup")
        first = run.step(source["batches"][126], [])
        if collected:
            assert run.mechanism_collection(2)["supervision"]["outcomes"] == [None] * 4
        second = run.step(source["batches"][127], feedback(native, first.predictions, 127))
        if resume:
            before = run.snapshot()
            run.close()
            run = open_session(native, source, consumer, path, resume=True, collection=collection)
            assert run.snapshot() == before
        final = run.step(source["batches"][128], feedback(native, second.predictions, 128))
        flows = tuple(sorted(run.fast_state()["references"]))
        state = consumer.predictor.export_state(run._fast_store.gather(run.fast_state(), flows))
        auxiliary = run._binding.restore(run, run._bundle(run.snapshot()["state"]))
        expected.append(
            (
                [(p.asset, p.value) for p in final.predictions],
                state,
                consumer.export_state(auxiliary),
                run.dendritic_event()["entries"],
            )
        )
        run.step(
            [], feedback(native, final.predictions, 129), kind="settlement", cutoff=moment(129)
        )
        assert run.dendritic_event() is None
        assert run.mechanism_event(5)["payload"]["dendritic_context"] is None
        run.close()
    assert_same(expected[1], expected[0])
    assert_same(expected[2], expected[0])


def test_interrupted_publication_does_not_advance_either_context_or_the_alternatives(
    native, source, tmp_path
):
    _, consumer = consumers(source, "observable")
    collection = MechanismCollectionConfig()
    path = tmp_path / "interrupted"
    run = open_session(native, source, consumer, path, collection=collection)
    run.step(source["batches"][125], [], kind="warmup")
    before, trace = run.snapshot(), run.dendritic_event()

    def fail(boundary):
        if boundary == native.Boundary.before_commit:
            raise RuntimeError("interrupción N2 de fixture")

    with pytest.raises(RuntimeError, match="interrupción N2"):
        run.step(source["batches"][126], [], fault=fail)
    run.close()
    run = open_session(native, source, consumer, path, resume=True, collection=collection)
    assert run.snapshot() == before and run.dendritic_event() == trace
    with pytest.raises(ValueError):
        run.mechanism_collection(2)
    assert len(run.step(source["batches"][126], []).predictions) == 4
    assert run.mechanism_collection(2)["supervision"]["outcomes"] == [None] * 4
    run.close()


def test_a_new_control_hash_does_not_hide_a_changed_dendritic_trace(native, source, tmp_path):
    import json

    _, consumer = consumers(source, "observable")
    with open_session(native, source, consumer, tmp_path / "trace") as run:
        run.step(source["batches"][125], [], kind="warmup")
        run.step(source["batches"][126], [])
        before = run.snapshot()
        bundle = run._bundle(before["state"])
        run._verify_bundle(bundle)
        control = json.loads(json.dumps(run._read(bundle["control"], "control")))
        control["dendritic_context"]["entries"][0][5] = 0
        changed = dict(bundle, control=run._stage(control, "control"))
        from mars_titan.memory import dendritic_session

        with pytest.raises(ValueError, match="traza N2"):
            dendritic_session.verify(run, control, run._rows(source["batches"][126]), moment(126))
        with pytest.raises(ValueError, match="traza N2"):
            run._verify_bundle(changed)
        assert run.snapshot() == before
        assert run.mechanism_event(2)["payload"]["dendritic_context"] == run.dendritic_event()


@pytest.mark.parametrize("foreign", [False, True])
def test_standalone_collection_builds_the_whole_source_and_refuses_its_replacement(
    native, source, tmp_path, foreign
):
    from types import SimpleNamespace

    from mars_titan.memory.mechanism_collection import MechanismCollector
    from mars_titan.models.titans.dendritic_context import DendriticPlan

    _, consumer = consumers(source, "observable")
    with open_session(native, source, consumer, tmp_path / "standalone") as run:
        run.step(source["batches"][125], [], kind="warmup")
        first = run.step(source["batches"][126], [])
        run.step(source["batches"][127], feedback(native, first.predictions, 127))
        before = run.snapshot()
        bundle = run._bundle(before["state"])
        auxiliary = run._binding.restore(run, bundle)
        rows = run._rows(source["batches"][128])
        bank, _ = run._bank(bundle["bank"])
        snapshot = run._binding.snapshot(bank, "a" * 64, moment(128))
        builder = MechanismCollector(
            consumer,
            auxiliary,
            rows=rows,
            cutoff=moment(128),
            context_id="a" * 64,
            matures_at=(moment(129),) * 4,
            snapshot=snapshot,
        )
        event = builder.emission_event()
        assert event.dendritic_source_id is not None
        if foreign:
            altered = [
                SimpleNamespace(
                    flow_id=r.flow_id,
                    sample_id=r.sample_id,
                    prediction_at=r.prediction_at,
                    input_available_at=r.input_available_at,
                    inputs=dict(r.inputs),
                )
                for r in rows
            ]
            altered[1].inputs["prices"] = altered[1].inputs["prices"].copy()
            altered[1].inputs["prices"][0, 3] += 0.2
            builder._dendritic_plan = DendriticPlan.create(
                consumer.predictor.config.active_dendrites,
                altered,
                seed=consumer.predictor.config.seed,
                input_contract=source["spec"].fingerprint(),
                context_id="a" * 64,
            )
            with pytest.raises(ValueError, match="origen común"):
                builder.emission_event()
        else:
            for block in (rows[:2], rows[2:]):
                flows = tuple(r.flow_id for r in block)
                builder.add(block, run._fast_store.gather(run.fast_state(), flows))
            assert len(builder.finish().payload()["groups"][0]["candidates"]) == 36
        assert run.snapshot() == before


@pytest.mark.cuda
def test_cuda_mechanisms_reproduce_the_next_fp32_prediction_after_resume(native, source, tmp_path):
    assert torch.cuda.is_available(), "La comprobación necesita CUDA disponible"
    prior = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        points = []
        for name, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
            base = engine(source, "observable", dtype=torch.float32, device=device)
            consumer = FrozenMechanismConsumer(base.predictor, readout=base.readout)
            assert consumer.predictor.head.weight.dtype is torch.float32
            assert consumer.predictor.head.weight.device == torch.device(device)
            path = tmp_path / name
            run = open_session(native, source, consumer, path)
            run.step(source["batches"][125], [], kind="warmup")
            first = run.step(source["batches"][126], [])
            second = run.step(source["batches"][127], feedback(native, first.predictions, 127))
            trace, snapshot = run.dendritic_event(), run.snapshot()
            run.close()
            run = open_session(native, source, consumer, path, resume=True)
            assert run.snapshot() == snapshot and run.dendritic_event() == trace
            final = run.step(source["batches"][128], feedback(native, second.predictions, 128))
            points.append(torch.tensor([p.value for p in final.predictions]))
            run.close()
        # Tolerancia técnica FP32 de los fixtures N2. No cambia la del corpus.
        torch.testing.assert_close(points[0], points[1], rtol=2e-5, atol=2e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prior
