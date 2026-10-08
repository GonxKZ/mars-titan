"""Cuatro recorridos C y retención con parámetros congelados y sin optimizador."""

import itertools
import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from test_financial_session import moment
from test_frozen_financial import frozen_backend as frozen_backend
from test_native_episode_backend import native as native

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.titans.episodic_readout import (
    EpisodicReadout,
    EpisodicReadoutConfig,
    copy_readout_parameters,
)
from mars_titan.models.titans.financial import (
    FinancialConfig,
    FinancialPredictor,
    copy_paired_parameters,
)
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training import corpus_targets
from mars_titan.training.cohort_contract import representation_hash, representation_identity
from mars_titan.training.corpus_inputs import CorpusDataset, _price_contexts
from mars_titan.training.prefix_eligibility import PrefixTargetVerifier
from tests.training.test_historical_corpus_inputs import historical_edition

FLOWS = tuple(f"US/T{index:04d}" for index in range(4))
DECISIONS = (125, 126, 127, 128, 130)


@pytest.fixture(scope="module", autouse=True)
def no_target_estimation():
    def forbidden(*args, **kwargs):
        pytest.fail("La prueba técnica no puede estimar etiquetas residuales")

    with pytest.MonkeyPatch.context() as guard:
        for name in ("prepare_corpus_targets", "residual_targets", "residual_targets_array"):
            guard.setattr(corpus_targets, name, forbidden)
        yield


def test_fixture_construction_does_not_estimate_residuals(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("El fixture de composición no puede ajustar etiquetas residuales")

    monkeypatch.setitem(globals(), "prepare_corpus_targets", forbidden)
    factory = SimpleNamespace(mktemp=lambda _: tmp_path)
    source = four_flow_source.__wrapped__(factory)
    assert set(source["batches"]) == set(DECISIONS)
    assert source["label_origin"] == "manual_fixture_no_estimation"


def manual_supervision(manifest, prepared, output):
    metadata = json.loads(manifest.read_text())
    clock = MarketClock("US", "2021-01-01", "2023-12-31")
    positions = {at: index for index, at in enumerate(clock.decisions)}
    assets, representation = [], None
    for asset in metadata["assets"]:
        relative = f"{asset['market']}/{asset['symbol']}"
        encoded = prepared.parent / "samples" / relative
        specification = representation_identity(
            json.loads((encoded / "manifest.json").read_text()), input_policy=HISTORICAL_MASKED
        )
        assert representation is None or representation == specification
        representation = specification
        dates = pq.read_table(
            encoded / "samples.parquet", columns=["prediction_at"], use_threads=False
        )["prediction_at"].to_pylist()
        labels = []
        for index, at in enumerate(dates):
            accepted = index in (0, 2)
            labels.append(
                dict(
                    sample_row=index,
                    prediction_at=at,
                    target_available_at=clock.decisions[positions[at] + 1] if accepted else None,
                    target=0.125 if accepted else None,
                    partition=("train" if index == 0 else "validation") if accepted else None,
                    reason="accepted"
                    if accepted
                    else "target_crosses_partition_boundary"
                    if index == 1
                    else "target_after_cutoff",
                    cohort_id=metadata["cohort_id"],
                )
            )
        destination = output / "labels" / relative
        destination.mkdir(parents=True)
        schema = corpus_targets.LABEL_SCHEMA.append(pa.field("cohort_id", pa.string()))
        path = destination / "labels.parquet"
        pq.write_table(pa.Table.from_pylist(labels, schema=schema), path)
        assets.append(
            dict(
                **asset,
                fingerprint="manual_fixture_no_estimation",
                samples=len(labels),
                prices_sha256=sha256(prepared / relative / "prices.parquet"),
                samples_sha256=sha256(encoded / "samples.parquet"),
                labels_sha256=sha256(path),
                representation_sha256=representation_hash(
                    specification, input_policy=HISTORICAL_MASKED
                ),
                counts=dict(train=1, validation=1),
            )
        )
    result = dict(
        metadata,
        kind="corpus_supervision",
        assets=assets,
        representation=representation,
        roots=dict(
            prepared=str(prepared),
            samples=str(prepared.parent / "samples"),
            labels=str(output / "labels"),
        ),
        counts=dict(train=len(assets), validation=len(assets)),
        configuration=dict(
            backend="manual_fixture_no_estimation", source_manifest_sha256=sha256(manifest)
        ),
    )
    path = output / "manifest.json"
    path.write_text(json.dumps(result))
    return path


@pytest.fixture(scope="module")
def four_flow_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("four_flow_controls")
    manifest, prepared = historical_edition(root)
    metadata = json.loads(manifest.read_text())
    asset, coverage = metadata["assets"][0], metadata["coverage"][0]
    assets, rows = [], []
    for flow in FLOWS:
        symbol = flow.split("/")[1]
        for base in (prepared, root / "samples"):
            destination = base / flow
            shutil.copytree(base / "US/AAA", destination)
            path = destination / "manifest.json"
            value = json.loads(path.read_text())
            path.write_text(json.dumps(dict(value, symbol=symbol)))
        assets.append(dict(asset, symbol=symbol))
        rows.append(dict(coverage, symbol=symbol))
    metadata.update(assets=assets, coverage=rows, candidate_count=4, samples=16)
    manifest.write_text(json.dumps(metadata))
    output = root / "supervised"
    dataset = CorpusDataset(
        manual_supervision(manifest, prepared, output), input_policy=HISTORICAL_MASKED
    )
    spec = FinancialInputSpec(
        source_sha256=dataset.identity,
        view_sha256="b" * 64,
        representation=dataset.manifest["representation"],
        dimensions=dict(prices=5, news=2, charts=1, fundamentals=3, macro=3),
        input_policy=HISTORICAL_MASKED,
    )
    codec = FrozenEpisodeCodec(spec)
    batches, encoded = {}, {}
    for index in DECISIONS:
        at = moment(index)
        batches[index] = []
        for asset in dataset.assets:
            flow = f"{asset['market']}/{asset['symbol']}"
            prices, _ = dataset._prices(asset)
            batch = validated_cpu_batch(
                dict(
                    inputs=dict(
                        prices=_price_contexts(prices, np.array([index]), 64),
                        news=np.zeros((1, 2), np.float32),
                        charts=np.array([[3.0]], np.float32),
                        fundamentals=np.zeros((1, 3), np.float32),
                        macro=np.zeros((1, 3), np.float32),
                    ),
                    presence=np.array([[True, False, True, False, False]]),
                    sample_ids=[f"{flow}/{at}"],
                    prediction_at=np.array([at], dtype="datetime64[us]"),
                    input_available_at=np.array([at], dtype="datetime64[us]"),
                ),
                spec,
            )
            batches[index].append(batch)
            encoded[(flow, at)] = codec.encode(batch)
    return dict(
        spec=spec,
        codec=codec,
        prefixes=PrefixTargetVerifier(dataset, source_manifest=manifest),
        batches=batches,
        encoded=encoded,
        label_origin=dataset.manifest["configuration"]["backend"],
    )


def paired_consumers(source):
    config = FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42)
    models, readers = {}, {}
    for mode in ("disabled", "diagnostic"):
        models[mode] = (
            FinancialPredictor(
                config,
                local_control=MACProjectionConfig(
                    mode=mode, rank=1, grid_size=4, frequency=1, max_flows=2, seed=91
                ),
                dtype=torch.float64,
                device="cpu",
            )
            .eval()
            .requires_grad_(False)
        )
        readers[mode] = (
            EpisodicReadout(
                EpisodicReadoutConfig(
                    source["codec"].fingerprint(),
                    hidden_size=32,
                    refinements=1,
                    neighbors=4,
                    seed=42,
                ),
                dtype=torch.float64,
                device="cpu",
            )
            .eval()
            .requires_grad_(False)
        )
    copied = copy_paired_parameters(models["disabled"], models["diagnostic"])
    copy_readout_parameters(readers["disabled"], readers["diagnostic"])
    assert copied["local_control"]["mac_sdpa_backend"] == "math"
    assert copied["runtime_state_transferred"] is False
    assert models["disabled"]._parameter_id == models["diagnostic"]._parameter_id
    return {
        mode: FrozenFinancialConsumer(model, readout=readers[mode])
        for mode, model in models.items()
    }


def tensor_leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu" and value.grad_fn is None and not value.requires_grad
        return {prefix: value.detach().clone()}
    result = {}
    items = (
        value.items()
        if isinstance(value, dict)
        else enumerate(value)
        if isinstance(value, (list, tuple))
        else ()
    )
    for name, child in items:
        result.update(tensor_leaves(child, f"{prefix}/{name}"))
    return result


def matured_record(native, source, outcome, identifier):
    prediction, label = outcome.prediction, outcome.label
    encoded = source["encoded"][(prediction.asset, prediction.decision_at)]
    record = native.MemoryRecord()
    record.id, record.decision_at = identifier, prediction.decision_at
    record.available_at, record.maturity_at = encoded.input_available_at[0], label.available_at
    record.key, record.value = encoded.key_inputs[0].tolist(), encoded.values[0].tolist()
    record.label, record.label_valid = label.value, True
    return record


def check_retention(native, run, source, commit, old, oracle, offered, policy):
    bundle = run._bundle(run.snapshot()["state"])
    bank, episodes = run._bank(bundle["bank"])
    assert bank.config.policy == policy
    records = bank.records()
    incoming = [
        matured_record(native, source, outcome, offered + index + 1)
        for index, outcome in enumerate(commit.applied)
    ]
    cutoff = bundle["cutoff"]
    for record in incoming:
        oracle.write(record, cutoff)
    offered += len(incoming)
    assert bank.seen == offered == run.snapshot()["applied"]
    assert len(records) == min(4, offered)
    for record in records:
        assert record.id in set(range(1, offered + 1))
        assert record.available_at <= record.decision_at < record.maturity_at <= cutoff
        episode = episodes[record.id]
        assert episode["error"] == record.label - episode["issued_prediction"]
    receipt = bank.receipt
    if incoming:
        assert receipt["policy"] == policy
        assert receipt["after_seen"] == offered
        assert receipt["retained_ids"] == sorted(record.id for record in records)
        assert receipt["client_ids"] == sorted([r.id for r in old] + [r.id for r in incoming])
        if bank.config.policy == "reservoir":
            assert receipt["retained_ids"] == sorted(r.id for r in oracle.retained_records())
            assert receipt["candidate_ids"] == []
        elif offered > 4:
            assert receipt["candidate_ids"] and receipt["variable_pairs"] > 0
            assert set(receipt["fixed_ids"]) <= set(receipt["retained_ids"])
            points = {r.id: r.key for r in old}
            points.update((r.id, native.normalize_key(r.key)) for r in incoming)

            def objective(ids):
                return math.fsum(
                    min(math.dist(point, points[i]) for i in ids) for point in points.values()
                )

            costs = [
                objective([*receipt["fixed_ids"], *candidate])
                for candidate in itertools.combinations(
                    receipt["candidate_ids"], 4 - len(receipt["fixed_ids"])
                )
            ]
            assert receipt["objective"] == pytest.approx(min(costs), abs=1e-12)
            assert objective(receipt["retained_ids"]) == pytest.approx(min(costs), abs=1e-12)
    return records, offered, receipt


def capture_control(run, mode, observations):
    bundle = run._bundle(run.snapshot()["state"])
    control = run._read(bundle["control"], "control")
    assert control["mac_updates"] == observations
    if not observations:
        assert control["reevaluations"] == 0
        return {}
    assert control["refinements"] == 1
    if mode == "disabled":
        assert control["selection"] is None and control["reevaluations"] == 0
        assert control["measurements"] == []
        return {}
    selection = control["selection"]
    assert tuple(flow for flow, _ in selection["flow_steps"]) == FLOWS
    assert tuple(selection["selected_flow_ids"]) == FLOWS[:2]
    assert control["reevaluations"] == 2
    assert control["group_estimated_bytes"] == selection["estimated_bytes"]
    values = {}
    for block in control["measurements"]:
        assert block["penalty"] is None
        for flow, value in zip(block["flow_ids"], block["angular_corrected_estimate"], strict=True):
            assert flow not in values and torch.isfinite(value)
            values[flow] = float(value)
    assert sorted(values) == list(FLOWS[:2])
    return values


def trajectory(
    native, source, consumer, output, policy, block_rows, *, reverse=False, recover=False
):
    mode = consumer.predictor.local_control.config.mode
    options = dict(
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(policy=policy, capacity=4, seed=73, frontier=1, new_candidates=2),
        phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
        world="four_flow_controls",
        fold="0",
        admission="m1",
        block_rows=block_rows,
    )
    scope = native.MemoryScope()
    scope.world, scope.partition, scope.fold, scope.representation = (
        "oracle",
        "validation",
        "0",
        "fixed64",
    )
    oracle = native.EpisodicMemory(scope, 73, 4, 2)
    run = FinancialSession(output, **options)
    previous, old, offered, observations = [], [], 0, 0
    points, controls, retained = {}, {}, []
    try:
        for index in (125, 126, 127, 128, 129, 130):
            at = moment(index)
            batches = [] if index == 129 else source["batches"][index]
            feedback = [
                native.Feedback(
                    p.id, 0, at, 0.125 * (int(p.asset[-1]) + 1) + 0.03125 * (index - 125)
                )
                for p in previous
            ]
            if reverse:
                batches, feedback = list(reversed(batches)), list(reversed(feedback))
            kind = "settlement" if index == 129 else "decision"
            step = dict(kind=kind, cutoff=at)
            if recover and index == 128:
                before = run.snapshot()
                latest = (output / "latest.json").read_bytes()

                def fail(boundary):
                    if boundary == native.Boundary.before_commit:
                        raise RuntimeError("Corte técnico antes de publicar")

                with pytest.raises(RuntimeError, match="Corte técnico"):
                    run.step(batches, feedback, fault=fail, **step)
                assert (output / "latest.json").read_bytes() == latest
                run.close()
                run = FinancialSession(output, **options, resume=True)
                assert run.snapshot() == before
            commit = run.step(batches, feedback, **step)
            previous = commit.predictions
            assert [p.asset for p in previous] == (list(FLOWS) if batches else [])
            observations += bool(batches)
            for prediction in commit.predictions:
                points[(prediction.asset, prediction.decision_at)] = prediction.value
            old, offered, receipt = check_retention(
                native, run, source, commit, old, oracle, offered, policy
            )
            retained.append([r.id for r in old])
            controls[index] = capture_control(run, mode, 4 if batches else 0)
            fast = run._fast_store.gather(run.fast_state(), FLOWS)
            assert fast.observed_steps.tolist() == [observations] * 4
            assert fast.mac.memory.steps.tolist() == [observations] * 4
        assert offered == 16 and run.diagnostics()["pending"] == 4
        result = dict(
            points=points,
            controls=controls,
            retained=retained,
            receipt=receipt,
            snapshot=run.snapshot(),
            fast=tensor_leaves(consumer.predictor.export_state_cpu(fast)),
        )
    finally:
        run.close()
    with FinancialSession(output, **options, resume=True) as restored:
        assert restored.snapshot() == result["snapshot"]
    consumer.verify()
    assert all(parameter.grad is None for parameter in consumer.predictor.parameters())
    return result


def assert_equivalent(left, right, *, exact):
    tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=1e-12, atol=1e-12)
    assert left["points"].keys() == right["points"].keys()
    keys = sorted(left["points"])
    np.testing.assert_allclose(
        [left["points"][k] for k in keys], [right["points"][k] for k in keys], **tolerance
    )
    assert left["retained"] == right["retained"]
    assert left["fast"].keys() == right["fast"].keys()
    for name, value in left["fast"].items():
        torch.testing.assert_close(value, right["fast"][name], **tolerance)


@pytest.mark.parametrize("policy", ["reservoir", "anchored"])
def test_frozen_control_and_retention_paths_overflow_recover_and_preserve_pairing(
    native, four_flow_source, tmp_path, policy
):
    rng = torch.random.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    consumers = paired_consumers(four_flow_source)
    results = {}
    for mode, consumer in consumers.items():
        root = tmp_path / mode
        reference = trajectory(native, four_flow_source, consumer, root / "reference", policy, 4)
        permuted = trajectory(
            native, four_flow_source, consumer, root / "permuted", policy, 1, reverse=True
        )
        recovered = trajectory(
            native, four_flow_source, consumer, root / "recovered", policy, 4, recover=True
        )
        assert_equivalent(reference, permuted, exact=False)
        assert_equivalent(reference, recovered, exact=True)
        assert reference["snapshot"] == recovered["snapshot"]
        for index in DECISIONS:
            for flow, estimate in reference["controls"][index].items():
                assert estimate == pytest.approx(permuted["controls"][index][flow], abs=1e-12)
        results[mode] = reference
    assert_equivalent(results["disabled"], results["diagnostic"], exact=True)
    assert results["disabled"]["receipt"] == results["diagnostic"]["receipt"]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.cuda.is_initialized() == cuda_initialized
    (tmp_path / "trace.json").write_text(
        json.dumps(
            {
                "policy": policy,
                "native_sha256": native.binary_sha256,
                "input_contract_id": four_flow_source["spec"].fingerprint(),
                "parameters_sha256": consumers["disabled"].predictor._parameter_id,
                "readout_parameters_sha256": consumers["disabled"].identity()[
                    "readout_parameters_sha256"
                ],
                "flows": 4,
                "issued_per_path": 20,
                "matured_per_path": 16,
                "capacity": 4,
                "paths_per_control": 3,
                "control_modes": list(results),
                "backend": "math",
                "fastpath": False,
                "K": 1,
                "admission": "m1",
                "optimizer_steps": 0,
                "label_origin": four_flow_source["label_origin"],
                "residual_estimation_calls": 0,
                "cuda_initialized_before": cuda_initialized,
                "cuda_state_unchanged": True,
                "retained_ids_by_event": results["disabled"]["retained"],
                "last_retention_receipt": results["disabled"]["receipt"],
                "C_prediction_and_fast_state_exact": True,
                "recovery_exact": True,
                "physical_partition_rtol": 1e-12,
                "physical_partition_atol": 1e-12,
            },
            indent=2,
        )
        + "\n"
    )
