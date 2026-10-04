"""Ejecución PPO nativa con pausas recuperables y selección temporal separada."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import read_tape, write_tape

ROOT = Path(__file__).resolve().parents[2]


def binary():
    result = Path(
        os.environ.get(
            "MARS_TITAN_PPO_EXECUTABLE", "build/native/native-ppo-release/mars-titan-ppo"
        )
    ).resolve()
    assert result.is_file(), "Compila mars-titan-ppo antes de ejecutar esta integración"
    return result


def read(path):
    return json.loads(path.read_text())


def scenario(tmp_path, partition, seed, name=None, parent="frozen-fixture"):
    prices = np.full((4, 2, 5), 10 + seed / 1000, dtype=np.float64)
    prices[:, :, 4] = 1000
    times = 1_672_531_200_000_000 + np.arange(4, dtype=np.int64) * 86_400_000_000
    tape = MarketTape(
        prices,
        times,
        ["FIC0", "FIC1"],
        np.full((4, 2), 0.02),
        domain="synthetic",
        currency="USD",
        partition=partition,
        parent_id=parent,
        source_identity={"generator": {"seed": seed}},
    )
    destination = tmp_path / (name or partition)
    write_tape(tape, destination)
    return destination


@pytest.fixture
def inputs(tmp_path):
    config = tmp_path / "config.json"
    config.write_text((ROOT / "configs/simulation/native-ppo-diagnostic.json").read_text())
    return config, scenario(tmp_path, "train", 7), scenario(tmp_path, "validation", 8)


def execute(
    inputs,
    output,
    *extra,
    diagnostic=True,
    success=True,
    paused=False,
    pass_fds=(),
    environment=None,
):
    config, train, validation = inputs
    command = [
        str(binary()),
        "--config",
        str(config),
        "--output",
        str(output),
        "--train-tape",
        str(train),
        "--validation-tape",
        str(validation),
    ]
    if diagnostic:
        command.extend(("--device", "cpu", "--diagnostic"))
    result = subprocess.run(
        [*command, *extra],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", **(environment or {})},
        pass_fds=pass_fds,
        capture_output=True,
        text=True,
        timeout=60,
    )
    expected_code = 2 if success and paused else 0 if success else 1
    assert result.returncode == expected_code, result.stderr
    if not success:
        assert result.stderr.startswith("Error: "), result.stderr
    for diagnostic_name in ("AddressSanitizer", "runtime error:", "UndefinedBehaviorSanitizer"):
        assert diagnostic_name not in result.stderr, result.stderr
    return result


def latest(output):
    index = read(output / "ppo-index.json")["payload"]
    return output / index["recent"][0]["bundle"]


def test_pause_resume_preserves_partial_rollout_and_final_parameters(inputs, tmp_path):
    started = datetime.now(UTC).replace(microsecond=0)
    uninterrupted, resumed = tmp_path / "continuous", tmp_path / "resumed"
    execute(inputs, uninterrupted)
    execute(inputs, resumed, "--stop-after", "8", paused=True)
    paused = read(resumed / "run.json")
    assert paused["status"] == "paused"
    assert paused["transitions"] == 8
    assert paused["partial_ticks"] == 4
    assert paused["optimizer_steps"] == 0
    execute(inputs, resumed, "--resume")
    expected, actual = read(uninterrupted / "run.json"), read(resumed / "run.json")
    for field in (
        "status",
        "transitions",
        "optimizer_steps",
        "invalid_transitions",
        "evaluations",
        "best",
    ):
        assert actual[field] == expected[field]
    assert actual["status"] == "completed"
    assert actual["transitions"] == 32
    assert actual["optimizer_steps"] == 8
    assert actual["final_test_opened"] is False
    assert actual["activity"] == "rl"
    assert actual["model"] == "ppo"
    assert actual["backend"] == "native_libtorch"
    assert actual["seed"] == 42
    assert actual["partition"] == "train"
    assert actual["global_step"] == actual["transitions"] == actual["total_steps"]
    assert actual["selection"]["policy"] == "greedy_argmax"
    assert started <= datetime.fromisoformat(actual["updated_at"]) <= datetime.now(UTC)
    import torch

    first = torch.jit.load(str(latest(uninterrupted) / "policy.pt"), map_location="cpu")
    second = torch.jit.load(str(latest(resumed) / "policy.pt"), map_location="cpu")
    for name, value in first.network.state_dict().items():
        assert torch.equal(value, second.network.state_dict()[name])
    previous = (resumed / "ppo-index.json").read_bytes()
    execute(inputs, resumed, "--resume")
    assert (resumed / "ppo-index.json").read_bytes() == previous
    assert len(list(resumed.glob("ppo-*/manifest.json"))) <= 3


def test_initial_policy_is_evaluated_and_selectable_before_training(inputs, tmp_path):
    output = tmp_path / "initial"
    execute(inputs, output, "--stop-after", "0", paused=True)
    report = read(output / "run.json")
    assert report["evaluations"] == 1
    assert report["best"]["transitions"] == 0
    assert report["best"]["optimizer_steps"] == 0
    assert read(output / "ppo-index.json")["payload"]["best"] is not None


@pytest.mark.parametrize("change", ["seed", "configuration", "source"])
def test_resume_rejects_changed_identity(inputs, tmp_path, change):
    output = tmp_path / "run"
    execute(inputs, output, "--stop-after", "8", paused=True)
    config, train, _ = inputs
    if change == "source":
        manifest = read(train / "manifest.json")
        manifest["identity"]["source"]["generator"]["seed"] = 11
        (train / "manifest.json").write_text(json.dumps(manifest))
    else:
        document = read(config)
        if change == "seed":
            document["training"]["seed"] = 43
        else:
            document["hyperparameters"]["learning_rate"] *= 2
        config.write_text(json.dumps(document))
    execute(inputs, output, "--resume", success=False)


@pytest.mark.parametrize(
    "problem", ["partition", "parent", "generator_seed", "seed", "budget", "environments"]
)
def test_invalid_experiments_fail_before_creating_output(inputs, tmp_path, problem):
    config, _, validation = inputs
    if problem in {"partition", "parent", "generator_seed"}:
        manifest = read(validation / "manifest.json")
        if problem == "partition":
            manifest["identity"]["partition"] = "train"
        elif problem == "parent":
            manifest["identity"]["parent_id"] = "other-parent"
        else:
            manifest["identity"]["source"]["generator"]["seed"] = 7
        (validation / "manifest.json").write_text(json.dumps(manifest))
    else:
        document = read(config)
        if problem == "seed":
            document["training"]["seed"] = 45
        elif problem == "budget":
            document["training"]["total_transitions"] = 64
        else:
            document["environments"] = 3
        config.write_text(json.dumps(document))
    output = tmp_path / "invalid"
    execute(inputs, output, success=False)
    assert not output.exists()


def test_cuda_requires_the_actual_inherited_lock_before_allocation(inputs, tmp_path):
    output = tmp_path / "cuda"
    with (tmp_path / "unrelated.lock").open("wb") as stream:
        result = execute(
            inputs,
            output,
            "--device",
            "cuda:0",
            "--gpu-lease-fd",
            str(stream.fileno()),
            "--vram-budget-bytes",
            str(256 * 1024**2),
            "--vram-total-bytes",
            str(8 * 1024**3),
            diagnostic=False,
            success=False,
            pass_fds=(stream.fileno(),),
            environment={"XDG_RUNTIME_DIR": str(tmp_path)},
        )
    assert "bloqueo" in result.stderr.lower()
    assert not output.exists()


def test_early_stopping_is_explicit_and_keeps_the_initial_candidate(inputs, tmp_path):
    config = inputs[0]
    document = read(config)
    document["selection"].update(early_stopping=True, patience=1, min_delta=1000)
    config.write_text(json.dumps(document))
    output = tmp_path / "early"
    execute(inputs, output)
    result = read(output / "run.json")
    assert result["status"] == "completed"
    assert result["stopping_reason"] == "early_stop"
    assert result["transitions"] == 16
    assert result["best"]["transitions"] == 0


def context(source, name):
    closes = pq.read_table(source / "market.parquet", columns=["close_time"])["close_time"]
    available = closes.to_numpy()[::2]
    table = pa.table(
        {
            "session": pa.array(range(4), type=pa.int32()),
            "feature": pa.array([0] * 4, type=pa.int32()),
            "value": pa.array([0.1] * 4, type=pa.float32()),
            "present": pa.array([True] * 4, type=pa.bool_()),
            "available_at": pa.array(available, type=pa.int64()),
        }
    )
    data = source / "context.parquet"
    pq.write_table(table, data)
    manifest = dict(
        schema_version=1,
        domain="synthetic",
        market_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        fields=[dict(name=name, unit="1")],
        file=dict(
            path="context.parquet",
            sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            bytes=data.stat().st_size,
        ),
    )
    (source / "context.json").write_text(json.dumps(manifest))


def test_context_columns_must_match_even_when_width_is_identical(inputs, tmp_path):
    context(inputs[1], "factor")
    context(inputs[2], "factor")
    execute(inputs, tmp_path / "compatible", "--stop-after", "0", paused=True)
    path = inputs[2] / "context.json"
    manifest = read(path)
    manifest["fields"][0]["name"] = "unrelated"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "mismatch"
    result = execute(inputs, output, success=False)
    assert "contexto" in result.stderr
    assert not output.exists()


def test_context_must_be_present_in_both_partitions(inputs, tmp_path):
    context(inputs[1], "factor")
    output = tmp_path / "missing-context"
    execute(inputs, output, success=False)
    assert not output.exists()


def test_replica_memory_is_rejected_from_metadata_before_loading_payloads(inputs, tmp_path):
    config, train, _ = inputs
    document = read(config)
    document["training"].update(total_transitions=8192, rollout_transitions=4096)
    document["environments"] = 4096
    config.write_text(json.dumps(document))
    manifest_path = train / "manifest.json"
    manifest = read(manifest_path)
    manifest["sessions"] = 8192
    manifest_path.write_text(json.dumps(manifest))
    declared = dict(
        schema_version=1,
        domain="synthetic",
        market_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        fields=[dict(name=f"context_{index}", unit="1") for index in range(512)],
        file=dict(path="context.parquet", sha256="a" * 64, bytes=100),
    )
    (train / "context.json").write_text(json.dumps(declared))
    lock = tmp_path / "mars-titan-scientific-gpu.lock"
    with lock.open("w+b") as stream:
        lock.chmod(0o600)
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = execute(
            inputs,
            tmp_path / "too-large",
            "--device",
            "cuda:0",
            "--gpu-lease-fd",
            str(stream.fileno()),
            "--vram-budget-bytes",
            str(256 * 1024**2),
            "--vram-total-bytes",
            str(8 * 1024**3),
            diagnostic=False,
            success=False,
            pass_fds=(stream.fileno(),),
            environment={"XDG_RUNTIME_DIR": str(tmp_path)},
        )
    assert "memoria" in result.stderr
    assert not (tmp_path / "too-large").exists()


def test_sealed_test_is_rejected_before_reading_context(inputs, tmp_path):
    source = inputs[2]
    manifest = read(source / "manifest.json")
    manifest["identity"]["partition"] = "test"
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "context.json").mkdir()
    result = execute(inputs, tmp_path / "sealed", success=False)
    assert "test" in result.stderr or "partición" in result.stderr
    assert not (tmp_path / "sealed").exists()


def test_completed_resume_verifies_the_selected_checkpoint(inputs, tmp_path):
    output = tmp_path / "run"
    execute(inputs, output)
    selected = read(output / "ppo-index.json")["payload"]["best"]["bundle"]
    (output / selected / "policy.pt").write_bytes(b"incomplete")
    previous = (output / "ppo-index.json").read_bytes()
    execute(inputs, output, "--resume", success=False)
    assert (output / "ppo-index.json").read_bytes() == previous


def pending_validation_checkpoint(output):
    index_path = output / "ppo-index.json"
    envelope = read(index_path)
    record = envelope["payload"]["recent"][0]
    directory = output / record["bundle"]
    metadata = read(directory / "metadata.json")
    assert metadata["transitions"] == 16
    assert metadata["progress"]["best"]["transitions"] == 0
    # Representa un corte después de Adam y antes de confirmar su validación.
    metadata["progress"].update(
        status="paused", evaluations=1, stale_evaluations=0, evaluated_optimizer_steps=0
    )
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    (directory / "metadata.json").write_bytes(encoded)
    manifest = read(directory / "manifest.json")
    manifest["files"]["metadata.json"] = dict(
        bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest()
    )
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (directory / "manifest.json").write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    replacement = dict(bundle=f"ppo-{digest}", sha256=digest)
    directory.rename(output / replacement["bundle"])
    for group in ("recent", "retired"):
        envelope["payload"][group] = [
            replacement if item == record else item for item in envelope["payload"][group]
        ]
    if envelope["payload"]["best"] == record:
        envelope["payload"]["best"] = replacement
    encoded = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    envelope["sha256"] = hashlib.sha256(encoded).hexdigest()
    index_path.write_text(json.dumps(envelope))


def test_pending_validation_can_stop_before_any_additional_transition(inputs, tmp_path):
    config = inputs[0]
    document = read(config)
    document["selection"].update(early_stopping=True, patience=1, min_delta=1000)
    config.write_text(json.dumps(document))
    output = tmp_path / "pending-evaluation"
    execute(inputs, output)
    pending_validation_checkpoint(output)
    execute(inputs, output, "--resume")
    result = read(output / "run.json")
    assert result["status"] == "completed"
    assert result["stopping_reason"] == "early_stop"
    assert result["transitions"] == 16
    assert result["partial_ticks"] == 0
    assert result["optimizer_steps"] == 4


def test_resume_recovers_a_cut_before_the_initial_checkpoint(inputs, tmp_path):
    template = tmp_path / "identity-template"
    execute(inputs, template, "--stop-after", "0", paused=True)
    previous = (template / "ppo-index.json").read_bytes()
    output = tmp_path / "initial-cut"
    output.mkdir()
    (output / "identity.json").write_bytes((template / "identity.json").read_bytes())
    envelope = read(template / "ppo-index.json")
    envelope["payload"].update(recent=[], best=None, retired=[])
    payload = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    envelope["sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "ppo-index.json").write_text(json.dumps(envelope))
    execute(inputs, output, "--resume")
    result = read(output / "run.json")
    assert result["status"] == "completed"
    assert result["transitions"] == 32
    assert result["optimizer_steps"] == 8
    assert result["best"] is not None
    assert (template / "ppo-index.json").read_bytes() == previous


@pytest.fixture
def adaptive_inputs(tmp_path, request):
    from mars_titan.simulation.adaptation_scenarios import prepare_adaptation_scenarios

    settings = dict(
        schema_version=1,
        families=["known_signal"],
        generator=dict(assets=2, sessions=12, warmup_sessions=4, regime_sessions=4),
        worlds=dict(train=17, validation=getattr(request, "param", 2), audit=1),
        seed_base=91,
        final_test_opened=False,
    )
    catalog = tmp_path / "catalog"
    index = prepare_adaptation_scenarios(settings, catalog)
    config = read(ROOT / "configs/simulation/native-ppo-diagnostic.json")
    config.update(
        schema_version=2,
        environments=16,
        evaluation_transitions=32,
        agent=dict(variant="ppo", trading_field=10, markov_fields=[], hmm_file=None),
    )
    config["training"].update(rollout_transitions=32)
    config["hyperparameters"]["minibatch_size"] = 16
    config_path = tmp_path / "adaptive.json"
    config_path.write_text(json.dumps(config))
    paths = {
        split: [catalog / row["path"] for row in index["records"] if row["split"] == split]
        for split in ("train", "validation", "audit")
    }
    return config_path, paths


def execute_adaptive(inputs, output, *extra, success=True, paused=False):
    config, paths = inputs
    primary = (config, paths["train"][0], paths["validation"][0])
    sources = [
        value
        for split, flag in (("train", "--train-tape"), ("validation", "--validation-tape"))
        for source in paths[split][1:]
        for value in (flag, str(source))
    ]
    return execute(primary, output, *sources, *extra, success=success, paused=paused)


def trace_rows(output):
    trace = output / "trace"
    records = read(trace / "trace-index.json")["payload"]["shards"]
    return [row for record in records for row in pq.read_table(trace / record["path"]).to_pylist()]


def unconfirmed_trace_tail(output):
    trace = output / "trace"
    envelope = read(trace / "trace-index.json")
    previous = envelope["payload"]["shards"][-1]
    table = pq.read_table(trace / previous["path"])
    rows = table.to_pylist()
    rows.append(dict(rows[-1], decision_id=previous["last"] + 1))
    temporary = trace / "unconfirmed.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    path = trace / f"trace-{digest}.parquet"
    temporary.rename(path)
    envelope["payload"]["shards"][-1] = dict(
        path=path.name,
        sha256=digest,
        bytes=path.stat().st_size,
        first=previous["first"],
        last=previous["last"] + 1,
        rows=len(rows),
    )
    envelope["payload"]["cursor"] += 1
    encoded = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    envelope["sha256"] = hashlib.sha256(encoded).hexdigest()
    (trace / "trace-index.json").write_text(json.dumps(envelope))
    return path


def test_adaptive_catalog_warmup_and_recovery_preserve_actual_source_indices(
    adaptive_inputs, tmp_path
):
    continuous, partial = tmp_path / "continuous", tmp_path / "partial"
    execute_adaptive(adaptive_inputs, continuous)
    execute_adaptive(adaptive_inputs, partial, "--stop-after", "16", paused=True)
    state = read(latest(partial) / "metadata.json")
    assert state["schema_version"] == 2
    assert state["source_indices"] == list(range(16))
    assert state["next_source"] == 16
    assert state["observed_transitions"] > state["transitions"] == 16
    unconfirmed = unconfirmed_trace_tail(partial)
    from scripts.explain_rl_decisions import explain

    assert f"Decisión {state['observed_transitions'] + 1}," not in explain(
        partial / "run.json", limit=200
    )
    execute_adaptive(adaptive_inputs, partial, "--resume")
    first, second = read(continuous / "run.json"), read(partial / "run.json")
    for key in ("transitions", "observed_transitions", "optimizer_steps", "evaluations", "best"):
        assert first[key] == second[key]
    assert second["agent_variant"] == "ppo" and second["analysis_domain"] == "technical"
    assert second["trace"]["confirmed_cursor"] == second["observed_transitions"]
    assert trace_rows(continuous) == trace_rows(partial)
    assert not unconfirmed.exists()
    assert 0 <= second["best"]["mean_max_drawdown"] <= 1
    metrics = second["best"]["validation_metrics"]
    assert [row["manifest_sha256"] for row in metrics] == [
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        for path in adaptive_inputs[1]["validation"]
    ]
    assert all(row["completed"] and row["steps"] == 11 for row in metrics)
    assert second["best"]["mean_max_drawdown"] == pytest.approx(
        np.mean([row["max_drawdown"] for row in metrics])
    )
    assert set(second["timings"]) == {
        "training_seconds",
        "evaluation_seconds",
        "checkpoint_seconds",
        "setup_seconds",
    }
    assert all(value > 0 for value in second["timings"].values())
    assert sum(second["timings"].values()) <= second["invocation_seconds"]
    import torch

    before = torch.jit.load(str(latest(continuous) / "policy.pt"), map_location="cpu")
    after = torch.jit.load(str(latest(partial) / "policy.pt"), map_location="cpu")
    for name, value in before.network.state_dict().items():
        assert torch.equal(value, after.network.state_dict()[name])


@pytest.mark.parametrize("variant", ["ppo_window", "ppo_gru", "ppo_episodic"])
def test_adaptive_initial_policies_share_explicit_selection_contract(
    adaptive_inputs, tmp_path, variant
):
    config, _ = adaptive_inputs
    document = read(config)
    document["agent"]["variant"] = variant
    config.write_text(json.dumps(document))
    output = tmp_path / variant
    execute_adaptive(adaptive_inputs, output, "--stop-after", "0", paused=True)
    report = read(output / "run.json")
    assert report["agent_variant"] == variant
    assert report["best"]["transitions"] == 0 and report["evaluations"] == 1


def test_adaptive_audit_sources_are_rejected_before_context_loading(adaptive_inputs, tmp_path):
    config, paths = adaptive_inputs
    audit = paths["audit"][0]
    (audit / "context.parquet").unlink()
    paths["validation"] = [audit]
    result = execute_adaptive(adaptive_inputs, tmp_path / "audit-leak", success=False)
    assert "auditoría" in result.stderr or "partición" in result.stderr
    assert not (tmp_path / "audit-leak").exists()


def test_adaptive_receipt_requires_the_declared_macro_contract(adaptive_inputs, tmp_path):
    _, paths = adaptive_inputs
    source = paths["train"][0]
    path = source / "manifest.json"
    manifest = read(path)
    manifest["identity"]["source"]["simulated_macro"] = []
    path.write_text(json.dumps(manifest))
    context_path = source / "context.json"
    context_manifest = read(context_path)
    context_manifest["market_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    context_path.write_text(json.dumps(context_manifest))
    output = tmp_path / "false-macro-coverage"
    result = execute_adaptive(adaptive_inputs, output, success=False)
    assert "macro" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("change", ["variant", "environments", "trading_field", "unexpected"])
def test_adaptive_invalid_contract_fails_before_creating_output(adaptive_inputs, tmp_path, change):
    config, _ = adaptive_inputs
    document = read(config)
    if change == "environments":
        document[change] = 8
    elif change == "variant":
        document["agent"][change] = "unknown"
    elif change == "trading_field":
        document["agent"][change] = 0
    else:
        document["agent"][change] = True
    config.write_text(json.dumps(document))
    output = tmp_path / "invalid-adaptive"
    execute_adaptive(adaptive_inputs, output, success=False)
    assert not output.exists()


def test_adaptive_validation_interval_does_not_evaluate_every_update(adaptive_inputs, tmp_path):
    config, _ = adaptive_inputs
    document = read(config)
    document["training"]["rollout_transitions"] = 16
    config.write_text(json.dumps(document))
    output = tmp_path / "sparse-validation"
    execute_adaptive(adaptive_inputs, output)
    report = read(output / "run.json")
    assert report["evaluations"] == 2
    assert report["catalog_train_sources"] == 17


def adaptive_hmm(inputs):
    pytest.importorskip("hmmlearn")
    from mars_titan.simulation.adaptation_scenarios import fit_hmm

    config, paths = inputs
    samples = []
    for path in paths["train"]:
        values = pq.read_table(path / "context.parquet")["value"].to_numpy().reshape(12, 11)
        samples.append(("train", values[:, [4, 5, 6]]))
    model = fit_hmm(samples)
    model["train_manifest_sha256"] = [
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() for path in paths["train"]
    ]
    path = config.with_name("hmm.json")
    path.write_text(json.dumps(model))
    document = read(config)
    document["agent"].update(
        variant="ppo_hmm",
        markov_fields=[4, 5, 6],
        hmm_file=dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    config.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize(
    "variant", ["ppo_hmm", "ppo_episodic_hmm", "ppo_recent_aux", "ppo_replay_aux"]
)
def test_adaptive_hmm_variants_use_verified_training_fit(adaptive_inputs, tmp_path, variant):
    adaptive_hmm(adaptive_inputs)
    config, _ = adaptive_inputs
    document = read(config)
    document["agent"]["variant"] = variant
    config.write_text(json.dumps(document))
    output = tmp_path / variant
    auxiliary = variant in {"ppo_recent_aux", "ppo_replay_aux"}
    if auxiliary:
        execute_adaptive(adaptive_inputs, output)
    else:
        execute_adaptive(adaptive_inputs, output, "--stop-after", "0", paused=True)
    report = read(output / "run.json")
    assert report["agent_variant"] == variant
    if auxiliary:
        assert report["auxiliary_steps"] > 0
        assert report["auxiliary_samples"] >= report["auxiliary_steps"]
        assert report["transitions"] == 32 and report["evaluations"] == 2
    else:
        assert report["evaluations"] == 1


@pytest.mark.parametrize("problem", ["hash", "fitted_sources", "future_fields"])
def test_adaptive_hmm_rejects_changed_or_reserved_evidence(adaptive_inputs, tmp_path, problem):
    path = adaptive_hmm(adaptive_inputs)
    config, _ = adaptive_inputs
    document = read(config)
    if problem == "hash":
        document["agent"]["hmm_file"]["sha256"] = "0" * 64
    else:
        model = read(path)
        if problem == "fitted_sources":
            model["train_manifest_sha256"][0] = "0" * 64
        else:
            model["inference"] = "full_sequence_smoothing"
        path.write_text(json.dumps(model))
        document["agent"]["hmm_file"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config.write_text(json.dumps(document))
    output = tmp_path / "invalid-hmm"
    execute_adaptive(adaptive_inputs, output, success=False)
    assert not output.exists()


def execute_audit(inputs, source_run, output, *extra, success=True):
    config, paths = inputs
    result = subprocess.run(
        [
            str(binary()),
            "--config",
            str(config),
            "--audit-run",
            str(source_run),
            "--output",
            str(output),
            "--audit-tape",
            str(paths["audit"][0]),
            "--device",
            "cpu",
            "--diagnostic",
            *extra,
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (0 if success else 1), result.stdout + result.stderr
    return result


def test_frozen_audit_compares_costs_without_changing_selection(adaptive_inputs, tmp_path):
    selected, output = tmp_path / "selected", tmp_path / "audit"
    execute_adaptive(adaptive_inputs, selected)
    previous_index = (selected / "ppo-index.json").read_bytes()
    previous_report = (selected / "run.json").read_bytes()
    execute_audit(adaptive_inputs, selected, output)
    report = read(output / "audit.json")
    assert report["schema_version"] == 2
    assert report["status"] == "completed"
    assert report["confirmed_episodes"] == report["total_episodes"] == 3
    assert [row["cost_bps"] for row in report["metrics"]] == [0, 10, 25]
    assert all(row["completed"] and row["steps"] == 11 for row in report["metrics"])
    assert len({row["manifest_sha256"] for row in report["metrics"]}) == 1
    assert all(
        row["mode"] in ("greedy", "warmup") and not row["learning_allowed"]
        for row in trace_rows(output)
    )
    assert report["trace"]["confirmed_cursor"] == len(trace_rows(output)) == 33
    previous_rows = trace_rows(output)
    journal_path = output / "audit-state.json"
    journal = read(journal_path)
    journal["payload"]["metrics"].pop()
    journal["payload"].update(status="paused", confirmed_cursor=22)
    encoded = json.dumps(journal["payload"], sort_keys=True, separators=(",", ":")).encode()
    journal["sha256"] = hashlib.sha256(encoded).hexdigest()
    journal_path.write_text(json.dumps(journal))
    execute_audit(adaptive_inputs, selected, output, "--resume")
    assert read(output / "audit.json")["metrics"] == report["metrics"]
    assert trace_rows(output) == previous_rows
    assert "metrics" not in read(output / "run.json")
    assert (selected / "ppo-index.json").read_bytes() == previous_index
    assert (selected / "run.json").read_bytes() == previous_report


def test_audit_rejects_unfinished_selection_before_reading_reserved_sources(
    adaptive_inputs, tmp_path
):
    selected, output = tmp_path / "paused", tmp_path / "audit"
    execute_adaptive(adaptive_inputs, selected, "--stop-after", "0", paused=True)
    (adaptive_inputs[1]["audit"][0] / "context.parquet").unlink()
    result = execute_audit(adaptive_inputs, selected, output, success=False)
    assert "cerrada" in result.stderr or "finaliz" in result.stderr
    assert not output.exists()


def test_parent_guard_accepts_the_actual_supervisor_and_rejects_another_pid(inputs, tmp_path):
    invalid = tmp_path / "foreign-parent"
    result = execute(inputs, invalid, "--parent-pid", "1", success=False)
    assert "padre" in result.stderr and not invalid.exists()
    valid = tmp_path / "actual-parent"
    execute(inputs, valid, "--parent-pid", str(os.getpid()), "--stop-after", "0", paused=True)
    assert read(valid / "run.json")["status"] == "paused"


@pytest.mark.parametrize("adaptive_inputs", [9], indirect=True)
def test_nine_complete_drawdowns_keep_the_selected_checkpoint_recoverable(
    adaptive_inputs, tmp_path
):
    import torch

    config, paths = adaptive_inputs
    document = read(config)
    document["environment"]["cost_bps"] = 0
    config.write_text(json.dumps(document))
    for index, source in enumerate(paths["validation"]):
        original = read_tape(source)
        prices = np.full_like(original.prices, 100.0)
        prices[:, :, 4] = 100000 + index
        prices[-1, :, 2:4] = 1e-200
        collapsed = MarketTape(
            prices,
            original.close_times,
            original.assets,
            original.scores,
            domain="synthetic",
            currency=original.currency,
            partition="validation",
            parent_id=original.identity["parent_id"],
            open_times=original.open_times,
            source_identity=original.identity["source"],
        )
        destination = tmp_path / f"collapse-{index}"
        write_tape(collapsed, destination)
        table = pq.read_table(source / "context.parquet")
        values = table["value"].to_numpy().copy().reshape(len(original), -1)
        values[:, 4:7] = 0
        values[-1, 4] = -1
        values[-1, 5] = np.std(values[:, 4])
        values[:, 7:10] = [3, 4, 1]
        table = table.set_column(
            table.column_names.index("value"),
            "value",
            pa.array(values.reshape(-1), type=pa.float32()),
        )
        pq.write_table(table, destination / "context.parquet")
        context = read(source / "context.json")
        context["market_manifest_sha256"] = hashlib.sha256(
            (destination / "manifest.json").read_bytes()
        ).hexdigest()
        context["file"].update(
            bytes=(destination / "context.parquet").stat().st_size,
            sha256=hashlib.sha256((destination / "context.parquet").read_bytes()).hexdigest(),
        )
        (destination / "context.json").write_text(json.dumps(context))
        paths["validation"][index] = destination

    template, output = tmp_path / "template", tmp_path / "controlled"
    execute_adaptive(adaptive_inputs, template, "--stop-after", "0", paused=True)
    source = latest(template)
    pending = output / "pending"
    pending.mkdir(parents=True)
    shutil.copyfile(template / "identity.json", output / "identity.json")
    shutil.copyfile(source / "rollout.pt", pending / "rollout.pt")
    policy = torch.jit.load(str(source / "policy.pt"), map_location="cpu")
    with torch.no_grad():
        for parameter in policy.network.state_dict().values():
            parameter.zero_()
        policy.network.output_bias[5] = 1
    torch.jit.save(policy, str(pending / "policy.pt"))
    metadata = read(source / "metadata.json")
    # Estado inicial controlado antes de su primera validación, con compra completa fija.
    metadata["progress"].update(
        best=None,
        evaluations=0,
        stale_evaluations=0,
        evaluated_optimizer_steps=None,
        evaluated_transitions=None,
    )
    (pending / "metadata.json").write_text(json.dumps(metadata))
    manifest = read(source / "manifest.json")
    for name in ("metadata.json", "policy.pt", "rollout.pt"):
        data = (pending / name).read_bytes()
        manifest["files"][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (pending / "manifest.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    selected = dict(bundle=f"ppo-{digest}", sha256=digest)
    pending.rename(output / selected["bundle"])
    envelope = read(template / "ppo-index.json")
    envelope["payload"].update(recent=[selected], best=None, retired=[])
    data = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    envelope["sha256"] = hashlib.sha256(data).hexdigest()
    (output / "ppo-index.json").write_text(json.dumps(envelope))
    execute_adaptive(adaptive_inputs, output, "--resume", "--stop-after", "0", paused=True)
    report = read(output / "run.json")
    assert len(report["best"]["validation_metrics"]) == 9
    assert all(row["max_drawdown"] == 1 for row in report["best"]["validation_metrics"])
    execute_adaptive(adaptive_inputs, output, "--resume", "--stop-after", "0", paused=True)
    assert read(output / "run.json")["best"]["mean_max_drawdown"] == 1


def convergence_config(adaptive_inputs, *, minimum=16, patience=1, variant="ppo"):
    config, _ = adaptive_inputs
    document = read(config)
    document.update(schema_version=3, evaluation_transitions=16)
    document["training"].update(total_transitions=32, rollout_transitions=16)
    document["selection"].update(
        early_stopping=True, min_transitions=minimum, patience=patience, min_delta=1000
    )
    document["agent"]["variant"] = variant
    config.write_text(json.dumps(document))
    return document


@pytest.mark.parametrize("variant", ["ppo"])
def test_convergence_minimum_inclusive_and_budget_preserve_initial_candidate(
    adaptive_inputs, tmp_path, variant
):
    convergence_config(adaptive_inputs, variant=variant)
    output = tmp_path / "minimum"
    execute_adaptive(adaptive_inputs, output, "--stop-after", "16", paused=True)
    paused = read(output / "run.json")
    assert paused["transitions"] == 16 and paused["stale_evaluations"] == 0
    assert paused["evaluations"] == 2 and paused["best"]["transitions"] == 0
    execute_adaptive(adaptive_inputs, output, "--resume")
    report = read(output / "run.json")
    assert report["schema_version"] == 3
    assert report["transitions"] == 32 and report["stale_evaluations"] == 1
    assert report["stopping_reason"] == "budget_exhausted"
    assert report["best"]["transitions"] == 0
    assert read(latest(output) / "metadata.json")["progress"]["status"] == "completed"


@pytest.mark.parametrize("variant", ["ppo"])
def test_convergence_plateau_before_budget_recovers_without_extra_updates(
    adaptive_inputs, tmp_path, variant
):
    convergence_config(adaptive_inputs, minimum=0, variant=variant)
    output = tmp_path / "plateau"
    execute_adaptive(adaptive_inputs, output, "--stop-after", "0", paused=True)
    execute_adaptive(adaptive_inputs, output, "--resume")
    report = read(output / "run.json")
    assert report["transitions"] == 16 and report["stale_evaluations"] == 1
    assert report["stopping_reason"] == "early_stop"
    assert report["best"]["transitions"] == 0
    before = (output / "ppo-index.json").read_bytes()
    execute_adaptive(adaptive_inputs, output, "--resume")
    assert (output / "ppo-index.json").read_bytes() == before


@pytest.mark.parametrize("minimum", [-1, 1, 48, True])
def test_convergence_rejects_invalid_minimum(adaptive_inputs, tmp_path, minimum):
    convergence_config(adaptive_inputs, minimum=minimum)
    execute_adaptive(adaptive_inputs, tmp_path / "invalid", success=False)


def test_convergence_dqn_rejects_minimum_before_learning_warmup(adaptive_inputs, tmp_path):
    convergence_config(adaptive_inputs, minimum=0, variant="double_dqn")
    result = execute_adaptive(adaptive_inputs, tmp_path / "untrained", success=False)
    assert "calentamiento" in result.stderr


def replace_confirmed_progress(output, **changes):
    envelope = read(output / "ppo-index.json")
    record = envelope["payload"]["recent"][0]
    source = output / record["bundle"]
    pending = output / "tampered"
    shutil.copytree(source, pending)
    metadata = read(pending / "metadata.json")
    metadata["progress"].update(changes)
    (pending / "metadata.json").write_text(json.dumps(metadata))
    manifest = read(pending / "manifest.json")
    data = (pending / "metadata.json").read_bytes()
    manifest["files"]["metadata.json"] = dict(
        bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
    )
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (pending / "manifest.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    record.update(bundle=f"ppo-{digest}", sha256=digest)
    pending.rename(output / record["bundle"])
    data = json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":")).encode()
    envelope["sha256"] = hashlib.sha256(data).hexdigest()
    (output / "ppo-index.json").write_text(json.dumps(envelope))


@pytest.mark.parametrize("audit", [False, True])
def test_convergence_rejects_patience_counted_at_minimum(adaptive_inputs, tmp_path, audit):
    convergence_config(adaptive_inputs)
    output = tmp_path / "tampered-state"
    execute_adaptive(adaptive_inputs, output, "--stop-after", "16", paused=True)
    replace_confirmed_progress(output, status="early_stopped", stale_evaluations=1)
    if audit:
        result = execute_audit(adaptive_inputs, output, tmp_path / "audit", success=False)
    else:
        result = execute_adaptive(adaptive_inputs, output, "--resume", success=False)
    assert "paciencia" in result.stderr


def test_convergence_audit_accepts_confirmed_plateau(adaptive_inputs, tmp_path):
    convergence_config(adaptive_inputs, minimum=0)
    output = tmp_path / "selected-convergence"
    execute_adaptive(adaptive_inputs, output)
    execute_audit(adaptive_inputs, output, tmp_path / "audit-convergence")


def test_convergence_resume_preserves_consumed_patience(adaptive_inputs, tmp_path):
    convergence_config(adaptive_inputs, minimum=0, patience=2)
    output = tmp_path / "consumed-patience"
    execute_adaptive(adaptive_inputs, output, "--stop-after", "16", paused=True)
    paused = read(output / "run.json")
    assert paused["stale_evaluations"] == 1 and paused["transitions"] == 16
    execute_adaptive(adaptive_inputs, output, "--resume")
    result = read(output / "run.json")
    assert result["stale_evaluations"] == 2 and result["transitions"] == 32
    assert result["stopping_reason"] == "budget_exhausted"


def heterogeneous_convergence_inputs(adaptive_inputs, tmp_path, patience):
    from mars_titan.simulation.adaptation_scenarios import prepare_adaptation_scenarios

    config, paths = adaptive_inputs
    settings = dict(
        schema_version=1,
        families=["known_signal"],
        generator=dict(assets=2, sessions=12, warmup_sessions=5, regime_sessions=4),
        worlds=dict(train=1, validation=1, audit=1),
        seed_base=191,
        final_test_opened=False,
    )
    other = tmp_path / "heterogeneous"
    catalog = prepare_adaptation_scenarios(settings, other)
    paths["train"][0] = other / next(
        row["path"] for row in catalog["records"] if row["split"] == "train"
    )
    convergence_config(adaptive_inputs, minimum=16, patience=patience)
    return config, paths


@pytest.mark.parametrize("pending_validation", [False, True])
def test_convergence_recovers_heterogeneous_warmup_cursor(
    adaptive_inputs, tmp_path, pending_validation
):
    import torch

    inputs = heterogeneous_convergence_inputs(adaptive_inputs, tmp_path, patience=8)
    output, full = tmp_path / "heterogeneous-resumed", tmp_path / "heterogeneous-full"
    execute_adaptive(inputs, output, "--stop-after", "16", paused=True)
    paused = read(output / "run.json")
    assert paused["transitions"] == 31 and paused["evaluations"] == 2
    assert paused["stale_evaluations"] == 1
    if pending_validation:
        replace_confirmed_progress(
            output,
            evaluations=1,
            stale_evaluations=0,
            evaluated_transitions=0,
            evaluated_optimizer_steps=0,
            evaluation_cursors=[0],
        )
    execute_adaptive(inputs, output, "--resume")
    execute_adaptive(inputs, full)
    resumed = read(output / "run.json")
    assert resumed["evaluation_cursors"] == [0, 31, 32]
    assert resumed["stale_evaluations"] == 2 and resumed["stopping_reason"] == "budget_exhausted"
    first = torch.jit.load(str(latest(output) / "policy.pt"), map_location="cpu").state_dict()
    second = torch.jit.load(str(latest(full) / "policy.pt"), map_location="cpu").state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
    from mars_titan.simulation.adaptive_campaign import Campaign

    campaign = object.__new__(Campaign)
    campaign.validate_convergence_receipt(
        output, resumed, read(inputs[0]), resumed["identity_sha256"]
    )


def test_convergence_admits_heterogeneous_plateau_and_audit(adaptive_inputs, tmp_path):
    from mars_titan.simulation.adaptive_campaign import Campaign

    inputs = heterogeneous_convergence_inputs(adaptive_inputs, tmp_path, patience=1)
    output = tmp_path / "heterogeneous-plateau"
    execute_adaptive(inputs, output)
    report = read(output / "run.json")
    assert report["transitions"] == 31 and report["stopping_reason"] == "early_stop"
    campaign = object.__new__(Campaign)
    campaign.validate_convergence_receipt(
        output, report, read(inputs[0]), report["identity_sha256"]
    )
    execute_adaptive(inputs, output, "--resume")
    execute_audit(inputs, output, tmp_path / "heterogeneous-audit")


def test_convergence_rejects_unbounded_history_before_runtime(adaptive_inputs, tmp_path):
    config = convergence_config(adaptive_inputs)
    config["training"]["total_transitions"] = 65536
    adaptive_inputs[0].write_text(json.dumps(config))
    result = execute_adaptive(adaptive_inputs, tmp_path / "oversized-selector", success=False)
    assert "4096" in result.stderr
    assert not (tmp_path / "oversized-selector").exists()
