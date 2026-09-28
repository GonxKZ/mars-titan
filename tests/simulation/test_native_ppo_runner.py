"""Ejecución PPO nativa con pausas recuperables y selección temporal separada."""

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.simulation.market import MarketTape
from mars_titan.simulation.storage import write_tape

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
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", **(environment or {})),
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
