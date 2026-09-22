import importlib
import json
import os
import random
import select
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module("mars_titan.training.checkpoints")


IDENTITY = {"data": "edition-a", "optimizer": "AdamW", "precision": "float32"}


class UnsupportedMapping(dict):
    pass


def save(directory, step, **options):
    return module().save_training_state(
        directory,
        {"global_step": step, "model": {"weight": torch.tensor([float(step)])}},
        identity=IDENTITY,
        **options,
    )


def test_two_recent_best_and_pinned_states_survive_retention(tmp_path):
    best = save(tmp_path, 1, best=True)
    pinned = save(tmp_path, 2, pin=True)
    discarded = save(tmp_path, 3)
    recent = save(tmp_path, 4)
    latest = save(tmp_path, 5)
    assert all(path.is_file() for path in (best, pinned, recent, latest))
    assert not discarded.exists()
    state = module().load_training_state(tmp_path, expected_identity=IDENTITY)
    assert state["global_step"] == 5
    torch.testing.assert_close(state["model"]["weight"], torch.tensor([5.0]))


@pytest.mark.parametrize("field", ["data", "optimizer", "precision"])
def test_changed_identity_is_rejected_without_overwriting(tmp_path, field):
    path = save(tmp_path, 1)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="identidad"):
        module().load_training_state(tmp_path, expected_identity={**IDENTITY, field: "other"})
    with pytest.raises(ValueError, match="identidad"):
        module().save_training_state(
            tmp_path, {"global_step": 2}, identity={**IDENTITY, field: "other"}
        )
    assert path.read_bytes() == original


def test_corrupt_latest_falls_back_explicitly_to_last_valid_state(tmp_path):
    save(tmp_path, 1)
    latest = save(tmp_path, 2)
    latest.write_bytes(b"incompleto")
    with pytest.warns(RuntimeWarning, match="anterior"):
        state = module().load_training_state(tmp_path, expected_identity=IDENTITY)
    assert state["global_step"] == 1
    save(tmp_path, 2)
    assert module().load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 2


def test_unowned_directory_and_symlink_are_not_adopted(tmp_path):
    (tmp_path / "original.txt").write_text("conservar")
    with pytest.raises(ValueError, match="directorio"):
        save(tmp_path, 1)
    assert (tmp_path / "original.txt").read_text() == "conservar"
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="enlace"):
        save(link, 1)


def test_disk_failure_preserves_previous_commit(tmp_path, monkeypatch):
    save(tmp_path, 1)
    checkpoint = module()

    def fail(*args, **kwargs):
        raise OSError(28, "Sin espacio")

    monkeypatch.setattr(checkpoint.torch, "save", fail)
    with pytest.raises(OSError):
        save(tmp_path, 2)
    assert checkpoint.load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 1


@pytest.mark.parametrize("metadata", [{Path("fold-1"): 1}, UnsupportedMapping(value=1)])
def test_unloadable_state_is_rejected_before_replacing_previous_commit(tmp_path, metadata):
    previous = save(tmp_path, 1)
    with pytest.raises(ValueError, match="estado|tipos"):
        module().save_training_state(
            tmp_path, {"global_step": 2, "metadata": metadata}, identity=IDENTITY
        )
    assert previous.is_file()
    assert module().load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 1


def test_oversized_identity_is_rejected_before_creating_directory(tmp_path):
    output = tmp_path / "checkpoints"
    with pytest.raises(ValueError, match="identidad"):
        module().save_training_state(output, {"global_step": 1}, identity={"data": "x" * 262144})
    assert not output.exists()


def test_killed_writer_never_publishes_partial_checkpoint(tmp_path):
    save(tmp_path, 1)
    script = """
import signal, sys, torch
from pathlib import Path
from mars_titan.training.checkpoints import save_training_state
def interrupted(state, stream):
    stream.write(b'incomplete')
    stream.flush()
    print('writing', flush=True)
    signal.pause()
torch.save = interrupted
save_training_state(Path(sys.argv[1]), {'global_step': 2},
 identity={'data':'edition-a','optimizer':'AdamW','precision':'float32'})
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)], stdout=subprocess.PIPE
    )
    try:
        assert select.select([process.stdout], [], [], 20)[0], "El proceso no llegó a escritura"
        assert process.stdout.readline().strip() == b"writing"
        process.kill()
        process.wait(timeout=10)
        assert (
            module().load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 1
        )
        save(tmp_path, 2)
        assert (
            module().load_training_state(tmp_path, expected_identity=IDENTITY)["global_step"] == 2
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdout.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_resume_reproduces_optimizer_rng_and_next_update_exactly(tmp_path, device):
    from mars_titan.budget_training import seed_run
    from mars_titan.profiling import CostProbe

    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.fail("Se requiere cuda:0 para comprobar la equivalencia neuronal")
    checkpoint = module()
    seed_run(42)
    dimensions = dict(prices=5, news=2, charts=1, fundamentals=1, macro=1)
    model = CostProbe("gru", dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def step():
        inputs = {
            name: torch.randn(3, 2, 5, device=device)
            if name == "prices"
            else torch.randn(3, size, device=device)
            for name, size in dimensions.items()
        }
        target = torch.full((3,), random.random() + float(np.random.random()), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    step()
    checkpoint.save_training_state(
        tmp_path,
        dict(
            global_step=1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=checkpoint.capture_rng(device),
            confirmed_cursor={"consumed": 3},
        ),
        identity=IDENTITY,
    )
    expected_loss = step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    state = checkpoint.load_training_state(tmp_path, expected_identity=IDENTITY)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    checkpoint.restore_rng(state["rng"], device)
    assert state["confirmed_cursor"]["consumed"] == 3
    assert torch.equal(step(), expected_loss)
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())


def test_signal_requests_stop_without_interrupting_current_operation():
    original = signal.getsignal(signal.SIGTERM)
    with module().StopRequest() as stop:
        assert not stop.requested
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.requested
    assert signal.getsignal(signal.SIGTERM) == original


def test_corrupt_index_is_rejected_instead_of_scanning_uncommitted_files(tmp_path):
    save(tmp_path, 1)
    (tmp_path / "latest.json").write_text(json.dumps({"schema_version": 100}))
    with pytest.raises(ValueError, match="manifiesto|esquema"):
        module().load_training_state(tmp_path, expected_identity=IDENTITY)
