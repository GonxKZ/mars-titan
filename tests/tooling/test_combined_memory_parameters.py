"""Contabilidad de parámetros a partir del contrato guardado, sin ajustar pesos."""

import json

import pytest
import torch

from benchmarks import combined_memory_parameters as inventory


def test_roles_separate_frozen_inert_and_trainable_tensors_without_counting_buffers():
    state = dict(
        a=torch.ones(2, 3),
        b=torch.ones(4, dtype=torch.float64),
        c=torch.ones(1),
        buffer=torch.ones(2),
        _extra_state={"version": 1},
    )
    identity = dict(
        parameter_roles=dict(shared=["a"]), frozen_parameters=["b"], inert_parameters=["c"]
    )
    result = inventory.parameter_counts(state, identity)
    assert result["stage_trainable_elements"] == 6
    assert result["registered_parameter_elements"] == 11
    assert result["registered_parameter_bytes"] == 60
    assert result["unclassified_tensor_elements"] == 2
    assert result["roles"]["shared"] == dict(elements=6, bytes=24)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "overlap", "nontensor"])
def test_incomplete_or_ambiguous_parameter_contracts_fail(fault):
    state = dict(a=torch.ones(2))
    identity = dict(parameter_roles=dict(shared=["a"]))
    if fault == "missing":
        identity["parameter_roles"]["shared"] = ["b"]
    elif fault == "duplicate":
        identity["parameter_roles"]["other"] = ["a"]
    elif fault == "overlap":
        identity["frozen_parameters"] = ["a"]
    else:
        state["a"] = {"fake": 2}
    with pytest.raises(ValueError):
        inventory.parameter_counts(state, identity)


def test_selected_checkpoints_separate_inference_size_and_stage_updates(tmp_path, monkeypatch):
    from mars_titan.training.checkpoints import save_training_state

    jobs = [
        dict(id="parent", kind="core"),
        dict(id="reader", kind="readout", parent="parent"),
        dict(id="frozen", kind="reference", parent="parent"),
        dict(id="adapter", kind="post", parent="parent"),
        dict(id="gru", kind="candidate"),
        dict(id="pending", kind="core"),
    ]
    receipt = dict(jobs=jobs)
    (tmp_path / "prepared.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(inventory, "verify_prepared", lambda _: receipt)
    for job in jobs[:-1]:
        folder = tmp_path / "runs" / job["id"]
        folder.mkdir(parents=True)
        report = folder / ("window.json" if job["kind"] == "candidate" else "run.json")
        report.write_text(json.dumps(dict(status="completed", final_test_opened=False)))
        if job["kind"] == "reference":
            continue
        state = dict(a=torch.ones(6))
        identity = dict(parameter_roles=dict(shared=["a"]))
        if job["kind"] == "post":
            identity = dict(parameter_roles=dict(adapters=["b"]), frozen_parameters=["a"])
            state["b"] = torch.ones(3)
        fit_folder = folder / ("run" if job["kind"] == "candidate" else "fit")
        saved = save_training_state(
            fit_folder / "checkpoints",
            dict(global_step=0, model=state),
            identity=identity,
            best=True,
        )
        (fit_folder / "run.json").write_text(
            json.dumps(
                dict(identity=identity, best_checkpoint=dict(sha256=inventory.sha256(saved)))
            )
        )
        save_training_state(
            fit_folder / "checkpoints",
            dict(global_step=1, model={k: v + 1 for k, v in state.items()}),
            identity=identity,
        )
    result = inventory.collect(tmp_path, tmp_path / "inventory.json")
    models = result["models"]
    assert models["reader"]["inference_registered_parameter_elements"] == 12
    assert models["frozen"]["stage_trainable_elements"] == 0
    assert models["frozen"]["inference_registered_parameter_elements"] == 6
    assert models["adapter"]["stage_trainable_elements"] == 3
    assert models["adapter"]["inference_registered_parameter_elements"] == 9
    assert models["gru"]["stage_trainable_elements"] == 6
    assert result["pending"] == ["pending"]
    with pytest.raises(FileExistsError):
        inventory.collect(tmp_path, tmp_path / "inventory.json")


def test_inventory_rejects_an_open_final_test(tmp_path, monkeypatch):
    monkeypatch.setattr(
        inventory, "verify_prepared", lambda _: dict(jobs=[dict(id="p", kind="core")])
    )
    folder = tmp_path / "runs/p"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps(dict(status="completed", final_test_opened=True)))
    with pytest.raises(ValueError, match="reserva"):
        inventory.collect(tmp_path, tmp_path / "out.json")


def test_inventory_cli_forwards_only_the_supplied_paths(tmp_path, monkeypatch, capsys):
    import sys

    called = []

    def collect(work, output):
        called.append((work, output))
        return dict(models={}, pending=[])

    monkeypatch.setattr(inventory, "collect", collect)
    monkeypatch.setattr(
        sys, "argv", ["inventory", "--work", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )
    inventory.main()
    assert called == [(tmp_path, tmp_path / "out.json")]
    assert json.loads(capsys.readouterr().out) == dict(models=0, pending=0)
