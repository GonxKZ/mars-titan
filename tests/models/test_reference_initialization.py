"""Inicialización separada del estado de recuperación de una ejecución."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.models.baselines.initialization")
    except ModuleNotFoundError:
        pytest.fail("Falta la inicialización validada de referencias")


def checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1)
    config = {
        "kind": "gru",
        "dimensions": {"prices": 5},
        "context": 64,
        "hidden_size": 32,
        "moving_average_kernel": None,
        "price_output_horizon": None,
        "loss": "mse",
    }
    hashes = {
        "samples.parquet": "data",
        "src/model.py": "code",
        "old/run/targets/A-targets.parquet": "labels",
    }
    state = {
        "model": {"weight": torch.tensor([[0.25, -0.5]]), "bias": torch.tensor([0.1])},
        "config": config,
        "input_hashes": hashes,
        "next_epoch": 3,
        "optimizer": {"state": "not_restored"},
    }
    path = tmp_path / "source.pt"
    torch.save(state, path)
    expected_hashes = {
        "samples.parquet": "data",
        "src/model.py": "code",
        "new/run/targets/A-targets.parquet": "labels",
    }
    return model, path, config, expected_hashes


def test_posttraining_loads_only_compatible_weights_and_preserves_source(tmp_path):
    torch = pytest.importorskip("torch")
    model, path, config, hashes = checkpoint(tmp_path)
    before = path.read_bytes()
    result = module().initialize_weights(
        model, path, config={**config, "loss": "mae"}, hashes=hashes
    )
    torch.testing.assert_close(model.weight, torch.tensor([[0.25, -0.5]]))
    torch.testing.assert_close(model.bias, torch.tensor([0.1]))
    assert result["source_completed_epochs"] == 3
    assert result["policy"] == "weights_only_new_optimizer_and_rng"
    assert len(result["sha256"]) == 64 and path.read_bytes() == before


@pytest.mark.parametrize("change", ["architecture", "labels", "code", "data"])
def test_incompatible_initialization_does_not_change_parameters(tmp_path, change):
    torch = pytest.importorskip("torch")
    model, path, config, hashes = checkpoint(tmp_path)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if change == "architecture":
        config = {**config, "kind": "dlinear"}
    else:
        key = {
            "labels": "new/run/targets/A-targets.parquet",
            "code": "src/model.py",
            "data": "samples.parquet",
        }[change]
        hashes = {**hashes, key: "changed"}
    with pytest.raises(ValueError, match="arquitectura|huellas"):
        module().initialize_weights(model, path, config=config, hashes=hashes)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_oversized_checkpoint_is_rejected_before_loading(tmp_path):
    model, path, config, hashes = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="presupuesto"):
        module().initialize_weights(model, path, config=config, hashes=hashes, max_bytes=1)


def test_compressed_checkpoint_is_rejected_before_expanding_storage(tmp_path):
    import zipfile

    model, path, config, hashes = checkpoint(tmp_path)
    compressed = tmp_path / "compressed.pt"
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for name in source.namelist():
            target.writestr(name, source.read(name))
    with pytest.raises(ValueError, match="compresión|presupuesto"):
        module().initialize_weights(model, compressed, config=config, hashes=hashes)


@pytest.mark.parametrize("fault", ["keys", "shape", "dtype", "nan", "epoch"])
def test_corrupt_checkpoint_never_partially_updates_model(tmp_path, fault):
    torch = pytest.importorskip("torch")
    model, path, config, hashes = checkpoint(tmp_path)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    state = torch.load(path, weights_only=True)
    if fault == "keys":
        state["model"].pop("bias")
    elif fault == "shape":
        state["model"]["bias"] = torch.ones(2)
    elif fault == "dtype":
        state["model"]["bias"] = state["model"]["bias"].double()
    elif fault == "nan":
        state["model"]["bias"][0] = float("nan")
    else:
        state["next_epoch"] = -1
    torch.save(state, path)
    with pytest.raises(ValueError, match="pesos|época"):
        module().initialize_weights(model, path, config=config, hashes=hashes)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
