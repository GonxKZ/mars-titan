import importlib
import os
import random

import numpy as np
import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def training_module():
    try:
        return importlib.import_module("mars_titan.budget_training")
    except ModuleNotFoundError:
        pytest.fail("Falta el entrenamiento supervisado de medición")


@pytest.mark.parametrize("kind", ["mlp", "gru"])
def test_real_labels_update_all_modalities_and_resume_exactly(tmp_path, kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible, no se sustituye por CPU")
    from mars_titan.profiling import CostProbe

    module = training_module()
    module.seed_run(42)
    dims = {"prices": 5, "news": 4, "charts": 3, "fundamentals": 2, "macro": 2}
    model = CostProbe(kind, dims, context=4).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = {k: torch.randn((3, 4, d) if k == "prices" else (3, d)) for k, d in dims.items()}
    batch = (inputs, torch.tensor([0.02, -0.03, 0.01]))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    module.train_step(model, optimizer, batch, torch.device("cuda:0"))
    assert all(not torch.equal(before[name], p) for name, p in model.named_parameters())
    path = tmp_path / "epoch-1.pt"
    module.save_checkpoint(
        path, model, optimizer, next_epoch=1, config={"seed": 42}, hashes={"input": "abc"}
    )
    expected_random = (random.random(), np.random.rand(), torch.rand(2, device="cuda:0"))
    module.train_step(model, optimizer, batch, torch.device("cuda:0"))
    expected = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert (
        module.load_checkpoint(path, model, optimizer, config={"seed": 42}, hashes={"input": "abc"})
        == 1
    )
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2, device="cuda:0"), expected_random[2])
    module.train_step(model, optimizer, batch, torch.device("cuda:0"))
    assert all(torch.equal(expected[name], p) for name, p in model.named_parameters())
    with pytest.raises(ValueError, match="entradas|configuración"):
        module.load_checkpoint(path, model, optimizer, config={"seed": 43}, hashes={"input": "abc"})


def test_partition_requires_target_maturity_before_training_cutoff():
    module = training_module()
    from datetime import UTC, datetime

    assert (
        module.target_partition(
            datetime(2022, 12, 30, tzinfo=UTC), datetime(2023, 1, 3, tzinfo=UTC)
        )
        is None
    )
    assert (
        module.target_partition(
            datetime(2022, 12, 29, tzinfo=UTC), datetime(2022, 12, 30, tzinfo=UTC)
        )
        == "train"
    )
    assert (
        module.target_partition(
            datetime(2023, 12, 29, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)
        )
        is None
    )
    assert (
        module.target_partition(datetime(2023, 1, 3, tzinfo=UTC), datetime(2023, 1, 4, tzinfo=UTC))
        == "validation"
    )


def test_validation_counts_partial_batches_without_updating_weights():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible, no se sustituye por CPU")
    from mars_titan.profiling import CostProbe

    module = training_module()
    dims = dict(prices=5, news=4, charts=3, fundamentals=2, macro=2)
    model = CostProbe("mlp", dims, context=4).to("cuda:0")
    batches = [
        (
            {k: torch.ones((n, 4, d) if k == "prices" else (n, d)) for k, d in dims.items()},
            torch.zeros(n),
        )
        for n in [3, 1]
    ]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    result = module.run_epoch(model, None, batches, torch.device("cuda:0"))
    assert result["samples"] == 4 and result["steps"] == 2
    assert result["elapsed_seconds"] > 0 and result["step_p99_ms"] > 0
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    with pytest.raises(ValueError, match="No hay muestras supervisadas"):
        module.run_epoch(model, None, [], torch.device("cuda:0"))


def test_process_tree_reports_current_process_memory():
    result = training_module().process_memory()
    assert result["rss_mib"] > 0 and result["pss_mib"] > 0
    assert result["processes"] >= 1


def test_labeled_records_exclude_missing_targets_and_other_partition():
    from datetime import UTC, datetime

    module = training_module()
    dates = [datetime(2022, 12, 28, tzinfo=UTC), datetime(2023, 1, 4, tzinfo=UTC)]
    records = [
        {"cursor": ("US/A", n), "prediction_at": day, "inputs": {"prices": n}}
        for n, day in enumerate(dates)
    ]
    targets = {
        "US/A": {dates[0].isoformat(): (0.03, "train"), dates[1].isoformat(): (-0.02, "validation")}
    }
    result = list(module.labeled_records(records, targets, "train"))
    assert len(result) == 1 and float(result[0][1]) == pytest.approx(0.03)
    assert result[0][0] == {"prices": 0}
    assert list(module.labeled_records(records, {}, "train")) == []


@pytest.mark.parametrize("options", [{"epochs": 0}, {"panel_sizes": [3]}, {"kinds": ["unknown"]}])
def test_budget_run_rejects_invalid_workload_before_touching_data(tmp_path, options):
    with pytest.raises(ValueError, match="carga"):
        training_module().train_budget_grid(
            tmp_path / "missing", tmp_path / "report.json", tmp_path / "out", **options
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "loss,expected_objective,expected_weight",
    [("mse", 5.0, 0.0), ("mae", 2.0, 0.8), ("huber", 0.875, 0.9)],
)
def test_objectives_keep_common_metrics_and_apply_the_expected_gradient(
    loss, expected_objective, expected_weight
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible, no se sustituye por CPU")

    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            return inputs["prices"] * self.weight

    model = ScalarModel().to("cuda:0")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = ({"prices": torch.tensor([1.0, 3.0])}, torch.zeros(2))
    result = training_module().run_epoch(
        model, optimizer, [batch], torch.device("cuda:0"), loss=loss, huber_delta=0.5
    )
    assert result["objective_loss"] == pytest.approx(expected_objective)
    assert result["diagnostic_mse"] == pytest.approx(5.0)
    assert result["diagnostic_mae"] == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(expected_weight, abs=1e-7)


@pytest.mark.parametrize("loss,delta", [("other", 0.1), ("huber", 0.0), ("huber", float("nan"))])
def test_invalid_loss_options_fail_before_touching_a_model(loss, delta):
    with pytest.raises(ValueError, match="pérdida|Huber"):
        training_module().run_epoch(None, None, [], None, loss=loss, huber_delta=delta)


def test_target_broadcasting_is_rejected_before_updating_weights():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA no está disponible, no se sustituye por CPU")

    class ColumnModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, inputs):
            return self.linear(inputs["prices"])

    model = ColumnModel().to("cuda:0")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    batch = ({"prices": torch.tensor([[1.0], [3.0]])}, torch.zeros(2))
    with pytest.raises(ValueError, match="forma"):
        training_module().train_step(model, optimizer, batch, torch.device("cuda:0"))
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
