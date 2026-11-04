"""La medida de coste de #791 recorre cada modelo declarado con formas pequeñas y no ajusta nada."""

import pytest
import torch

from benchmarks import mature_reliability_cost as cost


@pytest.fixture(autouse=True)
def restored_runtime():
    threads, fastpath = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def test_a_small_measure_covers_every_arm_and_keeps_the_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(cost, "EVENT_ROWS", (8,))
    output = tmp_path / "cost.json"
    receipt = cost.measure(output, rows=4, instants=3, repetitions=1, warmup=0, threads=1)
    names = list(cost.arms())
    assert names[0] == "reference" and list(receipt["models"]) == names
    models = receipt["models"]
    assert all(model["parameters_unchanged"] for model in models.values())
    assert models["reference"]["added_parameters"] == 0
    assert {models[name]["added_parameters"] for name in names[1:]} == {6}
    assert set(receipt["accumulator"]) == {f"{name}/8" for name in names[1:]}
    assert output.exists() and receipt["shape"] == dict(rows=4, instants=3, hidden_size=64)
    with pytest.raises(FileExistsError):
        cost.measure(output, rows=4, instants=3, repetitions=1, warmup=0, threads=1)
    with pytest.raises(ValueError, match="presupuesto"):
        cost.measure(
            tmp_path / "other.json", rows=4, instants=1, repetitions=1, warmup=0, threads=1
        )


def test_the_walk_feeds_the_signal_with_the_previous_labels(monkeypatch):
    spec = cost.specification()
    inputs = cost.batches(spec, 3, 3)
    targets = [torch.full((3,), value) for value in (0.01, -0.02, 0.03)]
    changes = cost.arms()["shared_signal"]
    config = cost.FinancialConfig(spec, variant="mac_online", seed=42, **(cost.BASE | changes))
    model = cost.FinancialPredictor(config, dtype=torch.float32)
    settled, settle = [], cost.ReliabilitySignal.settle

    def spy(self, matured):
        matured = list(matured)
        settled.append(matured)
        return settle(self, matured)

    monkeypatch.setattr(cost.ReliabilitySignal, "settle", spy)
    prepared = cost.walk(model, inputs, targets, differentiable=False)
    assert len(prepared) == 3 and len(settled) == 2
    for index, matured in enumerate(settled):
        issued = prepared[index].point_predictions.tolist()
        assert [row[1] for row in matured] == issued
        assert [row[2] for row in matured] == targets[index].tolist()
