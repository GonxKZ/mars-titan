"""La medida de coste de los objetivos de salida recorre cada modelo pedido y no ajusta nada."""

import json

import pytest
import torch

from benchmarks import objective_variants_cost as cost
from mars_titan.models.titans.skewed_t_objective import skewed_t_beta_nll

NAMES = ["reference_mae", "laplace_scale_head_v1", "skewed_t_objective_v1"]


@pytest.fixture(autouse=True)
def restored_runtime():
    threads, fastpath = torch.get_num_threads(), torch.backends.mha.get_fastpath_enabled()
    yield
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(fastpath)


def small(output, **options):
    values = dict(rows=4, instants=2, repetitions=1, warmup=0, threads=1, names=NAMES)
    return cost.measure(output, **values | options)


def test_a_small_measure_covers_f1_and_l2_and_keeps_the_parameters(tmp_path):
    output = tmp_path / "cost.json"
    small(output)
    receipt = json.loads(output.read_text())
    assert receipt["issues"] == [749, 787] and list(receipt["models"]) == NAMES
    assert receipt["optimizer_steps"] == 0 and receipt["training_runs"] == 0
    assert "src/mars_titan/models/titans/skewed_t_objective.py" in receipt["source_sha256"]
    models = receipt["models"]
    assert all(model["parameters_unchanged"] for model in models.values())
    # F1 añade una proyección de H a 1 y L2 tres, con sus sesgos.
    hidden = cost.BASE["hidden_size"]
    added = [models[name]["added_parameters"] for name in NAMES]
    assert added == [0, hidden + 1, 3 * (hidden + 1)]
    assert models["reference_mae"]["relative_to_control"]["backward_seconds"] == 1
    with pytest.raises(FileExistsError):
        small(output)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(rows=0), "presupuesto"),
        (dict(threads=9), "presupuesto"),
        (dict(names=NAMES[1:]), "primero el control"),
        (dict(names=[*NAMES, NAMES[2]]), "primero el control"),
        (dict(names=[NAMES[0], "unknown"]), "primero el control"),
    ],
)
def test_invalid_measures_are_refused_before_building_models(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        small(tmp_path / "cost.json", **options)
    assert not (tmp_path / "cost.json").exists()


def test_each_model_is_measured_with_its_own_loss():
    spec = cost.specification()
    inputs = cost.batches(spec, 3, 2)
    targets = [torch.full((3,), value) for value in (0.01, -0.02)]
    for name in ("reference_mae", "skewed_t_objective_v1"):
        config = cost.FinancialConfig(
            spec, variant="mac_online", seed=42, **(cost.BASE | cost.MODELS[name])
        )
        model = cost.FinancialPredictor(config, dtype=torch.float32)
        state = model.initial_state(inputs[0].flow_ids)
        prepared = []
        for batch in inputs:
            result = model.prepare(batch, state, session_complete=True)
            state = result.next_state
            prepared.append(result)
        value = cost._loss(model, prepared, targets, inputs)
        target = torch.cat(targets)
        if model.output_objective is None:
            points = torch.cat([result.point_predictions for result in prepared])
            expected = torch.nn.functional.l1_loss(points, target)
        else:
            outputs = torch.cat([result.outputs for result in prepared])
            expected = skewed_t_beta_nll(outputs, target)
        assert torch.equal(value, expected), name
