"""Recorrido cronológico con la escala de Laplace de F1 (#749), sin pasos de optimizador."""

import math

import pytest
import torch

from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.laplace_scale import LaplaceScaleHead
from mars_titan.training import financial_run
from mars_titan.training.financial_run import (
    ChronologicalInference,
    ChronologicalRecipe,
    ChronologicalTrainer,
)
from mars_titan.training.titans_walk_forward import PredictionRows
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    corpus,
    entries,
    named_records,
)

HEAD = LaplaceScaleHead(floor=1e-6, initial_scale=0.02)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("laplace-scale"))[1]


def predictor(streams, *, head=HEAD, scale_weight=None):
    # La memoria de la receta de campaña conserva la escala de la salida de MAC. Con la
    # memoria v1 del fixture el estado codificado es casi nulo y la escala no varía.
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        laplace_scale_head=head,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if scale_weight is not None:
        # Estado posterior simulado sin optimizador, con la escala dependiente de la entrada.
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            model.output_objective.weight.normal_(0.0, scale_weight, generator=generator)
        model._seal_parameters()
    return model


def same(left, right):
    return left is right if left is None or right is None else torch.equal(left, right)


def recipe(**options):
    values = dict(truncation=3, epochs=1, block_rows=2, max_grad_norm=None, selection=SELECTION)
    return ChronologicalRecipe(**values | options)


def trainer(streams, output, *, model=None, **options):
    return ChronologicalTrainer(
        model or predictor(streams),
        recipe(**options),
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def test_every_update_uses_the_beta_nll_and_matches_the_l1_control_at_zero_scale_weight(
    shared, tmp_path, monkeypatch
):
    calls, original = [], financial_run.laplace_beta_nll

    def spy(prediction, target):
        calls.append(prediction.shape)
        return original(prediction, target)

    monkeypatch.setattr(financial_run, "laplace_beta_nll", spy)
    engine = trainer(shared, tmp_path / "laplace")
    before = engine.predictor._parameter_id
    assert engine.run()["status"] == "completed"
    assert engine.predictor._parameter_id == before
    updates = entries(engine.audit, "update")
    assert len(calls) == len(updates) == engine.optimizer.calls > 2
    assert all(len(shape) == 2 and shape[1] == 2 for shape in calls)
    control = trainer(shared, tmp_path / "control", model=predictor(shared, head=None))
    control.run()
    laplace, reference = named_records(engine), named_records(control)
    assert len(laplace) == len(reference)
    for left, right in zip(laplace, reference, strict=True):
        assert torch.count_nonzero(left.pop("output_objective.weight"))
        assert left.pop("output_objective.bias").abs() > 0
        # Sin paso real los pesos de la escala siguen en cero y el tronco recibe el
        # gradiente del control: el de la mediana es exactamente el del error absoluto.
        assert left.keys() == right.keys()
        for name, value in left.items():
            assert same(value, right[name]), name


def test_replay_by_flow_blocks_reproduces_the_gradients(shared, tmp_path):
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        engine = trainer(
            shared,
            tmp_path / name,
            model=predictor(shared, scale_weight=0.05),
            accumulation_rows=rows,
        )
        engine.run()
        records.append(named_records(engine))
    whole, blocks = records
    assert len(whole) == len(blocks) > 2
    for left, right in zip(whole, blocks, strict=True):
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            else:
                torch.testing.assert_close(value, right[name], rtol=1e-10, atol=1e-14)
    assert any(torch.count_nonzero(record["fusion.0.weight"]) for record in whole)


def test_future_suffix_does_not_change_past_scales_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        streams = corpus(tmp_path / name, perturb_after=perturb)[1]
        engine = trainer(
            streams, tmp_path / name / "run", model=predictor(streams, scale_weight=0.05)
        )
        engine.run()
        engines.append(engine)
    base, future = engines
    steps = [u[2] for u in entries(base.audit, "update") if u[1] <= cut]
    assert steps and steps == [u[2] for u in entries(future.audit, "update") if u[1] <= cut]
    for left, right in zip(named_records(base)[: len(steps)], named_records(future), strict=False):
        for name, value in left.items():
            assert same(value, right[name]), name
    later = named_records(base)[len(steps)], named_records(future)[len(steps)]
    assert not torch.equal(later[0]["output_objective.weight"], later[1]["output_objective.weight"])


def test_emitted_rows_carry_the_laplace_quantiles_of_each_prediction(shared):
    model = predictor(shared, scale_weight=0.05)
    inference = ChronologicalInference(model, recipe())
    assert inference.emits_levels and not inference.quantiles
    rows = []
    metrics = inference.predict(shared["validation"], rows)
    assert len(rows) == metrics["labels"] > 0
    scales = []
    for _, _, issued, _, levels in rows:
        assert len(levels) == 5 and levels[2] == issued
        assert all(a <= b for a, b in zip(levels, levels[1:], strict=False))
        scales.append((levels[3] - levels[2]) / math.log(5))
    # Con pesos de escala no nulos, cada fila emite su propia escala.
    assert len({round(value, 12) for value in scales}) > 1 and min(scales) > 0
    table = PredictionRows(inference.emits_levels)
    for row in rows:
        if row[1] < 1_704_067_200_000_000:
            table.append(row)
    assert table.count + len(table.buffer) > 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(loss="mse"), "receta mae"),
        (dict(loss="huber"), "receta mae"),
        (dict(distillation_weight=0.5), "sin destilación"),
    ],
)
def test_recipes_without_the_declared_objective_are_refused(shared, tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        trainer(shared, tmp_path / "run", **options)


def test_posttraining_is_refused_until_it_is_declared(shared, tmp_path):
    with pytest.raises(ValueError, match="postentrenamiento"):
        ChronologicalTrainer(
            predictor(shared),
            recipe(),
            train=shared["train"],
            validation=shared["validation"],
            output=tmp_path / "run",
            optimizer_factory=RecordingOptimizer,
            posttraining=dict(kind="adapters"),
        )


def test_the_run_identity_records_the_scale_head(shared, tmp_path):
    engine = trainer(shared, tmp_path / "laplace")
    control = trainer(shared, tmp_path / "control", model=predictor(shared, head=None))
    assert engine.identity["predictor"]["laplace_scale_head"] == HEAD.identity()
    assert "laplace_scale_head" not in control.identity["predictor"]
    assert engine.run_id != control.run_id
