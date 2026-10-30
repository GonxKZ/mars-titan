"""Recorrido cronológico con la t asimétrica de L2 (#787), sin pasos de optimizador."""

import pytest
import torch

from mars_titan.models.titans import skewed_t_objective
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.skewed_t_objective import SkewedTObjective
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

OBJECTIVE = SkewedTObjective(initial_scale=0.02, initial_dof=6.0, max_skew=2.5)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("skewed-t"))[1]


def predictor(streams, *, objective=OBJECTIVE, weight=None):
    # La misma memoria que en las pruebas de F1: con la v1 del fixture el estado codificado es
    # casi nulo y σ, ξ y ν apenas variarían entre filas.
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        memory_residual_layer_norm=True,
        gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
        skewed_t_objective=objective,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if weight is not None:
        # Estado posterior simulado sin optimizador, con σ, ξ y ν dependientes de la entrada.
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            model.output_objective.weight.normal_(0.0, weight, generator=generator)
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


def score(outputs, target):
    """∂L/∂μ de la β-NLL en cada fila, la puntuación acotada de la t asimétrica."""
    point, scale, skew, dof = outputs.unbind(-1)
    side = torch.where(target < point, skew, 1 / skew)
    x = (target - point) / scale * side
    return -side * (dof + 1) * x / (dof + x.square()) / len(target)


@pytest.mark.parametrize("weight", [None, 0.05])
def test_every_update_uses_the_beta_nll_and_the_head_gets_the_t_score(
    shared, tmp_path, monkeypatch, weight
):
    calls, original = [], skewed_t_objective.skewed_t_beta_nll

    def spy(prediction, target):
        calls.append((prediction.detach().clone(), target.clone()))
        return original(prediction, target)

    # El objetivo de L2 calcula su pérdida con la función de su módulo.
    monkeypatch.setattr(skewed_t_objective, "skewed_t_beta_nll", spy)
    engine = trainer(shared, tmp_path / "skewed", model=predictor(shared, weight=weight))
    before = engine.predictor._parameter_id
    assert engine.run()["status"] == "completed"
    assert engine.predictor._parameter_id == before
    updates = entries(engine.audit, "update")
    records = named_records(engine)
    assert len(calls) == len(updates) == len(records) == engine.optimizer.calls > 2
    for (outputs, target), record in zip(calls, records, strict=True):
        assert outputs.shape == (len(target), 4)
        assert torch.count_nonzero(record["output_objective.weight"].abs().sum(1)) == 3
        assert torch.count_nonzero(record["output_objective.bias"]) == 3
        # μ solo entra en la pérdida por su fila, así que el sesgo de la cabeza escalar suma
        # las puntuaciones de las filas que maduran en la actualización.
        expected = score(outputs, target).sum().reshape(1)
        torch.testing.assert_close(record["head.bias"], expected, rtol=1e-10, atol=1e-15)


def test_replay_by_flow_blocks_reproduces_the_gradients(shared, tmp_path):
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        engine = trainer(
            shared, tmp_path / name, model=predictor(shared, weight=0.05), accumulation_rows=rows
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


def test_future_suffix_does_not_change_past_outputs_or_gradients(tmp_path):
    cut = decision("2022-12-09")
    engines = []
    for name, perturb in (("base", None), ("future", cut)):
        streams = corpus(tmp_path / name, perturb_after=perturb)[1]
        engine = trainer(streams, tmp_path / name / "run", model=predictor(streams, weight=0.05))
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


def test_emitted_rows_carry_the_median_and_the_quantiles_of_each_distribution(shared):
    model = predictor(shared, weight=0.05)
    inference = ChronologicalInference(model, recipe())
    assert inference.emits_levels and not inference.quantiles
    rows = []
    metrics = inference.predict(shared["validation"], rows)
    assert len(rows) == metrics["labels"] > 0
    asymmetry = []
    for _, _, issued, _, levels in rows:
        # La predicción emitida es la mediana de la distribución ajustada.
        assert len(levels) == 5 and levels[2] == issued
        assert all(a < b for a, b in zip(levels, levels[1:], strict=False))
        asymmetry.append((levels[3] - levels[2]) - (levels[2] - levels[1]))
    # Con ξ ≠ 1 los intervalos dejan de ser simétricos alrededor de la mediana, y cada fila
    # emite su propia forma.
    assert len({round(value, 12) for value in asymmetry}) > 1
    assert any(value > 0 for value in asymmetry) or any(value < 0 for value in asymmetry)
    table = PredictionRows(inference.emits_levels)
    for row in rows:
        if row[1] < 1_704_067_200_000_000:
            table.append(row)
    assert table.count + len(table.buffer) > 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(loss="mse"), "sustituye la pérdida mae"),
        (dict(loss="huber"), "sustituye la pérdida mae"),
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


def test_the_run_identity_records_the_objective(shared, tmp_path):
    engine = trainer(shared, tmp_path / "skewed")
    control = trainer(shared, tmp_path / "control", model=predictor(shared, objective=None))
    assert engine.identity["predictor"]["output_objective"] == OBJECTIVE.identity()
    assert "output_objective" not in control.identity["predictor"]
    assert engine.run_id != control.run_id
