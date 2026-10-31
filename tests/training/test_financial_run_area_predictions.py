"""Predicciones por área de H1 (#783) en el entrenador cronológico, sin pasos de optimizador.

El registrador de gradientes recorre el bucle hasta cada paso sin modificar pesos, así que
los votos siguen en cero durante todo el recorrido. Con `accumulation_rows`, el término de
las áreas se acumula por bloques de flujos con la misma fracción de etiquetas que la tarea
y se compara en float64 con el del tramo completo.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.holographic_network import HolographicNetworkConfig
from mars_titan.training.financial_run import ChronologicalInference
from tests.training.test_financial_run import explicit_fastpath as explicit_fastpath
from tests.training.test_financial_run import named_records, trainer
from tests.training.test_financial_run import shared as shared

# Con cinco áreas cada una es una modalidad y el corpus de prueba no necesita las familias.
FIVE_AREAS = HolographicNetworkConfig(areas=5)
VOTES = {
    f"holographic_network.{name}"
    for name in (
        "area_vote_norm.weight",
        "area_vote_norm.bias",
        "area_vote_weight",
        "area_vote_bias",
        "area_vote_logits",
    )
}


def elementwise(loss, error, delta):
    if loss == "mae":
        return abs(error)
    if loss == "mse":
        return error * error
    return 0.5 * error * error if abs(error) <= delta else delta * (abs(error) - 0.5 * delta)


@pytest.mark.parametrize("loss", ["mae", "mse", "huber"])
def test_the_area_loss_averages_the_present_areas_of_each_row(loss):
    areas = torch.tensor([[1.0, 5.0, -2.0], [2.0, 9.0, 0.5], [3.0, 3.0, 3.0]], dtype=torch.float64)
    alive = torch.tensor([[True, False, True], [True, True, True], [False, False, False]])
    target = torch.tensor([0.5, 1.0, 7.0], dtype=torch.float64)
    owner = SimpleNamespace(recipe=SimpleNamespace(loss=loss, huber_delta=1.5))
    value = ChronologicalInference._area_loss(owner, areas, alive, target)
    rows = []
    for r in range(3):
        errors = [float(areas[r, a] - target[r]) for a in range(3) if alive[r, a]]
        # Una fila sin áreas presentes aporta cero y sigue contando en la media.
        terms = [elementwise(loss, error, 1.5) for error in errors]
        rows.append(sum(terms) / len(terms) if terms else 0.0)
    assert float(value) == pytest.approx(sum(rows) / 3, rel=1e-15)


def test_the_split_separates_the_task_the_areas_and_their_presence():
    rows = torch.tensor([[0.1, 1.0, 2.0, 1.0, 0.0], [0.2, 3.0, 4.0, 0.0, 1.0]])
    main, areas, alive = ChronologicalInference._split(SimpleNamespace(area_weight=0.5), rows)
    assert torch.equal(main, rows[:, 0]) and torch.equal(areas, rows[:, 1:3])
    assert torch.equal(alive, torch.tensor([[True, False], [False, True]]))
    plain = ChronologicalInference._split(SimpleNamespace(area_weight=None), rows)
    assert plain[0] is rows and plain[1:] == (None, None)


def model(streams, weight):
    network = FIVE_AREAS if weight is None else replace(FIVE_AREAS, area_predictions=weight)
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        holographic_network=network,
    )
    return FinancialPredictor(config, dtype=torch.float64)


def recorded(streams, output, weight, **options):
    engine = trainer(streams, output, model=model(streams, weight), **options)
    report = engine.run()
    return report["history"][1]["train"], named_records(engine)


def test_with_zero_weight_and_no_clipping_the_shared_gradients_are_those_of_h1(shared, tmp_path):
    _, streams = shared
    plain_counters, plain = recorded(streams, tmp_path / "plain", None, max_grad_norm=None)
    counters, voting = recorded(streams, tmp_path / "zero", 0.0, max_grad_norm=None)
    assert "area_loss_sum" not in plain_counters and counters["area_loss_sum"] == 0
    assert len(voting) == len(plain) > 2
    for left, right in zip(voting, plain, strict=True):
        assert set(left) - set(right) == VOTES
        for key, value in right.items():
            if value is None:
                assert left[key] is None, key
            else:
                torch.testing.assert_close(left[key], value, rtol=0, atol=0)


def test_with_clipping_the_votes_only_change_the_common_factor_of_each_step(shared, tmp_path):
    _, streams = shared
    plain = recorded(streams, tmp_path / "plain", None)[1]
    voting = recorded(streams, tmp_path / "zero", 0.0)[1]
    factors = []
    for left, right in zip(voting, plain, strict=True):
        # La norma global incluye los votos, así que el recorte puede escalar distinto todo el
        # gradiente compartido, pero nunca cambiar su dirección.
        factor = left["head.bias"] / right["head.bias"]
        factors.append(float(factor))
        for key, value in right.items():
            if value is not None:
                torch.testing.assert_close(left[key], factor * value, rtol=1e-9, atol=1e-30)
    assert all(0 < value <= 1 for value in factors) and min(factors) < 1


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_block_accumulation_gives_the_gradient_of_the_whole_segment(shared, tmp_path, weight):
    _, streams = shared
    full_counters, full = recorded(streams, tmp_path / "full", weight)
    block_counters, blocks = recorded(streams, tmp_path / "blocks", weight, accumulation_rows=1)
    assert len(full) == len(blocks) > 2
    for left, right in zip(full, blocks, strict=True):
        assert left.keys() == right.keys()
        for key, value in left.items():
            if value is None:
                assert right[key] is None, key
            else:
                torch.testing.assert_close(right[key], value, rtol=1e-9, atol=1e-12)
    assert block_counters["area_loss_sum"] == pytest.approx(
        full_counters["area_loss_sum"], rel=1e-12
    )
    assert (full_counters["area_loss_sum"] > 0) is (weight > 0)


def test_the_weight_scales_only_the_area_term(shared, tmp_path):
    _, streams = shared
    runs = {
        w: recorded(streams, tmp_path / str(w), w, max_grad_norm=None)[1] for w in (0.0, 0.5, 1.0)
    }
    changed = set()
    for zero, half, one in zip(runs[0.0], runs[0.5], runs[1.0], strict=True):
        for name, value in zero.items():
            if value is None:
                assert half[name] is None and one[name] is None, name
                continue
            # El gradiente es el de la tarea más el peso por el de las áreas.
            torch.testing.assert_close(
                one[name] - half[name], half[name] - value, rtol=1e-7, atol=1e-12
            )
            if not torch.equal(half[name], value):
                changed.add(name)
    # Con los votos en cero, el término de las áreas solo mueve sus pesos y sus sesgos.
    assert changed == {"holographic_network.area_vote_weight", "holographic_network.area_vote_bias"}
