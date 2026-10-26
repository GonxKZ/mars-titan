"""Rangos gaussianos por sesión y pérdida de IC con cabeza de escala (F4 de #749).

Las pruebas calculan salidas y gradientes sin optimizador ni pasos.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import special, stats
from test_financial_adapter import DIMENSIONS, api, setup, specification

from mars_titan.models.titans.session_ic_objective import (
    RELATIVE_FLOOR,
    SessionICObjective,
    session_correlations,
    session_ic_loss,
    standardized_scores,
)
from mars_titan.models.titans.session_ranks import (
    SESSION_RANK_INPUTS,
    gaussian_ranks,
    rank_session_inputs,
)
from mars_titan.training.search_cases import checked_predictor_changes

OBJECTIVE = SessionICObjective(minimum_rows=3)
AT = 1_609_459_200_000_000


def session_batch(rows=6, *, at=AT, seed=5, markets=None):
    """Lote histórico de un instante con `rows` activos y presencias variadas."""
    generator = np.random.default_rng(seed)
    markets = markets or ["US"] * rows
    values = {
        name: generator.normal(size=(rows, 64, width) if name == "prices" else (rows, width))
        .astype(np.float32)
        .round(3)
        for name, width in DIMENSIONS.items()
    }
    presence = np.ones((rows, 5), dtype=np.bool_)
    presence[::3, 1] = False
    values["news"][::3] = 0
    values["fundamentals"][:, 1:] = [1, 0.5]
    values["macro"][:] = [0.2, 1, 0]
    names = [f"{market}/A{index:03}" for index, market in enumerate(markets)]
    raw = dict(
        inputs=values,
        presence=presence,
        sample_ids=[f"{name}/{at}" for name in names],
        market=markets,
        prediction_at=np.full(rows, at, dtype="datetime64[us]"),
        input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
    )
    return api().DecisionBatch.from_corpus(raw, specification(), dtype=torch.float64)


def model(**changes):
    base, _ = setup()
    return api().FinancialPredictor(replace(base.config, **changes), dtype=torch.float64)


def reference_scores(values, present):
    """Puntuaciones normales con `scipy.stats.rankdata` por columna."""
    expected = np.zeros_like(values)
    for column in range(values.shape[1]):
        rows = present[:, column]
        ranks = stats.rankdata(values[rows, column], method="average")
        expected[rows, column] = special.ndtri((ranks - 0.5) / rows.sum())
    return expected


def test_gaussian_ranks_match_the_reference_with_ties_and_absent_rows():
    values = np.array(
        [[0.3, 1.0, 4.0], [0.1, 2.0, 4.0], [0.3, 0.0, 4.0], [0.7, 5.0, 4.0], [0.2, 9.0, 4.0]]
    )
    present = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=bool)
    result = gaussian_ranks(torch.tensor(values), torch.tensor(present))
    expected = reference_scores(values, present)
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=1e-15)
    assert result[4, 0] == 0 and result[1, 1] == 0
    # Dos filas presentes empatadas reciben el rango medio, es decir, la mediana normal.
    assert torch.equal(result[:2, 2], torch.zeros(2, dtype=torch.float64))
    single = gaussian_ranks(torch.tensor([[3.0], [8.0]]), torch.tensor([[True], [False]]))
    assert single.tolist() == [[0.0], [0.0]]


def test_ranks_ignore_row_order_and_monotone_changes_of_a_column():
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(9, 4, generator=generator, dtype=torch.float64)
    present = torch.rand(9, 4, generator=generator) > 0.2
    result = gaussian_ranks(values, present)
    order = torch.randperm(9, generator=generator)
    torch.testing.assert_close(gaussian_ranks(values[order], present[order]), result[order])
    stretched = torch.exp(3 * values) + 7
    torch.testing.assert_close(gaussian_ranks(stretched, present), result, rtol=0, atol=0)


def test_session_inputs_rank_price_steps_and_fundamental_values_of_each_market():
    generator = torch.Generator().manual_seed(11)
    rows, markets = 7, ["US", "US", "CN", "US", "CN", "US", "CN"]
    prices = torch.randn(rows, 64, 6, generator=generator, dtype=torch.float64)
    prices[..., 5] = 1.0
    prices[1, 10:20, 5] = 0.0
    prices[1, 10:20, :5] = 0.0
    fundamentals = torch.randn(rows, 6, generator=generator, dtype=torch.float64)
    fundamentals[:, 2:4] = torch.tensor([[1.0, 0.0]]).repeat(rows, 1)
    fundamentals[:, 3][::2] = 1.0
    fundamentals[:, 1][fundamentals[:, 3] == 0] = 0.0
    fundamentals[:, 4:] = torch.rand(rows, 2, generator=generator, dtype=torch.float64)
    fundamentals[:, 5][fundamentals[:, 3] == 0] = 0.0
    others = {name: torch.randn(rows, 3, generator=generator) for name in ("news", "charts")}
    macro = torch.randn(1, 3, generator=generator).repeat(rows, 1)
    inputs = dict(prices=prices, fundamentals=fundamentals, macro=macro, **others)
    result = rank_session_inputs(inputs, markets)
    for market in ("US", "CN"):
        rows_of = [i for i, value in enumerate(markets) if value == market]
        selected = prices[rows_of]
        present = (selected[..., 5] > 0).unsqueeze(-1).expand(-1, -1, 5).reshape(len(rows_of), -1)
        expected = reference_scores(
            selected[..., :5].reshape(len(rows_of), -1).numpy(), present.numpy()
        )
        np.testing.assert_allclose(
            result["prices"][rows_of][..., :5].reshape(len(rows_of), -1).numpy(),
            expected,
            rtol=0,
            atol=1e-15,
        )
        data, masks = fundamentals[rows_of, :2].numpy(), fundamentals[rows_of, 2:4].numpy() > 0
        np.testing.assert_allclose(
            result["fundamentals"][rows_of, :2].numpy(),
            reference_scores(data, masks),
            rtol=0,
            atol=1e-15,
        )
    assert torch.equal(result["prices"][..., 5], prices[..., 5])
    assert torch.equal(result["prices"][1, 10:20], prices[1, 10:20])
    assert torch.equal(result["fundamentals"][:, 2:], fundamentals[:, 2:])
    for name in ("news", "charts", "macro"):
        assert result[name] is inputs[name]
    # Una fila de otro mercado no cambia los rangos del primero.
    alone = rank_session_inputs({k: v[[0, 1, 3, 5]] for k, v in inputs.items()}, ["US"] * 4)
    torch.testing.assert_close(alone["prices"], result["prices"][[0, 1, 3, 5]], rtol=0, atol=0)


def test_disabled_fields_keep_identity_parameters_and_predictions():
    base, batch = setup()
    explicit = model(session_rank_inputs=None, session_ic_objective=None)
    identity = base.config.identity()
    assert explicit.config.identity() == identity
    assert identity["output"] == "scalar_corpus_target"
    assert not {"session_rank_inputs", "output_objective"} & set(identity)
    assert base.output_objective is None and not base.session_level
    expected = base.prepare(batch, base.initial_state(batch.flow_ids))
    actual = explicit.prepare(batch, explicit.initial_state(batch.flow_ids))
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert actual.outputs is None and actual.input_digest == batch.input_digest


def test_variants_share_the_reference_parameters_and_add_only_the_scale():
    base, _ = setup()
    variant = model(session_rank_inputs=SESSION_RANK_INPUTS, session_ic_objective=OBJECTIVE)
    reference = dict(base.named_parameters())
    names = dict(variant.named_parameters())
    assert set(names) - set(reference) == {"output_objective.scale"}
    for name, value in reference.items():
        torch.testing.assert_close(names[name], value, rtol=0, atol=0)
    identity = variant.config.identity()
    assert identity["output"] == "session_ic_objective_v1"
    assert identity["output_objective"]["minimum_rows"] == 3
    assert identity["session_rank_inputs"]["name"] == SESSION_RANK_INPUTS


def test_session_operations_need_the_complete_session_in_one_block():
    batch = session_batch()
    for changes in (
        dict(session_rank_inputs=SESSION_RANK_INPUTS),
        dict(session_ic_objective=OBJECTIVE),
    ):
        predictor = model(**changes)
        assert predictor.session_level
        with pytest.raises(ValueError, match="sesión completa"):
            predictor.prepare(batch, predictor.initial_state(batch.flow_ids))
        predictor.prepare(batch, predictor.initial_state(batch.flow_ids), session_complete=True)


def test_ranked_predictor_equals_the_reference_on_inputs_ranked_by_hand():
    batch = session_batch()
    ranked = model(session_rank_inputs=SESSION_RANK_INPUTS)
    reference, _ = setup()
    by_hand = batch.derived(rank_session_inputs(batch.inputs, ["US"] * 6), "by_hand")
    expected = reference.prepare(by_hand, reference.initial_state(batch.flow_ids))
    actual = ranked.prepare(batch, ranked.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.next_state.mac.memory.weights[0],
        expected.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    # La huella devuelta sigue siendo la del lote recibido, no la del derivado.
    assert actual.input_digest == batch.input_digest != by_hand.input_digest


def test_scale_head_keeps_the_order_and_standardizes_each_market():
    markets = ["US", "CN", "US", "US", "CN", "US", "CN"]
    batch = session_batch(7, markets=markets)
    predictor = model(session_ic_objective=OBJECTIVE)
    prepared = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), session_complete=True
    )
    score, point = prepared.outputs.unbind(-1)
    torch.testing.assert_close(point, prepared.point_predictions, rtol=0, atol=0)
    raw = predictor.head(prepared.working_state).squeeze(-1)
    torch.testing.assert_close(score, raw, rtol=0, atol=0)
    for market in ("US", "CN"):
        rows = [i for i, value in enumerate(markets) if value == market]
        z = point[rows] / torch.nn.functional.softplus(predictor.output_objective.scale)
        centered = score[rows] - score[rows].mean()
        floor = RELATIVE_FLOOR * score[rows].square().mean() + torch.finfo(score.dtype).eps ** 2
        expected = centered / torch.sqrt(centered.square().mean() + floor)
        torch.testing.assert_close(z, expected, rtol=1e-12, atol=0)
        torch.testing.assert_close(z.mean(), torch.zeros((), dtype=z.dtype), rtol=0, atol=1e-12)
        assert torch.equal(score[rows].argsort(), point[rows].argsort())
    torch.testing.assert_close(
        torch.nn.functional.softplus(predictor.output_objective.scale),
        torch.tensor(1e-3, dtype=torch.float64),
    )
    assert standardized_scores(torch.tensor([2.0]), ["US"]).tolist() == [0.0]
    spread = standardized_scores(
        torch.tensor([1.0, 3.0, 2.0, 7.0], dtype=torch.float64), ["US"] * 4
    )
    torch.testing.assert_close(spread.square().mean(), torch.ones((), dtype=torch.float64))


def test_session_loss_matches_pearson_on_normal_scores_and_skips_small_sessions():
    generator = np.random.default_rng(4)
    sessions = [("US", 1)] * 6 + [("US", 2)] * 2 + [("CN", 1)] * 5
    score, target = generator.normal(size=13), generator.normal(0, 0.01, size=13)
    point = generator.normal(0, 0.001, size=13)
    outputs = torch.tensor(np.stack([score, point], axis=1))
    loss = session_ic_loss(outputs, torch.tensor(target), sessions, OBJECTIVE)
    correlations = []
    for key in (("US", 1), ("CN", 1)):
        rows = [i for i, value in enumerate(sessions) if value == key]
        ranks = stats.rankdata(target[rows], method="average")
        normal = special.ndtri((ranks - 0.5) / len(rows))
        correlations.append(stats.pearsonr(score[rows], normal).statistic)
    expected = -np.mean(correlations) + np.abs(target - point).mean()
    # El suelo relativo de 10⁻¹⁰ cambia la correlación en esa proporción.
    np.testing.assert_allclose(float(loss), expected, rtol=1e-8)
    alone = session_correlations(outputs[:, 0], torch.tensor(target), sessions, 3)
    np.testing.assert_allclose(alone.numpy(), correlations, rtol=1e-8)
    empty = session_ic_loss(outputs[6:8], torch.tensor(target[6:8]), sessions[6:8], OBJECTIVE)
    np.testing.assert_allclose(float(empty), np.abs(target[6:8] - point[6:8]).mean())


def test_session_loss_gradients_pass_gradcheck_and_match_the_pearson_derivative():
    generator = torch.Generator().manual_seed(8)
    sessions = [("US", 1)] * 5 + [("US", 2)] * 4
    target = torch.randn(9, generator=generator, dtype=torch.float64) / 100
    outputs = torch.randn(9, 2, generator=generator, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value: session_ic_loss(value, target, sessions, OBJECTIVE), (outputs,)
    )
    loss = session_ic_loss(outputs, target, sessions, OBJECTIVE)
    (gradient,) = torch.autograd.grad(loss, outputs)
    # Derivada de −ρ/|T| respecto a s_i: −(ẑ_g,i − ρ ẑ_s,i) / (|T| n σ_s).
    for rows in (list(range(5)), list(range(5, 9))):
        s = outputs[rows, 0].detach()
        labels = target[rows].unsqueeze(-1)
        g = gaussian_ranks(labels, torch.ones_like(labels, dtype=torch.bool)).squeeze(-1)
        sigma = s.std(correction=0)
        zs, zg = (s - s.mean()) / sigma, (g - g.mean()) / g.std(correction=0)
        rho = (zs * zg).mean()
        expected = -(zg - rho * zs) / (2 * len(rows) * sigma)
        torch.testing.assert_close(gradient[rows, 0], expected, rtol=1e-8, atol=1e-12)
    # El término MAE solo deriva respecto a la predicción: −w sign(y − ŷ) / N.
    expected = -torch.sign(target - outputs[:, 1].detach()) / 9
    torch.testing.assert_close(gradient[:, 1], expected, rtol=0, atol=1e-15)


def test_ranking_reaches_the_trunk_and_the_absolute_error_only_the_scale():
    batch = session_batch(8)
    predictor = model(session_rank_inputs=SESSION_RANK_INPUTS, session_ic_objective=OBJECTIVE)
    prepared = predictor.prepare(
        batch, predictor.initial_state(batch.flow_ids), differentiable=True, session_complete=True
    )
    target = torch.linspace(-0.02, 0.03, 8, dtype=torch.float64)
    sessions = [("US", AT)] * 8
    names, values = zip(*predictor.named_parameters(), strict=True)
    ranking = -session_correlations(prepared.outputs[:, 0], target, sessions, 3).mean()
    absolute = (target - prepared.outputs[:, 1]).abs().mean()
    first = torch.autograd.grad(ranking, values, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(absolute, values, allow_unused=True)
    active = {
        name
        for name, grad in zip(names, first, strict=True)
        if grad is not None and grad.abs().sum() > 0
    }
    assert {"head.weight", "fusion.0.weight"} <= active
    assert "output_objective.scale" not in active
    reached = {
        name
        for name, grad in zip(names, second, strict=True)
        if grad is not None and grad.abs().sum() > 0
    }
    assert reached == {"output_objective.scale"}


def test_scale_round_trips_through_the_state_dict_and_reproduces_predictions():
    batch = session_batch()
    source = model(session_ic_objective=OBJECTIVE)
    with torch.no_grad():
        source.output_objective.scale.fill_(-2.5)
    source._seal_parameters()
    target = model(session_ic_objective=OBJECTIVE)
    target.load_state_dict(source.state_dict())
    expected = source.prepare(batch, source.initial_state(batch.flow_ids), session_complete=True)
    actual = target.prepare(batch, target.initial_state(batch.flow_ids), session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    other = model(session_ic_objective=replace(OBJECTIVE, minimum_rows=4))
    with pytest.raises(ValueError):
        other.load_state_dict(source.state_dict())


def test_predictor_change_registry_admits_both_identities():
    changes = dict(session_rank_inputs=SESSION_RANK_INPUTS, session_ic_objective=asdict(OBJECTIVE))
    assert checked_predictor_changes(changes) == changes
    base, _ = setup()
    config = replace(base.config, **changes)
    assert config.session_ic_objective == OBJECTIVE


@pytest.mark.parametrize(
    "changes",
    [
        dict(session_rank_inputs="session_rank_inputs_v2"),
        dict(session_ic_objective=dict(minimum_rows=8, scale_weight=1.0)),
        dict(session_ic_objective=dict(minimum_rows=2, scale_weight=1.0, initial_scale=1e-3)),
        dict(session_ic_objective=dict(minimum_rows=8, scale_weight=0.0, initial_scale=1e-3)),
        dict(session_ic_objective=dict(minimum_rows=8, scale_weight=1.0, initial_scale=True)),
        dict(session_ic_objective=OBJECTIVE, head="quantile_head_v1"),
    ],
)
def test_invalid_declarations_are_rejected(changes):
    base, _ = setup()
    with pytest.raises(ValueError):
        replace(base.config, **changes)


def test_ranks_need_the_masked_policy():
    strict, _ = setup(historical=False)
    with pytest.raises(ValueError, match="máscaras"):
        replace(strict.config, session_rank_inputs=SESSION_RANK_INPUTS)


def test_declared_arms_build_with_their_control_and_stay_disabled_by_default():
    root = Path(__file__).resolve().parents[3]
    declaration = json.loads(
        (root / "configs/experiments/session-rank-objective-v1.json").read_text()
    )
    assert declaration["default_enabled"] is False
    assert declaration["final_test_access"] is False
    base, _ = setup()
    reference = dict(base.named_parameters())
    identities = set()
    for arm in declaration["arms"].values():
        changes = arm["predictor_changes"]
        assert checked_predictor_changes(changes or None) == (changes or None)
        assert arm["role"] == "control" or declaration["arms"][arm["control"]]["role"] == "control"
        predictor = model(**changes)
        identities.add(json.dumps(predictor.config.identity(), sort_keys=True))
        for key, value in reference.items():
            torch.testing.assert_close(
                dict(predictor.named_parameters())[key], value, rtol=0, atol=0
            )
    assert len(identities) == len(declaration["arms"])
