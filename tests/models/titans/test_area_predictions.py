"""Predicciones por área de H1 (#783): votos, lectura aprendida y su paso por el predictor.

Las pruebas calculan salidas y gradientes sobre lotes ficticios, sin optimizador ni pasos.
"""

import math
from dataclasses import asdict, replace

import pytest
import torch
from test_holographic_network import H1, H3, module_inputs, network, prepared, reference, setup
from torch.nn import functional as F

from mars_titan.models.titans.holographic_network import parse_holographic_network

VOTING = replace(H1, area_predictions=0.5)
VOTE_PARAMETERS = (
    "area_vote_norm.weight",
    "area_vote_norm.bias",
    "area_vote_weight",
    "area_vote_bias",
    "area_vote_logits",
)
AREA_KEYS = {"area_predictions", "area_votes", "area_readout", "area_output", "area_loss"}


def moved_votes(module, seed=13):
    """Llevar lejos de cero los votos y la lectura de un módulo H1 con áreas que votan."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in (module.area_vote_weight, module.area_vote_bias, module.area_vote_logits):
            value = torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
            parameter.copy_(value.to(parameter.dtype))


def test_the_field_is_optional_and_only_a_declared_weight_enters_the_identity():
    fields = asdict(H1)
    assert fields.pop("area_predictions") is None
    assert parse_holographic_network(fields) == H1
    assert parse_holographic_network(asdict(VOTING)) == VOTING
    assert not AREA_KEYS & set(H1.identity())
    identity = VOTING.identity()
    assert AREA_KEYS <= set(identity) and identity["area_predictions"] == 0.5
    assert replace(H1, area_predictions=0.0).identity() != identity
    for weight in (0.0, 4.0):
        assert replace(H1, area_predictions=weight).area_predictions == weight


@pytest.mark.parametrize("weight", [-0.1, 4.5, 1, math.nan, math.inf, True, "0.5"])
def test_invalid_weights_are_refused(weight):
    with pytest.raises(ValueError, match="decimal entre 0 y 4"):
        replace(H1, area_predictions=weight)


def test_hidden_groups_alone_have_no_areas_to_vote():
    with pytest.raises(ValueError, match="H3 no tiene áreas"):
        replace(H3, area_predictions=0.5)


def test_the_quantile_head_and_the_output_objectives_are_refused():
    from mars_titan.models.titans.session_ic_objective import SessionICObjective

    for changes in (
        dict(head="quantile_head_v1"),
        dict(session_ic_objective=SessionICObjective()),
    ):
        with pytest.raises(ValueError, match="cabeza escalar sin objetivo"):
            setup(holographic_network=VOTING, **changes)
        # Sin votos, H1 sigue admitiendo la misma declaración.
        setup(holographic_network=H1, **changes)


def test_votes_need_the_declared_field():
    with pytest.raises(ValueError, match="no declara predicciones por área"):
        network(H1).forward_with_votes(*module_inputs())


def test_the_votes_follow_their_equation_row_by_row():
    net = network(VOTING)
    moved_votes(net)
    blocks, presence, inputs, encoders = module_inputs()
    result, votes = net.forward_with_votes(blocks, presence, inputs, encoders)
    torch.testing.assert_close(result, net(blocks, presence, inputs, encoders), rtol=0, atol=0)
    # Los estados finales y la presencia salen de la referencia escrita área por área.
    final = []
    reference(net, blocks, presence, inputs, encoders, final=final)
    alive = torch.tensor([flags for _, flags in final])
    assert torch.equal(votes.alive, alive) and not alive.all() and alive.any(1).all()
    norm = net.area_vote_norm
    for r, (states, _) in enumerate(final):
        present = alive[r].nonzero().flatten().tolist()
        weights = torch.softmax(net.area_vote_logits[present], dim=0)
        for a in range(net.config.areas):
            if a not in present:
                assert votes.votes[r, a] == 0 and votes.weights[r, a] == 0
                continue
            state = F.layer_norm(states[a], (net.hidden_size,), norm.weight, norm.bias, norm.eps)
            expected = net.area_vote_weight[a] @ state + net.area_vote_bias[a]
            torch.testing.assert_close(votes.votes[r, a], expected)
            torch.testing.assert_close(votes.weights[r, a], weights[present.index(a)])
        torch.testing.assert_close(votes.shift[r], (votes.weights[r] * votes.votes[r]).sum())


def test_a_row_without_present_areas_keeps_its_output_and_a_finite_gradient():
    net = network(VOTING)
    moved_votes(net)
    blocks, presence, inputs, encoders = module_inputs()
    _, states, alive = net._run(blocks, presence, inputs, encoders)
    alive = alive.clone()
    alive[0] = False
    votes = net.area_votes(states, alive)
    assert votes.shift[0] == 0 and not votes.weights[0].any() and not votes.votes[0].any()
    votes.shift.sum().backward()
    for name in ("area_vote_logits", "area_vote_weight", "area_vote_bias"):
        assert torch.isfinite(getattr(net, name).grad).all(), name


def test_the_operation_counts_add_the_votes_only_when_declared():
    rows = 7
    plain, voting = network(H1).operation_counts(rows), network(VOTING).operation_counts(rows)
    assert "holographic_area_votes" not in plain
    assert voting == dict(plain, holographic_area_votes=rows * VOTING.areas)


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_the_votes_start_at_zero_and_the_predictor_emits_as_h1(weight):
    base, batch = setup(holographic_network=H1)
    voting, _ = setup(holographic_network=replace(H1, area_predictions=weight))
    names = dict(voting.named_parameters())
    reference = dict(base.named_parameters())
    assert set(names) - set(reference) == {f"holographic_network.{n}" for n in VOTE_PARAMETERS}
    for key, value in reference.items():
        assert torch.equal(names[key], value), key
    expected, actual = prepared(base, batch), prepared(voting, batch)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert expected.area_predictions is None and expected.area_votes is None
    alive = actual.area_votes.alive
    assert alive.shape == (len(batch.flow_ids), H1.areas) and alive.any(1).all()
    # Al construirse, la lectura reparte por igual entre las áreas presentes de cada fila.
    uniform = alive.to(actual.area_votes.weights.dtype) / alive.sum(1, keepdim=True)
    torch.testing.assert_close(actual.area_votes.weights, uniform, rtol=0, atol=1e-15)
    torch.testing.assert_close(
        actual.area_predictions,
        expected.point_predictions.unsqueeze(-1).expand_as(actual.area_predictions),
        rtol=0,
        atol=0,
    )


def test_the_output_is_the_reference_plus_the_weighted_votes():
    base, batch = setup(holographic_network=H1)
    voting, _ = setup(holographic_network=VOTING)
    moved_votes(voting.holographic_network)
    voting._seal_parameters()
    expected, actual = prepared(base, batch), prepared(voting, batch)
    votes = actual.area_votes
    assert votes.votes.abs().sum() > 0
    shift = (votes.weights * votes.votes).sum(1)
    torch.testing.assert_close(
        actual.point_predictions, expected.point_predictions + shift, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.area_predictions,
        expected.point_predictions.unsqueeze(-1) + votes.votes,
        rtol=0,
        atol=0,
    )


def test_the_area_loss_reaches_the_areas_and_the_encoders_but_not_the_output_path():
    for moved in (False, True):
        model, batch = setup(holographic_network=VOTING)
        if moved:
            moved_votes(model.holographic_network)
            model._seal_parameters()
        result = model.prepare(batch, model.initial_state(batch.flow_ids), differentiable=True)
        (result.area_predictions.abs() * result.area_votes.alive).sum().backward()
        names = {name for name, _ in model.named_parameters()}
        reached = {
            name
            for name, value in model.named_parameters()
            if value.grad is not None and bool(value.grad.abs().sum() > 0)
        }
        # La referencia entra sin grafo y la lectura de los votos no cambia la predicción de
        # cada área, así que la cabeza, la fusión, Titans-MAC, W_o y la lectura quedan fuera.
        assert not {name for name in reached if name.startswith(("head.", "fusion.", "mac."))}
        assert not reached & {
            "holographic_network.readout.weight",
            "holographic_network.area_vote_logits",
        }
        assert "holographic_network.area_vote_weight" in reached
        # Con los votos en cero, la pérdida de las áreas aún no llega al puente ni a los
        # codificadores de los bloques.
        upstream = {"holographic_network.qkv.weight"}
        upstream |= {name for name in names if name.startswith(("encoders.", "price_encoder."))}
        assert (upstream <= reached) if moved else not upstream & reached
