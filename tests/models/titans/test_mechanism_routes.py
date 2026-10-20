"""Conservar K1 para un retorno posterior sin repetir su lectura ni avanzar MAC."""

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT
from test_refinement_routes import fixture, reference

from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.refinement_routes import ROUTES, run_route_pair

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("route", ROUTES)
def test_pair_keeps_k1_and_only_executes_the_requested_path(route, device, monkeypatch):
    reader, snapshot, base = fixture()
    reader, base = reader.to(device), base.to(device)
    snapshot = EpisodeSnapshot.restore(
        snapshot.export_cpu(),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=10,
        dtype=torch.float64,
        device=device,
    )
    first_expected, _ = reference(reader, snapshot, base, "k1")
    final_expected, _ = reference(reader, snapshot, base, route)
    calls = []
    original = reader._step

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(reader, "_step", counted)
    first, final = run_route_pair(
        reader, base, snapshot, route=route, context_id=CONTEXT, cutoff=10
    )
    assert len(calls) == int(route[1])
    assert torch.equal(first.state, first_expected)
    assert torch.equal(final.state, final_expected)
    assert first.refinements == 1 and final.refinements == int(route[1])
    assert first.searches == 1 and final.searches == 1 + int("r" in route)
    assert not final.state.requires_grad


def test_both_outputs_keep_their_gradients_when_requested():
    reader, snapshot, base = fixture()
    base.requires_grad_()
    first, final = run_route_pair(
        reader,
        base,
        snapshot,
        route="k4rc",
        context_id=CONTEXT,
        cutoff=10,
        differentiable=True,
    )
    actual = torch.autograd.grad(first.state.square().sum() + final.state.square().sum(), base)[0]
    initial, _ = reference(reader, snapshot, base, "k1")
    last, _ = reference(reader, snapshot, base, "k4rc")
    expected = torch.autograd.grad(initial.square().sum() + last.square().sum(), base)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
