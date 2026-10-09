"""K con los episodios de la primera lectura frente a la selección en cada paso.

El modo `first_read` separa más cálculo de más evidencia: los pasos siguientes atienden los
mismos episodios con la consulta del estado refinado, sin volver a seleccionar.
"""

import math

import pytest
import torch
from test_episodic_readout import CODEC, CONTEXT, analytic, api, snapshot


def reader(**options):
    return api().EpisodicReadout(
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, **options), dtype=torch.float64
    )


def random_reader(seed, dtype=torch.float64, **options):
    model = api().EpisodicReadout(
        api().EpisodicReadoutConfig(CODEC, hidden_size=32, seed=seed, **options), dtype=dtype
    )
    return model


def random_snapshot(size, dtype, seed=3):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn((size, 64), generator=generator, dtype=torch.float64)
    keys = (keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)).float()
    # Las claves FP32 deben conservar la norma unitaria que exige la instantánea.
    keys = keys / torch.linalg.vector_norm(keys.double(), dim=1, keepdim=True).float()
    return api().EpisodeSnapshot.create(
        keys=keys.contiguous(),
        values=torch.randn((size, 64), generator=generator).float(),
        labels=torch.randn(size, generator=generator, dtype=torch.float64) / 10,
        ids=torch.arange(1, size + 1, dtype=torch.int64) * 7,
        decision_at=torch.ones(size, dtype=torch.int64),
        available_at=torch.ones(size, dtype=torch.int64),
        maturity_at=torch.full((size,), 2, dtype=torch.int64),
        cutoff=10,
        codec_id=CODEC,
        context_id=CONTEXT,
        device="cpu",
        dtype=dtype,
    )


def reselecting(model):
    """Parámetros con los que el segundo paso elegiría otro episodio si volviera a buscar."""
    analytic(model)
    with torch.no_grad():
        model.refinement.weight.zero_()
        model.refinement.bias[1] = 1
        model.step_logit.fill_(math.log(0.9 / 0.1))


def start():
    z = torch.zeros((1, 32), dtype=torch.float64)
    z[0, 0] = 0.5
    return z


def test_previous_selection_keeps_its_identity_and_the_new_mode_has_its_own():
    previous = api().EpisodicReadoutConfig(CODEC, hidden_size=32)
    fixed = api().EpisodicReadoutConfig(CODEC, hidden_size=32, episode_selection="first_read")
    identity = previous.identity()
    assert "episode_selection" not in identity
    assert identity["selection"] == "global_each_step_fp64_stable_low_id"
    assert fixed.identity()["episode_selection"] == "first_read"
    assert fixed.identity()["selection"] != identity["selection"]
    assert reader().get_extra_state() != reader(episode_selection="first_read").get_extra_state()


@pytest.mark.parametrize("value", ["each", "", None, 1])
def test_unknown_selection_modes_are_rejected(value):
    with pytest.raises(ValueError, match="per_step o first_read"):
        api().EpisodicReadoutConfig(CODEC, episode_selection=value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("differentiable", [False, True])
def test_one_refinement_gives_the_same_bits_in_both_modes(dtype, differentiable):
    data = random_snapshot(12, dtype)
    per_step = random_reader(5, dtype, neighbors=4)
    fixed = random_reader(5, dtype, neighbors=4, episode_selection="first_read")
    z = torch.randn((6, 32), generator=torch.Generator().manual_seed(9), dtype=dtype)
    options = dict(context_id=CONTEXT, cutoff=10, differentiable=differentiable)
    left, right = per_step(z, data, **options), fixed(z, data, **options)
    assert torch.equal(left.state, right.state)
    assert len(left.reads) == len(right.reads) == 1
    for name in ("values", "weights", "ids", "presence"):
        assert torch.equal(getattr(left.reads[0], name), getattr(right.reads[0], name))


@pytest.mark.parametrize("steps", [2, 4])
def test_later_steps_never_read_episodes_outside_the_first_selection(steps):
    per_step = reader(neighbors=1, refinements=steps)
    fixed = reader(neighbors=1, refinements=steps, episode_selection="first_read")
    reselecting(per_step)
    reselecting(fixed)
    options = dict(context_id=CONTEXT, cutoff=10)
    moved = per_step(start(), snapshot(3), **options)
    kept = fixed(start(), snapshot(3), **options)
    # El control demuestra que una nueva búsqueda cambiaría el episodio elegido.
    assert [read.ids.item() for read in moved.reads] == [10] + [20] * (steps - 1)
    assert [read.ids.item() for read in kept.reads] == [10] * steps
    assert {read.ids.item() for read in kept.reads} == {kept.reads[0].ids.item()}


def test_fixed_positions_recompute_softmax_weights_with_the_refined_query():
    model = reader(neighbors=2, refinements=2, episode_selection="first_read")
    reselecting(model)
    data = snapshot(3)
    result = model(start(), data, context_id=CONTEXT, cutoff=10)
    first, second = result.reads
    assert first.ids.tolist() == second.ids.tolist() == [[10, 20]]
    # Segunda lectura: consulta normalizada del estado tras el primer paso.
    state = model.refine(start(), start(), first)
    query = model.query_projection(state)
    query = query / torch.linalg.vector_norm(query, dim=-1, keepdim=True)
    keys = data._values[0][:2].to(torch.float64)
    expected = (keys @ query[0]).softmax(-1)
    torch.testing.assert_close(second.weights[0], expected, rtol=0, atol=1e-15)
    assert not torch.equal(first.weights, second.weights)


@pytest.mark.parametrize("steps", [2, 4])
def test_fixed_selection_is_differentiable_with_finite_difference_gradients(steps):
    model = random_reader(11, neighbors=3, refinements=steps, episode_selection="first_read")
    data = random_snapshot(9, torch.float64)
    z = torch.randn((2, 32), generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    z.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: model(x, data, context_id=CONTEXT, cutoff=10, differentiable=True).state,
        (z,),
        eps=1e-6,
        atol=1e-6,
        rtol=1e-5,
    )

    def through_query(value):
        # Sustituir funcionalmente la matriz de consulta conserva el resto del módulo.
        return torch.func.functional_call(
            model,
            {"query_projection.weight": value},
            (z.detach(), data),
            dict(context_id=CONTEXT, cutoff=10, differentiable=True),
        ).state

    probe = model.query_projection.weight.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(through_query, (probe,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_positions_must_match_the_batch_and_neighbors():
    model = reader(neighbors=2)
    data = snapshot(3)
    state = start()
    with pytest.raises(ValueError, match="posiciones fijadas"):
        model._read(state, data, CONTEXT, 10, torch.zeros((1, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match="posiciones fijadas"):
        model._read(state, data, CONTEXT, 10, torch.zeros((1, 2), dtype=torch.int32))


def test_empty_memory_and_no_bank_control_stay_empty_in_every_fixed_step():
    model = reader(neighbors=2, refinements=4, episode_selection="first_read")
    empty = model(start(), snapshot(0), context_id=CONTEXT, cutoff=10)
    assert all(read.ids.shape == (1, 0) and not read.presence.any() for read in empty.reads)
    control = reader(neighbors=2, refinements=4, episode_selection="first_read", mode="no_bank")
    absent = control(start())
    assert all(not read.presence.any() for read in absent.reads)


def test_parameters_copy_between_modes_and_keep_one_step_parity():
    source = random_reader(21, neighbors=4)
    target = random_reader(22, neighbors=4, episode_selection="first_read")
    receipt = api().copy_readout_parameters(source, target)
    assert receipt["snapshot_transferred"] is False
    data = random_snapshot(10, torch.float64)
    z = torch.randn((3, 32), generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    options = dict(context_id=CONTEXT, cutoff=10)
    assert torch.equal(source(z, data, **options).state, target(z, data, **options).state)
    with pytest.raises(ValueError):
        api().copy_readout_parameters(source, random_reader(22, neighbors=4, refinements=2))
