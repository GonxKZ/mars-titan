"""Variantes C1 y C6 del operador regional, sin ajustes ni datos financieros reales."""

import math
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.models.titans import regional_variants as rv
from mars_titan.models.titans.regional_fusion import RegionalFusion, RegionalFusionConfig

PRESENCE = torch.tensor(
    [
        [True, True, True, True, True],
        [True, False, True, False, True],
        [False, False, False, False, False],
        [True, True, False, True, True],
    ]
)


def operator(**options):
    return RegionalFusionConfig(**({"topology": "ring"} | options))


def mixing(mode="talking_heads", hidden=10, seed=3, **options):
    config = rv.RegionAxisMixingConfig(operator(**options), mixing=mode)
    return rv.RegionAxisMixing(hidden, config, seed=seed).double()


def workspace(write="competition", hidden=10, seed=3, slots=2, top_k=2, **options):
    config = rv.SharedWorkspaceConfig(
        operator(**options),
        slots=slots,
        write=write,
        write_top_k=top_k if write == "competition" else None,
    )
    return rv.SharedWorkspace(hidden, config, seed=seed).double()


def blocks(rows=4, hidden=10, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, 5, hidden, generator=generator, dtype=torch.float64)


class Skeleton(rv._RegionalVariant):
    """El operador original reconstruido sobre el calendario común de las variantes."""

    kind = "test_regional_skeleton"

    def _build(self, hidden_size):
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.local_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.summary_to_local = nn.Linear(hidden_size, hidden_size, bias=False)
        self.summary_norm = nn.LayerNorm(hidden_size)
        self.summary_update = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=False),
        )

    def _messages(self, normalized, allowed):
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        return F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=allowed.unsqueeze(1)
        ).squeeze(1)

    _initial_summary = rv.RegionAxisMixing._initial_summary
    _summary = rv.RegionAxisMixing._summary


class SkeletonConfig:
    def __init__(self, config):
        self.operator = config

    def __post_init__(self):
        self.operator.__post_init__()

    def identity(self):
        return asdict(self.operator)


@pytest.mark.parametrize("topology", ["complete", "ring", "self"])
@pytest.mark.parametrize("summary", [True, False])
@pytest.mark.parametrize("schedule", [(1, 1), (2, 2), (3, 1)])
def test_the_common_schedule_reproduces_the_original_operator_bit_for_bit(
    topology, summary, schedule
):
    cycles, steps = schedule
    config = RegionalFusionConfig(
        topology=topology, cycles=cycles, local_steps=steps, summary=summary
    )
    original = RegionalFusion(10, config, seed=5).double()
    rebuilt = Skeleton(10, SkeletonConfig(config), seed=5).double()
    rebuilt.load_state_dict(
        {name: value for name, value in original.state_dict().items() if name != "_extra_state"}
        | {"_extra_state": rebuilt.get_extra_state()}
    )
    x = blocks()
    torch.testing.assert_close(rebuilt(x, PRESENCE), original(x, PRESENCE), rtol=0, atol=0)
    torch.testing.assert_close(rebuilt(x, None), original(x, None), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mixing_starts_equal_to_its_control_and_only_the_mixing_separates_them(dtype):
    variant, control = mixing().to(dtype), mixing("none").to(dtype)
    shared = {name: value for name, value in variant.named_parameters() if "mix" not in name}
    assert shared.keys() == dict(control.named_parameters()).keys()
    for name, value in control.named_parameters():
        assert torch.equal(value, shared[name])
    assert {name for name, _ in variant.named_parameters()} - shared.keys() == {
        "mix_logits",
        "mix_weights",
    }
    x = blocks().to(dtype)
    assert torch.equal(variant(x, PRESENCE), control(x, PRESENCE))
    with torch.no_grad():
        variant.mix_logits.copy_(torch.full((5, 5), 0.2, dtype=dtype))
    assert not torch.equal(variant(x, PRESENCE), control(x, PRESENCE))


def test_written_and_dense_workspaces_start_from_the_same_parameters():
    competition, dense = workspace(), workspace("dense")
    assert competition.state_dict().keys() == dense.state_dict().keys()
    for name, value in competition.named_parameters():
        assert torch.equal(value, dict(dense.named_parameters())[name])
    x = blocks()
    assert not torch.equal(competition(x, PRESENCE), dense(x, PRESENCE))


@pytest.mark.parametrize("build", [mixing, lambda: mixing("none"), workspace])
def test_missing_regions_send_nothing_and_keep_their_blocks(build):
    model = build()
    original = blocks()
    changed = torch.where(PRESENCE[..., None], original, torch.full_like(original, 1e308))
    left, right = model(original, PRESENCE), model(changed, PRESENCE)
    torch.testing.assert_close(left[PRESENCE], right[PRESENCE], rtol=0, atol=0)
    torch.testing.assert_close(right[~PRESENCE], changed[~PRESENCE], rtol=0, atol=0)
    assert torch.isfinite(right).all()


@pytest.mark.parametrize("build", [mixing, workspace])
def test_rows_are_independent_and_the_working_state_resets(build):
    model = build()
    x = blocks()
    expected = model(x, PRESENCE)
    model(x * -3, PRESENCE)
    torch.testing.assert_close(model(x, PRESENCE), expected, rtol=0, atol=0)
    other = x.clone()
    other[3] += 5
    torch.testing.assert_close(model(other, PRESENCE)[:3], expected[:3], rtol=0, atol=0)
    order = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(model(x[order], PRESENCE[order]), expected[order], rtol=0, atol=0)


def test_mixed_weights_keep_each_head_inside_its_edges():
    """Con aristas propias y sin resumen, un nodo no recibe nada de otro aunque se mezcle."""
    model = mixing(topology="self", summary=False, cycles=2, local_steps=2)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(2)
        model.mix_logits.add_(torch.randn(5, 5, generator=generator, dtype=torch.float64))
        model.mix_weights.add_(torch.randn(5, 5, generator=generator, dtype=torch.float64))
    x = blocks()
    present = torch.ones(4, 5, dtype=torch.bool)
    changed = x.clone()
    changed[:, 1:] *= 7
    torch.testing.assert_close(model(x, present)[:, 0], model(changed, present)[:, 0])


def test_competition_writes_only_with_the_best_regions_of_each_slot():
    model = workspace(slots=1, top_k=1)
    local = blocks(rows=2)
    present = torch.tensor([[True, True, False, True, True], [False] * 5])
    active = present.any(1, keepdim=True)
    slots = model._initial_summary(local)
    weights = model._write_weights(slots, local, present, active)
    scores = model.write_query(model.slot_norm(slots)) @ model.write_key(local).transpose(1, 2)
    best = scores[0, 0].masked_fill(~present[0], -math.inf).argmax()
    expected = torch.zeros(5, dtype=torch.float64)
    expected[best] = 1
    torch.testing.assert_close(weights[0, 0], expected, rtol=0, atol=0)
    assert torch.isfinite(weights[1]).all()
    dense = workspace("dense", slots=1)
    weights = dense._write_weights(slots, local, present, active)
    assert bool((weights[0, 0] > 0).eq(present[0]).all())
    # Una fila sin regiones conserva sus ranuras y su salida es la entrada.
    torch.testing.assert_close(model(local, present)[1], local[1], rtol=0, atol=0)


def test_competition_is_dense_writing_when_k_regions_or_fewer_are_present():
    few = torch.tensor([[True, False, True, False, False], [False, False, False, True, False]])
    x = blocks(rows=2)
    torch.testing.assert_close(
        workspace(top_k=2)(x, few), workspace("dense")(x, few), rtol=0, atol=0
    )


@pytest.mark.parametrize("build", [mixing, workspace])
def test_executed_recurrence_gradients_match_finite_differences(build):
    model = build(hidden=5, cycles=2, local_steps=1)
    x = blocks(rows=2, hidden=5)
    present = torch.tensor([[True, True, False, True, True], [True] * 5])
    name = "mix_logits" if hasattr(model, "mix_logits") else "write_query.weight"
    weight = dict(model.named_parameters())[name].detach().clone()
    weight = (weight + 0.1 * torch.randn_like(weight)).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value, parameter: torch.func.functional_call(
            model, {name: parameter}, (value, present)
        ),
        (x.requires_grad_(), weight),
        fast_mode=True,
    )


def test_construction_and_calls_leave_the_global_generator_untouched():
    before = torch.get_rng_state().clone()
    for model in (mixing(), workspace(), workspace("dense")):
        model(blocks(), PRESENCE)
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoints_restore_the_same_variant_and_reject_another_contract():
    source, target = mixing(seed=1), mixing(seed=2)
    target.load_state_dict(source.state_dict())
    x = blocks()
    torch.testing.assert_close(target(x, PRESENCE), source(x, PRESENCE), rtol=0, atol=0)
    for other in (mixing("none"), mixing(topology="complete"), workspace()):
        with pytest.raises((RuntimeError, ValueError)):
            other.load_state_dict(source.state_dict())
    damaged = source.state_dict()
    damaged["edges"] = ~damaged["edges"]
    with pytest.raises(ValueError, match="aristas"):
        target.load_state_dict(damaged)


@pytest.mark.parametrize(
    "build",
    [
        lambda: rv.RegionAxisMixingConfig(operator(routing="uniform")),
        lambda: rv.RegionAxisMixingConfig(operator(), mixing="talking"),
        lambda: rv.SharedWorkspaceConfig(operator(summary=False)),
        lambda: rv.SharedWorkspaceConfig(operator(), slots=0),
        lambda: rv.SharedWorkspaceConfig(operator(), slots=5),
        lambda: rv.SharedWorkspaceConfig(operator(), write_top_k=0),
        lambda: rv.SharedWorkspaceConfig(operator(), write_top_k=5),
        lambda: rv.SharedWorkspaceConfig(operator(), write="dense"),
        lambda: rv.SharedWorkspaceConfig(operator(), write="sparse"),
        lambda: rv.parse_regional_variant(dict(operator=asdict(operator()))),
        lambda: rv.parse_regional_variant(dict(operator=dict(topology="ring"), mixing="none")),
        lambda: rv.RegionAxisMixing(4, rv.RegionAxisMixingConfig(operator()), seed=1),
    ],
)
def test_incomplete_or_invalid_declarations_are_rejected(build):
    with pytest.raises(ValueError):
        build()


def test_declarations_travel_as_json_and_name_their_control():
    for config in (
        rv.RegionAxisMixingConfig(operator()),
        rv.RegionAxisMixingConfig(operator(), mixing="none"),
        rv.SharedWorkspaceConfig(operator()),
        rv.SharedWorkspaceConfig(operator(), write="dense", write_top_k=None),
    ):
        assert rv.parse_regional_variant(asdict(config)) == config
    identities = {
        str(rv.RegionAxisMixingConfig(operator(), mixing=mode).identity())
        for mode in ("talking_heads", "none")
    }
    assert len(identities) == 2


@pytest.mark.parametrize(
    "bad",
    [
        lambda x, p: (x.float(), p),
        lambda x, p: (x[:, :4], p[:, :4]),
        lambda x, p: (x, p.int()),
        lambda x, p: (x.repeat(80, 1, 1), p.repeat(80, 1)),
    ],
)
def test_inputs_outside_the_contract_are_rejected(bad):
    model = rv.RegionAxisMixing(10, rv.RegionAxisMixingConfig(operator()), seed=3, max_batch=256)
    model = model.double()
    with pytest.raises(ValueError):
        model(*bad(blocks(), PRESENCE))


def test_autocast_does_not_relax_the_declared_precision():
    model = mixing().float()
    with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError, match="autocast"):
        model(blocks().float(), PRESENCE)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("build", [mixing, workspace])
def test_cuda_matches_cpu(build, dtype):
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build().to(dtype)
    x = blocks().to(dtype)
    expected = model(x, PRESENCE)
    result = model.to("cuda:0")(x.to("cuda:0"), PRESENCE.to("cuda:0")).cpu()
    tolerance = (
        dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    )
    torch.testing.assert_close(result, expected, **tolerance)
