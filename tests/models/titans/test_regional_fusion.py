"""Grafo de evidencia y ritmos de trabajo, sin ajustes ni datos financieros reales."""

import importlib
import math

import pytest
import torch


def graph(*, hidden=2, seed=37, **options):
    api = importlib.import_module("mars_titan.models.titans.regional_fusion")
    return api.RegionalFusion(hidden, api.RegionalFusionConfig(**options), seed=seed).double()


def test_uniform_one_step_matches_the_hand_computed_masked_mean():
    model = graph(cycles=1, local_steps=1, summary=False, routing="uniform")
    with torch.no_grad():
        model.qkv.weight.copy_(torch.eye(2).repeat(3, 1))
        model.local_out.weight.copy_(torch.eye(2))
        model.readout.weight.copy_(torch.eye(2))
    blocks = torch.tensor([[[1.0, -1.0], [0.0, 0.0], [99.0, 99.0], [7.0, 7.0], [8.0, 8.0]]])
    present = torch.tensor([[True, True, False, False, False]])
    amount = 0.125 / math.sqrt(1.0 + 1e-5)
    expected = blocks.double().clone()
    expected[:, :2] += torch.tensor([amount, -amount], dtype=torch.float64)
    torch.testing.assert_close(model(blocks.double(), present), expected, rtol=1e-14, atol=1e-14)


def test_missing_nodes_cannot_send_evidence_and_do_not_acquire_new_content():
    model = graph(cycles=2, local_steps=2)
    original = torch.arange(20, dtype=torch.float64).reshape(2, 5, 2) / 10
    present = torch.tensor([[True, False, True, True, False], [False] * 5])
    changed = torch.where(present[..., None], original, torch.full_like(original, 1e308))
    left, right = model(original, present), model(changed, present)
    torch.testing.assert_close(left[present], right[present], rtol=0, atol=0)
    torch.testing.assert_close(right[~present], changed[~present], rtol=0, atol=0)
    assert torch.isfinite(right).all()


def test_state_is_local_to_the_call_and_asset_order_is_preserved():
    model = graph(cycles=2, local_steps=2)
    inputs = torch.arange(30, dtype=torch.float64).reshape(3, 5, 2) / 7
    present = torch.ones(3, 5, dtype=torch.bool)
    expected = model(inputs, present)
    model(inputs * -3, present)
    torch.testing.assert_close(model(inputs, present), expected, rtol=0, atol=0)
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        model(inputs[order], present[order]), expected[order], rtol=0, atol=0
    )
    unrelated = inputs.clone()
    unrelated[2, :, 0] += 10
    torch.testing.assert_close(model(unrelated, present)[:2], expected[:2], rtol=0, atol=0)


def test_strict_fusion_without_a_mask_means_all_five_sources_are_present():
    model = graph()
    blocks = torch.arange(10, dtype=torch.float64).reshape(1, 5, 2) / 7
    torch.testing.assert_close(
        model(blocks, None),
        model(blocks, torch.ones(1, 5, dtype=torch.bool)),
        rtol=0,
        atol=0,
    )


def test_summary_connects_regions_only_when_enabled():
    blocks = torch.arange(20, dtype=torch.float64).reshape(1, 5, 4) / 3
    changed = blocks.clone()
    changed[:, 0, 0] += 3
    present = torch.ones(1, 5, dtype=torch.bool)
    disconnected = graph(hidden=4, topology="self", summary=False)
    connected = graph(hidden=4, topology="self", summary=True)
    torch.testing.assert_close(
        disconnected(blocks, present)[:, 1:], disconnected(changed, present)[:, 1:], rtol=0, atol=0
    )
    assert not torch.equal(connected(blocks, present)[:, 1:], connected(changed, present)[:, 1:])


def test_moving_summary_updates_between_local_steps_changes_the_recurrence():
    blocks = torch.arange(20, dtype=torch.float64).reshape(1, 5, 4) / 3
    present = torch.ones(1, 5, dtype=torch.bool)
    for summary in (False, True):
        together = graph(hidden=4, cycles=1, local_steps=2, summary=summary)
        separated = graph(hidden=4, cycles=2, local_steps=1, summary=summary)
        a, b = together(blocks, present), separated(blocks, present)
        if summary:
            assert not torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_the_updated_summary_reaches_the_next_local_cycle():
    model = graph(hidden=4, cycles=2, local_steps=1)
    blocks = torch.arange(20, dtype=torch.float64).reshape(1, 5, 4) / 3
    norm_inputs, messages, contexts = [], [], []
    handles = [
        model.local_norm.register_forward_pre_hook(
            lambda module, args: norm_inputs.append(args[0].detach().clone())
        ),
        model.local_out.register_forward_hook(
            lambda module, args, result: messages.append(result.detach().clone())
        ),
        model.summary_to_local.register_forward_hook(
            lambda module, args, result: contexts.append(result.detach().clone())
        ),
    ]
    try:
        model(blocks, torch.ones(1, 5, dtype=torch.bool))
    finally:
        for handle in handles:
            handle.remove()
    assert len(norm_inputs) == len(messages) == len(contexts) == 2
    expected = blocks + 0.25 * messages[0] + contexts[0].unsqueeze(1)
    torch.testing.assert_close(norm_inputs[1], expected, rtol=0, atol=0)


def test_constructor_and_forward_do_not_consume_the_global_generator():
    before = torch.get_rng_state().clone()
    model = graph(hidden=4)
    x = torch.arange(20, dtype=torch.float64).reshape(1, 5, 4)
    model(x, torch.ones(1, 5, dtype=torch.bool))
    assert torch.equal(before, torch.get_rng_state())


def test_executed_recurrence_gradients_match_finite_differences():
    model = graph(cycles=2, local_steps=1)
    x = torch.arange(10, dtype=torch.float64).reshape(1, 5, 2) / 7
    present = torch.tensor([[True, True, False, True, True]])
    weight = model.qkv.weight.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda value, qkv: torch.func.functional_call(model, {"qkv.weight": qkv}, (value, present)),
        (x.requires_grad_(), weight),
        fast_mode=True,
    )


def test_incompatible_or_nonfinite_checkpoint_is_rejected_before_copying_parameters():
    source = graph(seed=3)
    target = graph(seed=4, cycles=1)
    original = {name: value.clone() for name, value in target.named_parameters()}
    with pytest.raises(ValueError):
        target.load_state_dict(source.state_dict())
    for name, value in target.named_parameters():
        assert torch.equal(original[name], value)
    state = target.state_dict()
    state["qkv.weight"] = torch.full_like(state["qkv.weight"], float("nan"))
    with pytest.raises(ValueError):
        target.load_state_dict(state)
    for name, value in target.named_parameters():
        assert torch.equal(original[name], value)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "edges"])
def test_malformed_checkpoint_does_not_partially_replace_the_regional_parameters(fault):
    model = graph()
    state = model.state_dict()
    original = {name: value.clone() for name, value in model.named_parameters()}
    if fault == "missing":
        state.pop("qkv.weight")
    elif fault == "extra":
        state["unknown"] = torch.ones(1)
    elif fault == "shape":
        state["qkv.weight"] = torch.ones(1, dtype=torch.float64)
    elif fault == "dtype":
        state["qkv.weight"] = state["qkv.weight"].float()
    else:
        state["edges"] = ~state["edges"]
    with pytest.raises(ValueError):
        model.load_state_dict(state)
    for name, value in model.named_parameters():
        assert torch.equal(value, original[name])


@pytest.mark.parametrize(
    "name",
    [
        "qkv.weight",
        "local_norm.bias",
        "readout.weight",
        "summary_update.0.weight",
        "summary_to_local.weight",
    ],
)
def test_nonfinite_intermediate_parameters_are_not_hidden_by_empty_masks(name):
    model = graph()
    with torch.no_grad():
        dict(model.named_parameters())[name].fill_(float("nan"))
    with pytest.raises(ValueError, match="NaN|infinito"):
        model(torch.ones(1, 5, 2, dtype=torch.float64), torch.zeros(1, 5, dtype=torch.bool))


@pytest.mark.parametrize(
    "options",
    [
        dict(cycles=True),
        dict(local_steps=0),
        dict(cycles=4, local_steps=4),
        dict(routing="unknown"),
        dict(topology="unknown"),
        dict(summary=1),
        dict(local_gain=float("nan")),
        dict(summary_gain=-1.0),
    ],
)
def test_invalid_or_unbounded_graph_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        graph(**options)


@pytest.mark.parametrize(
    "bad", ["blocks_shape", "blocks_dtype", "empty", "oversized", "mask_shape", "mask_dtype", "nan"]
)
def test_invalid_inputs_are_rejected_before_a_result_is_returned(bad):
    model = graph()
    x = torch.ones(1, 5, 2, dtype=torch.float64)
    present = torch.ones(1, 5, dtype=torch.bool)
    if bad == "blocks_shape":
        x = x[:, :4]
    elif bad == "blocks_dtype":
        x = x.float()
    elif bad == "empty":
        x, present = x[:0], present[:0]
    elif bad == "oversized":
        x, present = x.expand(257, -1, -1), present.expand(257, -1)
    elif bad == "mask_shape":
        present = present[:, :4]
    elif bad == "mask_dtype":
        present = present.float()
    else:
        x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(x, present)


def test_autocast_does_not_silently_relax_the_declared_precision():
    model = graph().float()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            model(torch.ones(1, 5, 2), torch.ones(1, 5, dtype=torch.bool))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("routing", ["attention", "uniform"])
@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("topology", ["complete", "ring", "self"])
def test_cuda_controls_match_cpu_with_missing_and_empty_rows(dtype, routing, summary, topology):
    model = graph(hidden=8, topology=topology, routing=routing, summary=summary).to(dtype=dtype)
    x = torch.arange(120, dtype=dtype).reshape(3, 5, 8) / 13
    present = torch.tensor([[True] * 5, [True, False, True, False, True], [False] * 5])
    expected = model(x, present).detach()
    before = torch.cuda.get_rng_state(0).clone()
    model = model.to("cuda:0")
    actual = model(x.to("cuda:0"), present.to("cuda:0")).detach().cpu()
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch.equal(before, torch.cuda.get_rng_state(0))
