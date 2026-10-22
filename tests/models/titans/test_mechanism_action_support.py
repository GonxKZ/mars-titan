"""Máscara declarada de acciones observadas, con coeficientes manuales y sin ajuste."""

from copy import deepcopy

import pytest
import torch
from test_mechanism_policy import allowed, api, inputs, policy

DEVICES = ("cpu", pytest.param("cuda:0", marks=pytest.mark.cuda))


def masked(parent, mask):
    try:
        return api().MechanismPolicy(
            **{name: value for name, value in parent.named_buffers()},
            interval_weight=parent.interval_weight,
            min_gain=parent.min_gain,
            action_mask=mask,
        )
    except TypeError as error:
        pytest.fail(f"Falta la máscara explícita de acciones con objetivos observados: {error}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_unknown_actions_cannot_win_despite_large_supplied_coefficients(device, dtype):
    parent = policy(device=device, dtype=dtype)
    mask = torch.ones(36, dtype=torch.bool, device=device)
    mask[4] = False
    model = masked(parent, mask)
    values, available = inputs(model), allowed(model)
    before = available.clone()
    result = model(values, available)
    assert result.action_ids.tolist() == [3, 3]
    assert result.scores[0, 4] == 3.5
    assert result.available[:, 4].tolist() == [False, False]
    assert torch.equal(available, before)
    assert model.get_extra_state()["action_support"] == "declared_boolean_mask_not_fit_attestation"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_all_supported_actions_keep_the_unmasked_scores_choices_and_rng(device, dtype):
    parent = policy(device=device, dtype=dtype)
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(0) if device != "cpu" else None
    model = masked(parent, torch.ones(36, dtype=torch.bool, device=device))
    left, right = parent(inputs(parent), allowed(parent)), model(inputs(model), allowed(model))
    for name in vars(left):
        assert torch.equal(getattr(left, name), getattr(right, name))
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if gpu_rng is not None:
        assert torch.equal(gpu_rng, torch.cuda.get_rng_state(0))
    assert model.action_mask.dtype == torch.bool


@pytest.mark.parametrize("fault", ["baseline", "dtype", "shape", "device", "extra_dimension"])
def test_invalid_action_masks_cannot_create_a_policy(fault):
    parent = policy()
    mask = torch.ones(36, dtype=torch.bool)
    if fault == "baseline":
        mask[0] = False
    elif fault == "dtype":
        mask = mask.double()
    elif fault == "shape":
        mask = mask[:35]
    elif fault == "device":
        mask = mask.to("meta")
    else:
        mask = mask[None, :]
    with pytest.raises(ValueError):
        masked(parent, mask)


def test_old_policy_checkpoint_and_extra_state_stay_unchanged_without_a_mask():
    parent, restored = policy(), policy()
    state = deepcopy(parent.state_dict())
    assert "action_mask" not in state
    assert "action_support" not in state["_extra_state"]
    restored.load_state_dict(state)
    assert torch.equal(
        parent(inputs(parent), allowed(parent)).scores,
        restored(inputs(restored), allowed(restored)).scores,
    )


def test_masked_checkpoint_restores_the_next_choice_and_rejects_a_false_baseline():
    parent = policy()
    model = masked(parent, torch.tensor([True] + [False] * 35))
    restored = masked(parent, torch.ones(36, dtype=torch.bool))
    restored.load_state_dict(model.state_dict())
    assert restored(inputs(restored), allowed(restored)).action_ids.tolist() == [0, 0]
    previous = deepcopy(restored.state_dict())
    bad = deepcopy(previous)
    bad["costs"][4] = 12
    bad["action_mask"][0] = False
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
    for name, value in previous.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored.state_dict()[name], value)
        else:
            assert restored.state_dict()[name] == value


def test_declared_fit_receipt_is_preserved_without_attesting_that_it_was_executed():
    parent = policy()
    try:
        linked = api().MechanismPolicy(
            **dict(parent.named_buffers()),
            interval_weight=parent.interval_weight,
            min_gain=parent.min_gain,
            fit_receipt_sha256="a" * 64,
        )
    except TypeError:
        pytest.fail("Falta la referencia explícita al recibo de ajuste")
    assert linked.get_extra_state()["declared_fit_receipt_sha256"] == "a" * 64
    from mars_titan.models.titans.mechanism_selection import policy_signature

    assert policy_signature(linked, strong=True)[1]["coefficient_origin"] == (
        "supplied_with_declared_fit_receipt_not_independently_verified"
    )
    restored = api().MechanismPolicy(
        **dict(parent.named_buffers()),
        interval_weight=parent.interval_weight,
        min_gain=parent.min_gain,
        fit_receipt_sha256="a" * 64,
    )
    restored.load_state_dict(linked.state_dict())
    assert torch.equal(
        linked(inputs(linked), allowed(linked)).scores,
        restored(inputs(restored), allowed(restored)).scores,
    )
    with pytest.raises(ValueError):
        restored.load_state_dict(parent.state_dict())


def test_the_supplied_action_mask_cannot_change_the_owned_policy_buffer():
    parent = policy()
    supplied = torch.ones(36, dtype=torch.bool)
    model = masked(parent, supplied)
    supplied[4] = False
    assert model(inputs(model), allowed(model)).action_ids.tolist() == [4, 4]
    assert model.action_mask[4]


@pytest.mark.parametrize("value", ["", "A" * 64, True, 12])
def test_invalid_declared_receipt_does_not_create_a_policy(value):
    parent = policy()
    with pytest.raises(ValueError):
        api().MechanismPolicy(
            **dict(parent.named_buffers()),
            interval_weight=parent.interval_weight,
            min_gain=parent.min_gain,
            fit_receipt_sha256=value,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("interval_weight", -1.0),
        ("interval_weight", float("inf")),
        ("min_gain", True),
        ("min_gain", -0.1),
        ("fit_receipt_sha256", "invalid"),
    ],
)
def test_validation_rejects_invalid_scalars_changed_before_freezing(field, value):
    model = policy()
    setattr(model, field, value)
    with pytest.raises(ValueError):
        model.validate()
