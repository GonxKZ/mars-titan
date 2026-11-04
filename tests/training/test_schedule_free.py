"""AdamW sin calendario de F3 con la biblioteca fijada, sin pasos de optimizador."""

import pytest
import torch

from mars_titan.training.schedule_free import (
    SCHEDULE_FREE,
    AveragedPoint,
    ScheduleFree,
    is_schedule_free,
)

schedulefree = pytest.importorskip("schedulefree")


def parameters():
    generator = torch.Generator().manual_seed(4)
    return {
        "a": torch.nn.Parameter(torch.randn(3, 2, generator=generator, dtype=torch.float64)),
        "b": torch.nn.Parameter(torch.randn(2, generator=generator, dtype=torch.float64)),
    }


def groups(named):
    return [dict(params=[named["a"]], role="shared"), dict(params=[named["b"]], role="other")]


def test_build_uses_the_pinned_library_in_train_mode_with_the_recipe_rate():
    named = parameters()
    optimizer = ScheduleFree(warmup_steps=5).build(
        groups(named), learning_rate=1e-3, weight_decay=0.01
    )
    assert type(optimizer) is schedulefree.AdamWScheduleFree and is_schedule_free(optimizer)
    assert not is_schedule_free(torch.optim.AdamW(named.values()))
    for group in optimizer.param_groups:
        assert group["train_mode"] and group["lr"] == 1e-3 and group["weight_decay"] == 0.01
        assert group["betas"] == (0.9, 0.999) and group["eps"] == 1e-8
        assert group["warmup_steps"] == 5 and group["r"] == 0.0
        assert group["weight_lr_power"] == 2.0
    assert [group["role"] for group in optimizer.param_groups] == ["shared", "other"]
    assert all(not state for state in optimizer.state.values())


def test_the_averaged_point_is_what_eval_writes_and_leaves_y_untouched():
    named = parameters()
    optimizer = ScheduleFree(beta1=0.8).build(groups(named), learning_rate=1e-3, weight_decay=0)
    point = AveragedPoint(optimizer, named)
    before = {name: value.detach().clone() for name, value in named.items()}
    # Sin estado, antes del primer paso, x es y.
    assert all(torch.equal(point.emitted()[name], before[name]) for name in named)
    # z fijado a mano, no por un paso: x = lerp(y, z, 1 − 1/β₁).
    for value in named.values():
        optimizer.state[value]["z"] = value.detach() + 0.25
    emitted = point.emitted()
    assert all(torch.equal(value, before[name]) for name, value in named.items())
    optimizer.eval()
    for name, value in named.items():
        assert torch.equal(emitted[name], value.detach()), name
        expected = (before[name] - 0.2 * (before[name] + 0.25)) / 0.8
        torch.testing.assert_close(emitted[name], expected, rtol=1e-13, atol=1e-15)
    with pytest.raises(ValueError, match="modo de ajuste"):
        point.emitted()


def test_the_averaged_point_must_cover_the_trainable_parameters():
    named = parameters()
    optimizer = ScheduleFree().build(
        [dict(params=[named["a"]], role="shared")], learning_rate=1e-3, weight_decay=0
    )
    with pytest.raises(ValueError, match="exactamente"):
        AveragedPoint(optimizer, named).emitted()
    with pytest.raises(ValueError, match="parámetros ajustados"):
        AveragedPoint(optimizer, {})


def test_identity_declares_library_values_and_the_emitted_point():
    identity = ScheduleFree().identity()
    assert identity["name"] == SCHEDULE_FREE and identity["library"] == "schedulefree==1.4.1"
    assert identity["betas"] == [0.9, 0.999] and identity["warmup_steps"] == 0
    assert identity["validation_and_selection"] == "averaged_point_x_computed_out_of_place"
    assert ScheduleFree(warmup_steps=10).identity() != identity


@pytest.mark.parametrize(
    "options",
    [
        dict(warmup_steps=-1),
        dict(warmup_steps=1.0),
        dict(warmup_steps=True),
        dict(beta1=0.0),
        dict(beta1=1.0),
        dict(beta1=float("nan")),
        dict(beta1=True),
    ],
)
def test_invalid_configurations_are_rejected(options):
    with pytest.raises(ValueError):
        ScheduleFree(**options)


def test_another_library_version_is_rejected(monkeypatch):
    from mars_titan.training import schedule_free

    monkeypatch.setattr(schedule_free.metadata, "version", lambda name: "1.4")
    with pytest.raises(ValueError, match="1.4.1"):
        ScheduleFree().build(groups(parameters()), learning_rate=1e-3, weight_decay=0)
