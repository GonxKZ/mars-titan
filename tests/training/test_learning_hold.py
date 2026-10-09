import json

import pytest
import torch

from mars_titan.training.learning_hold import install_optimizer_guard, learning_blocked


class Blocked(Exception):
    pass


def _raise(reason):
    raise Blocked(reason)


def _hold(tmp_path, allowed):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": allowed}), encoding="utf-8")
    return path


def test_missing_hold_does_not_block(tmp_path):
    assert learning_blocked(tmp_path / "absent.json") is False
    assert install_optimizer_guard(_raise, tmp_path / "absent.json") is None


def test_allowed_hold_does_not_install_guard(tmp_path):
    path = _hold(tmp_path, True)
    assert learning_blocked(path) is False
    assert install_optimizer_guard(_raise, path) is None


@pytest.mark.parametrize("value", [None, "false", 0])
def test_ambiguous_hold_fails_fast(tmp_path, value):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": value}), encoding="utf-8")
    with pytest.raises(ValueError):
        learning_blocked(path)


@pytest.mark.parametrize("optimizer", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_blocked_step_stops_before_weights_change(tmp_path, optimizer):
    handle = install_optimizer_guard(_raise, _hold(tmp_path, False))
    assert handle is not None
    try:
        weight = torch.nn.Parameter(torch.ones(3))
        opt = optimizer([weight], lr=0.5)
        weight.sum().backward()
        before = weight.detach().clone()
        # La guarda de conftest puede actuar antes y omitir la prueba. Ambas detienen el paso.
        with pytest.raises((Blocked, pytest.skip.Exception), match="Bloqueo de aprendizaje"):
            opt.step()
        assert torch.equal(weight.detach(), before)
        assert not opt.state
    finally:
        handle.remove()
