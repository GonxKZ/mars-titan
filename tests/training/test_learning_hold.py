import json
import os
from pathlib import Path

import pytest
import torch

from mars_titan.training.learning_hold import (
    HOLD_ENV,
    LearningHoldError,
    install_optimizer_guard,
    learning_blocked,
    require_learning_allowed,
)

pytest_plugins = ["pytester"]
ROOT = Path(__file__).resolve().parents[2]


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


def test_required_learning_names_the_action_and_the_hold(tmp_path, monkeypatch):
    path = _hold(tmp_path, False)
    monkeypatch.setenv(HOLD_ENV, str(path))
    with pytest.raises(LearningHoldError) as error:
        require_learning_allowed("el ajuste de prueba")
    assert str(error.value) == (
        f"Bloqueo de aprendizaje vigente: el ajuste de prueba no se ejecuta mientras {path} "
        "no declare training_allowed verdadero"
    )


@pytest.mark.parametrize("allowed", [True, None])
def test_required_learning_passes_when_allowed_or_absent(tmp_path, monkeypatch, allowed):
    path = tmp_path / "absent.json" if allowed is None else _hold(tmp_path, allowed)
    monkeypatch.setenv(HOLD_ENV, str(path))
    assert require_learning_allowed("el ajuste de prueba") is None


def test_required_learning_rejects_an_ambiguous_hold(tmp_path, monkeypatch):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": "false"}), encoding="utf-8")
    monkeypatch.setenv(HOLD_ENV, str(path))
    with pytest.raises(ValueError):
        require_learning_allowed("el ajuste de prueba")


def test_hold_error_is_not_an_operational_error():
    # Los lanzadores capturan estos tipos para registrar intentos fallidos o códigos de salida.
    assert not issubclass(LearningHoldError, (OSError, ValueError, RuntimeError))


def test_conftest_skips_hold_stops_in_setup_and_call_but_not_other_errors(
    pytester, tmp_path, monkeypatch
):
    monkeypatch.setenv(HOLD_ENV, str(tmp_path / "absent.json"))
    # El conftest común importa tests.suite_support, así que la raíz también va en la ruta.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    pytester.makeconftest((ROOT / "tests/conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(
        """
        import pytest
        from mars_titan.training.learning_hold import LearningHoldError

        @pytest.fixture
        def stopped():
            raise LearningHoldError("Bloqueo de aprendizaje vigente: preparación")

        def test_setup(stopped):
            pass

        def test_call():
            raise LearningHoldError("Bloqueo de aprendizaje vigente: llamada")

        def test_other_error():
            raise RuntimeError("fallo ordinario")
        """
    )
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs")
    result.assert_outcomes(skipped=2, failed=1)
    result.stdout.fnmatch_lines(["*Bloqueo de aprendizaje vigente: preparación*"])
