"""Comprobaciones de datos diferidas en CUDA: mismos errores y mensajes, una sincronización.

En CPU las comprobaciones siguen siendo inmediatas. En CUDA, dentro de `deferred_checks`,
se resuelven al cerrar el tramo con el mismo `ValueError` y el mensaje de la primera
comprobación fallida en orden de programa.
"""

import pytest
import torch

from mars_titan.models.titans import state as checks

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere cuda:0")


def finite(device="cpu"):
    return torch.ones(3, device=device)


def broken(device="cpu"):
    return torch.tensor([1.0, float("nan")], device=device)


def test_cpu_checks_stay_immediate_inside_the_deferred_block():
    with pytest.raises(ValueError, match="^Entrada contiene NaN o infinito$"):
        with checks.deferred_checks():
            checks.check_finite(broken(), "Entrada")
            pytest.fail("La comprobación en CPU no puede diferirse")
    with pytest.raises(ValueError, match="^Negativo$"):
        with checks.deferred_checks():
            checks.require_true(torch.tensor(False), "Negativo")
            pytest.fail("La comprobación en CPU no puede diferirse")
    checks.check_finite(finite(), "Entrada")
    checks.require_true(torch.tensor(True), "Negativo")


def test_checks_outside_a_block_are_immediate():
    with pytest.raises(ValueError, match="^Entrada contiene NaN o infinito$"):
        checks.check_finite(broken(), "Entrada")
    assert checks._PENDING.get() is None


@cuda
def test_cuda_checks_resolve_at_the_end_with_the_first_message():
    reached = []
    with pytest.raises(ValueError, match="^Primera contiene NaN o infinito$"):
        with checks.deferred_checks():
            checks.check_finite(finite("cuda:0"), "Correcta")
            checks.check_finite(broken("cuda:0"), "Primera")
            checks.require_true(torch.tensor(False, device="cuda:0"), "Segunda")
            reached.append(True)
    assert reached == [True]
    assert checks._PENDING.get() is None


@cuda
def test_cuda_checks_pass_silently_and_deduplicate_unchanged_tensors():
    value = finite("cuda:0")
    with checks.deferred_checks():
        checks.check_finite(value, "Valor")
        checks.check_finite(value, "Valor")
        pending = checks._PENDING.get()
        assert len(pending.flags) == 1
        value.add_(1)
        checks.check_finite(value, "Valor")
        assert len(pending.flags) == 2
        checks.require_true(torch.tensor(True, device="cuda:0"), "Cierto")


@cuda
def test_each_check_sees_the_value_at_its_program_point():
    value = finite("cuda:0")
    with checks.deferred_checks():
        checks.check_finite(value, "Antes")
        value[0] = float("nan")
    with pytest.raises(ValueError, match="^Después contiene NaN o infinito$"):
        with checks.deferred_checks():
            checks.check_finite(value, "Después")


@cuda
def test_an_earlier_failed_check_wins_over_a_later_error():
    with pytest.raises(ValueError, match="^Primera contiene NaN o infinito$") as raised:
        with checks.deferred_checks():
            checks.check_finite(broken("cuda:0"), "Primera")
            raise KeyError("posterior")
    assert isinstance(raised.value.__cause__, KeyError)
    with pytest.raises(KeyError):
        with checks.deferred_checks():
            checks.check_finite(finite("cuda:0"), "Correcta")
            raise KeyError("posterior")


@cuda
def test_an_immediate_failure_resolves_the_earlier_deferred_checks_first():
    with pytest.raises(ValueError, match="^Primera contiene NaN o infinito$"):
        with checks.deferred_checks():
            checks.check_finite(broken("cuda:0"), "Primera")
            checks.check_finite(broken(), "En CPU")


@cuda
def test_nested_blocks_resolve_only_at_the_outer_end():
    reached = []
    with pytest.raises(ValueError, match="^Interna contiene NaN o infinito$"):
        with checks.deferred_checks():
            with checks.deferred_checks():
                checks.check_finite(broken("cuda:0"), "Interna")
            reached.append(True)
    assert reached == [True]
