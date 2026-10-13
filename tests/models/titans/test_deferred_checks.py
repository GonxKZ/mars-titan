"""Comprobaciones de datos diferidas en CUDA: mismos errores y mensajes, una sincronización.

En CPU las comprobaciones siguen siendo inmediatas. En CUDA, dentro de `deferred_checks`,
se resuelven al cerrar el tramo con el mismo `ValueError` y el mensaje de la primera
comprobación fallida en orden de programa.
"""

import warnings

import pytest
import torch

from mars_titan.models.titans import state as checks

cuda = pytest.mark.cuda


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


DEVICES = ["cpu", pytest.param("cuda:0", marks=cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_finite_indicator_agrees_with_isfinite(device, dtype):
    largest = torch.finfo(dtype).max
    cases = [
        torch.tensor([largest, -largest, torch.finfo(dtype).tiny / 4, 0.0, -0.0], dtype=dtype),
        torch.empty(0, dtype=dtype),
    ]
    for size in (1, 7, 4097, 2**20 + 3):
        for position in sorted({0, size // 2, size - 1}):
            for bad in (float("nan"), float("inf"), -float("inf")):
                value = torch.randn(size, dtype=torch.float64).to(dtype)
                value[position] = bad
                cases.append(value)
    # Vistas con pasos, que la reducción recorre sin copiarlas.
    view = torch.randn(64, 64, dtype=torch.float64).to(dtype)
    view[3, 7] = float("nan")
    cases.extend([view[:, ::2], view[::2, 1::2], view.t()])
    for value in cases:
        value = value.to(device)
        expected = bool(torch.isfinite(value).all())
        assert checks.is_finite(checks.finite_indicator(value)) is expected


@pytest.mark.parametrize("device", DEVICES)
def test_finite_indicator_keeps_the_general_check_for_other_types(device):
    for value, expected in (
        (torch.arange(5), True),
        (torch.tensor([True, False]), True),
        (torch.tensor([1 + 2j, complex("nan")]), False),
        # Componentes finitas cuyo módulo desbordaría en float32 siguen siendo finitas.
        (torch.tensor([3e38 + 3e38j], dtype=torch.complex64), True),
    ):
        indicator = checks.finite_indicator(value.to(device))
        assert indicator.dtype == torch.bool
        assert checks.is_finite(indicator) is expected


def test_finite_indicator_adds_no_autograd_graph():
    value = torch.randn(4, requires_grad=True) * 2
    indicator = checks.finite_indicator(value)
    assert not indicator.requires_grad
    assert indicator.grad_fn is None


@cuda
def test_cuda_flags_and_bounds_resolve_in_program_order():
    with pytest.raises(ValueError, match="^Bandera$"):
        with checks.deferred_checks():
            checks.check_finite(finite("cuda:0"), "Correcta")
            checks.require_true(torch.tensor(False, device="cuda:0"), "Bandera")
            checks.check_finite(broken("cuda:0"), "Después")
    with pytest.raises(ValueError, match="^Antes contiene NaN o infinito$"):
        with checks.deferred_checks():
            checks.check_finite(broken("cuda:0").double(), "Antes")
            checks.require_true(torch.tensor(False, device="cuda:0"), "Bandera")
            checks.require_finite(broken("cuda:0"), "Mensaje propio")
    with pytest.raises(ValueError, match="^Mensaje propio$"):
        with checks.deferred_checks():
            checks.require_finite(broken("cuda:0"), "Mensaje propio")


@cuda
def test_cuda_block_resolves_with_one_synchronization():
    values = [finite("cuda:0") * index for index in range(20)]
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with checks.deferred_checks():
                for value in values:
                    checks.check_finite(value, "Valor")
                    checks.require_true((value >= 0).all(), "Signo")
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert sum("synchroniz" in str(item.message) for item in caught) == 1


@cuda
def test_nested_block_forgets_its_tensors_but_keeps_their_checks():
    kept = finite("cuda:0")
    with pytest.raises(ValueError, match="^Interna contiene NaN o infinito$"):
        with checks.deferred_checks():
            pending = checks._PENDING.get()
            checks.check_finite(kept, "Externa")
            outer = pending.seen
            with checks.deferred_checks():
                assert checks._PENDING.get() is pending
                assert pending.seen == {}
                checks.check_finite(broken("cuda:0"), "Interna")
                assert len(pending.seen) == 1
            # El tramo interior ya no retiene su temporal, pero su comprobación sigue pendiente.
            assert pending.seen is outer
            assert [value is kept for value in outer.values()] == [True]
            assert len(pending.flags) == 2
