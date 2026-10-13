"""Rangos NVTX opcionales: apagados no hacen nada y encendidos abren y cierran cada fase."""

import importlib
import random
import threading

import numpy as np
import pytest
import torch

from mars_titan import nvtx_ranges


class Recorder:
    """Sustituto de `torch.cuda.nvtx` que guarda la pila de cada hilo y reenvía las marcas."""

    def __init__(self, forward=True):
        self.forward = forward
        self.events, self.depth, self.lock = [], {}, threading.Lock()

    def range_push(self, name):
        thread = threading.get_ident()
        with self.lock:
            self.depth[thread] = self.depth.get(thread, 0) + 1
            self.events.append(("push", thread, name))
        if self.forward:
            torch.cuda.nvtx.range_push(name)

    def range_pop(self):
        thread = threading.get_ident()
        with self.lock:
            self.depth[thread] = self.depth.get(thread, 0) - 1
            assert self.depth[thread] >= 0, "range_pop sin range_push en el mismo hilo"
            self.events.append(("pop", thread, None))
        if self.forward:
            torch.cuda.nvtx.range_pop()

    def names(self):
        return [name for kind, _, name in self.events if kind == "push"]

    def balanced(self):
        return all(depth == 0 for depth in self.depth.values())


@pytest.fixture
def recorder(monkeypatch):
    """Rangos encendidos con un registro. Reenvía a NVTX real si PyTorch trae CUDA."""
    recorded = Recorder(forward=torch.version.cuda is not None)
    monkeypatch.setattr(nvtx_ranges, "ENABLED", True)
    monkeypatch.setattr(nvtx_ranges, "_nvtx", lambda: recorded)
    return recorded


def test_disabled_ranges_return_the_same_objects_without_marks(monkeypatch):
    marks = Recorder(forward=False)
    monkeypatch.setattr(nvtx_ranges, "ENABLED", False)
    monkeypatch.setattr(nvtx_ranges, "_nvtx", lambda: marks)
    first, second = nvtx_ranges.phase("a"), nvtx_ranges.phase("b")
    assert first is second
    values = [1, 2, 3]
    assert nvtx_ranges.iterate("read", values) is values
    with first:
        pass
    assert marks.events == []


def test_enabled_phase_pops_its_range_when_the_body_fails(recorder):
    with pytest.raises(KeyError):
        with nvtx_ranges.phase("outer"):
            with nvtx_ranges.phase("inner"):
                raise KeyError("fallo")
    assert recorder.names() == ["outer", "inner"]
    assert [kind for kind, *_ in recorder.events] == ["push", "push", "pop", "pop"]
    assert recorder.balanced()


def test_enabled_phase_does_not_swallow_exceptions(recorder):
    with pytest.raises(ValueError, match="sigue"):
        with nvtx_ranges.phase("fase"):
            raise ValueError("sigue")


def test_iterate_brackets_each_next_and_not_the_consumer(recorder):
    seen = []

    def produce():
        for value in range(3):
            seen.append(("next", value, recorder.depth.get(threading.get_ident(), 0)))
            yield value

    for value in nvtx_ranges.iterate("read", produce()):
        # El cuerpo del bucle del consumidor queda fuera del rango de lectura.
        seen.append(("body", value, recorder.depth.get(threading.get_ident(), 0)))
    assert seen == [
        ("next", 0, 1),
        ("body", 0, 0),
        ("next", 1, 1),
        ("body", 1, 0),
        ("next", 2, 1),
        ("body", 2, 0),
    ]
    # Tres elementos y la llamada final que agota el generador.
    assert recorder.names() == ["read"] * 4
    assert recorder.balanced()


def test_iterate_closes_the_reader_when_the_consumer_stops(recorder):
    closed = []

    def produce():
        try:
            yield from range(10)
        finally:
            closed.append(True)

    # La prueba conserva el generador interno para que solo lo cierre el envoltorio.
    inner = produce()
    wrapper = nvtx_ranges.iterate("read", inner)
    for value in wrapper:
        if value == 1:
            break
    assert closed == []
    wrapper.close()
    assert closed == [True] and inner.gi_frame is None
    assert recorder.balanced()


def test_iterate_propagates_reader_failures_with_balanced_ranges(recorder):
    def produce():
        yield 0
        raise OSError("lectura")

    with pytest.raises(OSError, match="lectura"):
        list(nvtx_ranges.iterate("read", produce()))
    assert recorder.names() == ["read", "read"]
    assert recorder.balanced()


def test_ranges_are_balanced_per_thread(recorder):
    # Los cuatro hilos tienen su rango abierto a la vez, como el lector con prefetch y el
    # bucle principal. Cada pila se cierra en su propio hilo.
    barrier = threading.Barrier(4)

    def work():
        for _ in range(50):
            with nvtx_ranges.phase("hilo"):
                barrier.wait()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert recorder.names().count("hilo") == 200
    assert len(recorder.depth) == 4 and recorder.balanced()


def test_enabled_ranges_leave_random_states_untouched(recorder):
    states = (torch.get_rng_state(), random.getstate(), np.random.get_state()[1].copy())
    for _ in nvtx_ranges.iterate("read", range(3)):
        with nvtx_ranges.phase("fase"):
            pass
    assert torch.equal(states[0], torch.get_rng_state())
    assert states[1] == random.getstate()
    assert np.array_equal(states[2], np.random.get_state()[1])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("1", True), ("2", ValueError), ("true", ValueError)],
)
def test_environment_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(nvtx_ranges.ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(nvtx_ranges.ENVIRONMENT, value)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    if isinstance(expected, bool):
        assert nvtx_ranges._enabled() is expected
    else:
        with pytest.raises(expected):
            nvtx_ranges._enabled()


def test_enabled_without_cuda_fails_at_import(monkeypatch):
    monkeypatch.setenv(nvtx_ranges.ENVIRONMENT, "1")
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CUDA"):
        nvtx_ranges._enabled()


def test_module_reads_the_switch_once_at_import(monkeypatch):
    monkeypatch.setenv(nvtx_ranges.ENVIRONMENT, "0")
    try:
        module = importlib.reload(nvtx_ranges)
        assert module.ENABLED is False
        monkeypatch.setenv(nvtx_ranges.ENVIRONMENT, "1")
        # Cambiar la variable después de importar no enciende los rangos.
        assert module.phase("x") is module._OFF
    finally:
        # Volver a importar con la variable original para no alterar las demás pruebas.
        monkeypatch.undo()
        importlib.reload(nvtx_ranges)
