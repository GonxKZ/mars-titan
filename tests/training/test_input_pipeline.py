"""Tubería de lectura: orden, errores en su posición, límites y cierre de los hilos."""

import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mars_titan.training.input_pipeline import (
    DECODE_WORKERS_ENV,
    GROUP_CACHE_ENV,
    GROUP_CACHE_MIB,
    PREFETCH_ENV,
    PipelineOptions,
    background,
    ordered_map,
)


class Boom(Exception):
    pass


def jittered(value):
    time.sleep(random.Random(value).random() / 500)
    return value * value


def source(count, *, fail_at=None, log=None):
    try:
        for index in range(count):
            if index == fail_at:
                raise Boom(index)
            if log is not None:
                log.append(index)
            yield index
    finally:
        if log is not None:
            log.append("closed")


@pytest.mark.parametrize("workers", [1, 3, 8])
@pytest.mark.parametrize("lookahead", [1, 2, 9])
def test_ordered_map_returns_results_in_input_order(workers, lookahead):
    with ThreadPoolExecutor(workers) as pool:
        assert list(ordered_map(jittered, range(60), pool, lookahead)) == [v * v for v in range(60)]


def test_function_error_is_raised_after_the_previous_results():
    def function(value):
        if value == 7:
            raise Boom(value)
        return jittered(value)

    received = []
    with ThreadPoolExecutor(4) as pool, pytest.raises(Boom):
        for value in ordered_map(function, range(40), pool, 6):
            received.append(value)
    assert received == [v * v for v in range(7)]


def test_input_error_is_raised_after_the_submitted_results():
    received, log = [], []
    with ThreadPoolExecutor(4) as pool, pytest.raises(Boom):
        for value in ordered_map(jittered, source(30, fail_at=11, log=log), pool, 5):
            received.append(value)
    assert received == [v * v for v in range(11)]
    assert log[-1] == "closed"


def test_pending_tasks_never_exceed_the_lookahead():
    active, peak, lock = [0], [0], threading.Lock()
    release = threading.Event()

    def function(value):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        release.wait(0.01)
        with lock:
            active[0] -= 1
        return value

    submitted = []

    def items():
        for value in range(50):
            submitted.append(value)
            yield value

    with ThreadPoolExecutor(8) as pool:
        stream = ordered_map(function, items(), pool, 3)
        for consumed, value in enumerate(stream):
            assert value == consumed
            # Lo enviado nunca supera lo consumido más el adelanto declarado.
            assert len(submitted) <= consumed + 1 + 3
    assert peak[0] <= 3


def test_closing_the_map_cancels_pending_work_and_closes_the_input():
    log, calls = [], []

    def function(value):
        calls.append(value)
        time.sleep(0.002)
        return value

    with ThreadPoolExecutor(2) as pool:
        stream = ordered_map(function, source(1000, log=log), pool, 4)
        assert next(stream) == 0
        stream.close()
        finished = len(calls)
        time.sleep(0.02)
        assert len(calls) == finished <= 5
    assert log[-1] == "closed"


@pytest.mark.parametrize("lookahead", [0, -1, True, 1.5])
def test_invalid_lookahead_is_rejected(lookahead):
    with ThreadPoolExecutor(1) as pool, pytest.raises(ValueError):
        next(ordered_map(jittered, range(3), pool, lookahead))


@pytest.mark.parametrize("depth", [1, 2, 16])
def test_background_preserves_order_and_finishes(depth):
    assert list(background(iter(range(200)), depth)) == list(range(200))


def test_background_error_is_raised_after_the_previous_items():
    received, log = [], []
    with pytest.raises(Boom):
        for value in background(source(20, fail_at=9, log=log), 3):
            received.append(value)
    assert received == list(range(9))
    assert log[-1] == "closed"


def test_background_producer_stays_within_its_depth_and_stops_when_closed():
    log = []
    stream = background(source(10_000, log=log), 4)
    assert next(stream) == 0
    time.sleep(0.05)
    produced = [entry for entry in log if entry != "closed"]
    # Un elemento entregado, `depth` en la cola y uno esperando a entrar.
    assert len(produced) <= 1 + 4 + 1
    stream.close()
    assert log[-1] == "closed"
    assert not [t for t in threading.enumerate() if t.name == "mars-titan-prefetch"]


def test_background_runs_the_source_outside_the_consumer_thread():
    threads = []

    def items():
        for value in range(3):
            threads.append(threading.current_thread())
            yield value

    assert list(background(items(), 1)) == [0, 1, 2]
    assert threading.current_thread() not in threads


ABANDONED = """
from mars_titan.training.input_pipeline import background
def items():
    try:
        n = 0
        while True:
            yield n
            n += 1
    finally:
        print("cerrado", flush=True)
stream = background(items(), 2)
next(stream)
"""


SELF_CLOSED = """
import threading
from concurrent.futures import ThreadPoolExecutor
from mars_titan.training.input_pipeline import ordered_map
executor, gate, holder = ThreadPoolExecutor(1), threading.Event(), {}
def work(item):
    if item == 1:
        gate.wait()
        # Como si el recolector finalizara el recorrido abandonado dentro de su tarea.
        holder["stream"].close()
    return item
holder["stream"] = stream = ordered_map(work, range(4), executor, 2)
assert next(stream) == 0
gate.set()
print(executor.submit(int, 7).result(timeout=30), flush=True)
"""


def test_a_map_closed_inside_its_own_task_does_not_wait_for_itself(source_environment):
    result = subprocess.run(
        [sys.executable, "-c", SELF_CLOSED],
        capture_output=True,
        text=True,
        timeout=60,
        env=source_environment,
    )
    assert result.returncode == 0 and result.stdout.split() == ["7"]


def test_an_abandoned_producer_closes_its_source_before_the_interpreter_exits(source_environment):
    result = subprocess.run(
        [sys.executable, "-c", ABANDONED],
        capture_output=True,
        text=True,
        timeout=60,
        env=source_environment,
    )
    assert result.returncode == 0 and result.stdout.split() == ["cerrado"]


@pytest.mark.parametrize("depth", [0, -2, False])
def test_invalid_depth_is_rejected(depth):
    with pytest.raises(ValueError):
        next(background(iter(range(3)), depth))


def test_options_default_to_the_sequential_reader(monkeypatch):
    for name in (DECODE_WORKERS_ENV, PREFETCH_ENV, GROUP_CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    options = PipelineOptions.from_environment()
    assert options == PipelineOptions()
    assert (options.decode_workers, options.prefetch_batches, options.lookahead) == (0, 0, 0)
    assert options.group_cache_bytes == GROUP_CACHE_MIB * 1024**2


def test_options_round_trip_through_the_environment(monkeypatch):
    options = PipelineOptions(decode_workers=3, prefetch_batches=8, group_cache_mib=512)
    for name, value in options.environment().items():
        monkeypatch.setenv(name, value)
    assert PipelineOptions.from_environment() == options
    assert options.lookahead == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (DECODE_WORKERS_ENV, "-1"),
        (DECODE_WORKERS_ENV, "17"),
        (DECODE_WORKERS_ENV, "two"),
        (PREFETCH_ENV, "65"),
        (PREFETCH_ENV, " 4"),
        (GROUP_CACHE_ENV, "0"),
        (GROUP_CACHE_ENV, "8193"),
    ],
)
def test_invalid_environment_fails_fast(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        PipelineOptions.from_environment()


@pytest.mark.parametrize(
    "fields",
    [
        dict(decode_workers=-1),
        dict(decode_workers=True),
        dict(prefetch_batches=65),
        dict(group_cache_mib=0),
        dict(group_cache_mib=1.5),
    ],
)
def test_invalid_options_fail_fast(fields):
    with pytest.raises(ValueError):
        PipelineOptions(**fields)
