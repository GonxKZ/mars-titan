"""Brazos con reinicio de la memoria rápida (R0 y R63) sobre la ventana técnica del fixture.

R0 evalúa un tramo medido desde el estado inicial de cada flujo y del contexto de mercado al
abrirse la decisión, sin calcular el calentamiento; R63 además vuelve a ese estado cada 63
instantes medidos (aprobados por el coordinador el 22 de noviembre de 2026). Oráculos: sin
reinicio, la regeneración de un solo tramo repite bit a bit lo emitido; R0 es el recorrido sin
meses de calentamiento; un periodo que no se alcanza dentro del tramo es R0; con periodo k, los k
primeros instantes son los de R0 y los siguientes cambian; la red cargada no cambia.
"""

import json
import math
from types import SimpleNamespace

import pytest

from mars_titan.models.titans.financial import _tensor_digest
from mars_titan.training import prediction_regeneration as regeneration
from mars_titan.training import titans_walk_forward as wf
from mars_titan.training.financial_run import ChronologicalInference, FlowStates, MemoryReset, _Pass
from tests.training.test_titans_walk_forward import (  # noqa: F401
    _compare,
    base,
    learning_doubles_module,
    table,
    unfused_attention,
)

pytestmark = pytest.mark.usefixtures("learning_doubles")


def regenerated(base, output, **options):  # noqa: F811
    """El estado elegido de la ventana mac_online del fixture, repetido en su propia ventana."""
    anchor = base["root"] / "runs" / "mac_online"
    with unfused_attention():
        return wf.carry_titans(
            anchor, base["view"], base["view"], output, device="cpu", regenerate=True, **options
        )


def metrics(receipt, partition="evaluation"):
    return receipt["predictions"][partition]["metrics"]


def decision_instants(output):
    """Instantes de decisión con filas emitidas en la evaluación, en orden."""
    return sorted(set(table(output, "evaluation")["prediction_at"]))


@pytest.mark.parametrize("partition", ["calibration", "evaluation"])
def test_without_reset_a_single_partition_regenerates_bit_for_bit(base, tmp_path, partition):  # noqa: F811
    anchor = base["root"] / "runs" / "mac_online"
    receipt = regenerated(base, tmp_path / "again", partitions=(partition,))
    assert set(receipt["predictions"]) == {partition} and set(receipt["phases"]) == {partition}
    assert "memory_reset" not in receipt
    assert not {"memory_resets", "warmup_events_skipped"} & set(metrics(receipt, partition))
    report = json.loads((anchor / "run.json").read_text())
    originals = regeneration.originals(anchor, report)
    result = regeneration.compare({partition: originals[partition]}, tmp_path / "again", receipt)
    assert result["identical"] is True, result


def test_r0_is_the_pass_without_warmup_months(base, tmp_path, monkeypatch):  # noqa: F811
    r0 = regenerated(base, tmp_path / "r0", partitions=("evaluation",), memory_reset=MemoryReset())
    assert r0["memory_reset"] == MemoryReset().record() and r0["memory_reset"]["every"] is None
    counted = metrics(r0)
    assert counted["warmup_observations"] == 0 and counted["warmup_events_skipped"] > 0
    assert counted["memory_resets"] == 1 and counted["measured_instants"] > 0
    phases = wf.window_phases
    monkeypatch.setattr(wf, "window_phases", lambda fold, warmup, *rest: phases(fold, 0, *rest))
    cold = regenerated(base, tmp_path / "cold", partitions=("evaluation",))
    assert metrics(cold)["warmup_observations"] == 0
    assert _compare(tmp_path / "r0", tmp_path / "cold", "evaluation")
    # La ventana emitida se calentó un mes escribiendo en la memoria, y eso cambia lo emitido.
    assert not _compare(tmp_path / "r0", base["root"] / "runs" / "mac_online", "evaluation")


def test_a_period_that_the_partition_does_not_reach_is_r0(base, tmp_path):  # noqa: F811
    r0 = regenerated(base, tmp_path / "r0", partitions=("evaluation",), memory_reset=MemoryReset())
    measured = metrics(r0)["measured_instants"]
    for every in (measured, 10 * measured):
        receipt = regenerated(
            base,
            tmp_path / f"every-{every}",
            partitions=("evaluation",),
            memory_reset=MemoryReset(every=every),
        )
        assert receipt["memory_reset"]["every"] == every
        assert metrics(receipt)["memory_resets"] == 1
        assert _compare(tmp_path / "r0", tmp_path / f"every-{every}", "evaluation")


def test_a_periodic_reset_repeats_r0_until_its_first_reset_and_then_departs(base, tmp_path):  # noqa: F811
    every = 5
    r0 = regenerated(base, tmp_path / "r0", partitions=("evaluation",), memory_reset=MemoryReset())
    periodic = regenerated(
        base, tmp_path / "periodic", partitions=("evaluation",), memory_reset=MemoryReset(every)
    )
    measured = metrics(r0)["measured_instants"]
    assert measured > 2 * every
    assert metrics(periodic)["memory_resets"] == math.ceil(measured / every)
    assert metrics(periodic)["measured_instants"] == measured
    instants = decision_instants(tmp_path / "r0")
    assert _compare(tmp_path / "r0", tmp_path / "periodic", "evaluation", until=instants[every - 1])
    assert not _compare(tmp_path / "r0", tmp_path / "periodic", "evaluation", until=instants[every])
    # Las filas, sus objetivos y lo que madura son los mismos: solo cambia lo predicho.
    left, right = table(tmp_path / "r0", "evaluation"), table(tmp_path / "periodic", "evaluation")
    assert left["sample_id"] == right["sample_id"] and left["target"] == right["target"]


def test_resets_do_not_change_the_loaded_network(base, tmp_path, monkeypatch):  # noqa: F811
    seen, original = [], ChronologicalInference.predict

    def digests(predictor):
        return {
            name: _tensor_digest(value)
            for name, value in predictor.state_dict().items()
            if hasattr(value, "dtype")
        }

    def recorded(self, source, rows, *, stop=None):
        before = digests(self.predictor)
        result = original(self, source, rows, stop=stop)
        seen.append((self.memory_reset, before, digests(self.predictor)))
        return result

    monkeypatch.setattr(ChronologicalInference, "predict", recorded)
    regenerated(base, tmp_path / "r", partitions=("evaluation",), memory_reset=MemoryReset(3))
    assert [reset for reset, _, _ in seen] == [MemoryReset(3)]
    assert all(before == after for _, before, after in seen)


@pytest.mark.parametrize(
    ("every", "cleared"),
    [(None, (True, False, False, False, False)), (2, (True, False, True, False, True))],
)
def test_a_reset_returns_each_flow_and_the_market_context_to_their_initial_state(every, cleared):
    """El contexto de mercado (PT4) no existe en el fixture de la ventana: se comprueba aquí."""
    engine = SimpleNamespace(memory_reset=MemoryReset(every))
    source = SimpleNamespace(phase=SimpleNamespace(decision_start=10))
    run = _Pass(flows=FlowStates({"US/A": (None, 0)}), pending={("US/A", 9): 0.5})
    run.context = "context"
    run.counters.update(memory_resets=0, warmup_events_skipped=0)
    # El calentamiento no se observa ni reinicia nada.
    assert ChronologicalInference._reset(engine, run, source, SimpleNamespace(at=9)) is False
    assert "US/A" in run.flows and run.context == "context"
    assert run.counters["warmup_events_skipped"] == 1
    for instants, expected in enumerate(cleared):
        run.flows, run.context, run.instants = FlowStates({"US/A": (None, 0)}), "context", instants
        assert ChronologicalInference._reset(engine, run, source, SimpleNamespace(at=10)) is True
        assert ("US/A" not in run.flows) is expected and (run.context is None) is expected
    assert run.counters["memory_resets"] == sum(cleared)
    # Lo pendiente de madurar se conserva.
    assert run.pending == {("US/A", 9): 0.5}


@pytest.mark.parametrize("every", [0, -1, 1.5, True, "63"])
def test_a_reset_period_is_a_positive_number_of_instants(every):
    with pytest.raises(ValueError, match="instantes medidos"):
        MemoryReset(every)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(regenerate=False, memory_reset=MemoryReset()), "Solo una regeneración"),
        (dict(regenerate=False, partitions=("evaluation",)), "Solo una regeneración"),
        (dict(partitions=("evaluation",), modality_ablation="mask_news"), "Solo una regeneración"),
        (dict(partitions=()), "tramos elegidos"),
        (dict(partitions=("evaluation", "calibration")), "tramos elegidos"),
        (dict(partitions=("evaluation", "evaluation")), "tramos elegidos"),
        (dict(partitions=("test",)), "tramos elegidos"),
        (dict(partitions=["evaluation"]), "tramos elegidos"),
        (dict(memory_reset=63), "MemoryReset"),
    ],
)
def test_resets_and_chosen_partitions_only_regenerate_the_anchor_window(
    base,  # noqa: F811
    tmp_path,
    options,
    message,
):
    options = dict(dict(regenerate=True), **options)
    anchor = base["root"] / "runs" / "mac_online"
    with pytest.raises(ValueError, match=message), unfused_attention():
        wf.carry_titans(anchor, base["view"], base["view"], tmp_path / "x", device="cpu", **options)
    assert not (tmp_path / "x").exists()
