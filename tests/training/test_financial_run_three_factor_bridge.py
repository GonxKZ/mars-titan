"""Recorrido cronológico con el puente de tres factores (#784, P1), sin pasos de optimizador."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mars_titan.memory.mature_reliability import ReliabilitySignal
from mars_titan.memory.three_factor_bridge import BridgePlasticity
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.holographic_network import HolographicNetworkConfig
from mars_titan.models.titans.plastic_bridge import (
    MATURE_DIRECTIONAL_HIT,
    MEAN_THIRD_FACTOR,
    ThreeFactorBridge,
)
from mars_titan.models.titans.reliability_gates import MatureReliability
from mars_titan.training import financial_run, titans_walk_forward
from mars_titan.training.checkpoints import load_training_state
from mars_titan.training.financial_run import ChronologicalInference
from tests.training.chronological_fixture import decision
from tests.training.test_financial_run import (
    StopAtStep,
    corpus,
    entries,
    named_records,
    trainer,
)

H1 = replace(HolographicNetworkConfig(), areas=5)
# Una tasa alta hace visibles los cambios de los pesos en los pocos eventos del corpus.
P1 = ThreeFactorBridge(learning_rate=0.5)
COMMON = ThreeFactorBridge(source=MATURE_DIRECTIONAL_HIT, time_constant=None, learning_rate=0.5)
BRIDGES = [P1, replace(P1, kind=MEAN_THIRD_FACTOR), COMMON]
IDS = ["error_gap", "mean_control", "common_signal"]
MODULE = "mars_titan.memory.three_factor_bridge"


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("three-factor-bridge"))[1]


def predictor(streams, bridge=None, *, common=False):
    """H1 con la lectura distinta de cero, para que los pesos del puente cambien la salida.

    Con `common`, o con la fuente común de P1, el predictor declara la señal de #791.
    """
    common = common or (bridge is not None and bridge.source == MATURE_DIRECTIONAL_HIT)
    config = FinancialConfig(
        streams["train"].specification(),
        variant="mac_online",
        hidden_size=32,
        holographic_network=H1,
        three_factor_bridge=bridge,
        mature_reliability=MatureReliability() if common else None,
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    with torch.no_grad():
        model.holographic_network.readout.weight.fill_(0.05)
    model._seal_parameters()
    return model


def reference(streams, bridge):
    """El mismo predictor sin el puente: H1, y la señal de #791 con la fuente común."""
    return predictor(streams, common=bridge.source == MATURE_DIRECTIONAL_HIT)


def moved_weights(run):
    return run.bridge is not None and any(
        state["weights"].any() for state in run.bridge._markets.values()
    )


def assert_same_gradients(records, expected, **tolerance):
    assert len(records) == len(expected) > 2
    for left, right in zip(records, expected, strict=True):
        assert left.keys() == right.keys()
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            else:
                options = tolerance or dict(rtol=0, atol=0)
                torch.testing.assert_close(value, right[name], **options)


@pytest.mark.parametrize("bridge", BRIDGES, ids=IDS)
def test_the_bridge_reproduces_its_reference_until_its_weights_move(
    shared, tmp_path, monkeypatch, bridge
):
    control = trainer(shared, tmp_path / "control", model=reference(shared, bridge))
    control.run()
    moves, labels = [], ChronologicalInference._labels

    def matured(self, run, event, *, train):
        result = labels(self, run, event, train=train)
        moves.append((train, event.at, moved_weights(run)))
        return result

    monkeypatch.setattr(ChronologicalInference, "_labels", matured)
    model = predictor(shared, bridge)
    engine = trainer(shared, tmp_path / "bridge", model=model)
    assert engine.run()["status"] == "completed"
    assert engine.predictor._parameter_id == model._parameter_id
    first = min(at for train, at, moved in moves if train and moved)
    base, plain = (entries(e.audit, "prediction", "train") for e in (engine, control))
    # Con los pesos a cero la salida es la de la referencia.
    assert [e for e in base if e[3] < first] == [e for e in plain if e[3] < first]
    assert [e for e in base if e[3] >= first] != [e for e in plain if e[3] >= first]
    # Los pesos se mueven antes de la primera actualización, así que los gradientes ya no
    # coinciden. El puente tampoco cambia el calendario ni las etiquetas de cada paso.
    for kind in ("update", "label"):
        assert entries(engine.audit, kind) == entries(control.audit, kind)
    assert len(named_records(engine)) == len(named_records(control)) > 2


def recorded_calls(monkeypatch):
    """Preparaciones, trazas guardadas, etiquetas y cierres del recorrido, en su orden."""
    calls = []
    prepare, labels, close = (
        FinancialPredictor.prepare,
        ChronologicalInference._labels,
        ChronologicalInference._close,
    )
    hold, settle, signal = BridgePlasticity.hold, BridgePlasticity.settle, ReliabilitySignal.settle

    def prepared(self, batch, state, **options):
        result = prepare(self, batch, state, **options)
        emitted = result.point_predictions.detach().tolist()
        calls.append(
            (
                "prepare",
                batch.flow_ids,
                batch.prediction_at,
                options["bridge_bias"].clone(),
                result.bridge_traces.clone(),
                emitted,
            )
        )
        return result

    def matured(self, run, event, *, train):
        calls.append(("labels", event.at))
        return labels(self, run, event, train=train)

    def held(self, keys, traces):
        calls.append(("hold", list(keys), np.array(traces, copy=True)))
        return hold(self, keys, traces)

    def settled(self, rows, expectations=None):
        calls.append(("settle", list(rows), copy.deepcopy(expectations)))
        return settle(self, rows, expectations)

    def signalled(self, rows):
        calls.append(("signal", list(rows)))
        return signal(self, rows)

    def closed(self, run):
        calls.append(("close",))
        return close(self, run)

    monkeypatch.setattr(FinancialPredictor, "prepare", prepared)
    monkeypatch.setattr(ChronologicalInference, "_labels", matured)
    monkeypatch.setattr(ChronologicalInference, "_close", closed)
    monkeypatch.setattr(BridgePlasticity, "hold", held)
    monkeypatch.setattr(BridgePlasticity, "settle", settled)
    monkeypatch.setattr(ReliabilitySignal, "settle", signalled)
    return calls


def replayed(calls, bridge, network):
    """Repetir el recorrido con otra instancia de la regla y comprobar cada lectura.

    La regla ya se compara con una referencia escrita aparte en `tests/memory`. Aquí se
    comprueba el recorrido: los pesos que lee cada decisión, las trazas que guarda, las
    predicciones con las que madura cada etiqueta y, con la fuente común, que el acierto
    esperado se lee antes de que la señal reciba las etiquetas del evento.
    """

    def fresh():
        return (
            BridgePlasticity(bridge, network.plastic_edges, network.config.catalog_areas),
            ReliabilitySignal(MatureReliability()),
        )

    plastic, signal = fresh()
    traces, issued, latest, settled, read = {}, {}, None, False, 0
    for call in calls:
        kind = call[0]
        if kind == "close":
            plastic, signal = fresh()
        elif kind == "prepare":
            _, flows, moments, bias, trace, emitted = call
            assert latest is None or all(moment >= latest for moment in moments)
            expected = torch.tensor(plastic.biases(flows), dtype=bias.dtype)
            assert torch.equal(bias, expected)
            read += bool(torch.count_nonzero(bias))
            for row, key in enumerate(zip(flows, moments, strict=True)):
                traces[key], issued[key] = trace[row], emitted[row]
        elif kind == "hold":
            _, keys, held = call
            # Cada traza guardada es la que devolvió la red al emitir su fila.
            assert np.array_equal(held, torch.stack([traces[key] for key in keys]).numpy())
            plastic.hold(keys, held)
        elif kind == "labels":
            latest, settled = call[1], False
        elif kind == "settle":
            _, rows, expectations = call
            assert not settled
            for flow, decision_at, value, _ in rows:
                # Cada etiqueta llega después de su decisión y madura con lo emitido.
                assert decision_at < latest and value == issued[flow, decision_at]
            if bridge.source == MATURE_DIRECTIONAL_HIT:
                markets = {flow.split("/", 1)[0] for flow, *_ in rows}
                assert expectations == signal.expected_hits(markets)
            else:
                assert expectations is None
            plastic.settle(rows, expectations)
        else:
            settled = True
            signal.settle(call[1])
    return read


@pytest.mark.parametrize("bridge", BRIDGES, ids=IDS)
def test_each_decision_reads_the_weights_of_the_labels_matured_before_it(
    shared, tmp_path, monkeypatch, bridge
):
    calls = recorded_calls(monkeypatch)
    engine = trainer(shared, tmp_path / "run", model=predictor(shared, bridge))
    engine.run()
    # La repetición usa las clases originales, sin registrar sus propias llamadas.
    monkeypatch.undo()
    # El ajuste y cada validación son recorridos distintos, y cada uno empieza sin pesos.
    assert sum(call[0] == "close" for call in calls) >= 2
    assert sum(call[0] == "hold" for call in calls) > 4
    assert replayed(calls, bridge, engine.predictor.holographic_network) > 0
    first = next(call for call in calls if call[0] == "prepare")
    assert not torch.count_nonzero(first[3])


@pytest.mark.parametrize("bridge", [P1, COMMON], ids=["error_gap", "common_signal"])
def test_altering_the_data_after_a_cut_changes_nothing_before_it(tmp_path, bridge):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb in (("base", None), ("future", cut)):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        model = predictor(streams, bridge)
        runs[name] = trainer(streams, tmp_path / name / "run", model=model, truncation=2)
        runs[name].run()
    base, future = (entries(runs[n].audit, "prediction") for n in runs)
    assert [e for e in base if e[3] <= cut] == [e for e in future if e[3] <= cut]
    assert [e for e in base if e[3] > cut] != [e for e in future if e[3] > cut]
    steps = [u[2] for u in entries(runs["base"].audit, "update") if u[1] <= cut]
    assert steps
    for left, right in zip(
        named_records(runs["base"])[: len(steps)], named_records(runs["future"]), strict=False
    ):
        assert left.keys() == right.keys()
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_resume_after_interruption_reproduces_the_continuous_run(shared, tmp_path):
    options = dict(epochs=2, fixed=True, checkpoint_updates=2)

    def engine(name):
        return trainer(shared, tmp_path / name, model=predictor(shared, P1), **options)

    continuous = engine("continuous")
    expected = continuous.run()
    first = engine("interrupted")
    stop = StopAtStep(15)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    saved = state["run"]["bridge"]
    assert saved["configuration"] == P1.identity()
    assert any(market["weights"].any() for market in saved["markets"].values())
    pending = saved["pending"]
    assert pending["flows"] and pending["traces"].shape == (len(pending["flows"]), 10)
    # Las trazas guardadas son exactamente las de las predicciones pendientes.
    first._restore(copy.deepcopy(state["run"]))
    for change in (
        lambda run: run["bridge"]["pending"].update(
            flows=pending["flows"][1:],
            decision_at=pending["decision_at"][1:],
            traces=pending["traces"][1:],
        ),
        lambda run: run.update(bridge=None),
    ):
        broken = copy.deepcopy(state["run"])
        change(broken)
        with pytest.raises(ValueError, match="no son las de las predicciones pendientes"):
            first._restore(broken)
    second = engine("interrupted")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert_same_gradients(joined, named_records(continuous))
    assert resumed["history"] == expected["history"]


@pytest.mark.parametrize("bridge", [P1, COMMON], ids=["error_gap", "common_signal"])
def test_replay_by_flow_blocks_keeps_the_weights_of_each_emission(shared, tmp_path, bridge):
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        model = predictor(shared, bridge)
        engine = trainer(shared, tmp_path / name, model=model, accumulation_rows=rows)
        engine.run()
        records.append(named_records(engine))
    whole, blocks = records
    assert_same_gradients(whole, blocks, rtol=1e-10, atol=1e-14)


def test_the_checkpoint_and_the_code_identity_follow_the_declared_bridge(shared, tmp_path):
    engine = trainer(shared, tmp_path / "bridge", model=predictor(shared, P1))
    control = trainer(shared, tmp_path / "control", model=predictor(shared))
    payload = engine._export(financial_run._Pass())
    assert payload["bridge"] is None
    plain = control._export(financial_run._Pass())
    assert "bridge" not in plain
    for target, saved in ((control, payload), (engine, plain)):
        with pytest.raises(ValueError, match="puente de tres factores"):
            target._restore(saved)
    assert engine._restore(payload).bridge is None
    # La configuración está en la implementación de todo predictor. La regla solo entra
    # en la identidad del código cuando se declara.
    assert "mars_titan.models.titans.plastic_bridge" in control._code()
    assert MODULE in engine._code() and MODULE not in control._code()
    # El recorrido de Titans guarda las dos en su identidad.
    assert {"mars_titan.models.titans.plastic_bridge", MODULE} <= set(titans_walk_forward._CODE)
