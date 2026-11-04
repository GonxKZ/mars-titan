"""Recorrido cronológico con la fiabilidad madura de #791, sin pasos de optimizador."""

import math

import pytest
import torch

from mars_titan.memory.mature_reliability import ReliabilitySignal
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.reliability_gates import MatureReliability
from mars_titan.training import financial_run
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

MIXING = [[1.5, -1.0, 0.5], [-0.8, 1.2, 2.0]]
SOURCES = [
    dict(),
    dict(source="shuffled_in_time", seed=3),
    dict(source="titans_surprise"),
    dict(independent=True, seed=2),
]
IDS = ["mature", "shuffled", "surprise", "independent"]


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    return corpus(tmp_path_factory.mktemp("mature-reliability"))[1]


def predictor(streams, reliability=None, mixing=None):
    specification = streams["train"].specification()
    config = FinancialConfig(
        specification, variant="mac_online", hidden_size=32, mature_reliability=reliability
    )
    model = FinancialPredictor(config, dtype=torch.float64)
    if mixing is not None:
        with torch.no_grad():
            model.reliability_gates.mixing.copy_(torch.tensor(mixing, dtype=torch.float64))
        model._seal_parameters()
    return model


def assert_same_gradients(records, reference):
    assert len(records) == len(reference) > 2
    for left, right in zip(records, reference, strict=True):
        assert left.keys() == right.keys()
        for name, value in left.items():
            if value is None:
                assert right[name] is None, name
            else:
                torch.testing.assert_close(value, right[name], rtol=0, atol=0)


@pytest.mark.parametrize("options", SOURCES, ids=IDS)
def test_a_new_signal_reproduces_the_reference_run_and_reaches_its_mixing(
    shared, tmp_path, options
):
    control = trainer(shared, tmp_path / "control", model=predictor(shared))
    control.run()
    model = predictor(shared, MatureReliability(**options))
    engine = trainer(shared, tmp_path / "gated", model=model)
    assert engine.run()["status"] == "completed"
    assert engine.audit == control.audit
    records = named_records(engine)
    mixing = [record.pop("reliability_gates.mixing") for record in records]
    assert_same_gradients(records, named_records(control))
    assert any(value is not None and torch.count_nonzero(value) for value in mixing)
    assert engine.predictor._parameter_id == model._parameter_id


def recorded_calls(monkeypatch):
    calls = []
    labels, prepare, close = (
        ChronologicalInference._labels,
        FinancialPredictor.prepare,
        ChronologicalInference._close,
    )

    def matured(self, run, event, *, train):
        calls.append(("labels", event.at, list(event.labels)))
        return labels(self, run, event, train=train)

    def prepared(self, batch, state, **options):
        signal = options["reliability"].clone()
        result = prepare(self, batch, state, **options)
        emitted = result.point_predictions.detach().tolist()
        calls.append(("prepare", batch.flow_ids, batch.prediction_at, signal, emitted))
        return result

    def closed(self, run):
        calls.append(("close",))
        return close(self, run)

    monkeypatch.setattr(ChronologicalInference, "_labels", matured)
    monkeypatch.setattr(FinancialPredictor, "prepare", prepared)
    monkeypatch.setattr(ChronologicalInference, "_close", closed)
    return calls


def replayed_signal(calls, config):
    """Señal esperada en cada preparación, reconstruida aparte con la fórmula de #791.

    Cada etiqueta se compara con la última predicción preparada para su flujo e instante, que
    es la emitida en el recorrido en curso.
    """
    decays = [math.exp(-1.0 / value) for value in config.time_constants]
    markets, expected, latest, issued = {}, [], None, {}
    for call in calls:
        if call[0] == "close":
            markets = {}
        elif call[0] == "labels":
            latest = call[1]
            groups = {}
            for flow, decision_at, outcome in call[2]:
                # Cada etiqueta llega después de su decisión y se compara con lo emitido.
                assert decision_at < latest
                hit = math.copysign(1, issued[flow, decision_at]) * math.copysign(1, outcome)
                hit = 0.0 if issued[flow, decision_at] == 0 or outcome == 0 else hit
                groups.setdefault(flow.split("/", 1)[0], []).append(hit)
            for market, hits in sorted(groups.items()):
                values = markets.setdefault(market, [0.0] * len(decays))
                score = math.fsum(hits) / len(hits)
                for index, decay in enumerate(decays):
                    values[index] = decay * values[index] + (1.0 - decay) * score
        else:
            _, flows, moments, signal, emitted = call
            issued.update(zip(zip(flows, moments, strict=True), emitted, strict=True))
            assert latest is None or all(moment >= latest for moment in moments)
            rows = [
                [list(markets.get(flow.split("/", 1)[0], [0.0] * len(decays)))] * 2
                for flow in flows
            ]
            expected.append((signal, torch.tensor(rows, dtype=torch.float64)))
    return expected


def test_each_decision_reads_the_labels_matured_before_it_and_nothing_later(
    shared, tmp_path, monkeypatch
):
    calls = recorded_calls(monkeypatch)
    config = MatureReliability()
    engine = trainer(shared, tmp_path / "run", model=predictor(shared, config, MIXING))
    engine.run()
    expected = replayed_signal(calls, config)
    # El ajuste y cada validación son recorridos distintos, y cada uno empieza neutral.
    assert len(expected) > 4 and sum(call[0] == "close" for call in calls) >= 2
    for signal, reference in expected:
        assert torch.equal(signal, reference)
    assert any(torch.count_nonzero(signal) for signal, _ in expected)
    assert not torch.count_nonzero(expected[0][0])


def test_the_signal_changes_predictions_only_after_a_label_matures(tmp_path):
    cut = decision("2022-12-09")
    runs = {}
    for name, perturb, reliability in (
        ("control", None, None),
        ("base", None, MatureReliability()),
        ("future", cut, MatureReliability()),
    ):
        _, streams = corpus(tmp_path / name, perturb_after=perturb)
        model = predictor(streams, reliability, None if reliability is None else MIXING)
        runs[name] = trainer(streams, tmp_path / name / "run", model=model, truncation=2)
        runs[name].run()
    control, base, future = (entries(runs[n].audit, "prediction") for n in runs)
    first = min(e[4] for e in entries(runs["base"].audit, "label"))
    # Hasta la primera etiqueta madura la señal es neutral y las predicciones coinciden.
    assert [e for e in base if e[3] < first] == [e for e in control if e[3] < first]
    assert [e for e in base if e[3] >= first] != [e for e in control if e[3] >= first]
    # Alterar lo posterior a `cut` no cambia nada anterior, tampoco los gradientes.
    assert [e for e in base if e[3] <= cut] == [e for e in future if e[3] <= cut]
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
    config = MatureReliability(source="shuffled_in_time", seed=8)

    def engine(name):
        return trainer(shared, tmp_path / name, model=predictor(shared, config, MIXING), **options)

    continuous = engine("continuous")
    expected = continuous.run()
    first = engine("interrupted")
    stop = StopAtStep(15)
    stop.trainer = first
    assert first.run(stop=stop)["status"] == "paused"
    state = load_training_state(
        tmp_path / "interrupted/checkpoints", expected_identity=first.identity
    )
    saved = state["run"]["reliability"]
    assert saved["configuration"] == config.identity()
    assert saved["markets"]["US"][0]["updates"] > 0
    second = engine("interrupted")
    resumed = second.run(resume=True)
    assert resumed["status"] == "completed"
    assert first.audit + second.audit == continuous.audit
    joined = named_records(first) + named_records(second)
    assert_same_gradients(joined, named_records(continuous))
    assert resumed["history"] == expected["history"]


def test_replay_by_flow_blocks_keeps_the_signal_of_each_emission(shared, tmp_path):
    records = []
    for name, rows in (("whole", None), ("blocks", 1)):
        model = predictor(shared, MatureReliability(), MIXING)
        engine = trainer(shared, tmp_path / name, model=model, accumulation_rows=rows)
        engine.run()
        records.append(named_records(engine))
    whole, blocks = records
    assert len(whole) == len(blocks) > 2
    for left, right in zip(whole, blocks, strict=True):
        for name, value in left.items():
            if value is None or right[name] is None:
                assert value is right[name], name
            else:
                torch.testing.assert_close(value, right[name], rtol=1e-10, atol=1e-14)
    assert any(torch.count_nonzero(record["reliability_gates.mixing"]) for record in whole)


def test_the_surprise_control_observes_only_the_emitted_rows(shared, tmp_path, monkeypatch):
    observed, observe = [], ReliabilitySignal.observe

    def spy(self, flows, losses):
        observed.extend(zip(flows, losses, strict=True))
        return observe(self, flows, losses)

    monkeypatch.setattr(ReliabilitySignal, "observe", spy)
    model = predictor(shared, MatureReliability(source="titans_surprise"), MIXING)
    engine = trainer(shared, tmp_path / "run", model=model)
    engine.run()
    emitted = entries(engine.audit, "prediction")
    assert len(observed) == len(emitted) > 0
    assert [flow for flow, _ in observed] == [entry[2] for entry in emitted]
    assert all(loss >= 0 for _, loss in observed)


def test_the_checkpoint_and_the_code_identity_follow_the_declared_signal(shared, tmp_path):
    engine = trainer(shared, tmp_path / "gated", model=predictor(shared, MatureReliability()))
    control = trainer(shared, tmp_path / "control", model=predictor(shared))
    payload = engine._export(financial_run._Pass())
    assert payload["reliability"] is None
    plain = control._export(financial_run._Pass())
    assert "reliability" not in plain
    for target, saved in ((control, payload), (engine, plain)):
        with pytest.raises(ValueError, match="señal de fiabilidad"):
            target._restore(saved)
    assert engine._restore(payload).reliability is None
    # Las puertas están en la implementación de todo predictor, porque `financial` importa
    # siempre su configuración. El acumulador de la señal solo entra cuando se declara.
    assert "mars_titan.models.titans.reliability_gates" in control._code()
    assert "mars_titan.memory.mature_reliability" in engine._code()
    assert "mars_titan.memory.mature_reliability" not in control._code()
