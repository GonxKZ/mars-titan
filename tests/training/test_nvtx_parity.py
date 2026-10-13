"""Los rangos NVTX no cambian ningún bit de los recorridos de Titans y del lector MARS-TITAN.

Cada prueba repite el mismo recorrido con los rangos apagados y encendidos, con el optimizador
registrador que no modifica pesos. Predicciones, gradientes, pasos y etiquetas deben coincidir
exactamente y los rangos deben quedar cerrados en cada hilo.
"""

import pytest
import torch

from mars_titan import nvtx_ranges
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training import mars_titan_run as mt
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.tooling.test_nvtx_ranges import Recorder
from tests.training.test_financial_run import SELECTION, RecordingOptimizer, entries, named_records
from tests.training.test_financial_run import shared as shared
from tests.training.test_mars_titan_run import build, gradients, recipe
from tests.training.test_mars_titan_run import native as native

CONTROL = dict(rank=2, frequency=2, seed=91, grid_size=16, threshold=0.0, max_flows=1)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def ranged(monkeypatch, enabled):
    """Encender o apagar los rangos y devolver el registro de marcas."""
    recorded = Recorder(forward=torch.version.cuda is not None)
    monkeypatch.setattr(nvtx_ranges, "ENABLED", enabled)
    monkeypatch.setattr(nvtx_ranges, "_nvtx", lambda: recorded)
    return recorded


def titans(streams, output, *, mode, rows, device):
    weight = 0.5 if mode == "penalty" else 0.0
    config = FinancialConfig(
        streams["train"].specification(), variant="mac_online", hidden_size=32, seed=42
    )
    predictor = FinancialPredictor(
        config,
        local_control=MACProjectionConfig(mode=mode, weight=weight, **CONTROL),
        dtype=torch.float64,
        device=device,
    )
    plan = ChronologicalRecipe(
        truncation=3, epochs=1, block_rows=2, selection=SELECTION, accumulation_rows=rows
    )
    return ChronologicalTrainer(
        predictor,
        plan,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


def same_records(left, right):
    assert len(left) == len(right) > 0
    for expected, actual in zip(left, right, strict=True):
        assert expected.keys() == actual.keys()
        for name, value in expected.items():
            if value is None:
                assert actual[name] is None
            else:
                assert torch.equal(value, actual[name]), name


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Falta cuda:0"),
        ),
    ],
)
@pytest.mark.parametrize("rows", [None, 2])
@pytest.mark.parametrize("mode", ["disabled", "penalty"])
def test_titans_pass_is_bit_identical_with_ranges(
    shared, tmp_path, monkeypatch, mode, rows, device
):
    _, streams = shared
    engines, reports, marks = {}, {}, {}
    for enabled in (False, True):
        marks[enabled] = ranged(monkeypatch, enabled)
        engine = titans(streams, tmp_path / str(enabled), mode=mode, rows=rows, device=device)
        reports[enabled] = engine.run()
        engines[enabled] = engine
    off, on = engines[False], engines[True]
    assert reports[False]["status"] == reports[True]["status"] == "completed"
    assert off.audit == on.audit
    assert off.optimizer.calls == on.optimizer.calls > 0
    same_records(named_records(off), named_records(on))
    assert reports[False]["history"] == reports[True]["history"]
    assert reports[False]["selection"] == reports[True]["selection"]
    assert marks[False].events == []
    names = marks[True].names()
    assert marks[True].balanced()
    assert names.count("titans.optimizer") == on.optimizer.calls
    assert names.count("titans.update") >= on.optimizer.calls
    assert names.count("titans.backward") == on.optimizer.calls
    assert names.count("titans.penalty") > 0 if mode == "penalty" else "titans.penalty" not in names
    assert {"titans.read", "titans.observe", "titans.prepare", "reader.decode"} <= set(names)
    assert entries(on.audit, "prediction")


@pytest.mark.parametrize("admission", ["m0", "m1", "m3"])
@pytest.mark.usefixtures("learning_doubles")
def test_readout_pass_is_bit_identical_with_ranges(
    shared, tmp_path, monkeypatch, native, admission
):
    _, streams = shared
    engines, reports, marks = {}, {}, {}
    for enabled in (False, True):
        marks[enabled] = ranged(monkeypatch, enabled)
        engine = build(
            streams,
            tmp_path / str(enabled),
            native,
            admission=admission,
            plan=recipe(epochs=2),
        )
        before = mt._parameters_digest(engine.readout)
        reports[enabled] = engine.run()
        assert mt._parameters_digest(engine.readout) == before
        engines[enabled] = engine
    off, on = engines[False], engines[True]
    assert reports[False]["status"] == reports[True]["status"] == "completed"
    assert off.audit == on.audit
    assert off.optimizer.calls == on.optimizer.calls > 0
    same_records(gradients(off), gradients(on))
    assert reports[False]["history"] == reports[True]["history"]
    assert marks[False].events == []
    names = marks[True].names()
    assert marks[True].balanced()
    assert names.count("readout.optimizer") == on.optimizer.calls
    assert names.count("readout.replay") == on.optimizer.calls
    assert {"readout.read", "readout.observe", "readout.admit", "reader.decode"} <= set(names)
    assert ("readout.snapshot" in names) is (admission != "m0")
