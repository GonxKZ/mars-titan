"""Comprobación CUDA del ajuste del núcleo con la penalización C, sin pasos de optimizador.

Se ejecuta de forma explícita. Compara un recorrido de ajuste con su validación en CPU y en
el dispositivo indicado, con los mismos parámetros iniciales, la misma base de C y el
optimizador que solo registra gradientes. Cubre B (C disabled) y B+C (penalty), sin
acumulación y con `accumulation_rows=2`.
`MARS_TITAN_CM_CONTROL_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
`MARS_TITAN_CM_CONTROL_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.local_control import MACProjectionConfig
from mars_titan.training.financial_run import ChronologicalRecipe, ChronologicalTrainer
from tests.training.test_financial_run import (
    SELECTION,
    RecordingOptimizer,
    corpus,
    entries,
    named_records,
)

DEVICE = os.environ.get("MARS_TITAN_CM_CONTROL_CHECK_DEVICE", "cuda:0")
TOLERANCES = {torch.float32: (2e-4, 2e-6), torch.float64: (1e-8, 1e-10)}
CONTROL = dict(rank=2, frequency=2, seed=91, grid_size=16, threshold=0.0, max_flows=1)
pytestmark = pytest.mark.skipif(
    DEVICE == "cuda:0" and not torch.cuda.is_available(), reason="Falta cuda:0"
)


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def trainer(streams, output, *, mode, dtype, device, rows):
    weight = 0.5 if mode == "penalty" else 0.0
    config = FinancialConfig(
        streams["train"].specification(), variant="mac_online", hidden_size=32, seed=42
    )
    predictor = FinancialPredictor(
        config,
        local_control=MACProjectionConfig(mode=mode, weight=weight, **CONTROL),
        dtype=dtype,
        device=device,
    )
    recipe = ChronologicalRecipe(
        truncation=3, epochs=1, block_rows=2, selection=SELECTION, accumulation_rows=rows
    )
    return ChronologicalTrainer(
        predictor,
        recipe,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


@pytest.mark.parametrize("rows", [None, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["disabled", "penalty"])
def test_device_pass_matches_cpu_without_optimizer_steps(tmp_path, mode, dtype, rows):
    _, streams = corpus(tmp_path / "corpus")
    rtol, atol = TOLERANCES[dtype]
    cuda = DEVICE.startswith("cuda")
    engines, seconds, reports = {}, {}, {}
    for name, device in (("cpu", "cpu"), ("device", DEVICE)):
        if cuda and name == "device":
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        engines[name] = trainer(
            streams, tmp_path / name, mode=mode, dtype=dtype, device=device, rows=rows
        )
        reports[name] = engines[name].run()
        if cuda and name == "device":
            torch.cuda.synchronize(0)
        seconds[name] = time.perf_counter() - start
    cpu, device = engines["cpu"], engines["device"]
    assert device.identity["local_control"] == cpu.identity["local_control"]
    left, right = entries(cpu.audit, "prediction"), entries(device.audit, "prediction")
    assert [e[:4] for e in left] == [e[:4] for e in right]
    torch.testing.assert_close(
        torch.tensor([e[4] for e in right], dtype=torch.float64),
        torch.tensor([e[4] for e in left], dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )
    assert entries(cpu.audit, "update") == entries(device.audit, "update")
    records = named_records(cpu), named_records(device)
    assert len(records[0]) == len(records[1]) > 3
    for expected, actual in zip(*records, strict=True):
        for name, value in expected.items():
            if value is None:
                assert actual[name] is None
            else:
                torch.testing.assert_close(actual[name].cpu(), value, rtol=rtol, atol=atol)
    train = [report["history"][-1]["train"] for report in reports.values()]
    for key in ("updates", "labels_in_loss", "control_groups", "control_flows"):
        assert train[0].get(key) == train[1].get(key)
    destination = os.environ.get("MARS_TITAN_CM_CONTROL_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                device=DEVICE,
                dtype=str(dtype),
                mode=mode,
                accumulation_rows=rows,
                updates=len(records[0]),
                control_groups=train[1].get("control_groups"),
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
