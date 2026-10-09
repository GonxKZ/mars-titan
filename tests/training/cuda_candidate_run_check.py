"""Comprobación CUDA del entrenador de la GRU candidata, sin pasos de optimizador.

Se ejecuta de forma explícita. Compara un recorrido de ajuste y una validación en CPU y en
el dispositivo indicado, con los mismos parámetros iniciales y el optimizador que solo
registra gradientes. `MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE=cpu` ensaya la lógica sin GPU
y no acredita CUDA. `MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from tests.training.test_candidate_run import SELECTION, entries, named_records, trainer
from tests.training.test_financial_run import corpus

DEVICE = os.environ.get("MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE", "cuda:0")
TOLERANCES = {torch.float32: (2e-4, 2e-6), torch.float64: (1e-8, 1e-10)}
pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    or (DEVICE == "cuda:0" and not torch.cuda.is_available()),
    reason="Faltan el enlace nativo o cuda:0",
)


def adapters(streams, dtype):
    from mars_titan.models.candidate.input_adapter import CandidateInputAdapter

    specification = streams["train"].specification()
    cpu = CandidateInputAdapter(specification, dtype=dtype)
    device = CandidateInputAdapter.restore(cpu.export_state(), specification, device=DEVICE)
    return cpu, device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("refinements", [1, 4])
def test_device_pass_matches_cpu_without_optimizer_steps(tmp_path, dtype, refinements):
    _, streams = corpus(tmp_path / "corpus")
    rtol, atol = TOLERANCES[dtype]
    cuda = DEVICE.startswith("cuda")
    engines, seconds = {}, {}
    for name, model in zip(("cpu", "device"), adapters(streams, dtype), strict=True):
        if cuda and name == "device":
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        engines[name] = trainer(
            streams, tmp_path / name, model=model, refinements=refinements, selection=SELECTION
        )
        engines[name].run()
        if cuda and name == "device":
            torch.cuda.synchronize(0)
        seconds[name] = time.perf_counter() - start
    cpu, device = engines["cpu"], engines["device"]
    left = entries(cpu.audit, "prediction")
    right = entries(device.audit, "prediction")
    assert [e[:4] + e[5:] for e in left] == [e[:4] + e[5:] for e in right]
    torch.testing.assert_close(
        torch.tensor([e[4] for e in right], dtype=torch.float64),
        torch.tensor([e[4] for e in left], dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )
    assert entries(cpu.audit, "admit") == entries(device.audit, "admit")
    records = named_records(cpu), named_records(device)
    assert len(records[0]) == len(records[1]) > 3
    for expected, actual in zip(*records, strict=True):
        for name, value in expected.items():
            if value is None:
                assert actual[name] is None
            else:
                torch.testing.assert_close(actual[name].cpu(), value, rtol=rtol, atol=atol)
    assert device.model.parameter_fingerprint() == cpu.model.parameter_fingerprint()
    destination = os.environ.get("MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                device=DEVICE,
                dtype=str(dtype),
                refinements=refinements,
                updates=len(records[0]),
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
