"""Comprobación CUDA de la corrección B6 de las ventanas, sin pasos de optimizador.

Se ejecuta de forma explícita. Recorre los seis eventos de cuatro flujos de
`test_mars_titan_correction_parity` con el mismo padre Titans-MAC congelado en CPU y en
`cuda:0`, en FP32 y FP64, con las reglas delta y proximal y con la clave constante. A vive
en CPU y FP64 en los dos casos, así que solo cambia la predicción del núcleo. Un segundo
recorrido en `cuda:0` debe repetir el primero bit a bit. `MARS_TITAN_B6_CHECK_REPORT`
guarda las medidas en JSON.
"""

import json
import os
import resource
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_financial_session_associative import CONSTANT, DELTA
from test_financial_session_controls import four_flow_source as four_flow_source
from test_mars_titan_session_parity import PHASE, events

from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.financial_run import ChronologicalRecipe
from mars_titan.training.mars_titan_correction import CorrectionInference

PROXIMAL = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25, forgetting=0.01))
CORRECTIONS = dict(delta=DELTA, proximal=PROXIMAL, constant=CONSTANT)
TOLERANCES = {torch.float32: (5e-5, 3e-6), torch.float64: (2e-9, 2e-10)}


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def walk(source, correction, dtype, device, state=None):
    config = FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42)
    predictor = FinancialPredictor(config, dtype=dtype, device=device)
    if state is not None:
        predictor.load_state_dict(state)
    predictor.eval().requires_grad_(False)
    inference = CorrectionInference(
        predictor, ChronologicalRecipe(block_rows=4), correction, source["codec"], audit=True
    )
    rows = []
    metrics = inference._pass(SimpleNamespace(phase=PHASE), events(source), rows=rows)
    emitted = [entry[2:] for entry in inference.audit if entry[0] == "emitted"]
    return predictor, inference, metrics, emitted


def test_cuda_window_correction_matches_cpu_and_repeats(four_flow_source, tmp_path):
    started = time.perf_counter()
    assert torch.cuda.is_available(), "La comprobación requiere CUDA sin fallback"
    device = torch.device("cuda:0")
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for dtype in (torch.float32, torch.float64):
        rtol, atol = TOLERANCES[dtype]
        for name, correction in CORRECTIONS.items():
            parent, cpu, _, expected = walk(four_flow_source, correction, dtype, "cpu")
            state = parent.state_dict()
            model, gpu, metrics, actual = walk(four_flow_source, correction, dtype, device, state)
            _, again, _, repeated = walk(four_flow_source, correction, dtype, device, state)
            assert len(actual) == len(expected) == 20
            difference = np.subtract([e[2] for e in actual], [e[2] for e in expected])
            assert [entry[:2] for entry in actual] == [entry[:2] for entry in expected]
            np.testing.assert_allclose(
                [entry[2] for entry in actual],
                [entry[2] for entry in expected],
                rtol=rtol,
                atol=atol,
            )
            assert gpu.memory.writes == cpu.memory.writes == metrics["associative_writes"] == 16
            assert gpu.memory.matrix.device.type == "cpu"
            assert gpu.memory.matrix.dtype == torch.float64
            torch.testing.assert_close(gpu.memory.matrix, cpu.memory.matrix, rtol=rtol, atol=atol)
            assert repeated == actual and torch.equal(again.memory.matrix, gpu.memory.matrix)
            assert all(
                p.device == device and not p.requires_grad and p.grad is None
                for p in model.parameters()
            )
            after = model.state_dict()
            assert all(
                torch.equal(after[k].cpu(), v) for k, v in state.items() if torch.is_tensor(v)
            )
            records.append(
                dict(
                    dtype=str(dtype),
                    correction=name,
                    rule=correction.memory.rule,
                    key=correction.key,
                    emissions=len(actual),
                    writes=gpu.memory.writes,
                    rtol=rtol,
                    atol=atol,
                    max_emission_error=float(np.max(np.abs(difference))),
                    max_matrix_error=float(
                        torch.max(torch.abs(gpu.memory.matrix - cpu.memory.matrix))
                    ),
                    repeated_bit_for_bit=True,
                )
            )
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    torch.cuda.synchronize(device)
    report = dict(
        kind="mars_titan_correction_cuda_check",
        records=records,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        hardware=hardware.strip(),
        tf32=False,
        mha_fastpath=False,
        peak_torch_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_rng_unchanged=True,
        cuda_rng_unchanged=True,
        optimizer_steps=0,
        scope="Fixture de cuatro flujos, sin corpus real ni ajuste.",
    )
    destination = os.environ.get("MARS_TITAN_B6_CHECK_REPORT")
    path = Path(destination) if destination else tmp_path / "mars-titan-correction-cuda.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
