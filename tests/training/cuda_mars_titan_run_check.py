"""Comprobación CUDA del lector episódico de MARS-TITAN, sin pasos de optimizador.

Se ejecuta de forma explícita. Compara un recorrido de ajuste con su validación en CPU y en
el dispositivo indicado, con el mismo padre Titans-MAC congelado, los mismos parámetros
iniciales del lector y el optimizador que solo registra gradientes. Cubre M1 con K = 1 y
K = 4 en sus dos modos de selección de episodios.
`MARS_TITAN_MARS_RUN_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
`MARS_TITAN_MARS_RUN_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from mars_titan.memory.episodic_codec import FrozenEpisodeCodec
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training import mars_titan_run as mt
from tests.training.test_financial_run import RecordingOptimizer, corpus, entries
from tests.training.test_mars_titan_run import gradients, recipe

DEVICE = os.environ.get("MARS_TITAN_MARS_RUN_CHECK_DEVICE", "cuda:0")
TOLERANCES = {torch.float32: (2e-4, 2e-6), torch.float64: (1e-8, 1e-10)}
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
        or (DEVICE == "cuda:0" and not torch.cuda.is_available()),
        reason="Faltan el enlace nativo o cuda:0",
    ),
    pytest.mark.usefixtures("learning_doubles"),
]


@pytest.fixture(autouse=True)
def explicit_fastpath():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def engine(streams, output, *, dtype, device, refinements, episodes, source=None):
    specification = streams["train"].specification()
    parent = FinancialPredictor(
        FinancialConfig(
            specification, variant="mac_online", hidden_size=32, seed=42, head="quantile_head_v1"
        ),
        dtype=dtype,
        device=device,
    )
    codec = FrozenEpisodeCodec(specification)
    readout = EpisodicReadout(
        EpisodicReadoutConfig(
            codec.fingerprint(),
            hidden_size=32,
            refinements=refinements,
            max_working_bytes=128 * 1024**2,
            seed=42,
            episode_selection=episodes,
        ),
        dtype=dtype,
        device=device,
    )
    if source is not None:
        # El dispositivo parte exactamente de los parámetros del recorrido en CPU.
        parent.load_state_dict(source.predictor.state_dict())
        readout.load_state_dict(source.readout.state_dict())
    plan = recipe()
    return mt.ReadoutTrainer(
        parent.eval().requires_grad_(False),
        readout,
        plan,
        admission="m1",
        retention=mt.retention_config(plan, "m1"),
        native=load_native(),
        codec=codec,
        train=streams["train"],
        validation=streams["validation"],
        output=output,
        world="cuda_check",
        fold="0",
        optimizer_factory=RecordingOptimizer,
        audit=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("refinements", "episodes"), [(1, "per_step"), (4, "per_step"), (4, "first_read")]
)
def test_device_pass_matches_cpu_without_optimizer_steps(tmp_path, dtype, refinements, episodes):
    _, streams = corpus(tmp_path / "corpus")
    rtol, atol = TOLERANCES[dtype]
    cuda = DEVICE.startswith("cuda")
    options = dict(dtype=dtype, refinements=refinements, episodes=episodes)
    engines, seconds = {}, {}
    for name, device in (("cpu", "cpu"), ("device", DEVICE)):
        if cuda and name == "device":
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        engines[name] = engine(
            streams, tmp_path / name, device=device, source=engines.get("cpu"), **options
        )
        before = mt._parameters_digest(engines[name].readout)
        engines[name].run()
        assert mt._parameters_digest(engines[name].readout) == before
        if cuda and name == "device":
            torch.cuda.synchronize(0)
        seconds[name] = time.perf_counter() - start
    cpu, device = engines["cpu"], engines["device"]
    left, right = entries(cpu.audit, "prediction"), entries(device.audit, "prediction")
    assert [e[:4] + e[5:] for e in left] == [e[:4] + e[5:] for e in right]
    torch.testing.assert_close(
        torch.tensor([e[4] for e in right], dtype=torch.float64),
        torch.tensor([e[4] for e in left], dtype=torch.float64),
        rtol=rtol,
        atol=atol,
    )
    assert entries(cpu.audit, "admit") == entries(device.audit, "admit")
    records = gradients(cpu), gradients(device)
    assert len(records[0]) == len(records[1]) > 3
    for expected, actual in zip(*records, strict=True):
        for name, value in expected.items():
            if value is None:
                assert actual[name] is None
            else:
                torch.testing.assert_close(actual[name].cpu(), value, rtol=rtol, atol=atol)
    assert device.predictor._parameter_id == cpu.predictor._parameter_id
    destination = os.environ.get("MARS_TITAN_MARS_RUN_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                device=DEVICE,
                dtype=str(dtype),
                refinements=refinements,
                episodes=episodes,
                updates=len(records[0]),
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
