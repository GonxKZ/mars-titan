"""Comprobación CUDA del lector episódico de MARS-TITAN, sin pasos de optimizador.

Se ejecuta de forma explícita. Compara un recorrido de ajuste con su validación en CPU y en
el dispositivo indicado, con el mismo padre Titans-MAC congelado, los mismos parámetros
iniciales del lector y el optimizador que solo registra gradientes. Cubre M1 con K = 1 y
K = 4 en sus dos modos de selección de episodios, y M3 con K = 1 y con K = 4 y episodios
fijos. M3 estima sus escalas en CPU con el tramo de entrenamiento de la fixture.
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
from tests.training.test_mars_titan_run import gradients, recipe, scalers_for

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


def engine(streams, output, *, dtype, device, admission, refinements, episodes, source=None):
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
    extra = dict(scalers=scalers_for(streams, plan)) if admission == "m3" else {}
    return mt.ReadoutTrainer(
        parent.eval().requires_grad_(False),
        readout,
        plan,
        admission=admission,
        retention=mt.retention_config(plan, admission, **extra),
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


def compare_admissions(cpu, device, rtol, atol):
    """Mismos IDs, índices y rasgos de entrada. El error y la puntuación M3, con tolerancia."""
    left, right = entries(cpu.audit, "admit"), entries(device.audit, "admit")
    assert [e[:6] for e in left] == [e[:6] for e in right]
    assert all(len(e) == len(f) for e, f in zip(left, right, strict=True))
    rows = [
        (expected, actual)
        for e, f in zip(left, right, strict=True)
        if len(e) == 7
        for expected, actual in zip(e[6], f[6], strict=True)
    ]
    # ID, anomalía, relevancia y máscara salen de las entradas CPU de la decisión.
    exact = [((e[0], e[2], e[3], e[5]), (f[0], f[2], f[3], f[5])) for e, f in rows]
    assert all(expected == actual for expected, actual in exact)
    if rows:
        torch.testing.assert_close(
            torch.tensor([[r[1], r[4]] for _, r in rows], dtype=torch.float64),
            torch.tensor([[r[1], r[4]] for r, _ in rows], dtype=torch.float64),
            rtol=rtol,
            atol=atol,
        )
    return len(rows)


def selective_margin(engine):
    """Menor distancia entre la puntuación M3 elegida y la mejor descartada en cada evento.

    Compiten las ofertas del evento y el selectivo anterior. Los IDs empiezan en 1 en cada
    recorrido, porque cada uno parte de un banco vacío. Un margen amplio frente a la
    tolerancia del dispositivo evita que los IDs dependan del redondeo.
    """
    margins, known, before = [], {}, set()
    for entry in entries(engine.audit, "admit"):
        if min(entry[3]) == 1:
            known, before = {}, set()
        known.update((row[0], row[4]) for row in entry[6])
        selected = set(entry[5]["selective"])
        contenders = ({row[0] for row in entry[6]} | before) - selected
        if contenders and selected:
            margins.append(min(known[i] for i in selected) - max(known[i] for i in contenders))
        before = selected
    return min(margins)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("admission", "refinements", "episodes"),
    [
        ("m1", 1, "per_step"),
        ("m1", 4, "per_step"),
        ("m1", 4, "first_read"),
        ("m3", 1, "per_step"),
        ("m3", 4, "first_read"),
    ],
)
def test_device_pass_matches_cpu_without_optimizer_steps(
    tmp_path, dtype, admission, refinements, episodes
):
    _, streams = corpus(tmp_path / "corpus")
    rtol, atol = TOLERANCES[dtype]
    cuda = DEVICE.startswith("cuda")
    options = dict(dtype=dtype, admission=admission, refinements=refinements, episodes=episodes)
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
    scored = compare_admissions(cpu, device, rtol, atol)
    assert (scored > 0) == (admission == "m3")
    margin = selective_margin(cpu) if admission == "m3" else None
    assert margin is None or margin > 1e-4
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
                admission=admission,
                scored_offers=scored,
                minimum_selective_margin=margin,
                refinements=refinements,
                episodes=episodes,
                updates=len(records[0]),
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
