"""Comprobación CUDA del entrenador de la GRU candidata, sin pasos de optimizador.

Se ejecuta de forma explícita. Compara un recorrido de ajuste y una validación en CPU y en
el dispositivo indicado, con los mismos parámetros iniciales y el optimizador que solo
registra gradientes. También compara el ajuste de una ventana v2 y su traslado a la
siguiente, incluido el traslado en el dispositivo del estado elegido en CPU, con el
calentamiento de 12 meses de la receta de campaña: ajuste y traslados registran esos meses
y las mismas fases en CPU y en el dispositivo, y los tramos medidos observan entradas
anteriores a su inicio.
`MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
`MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

from tests.training.test_candidate_run import SELECTION, entries, named_records, trainer
from tests.training.test_candidate_walk_forward import WARMUP
from tests.training.test_financial_run import corpus

DEVICE = os.environ.get("MARS_TITAN_CANDIDATE_RUN_CHECK_DEVICE", "cuda:0")
TOLERANCES = {torch.float32: (2e-4, 2e-6), torch.float64: (1e-8, 1e-10)}
# Como en test_candidate_run.py, el optimizador solo registra gradientes.
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
        or (DEVICE == "cuda:0" and not torch.cuda.is_available()),
        reason="Faltan el enlace nativo o cuda:0",
    ),
    pytest.mark.usefixtures("learning_doubles"),
]


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


def walk_forward_views(root):
    from mars_titan.training import masked_campaign as engine
    from tests.training.test_masked_campaign import write_campaign
    from tests.training.test_walk_forward_v2_views import fixture

    data = fixture(root / "data", ("US",))
    campaign = write_campaign(root / "config", scopes=("US",))
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    return {name: Path(record["path"]) for name, record in prepared["US"]["windows"].items()}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_device_walk_forward_matches_cpu_without_optimizer_steps(tmp_path, dtype):
    """Ajuste de ventana y traslado en CPU y en el dispositivo, con las mismas filas."""
    import pyarrow.parquet as pq

    from mars_titan.models.quantile_head import QUANTILE_COLUMNS
    from mars_titan.training import candidate_run
    from mars_titan.training import candidate_walk_forward as walk
    from mars_titan.training import masked_campaign as engine
    from tests.training.test_financial_run import RecordingOptimizer

    views = walk_forward_views(tmp_path / "views")
    # El mismo calentamiento que declara la receta de campaña de la candidata.
    warmup = WARMUP
    assert warmup == 12
    recipe = candidate_run.CandidateRecipe(
        update_instants=2, epochs=1, block_rows=2, bank_capacity=4, selection=SELECTION
    )
    model = dict(feature_seed=43, key_seed=44, dtype=dtype)
    rtol, atol = TOLERANCES[getattr(torch, dtype)]
    cuda = DEVICE.startswith("cuda")
    reports, seconds = {}, {}
    for name, target in (("cpu", "cpu"), ("device", DEVICE)):
        if cuda and name == "device":
            torch.cuda.reset_peak_memory_stats(0)
        start = time.perf_counter()
        fit = walk.fit_window(
            views["fold-000"],
            tmp_path / name / "fit",
            recipe,
            seed=42,
            model=model,
            parent_id="US/fold-000/gru_episodic",
            device=target,
            warmup_months=warmup,
            optimizer_factory=RecordingOptimizer,
        )
        carried = walk.carry_window(
            tmp_path / name / "fit",
            views["fold-000"],
            views["fold-001"],
            tmp_path / name / "carry",
            parent_id="US/fold-000/gru_episodic",
            device=target,
        )
        if cuda and name == "device":
            torch.cuda.synchronize(0)
        seconds[name] = time.perf_counter() - start
        reports[name] = (fit, carried)
    # El estado del ancla en CPU también se traslada en el dispositivo.
    crossed = walk.carry_window(
        tmp_path / "cpu" / "fit",
        views["fold-000"],
        views["fold-001"],
        tmp_path / "crossed",
        parent_id="US/fold-000/gru_episodic",
        device=DEVICE,
    )
    cpu, device = reports["cpu"], reports["device"]
    # Ajuste y traslados repiten el calentamiento con las mismas fases en los dos dispositivos.
    phases = {}
    for name, report in (("fit", cpu[0]), ("carry", cpu[1])):
        phases[name] = {key: value["phase"] for key, value in report["sources"].items()}
    for report in (*cpu, *device, crossed):
        assert report["bank_policy"]["warmup_months"] == warmup
    assert walk.anchor_warmup(cpu[0]) == walk.anchor_warmup(device[0]) == warmup
    assert {key: value["phase"] for key, value in device[0]["sources"].items()} == phases["fit"]
    for report in (device[1], crossed):
        assert {k: v["phase"] for k, v in report["sources"].items()} == phases["carry"]
    measured = phases["carry"]["evaluation"]
    assert measured["warmup_start"] < measured["decision_start"]
    pairs = [
        (tmp_path / "cpu/fit", tmp_path / "device/fit", cpu[0], device[0]),
        (tmp_path / "cpu/carry", tmp_path / "device/carry", cpu[1], device[1]),
        (tmp_path / "cpu/carry", tmp_path / "crossed", cpu[1], crossed),
    ]
    # Sin pasos, el estado elegido es la inicialización de la semilla en ambos dispositivos.
    assert device[0]["checkpoint"]["parameters_sha256"] == cpu[0]["checkpoint"]["parameters_sha256"]
    for left_folder, right_folder, left, right in pairs:
        assert set(left["predictions"]) == set(right["predictions"])
        for partition, record in left["predictions"].items():
            expected = pq.read_table(left_folder / record["path"])
            actual = pq.read_table(right_folder / right["predictions"][partition]["path"])
            assert engine._rows_digest(actual) == engine._rows_digest(expected)
            for column in ("prediction", *QUANTILE_COLUMNS):
                torch.testing.assert_close(
                    torch.from_numpy(actual[column].to_numpy().copy()),
                    torch.from_numpy(expected[column].to_numpy().copy()),
                    rtol=rtol,
                    atol=atol,
                )
    destination = os.environ.get("MARS_TITAN_CANDIDATE_RUN_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                check="walk_forward",
                device=DEVICE,
                dtype=dtype,
                warmup_months=warmup,
                phases=phases,
                seconds=seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
