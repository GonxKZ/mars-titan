"""Comprobación CUDA de la regeneración bit a bit, sin pasos de optimizador.

Se ejecuta de forma explícita. La retención v2 solo libera una tabla por fila si su
regeneración desde el estado elegido sale idéntica en el mismo dispositivo. Aquí cada
familia se ajusta en el dispositivo indicado con un optimizador que solo registra
gradientes, en FP32 estricto y en FP32 (la precisión de la campaña), y después se regenera
en ese mismo dispositivo con `regenerate=True`. Se exige igualdad bit a bit de validación,
calibración y evaluación con la huella de contenido registrada.

Cubre Titans-MAC y, con el enlace episódico nativo, el lector M1 de MARS-TITAN y la GRU
episódica. `MARS_TITAN_REGENERATION_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no
acredita CUDA. `MARS_TITAN_REGENERATION_CHECK_REPORT` guarda el resultado en JSON.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_EPISODIC_NATIVE=<enlace> \\
        uv run --no-sync pytest -q tests/training/cuda_regeneration_check.py
"""

import json
import os
from pathlib import Path

import pytest
import torch

from mars_titan.training import prediction_regeneration as regeneration
from mars_titan.training import titans_walk_forward as wf
from tests.training.test_titans_walk_forward import Factory, recipe, unfused_attention, views
from tests.training.test_titans_walk_forward import (
    learning_doubles_module as learning_doubles_module,
)

DEVICE = os.environ.get("MARS_TITAN_REGENERATION_CHECK_DEVICE", "cuda:0")
HELD_OUT = {"validation", "calibration", "evaluation"}
NATIVE = os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
pytestmark = [
    pytest.mark.skipif(DEVICE == "cuda:0" and not torch.cuda.is_available(), reason="Falta cuda:0"),
    pytest.mark.usefixtures("learning_doubles"),
]
RESULTS = {}


@pytest.fixture(scope="module", autouse=True)
def strict_numerics():
    """FP32 estricto en el ajuste y en la regeneración, como exige la campaña."""
    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    numerics = regeneration.strict_fp32()
    yield numerics
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before[:2]
    torch.set_float32_matmul_precision(before[2])
    path = os.environ.get("MARS_TITAN_REGENERATION_CHECK_REPORT")
    if path:
        Path(path).write_text(
            json.dumps(
                dict(
                    device=DEVICE,
                    gpu=torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else None,
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(),
                    numerics=numerics,
                    cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    results=RESULTS,
                    optimizer_steps=0,
                ),
                indent=1,
            )
        )


def float32_recipe(root):
    """Receta reducida de las pruebas con la precisión FP32 de la campaña."""
    path = recipe(root)
    document = json.loads(path.read_text())
    document["predictor"]["dtype"] = "float32"
    path.write_text(json.dumps(document))
    return path


def check(name, folder, report, destination, produced):
    result = regeneration.compare(regeneration.originals(folder, report), destination, produced)
    RESULTS[name] = {
        partition: dict(identical=item["identical"], same_file=item["same_file"])
        for partition, item in result["partitions"].items()
    }
    assert set(result["partitions"]) == HELD_OUT
    assert result["identical"] is True, result


@pytest.fixture(scope="module")
def titans(tmp_path_factory, learning_doubles_module):
    root = tmp_path_factory.mktemp("cuda-regeneration")
    view, protocol = views(root / "base")
    output = root / "titans"
    with unfused_attention():
        report = wf.run_titans_window(
            view,
            protocol,
            "fold-000",
            float32_recipe(root),
            variant="mac_online",
            seed=42,
            output=output,
            device=DEVICE,
            indices=root / "indices",
            optimizer_factory=Factory(),
        )
    return dict(root=root, view=view, output=output, report=report)


def test_titans_mac_regenerates_its_rows_on_the_same_device(titans, tmp_path):
    with unfused_attention():
        produced = wf.carry_titans(
            titans["output"],
            titans["view"],
            titans["view"],
            tmp_path / "again",
            device=DEVICE,
            regenerate=True,
        )
    check("titans_mac", titans["output"], titans["report"], tmp_path / "again", produced)


@pytest.mark.skipif(not NATIVE, reason="Falta el enlace episódico nativo")
def test_the_mars_titan_reader_regenerates_its_rows_on_the_same_device(titans, tmp_path):
    from mars_titan.training import mars_titan_walk_forward as mw
    from tests.training.test_mars_titan_walk_forward import M1, readout_recipe

    output = titans["root"] / "mars"
    with unfused_attention():
        report = mw.run_mars_titan_window(
            titans["view"],
            titans["output"],
            readout_recipe(titans["root"]),
            components=M1,
            seed=42,
            output=output,
            search_case="lr1e-4",
            device=DEVICE,
            indices=titans["root"] / "mars-indices",
            optimizer_factory=Factory(),
        )
        produced = mw.carry_mars_titan(
            output,
            titans["view"],
            titans["view"],
            tmp_path / "again",
            device=DEVICE,
            regenerate=True,
        )
    check("mars_titan", output, report, tmp_path / "again", produced)


@pytest.mark.skipif(not NATIVE, reason="Falta el enlace episódico nativo")
def test_the_episodic_gru_regenerates_its_rows_on_the_same_device(titans, tmp_path):
    from mars_titan.training import candidate_walk_forward as candidate
    from tests.training.test_candidate_walk_forward import SMALL
    from tests.training.test_financial_run import RecordingOptimizer

    output = titans["root"] / "episodic"
    model = dict(feature_seed=43, key_seed=44, dtype="float32")
    report = candidate.fit_window(
        titans["view"],
        output,
        SMALL,
        seed=42,
        model=model,
        parent_id="US/fold-000/gru_episodic",
        device=DEVICE,
        optimizer_factory=RecordingOptimizer,
    )
    produced = candidate.carry_window(
        output,
        titans["view"],
        titans["view"],
        tmp_path / "again",
        parent_id="US/fold-000/gru_episodic",
        device=DEVICE,
        regenerate=True,
    )
    check("episodic_gru", output, report, tmp_path / "again", produced)
