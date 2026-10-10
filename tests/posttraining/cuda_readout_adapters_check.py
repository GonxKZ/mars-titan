"""Comprobación CUDA de los adaptadores de los lectores episódicos, sin pasos de optimizador.

Se ejecuta de forma explícita (por ejemplo con `memslot gpu`). La campaña base es la reducida
de `test_mars_titan_campaign.py`, ejecutada en CPU con el registrador que no cambia pesos, y
el padre es el lector M1 elegido en su ventana reentrenada. En el dispositivo, cada caso de
la matriz v3 para el lector con banco (núcleo, lectura episódica, ambos y la continuación
completa) se ajusta por etapas con las filas nuevas de la ventana siguiente y debe emitir
exactamente las filas del padre congelado en ese mismo dispositivo. Las filas del padre
congelado deben coincidir entre CPU y el dispositivo dentro de las tolerancias de la
precisión del padre (float64 en la receta reducida de las pruebas, que el lector hereda), y
el caso con núcleo y lectura debe recorrer los mismos pasos en los dos, con gradientes
iguales dentro de esas tolerancias.

Necesita el enlace nativo del banco episódico y `CUBLAS_WORKSPACE_CONFIG`, que exigen los
recorridos en `cuda:0`. `MARS_TITAN_READOUT_ADAPTERS_CHECK_DEVICE=cpu` ensaya la lógica sin
GPU y no acredita CUDA. `MARS_TITAN_READOUT_ADAPTERS_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.posttraining import chronological_matrix as cm
from mars_titan.posttraining import chronological_windows as cw
from mars_titan.posttraining.adapter_matrix import read_matrix
from mars_titan.training.titans_walk_forward import unfused_attention
from tests.posttraining.test_readout_windows import MATRIX, receipt
from tests.training.test_mars_titan_campaign import ARM
from tests.training.test_mars_titan_campaign import campaign_run as campaign_run
from tests.training.test_mars_titan_campaign import explicit_fastpath as explicit_fastpath
from tests.training.test_titans_campaign import permitted as permitted
from tests.training.test_titans_walk_forward import Factory

DEVICE = os.environ.get("MARS_TITAN_READOUT_ADAPTERS_CHECK_DEVICE", "cuda:0")
# Tolerancias por precisión, las de las demás comprobaciones CUDA de los lectores.
TOLERANCES = {"float32": dict(rtol=2e-4, atol=2e-6), "float64": dict(rtol=1e-8, atol=1e-10)}
PARTITIONS = ("validation", "calibration", "evaluation")
pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    or (DEVICE == "cuda:0" and not torch.cuda.is_available()),
    reason="Faltan el enlace nativo o cuda:0",
)


@pytest.fixture(autouse=True)
def strict_numerics():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def table(path):
    return pq.read_table(path).sort_by("sample_id")


@pytest.fixture(scope="module")
def chosen(campaign_run):  # noqa: F811
    """Brazo M1 elegido en la ventana reentrenada, sus vistas y los casos de la matriz."""
    matrix, digest = read_matrix(MATRIX)
    matrix["budget"].update(epochs=1)
    fitted, carried = campaign_run.windows[0], campaign_run.windows[1]
    searches = {
        name: receipt(campaign_run, f"US/{fitted}/{ARM}/search-{name}")
        for name in ("lr1e-4", "lr1e-3")
    }
    best = min(searches, key=lambda name: (searches[name]["score"], name))
    prepared = campaign_run.prepared["US"]["windows"]
    cases = {
        item["id"].split("/", 1)[1]: item["case"]
        for item in cm.cases(matrix, digest, "mars_titan", bank=True)
        if item["case"]["seed"] == 42
    }
    # El lector hereda la precisión del Titans-MAC padre de la receta reducida.
    titans = json.loads((campaign_run.root / "config" / "titans.json").read_text())
    dtype = titans["predictor"]["dtype"]
    assert dtype == "float64"
    return dict(
        dtype=dtype,
        tolerance=TOLERANCES[dtype],
        arm=(campaign_run.output / searches[best]["report"]["path"]).parent,
        fitted=Path(prepared[fitted]["path"]),
        carried=Path(prepared[carried]["path"]),
        matrix=matrix,
        digest=digest,
        cases=cases,
    )


def staged(chosen, name, output, device, *, carried=True):
    """Caso con el registrador de gradientes, por etapas en la ventana siguiente o, sin
    `carried`, en la ventana reentrenada del padre, que tiene más filas de ajuste."""
    factory = Factory()
    with unfused_attention():
        report = cw.run_readout_posttraining(
            chosen["arm"],
            chosen["carried"] if carried else chosen["fitted"],
            output,
            case=chosen["cases"][name],
            matrix=chosen["matrix"],
            digest=chosen["digest"],
            device=device,
            optimizer_factory=factory,
            parent_view=chosen["fitted"] if carried else None,
        )
    assert report["status"] == "completed", name
    assert factory.instances and all(item.calls > 0 for item in factory.instances), name
    return report, factory


def frozen(chosen, output, device):
    with unfused_attention():
        return cw.frozen_readout(
            chosen["arm"], chosen["fitted"], chosen["carried"], output, device=device
        )


def save(entry):
    destination = os.environ.get("MARS_TITAN_READOUT_ADAPTERS_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(entry)
        path.write_text(json.dumps(report, indent=2))


def test_null_adapters_reproduce_the_frozen_parent_on_the_device(chosen, tmp_path):
    cuda = DEVICE.startswith("cuda")
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    parents = {
        name: (tmp_path / f"frozen-{name}", frozen(chosen, tmp_path / f"frozen-{name}", device))
        for name, device in (("cpu", "cpu"), ("device", DEVICE))
    }
    (cpu_folder, cpu_frozen), (device_folder, device_frozen) = parents["cpu"], parents["device"]
    assert cpu_frozen["status"] == device_frozen["status"] == "completed"
    differences = {}
    for partition in PARTITIONS:
        left = table(cpu_folder / cpu_frozen["predictions"][partition]["path"])
        right = table(device_folder / device_frozen["predictions"][partition]["path"])
        assert left.num_rows == right.num_rows > 0, partition
        assert left["sample_id"].equals(right["sample_id"]) and left["target"].equals(
            right["target"]
        )
        worst = 0.0
        for name in QUANTILE_COLUMNS:
            expected = torch.from_numpy(left[name].to_numpy().astype("float64"))
            actual = torch.from_numpy(right[name].to_numpy().astype("float64"))
            torch.testing.assert_close(actual, expected, **chosen["tolerance"])
            worst = max(worst, float((actual - expected).abs().max()))
        differences[partition] = worst
    assert set(chosen["cases"]) == {
        "full_continuation",
        "core",
        "episodic_readout",
        "core+episodic_readout",
    }
    arms = {}
    for name in chosen["cases"]:
        output = tmp_path / "staged" / name
        report, factory = staged(chosen, name, output, DEVICE)
        for partition in PARTITIONS:
            mine = table(output / report["predictions"][partition]["path"])
            theirs = table(device_folder / device_frozen["predictions"][partition]["path"])
            for column in ("sample_id", "target", *QUANTILE_COLUMNS):
                assert mine[column].equals(theirs[column]), (name, partition, column)
        arms[name] = dict(
            updates=sum(item.calls for item in factory.instances),
            rows_equal_to_frozen_parent=True,
        )
    # Todos los casos recorren el mismo número de actualizaciones.
    assert len({arm["updates"] for arm in arms.values()}) == 1
    if cuda:
        torch.cuda.synchronize(0)
    save(
        dict(
            check="null_adapters",
            device=DEVICE,
            arms=arms,
            frozen_parent_max_abs_difference_cpu_device=differences,
            dtype=chosen["dtype"],
            tolerance=chosen["tolerance"],
            seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
            torch=torch.__version__,
        )
    )


def test_device_gradients_of_the_core_and_the_reader_match_the_cpu(chosen, tmp_path):
    runs = {
        name: staged(chosen, "core+episodic_readout", tmp_path / name, device, carried=False)[1]
        for name, device in (("cpu", "cpu"), ("device", DEVICE))
    }
    cpu, device = runs["cpu"], runs["device"]
    roles = [[group["role"] for group in item.param_groups] for item in device.instances]
    assert roles == [[group["role"] for group in item.param_groups] for item in cpu.instances]
    assert len(cpu.records) == len(device.records) > 1
    maximum = 0.0
    for expected, actual in zip(cpu.records, device.records, strict=True):
        assert len(expected) == len(actual)
        for wanted, found in zip(expected, actual, strict=True):
            assert (wanted is None) == (found is None)
            if wanted is None:
                continue
            assert found.device == torch.device(DEVICE)
            torch.testing.assert_close(found.cpu(), wanted, **chosen["tolerance"])
            maximum = max(maximum, float((found.cpu() - wanted).abs().max()))
    save(
        dict(
            check="core_and_reader_gradients",
            device=DEVICE,
            updates=len(device.records),
            roles=roles,
            max_abs_gradient_difference_cpu_device=maximum,
            dtype=chosen["dtype"],
            tolerance=chosen["tolerance"],
            torch=torch.__version__,
        )
    )
