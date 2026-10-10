"""Comprobación CUDA de los adaptadores de la GRU candidata, sin pasos de optimizador.

Se ejecuta de forma explícita (por ejemplo con `memslot gpu`). El padre es la ventana técnica
de `test_candidate_adapters.py`, ajustada en CPU con el registrador que no cambia pesos, en
float64 y con el calentamiento de 12 meses de la receta. En el dispositivo, cada caso de la
matriz para la GRU candidata (la cabeza con su corrección a cero y la continuación
completa) se ajusta por etapas con las filas nuevas de la ventana siguiente y debe emitir
exactamente las filas del padre congelado en ese mismo dispositivo. Las filas del padre
congelado deben coincidir entre CPU y el dispositivo dentro de las tolerancias de float64, y
el ajuste de la cabeza debe recorrer los mismos pasos en los dos, con gradiente solo en su
corrección y valores iguales dentro de esas tolerancias.

`MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
`MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_REPORT` guarda las medidas en JSON.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.posttraining import candidate_adapters as ca
from tests.posttraining.test_candidate_adapters import FIRST, NEXT, SCOPE, cases, rows
from tests.posttraining.test_candidate_adapters import allowed as allowed
from tests.posttraining.test_candidate_adapters import matrix as matrix
from tests.posttraining.test_candidate_adapters import parent as parent
from tests.posttraining.test_candidate_adapters import views as views
from tests.training.test_candidate_walk_forward import WARMUP
from tests.training.test_financial_run import RecordingOptimizer

DEVICE = os.environ.get("MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_DEVICE", "cuda:0")
TOLERANCE = dict(rtol=1e-8, atol=1e-10)
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


def staged(parent, views, matrix, case, output, device, *, window=NEXT):
    """Caso de la matriz sin cambiar pesos, por etapas en la ventana siguiente o, con
    `window=FIRST`, en la propia ventana del padre, que tiene más filas de ajuste."""
    document, digest = matrix
    made = []

    def factory(groups):
        made.append(RecordingOptimizer(groups))
        return made[-1]

    report = ca.run_candidate_posttraining(
        parent,
        views.windows[SCOPE][window],
        output,
        case=case,
        matrix=document,
        digest=digest,
        device=device,
        optimizer_factory=factory,
        parent_view=views.windows[SCOPE][FIRST] if window == NEXT else None,
    )
    assert report["status"] == "completed" and made and made[0].calls > 0
    return report, made[0]


def gradients(optimizer):
    """Gradientes registrados en cada paso, en el orden de los grupos del optimizador."""
    order = [id(value) for group in optimizer.param_groups for value in group["params"]]
    return [[record[key] for key in order] for record in optimizer.records]


def column(folder, record, name):
    return torch.from_numpy(np.array(rows(folder, record)[name].to_numpy(), dtype=np.float64))


def test_null_adapters_reproduce_the_frozen_parent_on_the_device(
    views, parent, matrix, allowed, tmp_path
):
    cuda = DEVICE.startswith("cuda")
    if cuda:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    frozen = {}
    for name, device in (("cpu", "cpu"), ("device", DEVICE)):
        output = tmp_path / f"frozen-{name}"
        receipt = ca.frozen_candidate(
            parent, views.windows[SCOPE][FIRST], views.windows[SCOPE][NEXT], output, device=device
        )
        assert receipt["status"] == "completed" and receipt["warmup_months"] == WARMUP == 12
        frozen[name] = (output, receipt)
    (cpu_folder, cpu_frozen), (device_folder, device_frozen) = frozen["cpu"], frozen["device"]
    differences = {}
    for partition in ca.PREDICTED:
        left = rows(cpu_folder, cpu_frozen["predictions"][partition])
        right = rows(device_folder, device_frozen["predictions"][partition])
        assert left.num_rows == right.num_rows > 0, partition
        assert left.select(["asset_id", "prediction_at", "target"]).equals(
            right.select(["asset_id", "prediction_at", "target"])
        )
        expected = column(cpu_folder, cpu_frozen["predictions"][partition], "prediction")
        actual = column(device_folder, device_frozen["predictions"][partition], "prediction")
        torch.testing.assert_close(actual, expected, **TOLERANCE)
        differences[partition] = float((actual - expected).abs().max())
    arms = {}
    for name, case in cases(matrix).items():
        output = tmp_path / f"staged-{name}"
        report, optimizer = staged(parent, views, matrix, case, output, DEVICE)
        for partition in ca.PREDICTED:
            mine = rows(output, report["predictions"][partition])
            theirs = rows(device_folder, device_frozen["predictions"][partition])
            assert mine.equals(theirs), (name, partition)
        arms[name] = dict(updates=optimizer.calls, rows_equal_to_frozen_parent=True)
    assert set(arms) == {"head", "full_continuation"}
    if cuda:
        torch.cuda.synchronize(0)
    destination = os.environ.get("MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                check="null_adapters",
                device=DEVICE,
                dtype="float64",
                warmup_months=WARMUP,
                arms=arms,
                frozen_parent_max_abs_difference_cpu_device=differences,
                tolerance=TOLERANCE,
                seconds=time.perf_counter() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(0) if cuda else None,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))


def test_device_gradients_reach_only_the_head_correction_and_match_the_cpu(
    views, parent, matrix, allowed, tmp_path
):
    case = cases(matrix)["head"]
    runs = {}
    for name, device in (("cpu", "cpu"), ("device", DEVICE)):
        report, optimizer = staged(
            parent, views, matrix, case, tmp_path / name, device, window=FIRST
        )
        assert [group["role"] for group in optimizer.param_groups] == ["head_adapters"]
        runs[name] = (report, optimizer)
    (cpu, cpu_optimizer), (device, device_optimizer) = runs["cpu"], runs["device"]
    assert cpu["global_step"] == device["global_step"] == device_optimizer.calls > 1
    left, right = gradients(cpu_optimizer), gradients(device_optimizer)
    assert len(left) == len(right) == device_optimizer.calls
    maximum = 0.0
    for expected, actual in zip(left, right, strict=True):
        assert len(expected) == len(actual) == 2
        for wanted, found in zip(expected, actual, strict=True):
            assert wanted is not None and found is not None
            assert found.device == torch.device(DEVICE)
            torch.testing.assert_close(found.cpu(), wanted, **TOLERANCE)
            maximum = max(maximum, float((found.cpu() - wanted).abs().max()))
    destination = os.environ.get("MARS_TITAN_CANDIDATE_ADAPTERS_CHECK_REPORT")
    if destination:
        path = Path(destination)
        report = json.loads(path.read_text()) if path.exists() else []
        report.append(
            dict(
                check="head_gradients",
                device=DEVICE,
                updates=device_optimizer.calls,
                max_abs_gradient_difference_cpu_device=maximum,
                tolerance=TOLERANCE,
                torch=torch.__version__,
            )
        )
        path.write_text(json.dumps(report, indent=2))
