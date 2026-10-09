"""Comprobación CUDA de los adaptadores de Titans-MAC, sin pasos de optimizador.

Se ejecuta de forma explícita (por ejemplo con `memslot gpu`). Usa float64, como los brazos
de la campaña con máscaras, y el registrador de gradientes que no modifica pesos:

- en el dispositivo, cada brazo con correcciones nulas emite exactamente las predicciones y
  el registro del padre congelado en ese mismo dispositivo;
- las predicciones del brazo con todos sus puntos coinciden entre CPU y el dispositivo con
  tolerancias declaradas;
- el ajuste recorre los mismos pasos en los dos dispositivos, con gradiente solo en los
  adaptadores y valores iguales dentro de esas tolerancias.

`MARS_TITAN_TITANS_ADAPTERS_CHECK_DEVICE=cpu` ensaya la lógica sin GPU y no acredita CUDA.
"""

import os

import pytest
import torch

from mars_titan.models.predictive_adaptation import adapter_names
from tests.posttraining.test_titans_adapters import (
    adapted,
    arm_points,
    clone,
    posttrainer,
    predictor,
    recipe,
)
from tests.posttraining.test_titans_adapters import matrix as matrix
from tests.posttraining.test_titans_adapters import shared as shared
from tests.training.test_financial_run import named_records

DEVICE = os.environ.get("MARS_TITAN_TITANS_ADAPTERS_CHECK_DEVICE", "cuda:0")
TOLERANCE = dict(rtol=1e-8, atol=1e-10)
pytestmark = pytest.mark.skipif(
    DEVICE == "cuda:0" and not torch.cuda.is_available(), reason="Falta cuda:0"
)


@pytest.fixture(autouse=True)
def strict_numerics():
    fastpath = torch.backends.mha.get_fastpath_enabled()
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.mha.set_fastpath_enabled(fastpath)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


def evaluated(model, streams):
    from mars_titan.training.financial_run import ChronologicalInference

    engine = ChronologicalInference(model, recipe(), audit=True)
    return engine.evaluate(streams["validation"]), engine.audit


def predictions(audit):
    return [entry for entry in audit if entry[0] == "prediction"]


@pytest.mark.parametrize("variant", ["transformer_direct", "mac_online"])
def test_null_adapters_reproduce_the_frozen_parent_on_the_device(shared, matrix, variant):
    _, streams = shared
    parent = predictor(streams, variant).eval()
    frozen = clone(parent).requires_grad_(False).to(DEVICE)
    expected, audit = evaluated(frozen, streams)
    for arm, points in arm_points(matrix, variant).items():
        model = adapted(parent, matrix, points).eval().to(DEVICE)
        assert model.head.weight.device == torch.device(DEVICE)
        assert evaluated(model, streams) == (expected, audit), arm


def test_device_predictions_match_the_cpu_within_tolerance(shared, matrix):
    _, streams = shared
    parent = predictor(streams, "mac_online").eval()
    points = arm_points(matrix, "mac_online")["head+readout+fusion"]
    _, cpu = evaluated(adapted(parent, matrix, points).eval(), streams)
    _, device = evaluated(adapted(parent, matrix, points).eval().to(DEVICE), streams)
    cpu, device = predictions(cpu), predictions(device)
    assert len(cpu) == len(device) > 0
    assert [entry[:4] for entry in device] == [entry[:4] for entry in cpu]
    torch.testing.assert_close(
        torch.tensor([entry[4] for entry in device], dtype=torch.float64),
        torch.tensor([entry[4] for entry in cpu], dtype=torch.float64),
        **TOLERANCE,
    )


def test_device_gradients_reach_only_the_adapters_and_match_the_cpu(shared, matrix, tmp_path):
    _, streams = shared
    parent = predictor(streams, "mac_online")
    points = arm_points(matrix, "mac_online")["head+readout+fusion"]
    runs = {}
    for name, device in (("cpu", "cpu"), ("device", DEVICE)):
        model = adapted(parent, matrix, points).to(device)
        engine = posttrainer(streams, tmp_path / name, model, max_grad_norm=None)
        assert engine.run()["status"] == "completed"
        runs[name] = (engine, named_records(engine), adapter_names(model))
    (cpu, cpu_records, names), (device, device_records, _) = runs["cpu"], runs["device"]
    assert len(device_records) == len(cpu_records) == device.optimizer.calls > 3
    assert [e for e in device.audit if e[0] != "prediction"] == [
        e for e in cpu.audit if e[0] != "prediction"
    ]
    for left, right in zip(device_records, cpu_records, strict=True):
        assert set(left) == set(right) == set(names)
        for name in names:
            assert left[name].device == torch.device(DEVICE), name
            torch.testing.assert_close(left[name].cpu(), right[name], **TOLERANCE)
