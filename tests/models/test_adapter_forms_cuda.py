"""Formas de adaptador de #444 con correcciones nulas en cuda:0, sin pasos de optimizador.

Se omiten sin GPU. Repiten en el dispositivo lo que `test_adapter_forms.py` comprueba en
CPU: en inferencia el brazo emite los mismos bits que el padre y los pesos efectivos de las
formas por tensor son los del padre bit a bit. Interesan sobre todo dos rutas que la CPU no
recorre, los sesgos de las recurrentes con cuDNN (BitFit) y las normas por fila de DoRA
calculadas en la GPU. Frente a CPU se aceptan las tolerancias de
`test_predictive_adapters_cuda.py`, sin TF32.
"""

import pytest
import torch
from torch.nn.utils import parametrize

from mars_titan.models.predictive_adaptation import adapted_copy, adapter_names
from tests.models.test_adapter_forms import CASES, arm_targets, batch, parent, same_bits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0")


@pytest.mark.parametrize(("kind", "arm"), CASES)
def test_cuda_null_forms_reproduce_the_parent_and_match_cpu(kind, arm, monkeypatch):
    # TF32 cambiaría la referencia float32 frente a CPU.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    device = torch.device("cuda:0")
    cpu_parent = parent(kind)
    cpu_child = adapted_copy(cpu_parent, arm_targets(cpu_parent, arm), seed=11)
    gpu_parent = parent(kind).to(device)
    gpu_child = adapted_copy(gpu_parent, arm_targets(gpu_parent, arm), seed=11)
    names = adapter_names(gpu_child)
    assert names and all(gpu_child.get_parameter(name).device == device for name in names)
    inputs, presence = batch()
    gpu_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        expected = gpu_parent(gpu_inputs, presence.to(device))
        assert same_bits(gpu_child(gpu_inputs, presence.to(device)), expected)
        reference = cpu_child(inputs, presence)
    torch.testing.assert_close(expected.cpu(), reference, rtol=1e-4, atol=1e-5)
    gpu_parent.train()
    gpu_child.train()
    # Con correcciones nulas, cada peso efectivo de las formas por tensor es el del padre.
    for name, module in gpu_child.named_modules():
        if parametrize.is_parametrized(module):
            for tensor in module.parametrizations:
                original = getattr(gpu_parent.get_submodule(name), tensor)
                assert same_bits(getattr(module, tensor).detach(), original), (name, tensor)
    output = gpu_child(gpu_inputs, presence.to(device))
    # En entrenamiento algunas proyecciones cambian de núcleo según requieran gradiente,
    # igual que en `test_predictive_adapters_cuda.py`, así que se compara con tolerancia.
    torch.testing.assert_close(
        output.detach(), gpu_parent(gpu_inputs, presence.to(device)), rtol=1e-6, atol=1e-7
    )
    output.square().sum().backward()
    for name, value in gpu_child.named_parameters():
        assert (value.grad is not None) == value.requires_grad, name
        if value.grad is not None:
            assert torch.isfinite(value.grad).all(), name
