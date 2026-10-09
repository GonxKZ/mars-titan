"""Adaptadores nulos en cuda:0, sin pasos de optimizador. Se omiten sin GPU."""

import pytest
import torch

from mars_titan.models.baselines.multimodal import PRESENCE_FUSION
from mars_titan.models.predictive_adaptation import adapted_copy
from tests.models.test_predictive_adapters import FAMILIES, batch, parent, targets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere cuda:0")


@pytest.mark.parametrize("kind", FAMILIES)
def test_cuda_zero_adapters_reproduce_the_parent_and_match_cpu(kind, monkeypatch):
    # TF32 cambiaría la referencia float32 frente a CPU.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    device = torch.device("cuda:0")
    cpu_parent = parent(kind)
    cpu_child = adapted_copy(cpu_parent, targets(kind), seed=11)
    gpu_parent = parent(kind).to(device)
    gpu_child = adapted_copy(gpu_parent, targets(kind), seed=11)
    assert all(value.device == device for value in gpu_child.parameters() if value.requires_grad)
    inputs, presence = batch()
    gpu_inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        expected = gpu_parent(gpu_inputs, presence.to(device))
        assert torch.equal(gpu_child(gpu_inputs, presence.to(device)), expected)
        reference = cpu_child(inputs, presence)
    torch.testing.assert_close(expected.cpu(), reference, rtol=1e-4, atol=1e-5)
    gpu_parent.train()
    gpu_child.train()
    output = gpu_child(gpu_inputs, presence.to(device))
    assert torch.equal(output.detach(), gpu_parent(gpu_inputs, presence.to(device)))
    output.square().sum().backward()
    for name, value in gpu_child.named_parameters():
        assert (value.grad is not None) == value.requires_grad, name
    assert PRESENCE_FUSION == cpu_parent.configuration["mask_fusion"]
