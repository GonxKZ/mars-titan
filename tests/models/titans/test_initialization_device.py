"""La inicialización local no depende del dispositivo ambiental de PyTorch."""

import pytest
import torch

from mars_titan.models.titans import MACConfig, MemoryConfig, NeuralMemory, TitansMAC


@pytest.mark.parametrize("kind", ["memory", "mac"])
def test_explicit_cpu_initialization_overrides_ambient_device_without_changing_rng(kind):
    config = MemoryConfig(dim=4, depth=2)

    def build():
        if kind == "memory":
            return NeuralMemory(config, device="cpu", dtype=torch.float64)
        return TitansMAC(
            MACConfig(memory=config, heads=2, persistent_tokens=2),
            device="cpu",
            dtype=torch.float64,
        )

    expected = build()
    before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        actual = build()
    assert torch.equal(before, torch.random.get_rng_state())
    for (name, left), (other, right) in zip(
        expected.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert name == other
        assert right.device == torch.device("cpu")
        torch.testing.assert_close(left, right, rtol=0, atol=0)
