"""Comparación de las seis rutas CUDA FP32 con la referencia CPU FP64."""

import pytest
import torch
from test_bridge_episodic_readout import CODEC, configuration, source

from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.regional_routes import run_route_pair

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("route", ["k1", "k2c", "k2r", "k4cc", "k4cr", "k4rc"])
def test_fp32_cuda_route_keeps_common_evidence_and_the_declared_output_tolerance(route):
    reference = EpisodicReadout(configuration(), dtype=torch.float64)
    actual = EpisodicReadout(configuration(), dtype=torch.float64)
    actual.load_state_dict(reference.state_dict())
    actual.to(device="cuda:0", dtype=torch.float32)
    snapshots = [
        EpisodeSnapshot.create(
            **source(16), codec_id=CODEC, context_id="b" * 64, cutoff=10, device=device, dtype=dtype
        )
        for device, dtype in (("cpu", torch.float64), ("cuda:0", torch.float32))
    ]
    base = torch.linspace(-0.5, 0.5, 64, dtype=torch.float64).reshape(2, 32)
    _, expected = run_route_pair(
        reference, base, snapshots[0], route=route, context_id="b" * 64, cutoff=10
    )
    _, observed = run_route_pair(
        actual,
        base.to(device="cuda:0", dtype=torch.float32),
        snapshots[1],
        route=route,
        context_id="b" * 64,
        cutoff=10,
    )
    torch.testing.assert_close(observed.state.cpu().double(), expected.state, rtol=2e-5, atol=1e-6)
    for left, right in zip(observed.reads, expected.reads, strict=True):
        assert torch.equal(left.bridge.evidence_ids.cpu(), right.bridge.evidence_ids)
        torch.testing.assert_close(
            left.bridge.evidence_weights.cpu().double(),
            right.bridge.evidence_weights,
            rtol=2e-5,
            atol=1e-6,
        )
