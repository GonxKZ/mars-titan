"""Reconstrucciones analíticas de fixtures, sin entrenamiento ni resultados financieros."""

import math
import random

import pytest
import torch

from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig, SharedEvidenceBridge

TOLERANCE = dict(atol=1e-12, rtol=1e-12)
CODEC, CONTEXT = "a" * 64, "b" * 64
A = math.tanh(1.0)
B = math.tanh(2 * A / 3)
C = math.tanh(A / 3)
D = math.tanh(4 * A / 3)
E = math.tanh(8 * A / 9)
F = math.tanh(2 * A / 9)
SELF_OUTPUT = ((A + D) / (A + B), -1.0)
MASKS = ((True, True, True), (True, False, True), (False, True, True), (False, False, False))
READ_FIELDS = (
    "values",
    "region_values",
    "evidence_weights",
    "bridge_weights",
    "evidence_ids",
    "selected_positions",
)


def floats(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def evidence(*, second=1.0, keep_second=True):
    count = 2 if keep_second else 1
    values = torch.zeros((count, 64), dtype=torch.float32, device="cpu")
    values[0, 0] = 1
    if keep_second:
        values[1, 1] = second
    return EpisodeSnapshot.create(
        keys=torch.eye(64, dtype=torch.float32, device="cpu")[:count],
        values=values,
        labels=torch.zeros(count, dtype=torch.float64, device="cpu"),
        ids=torch.arange(1, count + 1, dtype=torch.int64, device="cpu"),
        decision_at=torch.full((count,), 2, dtype=torch.int64, device="cpu"),
        available_at=torch.full((count,), 1, dtype=torch.int64, device="cpu"),
        maturity_at=torch.full((count,), 3, dtype=torch.int64, device="cpu"),
        cutoff=4,
        codec_id=CODEC,
        context_id=CONTEXT,
        dtype=torch.float64,
        device="cpu",
    )


@pytest.fixture
def source():
    return evidence()


def analytic_bridge(*, topology="ring", variant="cooperation", nonlinear=True, cycles=0):
    regions = 2 if variant == "cancellation" else 1 if variant == "single" else 3
    model = SharedEvidenceBridge(
        EvidenceBridgeConfig(
            codec_id=CODEC,
            hidden_size=2,
            neighbors=2,
            regions=tuple(f"reader_{index}" for index in range(regions)),
            topology=topology,
            cycles=cycles,
            temperature=1 / math.log(3),
            bridge_gain=1.0 if variant == "cancellation" else 0.5,
            recurrence=BridgeRecurrenceConfig(1, regions) if nonlinear else None,
        ),
        dtype=torch.float64,
        device="cpu",
    )
    matrices = floats((((4 / 3, -1 / 3), (0, 1)), ((1, 0), (-1 / 3, 4 / 3)), ((2, 0), (0, -2))))
    if variant == "cancellation":
        matrices[1] = floats(((-4, 3), (0, -1)))
    with torch.no_grad():
        # Los parámetros fijan el fixture. El buffer de topología queda intacto.
        for parameter in model.parameters():
            parameter.zero_()
        model.query.bias[0] = 1
        model.value.weight[0, 0] = 1
        model.value.weight[1, 1] = 1
        if regions > 1 and variant != "redundant":
            model.region_query[1, 0, :2] = floats((-1, 1))
        if regions == 3 and variant != "redundant":
            model.region_query[2, 0, :2] = floats((1 / math.sqrt(2) - 1, 1 / math.sqrt(2)))
        model.region_value.copy_(
            matrices[0].expand(regions, -1, -1) if variant == "redundant" else matrices[:regions]
        )
        if variant == "direct":
            direct = torch.eye(2, dtype=torch.float64, device="cpu") / A
            model.output.weight[:, :2] = direct
            model.output.weight[:, 2:4] = direct
        elif variant == "cancellation":
            model.output.weight[:, :2] = torch.eye(2, dtype=torch.float64, device="cpu") / A
        else:
            model.output.weight[:, :2] = torch.diag(floats((2 / (A + B), 2 / C)))
    return model.eval().requires_grad_(False)


def read(model, source, available=None):
    batch = 1 if available is None else len(available)
    state = floats(((1, 0),)).repeat(batch, 1)
    with torch.no_grad():
        return model(state, source, context_id=CONTEXT, cutoff=4, available=available)


def assert_same_reads(actual, expected):
    for name in READ_FIELDS:
        left, right = getattr(actual, name), getattr(expected, name)
        if left.is_floating_point():
            torch.testing.assert_close(left, right, **TOLERANCE)
        else:
            assert torch.equal(left, right)
    assert actual.snapshot_sha256 == expected.snapshot_sha256


def test_distinct_views_contribute_through_the_ring_with_a_fixed_receiver_output(source):
    isolated = read(analytic_bridge(nonlinear=False), source)
    connected = read(analytic_bridge(), source)
    torch.testing.assert_close(
        isolated.evidence_weights,
        floats((((3 / 4, 1 / 4), (1 / 4, 3 / 4), (1 / 2, 1 / 2)),)),
        **TOLERANCE,
    )
    torch.testing.assert_close(
        isolated.region_values, floats((((A, 0), (0, A), (A, -A)),)), **TOLERANCE
    )
    torch.testing.assert_close(isolated.values, floats(((2 * A / (A + B), 0),)), **TOLERANCE)
    assert float((isolated.values - floats(((1, 1),))).square().sum()) > 1
    torch.testing.assert_close(
        connected.region_values[:, 0], floats((((A + B) / 2, C / 2),)), **TOLERANCE
    )
    torch.testing.assert_close(connected.values, floats(((1, 1),)), **TOLERANCE)
    torch.testing.assert_close(
        connected.bridge_weights,
        floats((((1 / 2, 1 / 2, 0), (0, 1 / 2, 1 / 2), (1 / 2, 0, 1 / 2)),)),
        **TOLERANCE,
    )
    assert connected.evidence_ids.tolist() == isolated.evidence_ids.tolist() == [[1, 2]]
    single = read(analytic_bridge(variant="single", nonlinear=False), source)
    torch.testing.assert_close(single.values, isolated.values, **TOLERANCE)


@pytest.mark.parametrize(
    "topology,variant,expected,minimum_error",
    [
        ("self", "cooperation", SELF_OUTPUT, 4),
        ("complete", "cooperation", ((A + E) / (A + B), -F / C), 1),
        ("ring", "redundant", SELF_OUTPUT, 4),
    ],
)
def test_self_interference_and_duplicate_readers_have_the_declared_negative_results(
    source, topology, variant, expected, minimum_error
):
    result = read(analytic_bridge(topology=topology, variant=variant), source)
    torch.testing.assert_close(result.values, floats((expected,)), **TOLERANCE)
    assert float((result.values - floats(((1, 1),))).square().sum()) > minimum_error
    assert result.evidence_ids.tolist() == [[1, 2]]
    assert torch.isfinite(result.values).all()


def test_direct_aggregation_without_messages_is_a_successful_same_size_control(source):
    direct = analytic_bridge(variant="direct", nonlinear=False)
    connected = analytic_bridge()
    assert sum(value.numel() for value in direct.parameters()) == sum(
        value.numel() for value in connected.parameters()
    )
    result = read(direct, source)
    torch.testing.assert_close(result.values, floats(((1, 1),)), **TOLERANCE)
    assert result.evidence_ids.tolist() == [[1, 2]]


@pytest.mark.parametrize("nonlinear,cycles", [(False, 1), (True, 0)])
def test_aggregation_can_cancel_distinct_views_without_deleting_canonical_evidence(
    source, nonlinear, cycles
):
    before = source.export_cpu()
    isolated = read(analytic_bridge(variant="cancellation", nonlinear=False), source)
    mixed = read(
        analytic_bridge(
            topology="complete", variant="cancellation", nonlinear=nonlinear, cycles=cycles
        ),
        source,
    )
    torch.testing.assert_close(isolated.region_values, floats((((A, 0), (-A, 0)),)), **TOLERANCE)
    torch.testing.assert_close(isolated.values, floats(((1, 0),)), **TOLERANCE)
    torch.testing.assert_close(
        mixed.region_values, torch.zeros_like(mixed.region_values), **TOLERANCE
    )
    torch.testing.assert_close(mixed.values, floats(((0, 0),)), **TOLERANCE)
    assert float((mixed.values - floats(((1, 0),))).square().sum()) == pytest.approx(
        1, abs=1e-12, rel=1e-12
    )
    assert mixed.evidence_ids.tolist() == [[1, 2]]
    after = source.export_cpu()
    assert after["sha256"] == before["sha256"]
    for name, value in before["tensors"].items():
        assert torch.equal(value, after["tensors"][name])


def test_removing_an_antagonistic_reader_restores_the_linear_read_without_erasing_it(source):
    model = analytic_bridge(topology="complete", variant="cancellation", nonlinear=False, cycles=1)
    masked = read(model, source, torch.tensor(((True, False),), device="cpu"))
    torch.testing.assert_close(masked.values, floats(((1, 0),)), **TOLERANCE)
    torch.testing.assert_close(masked.bridge_weights, floats((((1, 0), (0, 0)),)), **TOLERANCE)
    assert masked.evidence_ids.tolist() == [[1, 2]] and source.count == 2


def test_reader_withdrawal_has_finite_outputs_and_preserves_the_shared_snapshot(source):
    before = source.export_cpu()
    masks = torch.tensor(MASKS, dtype=torch.bool, device="cpu")
    result = read(analytic_bridge(), source, masks)
    torch.testing.assert_close(
        result.values, floats(((1, 1), SELF_OUTPUT, (0, 0), (0, 0))), **TOLERANCE
    )
    assert torch.isfinite(result.values).all()
    assert result.evidence_ids.tolist() == [[1, 2]] * len(masks)
    assert torch.count_nonzero(result.evidence_weights[~masks]) == 0
    assert torch.count_nonzero(result.region_values[~masks]) == 0
    absent_edges = ~masks[:, :, None] | ~masks[:, None, :]
    assert torch.count_nonzero(result.bridge_weights[absent_edges]) == 0
    assert source.export_cpu()["sha256"] == before["sha256"]


@pytest.mark.parametrize("split", [1, 2])
def test_row_permutation_and_two_block_partitions_preserve_every_derived_read(source, split):
    model = analytic_bridge()
    masks = torch.tensor(MASKS, dtype=torch.bool, device="cpu")
    full = read(model, source, masks)
    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.int64, device="cpu")
    inverse = torch.argsort(permutation)
    permuted = read(model, source, masks[permutation])
    chunks = (read(model, source, masks[:split]), read(model, source, masks[split:]))
    for name in READ_FIELDS:
        expected = getattr(full, name)
        reordered = getattr(permuted, name)[inverse]
        blocked = torch.cat([getattr(chunk, name) for chunk in chunks])
        if expected.is_floating_point():
            torch.testing.assert_close(reordered, expected, **TOLERANCE)
            torch.testing.assert_close(blocked, expected, **TOLERANCE)
        else:
            assert torch.equal(reordered, expected) and torch.equal(blocked, expected)
    assert permuted.snapshot_sha256 == full.snapshot_sha256
    assert all(chunk.snapshot_sha256 == full.snapshot_sha256 for chunk in chunks)


def test_random_withdrawal_uses_an_independent_stream_and_fixed_row_masks(source):
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state().clone()
    row_ids = ("fixture_0", "fixture_1", "fixture_2", "fixture_3")
    generator = random.Random(20261071)
    removed = {row: generator.randrange(3) for row in sorted(row_ids)}
    masks = torch.tensor(
        [[region != removed[row] for region in range(3)] for row in row_ids],
        dtype=torch.bool,
        device="cpu",
    )
    expected = ((0, 0), SELF_OUTPUT, (1, 1))
    model = analytic_bridge()
    actual = read(model, source, masks)
    torch.testing.assert_close(
        actual.values, floats([expected[removed[row]] for row in row_ids]), **TOLERANCE
    )
    reordered = read(model, source, masks.flip(0))
    torch.testing.assert_close(actual.values, reordered.values.flip(0), **TOLERANCE)
    chunks = (read(model, source, masks[:1]), read(model, source, masks[1:]))
    torch.testing.assert_close(
        actual.values, torch.cat([chunk.values for chunk in chunks]), **TOLERANCE
    )
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_canonical_erasure_has_a_new_identity_and_cannot_distinguish_erased_contents():
    positive, negative = evidence(), evidence(second=-1)
    model = analytic_bridge()
    assert positive.export_cpu()["sha256"] != negative.export_cpu()["sha256"]
    assert not torch.allclose(
        read(model, positive).values, read(model, negative).values, **TOLERANCE
    )
    erased_positive = evidence(keep_second=False)
    erased_negative = evidence(second=-1, keep_second=False)
    assert erased_positive.export_cpu()["sha256"] == erased_negative.export_cpu()["sha256"]
    assert erased_positive.export_cpu()["sha256"] != positive.export_cpu()["sha256"]
    left, right = read(model, erased_positive), read(model, erased_negative)
    assert_same_reads(left, right)
    assert left.evidence_ids.tolist() == [[1]]
    assert not (
        torch.allclose(left.values, floats(((1, 1),)), **TOLERANCE)
        and torch.allclose(right.values, floats(((1, -1),)), **TOLERANCE)
    )
