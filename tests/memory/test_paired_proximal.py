"""Una factorización para dos estados proximales sobre la misma cohorte madura."""

from dataclasses import replace

import pytest
import torch

from mars_titan.memory.associative_memory import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureFeedback,
)
from mars_titan.memory.paired_proximal import write_proximal_pair


def inputs(rows=7, size=5, rate=0.25, weighted=False):
    generator = torch.Generator().manual_seed(20261026)
    keys = torch.randn((rows, size), dtype=torch.float64, generator=generator)
    keys = keys / torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
    values = torch.randn((rows, 1), dtype=torch.float64, generator=generator)
    other = torch.randn((rows, size), dtype=torch.float64, generator=generator)
    weights = (
        torch.arange(1, rows + 1, dtype=torch.float64)
        if weighted
        else torch.ones(rows, dtype=torch.float64)
    )
    weights = weights / weights.sum()
    feedback = MatureFeedback(
        torch.arange(1, rows + 1),
        torch.full((rows,), 5),
        torch.full((rows,), 10),
        keys,
        values,
        weights,
    )
    config = AssociativeMemoryConfig("proximal", size, 1, rate=rate, forgetting=0.1)
    left = AssociativeMemory(
        config, torch.randn((size, 1), dtype=torch.float64, generator=generator)
    )
    right = AssociativeMemory(
        replace(config, value_size=size),
        torch.randn((size, size), dtype=torch.float64, generator=generator),
    )
    return left, right, feedback, other


@pytest.mark.parametrize("rows", [1, 7, 256])
@pytest.mark.parametrize("size", [5, 64])
@pytest.mark.parametrize("rate", [0.0, 0.25, 3.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_pair_matches_two_independent_writes_bit_for_bit(rows, size, rate, weighted):
    left, right, feedback, other = inputs(rows, size, rate, weighted)
    expected = (
        left.write(feedback, cutoff=10),
        right.write(replace(feedback, values=other), cutoff=10),
    )
    actual = write_proximal_pair(left, right, feedback, other, cutoff=10)
    for value, reference in zip(actual, expected, strict=True):
        assert torch.equal(value.matrix, reference.matrix)
        assert value.export()["matrix_sha256"] == reference.export()["matrix_sha256"]
        assert (value.writes, value.cursor) == (reference.writes, reference.cursor)
        assert not value.matrix.requires_grad
    assert left.writes == right.writes == 0


def test_pair_uses_one_factor_and_keeps_canonical_alignment(monkeypatch):
    left, right, feedback, other = inputs()
    reverse = torch.arange(6, -1, -1)
    reversed_feedback = replace(
        feedback,
        **{
            name: getattr(feedback, name)[reverse]
            for name in ("ids", "decision_at", "available_at", "keys", "values", "weights")
        },
    )
    expected = (
        left.write(feedback, cutoff=10),
        right.write(replace(feedback, values=other), cutoff=10),
    )
    calls, original = [], torch.linalg.cholesky_ex

    def factor(matrix):
        calls.append(tuple(matrix.shape))
        return original(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", factor)
    actual = write_proximal_pair(left, right, reversed_feedback, other[reverse], cutoff=10)
    assert calls == [(5, 5)]
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a.matrix, b.matrix)


def test_pair_handles_rank_deficient_and_zero_weight_rows():
    left, right, feedback, other = inputs()
    keys = torch.zeros_like(feedback.keys)
    keys[:, 0] = 1
    weights = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    feedback = replace(feedback, keys=keys, weights=weights)
    actual = write_proximal_pair(left, right, feedback, other, cutoff=10)
    assert torch.equal(actual[0].matrix, left.write(feedback, cutoff=10).matrix)
    assert torch.equal(
        actual[1].matrix, right.write(replace(feedback, values=other), cutoff=10).matrix
    )


def test_empty_pair_does_not_apply_forgetting_or_replace_the_states():
    left, right, feedback, other = inputs()
    feedback = replace(
        feedback,
        **{
            name: getattr(feedback, name)[:0]
            for name in ("ids", "decision_at", "available_at", "keys", "values")
        },
        weights=None,
    )
    actual = write_proximal_pair(left, right, feedback, other[:0], cutoff=10)
    assert actual[0] is left and actual[1] is right


@pytest.mark.parametrize(
    "change", [dict(rule="delta"), dict(key_size=6), dict(rate=0.5), dict(forgetting=0.2)]
)
def test_pair_rejects_different_rules_dimensions_or_hyperparameters(change):
    left, right, feedback, other = inputs()
    right = AssociativeMemory(replace(right.config, **change))
    with pytest.raises(ValueError, match="compartir"):
        write_proximal_pair(left, right, feedback, other, cutoff=10)


def test_pair_rejects_unsynchronised_counters_before_updating():
    left, right, feedback, other = inputs()
    right = right.write(replace(feedback, values=other), cutoff=10)
    with pytest.raises(ValueError, match="compartir"):
        write_proximal_pair(left, right, feedback, other, cutoff=20)


def test_pair_rejects_future_and_repeated_feedback():
    left, right, feedback, other = inputs()
    with pytest.raises(ValueError, match="antes del corte"):
        write_proximal_pair(left, right, feedback, other, cutoff=9)
    left, right = write_proximal_pair(left, right, feedback, other, cutoff=10)
    with pytest.raises(ValueError, match="orden canónico"):
        write_proximal_pair(left, right, feedback, other, cutoff=10)


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "device"])
def test_pair_rejects_invalid_companion_values_without_mutating_the_states(kind):
    left, right, feedback, other = inputs()
    before = left.matrix, right.matrix
    if kind == "shape":
        other = other[:, :1]
    elif kind == "dtype":
        other = other.float()
    elif kind == "nan":
        other[0, 0] = float("nan")
    else:
        other = torch.empty(other.shape, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError):
        write_proximal_pair(left, right, feedback, other, cutoff=10)
    assert torch.equal(left.matrix, before[0]) and torch.equal(right.matrix, before[1])


def test_pair_reports_overflow_and_does_not_publish_the_other_state():
    left, right, feedback, other = inputs(rows=1, size=1, rate=3)
    feedback = replace(feedback, keys=torch.ones_like(feedback.keys))
    other.fill_(torch.finfo(torch.float64).max)
    before = left.matrix, right.matrix
    with pytest.raises(ValueError, match="no finito"):
        write_proximal_pair(left, right, feedback, other, cutoff=10)
    assert torch.equal(left.matrix, before[0]) and torch.equal(right.matrix, before[1])


def test_pair_rejects_non_associative_states():
    left, right, feedback, other = inputs()
    with pytest.raises(ValueError, match="compartir"):
        write_proximal_pair(None, right, feedback, other, cutoff=10)
    with pytest.raises(ValueError, match="compartir"):
        write_proximal_pair(left, object(), feedback, other, cutoff=10)


def test_pair_propagates_a_factorization_failure_reported_by_the_library(monkeypatch):
    left, right, feedback, other = inputs()
    before = left.matrix, right.matrix
    original = torch.linalg.cholesky_ex

    def failed(matrix):
        factor, info = original(matrix)
        return factor, torch.ones_like(info)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", failed)
    with pytest.raises(ValueError, match="definido positivo"):
        write_proximal_pair(left, right, feedback, other, cutoff=10)
    assert torch.equal(left.matrix, before[0]) and torch.equal(right.matrix, before[1])
