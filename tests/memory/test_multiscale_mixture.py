"""E1 frente a decaimiento temporal, Hedge y resultados emitidos guardados."""

import pytest
import torch

from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.multiscale_mixture import DAY_US, MultiscaleMemory, MultiscaleMixtureConfig

BASE = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.5))


def feedback(memory, at, *, values=(0.02, -0.01)):
    keys = torch.zeros(2, 64, dtype=torch.float64)
    keys[0, 0] = 1
    keys[1, 1] = 1
    return BASE.feedback(
        ids=[memory.writes + 1, memory.writes + 2],
        decision_at=[at - 1] * 2,
        available_at=[at] * 2,
        keys=keys,
        values=values,
    )


def test_calendar_decay_uses_elapsed_time_without_rebasing_on_read():
    config = MultiscaleMixtureConfig(half_lives_days=(1.0, 2.0, 4.0))
    empty = MultiscaleMemory(BASE, config)
    first = feedback(empty, 10)
    memory = empty.write(
        first, cutoff=10, issued_predictions=torch.zeros(2, 3, dtype=torch.float64)
    )
    before = memory.export()
    prediction, parts = memory.predict(first.keys, at=10 + DAY_US)
    initial = memory.predict(first.keys, at=10)[1]
    expected = initial * torch.tensor([0.5, 2 ** (-0.5), 2 ** (-0.25)], dtype=torch.float64)
    torch.testing.assert_close(parts, expected, rtol=1e-15, atol=0)
    torch.testing.assert_close(prediction[:, 0], expected.mean(dim=1), rtol=0, atol=0)
    assert memory.clock == 10 and memory.export()["state_sha256"] == before["state_sha256"]
    with pytest.raises(ValueError):
        memory.predict(first.keys, at=9)


def test_hedge_uses_the_issued_experts_not_current_memories():
    config = MultiscaleMixtureConfig(
        half_lives_days=(1.0, 2.0, 4.0), weight_rate=2.0, loss_scale=0.02
    )
    memory = MultiscaleMemory(BASE, config)
    cohort = feedback(memory, 10, values=(0.02, 0.02))
    issued = torch.tensor([[0.02, 0.0, -0.02], [0.02, 0.0, -0.02]], dtype=torch.float64)
    updated = memory.write(cohort, cutoff=10, issued_predictions=issued)
    expected = torch.exp(torch.tensor([0.0, -2.0, -2.0], dtype=torch.float64))
    expected /= expected.sum()
    torch.testing.assert_close(updated.weights, expected, rtol=1e-15, atol=0)
    assert torch.equal(memory.weights, torch.full((3,), 1 / 3, dtype=torch.float64))
    assert updated.cohorts == 1 and updated.writes == 2


def test_restore_preserves_weights_clock_and_the_next_publication():
    config = MultiscaleMixtureConfig()
    memory = MultiscaleMemory(BASE, config)
    cohort = feedback(memory, 10)
    memory = memory.write(
        cohort, cutoff=10, issued_predictions=torch.zeros(2, 3, dtype=torch.float64)
    )
    restored = MultiscaleMemory.restore(BASE, config, memory.export())
    a, parts = memory.predict(cohort.keys, at=20)
    assert torch.equal(a, restored.predict(cohort.keys, at=20)[0])
    second = feedback(memory, 20)
    left = memory.write(second, cutoff=20, issued_predictions=parts)
    right = restored.write(second, cutoff=20, issued_predictions=parts)
    assert left.export()["state_sha256"] == right.export()["state_sha256"]


def test_mature_correction_connects_the_factor_and_rejects_double_forgetting():
    correction = MatureCorrection(BASE.memory, multiscale=MultiscaleMixtureConfig())
    assert isinstance(correction.empty(), MultiscaleMemory)
    assert correction.restore(correction.empty().export()).clock is None
    with pytest.raises(ValueError):
        MatureCorrection(
            AssociativeMemoryConfig("proximal", forgetting=0.01),
            multiscale=MultiscaleMixtureConfig(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(half_lives_days=[]),
        dict(half_lives_days=[1] * 5),
        dict(half_lives_days=[2, 1]),
        dict(half_lives_days=[False]),
        dict(weighting="unknown"),
        dict(weighting="uniform"),
        dict(weight_rate=float("inf")),
        dict(loss_scale=0),
    ],
)
def test_invalid_configs_are_rejected(changes):
    with pytest.raises(ValueError):
        MultiscaleMixtureConfig(**changes)


def test_invalid_read_does_not_construct_decayed_copies(monkeypatch):
    from mars_titan.memory.associative_memory import AssociativeMemory

    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig())

    def forbidden(*args):
        raise AssertionError("Se intentó preparar estado con una consulta inválida")

    monkeypatch.setattr(AssociativeMemory, "decay_coefficients", forbidden)
    with pytest.raises(ValueError):
        memory.predict(torch.zeros(1, 64), at=10)


def test_non_sequence_expert_input_is_rejected_before_iterating_it():
    def forbidden():
        raise AssertionError("No se debe consumir un iterable externo sin límite")
        yield

    with pytest.raises(ValueError):
        MultiscaleMemory(BASE, MultiscaleMixtureConfig(), experts=forbidden())


@pytest.mark.parametrize("kind", ["future", "wrong_shape", "wrong_dtype", "nonfinite", "overflow"])
def test_bad_mature_results_or_issued_parts_are_rejected(kind):
    from dataclasses import replace

    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig())
    data = feedback(memory, 10)
    parts = torch.zeros(2, 3, dtype=torch.float64)
    if kind == "future":
        data = replace(data, available_at=data.available_at + 1)
    if kind == "wrong_shape":
        parts = parts[:1]
    if kind == "wrong_dtype":
        parts = parts.float()
    if kind == "nonfinite":
        parts[0, 0] = float("nan")
    if kind == "overflow":
        parts.fill_(1e308)
        data = replace(data, values=torch.full((2, 1), -1e308, dtype=torch.float64))
    with pytest.raises(ValueError):
        memory.write(data, cutoff=10, issued_predictions=parts)


def test_order_of_assets_does_not_change_the_published_state():
    from dataclasses import replace

    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig())
    data = feedback(memory, 10)
    parts = torch.tensor([[0.02, 0.0, 0.01], [-0.01, 0.02, 0.0]], dtype=torch.float64)
    order = torch.tensor([1, 0])
    reversed_data = replace(
        data,
        **{
            name: getattr(data, name)[order]
            for name in ("ids", "decision_at", "available_at", "keys", "values", "weights")
        },
    )
    left = memory.write(data, cutoff=10, issued_predictions=parts)
    right = memory.write(reversed_data, cutoff=10, issued_predictions=parts[order])
    assert left.export()["state_sha256"] == right.export()["state_sha256"]
    assert left.numeric_state_bytes == 3 * 64 * 8 + 3 * 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("writes", True),
        ("clock", 20),
        ("cohorts", 2),
        ("matrix_sha256", "wrong"),
        ("configuration", {}),
    ],
)
def test_restore_rejects_corrupted_state(field, value):
    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig())
    data = feedback(memory, 10)
    memory = memory.write(
        data, cutoff=10, issued_predictions=torch.zeros(2, 3, dtype=torch.float64)
    )
    payload = memory.export()
    payload[field] = value
    with pytest.raises(ValueError):
        MultiscaleMemory.restore(BASE, memory.mixture, payload)


def test_clipped_loss_of_a_mixture_need_not_be_below_the_mixture_of_clipped_losses():
    values = torch.tensor([-100.0, 100.0], dtype=torch.float64)
    target = 100.0
    mixture_loss = (values.mean() - target).abs().clamp(max=1.0)
    average_loss = (values - target).abs().clamp(max=1.0).mean()
    assert mixture_loss == 1.0 and average_loss == 0.5


def test_uniform_weights_do_not_adapt_and_empty_feedback_does_not_advance_time():
    config = MultiscaleMixtureConfig(weighting="uniform", weight_rate=0.0)
    memory = MultiscaleMemory(BASE, config)
    data = feedback(memory, 10, values=(0.02, 0.02))
    parts = torch.tensor([[0.02, 0.0, -0.02], [0.02, 0.0, -0.02]], dtype=torch.float64)
    written = memory.write(data, cutoff=10, issued_predictions=parts)
    assert torch.equal(written.weights, torch.full((3,), 1 / 3, dtype=torch.float64))
    empty = BASE.feedback(
        ids=[],
        decision_at=[],
        available_at=[],
        keys=torch.empty(0, 64, dtype=torch.float64),
        values=[],
    )
    assert (
        written.write(empty, cutoff=20, issued_predictions=torch.empty(0, 3, dtype=torch.float64))
        is written
    )
    assert written.clock == 10


def test_query_gradient_passes_gradcheck_inside_the_key_norm_constraint():
    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig())
    data = feedback(memory, 10)
    memory = memory.write(
        data, cutoff=10, issued_predictions=torch.zeros(2, 3, dtype=torch.float64)
    )
    query = (data.keys * 0.5).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda q: memory.read(q, at=10 + DAY_US), (query,), eps=1e-6, atol=1e-8, rtol=1e-5
    )


def test_very_old_coefficients_can_underflow_without_nan_or_mutating_the_state():
    memory = MultiscaleMemory(BASE, MultiscaleMixtureConfig(half_lives_days=(1.0, 2.0, 4.0)))
    data = feedback(memory, 10)
    memory = memory.write(
        data, cutoff=10, issued_predictions=torch.zeros(2, 3, dtype=torch.float64)
    )
    before = memory.export()["state_sha256"]
    output = memory.read(data.keys, at=2**63 - 1)
    assert torch.equal(output, torch.zeros_like(output))
    assert memory.export()["state_sha256"] == before
