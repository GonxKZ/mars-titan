"""El benchmark de F2 y F3 mide sin pasos de optimizador y sin cambiar parámetros."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

pytest.importorskip("schedulefree")

from benchmarks.weight_options_cost import measure  # noqa: E402


def test_the_measure_covers_each_cost_and_leaves_the_parameters_unchanged():
    with sdpa_kernel(SDPBackend.MATH):
        result = measure(rows=4, instants=1, repetitions=1, warmup=0)
    assert set(result["timings"]) == {
        "weight_ema_update",
        "weight_ema_validation_swap",
        "schedule_free_point",
        "schedule_free_validation_swap",
        "shrink_perturb_once",
        "inference_reference",
    }
    assert all(value["median"] > 0 for value in result["timings"].values())
    assert result["parameters_unchanged"] is True
    memory, parameters = result["memory"], result["parameters"]
    assert memory["weight_ema_shadow_bytes"] == 4 * parameters["trainable"]
    assert memory["weight_ema_extra_state_bytes"] == (
        memory["model_state_bytes"] + memory["weight_ema_shadow_bytes"]
    )
    assert torch.get_default_dtype() == torch.float32
