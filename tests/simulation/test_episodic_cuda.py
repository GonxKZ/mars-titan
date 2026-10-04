"""Recuperación de los comparadores episódicos con CUDA y memoria reconstruida."""

import json
import os

import pytest
from test_native_ppo_runner import adaptive_hmm, execute, latest, read, trace_rows
from test_native_ppo_runner import adaptive_inputs as adaptive_inputs

from mars_titan.training.experiment_resources import GpuLease


def adam_states(policy):
    """Relacionar los momentos con el orden de parámetros, sin comparar direcciones de proceso."""
    groups = policy.optimizer.param_groups
    result = []
    for group_index in range(int(getattr(groups, "param_groups/size").item())):
        group = getattr(groups, f"param_groups/{group_index}")
        for index in range(int(getattr(group, "params/size").item())):
            identifier = str(getattr(group, f"params/{index}"))
            state = getattr(policy.optimizer.state, identifier)
            result.append((state.step, state.exp_avg, state.exp_avg_sq))
    return result


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="Requiere una ventana CUDA exclusiva y activación explícita",
)
@pytest.mark.parametrize("variant", ["ppo_episodic", "ppo_episodic_hmm"])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_episodic_cuda_resume_preserves_policy_trace_and_selection(
    adaptive_inputs, tmp_path, variant, seed
):
    import torch

    if variant == "ppo_episodic_hmm":
        adaptive_hmm(adaptive_inputs)
    config, paths = adaptive_inputs
    document = read(config)
    document["agent"]["variant"] = variant
    document["training"].update(seed=seed, total_transitions=96)
    config.write_text(json.dumps(document))
    primary = config, paths["train"][0], paths["validation"][0]
    sources = [
        value
        for split in ("train", "validation")
        for source in paths[split][1:]
        for value in (f"--{split}-tape", str(source))
    ]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    with GpuLease(max_vram=1024**3) as lease:
        torch.cuda.set_device("cuda:0")
        descriptor = lease.handle.fileno()
        device = [
            "--device",
            "cuda:0",
            "--gpu-lease-fd",
            str(descriptor),
            "--vram-budget-bytes",
            str(lease.record["max_vram_bytes"]),
            "--vram-total-bytes",
            str(torch.cuda.get_device_properties(0).total_memory),
        ]
        options = dict(
            diagnostic=False,
            pass_fds=(descriptor,),
            environment={
                "CUDA_VISIBLE_DEVICES": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        execute(primary, continuous, *sources, *device, **options)
        execute(primary, resumed, *sources, *device, "--stop-after", "48", paused=True, **options)
        assert read(resumed / "run.json")["transitions"] == 48
        execute(primary, resumed, *sources, *device, "--resume", **options)
    expected, actual = read(continuous / "run.json"), read(resumed / "run.json")
    for field in ("transitions", "observed_transitions", "optimizer_steps", "evaluations", "best"):
        assert actual[field] == expected[field]
    assert actual["device"] == "cuda:0" and actual["diagnostic"] is False
    assert actual["status"] == "completed" and actual["transitions"] == 96
    assert actual["optimizer_steps"] > 0 and actual["final_test_opened"] is False
    assert trace_rows(continuous) == trace_rows(resumed)
    archives = [
        torch.jit.load(str(latest(output) / "policy.pt"), map_location="cpu")
        for output in (continuous, resumed)
    ]
    states = [archive.state_dict() for archive in archives]
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    expected_adam, actual_adam = map(adam_states, archives)
    assert len(expected_adam) == len(actual_adam) == 6
    for expected, actual in zip(expected_adam, actual_adam, strict=True):
        assert expected[0] == actual[0]
        assert torch.equal(expected[1], actual[1])
        assert torch.equal(expected[2], actual[2])
