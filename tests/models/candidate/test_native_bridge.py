"""Enlace tensorial del candidato nativo, sin ajuste ni etiquetas calculadas."""

import os

import pytest
import torch

from mars_titan.memory.native_backend import load_native

pytestmark = pytest.mark.skipif(
    not os.environ.get("MARS_TITAN_EPISODIC_NATIVE"), reason="Falta el enlace nativo compilado"
)


def test_native_candidate_symbols_are_explicit():
    native = load_native()
    assert native.candidate_abi_version == 1
    config = native.CandidateConfig()
    config.normalization_id = "manual-fixture-v1"
    model = native.Candidate(config, "float64", "cpu")
    assert model.config.input_policy == "strict_inputs_v1"
    assert len(model.parameter_fingerprint()) == 64
    assert all(value.dtype == torch.float64 for value in model.named_parameters().values())


def configured():
    native = load_native()
    config = native.CandidateConfig()
    config.dimensions = [5, 3, 4, 3, 3]
    config.normalization_id = "manual-historical-v1"
    config.input_policy = "historical_masked_2000_v1"
    return native, config


def values():
    return [
        torch.zeros(1, 64, 5, dtype=torch.float64),
        torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64),
        torch.ones(1, 4, dtype=torch.float64),
        torch.tensor([[0, 1, 0]], dtype=torch.float64),
        torch.tensor([[0, 1, 0]], dtype=torch.float64),
        torch.ones(1, 5, dtype=torch.bool),
    ]


def test_tensor_caster_preserves_external_derivatives():
    native, config = configured()
    model = native.Candidate(config, "float64", "cpu")
    data = values()
    news = data[1].clone().requires_grad_()

    def function(value):
        inputs = native.CandidateInputs(data[0], value, *data[2:])
        return model.forward(inputs, model.empty_memory(), 2).quantiles[:, 2]

    assert torch.autograd.gradcheck(function, (news,), eps=1e-6, atol=1e-7, rtol=1e-5)
    assert all(p.grad is None for p in model.named_parameters().values())


def test_config_is_owned_and_constructor_ignores_ambient_device():
    native, config = configured()
    with torch.device("meta"):
        model = native.Candidate(config, "float64", "cpu")
    config.input_policy = "unknown"
    model.config.input_policy = "unknown"
    assert model.config.input_policy == "historical_masked_2000_v1"
    assert all(p.device.type == "cpu" for p in model.named_parameters().values())


@pytest.mark.parametrize("kind", ["dtype", "device", "shape", "presence", "inf", "fraction"])
def test_native_boundary_rejects_incompatible_tensors(kind):
    native, config = configured()
    model = native.Candidate(config, "float64", "cpu")
    data = values()
    if kind == "dtype":
        data[1] = data[1].float()
    elif kind == "device":
        data[1] = torch.empty_like(data[1], device="meta")
    elif kind == "shape":
        data[1] = data[1][:, :2]
    elif kind == "presence":
        data[-1] = data[-1].double()
    elif kind == "inf":
        data[1][0, 0] = float("inf")
    else:
        data[3][0, 1] = 1.00000001
    with pytest.raises(ValueError):
        model.encode(native.CandidateInputs(*data))


@pytest.mark.parametrize(
    "dtype,device", [("float16", "cpu"), ("float64", "meta"), ("float32", "cuda:1")]
)
def test_native_configuration_rejects_unsupported_execution(dtype, device):
    native, config = configured()
    with pytest.raises(ValueError):
        native.Candidate(config, dtype, device)


def test_archive_requires_policy_and_bounds_payload_before_copying():
    native, config = configured()
    model = native.Candidate(config, "float64", "cpu")
    payload = model.save_state()
    with pytest.raises(ValueError):
        native.Candidate.load_state(payload)
    restored = native.Candidate.load_state(payload, "cpu", config.input_policy)
    assert restored.parameter_fingerprint() == model.parameter_fingerprint()
    for invalid in (b"", b"broken archive", b"x" * (128 * 1024**2 + 1)):
        with pytest.raises(ValueError):
            native.Candidate.load_state(invalid, "cpu", config.input_policy)
    with pytest.raises(TypeError):
        native.Candidate.load_state("not bytes", "cpu", config.input_policy)


def test_transfer_rejects_shared_storage_and_nonfinite_origin_before_copying():
    native, config = configured()
    target = native.Candidate(config, "float64", "cpu")
    config.input_policy = "strict_inputs_v1"
    source = native.Candidate(config, "float64", "cpu")
    before = target.parameter_fingerprint()
    source.named_parameters()["head_bias"].data = target.named_parameters()["head_bias"].data
    with pytest.raises(ValueError):
        target.transfer_strict_parameters(source)
    assert target.parameter_fingerprint() == before
    source = native.Candidate(config, "float64", "cpu")
    with torch.no_grad():
        source.named_parameters()["head_bias"][2] = float("nan")
    with pytest.raises(ValueError):
        target.transfer_strict_parameters(source)
    assert target.parameter_fingerprint() == before
