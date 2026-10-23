"""Backend de precomputación opcional, con la referencia y sus archivos compatibles."""

from copy import deepcopy

import pytest
from test_kronlstm import api, encoder, prices

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]


def test_default_reference_keeps_the_legacy_configuration_cache_and_checkpoint():
    import torch

    legacy = dict(
        schema_version=1,
        kind="compact_price_kronlstm",
        input_size=3,
        context=8,
        hidden_size=32,
        layers=2,
        heads=4,
        rule="scalar",
        intermediate_size=256,
        upstream_commit="3d800c9a153a88ee9624fe6acf7f39a58017a17e",
        causal=True,
        state_policy="reset_per_window",
        aggregation="last_token",
        backend="pytorch_reference_cpu_qkv64_v1",
        cpu_float32_qkv_accumulation="float64",
        conv_size=4,
        normalize_v=True,
        normalize_qk=True,
        decouple_erase=True,
        allow_neg_eigval=False,
        decay_gate="vector",
        activation="swiglu",
        normalization="rms_norm",
        norm_eps=1e-5,
        gate_precision="float32",
        v_norm_floor=1e-18,
        output_gate_bias=3.0,
        max_batch=4,
        max_recurrent_elements=2**24,
        dtypes=["float32", "float64"],
    )
    block_contract = dict(
        kind="kron_block",
        hidden=32,
        heads=4,
        head_width=8,
        rule="scalar",
        conv_size=4,
        normalize_qk=True,
        normalize_v=True,
        decay_gate="vector",
        decouple_erase=True,
        gate_precision="float32",
        cpu_float32_qkv_accumulation="float64",
        activation="swiglu",
    )
    default, explicit = encoder().eval(), encoder(execution="reference").eval()
    assert default.configuration == explicit.configuration == legacy
    assert all(block.configuration == block_contract for block in default.blocks)
    incoming = deepcopy(default.state_dict())
    assert incoming["_extra_state"] == legacy
    explicit.load_state_dict(incoming)
    with torch.no_grad():
        left, left_cache = default.encode_steps(prices("cpu", "float64")[:, :3])
        right, right_cache = explicit.encode_steps(prices("cpu", "float64")[:, :3])
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for key in ("configuration", "weights_sha256", "cursor"):
        assert left_cache[key] == right_cache[key]
    rebound = explicit.restore_state(default.export_state(left_cache))
    assert rebound["configuration"] == legacy and rebound["cursor"] == 3


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("rule", ["scalar", "vector"])
def test_precomputed_encoder_preserves_parameters_outputs_and_every_gradient(device, dtype, rule):
    import torch

    reference = encoder(device=device, dtype=dtype, rule=rule, execution="reference").eval()
    candidate = encoder(device=device, dtype=dtype, rule=rule, execution="precomputed").eval()
    assert candidate.configuration["backend"] == "pytorch_precomputed_cpu_qkv64_v1"
    left, right = dict(reference.named_parameters()), dict(candidate.named_parameters())
    assert left.keys() == right.keys()
    assert [(name, type(module)) for name, module in reference.named_modules()] == [
        (name, type(module)) for name, module in candidate.named_modules()
    ]
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(0).clone() if device == "cuda:0" else None
    results, gradients = [], []
    for model in (reference, candidate):
        inputs = prices(device, dtype).requires_grad_()
        result = model.encode_sequence(inputs)
        results.append(result)
        gradients.append(torch.autograd.grad(result.square().mean(), (inputs, *model.parameters())))
    forward = dict(rtol=1e-5, atol=1e-6) if dtype == "float32" else dict(rtol=1e-12, atol=1e-12)
    backward = dict(rtol=2e-5, atol=2e-6) if dtype == "float32" else dict(rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(results[0], results[1], **forward)
    for left_grad, right_grad in zip(*gradients, strict=True):
        assert torch.isfinite(left_grad).all() and torch.isfinite(right_grad).all()
        torch.testing.assert_close(left_grad, right_grad, **backward)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)


@pytest.mark.parametrize("execution", [None, False, "", "auto", 1])
def test_invalid_execution_fails_before_consuming_initialization_rng(execution):
    import torch

    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="execution|ejecución|backend"):
        api().CompactPriceKronLSTM(3, hidden_size=32, execution=execution)
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_each_backend_restores_its_cache_and_rejects_cross_backend_payloads(device, dtype):
    import torch

    inputs = prices(device, dtype)
    for execution, other in (("reference", "precomputed"), ("precomputed", "reference")):
        model = encoder(device=device, dtype=dtype, execution=execution).eval()
        restored = encoder(device=device, dtype=dtype, execution=execution).eval()
        incompatible = encoder(device=device, dtype=dtype, execution=other).eval()
        restored.load_state_dict(deepcopy(model.state_dict()))
        with torch.no_grad():
            expected = model.encode_sequence(inputs)
            first, cache = model.encode_steps(inputs[:, :3])
            payload = model.export_state(cache)
            previous = deepcopy(cache)
            resumed = restored.restore_state(payload)
            second, finished = restored.encode_steps(inputs[:, 3:], resumed)
        assert cache["cursor"] == resumed["cursor"] == 3 and finished["cursor"] == 8
        torch.testing.assert_close(
            torch.cat((first, second), dim=1), expected, rtol=1e-5, atol=1e-6
        )
        for actual, before in zip(cache["layers"], previous["layers"], strict=True):
            for name in actual:
                torch.testing.assert_close(actual[name], before[name], rtol=0, atol=0)
        with pytest.raises(ValueError, match="contrato|configuración|backend"):
            incompatible.restore_state(payload)
        checkpoint = deepcopy(model.state_dict())
        checkpoint["projection.weight"].add_(1)
        before = incompatible.projection.weight.detach().clone()
        with pytest.raises(ValueError, match="contrato|configuración|backend"):
            incompatible.load_state_dict(checkpoint)
        torch.testing.assert_close(incompatible.projection.weight, before, rtol=0, atol=0)
