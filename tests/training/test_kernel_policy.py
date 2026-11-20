"""Política FP32 estricta de los ajustes neuronales y su identidad en recetas y recorridos.

La campaña fija FP32 sin TF32 en cuBLAS ni en cuDNN. Una receta sin precisión declarada
conserva la configuración del proceso y su identidad anterior. No se ejecuta ningún paso.
"""

import subprocess
import sys

import pytest
import torch

from mars_titan.models.titans.config import MAX_BLOCK_ROWS, MAX_STATE_BYTES
from mars_titan.models.titans.episodic_readout import EpisodicReadoutConfig
from mars_titan.training import kernel_policy as policy
from mars_titan.training.candidate_run import CandidateRecipe
from mars_titan.training.financial_run import ChronologicalRecipe
from mars_titan.training.mars_titan_run import ReadoutRecipe
from tests.training.test_financial_run import (
    explicit_fastpath,  # noqa: F401
    shared,  # noqa: F401
    trainer,
)

RECIPES = (ChronologicalRecipe, ReadoutRecipe, CandidateRecipe)


@pytest.fixture(autouse=True)
def restore_flags():
    flags = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    attention = policy.attention_backends()
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
    torch.set_float32_matmul_precision(flags[2])
    torch.backends.cuda.enable_flash_sdp(attention["flash"])
    torch.backends.cuda.enable_mem_efficient_sdp(attention["mem_efficient"])
    torch.backends.cuda.enable_cudnn_sdp(attention["cudnn"])
    torch.backends.cuda.enable_math_sdp(attention["math"])


def test_fp32_strict_disables_tf32_in_cublas_and_cudnn():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    identity = policy.apply_kernel_policy(policy.FP32_STRICT)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert identity["precision"] == "fp32_strict"
    assert identity["matmul_allow_tf32"] is False and identity["cudnn_allow_tf32"] is False
    assert identity["retired_native_overrides"] == ["bmm"]
    assert not any(item.startswith("bmm/") for item in identity["native_overrides"])
    assert policy.kernel_policy_identity(policy.FP32_STRICT) == identity


@pytest.mark.parametrize(
    "flag",
    [
        lambda: setattr(torch.backends.cudnn, "allow_tf32", True),
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", True),
        lambda: torch.set_float32_matmul_precision("high"),
    ],
)
def test_identity_fails_when_the_process_leaves_the_policy(flag):
    identity = policy.apply_kernel_policy(policy.FP32_STRICT)
    flag()
    with pytest.raises(ValueError, match="política de precisión"):
        policy.kernel_policy_identity(policy.FP32_STRICT)
    with pytest.raises(ValueError, match="política de precisión"):
        policy.require_policy(policy.FP32_STRICT, identity)


@pytest.mark.parametrize("precision", ["tf32", "bf16", "FP32_STRICT", True, 1, ""])
def test_only_fp32_strict_is_accepted(precision):
    assert not policy.valid_precision(precision)
    with pytest.raises(ValueError):
        policy.apply_kernel_policy(precision)
    for recipe in RECIPES:
        with pytest.raises(ValueError):
            recipe(precision=precision)


def test_undeclared_precision_keeps_the_process_and_the_previous_identity():
    torch.backends.cudnn.allow_tf32 = True
    assert policy.declared_policy(None) is None
    assert torch.backends.cudnn.allow_tf32 is True
    policy.require_policy(None, None)
    for recipe in RECIPES:
        assert "precision" not in recipe().identity()
        declared = recipe(precision=policy.FP32_STRICT).identity()
        assert declared["precision"] == "fp32_strict"
        assert {k: v for k, v in declared.items() if k != "precision"} == recipe().identity()


def test_block_bounds_share_one_constant():
    assert ChronologicalRecipe(block_rows=MAX_BLOCK_ROWS, accumulation_rows=MAX_BLOCK_ROWS)
    assert ReadoutRecipe(block_rows=MAX_BLOCK_ROWS, max_working_bytes=MAX_STATE_BYTES)
    assert EpisodicReadoutConfig("a" * 64, max_batch=MAX_BLOCK_ROWS)
    for options in (
        dict(block_rows=MAX_BLOCK_ROWS + 1),
        dict(max_working_bytes=MAX_STATE_BYTES + 1),
    ):
        with pytest.raises(ValueError):
            ReadoutRecipe(**options)
    with pytest.raises(ValueError):
        EpisodicReadoutConfig("a" * 64, max_working_bytes=MAX_STATE_BYTES + 1)


def test_declared_precision_enters_the_run_identity_and_is_checked(shared, tmp_path):  # noqa: F811
    _, streams = shared
    torch.backends.cudnn.allow_tf32 = True
    plain = trainer(streams, tmp_path / "plain")
    assert "kernel_policy" not in plain.identity
    assert plain.identity["numerics"]["cudnn_allow_tf32"] is True
    strict = trainer(streams, tmp_path / "strict", precision=policy.FP32_STRICT)
    assert strict.identity["kernel_policy"]["precision"] == "fp32_strict"
    assert strict.identity["numerics"]["cudnn_allow_tf32"] is False
    assert strict.identity["recipe"]["precision"] == "fp32_strict"
    assert strict.run_id != plain.run_id
    strict._check_runtime()
    torch.backends.cudnn.allow_tf32 = True
    with pytest.raises(ValueError, match="configuración numérica"):
        strict._check_runtime()


def test_fp32_ieee_attention_leaves_only_the_math_kernel():
    identity = policy.apply_kernel_policy(policy.FP32_IEEE_ATTENTION)
    assert policy.attention_backends() == policy.MATH_ONLY == identity["attention_backends"]
    assert identity["precision"] == "fp32_ieee_attention_v1"
    assert policy.kernel_policy_identity(policy.FP32_IEEE_ATTENTION) == identity
    # El mismo proceso cumple también fp32_strict, cuya identidad no nombra la atención.
    strict = policy.kernel_policy_identity(policy.FP32_STRICT)
    assert "attention_backends" not in strict
    shared_fields = {k: v for k, v in identity.items() if k != "attention_backends"}
    assert shared_fields == dict(strict, precision="fp32_ieee_attention_v1")


@pytest.mark.parametrize(
    "leave",
    [
        lambda: torch.backends.cuda.enable_flash_sdp(True),
        lambda: torch.backends.cuda.enable_mem_efficient_sdp(True),
        lambda: torch.backends.cuda.enable_cudnn_sdp(True),
        lambda: torch.backends.cuda.enable_math_sdp(False),
    ],
    ids=["flash", "mem_efficient", "cudnn", "math"],
)
def test_the_ieee_identity_fails_when_the_attention_leaves_the_math_kernel(leave):
    identity = policy.apply_kernel_policy(policy.FP32_IEEE_ATTENTION)
    leave()
    with pytest.raises(ValueError, match="política de precisión"):
        policy.kernel_policy_identity(policy.FP32_IEEE_ATTENTION)
    with pytest.raises(ValueError, match="política de precisión"):
        policy.require_policy(policy.FP32_IEEE_ATTENTION, identity)
    # fp32_strict no restringe la atención: su comprobación no depende de estos núcleos.
    policy.kernel_policy_identity(policy.FP32_STRICT)


def test_fp32_strict_does_not_touch_the_attention_kernels():
    before = policy.attention_backends()
    policy.apply_kernel_policy(policy.FP32_STRICT)
    assert policy.attention_backends() == before


def test_the_ieee_precision_enters_every_recipe_identity():
    for recipe in RECIPES:
        declared = recipe(precision=policy.FP32_IEEE_ATTENTION).identity()
        assert declared["precision"] == "fp32_ieee_attention_v1"
        assert {k: v for k, v in declared.items() if k != "precision"} == recipe().identity()


CUDA_CHECK = """
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as attention

from mars_titan.training import kernel_policy as policy

generator = torch.Generator(device="cuda").manual_seed(7)
q, k, v = (
    torch.randn(512, 2, 9, 32, device="cuda", generator=generator, requires_grad=True)
    for _ in range(3)
)
with sdpa_kernel(SDPBackend.MATH):
    reference = attention(q, k, v).detach()
exact = attention(q.detach().double(), k.detach().double(), v.detach().double())
policy.apply_kernel_policy(policy.FP32_IEEE_ATTENTION)
gradients = []
for _ in range(2):
    output = attention(q, k, v)
    gradients.append(torch.autograd.grad(output.square().sum(), (q, k, v)))
    assert torch.equal(output.detach(), reference)
    error = (output.detach().double() - exact).abs().max() / exact.abs().max()
    assert error.item() < 4e-7, error.item()
for first, second in zip(*gradients, strict=True):
    assert torch.equal(first, second)
print("ok")
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Necesita CUDA")
def test_on_cuda_the_ieee_attention_is_the_math_kernel_and_repeats_its_gradient():
    # En otro proceso: iniciar CUDA aquí cambiaría lo que ven las pruebas del determinismo.
    done = subprocess.run(
        [sys.executable, "-c", CUDA_CHECK], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]
