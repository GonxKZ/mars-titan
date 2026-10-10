"""Política FP32 estricta de los ajustes neuronales y su identidad en recetas y recorridos.

La campaña fija FP32 sin TF32 en cuBLAS ni en cuDNN. Una receta sin precisión declarada
conserva la configuración del proceso y su identidad anterior. No se ejecuta ningún paso.
"""

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
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[:2]
    torch.set_float32_matmul_precision(flags[2])


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
