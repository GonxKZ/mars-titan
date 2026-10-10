"""Anclas neuronales con precisión declarada y lote ampliado, en CPU y sin ajustar nada.

Un proceso nuevo empieza con los valores de PyTorch. La predicción trasladada debe fijar
la política del ancla antes de comparar su entorno, y reconstruir el Transformer con el
lote de su identidad para aceptar su contrato.
"""

import pytest
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.training import carried_predictions as carry
from mars_titan.training import kernel_policy as policy
from tests.training.test_carried_predictions import cpu, neural_anchor, views  # noqa: F401


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


def carried(tmp_path, views, name):  # noqa: F811
    return carry.carry_reference(
        tmp_path / "anchor",
        views.view(0),
        views.view(2),
        tmp_path / name,
        batch_size=2,
        input_policy=HISTORICAL_MASKED,
    )


def test_strict_anchor_is_carried_from_a_fresh_process(views, tmp_path, cpu):  # noqa: F811
    neural_anchor(tmp_path / "anchor", views.view(0), precision=policy.FP32_STRICT)
    torch.backends.cudnn.allow_tf32 = True
    receipt = carried(tmp_path, views, "carry")
    assert receipt["predictions"]
    assert torch.backends.cudnn.allow_tf32 is False


def test_a_legacy_anchor_keeps_the_process_numerics(views, tmp_path, cpu):  # noqa: F811
    torch.backends.cudnn.allow_tf32 = True
    neural_anchor(tmp_path / "anchor", views.view(0))
    carried(tmp_path, views, "carry")
    assert torch.backends.cudnn.allow_tf32 is True
    # Si otro trabajo del proceso fija la política, el ancla sin precisión ya no coincide.
    policy.declared_policy(policy.FP32_STRICT)
    with pytest.raises(ValueError, match="entorno o el código"):
        carried(tmp_path, views, "other")


def test_an_anchor_with_another_recorded_policy_is_rejected(views, tmp_path, cpu, monkeypatch):  # noqa: F811
    neural_anchor(tmp_path / "anchor", views.view(0), precision=policy.FP32_STRICT)
    # Otra versión de PyTorch con otra sustitución nativa registra otra política.
    monkeypatch.setattr(policy, "native_overrides", lambda: ["mm/CUDA/triton"])
    with pytest.raises(ValueError, match="entorno o el código"):
        carried(tmp_path, views, "carry")
    assert not (tmp_path / "carry").exists()


def test_wide_transformer_anchor_rebuilds_its_batch_contract(views, tmp_path, cpu):  # noqa: F811
    neural_anchor(tmp_path / "anchor", views.view(0), kind="transformer", batch_size=512)
    receipt = carried(tmp_path, views, "carry")
    assert receipt["predictions"]
