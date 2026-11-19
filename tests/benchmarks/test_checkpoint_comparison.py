import pytest
import torch

from benchmarks.comparison_efficiency_20261118.ajustes_con_checkpoint import state_fingerprint


def test_checkpoint_fingerprint_is_order_independent_and_bit_exact():
    value = torch.tensor([[-0.0, 1.0], [2.0, 3.0]])
    a = state_fingerprint({"weights": value, "score": -0.0})
    assert a == state_fingerprint({"score": -0.0, "weights": value.clone()})
    assert a["tensors"] == 1 and a["tensor_values"] == 4
    assert a != state_fingerprint({"weights": value, "score": 0.0})
    assert a != state_fingerprint({"weights": value + 0.0, "score": -0.0})
    assert a != state_fingerprint({"weights": value.T.contiguous().T, "score": -0.0})
    assert a != state_fingerprint({"weights": value.double(), "score": -0.0})


def test_unknown_checkpoint_object_is_rejected():
    with pytest.raises(ValueError, match="sin huella"):
        state_fingerprint({"model": object()})
