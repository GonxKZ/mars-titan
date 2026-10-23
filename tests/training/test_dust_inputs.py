"""Validar entradas de Dust antes del cálculo, sin alterar pesos ni aleatoriedad."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest
import torch
from test_dust_financial import assert_unchanged, estimator_for, financial_case, snapshots
from test_financial_adapter import setup


@pytest.fixture(scope="module")
def cpu_case():
    model, batch = financial_case()
    return model, batch, estimator_for(model)


@pytest.fixture(scope="module")
def cuda_case():
    model, batch = financial_case(device="cuda:0")
    return model, batch, estimator_for(model)


def rejected_before_core(case, inputs, presence, monkeypatch, *, rows=2, match=None):
    model, _, estimator = case
    weights = snapshots(model)
    private = estimator.state_dict()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = (
        torch.cuda.get_rng_state(0).clone() if estimator.adapter.device.type == "cuda" else None
    )
    tensors = (
        {
            name: value.detach().clone()
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        if isinstance(inputs, Mapping)
        else {}
    )
    flags = presence.clone() if isinstance(presence, torch.Tensor) else None
    target = torch.zeros(rows, dtype=estimator.adapter.dtype, device=estimator.adapter.device)

    def forbidden(*args, **kwargs):
        pytest.fail("Una entrada rechazada alcanzó el núcleo de Dust")

    monkeypatch.setattr(estimator.adapter, "forward", forbidden)
    monkeypatch.setattr(estimator.adapter, "_instrumented", forbidden)
    with pytest.raises(ValueError, match=match):
        estimator.estimate(inputs, presence, target)
    assert_unchanged(model, weights)
    after = estimator.state_dict()
    assert torch.equal(after["rng"], private["rng"])
    assert after["steps"] == private["steps"]
    assert after["totals"] == private["totals"]
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
    for name, expected in tensors.items():
        actual = inputs[name]
        assert actual.layout == expected.layout
        if actual.layout != torch.strided:
            actual, expected = actual.to_dense(), expected.to_dense()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if flags is not None:
        assert torch.equal(presence, flags)


@pytest.mark.parametrize(
    "fault",
    [
        "not_mapping",
        "missing_modality",
        "extra_modality",
        "prices_not_tensor",
        "prices_rank",
        "batch_zero",
        "batch_above_limit",
        "modality_not_tensor",
        "price_width",
        "modality_width",
        "modality_dtype",
        "sparse_modality",
        "presence_not_tensor",
        "presence_float",
        "presence_shape",
        "presence_missing",
        "nan_in_absent_modality",
        "infinity_in_absent_modality",
    ],
)
def test_invalid_inputs_fail_before_clean_or_perturbed_core(cpu_case, monkeypatch, fault):
    _, batch, estimator = cpu_case
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    presence = batch.presence.clone()
    rows = len(presence)
    if fault == "not_mapping":
        inputs = None
    elif fault == "missing_modality":
        inputs.pop("news")
    elif fault == "extra_modality":
        inputs["unknown"] = inputs["news"].clone()
    elif fault == "prices_not_tensor":
        inputs["prices"] = inputs["prices"].tolist()
    elif fault == "prices_rank":
        inputs["prices"] = inputs["prices"][:, 0]
    elif fault == "batch_zero":
        inputs = {name: value[:0] for name, value in inputs.items()}
        presence, rows = presence[:0], 0
    elif fault == "batch_above_limit":
        rows = estimator.adapter.max_batch + 1
        indices = torch.zeros(rows, dtype=torch.long)
        inputs = {name: value.index_select(0, indices) for name, value in inputs.items()}
        presence = presence.index_select(0, indices)
    elif fault == "modality_not_tensor":
        inputs["news"] = inputs["news"].tolist()
    elif fault == "price_width":
        inputs["prices"] = inputs["prices"][..., :-1]
    elif fault == "modality_width":
        inputs["news"] = inputs["news"][:, :-1]
    elif fault == "modality_dtype":
        inputs["macro"] = inputs["macro"].float()
    elif fault == "sparse_modality":
        inputs["charts"] = inputs["charts"].to_sparse()
    elif fault == "presence_not_tensor":
        presence = presence.tolist()
    elif fault == "presence_float":
        presence = presence.double()
    elif fault == "presence_shape":
        presence = presence[:, :-1]
    elif fault == "presence_missing":
        presence = None
    elif fault == "nan_in_absent_modality":
        assert presence[1, 1].item() is False
        inputs["news"][1, 0] = float("nan")
    elif fault == "infinity_in_absent_modality":
        assert presence[1, 3].item() is False
        inputs["fundamentals"][1, 0] = float("inf")
    rejected_before_core(cpu_case, inputs, presence, monkeypatch, rows=rows)


def test_cpu_autocast_is_rejected_before_any_core(cpu_case, monkeypatch):
    _, batch, _ = cpu_case
    with torch.autocast("cpu", dtype=torch.bfloat16):
        rejected_before_core(cpu_case, batch.inputs, batch.presence, monkeypatch, match="autocast")


@pytest.mark.cuda
def test_cuda_autocast_is_rejected_before_any_core(cuda_case, monkeypatch):
    _, batch, _ = cuda_case
    with torch.autocast("cuda", dtype=torch.float16):
        rejected_before_core(cuda_case, batch.inputs, batch.presence, monkeypatch, match="autocast")


@pytest.mark.cuda
@pytest.mark.parametrize("field", ["news", "presence"])
def test_cuda_inputs_cannot_mix_cpu_tensors_before_any_core(cuda_case, monkeypatch, field):
    _, batch, _ = cuda_case
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    presence = batch.presence.clone()
    if field == "presence":
        presence = presence.cpu()
    else:
        inputs[field] = inputs[field].cpu()
    rejected_before_core(cuda_case, inputs, presence, monkeypatch)


def test_valid_readonly_mapping_passes_the_real_metadata_validator(cpu_case):
    model, batch, estimator = cpu_case
    weights = snapshots(model)
    parent_rng = torch.random.get_rng_state().clone()
    rows = estimator.adapter.validate_inputs(MappingProxyType(dict(batch.inputs)), batch.presence)
    assert rows == len(batch.flow_ids)
    assert_unchanged(model, weights)
    assert torch.equal(torch.random.get_rng_state(), parent_rng)


def test_strict_policy_accepts_presence_omission_in_the_real_validator():
    model, batch = setup("transformer_direct", historical=False)
    estimator = estimator_for(model)
    weights = snapshots(model)
    parent_rng = torch.random.get_rng_state().clone()
    assert model.masked is False
    assert estimator.adapter.validate_inputs(batch.inputs, None) == len(batch.flow_ids)
    assert_unchanged(model, weights)
    assert torch.equal(torch.random.get_rng_state(), parent_rng)
