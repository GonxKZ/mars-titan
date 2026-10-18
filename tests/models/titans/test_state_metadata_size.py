"""Recuento exacto y acotado de metadatos, frente a su serialización completa."""

from dataclasses import replace

import pytest
import torch

from mars_titan.models.titans.config import canonical
from tests.models.titans.test_financial_adapter import VARIANTS, api, on_device, setup


def serialized_size(model, state):
    # El formato completo anterior es el oráculo, independiente del recuento optimizado.
    return len(
        canonical(
            model._metadata(state.flow_ids, state.last_sample_ids, state.last_prediction_at)
        ).encode()
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_and_advanced_state_count_the_exact_serialized_bytes(variant, device, dtype):
    model, batch = on_device(device, dtype, variant)
    initial = model.initial_state(batch.flow_ids)
    for state in (initial, model.prepare(batch, initial).next_state):
        usage = model.state_usage(state)
        assert usage["metadata_bytes"] == serialized_size(model, state)
        assert usage["total_bytes"] == usage["metadata_bytes"] + usage["tensor_bytes"]
        restored = model.restore_state(model.export_state(state))
        assert model.state_usage(restored) == usage


def test_repeated_sizing_does_not_rebuild_the_invariant_configuration(monkeypatch):
    model, batch = setup()
    state = model.initial_state(batch.flow_ids)
    expected = serialized_size(model, state)
    original, calls = model.get_extra_state, []

    def counted():
        calls.append(1)
        return original()

    monkeypatch.setattr(model, "get_extra_state", counted)
    for _ in range(20):
        assert model.state_usage(state)["metadata_bytes"] == expected
    assert len(calls) <= 1


def test_sizing_follows_mode_dtype_and_load_without_changing_the_checkpoint_schema():
    model, batch = setup()
    checkpoint_keys = tuple(model.state_dict())
    sizes = []
    for training in (True, False, True):
        model.train(training)
        state = model.initial_state(batch.flow_ids)
        expected = serialized_size(model, state)
        assert model.state_usage(state)["metadata_bytes"] == expected
        sizes.append(expected)
    assert sizes[1] == sizes[0] + 1  # JSON false ocupa un byte más que true.
    assert sizes[2] == sizes[0]
    for dtype in (torch.float32, torch.float64):
        model.to(dtype=dtype)
        state = model.initial_state(batch.flow_ids)
        expected = serialized_size(model, state)
        assert model.state_usage(state)["metadata_bytes"] == expected
        model.load_state_dict(model.state_dict())
        assert model.state_usage(model.initial_state(batch.flow_ids))["metadata_bytes"] == expected
        assert tuple(model.state_dict()) == checkpoint_keys


def test_cached_size_keeps_json_escaping_in_the_configuration():
    model, batch = setup()
    representation = model.config.inputs.representation
    representation["encoders"] = {"fixture": 'línea\n"base"\\v1'}
    spec = api().FinancialInputSpec(
        source_sha256=model.config.inputs.source_sha256,
        view_sha256=model.config.inputs.view_sha256,
        representation=representation,
        dimensions=model.config.inputs.dimensions,
        input_policy=model.config.inputs.input_policy,
    )
    model = api().FinancialPredictor(replace(model.config, inputs=spec), dtype=torch.float64)
    state = model.initial_state(batch.flow_ids)
    assert model.state_usage(state)["metadata_bytes"] == serialized_size(model, state)


def test_the_exact_state_budget_accepts_its_limit_and_rejects_one_byte_less():
    model, batch = setup()
    model = api().FinancialPredictor(
        replace(model.config, max_state_bytes=99999), dtype=torch.float64
    )
    state = model.initial_state(batch.flow_ids)
    required = model.state_usage(state)["total_bytes"]
    assert 10000 < required < 99999
    exact = api().FinancialPredictor(
        replace(model.config, max_state_bytes=required), dtype=torch.float64
    )
    assert exact.state_usage(exact.initial_state(batch.flow_ids))["total_bytes"] == required
    smaller = api().FinancialPredictor(
        replace(model.config, max_state_bytes=required - 1), dtype=torch.float64
    )
    with pytest.raises(ValueError, match="presupuesto de bytes"):
        smaller.initial_state(batch.flow_ids)
