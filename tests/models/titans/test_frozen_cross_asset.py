"""Consumidor congelado con atención entre activos de la misma sesión (C5)."""

import hashlib
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_cross_asset_attention import CONTROL, MARKETS, VARIANT
from test_episodic_readout import CONTEXT, reader
from test_financial_adapter import setup
from test_frozen_financial import bank, consumer, frozen
from test_session_rank_objective import session_batch

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.cross_asset_attention import (
    CrossAssetAttention,
    CrossAssetAttentionConfig,
)
from mars_titan.models.titans.episodic_readout import apply_episodic_readout

CROSS_ASSET_MODULE = "mars_titan.models.titans.cross_asset_attention"


@pytest.fixture(autouse=True)
def frozen_backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


def _c5_predictor(config=VARIANT, *, variant="mac_online", nonzero_out=False):
    base, batch = setup(variant)
    predictor = type(base)(replace(base.config, cross_asset_attention=config), dtype=torch.float64)
    if nonzero_out:
        with torch.no_grad():
            generator = torch.Generator().manual_seed(17)
            weight = predictor.cross_asset_attention.out.weight
            weight.copy_(torch.randn(weight.shape, generator=generator, dtype=weight.dtype) / 4)
        predictor._seal_parameters()
    predictor.eval().requires_grad_(False)
    return predictor, batch


def _c5_consumer(config=VARIANT, *, variant="mac_online", refinements=None, nonzero_out=False):
    predictor, batch = _c5_predictor(config, variant=variant, nonzero_out=nonzero_out)
    extension = None if refinements is None else reader(refinements=refinements)
    if extension is not None:
        extension.eval().requires_grad_(False)
    return consumer(predictor, extension), batch


def _with_two_instants(batch):
    """Construir una entrada con dos instantes de decisión distintos."""
    instants = (batch.prediction_at[0], batch.prediction_at[0] + 1, *batch.prediction_at[2:])
    sample_ids = tuple(f"{flow}/{at}" for flow, at in zip(batch.flow_ids, instants, strict=True))
    try:
        return replace(batch, prediction_at=instants, sample_ids=sample_ids)
    except (TypeError, ValueError):
        return SimpleNamespace(
            flow_ids=batch.flow_ids,
            sample_ids=sample_ids,
            prediction_at=instants,
            inputs=batch.inputs,
            presence=batch.presence,
            input_contract_id=batch.input_contract_id,
            input_digest=batch.input_digest,
        )


def test_identity_without_cross_asset_attention_matches_existing_functions(monkeypatch, tmp_path):
    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    base_engine, _ = frozen("mac_online")
    base_model, _ = setup("mac_online")
    explicit_model = (
        type(base_model)(
            replace(base_model.config, cross_asset_attention=None), dtype=torch.float64
        )
        .eval()
        .requires_grad_(False)
    )
    explicit_engine = consumer(explicit_model)

    assert CROSS_ASSET_MODULE not in api._SOURCE_MODULES
    assert api.declared_sources(base_engine.predictor.config) == ()
    assert api.declared_sources(explicit_model.config) == ()
    assert api.declared_sources(SimpleNamespace()) == ()
    assert api.declared_sources(
        SimpleNamespace(technical_branch="declared", cross_asset_attention=None)
    ) == ("mars_titan.models.titans.technical_branch",)

    expected_code = api._implementation(kronlstm=explicit_model.config.temporal_encoder is not None)
    signature, modes = api._execution_signature((("predictor", explicit_model),))
    expected_identity = dict(
        schema_version=1,
        recipe="frozen_financial_mac_then_episodic_v1",
        predictor=explicit_model.get_extra_state(),
        predictor_parameters_sha256=explicit_model._parameter_id,
        readout=None,
        readout_parameters_sha256=None,
        implementation=expected_code,
        numerics=api._numerics(),
        modules=modes,
        dtype=str(explicit_model.head.weight.dtype),
        device=str(explicit_model.head.weight.device),
        shared_parameters="frozen_no_optimizer",
        memory_updates="once_per_observation",
        local_control_scope="mac_transition_before_episodic_readout",
    )
    expected_model_id = hashlib.sha256(canonical(expected_identity).encode()).hexdigest()

    assert CROSS_ASSET_MODULE not in base_engine.identity()["implementation"]
    assert "predictor/cross_asset_attention" not in base_engine.identity()["modules"]
    assert "cross_asset_attention" not in base_engine.identity()["predictor"]["configuration"]
    assert base_engine.identity()["implementation"] == expected_code
    assert explicit_engine.identity() == base_engine.identity()
    assert explicit_engine.model_id == base_engine.model_id == expected_model_id
    assert explicit_engine._signature == signature

    with_readout, _ = frozen("mac_online", refinements=1)
    assert CROSS_ASSET_MODULE not in with_readout.identity()["implementation"]
    assert set(with_readout.identity()["implementation"]) == set(expected_code)

    # Cambiar cross_asset_attention.py no invalida un consumidor que no declara C5.
    original = api.importlib.import_module
    changed = tmp_path / "different_c5.py"
    changed.write_text("different implementation\n")

    def redirected(name):
        if name == CROSS_ASSET_MODULE:
            return SimpleNamespace(__file__=str(changed))
        return original(name)

    monkeypatch.setattr(api.importlib, "import_module", redirected)
    base_engine.verify()


def test_cross_asset_consumer_registers_identity_and_changes_with_config():
    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    base_engine, _ = frozen("mac_online")
    same_session_4, _ = _c5_consumer(VARIANT)
    same_session_4_repeat, _ = _c5_consumer(VARIANT)
    self_only_4, _ = _c5_consumer(CONTROL)
    same_session_2, _ = _c5_consumer(CrossAssetAttentionConfig(heads=2, mode="same_session"))

    assert api.declared_sources(same_session_4.predictor.config) == (CROSS_ASSET_MODULE,)
    assert api.declared_sources(
        SimpleNamespace(technical_branch="declared", cross_asset_attention=VARIANT)
    ) == ("mars_titan.models.titans.technical_branch", CROSS_ASSET_MODULE)

    identity = same_session_4.identity()
    expected_code = api._implementation(
        kronlstm=same_session_4.predictor.config.temporal_encoder is not None,
        declared=(CROSS_ASSET_MODULE,),
    )
    assert identity["implementation"] == expected_code
    assert identity["predictor"]["configuration"]["cross_asset_attention"] == VARIANT.identity()

    hidden = same_session_4.predictor.config.hidden_size
    layer_entry = identity["modules"]["predictor/cross_asset_attention"]
    assert layer_entry == dict(
        type="mars_titan.models.titans.cross_asset_attention.CrossAssetAttention",
        training=False,
        settings=dict(
            cross_asset_attention=VARIANT.identity(),
            hidden_size=hidden,
            head_dim=hidden // VARIANT.heads,
        ),
    )

    assert same_session_4.model_id == same_session_4_repeat.model_id
    model_ids = {
        base_engine.model_id,
        same_session_4.model_id,
        self_only_4.model_id,
        same_session_2.model_id,
    }
    assert len(model_ids) == 4
    same_session_4.verify()
    self_only_4.verify()
    same_session_2.verify()

    with_readout, _ = _c5_consumer(VARIANT, refinements=2)
    assert CROSS_ASSET_MODULE in with_readout.identity()["implementation"]
    assert with_readout.model_id != same_session_4.model_id
    with_readout.verify()


def test_cross_asset_source_change_invalidates_strong_verification(monkeypatch, tmp_path):
    api = importlib.import_module("mars_titan.models.titans.frozen_financial")
    engine, _ = _c5_consumer(VARIANT)
    original = api.importlib.import_module
    changed = tmp_path / "different_cross_asset.py"
    changed.write_text("different implementation\n")

    def redirected(name):
        if name == CROSS_ASSET_MODULE:
            return SimpleNamespace(__file__=str(changed))
        return original(name)

    monkeypatch.setattr(api.importlib, "import_module", redirected)
    with pytest.raises(ValueError, match="código"):
        engine.verify()


@pytest.mark.parametrize("change", ["config", "head_dim", "train", "data_bypass"])
def test_cross_asset_tampering_after_sealing_is_detected_by_verify(change):
    engine, _ = _c5_consumer(VARIANT, nonzero_out=True)
    layer = engine.predictor.cross_asset_attention
    if change == "config":
        layer.config = CONTROL
    elif change == "head_dim":
        layer.head_dim = layer.head_dim // 2
    elif change == "train":
        layer.train()
    else:
        layer.out.weight.data.add_(1)
    with pytest.raises(ValueError):
        engine.verify(strong=change == "data_bypass")


@pytest.mark.parametrize("tamper", ["config_mismatch", "missing_module", "bad_head_dim", "extra"])
def test_inconsistent_cross_asset_module_is_rejected_at_construction(tamper):
    if tamper == "extra":
        predictor, _ = setup("mac_online")
        predictor.cross_asset_attention = CrossAssetAttention(
            predictor.config.hidden_size, VARIANT, seed=predictor.config.seed, dtype=torch.float64
        )
        predictor._seal_parameters()
        predictor.eval().requires_grad_(False)
    else:
        predictor, _ = _c5_predictor(VARIANT)
        if tamper == "config_mismatch":
            predictor.cross_asset_attention.config = CONTROL
        elif tamper == "missing_module":
            predictor.cross_asset_attention = None
        else:
            predictor.cross_asset_attention.head_dim = 1
    with pytest.raises(ValueError, match="atención entre activos"):
        consumer(predictor)


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
def test_consumer_matches_predictor_bitwise_on_complete_session(config):
    engine, _ = _c5_consumer(config, nonzero_out=True)
    batch = session_batch(6, markets=list(MARKETS))
    state = engine.predictor.initial_state(batch.flow_ids)

    base, _ = setup("mac_online")
    base.eval().requires_grad_(False)
    baseline = base.prepare(batch, base.initial_state(batch.flow_ids))

    expected = engine.predictor.prepare(batch, state, session_complete=True)
    assert not torch.equal(expected.point_predictions, baseline.point_predictions)

    actual = engine.prepare(batch, state, context_id=CONTEXT, session_complete=True)
    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    assert actual.next_state.observed_steps.tolist() == expected.next_state.observed_steps.tolist()
    torch.testing.assert_close(
        actual.next_state.mac.memory.weights[0],
        expected.next_state.mac.memory.weights[0],
        rtol=0,
        atol=0,
    )
    assert not actual.point_predictions.requires_grad

    warmup = engine.prepare(batch, state, context_id=CONTEXT, warmup=True, session_complete=True)
    assert warmup.point_predictions is None and warmup.readout is None
    assert warmup.next_state.observed_steps.tolist() == expected.next_state.observed_steps.tolist()


def test_consumer_with_readout_matches_predictor_and_readout_bitwise():
    engine, batch = _c5_consumer(VARIANT, refinements=2, nonzero_out=True)
    state = engine.predictor.initial_state(batch.flow_ids)
    snapshot = bank(batch)

    prepared = engine.predictor.prepare(batch, state, differentiable=False, session_complete=True)
    expected = apply_episodic_readout(
        prepared,
        engine.predictor.head,
        engine.readout,
        snapshot=snapshot,
        context_id=CONTEXT,
        cutoff=batch.prediction_at[0],
        differentiable=False,
    )
    actual = engine.prepare(
        batch, state, snapshot=snapshot, context_id=CONTEXT, session_complete=True
    )

    torch.testing.assert_close(actual.point_predictions, expected.point_predictions, rtol=0, atol=0)
    torch.testing.assert_close(actual.readout.state, expected.readout.state, rtol=0, atol=0)
    assert actual.next_state.mac.memory.steps.tolist() == [1] * len(batch.flow_ids)
    assert not actual.point_predictions.requires_grad
    assert all(p.grad is None for p in engine.predictor.parameters())


@pytest.mark.parametrize("config", [VARIANT, CONTROL])
@pytest.mark.parametrize("session_complete", [False, None, 1])
def test_incomplete_session_is_rejected_for_cross_asset_consumer(config, session_complete):
    engine, _ = _c5_consumer(config, nonzero_out=True)
    batch = session_batch(6, markets=list(MARKETS))
    state = engine.predictor.initial_state(batch.flow_ids)

    with pytest.raises(ValueError, match="sesión completa"):
        engine.prepare(batch, state, context_id=CONTEXT)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.prepare(batch, state, context_id=CONTEXT, session_complete=session_complete)
    with pytest.raises(ValueError, match="sesión completa"):
        engine.prepare(
            batch, state, context_id=CONTEXT, warmup=True, session_complete=session_complete
        )


@pytest.mark.parametrize("warmup", [False, True])
def test_mixed_instants_are_rejected_even_with_session_complete(warmup):
    engine, _ = _c5_consumer(VARIANT, nonzero_out=True)
    batch = session_batch(6, markets=list(MARKETS))
    state = engine.predictor.initial_state(batch.flow_ids)
    mixed = _with_two_instants(batch)

    with pytest.raises(ValueError, match="evento y corte únicos"):
        engine.prepare(mixed, state, context_id=CONTEXT, warmup=warmup, session_complete=True)
