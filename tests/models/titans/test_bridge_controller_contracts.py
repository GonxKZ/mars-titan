"""Identidad del selector y límite de trazas sobre toda la cohorte."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_bridge_episodic_readout import CODEC, bank, configuration, source
from test_bridge_mechanism_routes import CONTRACT, engine
from test_frozen_financial import frozen_backend as frozen_backend
from test_market_memory import pt4
from test_mechanism_policy import policy

from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.mechanism_catalog import MechanismAction
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.mechanism_policy import MechanismPolicy


def bound_policy(catalog_id):
    parent = policy()
    return MechanismPolicy(
        **dict(parent.named_buffers()),
        interval_weight=parent.interval_weight,
        min_gain=parent.min_gain,
        catalog_id=catalog_id,
    )


@pytest.mark.parametrize("identity", [None, "a" * 64])
def test_a_policy_from_another_reader_contract_cannot_choose_regional_actions(identity):
    parent = engine()
    with pytest.raises(ValueError):
        FrozenMechanismConsumer(
            parent.predictor,
            readout=parent.readout,
            catalog=parent.catalog,
            policy=bound_policy(identity),
        )


def test_bound_policy_checkpoint_and_frozen_identity_preserve_the_declared_catalog():
    parent = engine()
    selector = bound_policy(parent.catalog.fingerprint())
    restored = bound_policy(parent.catalog.fingerprint())
    restored.load_state_dict(deepcopy(selector.state_dict()))
    consumer = FrozenMechanismConsumer(
        parent.predictor,
        readout=parent.readout,
        catalog=parent.catalog,
        policy=restored,
    )
    assert (
        consumer.identity()["selector"]["configuration"]["catalog_id"]
        == parent.catalog.fingerprint()
    )
    restored.catalog_id = "c" * 64
    with pytest.raises(ValueError):
        consumer.verify()


def test_logical_cohort_trace_budget_is_checked_before_any_mac_observation():
    parent = engine()
    config = configuration()
    config = replace(
        config,
        max_batch=1,
        max_working_bytes=512 * 1024,
        evidence_bridge=replace(config.evidence_bridge, max_batch=1, max_working_bytes=512 * 1024),
    )
    reader = EpisodicReadout(config, dtype=torch.float64).eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(parent.predictor, readout=reader, catalog=parent.catalog)
    at = 1_609_459_200_000_000
    snapshot = bank("cpu", cutoff=at)
    reader._validate_read(torch.zeros((1, 32), dtype=torch.float64), snapshot, "b" * 64, at)
    before = consumer.initial_state()
    with pytest.raises(ValueError, match="antes de MAC"):
        consumer.begin(
            before,
            flow_ids=tuple(f"US/T{i:04d}" for i in range(128)),
            cutoff=at,
            context_id="b" * 64,
            matures_at=(at + 10,) * 128,
            snapshot=snapshot,
        )
    assert before.cutoff is None


def test_old_policy_identity_does_not_gain_an_implicit_regional_catalog():
    assert "catalog_id" not in policy().get_extra_state()
    model = pt4(None, head=QUANTILE_HEAD).eval().requires_grad_(False)
    FrozenMechanismConsumer(model, policy=policy())
    with pytest.raises(ValueError):
        bound_policy(CONTRACT)


@pytest.mark.parametrize("planning_rows,expected", [(256, "k1"), (64, "k4cr")])
def test_reader_planning_bound_can_trigger_the_declared_k1_without_weakening_the_guard(
    planning_rows, expected
):
    parent = engine()
    config = configuration()
    config = replace(
        config,
        max_batch=planning_rows,
        evidence_bridge=replace(config.evidence_bridge, max_batch=planning_rows),
    )
    reader = EpisodicReadout(config, dtype=torch.float64).eval().requires_grad_(False)
    consumer = FrozenMechanismConsumer(parent.predictor, readout=reader, catalog=parent.catalog)
    at = 1_609_459_200_000_000
    data = source(256)
    data["keys"] = torch.eye(64, dtype=torch.float32).repeat(4, 1)
    snapshot = EpisodeSnapshot.create(
        **data,
        codec_id=CODEC,
        context_id="b" * 64,
        cutoff=at,
        dtype=torch.float64,
    )
    event = consumer.begin(
        consumer.initial_state(),
        flow_ids=tuple(f"US/T{i:04d}" for i in range(256)),
        cutoff=at,
        context_id="b" * 64,
        matures_at=(at + 10,) * 256,
        snapshot=snapshot,
        action=MechanismAction(route="k4cr"),
    )
    plan = dict(event.plans)["US"]
    assert plan.applied.route == expected
    assert plan.reasons == (("reader_budget",) if planning_rows == 256 else ())
