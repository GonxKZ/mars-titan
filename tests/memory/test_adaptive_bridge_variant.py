"""La declaración de componentes conserva y reconstruye la parada opcional."""

from copy import deepcopy

import pytest
import torch
from test_financial_session_controls import four_flow_source as four_flow_source
from test_frozen_financial import frozen_backend as frozen_backend
from test_shared_bridge_variant import GEOMETRY, selected
from test_shared_bridge_variant import predictor as predictor

from mars_titan.memory import mars_titan_variant as api
from mars_titan.models.titans.episodic_readout import EpisodicReadout


@pytest.mark.parametrize("backend", ("adaptive_reference_full", "adaptive_compact"))
def test_variant_roundtrip_constructs_the_declared_stopping_without_an_extra_component(
    predictor, backend
):
    geometry = dict(
        GEOMETRY,
        cycles=0,
        recurrence=dict(
            steps=4,
            max_node_updates_per_row=1216,
            stopping=dict(atol=0.001, rtol=0.0, backend=backend),
        ),
    )
    variant = selected(predictor, geometry=geometry)
    config = variant.readout_config("a" * 64, hidden_size=32)
    assert config.evidence_bridge.recurrence.stopping.backend == backend
    assert config.identity()["evidence_bridge"]["kind"] == "shared_evidence_bridge_adaptive_v1"
    reader = (
        EpisodicReadout(config, device=predictor.head.weight.device, dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    consumer = variant.consumer(predictor, reader)
    assert consumer.readout.config.identity() == config.identity()
    assert (
        variant.components["evidence_bridge"]["recurrence"]["stopping"]
        == geometry["recurrence"]["stopping"]
    )
    restored = api.select_variant(
        api.load_declaration(), deepcopy(variant.components), base=variant.base
    )
    assert restored.identity() == variant.identity()


def test_active_stopping_requires_its_explicit_component_declaration(predictor):
    declaration = deepcopy(api.load_declaration())
    declaration["components"]["evidence_bridge"]["recurrence"].pop("stopping", None)
    geometry = dict(
        GEOMETRY,
        cycles=0,
        recurrence=dict(
            steps=4,
            max_node_updates_per_row=1216,
            stopping=dict(atol=0.001, rtol=0.0),
        ),
    )
    with pytest.raises(ValueError, match="parada"):
        api.select_variant(
            declaration,
            dict(episodic_bank="m1", evidence_bridge=geometry),
            base=api.core_identity(predictor, "fixture_bridge_catalog"),
        )
