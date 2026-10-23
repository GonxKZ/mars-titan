"""Identificar las etapas regionales activas sin ejecutar el predictor ni una sesión."""

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_bridge_episodic_readout import configuration as base_readout_config
from test_financial_adapter import specification
from test_frozen_financial import frozen_backend as frozen_backend

from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.models.titans.regional_hierarchy import RegionalHierarchyConfig


@pytest.mark.parametrize("family", ("program", "hierarchy"))
@pytest.mark.parametrize("stages", ("solo_pre", "solo_post", "ambos"))
def test_frozen_composition_identifies_only_the_active_regional_stages(family, stages):
    active_pre, active_post = stages != "solo_post", stages != "solo_pre"
    execution = (
        RegionalExecutionConfig(program="summary_interleaved_v1")
        if family == "program"
        else RegionalHierarchyConfig()
    )
    config = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        max_batch=8,
        head=QUANTILE_HEAD,
        session_context="market_memory_v1",
        regional_fusion=RegionalFusionConfig(
            topology="self", routing="uniform", cycles=2, local_steps=2, summary=True
        ),
        regional_execution=execution if active_pre else None,
    )
    predictor = FinancialPredictor(config, device="cpu", dtype=torch.float64)
    predictor.eval().requires_grad_(False)
    reader = None
    if active_post:
        base = base_readout_config(refinements=1, episode_selection="first_read")
        readout_config = replace(
            base,
            max_batch=2,
            evidence_bridge=replace(
                base.evidence_bridge,
                max_batch=2,
                regions=("view_1", "view_2", "view_3", "view_4"),
                topology="self",
                cycles=0,
                hierarchy=BridgeHierarchyConfig(),
            ),
        )
        reader = EpisodicReadout(readout_config, device="cpu", dtype=torch.float64)
        reader.eval().requires_grad_(False)
    contract = "shared_evidence_bridge_hierarchy_v1" if active_post else "legacy"
    consumer = FrozenMechanismConsumer(
        predictor,
        readout=reader,
        catalog=MechanismCatalog(
            reader_contract=contract, regional_execution=config.regional_execution
        ),
    )
    identity = consumer.identity()
    assert identity["schema_version"] == (6 if stages == "ambos" else 5)
    recipes = {
        "solo_pre": "frozen_pre_mac_regional_program_cohort_v1",
        "solo_post": "frozen_hierarchical_regional_mechanism_cohort_v1",
        "ambos": "frozen_pre_mac_hierarchical_bridge_cohort_v1",
    }
    assert identity["recipe"] == recipes[stages]
    assert ("regional_execution" in identity) == active_pre
    sources = []
    if active_pre:
        assert identity["regional_execution"] == predictor.regional_execution.get_extra_state()
        sources.append("regional_execution")
        if family == "hierarchy":
            sources.append("regional_hierarchy")
    if active_post:
        assert identity["readout"]["configuration"]["evidence_bridge"]["kind"] == contract
        sources.append("bridge_hierarchy")
    else:
        assert identity["readout"] is None
    for name in sources:
        module = importlib.import_module(f"mars_titan.models.titans.{name}")
        expected = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert identity["implementation"][module.__name__] == expected
    consumer.verify()
