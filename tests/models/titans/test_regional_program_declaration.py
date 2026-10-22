"""Contratos de los brazos declarados, sin construir modelos ni ejecutar ajustes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_financial_adapter import specification

from mars_titan.models.titans.config import canonical
from mars_titan.models.titans.financial import FinancialConfig
from mars_titan.models.titans.mechanism_catalog import (
    MechanismAction,
    MechanismAvailability,
    MechanismCatalog,
)
from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
from mars_titan.training.search_cases import checked_predictor_changes
from mars_titan.training.titans_walk_forward import predictor_options

ROOT = Path(__file__).resolve().parents[3]
DECLARATION = ROOT / "configs/experiments/regional-program-v1.json"
CONTROL = "legacy_post_fusion"


def declaration():
    assert DECLARATION.is_file(), "Falta la declaración finita de programas regionales"
    return json.loads(DECLARATION.read_text())


def configurations(document):
    recipe = json.loads((ROOT / document["base_recipe"]).read_text())
    common = dict(recipe, predictor=recipe["predictor"] | document["common_predictor"])
    return {
        name: FinancialConfig(
            specification(),
            variant=document["variant"],
            **predictor_options(common, {"predictor_changes": arm["predictor_changes"]}),
        )
        for name, arm in document["arms"].items()
    }


def test_declaration_is_disabled_and_preserves_the_closed_scientific_scope():
    document = declaration()
    assert document["default_enabled"] is False
    assert document["final_test_access"] is False
    assert document["status"] == "declared_not_executed"
    assert document["executions"] == 0
    assert document["confirmed_hypotheses"] == []
    arms = document["arms"]
    assert set(arms) == {
        CONTROL,
        "fixed_at_end",
        "fixed_interleaved",
        "contextual_norm",
        "fixed_at_end_no_summary",
        "fixed_interleaved_no_summary",
    }
    assert len({arm["variant_id"] for arm in arms.values()}) == len(arms)
    assert all(arm["baseline_explicit"] == CONTROL for arm in arms.values())


def test_every_arm_builds_the_same_financial_base_and_a_compatible_catalog():
    document = declaration()
    configs = configurations(document)
    action = MechanismAction(**document["mechanism_action"])
    assert action == MechanismAction("parent", "k1", "none")
    identities, fingerprints = {}, set()
    for name, config in configs.items():
        changes = document["arms"][name]["predictor_changes"]
        assert checked_predictor_changes(changes) == changes
        assert set(changes) <= {"regional_fusion", "regional_execution"}
        assert config.variant == "mac_online"
        assert config.session_context == "market_memory_v1"
        assert type(config.regional_fusion) is RegionalFusionConfig
        assert config.regional_fusion.topology == "ring"
        assert (config.regional_fusion.cycles, config.regional_fusion.local_steps) == (2, 2)
        catalog = MechanismCatalog(regional_execution=config.regional_execution)
        assert MechanismCatalog.restore(catalog.identity()) == catalog
        assert len(catalog.actions()) == 36
        resolved = catalog.resolve(
            action, MechanismAvailability(context=True, quantiles=True), rows=1
        )
        assert resolved.applied == action and not resolved.reasons
        identity = config.identity()
        fingerprints.add(hashlib.sha256(canonical(identity).encode()).hexdigest())
        identities[name] = {
            key: value
            for key, value in identity.items()
            if key not in {"regional_fusion", "regional_execution"}
        }
    assert len(fingerprints) == len(configs)
    assert all(identity == identities[CONTROL] for identity in identities.values())
    assert configs[CONTROL].regional_execution is None
    assert "regional_execution" not in configs[CONTROL].identity()
    for name in set(configs) - {CONTROL}:
        assert type(configs[name].regional_execution) is RegionalExecutionConfig
        assert (
            MechanismCatalog(regional_execution=configs[name].regional_execution).fingerprint()
            != MechanismCatalog().fingerprint()
        )


def test_programs_and_summary_ablations_keep_their_declared_semantics():
    configs = configurations(declaration())
    for name, program in (
        ("fixed_at_end", "summary_at_end_v1"),
        ("fixed_interleaved", "summary_interleaved_v1"),
    ):
        enabled, disabled = configs[name], configs[name + "_no_summary"]
        assert enabled.regional_execution == RegionalExecutionConfig(program=program)
        assert enabled.regional_fusion.summary is True
        assert disabled.regional_fusion == replace(enabled.regional_fusion, summary=False)
        assert disabled == replace(enabled, regional_fusion=disabled.regional_fusion)
    contextual = configs["contextual_norm"]
    assert contextual.regional_execution == RegionalExecutionConfig(
        selection="context_norm_v1", program="summary_at_end_v1", context_threshold=1e-9
    )
    with pytest.raises(ValueError, match="padre PT4"):
        replace(contextual, session_context=None)
    with pytest.raises(ValueError):
        checked_predictor_changes({"session_context": contextual.session_context})
