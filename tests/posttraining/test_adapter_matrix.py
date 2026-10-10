"""Matriz finita de adaptadores: combinaciones, presupuesto, identidad y destinos."""

import copy
import itertools
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.predictive_adaptation import adapted_copy, trainable_parameters
from mars_titan.posttraining import adapter_matrix
from mars_titan.posttraining.run import validate_case

CONFIG = Path("configs/posttraining/adapter-matrix-v1.json")
DIMENSIONS = dict(prices=5, news=7, charts=6, fundamentals=3, macro=6)


def matrix():
    return adapter_matrix.read_matrix(CONFIG)


def parent(kind, layers=2):
    torch.manual_seed(1)
    return MultimodalReference(
        kind,
        DIMENSIONS,
        context=8,
        hidden_size=32,
        layers=layers,
        dropout=0.1,
        transformer=dict(heads=2, feedforward_multiplier=2) if kind == "transformer" else None,
        mask_fusion=PRESENCE_FUSION,
    ).requires_grad_(False)


def test_repository_matrix_declares_every_combination_once():
    declared, digest = matrix()
    assert declared["input_policy"] == HISTORICAL_MASKED
    combinations = {tuple(arm["points"]) for arm in declared["arms"] if "overrides" not in arm}
    assert combinations == {
        combination
        for size in (1, 2, 3)
        for combination in itertools.combinations(adapter_matrix.POINTS, size)
    }
    assert [arm["id"] for arm in declared["arms"] if "overrides" in arm] == ["fusion_full_rank"]
    for family in adapter_matrix.FAMILIES:
        arms = adapter_matrix.arms(declared, family)
        readout = family in adapter_matrix.READOUT_FAMILIES
        assert len(arms) == (8 if readout else 4)
        assert all(readout or "readout" not in arm["points"] for arm in arms)
        cases = adapter_matrix.cases(declared, digest, family)
        assert len(cases) == len(declared["budget"]["seeds"]) * (len(arms) + 2)
        assert len({item["id"] for item in cases}) == len(cases)
        for item in cases:
            validate_case(item["case"])
            assert item["case"]["condition"] == "real"
            assert item["case"]["selection"] == declared["selection"]
            if item["control"] is None:
                assert item["case"]["adapter"]["matrix_sha256"] == digest
                assert item["case"]["adapter"]["input_policy"] == HISTORICAL_MASKED


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_plan_fixes_equal_updates_and_counts_every_trainable_parameter(family):
    declared, digest = matrix()
    model = parent(family)
    rows = adapter_matrix.plan(
        declared, digest, family, model, updates_per_epoch=13, linear_features=1734
    )
    assert rows[0] == dict(
        id="frozen_parent",
        control="frozen_parent",
        trainable_parameters=0,
        state_bytes=0,
        updates=0,
        invalidates=[],
        case=None,
    )
    assert {row["updates"] for row in rows[1:]} == {declared["budget"]["epochs"] * 13}
    full = sum(value.numel() for value in model.parameters())
    for row in rows[1:]:
        assert row["state_bytes"] == 12 * row["trainable_parameters"]
        if row["control"] == "linear_residual":
            assert row["trainable_parameters"] == 1735 and row["invalidates"] == []
        elif row["control"] == "full_continuation":
            assert row["trainable_parameters"] == full
            assert "cached_parent_predictions" in row["invalidates"]
        else:
            adapter = row["case"]["adapter"]
            child = adapted_copy(
                model,
                adapter_matrix.targets(adapter, model),
                seed=adapter_matrix.adapter_seed(row["case"]),
            )
            assert trainable_parameters(child) == row["trainable_parameters"] < full
    by_arm = {row["id"].split("/")[1]: row["trainable_parameters"] for row in rows[1:4]}
    hidden, fusion_in = 32, 5 * 32 + 5
    assert by_arm == {
        "linear_residual": 1735,
        "full_continuation": full,
        "head": hidden + 1,
    }
    arms = {row["id"].split("/")[1]: row for row in rows if row["id"].startswith("seed-42/")}
    assert arms["fusion"]["trainable_parameters"] == 4 * (hidden + fusion_in)
    assert arms["fusion_full_rank"]["trainable_parameters"] == hidden * fusion_in
    if family == "transformer":
        # Dos capas con consulta y salida de rango 4: 2 × 2 × 4 × (32 + 32).
        assert arms["readout"]["trainable_parameters"] == 1024
        assert arms["head+readout+fusion"]["trainable_parameters"] == 33 + 1024 + 4 * 197


def test_targets_keep_keys_values_and_persistent_states_frozen():
    declared, digest = matrix()
    model = parent("transformer")
    case = next(
        item["case"]
        for item in adapter_matrix.cases(declared, digest, "transformer")
        if item["id"] == "seed-42/readout"
    )
    description = adapter_matrix.describe(case["adapter"], model)
    packed = [row for row in description["targets"] if row["tensor"] == "in_proj_weight"]
    assert [row["rows"] for row in packed] == [[0, 32], [0, 32]]
    assert all(row["shape"] == [96, 32] for row in packed)
    with pytest.raises(ValueError, match="lectura"):
        adapter_matrix.targets(case["adapter"], parent("gru"))
    with pytest.raises(ValueError, match="neuronales"):
        adapter_matrix.arms(declared, "ridge")


def test_adapter_seed_is_fixed_by_seed_arm_and_matrix():
    declared, digest = matrix()
    cases = {item["id"]: item["case"] for item in adapter_matrix.cases(declared, digest, "gru")}
    seeds = {
        key: adapter_matrix.adapter_seed(case) for key, case in cases.items() if "adapter" in case
    }
    assert len(set(seeds.values())) == len(seeds)
    assert seeds == {
        key: adapter_matrix.adapter_seed(case) for key, case in cases.items() if "adapter" in case
    }
    other = {item["id"]: item["case"] for item in adapter_matrix.cases(declared, "b" * 64, "gru")}
    assert adapter_matrix.adapter_seed(other["seed-42/head"]) != seeds["seed-42/head"]


def mutate(change):
    declared = copy.deepcopy(json.loads(CONFIG.read_text()))
    change(declared)
    return declared


INVALID = {
    "missing_combination": lambda m: m["arms"].pop(3),
    "duplicated_arm": lambda m: m["arms"].append(dict(m["arms"][0])),
    "four_points": lambda m: m["arms"].append(
        dict(id="x", points=["head", "readout", "fusion", "head"])
    ),
    "unordered_points": lambda m: m["arms"].__setitem__(
        3, dict(id="readout+head", points=["readout", "head"])
    ),
    "renamed_arm": lambda m: m["arms"][0].__setitem__("id", "cabeza"),
    "patience": lambda m: m["selection"].__setitem__("patience", 2),
    "objective": lambda m: m["objectives"].__setitem__("full_continuation", "neural_mse"),
    "head_low_rank": lambda m: m["points"].__setitem__(
        "head", dict(form="low_rank", rank=1, alpha=1.0, invalidates=["cached_parent_predictions"])
    ),
    "rank": lambda m: m["points"]["fusion"].__setitem__("rank", 0),
    "unknown_state": lambda m: m["points"]["fusion"].__setitem__("invalidates", ["weights"]),
    "same_override": lambda m: m["arms"][-1]["overrides"].__setitem__(
        "fusion", dict(m["points"]["fusion"])
    ),
    "test_opened": lambda m: m.__setitem__("final_test_opened", True),
    "too_many_arms": lambda m: m["arms"].extend(
        dict(
            id=f"extra-{i}",
            points=["fusion"],
            overrides={"fusion": dict(form="residual", invalidates=["fused_representations"])},
        )
        for i in range(5)
    ),
    "readout_in_gru": lambda m: m["architectures"]["executable"].__setitem__(
        "gru", ["head", "readout", "fusion"]
    ),
    "titans_memory_adapter": lambda m: m["architectures"]["pending"]["titans_mac"]["targets"][
        "readout"
    ].append("mac.memory.key_projection"),
    "budget_epochs": lambda m: m["budget"].__setitem__("epochs", 0),
    "control_missing": lambda m: m["controls"].pop(),
}


@pytest.mark.parametrize("name", sorted(INVALID))
def test_matrix_rejects_open_or_unequal_designs(name):
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(mutate(INVALID[name]))


def test_pending_targets_exist_and_avoid_memory_states():
    from test_financial_adapter import specification

    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor

    pending = matrix()[0]["architectures"]["pending"]
    models = dict(
        titans_mac=FinancialPredictor(
            FinancialConfig(specification(), variant="mac_online", hidden_size=32)
        ),
        mars_titan_episodic_readout=EpisodicReadout(
            EpisodicReadoutConfig("a" * 64, hidden_size=32)
        ),
    )
    for name, model in models.items():
        declared = pending[name]
        modules = dict(model.named_modules())
        names = {key for key, _ in model.named_parameters()} | set(modules)
        for paths in declared["targets"].values():
            for path in paths:
                assert isinstance(modules[path], nn.Linear), path
        for path in declared["frozen"]:
            assert any(key == path or key.startswith(path + ".") for key in names), path
    assert pending["episodic_gru_candidate"]["targets"] == {}


SECOND = Path("configs/posttraining/adapter-matrix-v2.json")
REASON = "Motivo declarado antes de ajustar, con más de veinte caracteres."


def quantile_parent(kind):
    torch.manual_seed(1)
    return MultimodalReference(
        kind,
        DIMENSIONS,
        context=8,
        hidden_size=32,
        layers=2,
        dropout=0.1,
        transformer=dict(heads=2, feedforward_multiplier=2) if kind == "transformer" else None,
        mask_fusion=PRESENCE_FUSION,
        head="quantile_head_v1",
    ).requires_grad_(False)


def test_second_version_only_adds_objectives_for_quantile_parents():
    first, _ = matrix()
    second, digest = adapter_matrix.read_matrix(SECOND)
    ignored = {"schema_version", "objectives"}
    assert {k: v for k, v in first.items() if k not in ignored} == {
        k: v for k, v in second.items() if k not in ignored
    }
    assert second["schema_version"] == 2
    assert adapter_matrix.objectives(second, "scalar") == first["objectives"]
    assert adapter_matrix.objectives(second, "quantile_head_v1")["adapters"] == "neural_pinball"
    with pytest.raises(ValueError, match="escalar"):
        adapter_matrix.objectives(first, "quantile_head_v1")
    for family in adapter_matrix.FAMILIES:
        scalar = adapter_matrix.cases(second, digest, family)
        quantile = adapter_matrix.cases(second, digest, family, head="quantile_head_v1")
        assert [item["id"] for item in quantile] == [
            item["id"] for item in scalar if item["control"] != "linear_residual"
        ]
        for left, right in zip(
            [item for item in scalar if item["control"] != "linear_residual"],
            quantile,
            strict=True,
        ):
            assert {k: v for k, v in left["case"].items() if k != "mode"} == {
                k: v for k, v in right["case"].items() if k != "mode"
            }


@pytest.mark.parametrize("family", ["gru", "transformer"])
def test_quantile_plan_keeps_equal_updates_without_the_excluded_control(family):
    second, digest = adapter_matrix.read_matrix(SECOND)
    first, first_digest = matrix()
    model = quantile_parent(family)
    with pytest.raises(ValueError, match="escalar"):
        adapter_matrix.plan(
            first, first_digest, family, model, updates_per_epoch=13, linear_features=1734
        )
    rows = adapter_matrix.plan(
        second, digest, family, model, updates_per_epoch=13, linear_features=1734
    )
    assert rows[0]["control"] == "frozen_parent" and rows[0]["updates"] == 0
    assert {row["updates"] for row in rows[1:]} == {second["budget"]["epochs"] * 13}
    assert all(row["control"] != "linear_residual" for row in rows)
    assert {row["case"]["mode"] for row in rows[1:]} == {"neural_pinball"}
    arms = 8 if family == "transformer" else 4
    assert len(rows) == 1 + len(second["budget"]["seeds"]) * (arms + 1)
    by_arm = {row["id"]: row["trainable_parameters"] for row in rows[1:]}
    # La cabeza de cuantiles tiene cinco salidas: 5 × 32 pesos y cinco sesgos.
    assert by_arm["seed-42/head"] == 5 * 32 + 5
    assert by_arm["seed-42/full_continuation"] == sum(v.numel() for v in model.parameters())


def mutate_second(change):
    declared = copy.deepcopy(json.loads(SECOND.read_text()))
    change(declared["objectives"])
    return declared


INVALID_SECOND = {
    "quantile_mae": lambda o: o["quantile_head_v1"].__setitem__("adapters", "neural_mae"),
    "quantile_continuation": lambda o: o["quantile_head_v1"].__setitem__(
        "full_continuation", "neural_mae"
    ),
    "quantile_point": lambda o: o["quantile_head_v1"].update(
        adapters="neural_mae", full_continuation="neural_mae"
    ),
    "scalar_pinball": lambda o: o["scalar"].update(
        adapters="neural_pinball", full_continuation="neural_pinball"
    ),
    "quantile_linear": lambda o: o["quantile_head_v1"].__setitem__("linear_residual", "mae"),
    "short_reason": lambda o: o["quantile_head_v1"].__setitem__(
        "linear_residual", {"excluded": "no"}
    ),
    "scalar_excluded": lambda o: o["scalar"].__setitem__("linear_residual", {"excluded": REASON}),
    "missing_head": lambda o: o.pop("quantile_head_v1"),
    "unknown_head": lambda o: o.__setitem__("median_head", dict(o["scalar"])),
    "flat": lambda o: o.update(adapters="neural_mae"),
}


@pytest.mark.parametrize("name", sorted(INVALID_SECOND))
def test_second_version_rejects_objectives_that_do_not_fit_the_head(name):
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(mutate_second(INVALID_SECOND[name]))


def test_versions_do_not_mix_objective_layouts():
    nested = copy.deepcopy(json.loads(SECOND.read_text()))
    nested["schema_version"] = 1
    flat = copy.deepcopy(json.loads(CONFIG.read_text()))
    flat["schema_version"] = 2
    for value in (nested, flat):
        with pytest.raises(ValueError, match="versión|objetivo"):
            adapter_matrix.validate_matrix(value)


THIRD = Path("configs/posttraining/adapter-matrix-v3.json")


def test_third_version_only_adds_the_chronological_designs():
    from mars_titan.posttraining import chronological_matrix as cm

    second, _ = adapter_matrix.read_matrix(SECOND)
    third, digest = adapter_matrix.read_matrix(THIRD)
    ignored = {"schema_version", "architectures"}
    assert {k: v for k, v in second.items() if k not in ignored} == {
        k: v for k, v in third.items() if k not in ignored
    }
    assert third["architectures"]["executable"] == second["architectures"]["executable"]
    assert third["architectures"]["pending"] == {}
    for family in adapter_matrix.FAMILIES:
        assert adapter_matrix.cases(
            third, digest, family, head="quantile_head_v1"
        ) == adapter_matrix.cases(second, digest, family, head="quantile_head_v1")
    per_seed = {
        ("titans_mac", "transformer_direct", True): 5,
        ("titans_mac", "mac_disabled", True): 5,
        ("titans_mac", "mac_frozen", True): 9,
        ("titans_mac", "mac_online", True): 9,
        ("mars_titan", None, True): 4,
        ("mars_titan", None, False): 2,
        ("cm_v1", None, True): 4,
        ("episodic_gru", None, True): 2,
    }
    seeds = len(third["budget"]["seeds"])
    for (family, variant, bank), count in per_seed.items():
        rows = cm.cases(third, digest, family, variant=variant, bank=bank)
        assert len(rows) == seeds * count, (family, variant, bank)
        assert {row["case"]["objective"] for row in rows} == {"neural_pinball"}
        assert sum(row["control"] == "full_continuation" for row in rows) == seeds
        assert len({row["id"] for row in rows}) == len(rows)
        for row in rows:
            cm.validate_case(row["case"])


def test_chronological_cases_fix_the_recipe_and_component_seeds():
    from mars_titan.posttraining import chronological_matrix as cm

    third, digest = adapter_matrix.read_matrix(THIRD)
    rows = {row["id"]: row["case"] for row in cm.cases(third, digest, "mars_titan")}
    both = rows["seed-42/core+episodic_readout"]
    options = cm.recipe_options(both)
    assert options["loss"] == "pinball" and options["epochs"] == third["budget"]["epochs"]
    assert options["learning_rate"] == third["budget"]["learning_rate"]
    assert options["max_grad_norm"] == third["budget"]["clip_norm"]
    # Sin paciencia la selección recorre el presupuesto y conserva el mejor estado.
    assert options["selection"]["patience"] == third["budget"]["epochs"]
    core, episodic = (cm.component_seed(both, name) for name in cm.COMPONENTS)
    assert core != episodic
    assert cm.component_seed(rows["seed-43/core+episodic_readout"], "core") != core
    assert cm.component_seed(both, "core") == core
    online = third["architectures"]["chronological"]["titans_mac"]["variants"]["mac_online"]
    assert both["adapter"]["core"] == {name: third["points"][name] for name in online}
    with pytest.raises(ValueError):
        cm.validate_case(dict(both, control="full_continuation"))
    with pytest.raises(ValueError):
        cm.validate_case(dict(both, objective="neural_mae"))
    with pytest.raises(ValueError, match="cronológico"):
        cm.design(third, "transformer")


def mutate_third(change):
    declared = copy.deepcopy(json.loads(THIRD.read_text()))
    change(declared["architectures"]["chronological"])
    return declared


INVALID_THIRD = {
    "memory_target": lambda c: c["titans_mac"]["targets"]["readout"].append(
        "mac.memory.key_projection"
    ),
    "persistent_target": lambda c: c["titans_mac"]["targets"]["fusion"].append("mac.persistent"),
    "unfrozen_memory": lambda c: c["titans_mac"]["frozen"].remove("mac.memory"),
    "silent_inapplicable": lambda c: c["titans_mac"].__setitem__("inapplicable", {}),
    "short_inapplicable": lambda c: c["titans_mac"]["inapplicable"].__setitem__("readout", "no"),
    "online_without_readout": lambda c: c["titans_mac"]["variants"].__setitem__(
        "mac_online", ["head", "fusion"]
    ),
    "missing_variant": lambda c: c["titans_mac"]["variants"].pop("mac_frozen"),
    "core_other_variant": lambda c: c["episodic_readout"]["components"]["core"].__setitem__(
        "variant", "mac_frozen"
    ),
    "joined_components": lambda c: c["episodic_readout"].__setitem__(
        "arms", [["core", "episodic_readout"]]
    ),
    "frozen_readout_target": lambda c: c["episodic_readout"]["frozen"].append("query_projection"),
    "no_bank_reason": lambda c: c["episodic_readout"].__setitem__("without_bank", ""),
    "candidate_fusion": lambda c: c["episodic_gru"].__setitem__("points", ["head", "fusion"]),
    "candidate_silent_exclusion": lambda c: c["episodic_gru"]["excluded"].pop("fusion"),
    "missing_design": lambda c: c.pop("episodic_gru"),
}


@pytest.mark.parametrize("name", sorted(INVALID_THIRD))
def test_third_version_rejects_open_or_memory_touching_designs(name):
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(mutate_third(INVALID_THIRD[name]))


def test_second_version_does_not_accept_the_chronological_section():
    declared = copy.deepcopy(json.loads(THIRD.read_text()))
    declared["schema_version"] = 2
    with pytest.raises(ValueError):
        adapter_matrix.validate_matrix(declared)
