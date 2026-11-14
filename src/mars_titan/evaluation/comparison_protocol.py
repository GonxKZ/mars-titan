"""Contrastes e inferencia fijados antes de ejecutar la campaña de comparación."""

import copy
import math

METRICS = ("mae_raw", "mae_calibrated", "pearson_ic", "rank_ic")
FAMILIES = {
    "primary": [
        ["mars_titan_m3", "titans_mac"],
        ["mars_titan_m3", "zero"],
        ["area_network_asset_groups", "titans_mac"],
        ["area_network_asset_groups", "zero"],
    ],
    "references": [
        [name, "zero"] for name in ("ridge", "rnn", "lstm", "gru", "dlinear", "transformer")
    ],
    "neural_memory": [
        ["titans_mac", name] for name in ("transformer", "titans_mac_disabled", "titans_mac_frozen")
    ],
    "episodic_memory": [
        [name, "mars_titan_m0"] for name in ("mars_titan_m1", "mars_titan_m2", "mars_titan_m3")
    ],
    "admission": [
        ["mars_titan_m2", "mars_titan_m1"],
        ["mars_titan_m3", "mars_titan_m1"],
        ["mars_titan_m3", "mars_titan_m2"],
    ],
    "mars_titan": [["mars_titan_m3", name] for name in ("titans_mac", "transformer", "zero")],
    "refinements": [[name, "mars_titan_m3"] for name in ("mars_titan_m3_k2", "mars_titan_m3_k4")],
    "evidence_bridge": [
        [name, control]
        for name in ("mars_titan_m3_bridge_ring", "mars_titan_m3_bridge_complete")
        for control in ("mars_titan_m3_bridge_self", "mars_titan_m3")
    ],
    "areas": [
        ["area_network", name]
        for name in ("titans_mac", "area_network_membership_only", "area_network_hidden_only")
    ],
    "asset_groups": [
        ["asset_groups", "asset_groups_shuffled"],
        ["asset_groups_shuffled", "titans_mac"],
    ],
    "combined_network": [
        ["area_network_asset_groups", name]
        for name in ("titans_mac", "area_network", "asset_groups")
    ],
    "cross_asset": [
        ["cross_asset_attention", "cross_asset_attention_self"],
        ["cross_asset_attention_self", "titans_mac"],
    ],
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(models):
    """Especificación común, reducida a las identidades presentes en una declaración."""
    names = set(models) | {"zero"}
    families = {
        family: [pair for pair in pairs if set(pair) <= names] for family, pairs in FAMILIES.items()
    }
    families = {name: pairs for name, pairs in families.items() if pairs}
    _require(families, "El análisis necesita contrastes declarados para sus modelos")
    return dict(
        schema_version=1,
        kind="comparison_campaign_analysis_v1",
        metrics=list(METRICS),
        primary_metric="mae_raw",
        primary_family="primary" if "primary" in families else next(iter(families)),
        seed_aggregation="mean_of_session_metrics",
        session_weighting="equal",
        bootstrap=dict(
            block_days=16,
            sensitivity_block_days=[8, 32],
            replicates=2000,
            seed=20261114,
            confidence=0.95,
            stratification="origin",
        ),
        families=families,
    )


def checked(options, models):
    """Validar los límites y el significado de la inferencia que lleva el paquete."""
    _require("zero" not in models, "zero está reservado para la referencia sin aprendizaje")
    _require(isinstance(options, dict), "Falta la declaración del análisis")
    result = copy.deepcopy(options)
    _require(
        result.get("schema_version") == 1
        and result.get("kind") == "comparison_campaign_analysis_v1"
        and result.get("metrics") == list(METRICS)
        and result.get("primary_metric") == "mae_raw"
        and result.get("seed_aggregation") == "mean_of_session_metrics"
        and result.get("session_weighting") == "equal",
        "El análisis no conserva sus métricas, agregación o criterio principal",
    )
    bootstrap = result.get("bootstrap")
    _require(isinstance(bootstrap, dict), "Falta el bootstrap declarado")
    _require(
        type(bootstrap.get("replicates")) is int
        and 32 <= bootstrap["replicates"] <= 100_000
        and type(bootstrap.get("seed")) is int
        and 0 <= bootstrap["seed"] < 2**32
        and type(bootstrap.get("block_days")) is int
        and 1 <= bootstrap["block_days"] <= 512
        and type(bootstrap.get("confidence")) in (float, int)
        and math.isfinite(bootstrap["confidence"])
        and 0.5 < bootstrap["confidence"] < 1
        and bootstrap.get("stratification") == "origin",
        "El bootstrap necesita parámetros finitos y acotados y estratos por origen",
    )
    sensitivity = bootstrap.get("sensitivity_block_days")
    _require(
        isinstance(sensitivity, list)
        and len(sensitivity) <= 8
        and all(type(value) is int and 1 <= value <= 512 for value in sensitivity)
        and len(set(sensitivity)) == len(sensitivity)
        and bootstrap["block_days"] not in sensitivity,
        "Las longitudes de sensibilidad deben ser distintas y acotadas",
    )
    families = result.get("families")
    _require(
        isinstance(families, dict)
        and 1 <= len(families) <= 32
        and result.get("primary_family") in families,
        "Faltan las familias declaradas o su familia principal",
    )
    names = set(models) | {"zero"}
    for family, pairs in families.items():
        _require(
            isinstance(family, str) and family.isidentifier(),
            "La familia necesita un identificador",
        )
        _require(
            isinstance(pairs, list)
            and 1 <= len(pairs) <= 64
            and all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(name, str) and name in names for name in pair)
                and pair[0] != pair[1]
                for pair in pairs
            )
            and len({tuple(pair) for pair in pairs}) == len(pairs),
            "Los contrastes deben nombrar modelos distintos y no repetirse",
        )
    return result
