"""Pruebas TDD exhaustivas del Catálogo Unificado de 4 Familias TFM (`tfm_catalog.py`).

Verifica de forma estricta:
  1. `TFM_FAMILIES` (4 familias estrictamente independientes) y `ModelVariantSpec` inmutable
     (validación de `max_loops` `1..4`, mercados `US`/`CN` separados FM-8, `order_level_mode`
     FM-7, `builder_target`, y regla anti-sopa entre familias).
  2. Cobertura sin omisiones en `TFM_VARIANT_CATALOG` de TODAS las variaciones de las 4 familias
     (`#22`, `#32`, `#61`):
       - Familia 1 (`titans_original`): Entrega 2 baselines + RL + `titans_mac` + `mars_titan`
         `M0..M3`/`K=1,2,4`/`familiarity_gate_r3` + DGX v12 (`tabm`/`crrm`/`lfm_sw`/`csos`) +
         `titans_neuromod_sota` + su gemelo `noic_pinball`.
       - Familia 2 (`holographic_brain`): `h1_single_zero_{5,19,38}`, `h2_catalog_only`,
         `h3_hidden_groups`, `h4_asset_groups`, `hsi_session_hologram`, `csrf_hadamard_resonator`
         + los 9 presets standalone de `neuro_brain_sota.py`.
       - Familia 3 (`mmarrs_titan`): las 16 variaciones y controles emparejados (`v00..v13c`,
         `max_loops <= 3` en los bucles modernos).
       - Familia 4 (`titans_cognitive_core`): los 8 modos y controles (`cog_no_tools_k1..noic`).
  3. Consultas `list_variants`, `get_variant`, `build_ablation_matrix` (rechaza `"US+CN"` por
     FM-8, preserva sello `2024+` y `market_model` CSI 300 para CN), `validate_catalog_integrity`
     (detecta controles rotos, `max_loops > 4`, y sopas sin control) y serialización compacta
     `.json` / `.json.gz`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mars_titan.models.mmarrs.config import VARIATION_REGISTRY, variation_config
from mars_titan.models.mmarrs.model import build_variation
from mars_titan.models.mmarrs.neuro_brain_sota import NEURO_BRAIN_ABLATIONS
from mars_titan.models.mmarrs.tfm_catalog import (
    DEFAULT_MAX_LOOPS,
    MAX_ALLOWED_LOOPS,
    ORDER_LEVEL_MODES,
    TFM_FAMILIES,
    TFM_MARKETS,
    TFM_ORIGINS,
    TFM_VARIANT_ALIASES,
    TFM_VARIANT_CATALOG,
    ModelVariantSpec,
    build_ablation_matrix,
    get_variant,
    list_variants,
    resolve_variant_id,
    serialize_catalog_json,
    validate_catalog_integrity,
)


def test_constants_and_model_variant_spec_validation() -> None:
    """`ModelVariantSpec` valida inmutabilidad, `max_loops<=4`, FM-8 mercados y anti-sopa."""
    assert TFM_FAMILIES == (
        "titans_original",
        "holographic_brain",
        "mmarrs_titan",
        "titans_cognitive_core",
    )
    assert TFM_ORIGINS == (
        "entrega2_pdf",
        "dgx_v10_v12",
        "sota_innovation",
        "matched_control",
    )
    assert TFM_MARKETS == ("US", "CN")
    assert ORDER_LEVEL_MODES == ("decoupled_f7b", "noic_pinball_twin", "raw")
    assert DEFAULT_MAX_LOOPS == 3
    assert MAX_ALLOWED_LOOPS == 4

    spec = get_variant("v13_mmarrs_full")
    assert isinstance(spec, ModelVariantSpec)
    assert spec.is_independent_family is True
    assert len(spec.fingerprint_sha256()) == 64

    with pytest.raises(AttributeError):
        spec.max_loops = 2  # type: ignore[misc]

    # Tope duro `max_loops <= 4`
    with pytest.raises(ValueError):
        ModelVariantSpec(
            variant_id="bad_loops",
            family="mmarrs_titan",
            origin="sota_innovation",
            description="Too many loops",
            matched_control_id="v00_dense_iso_flop",
            supports_markets=("US", "CN"),
            max_loops=5,
            order_level_mode="decoupled_f7b",
            builder_target="mars_titan.models.mmarrs.model:build_variation[v13_mmarrs_titan_full]",
            config_overrides={},
            tags=("test",),
        )

    # FM-8: prohibido mezclar mercados `US+CN` en `supports_markets`
    with pytest.raises(ValueError):
        ModelVariantSpec(
            variant_id="bad_market",
            family="titans_original",
            origin="sota_innovation",
            description="Pooled US+CN forbidden",
            matched_control_id="zero",
            supports_markets=("US+CN",),
            max_loops=1,
            order_level_mode="decoupled_f7b",
            builder_target="mars_titan.models.titans:TitansMAC",
            config_overrides={},
            tags=("test",),
        )

    # Anti-sopa: puente entre familias no puede mezclar >= 2 familias externas ni carecer de control
    with pytest.raises(ValueError):
        ModelVariantSpec(
            variant_id="bad_soup",
            family="mmarrs_titan",
            origin="sota_innovation",
            description="Uncontrolled multi-family soup",
            matched_control_id="bad_soup",
            supports_markets=("US", "CN"),
            max_loops=3,
            order_level_mode="decoupled_f7b",
            builder_target="mars_titan.models.mmarrs.model:build_variation[v13_mmarrs_titan_full]",
            config_overrides={},
            tags=("soup",),
            cross_family_bridge_of=("holographic_brain", "titans_cognitive_core"),
        )


def test_zero_omission_four_family_catalog_coverage() -> None:
    """Todas las variaciones de las 4 familias (`#22`, `#32`, `#61`) están presentes sin omisión."""
    report = validate_catalog_integrity()
    assert report["valid"] is True
    assert report["total_variants"] == len(TFM_VARIANT_CATALOG)
    assert set(report["counts_by_family"].keys()) == set(TFM_FAMILIES)

    # Familia 1 (`titans_original`)
    f1_ids = {v.variant_id for v in list_variants(family="titans_original")}
    expected_f1 = {
        "zero",
        "past_median",
        "momentum_reversal",
        "ridge",
        "xgboost",
        "lightgbm",
        "rnn",
        "lstm",
        "gru",
        "dlinear",
        "transformer_compact",
        "kronlstm",
        "rl_ppo",
        "rl_double_dqn",
        "rl_klpo",
        "rl_grpo",
        "titans_mac_disabled",
        "titans_mac_frozen",
        "titans_mac_online",
        "titans_mac_noic_pinball",
        "mars_titan_m0_no_bank",
        "mars_titan_m1_uniform",
        "mars_titan_m2_mature_error",
        "mars_titan_m3_k1",
        "mars_titan_m3_k2",
        "mars_titan_m3_k4",
        "mars_titan_m3_familiarity_gate_r3",
        "titans_tabm",
        "titans_crrm",
        "titans_lfm_sw",
        "titans_csos",
        "titans_neuromod_sota",
        "titans_neuromod_sota_noic_pinball",
        "t15a_dual_axis_mem",
        "t15b_full_v2",
        "t15c_noic_twin",
    }
    assert expected_f1.issubset(f1_ids)

    # Familia 2 (`holographic_brain`) — incluye los 9 presets v1 + 4 presets v2 SIN `nb_nb_*`
    f2_ids = {v.variant_id for v in list_variants(family="holographic_brain")}
    expected_f2 = {
        "h1_single_zero_5",
        "h1_single_zero_19",
        "h1_single_zero_38",
        "h2_catalog_only",
        "h3_hidden_groups",
        "h4_asset_groups",
        "hsi_session_hologram",
        "csrf_hadamard_resonator",
    } | {
        preset if preset.startswith("nb_") else f"nb_{preset}"
        for preset in NEURO_BRAIN_ABLATIONS
    }
    assert expected_f2.issubset(f2_ids)
    assert not any(vid.startswith("nb_nb_") for vid in f2_ids)
    assert {
        "nb_v2_fhrr_lgn_pi",
        "nb_v2_full",
        "nb_v2_noic_twin",
        "nb_v2_moe_soft_k3",
    }.issubset(f2_ids)

    # Familia 3 (`mmarrs_titan`) — alineación 1:1 con Familia 3 en `VARIATION_REGISTRY` (v1 + v2)
    f3_ids = {v.variant_id for v in list_variants(family="mmarrs_titan")}
    expected_f3 = {
        k for k, spec_d in VARIATION_REGISTRY.items() if spec_d.get("family") == "mmarrs_titan"
    }
    assert f3_ids == expected_f3
    assert {
        "v14a_tied_loop",
        "v14b_tied_xasset",
        "v14c_tied_xasset_cf",
        "v14c_placebo",
        "v14c_noic_twin",
    }.issubset(f3_ids)
    # Todos los bucles de Familia 3 respetan el tope `<= 3` por defecto
    for vid in expected_f3:
        assert get_variant(vid).max_loops <= 3

    # Familia 4 (`titans_cognitive_core`) — las 8 variantes v1 + 5 variantes v2 + alias v1
    f4_ids = {v.variant_id for v in list_variants(family="titans_cognitive_core")}
    expected_f4 = {
        "cog_no_tools_k1",
        "cog_no_tools_k3",
        "cog_uniform_tools_k3",
        "cog_soft_tools_k3",
        "cog_sparsemax_tools_k3",
        "cog_no_context_eng_k3",
        "cog_core_plus_moe_k3",
        "cog_core_noic_twin",
        "cog_v2_10tools_uniform_k3",
        "cog_v2_10tools_shuffled_cf_placebo_k3",
        "cog_v2_10tools_soft_k3",
        "cog_v2_10tools_sparsemax_k3",
        "cog_v2_10tools_noic_twin",
    }
    assert expected_f4.issubset(f4_ids)
    assert resolve_variant_id("cog_v1_6tools_k3") == "cog_soft_tools_k3"
    assert get_variant("cog_v1_6tools_k3") == get_variant("cog_soft_tools_k3")


def test_query_helpers_ablation_matrix_and_fm8_market_enforcement() -> None:
    """`build_ablation_matrix` separa US/CN (FM-8), respeta sello 2024+ y controles."""
    # Rechazar mercados conjuntos / inválidos (FM-8)
    for bad_mkt in ("US+CN", "ALL", "EU", ""):
        with pytest.raises(ValueError):
            build_ablation_matrix("mmarrs_titan", market=bad_mkt)

    with pytest.raises(ValueError):
        build_ablation_matrix("unknown_family", market="US")
    with pytest.raises(KeyError):
        get_variant("nonexistent_variant")

    for fam in TFM_FAMILIES:
        mat_us = build_ablation_matrix(fam, market="US")
        mat_cn = build_ablation_matrix(fam, market="CN")
        assert mat_us["family"] == fam
        assert mat_us["market"] == "US"
        assert mat_cn["market"] == "CN"
        assert mat_us["markets_strictly_separate"] is True
        assert mat_us["sealed_test_start_year"] == 2024
        assert mat_cn["market_model"] == "csi300_required"
        assert mat_us["market_model"] == "spy_ols_252d"
        assert len(mat_us["controls"]) >= 1
        assert len(mat_us["candidates"]) >= 1
        assert mat_us["n_variants"] == len(mat_us["variants"])
        assert mat_us["matrix_sha256"] != mat_cn["matrix_sha256"]

    # Excluir puentes inter-familia (`include_bridges=False`) deja solo variantes 100 % puras
    pure_f2 = build_ablation_matrix("holographic_brain", market="US", include_bridges=False)
    pure_ids = {v["variant_id"] for v in pure_f2["variants"]}
    assert "nb_full" in pure_ids
    assert "nb_holographic_moe_soft_k3" not in pure_ids

    # Filtros de `list_variants`
    controls = list_variants(origin="matched_control")
    assert all(v.origin == "matched_control" for v in controls)
    assert len(controls) >= 10


def test_validate_catalog_integrity_catches_broken_controls_and_cross_family_soup() -> None:
    """`validate_catalog_integrity` falla explícitamente ante controles ausentes o sopas."""
    broken_ctrl = ModelVariantSpec(
        variant_id="orphan_var",
        family="mmarrs_titan",
        origin="sota_innovation",
        description="Missing matched control",
        matched_control_id="does_not_exist",
        supports_markets=("US", "CN"),
        max_loops=3,
        order_level_mode="decoupled_f7b",
        builder_target="mars_titan.models.mmarrs.model:build_variation[v07_moe_double_loop]",
        config_overrides={},
        tags=("broken",),
    )
    bad_cat = dict(TFM_VARIANT_CATALOG)
    bad_cat["orphan_var"] = broken_ctrl
    with pytest.raises(ValueError, match="matched_control_id"):
        validate_catalog_integrity(bad_cat)

    # Sopa sin control dentro de la misma familia (puente cuyo control pertenece a otra familia)
    cross_ctrl_soup = ModelVariantSpec(
        variant_id="bad_bridge_ctrl",
        family="holographic_brain",
        origin="sota_innovation",
        description="Bridge pointing outside its own family",
        matched_control_id="v00_dense_iso_flop",
        supports_markets=("US", "CN"),
        max_loops=3,
        order_level_mode="decoupled_f7b",
        builder_target="mars_titan.models.mmarrs.neuro_brain_sota:build_neuro_brain_ablation[full]",
        config_overrides={},
        tags=("bridge",),
        cross_family_bridge_of=("mmarrs_titan",),
    )
    bad_cat2 = dict(TFM_VARIANT_CATALOG)
    bad_cat2["bad_bridge_ctrl"] = cross_ctrl_soup
    with pytest.raises(ValueError, match="same family"):
        validate_catalog_integrity(bad_cat2)


def test_compact_json_and_gzip_catalog_serialization(tmp_path: Path) -> None:
    """Serialización compacta `.json` y `.json.gz` preserva todo el catálogo sin pérdida."""
    raw_path = tmp_path / "tfm_catalog.json"
    gz_path = tmp_path / "tfm_catalog.json.gz"

    raw_text = serialize_catalog_json(raw_path)
    gz_text = serialize_catalog_json(gz_path)
    assert raw_text == gz_text

    parsed_raw = json.loads(raw_path.read_text(encoding="utf-8"))
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        parsed_gz = json.loads(fh.read())

    assert parsed_raw == parsed_gz
    assert parsed_raw["total_variants"] == len(TFM_VARIANT_CATALOG)
    assert len(parsed_raw["catalog_sha256"]) == 64
    assert gz_path.read_bytes()[:2] == b"\x1f\x8b"
    assert gz_path.stat().st_size < raw_path.stat().st_size * 0.45


def test_p1_6_family3_registry_alignment_aliases_and_build_variation_compatibility() -> None:
    """P1-6: Familia 3 alinea 1:1 con `VARIATION_REGISTRY`, aliases y `build_variation`."""
    legacy_aliases = {
        "v00_dense_iso_flop": "v00_dense_baseline",
        "v02_neuromod_only": "v02_titans_neuromod_only",
        "v05_moe_time_shuffled_placebo": "v05_moe_shuffled_ctx_placebo",
        "v06_loop_only_k3": "v06_double_loop_pc_trm_only",
        "v07_moe_loop_k3": "v07_moe_double_loop",
        "v08_holo_bridge_comparison": "v08_holographic_3l_hrr_only",
        "v09_holo_moe_bridge_comparison": "v09_holographic_3l_moe_loop",
        "v10_rmt_hypergraph_only": "v10_cross_asset_hypergraph_only",
        "v11_moe_agent_p_plus": "v11_moe_plus_agent_core",
        "v12_moe_loop_p_plus": "v12_moe_loop_plus_agent_core",
        "v13_mmarrs_full": "v13_mmarrs_titan_full",
    }
    for old_id, canonical_id in legacy_aliases.items():
        assert TFM_VARIANT_ALIASES[old_id] == canonical_id
        assert resolve_variant_id(old_id) == canonical_id
        assert get_variant(old_id) == get_variant(canonical_id)

    # Todas las variaciones con `build_variation` (Familia 3 + Familia 1) construyen sin error!
    mmarrs_specs = [
        s
        for s in TFM_VARIANT_CATALOG.values()
        if s.builder_target.startswith("mars_titan.models.mmarrs.model:build_variation")
    ]
    assert len(mmarrs_specs) >= len(VARIATION_REGISTRY)
    for spec in mmarrs_specs:
        assert spec.registry_key in VARIATION_REGISTRY
        if spec.family == "mmarrs_titan":
            assert spec.variant_id == spec.registry_key
        cfg = variation_config(spec.registry_key, **spec.config_overrides)
        assert cfg.variation == spec.registry_key
        model = build_variation(
            spec.registry_key,
            **spec.config_overrides,
            hidden_dim=16,
            n_cortical_areas=8,
            n_hyperedges=4,
            hrr_max_slots=16,
            allow_short_synthetic=True,
        )
        assert model.config.variation == spec.registry_key

    # Verificar que v13b y v13c activan exactamente sus banderas SOTA / FM-7 en MMARRSConfig
    s_13b = get_variant("v13b_mmarrs_multimodal_sota")
    cfg_13b = variation_config(s_13b.registry_key, **s_13b.config_overrides)
    assert cfg_13b.use_multimodal_sota is True
    assert cfg_13b.ssl_weight == pytest.approx(0.02)

    s_13c = get_variant("v13c_mmarrs_noic_pinball_twin")
    cfg_13c = variation_config(s_13c.registry_key, **s_13c.config_overrides)
    assert cfg_13c.noic_pinball_twin is True


def test_p1_6_build_ablation_matrix_n_trials_accounts_for_seeds_and_hpo() -> None:
    """P1-6: `n_trials` multiplica `n_variants * len(seeds) * n_hpo_trials_per_variant`."""
    mat_def = build_ablation_matrix("mmarrs_titan", market="US", seeds=(42, 43, 44))
    assert mat_def["n_seeds"] == 3
    assert mat_def["n_hpo_trials_per_variant"] == 1
    assert mat_def["n_trials"] == 3 * mat_def["n_variants"]

    mat_hpo = build_ablation_matrix(
        "mmarrs_titan",
        market="CN",
        seeds=(42, 43, 44, 45),
        n_hpo_trials_per_variant=6,
    )
    assert mat_hpo["n_seeds"] == 4
    assert mat_hpo["n_hpo_trials_per_variant"] == 6
    assert mat_hpo["n_trials"] == 4 * 6 * mat_hpo["n_variants"]

    with pytest.raises(ValueError):
        build_ablation_matrix("mmarrs_titan", market="US", n_hpo_trials_per_variant=0)


def test_wave2_v2_catalog_variants_instantiate_cleanly_via_build_family_model() -> None:
    """Wave-2: todas las variantes v2 (F1/F2/F3/F4) + controles y alias construyen limpias."""
    from mars_titan.models.mmarrs.model import build_family_model

    v2_by_family = {
        "titans_original": ("t15a_dual_axis_mem", "t15b_full_v2", "t15c_noic_twin"),
        "holographic_brain": (
            "nb_v2_fhrr_lgn_pi",
            "nb_v2_full",
            "nb_v2_noic_twin",
            "nb_v2_moe_soft_k3",
        ),
        "mmarrs_titan": (
            "v14a_tied_loop",
            "v14b_tied_xasset",
            "v14c_tied_xasset_cf",
            "v14c_placebo",
            "v14c_noic_twin",
        ),
        "titans_cognitive_core": (
            "cog_v2_10tools_uniform_k3",
            "cog_v2_10tools_shuffled_cf_placebo_k3",
            "cog_v2_10tools_soft_k3",
            "cog_v2_10tools_sparsemax_k3",
            "cog_v2_10tools_noic_twin",
        ),
    }
    dims = dict(
        prices_dim=16,
        fundamentals_dim=8,
        macro_dim=6,
        news_dim=8,
        charts_dim=8,
        physics_dim=12,
        context_dim=10,
        exposures_dim=4,
        rmt_dim=4,
        hidden_dim=16,
        allow_short_synthetic=True,
    )
    fps: set[str] = set()
    for fam, variants in v2_by_family.items():
        for vid in variants:
            spec = get_variant(vid)
            assert spec.family == fam
            model = build_family_model(vid, market="US", family=fam, **dims)
            audit = model.parameter_audit()
            assert audit["family"] == fam
            fps.add(str(audit["fingerprint"]))
            if spec.order_level_mode == "noic_pinball_twin":
                assert audit["noic_pinball_twin"] is True

    assert len(fps) == sum(len(v) for v in v2_by_family.values())

    # Alias F4 v1 -> `cog_soft_tools_k3`
    m_alias = build_family_model("cog_v1_6tools_k3", market="US", **dims)
    assert m_alias.config.family == "titans_cognitive_core"
    assert m_alias.config.cognitive_toolbox_version == "v1_6tools"
    assert m_alias.config.cognitive_tool_mode == "soft"

    # Anti-sopa: pedir familia equivocada para v2 debe lanzar ValueError
    with pytest.raises(ValueError, match="Incompatibilidad de familia"):
        build_family_model("t15b_full_v2", family="mmarrs_titan", **dims)
    with pytest.raises(ValueError, match="Incompatibilidad de familia"):
        build_family_model("cog_v2_10tools_soft_k3", family="mmarrs_titan", **dims)
    with pytest.raises(ValueError, match="Incompatibilidad de familia"):
        build_family_model("nb_v2_full", family="titans_original", **dims)
