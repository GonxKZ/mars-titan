"""Pruebas TDD del Ledger Permanente de Memoria de Investigación (`ResearchMemoryLedger`).

Verifica de forma estricta:
  1. `ExperimentRecord`: inmutabilidad, validación de las 4 familias independientes, mercados
     (`US|CN`), veredictos (`KEEP|REJECT|INVESTIGATE|BUG_CAUGHT`), consistencia de
     `GainValidationGate` y serialización determinista.
  2. `ResearchMemoryLedger`: persistencia JSONL append-only (`experiment_memory.jsonl`),
     recarga fiel, consulta `already_tested(fingerprint_sha256)` y rechazo de duplicados opcional.
  3. Consultas de investigación: `best_by_family(metric=...)` (anidado 6D y plano, mayor/menor
     mejor, aislamiento de `REJECT`/`BUG_CAUGHT` y separación `US|CN`), `rejected_hypotheses(...)`
     y cola de hipótesis abiertas.
  4. `export_colab_champions_manifest(top_k_per_family=2)`: integración real con
     `GainValidationGate` exportando EXCLUSIVAMENTE variaciones `KEEP` certificadas `PASS` listas
     para pre-entrenamiento `2000–2023` en Google Colab RTX PRO 6000 (`sm_120`, `2024+` sellado).
  5. `render_markdown(path)` y verificación del documento semilla
     `docs/research/mmarrs_experiment_memory.md` con todos los hallazgos y bugs reales de la sesión.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mars_titan.models.mmarrs.long_running_harness import (
    DEFAULT_MATCHED_CONTROLS,
    GainValidationGate,
)
from mars_titan.models.mmarrs.research_memory import (
    EXPERIMENT_VERDICTS,
    MARKETS,
    RESEARCH_FAMILIES,
    ExperimentRecord,
    ResearchMemoryLedger,
    build_initial_session_records,
    canonical_experiment_fingerprint,
)


def _six_dim_report(
    *,
    rank_ic: float,
    pure_alpha_ic: float,
    t_nw: float = 3.2,
    crps: float = 0.0085,
    cov_80: float = 0.80,
    sr1: float = 1.45,
    sr2: float = 0.95,
    dsr: float = 0.92,
) -> dict[str, dict[str, float]]:
    return dict(
        dim2_ranking=dict(
            rank_ic_mean=rank_ic,
            pure_alpha_ic_mean=pure_alpha_ic,
            rank_ic_t_nw=t_nw,
        ),
        dim4_uncertainty=dict(
            crps=crps,
            cov_80=cov_80,
        ),
        dim6_economics=dict(
            net_sharpe_1x=sr1,
            net_sharpe_2x=sr2,
            dsr=dsr,
        ),
    )


def _make_record(
    *,
    experiment_id: str,
    family: str = "mmarrs_titan",
    variation: str = "v13_mmarrs_titan_full",
    market: str = "US",
    verdict: str = "KEEP",
    rank_ic: float = 0.120,
    crps: float = 0.0082,
    sr1: float = 1.50,
    gain_gate_passed: bool = False,
    gain_verdict: dict[str, object] | None = None,
    fingerprint_sha256: str | None = None,
    failure_mode: str = "none",
    next_hypothesis: str = "Verify on 2000-2023 full history",
) -> ExperimentRecord:
    fp = fingerprint_sha256 or canonical_experiment_fingerprint(
        family=family,
        variation=variation,
        market=market,
        experiment_id=experiment_id,
    )
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp_utc="2026-11-23T11:00:00+00:00",
        family=family,
        variation=variation,
        market=market,
        hypothesis=f"Test {variation} in {family} ({market})",
        changes_made=f"Configured {variation}",
        metrics_6d=_six_dim_report(
            rank_ic=rank_ic, pure_alpha_ic=rank_ic * 0.9, crps=crps, sr1=sr1
        ),
        vs_controls_delta=dict(dense_iso_flop=dict(delta_rank_ic=0.004, delta_net_sharpe_1x=0.25)),
        verdict=verdict,
        failure_mode_or_root_cause=failure_mode,
        lessons_learned=f"Observed rank_ic={rank_ic:.4f}",
        next_hypothesis=next_hypothesis,
        hardware_env="cpu_test_fp32_strict",
        fingerprint_sha256=fp,
        gain_gate_passed=gain_gate_passed,
        gain_verdict=gain_verdict,
    )


def test_experiment_record_validation_immutability_and_roundtrip() -> None:
    """`ExperimentRecord` valida familias/mercados/veredictos, es inmutable y serializa exacto."""
    assert RESEARCH_FAMILIES == (
        "titans_original",
        "holographic_brain",
        "mmarrs_titan",
        "titans_cognitive_core",
    )
    assert MARKETS == ("US", "CN")
    assert EXPERIMENT_VERDICTS == ("KEEP", "REJECT", "INVESTIGATE", "BUG_CAUGHT")

    rec = _make_record(experiment_id="exp_001", gain_gate_passed=True)
    with pytest.raises(AttributeError):
        rec.verdict = "REJECT"  # type: ignore[misc]

    rebuilt = ExperimentRecord.from_dict(rec.to_dict())
    assert rebuilt == rec

    # Validaciones de contrato estricto
    with pytest.raises(ValueError):
        _make_record(experiment_id="bad_fam", family="blended_soup")
    with pytest.raises(ValueError):
        _make_record(experiment_id="bad_mkt", market="US+CN")
    with pytest.raises(ValueError):
        _make_record(experiment_id="bad_ver", verdict="MAYBE")
    with pytest.raises(ValueError):
        _make_record(experiment_id="", variation="v00")
    with pytest.raises(ValueError):
        # Contradicción prohibida: `gain_gate_passed=True` con `gain_verdict` REJECT
        _make_record(
            experiment_id="bad_gate",
            verdict="KEEP",
            gain_gate_passed=True,
            gain_verdict=dict(verdict="REJECT", promoted=False),
        )


def test_ledger_append_only_jsonl_reload_and_already_tested(tmp_path: Path) -> None:
    """`ResearchMemoryLedger` persiste en JSONL append-only, recarga fielmente y deduplica."""
    db_path = tmp_path / "experiment_memory.jsonl"
    ledger = ResearchMemoryLedger(db_path)

    r1 = _make_record(experiment_id="exp_t1", family="titans_original", variation="t02_online")
    r2 = _make_record(
        experiment_id="exp_h1",
        family="holographic_brain",
        variation="h04_hsi",
        market="CN",
        verdict="INVESTIGATE",
    )
    # Incluir float NaN en métricas para confirmar sanitización estricta RFC 8259
    r3 = ExperimentRecord(
        experiment_id="exp_bug1",
        timestamp_utc="2026-11-23T11:05:00+00:00",
        family="titans_cognitive_core",
        variation="c02_cog_core_soft_tools",
        market="US",
        hypothesis="BF16 linalg.solve stability check",
        changes_made="Ran raw BF16 without upcast",
        metrics_6d=dict(rank_ic_mean=float("nan"), crps=float("inf")),
        vs_controls_delta={},
        verdict="BUG_CAUGHT",
        failure_mode_or_root_cause="BF16 linalg.solve and FP16 -1e6 mask overflow to NaN",
        lessons_learned="Always upcast sensitive linear algebra via _stable_calc_dtype",
        next_hypothesis="Re-test with _stable_calc_dtype FP32 upcast",
        hardware_env="blackwell_sm120_bf16_amp",
        fingerprint_sha256="a" * 64,
    )

    ledger.record(r1)
    ledger.record(r2)
    ledger.record(r3)

    assert ledger.already_tested(r1.fingerprint_sha256) is True
    assert ledger.already_tested(r3.fingerprint_sha256) is True
    assert ledger.already_tested("f" * 64) is False

    with pytest.raises(ValueError):
        ledger.record(r1, allow_duplicate_fingerprint=False)

    # Archivo JSONL estricto: exactamente 3 líneas parseables sin sobrescritura
    lines = db_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed_bug = json.loads(lines[2])
    assert parsed_bug["metrics_6d"]["rank_ic_mean"] is None
    assert parsed_bug["metrics_6d"]["crps"] is None

    # Recarga en una nueva instancia conserva todo y sigue en modo append-only
    reloaded = ResearchMemoryLedger(db_path)
    assert len(reloaded.all_records()) == 3
    assert reloaded.already_tested(r1.fingerprint_sha256) is True
    assert reloaded.already_tested(r2.fingerprint_sha256) is True

    r4 = _make_record(
        experiment_id="exp_m1", family="mmarrs_titan", variation="v07_moe_double_loop"
    )
    reloaded.record(r4)
    assert len(db_path.read_text(encoding="utf-8").splitlines()) == 4


def test_query_apis_best_by_family_and_rejected_hypotheses(tmp_path: Path) -> None:
    """`best_by_family` excluye REJECT/BUG_CAUGHT, soporta menor-mejor y filtra por mercado."""
    ledger = ResearchMemoryLedger(tmp_path / "memory.jsonl")

    # Titans Original: KEEP 0.1163 vs KEEP 0.1107 vs REJECT (falso IC 0.45 por bug/fuga!)
    ledger.record(
        _make_record(
            experiment_id="t_base",
            family="titans_original",
            variation="t02_base",
            rank_ic=0.1107,
            crps=0.0095,
            verdict="KEEP",
        )
    )
    ledger.record(
        _make_record(
            experiment_id="t_best",
            family="titans_original",
            variation="t10_xrank_ic",
            rank_ic=0.1163,
            crps=0.0080,
            verdict="KEEP",
        )
    )
    ledger.record(
        _make_record(
            experiment_id="t_leak_bug",
            family="titans_original",
            variation="t_bad_cov",
            rank_ic=0.4500,
            crps=0.0020,
            verdict="BUG_CAUGHT",
            failure_mode="alpha_evaluation covariate leaked target",
        )
    )
    ledger.record(
        _make_record(
            experiment_id="t_plr_rej",
            family="titans_original",
            variation="t_plr",
            rank_ic=0.1010,
            crps=0.0110,
            verdict="REJECT",
            failure_mode="PLR periodic embeddings degraded IC by -0.015",
        )
    )
    # Holographic Brain: US vs CN separados
    ledger.record(
        _make_record(
            experiment_id="h_us",
            family="holographic_brain",
            variation="h06_holo3l_us",
            market="US",
            rank_ic=0.1180,
            crps=0.0084,
            verdict="KEEP",
        )
    )
    ledger.record(
        _make_record(
            experiment_id="h_cn",
            family="holographic_brain",
            variation="h06_holo3l_cn",
            market="CN",
            rank_ic=0.1320,
            crps=0.0090,
            verdict="KEEP",
        )
    )

    best_ic = ledger.best_by_family(metric="rank_ic_mean")
    assert best_ic["titans_original"].experiment_id == "t_best"
    assert best_ic["holographic_brain"].experiment_id == "h_cn"

    best_us = ledger.best_by_family(metric="rank_ic_mean", market="US")
    assert best_us["holographic_brain"].experiment_id == "h_us"

    # Métrica menor-es-mejor (`crps`)
    best_crps = ledger.best_by_family(metric="crps")
    assert best_crps["titans_original"].experiment_id == "t_best"
    assert best_crps["holographic_brain"].experiment_id == "h_us"

    # `rejected_hypotheses` global y filtrada por familia
    all_rej = ledger.rejected_hypotheses()
    assert {r.experiment_id for r in all_rej} == {"t_leak_bug", "t_plr_rej"}
    rej_only = ledger.rejected_hypotheses(family="titans_original", include_bugs=False)
    assert [r.experiment_id for r in rej_only] == ["t_plr_rej"]
    assert ledger.rejected_hypotheses(family="holographic_brain") == []


def test_export_colab_champions_manifest_requires_keep_and_gain_validation_gate(
    tmp_path: Path,
) -> None:
    """`export_colab_champions_manifest` exporta SOLO `KEEP` con `GainValidationGate` PASS."""
    ledger = ResearchMemoryLedger(tmp_path / "memory.jsonl")
    gate = GainValidationGate()

    controls = {
        role: _six_dim_report(rank_ic=0.105, pure_alpha_ic=0.095, crps=0.0095, sr1=1.10, sr2=0.65)
        for role in DEFAULT_MATCHED_CONTROLS
    }

    # 3 candidatos PASS en `mmarrs_titan` (top_k=2 debe elegir solo los 2 mejores)
    for idx, (var, ric) in enumerate(
        [
            ("v07_moe_double_loop", 0.119),
            ("v13_mmarrs_titan_full", 0.126),
            ("v13b_mmarrs_multimodal_sota", 0.131),
        ]
    ):
        rep = _six_dim_report(
            rank_ic=ric, pure_alpha_ic=ric * 0.92, crps=0.0080, sr1=1.60, sr2=1.05
        )
        verdict = gate.evaluate(var, rep, controls)
        assert verdict.verdict == "PASS" and verdict.promoted is True
        base_rec = _make_record(
            experiment_id=f"mm_{idx}",
            family="mmarrs_titan",
            variation=var,
            rank_ic=ric,
            verdict="KEEP",
        )
        ledger.record_with_gain_gate(base_rec, verdict)

    # 1 candidato PASS en `titans_cognitive_core`
    cog_rep = _six_dim_report(rank_ic=0.123, pure_alpha_ic=0.114, crps=0.0081, sr1=1.55, sr2=0.98)
    cog_verdict = gate.evaluate("c02_cog_core_soft_tools", cog_rep, controls)
    assert cog_verdict.promoted is True
    ledger.record_with_gain_gate(
        _make_record(
            experiment_id="cc_0",
            family="titans_cognitive_core",
            variation="c02_cog_core_soft_tools",
            rank_ic=0.123,
            verdict="KEEP",
        ),
        cog_verdict,
    )

    # Candidato `KEEP` SIN pasar `GainValidationGate` (IC enorme 0.95) -> DEBE QUEDAR FUERA!
    ledger.record(
        _make_record(
            experiment_id="uncertified_keep",
            family="mmarrs_titan",
            variation="v_unverified",
            rank_ic=0.950,
            verdict="KEEP",
            gain_gate_passed=False,
        )
    )
    # Bypass `gain_gate_passed=True` SIN `gain_verdict` real -> TAMBIÉN BLOQUEADO!
    ledger.record(
        _make_record(
            experiment_id="bypass_no_verdict",
            family="mmarrs_titan",
            variation="v_bypass",
            rank_ic=0.990,
            verdict="KEEP",
            gain_gate_passed=True,
            gain_verdict=None,
        )
    )

    # Candidato evaluado con `GainValidationGate` que FALLA -> degradado a REJECT y EXCLUIDO!
    weak_rep = _six_dim_report(rank_ic=0.104, pure_alpha_ic=0.094, crps=0.0110, sr1=0.90, sr2=-0.10)
    weak_verdict = gate.evaluate("v_weak", weak_rep, controls)
    assert weak_verdict.verdict == "REJECT"
    stored_weak = ledger.record_with_gain_gate(
        _make_record(
            experiment_id="failed_gate",
            family="holographic_brain",
            variation="h_weak",
            rank_ic=0.104,
            verdict="KEEP",
        ),
        weak_verdict,
    )
    assert stored_weak.verdict == "REJECT"
    assert stored_weak.gain_gate_passed is False

    out_manifest = tmp_path / "colab_champions.json"
    manifest = ledger.export_colab_champions_manifest(
        top_k_per_family=2,
        output_path=out_manifest,
    )

    assert out_manifest.is_file()
    persisted = json.loads(out_manifest.read_text(encoding="utf-8"))
    assert persisted["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["sealed_test_start_year"] == 2024
    assert manifest["markets_strictly_separate"] is True
    assert "sm_120" in manifest["hardware_target"]

    mm_champs = manifest["champions_by_family"]["mmarrs_titan"]
    assert [c["variation"] for c in mm_champs] == [
        "v13b_mmarrs_multimodal_sota",
        "v13_mmarrs_titan_full",
    ]
    assert all(c["gain_gate_passed"] is True and c["verdict"] == "KEEP" for c in mm_champs)
    assert len(manifest["champions_by_family"]["titans_cognitive_core"]) == 1
    assert manifest["champions_by_family"]["holographic_brain"] == []
    assert manifest["champions_by_family"]["titans_original"] == []
    assert manifest["total_champions"] == 3


def test_render_markdown_and_seeded_research_memory_doc(tmp_path: Path) -> None:
    """`render_markdown` y `docs/research/mmarrs_experiment_memory.md` incluyen los hallazgos."""
    seed_records = build_initial_session_records()
    assert len(seed_records) >= 12
    assert {r.family for r in seed_records} == set(RESEARCH_FAMILIES)
    assert {"KEEP", "REJECT", "INVESTIGATE", "BUG_CAUGHT"}.issubset(
        {r.verdict for r in seed_records}
    )

    ledger = ResearchMemoryLedger(tmp_path / "seeded.jsonl")
    for rec in seed_records:
        ledger.record(rec, allow_duplicate_fingerprint=False)

    rendered_path = tmp_path / "rendered.md"
    md = ledger.render_markdown(rendered_path)
    assert rendered_path.is_file()
    assert rendered_path.read_text(encoding="utf-8") == md

    required_sections = (
        "Verified Gains Leaderboard",
        "Graveyard of Rejected Ideas & Root Causes",
        "Open Hypotheses Queue",
        "titans_original",
        "holographic_brain",
        "mmarrs_titan",
        "titans_cognitive_core",
    )
    for token in required_sections:
        assert token in md, f"Falta sección/familia en render_markdown: {token}"

    # Verificar el documento permanente sembrado en `docs/research/mmarrs_experiment_memory.md`
    repo_root = Path(__file__).resolve().parents[2]
    doc_path = repo_root / "docs" / "research" / "mmarrs_experiment_memory.md"
    assert doc_path.is_file(), f"Falta documento permanente: {doc_path}"
    doc_text = doc_path.read_text(encoding="utf-8")

    required_findings = (
        "alpha_evaluation",
        "Almgren-Chriss",
        "tl.constexpr",
        "DMA",
        "bfloat16",
        "linalg.solve",
        "FM-7",
        "FM-8",
        "PLR",
        "55-session",
        "150K",
        "2000",
        "2024",
        "titans_original",
        "holographic_brain",
        "mmarrs_titan",
        "titans_cognitive_core",
        "Verified Gains Leaderboard",
        "Graveyard of Rejected Ideas",
        "Open Hypotheses Queue",
    )
    for token in required_findings:
        assert token in doc_text, f"Falta hallazgo/bug obligatorio en {doc_path}: {token}"


def test_export_colab_champions_keeps_us_and_cn_strictly_separate(tmp_path: Path) -> None:
    """Con `market=None`, CN de IC alto nunca eclipsa a US (`top_k` por `(familia, mercado)`)."""
    ledger = ResearchMemoryLedger(tmp_path / "us_cn_memory.jsonl")
    gate = GainValidationGate()
    controls = {
        role: _six_dim_report(rank_ic=0.100, pure_alpha_ic=0.090, crps=0.0095, sr1=1.00, sr2=0.55)
        for role in DEFAULT_MATCHED_CONTROLS
    }

    # 2 campeones CN con IC muy alto (0.155, 0.148) + 2 campeones US (0.125, 0.118)
    for exp_id, mkt, var, ric in [
        ("cn_1", "CN", "v13_cn_a", 0.155),
        ("cn_2", "CN", "v13_cn_b", 0.148),
        ("us_1", "US", "v13_us_a", 0.125),
        ("us_2", "US", "v13_us_b", 0.118),
    ]:
        rep = _six_dim_report(rank_ic=ric, pure_alpha_ic=ric * 0.9, crps=0.0080, sr1=1.60, sr2=1.05)
        verdict = gate.evaluate(var, rep, controls)
        assert verdict.promoted is True
        ledger.record_with_gain_gate(
            _make_record(
                experiment_id=exp_id,
                family="mmarrs_titan",
                variation=var,
                market=mkt,
                rank_ic=ric,
                verdict="KEEP",
            ),
            verdict,
        )

    manifest = ledger.export_colab_champions_manifest(top_k_per_family=1)
    by_fm = manifest["champions_by_family_and_market"]["mmarrs_titan"]
    assert [c["variation"] for c in by_fm["US"]] == ["v13_us_a"]
    assert [c["variation"] for c in by_fm["CN"]] == ["v13_cn_a"]
    # `champions_by_family` preserva tanto el campeón US como el CN (nunca eclipsados!)
    combined = manifest["champions_by_family"]["mmarrs_titan"]
    assert {(c["market"], c["variation"]) for c in combined} == {
        ("US", "v13_us_a"),
        ("CN", "v13_cn_a"),
    }
    assert manifest["total_champions"] == 2


def test_render_markdown_sorts_zero_rank_ic_above_negative_ic(tmp_path: Path) -> None:
    """`rank_ic_mean == 0.0` debe ordenarse ESTRICTAMENTE por encima de `-0.0500`."""
    ledger = ResearchMemoryLedger(tmp_path / "zero_sort.jsonl")
    ledger.record(
        _make_record(
            experiment_id="neg_ic_exp",
            family="titans_original",
            variation="t_neg",
            rank_ic=-0.0500,
            verdict="KEEP",
        )
    )
    ledger.record(
        _make_record(
            experiment_id="zero_ic_exp",
            family="titans_original",
            variation="t_zero",
            rank_ic=0.0,
            verdict="KEEP",
        )
    )
    md = ledger.render_markdown()
    lb_section = md.split("## Verified Gains Leaderboard", 1)[1].split(
        "## Per-Family Experiment Ledgers", 1
    )[0]
    assert lb_section.index("zero_ic_exp") < lb_section.index("neg_ic_exp")


def test_ledger_native_jsonl_gz_append_only_roundtrip_and_size_reduction(tmp_path: Path) -> None:
    """`.jsonl.gz` es estrictamente append-only, legible tras recarga, sin pérdida y compacto."""
    gz_path = tmp_path / "experiment_memory.jsonl.gz"
    raw_path = tmp_path / "experiment_memory.jsonl"

    seed = build_initial_session_records()
    ledger_gz = ResearchMemoryLedger(gz_path)
    ledger_raw = ResearchMemoryLedger(raw_path)

    # Escribir la primera mitad y recargar para comprobar append concatenado multi-sesión
    half = len(seed) // 2
    for rec in seed[:half]:
        ledger_gz.record(rec, allow_duplicate_fingerprint=False)
        ledger_raw.record(rec, allow_duplicate_fingerprint=False)

    reloaded_gz = ResearchMemoryLedger(gz_path)
    assert len(reloaded_gz.all_records()) == half
    for rec in seed[half:]:
        reloaded_gz.record(rec, allow_duplicate_fingerprint=False)
        ledger_raw.record(rec, allow_duplicate_fingerprint=False)

    final_gz = ResearchMemoryLedger(gz_path)
    final_raw = ResearchMemoryLedger(raw_path)
    assert final_gz.all_records() == final_raw.all_records()
    assert gz_path.read_bytes()[:2] == b"\x1f\x8b"
    assert gz_path.stat().st_size < raw_path.stat().st_size * 0.65


def test_p2_export_colab_champions_manifest_market_train_window_and_precision_mode(
    tmp_path: Path,
) -> None:
    """P2: `train_window` respeta mercado (US/CN) y `precision_mode` es configurable."""
    ledger = ResearchMemoryLedger(tmp_path / "p2_manifest.jsonl")
    gate = GainValidationGate()
    controls = {
        role: _six_dim_report(rank_ic=0.100, pure_alpha_ic=0.090, crps=0.0095, sr1=1.00, sr2=0.55)
        for role in DEFAULT_MATCHED_CONTROLS
    }

    us_rep = _six_dim_report(rank_ic=0.128, pure_alpha_ic=0.118, crps=0.0080, sr1=1.60, sr2=1.05)
    us_verdict = gate.evaluate("v13_mmarrs_titan_full", us_rep, controls)
    assert us_verdict.promoted is True
    ledger.record_with_gain_gate(
        ExperimentRecord(
            experiment_id="us_bf16_champ",
            timestamp_utc="2026-11-23T11:30:00+00:00",
            family="mmarrs_titan",
            variation="v13_mmarrs_titan_full",
            market="US",
            hypothesis="US champion trained under BF16 AMP on Blackwell",
            changes_made="bf16_amp run",
            metrics_6d=us_rep,
            vs_controls_delta={},
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned="BF16 AMP clean",
            next_hypothesis="Deploy to Colab",
            hardware_env="rtx_pro_6000_sm120_bf16_amp",
            fingerprint_sha256=canonical_experiment_fingerprint(exp="us_bf16_champ"),
        ),
        us_verdict,
    )

    cn_rep = _six_dim_report(rank_ic=0.145, pure_alpha_ic=0.132, crps=0.0081, sr1=1.65, sr2=1.08)
    cn_verdict = gate.evaluate("v13b_mmarrs_multimodal_sota", cn_rep, controls)
    assert cn_verdict.promoted is True
    ledger.record_with_gain_gate(
        ExperimentRecord(
            experiment_id="cn_fp32_champ",
            timestamp_utc="2026-11-23T11:35:00+00:00",
            family="mmarrs_titan",
            variation="v13b_mmarrs_multimodal_sota",
            market="CN",
            hypothesis="CN champion trained under FP32 strict on CSI 300 2006-2022",
            changes_made="fp32_strict run",
            metrics_6d=cn_rep,
            vs_controls_delta={},
            verdict="KEEP",
            failure_mode_or_root_cause="none",
            lessons_learned="CN window starts 2006",
            next_hypothesis="Deploy CN to Colab",
            hardware_env="dgx_fp32_strict_cn",
            fingerprint_sha256=canonical_experiment_fingerprint(exp="cn_fp32_champ"),
        ),
        cn_verdict,
    )

    # 1) Mercado US: train_window="2000-2022" y preserva precisión `bf16_amp` del registro
    man_us = ledger.export_colab_champions_manifest(top_k_per_family=2, market="US")
    assert man_us["train_window"] == "2000-2022"
    assert man_us["train_windows_by_market"] == {"US": "2000-2022"}
    assert man_us["precision_mode"] == "bf16_amp"
    us_entry = man_us["champions_by_family_and_market"]["mmarrs_titan"]["US"][0]
    assert us_entry["train_window"] == "2000-2022"
    assert us_entry["precision_mode"] == "bf16_amp"

    # 2) Mercado CN: train_window="2006-2022" y preserva precisión `fp32_strict` del registro
    man_cn = ledger.export_colab_champions_manifest(top_k_per_family=2, market="CN")
    assert man_cn["train_window"] == "2006-2022"
    assert man_cn["train_windows_by_market"] == {"CN": "2006-2022"}
    assert man_cn["precision_mode"] == "fp32_strict"
    cn_entry = man_cn["champions_by_family_and_market"]["mmarrs_titan"]["CN"][0]
    assert cn_entry["train_window"] == "2006-2022"
    assert cn_entry["precision_mode"] == "fp32_strict"

    # 3) Override explícito de `precision_mode`
    man_cn_bf16 = ledger.export_colab_champions_manifest(
        top_k_per_family=2, market="CN", precision_mode="bf16_amp"
    )
    assert man_cn_bf16["train_window"] == "2006-2022"
    assert man_cn_bf16["precision_mode"] == "bf16_amp"

    # 4) Exportación combinada (`market=None`) conserva ventanas separadas y precisión por campeón
    man_all = ledger.export_colab_champions_manifest(top_k_per_family=2, market=None)
    assert man_all["train_windows_by_market"] == {"US": "2000-2022", "CN": "2006-2022"}
    assert man_all["train_window"] == {"US": "2000-2022", "CN": "2006-2022"}
    assert man_all["precision_mode"] == "bf16_amp"
    assert (
        man_all["champions_by_family_and_market"]["mmarrs_titan"]["US"][0]["precision_mode"]
        == "bf16_amp"
    )
    assert (
        man_all["champions_by_family_and_market"]["mmarrs_titan"]["CN"][0]["precision_mode"]
        == "fp32_strict"
    )

    with pytest.raises(ValueError):
        ledger.export_colab_champions_manifest(precision_mode="")
