"""Pruebas TDD del arnés de larga duración `LongRunningTitansHarness` (2000–2023+ -> 2024+ sellado).

Verifica de forma estricta:
  1. `HarnessConfig`: contrato inmutable y validación de modos (`full_history_final`,
     `expanding_walk_forward`) y umbrales de salud/compactación/sello 2024+.
  2. Fase Inicializadora (`initialize`): creación de `manifest.json` con huella SHA-256,
     telemetría RTX PRO 6000 (`sm_120`), contrato de variables, estado del sello 2024+,
     e inicialización append-only de `progress_ledger.jsonl` sin sobrescribir progreso previo.
  3. Trabajador Incremental (`run_incremental_sessions`): ejecución por tramos -> parada a mitad ->
     reanudación desde checkpoint atómico con continuidad exacta, cero sesiones duplicadas y
     ledger JSONL válido.
  4. `SessionHealthGate`: detección de NaN/Inf en pesos y predicciones, cuantiles cruzados,
     violación de cota PT1 de Lyapunov, picos OOD de Hopfield, desbalanceo MoE y alerta CUSUM;
     rollback automático seguro ante inyección de NaN (`auto_recover_on_nan`).
  5. Compactación Zero-Drift (`compact_working_state`): amortigua la deriva rápida (`fast`,
     `fast_mom`) preservando bit-a-bit los bancos cristalizados `medium`/`slow` y `HRR-Hopfield`.
  6. Sello estricto `2024+`: rechazo categórico a incluir sesiones `year >= 2024` en
     entrenamiento, ajuste o splits walk-forward.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from mars_titan.data.market_physics_v4_gpu import ZeroCopySessionStore, ZeroCopySessionView
from mars_titan.models.mmarrs import long_running_harness as lrh
from mars_titan.models.mmarrs.full_history_train import FullHistorySplitConfig
from mars_titan.models.mmarrs.long_running_harness import (
    DEFAULT_MATCHED_CONTROLS,
    PIPELINE_STAGES,
    GainValidationGate,
    GainVerdict,
    HarnessConfig,
    HealthReport,
    LongRunningTitansHarness,
    SealedTestViolationError,
    SessionHealthGate,
    _align_w_prev_by_asset_ids,
    compact_working_state,
    read_progress_ledger,
    required_purge_embargo,
)
from mars_titan.models.mmarrs.model import build_variation


def _make_synthetic_sessions(
    years: list[int],
    *,
    n_assets: int = 16,
    seed: int = 123,
) -> list[dict[str, torch.Tensor]]:
    """Genera sesiones sintéticas multimodales deterministas compatibles con `MMARRSTitan`."""
    rng = np.random.default_rng(seed)
    sessions: list[dict[str, torch.Tensor]] = []
    for _ in years:

        def _t(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.tensor(rng.standard_normal(shape), dtype=torch.float32)

        prices = _t((n_assets, 16))
        physics = _t((n_assets, 12))
        mask = torch.ones((n_assets, 5), dtype=torch.float32)
        mask[0::5, 3] = 0.0
        y1 = 0.008 * prices[:, 0] + 0.006 * physics[:, 0] + 0.010 * _t((n_assets,))
        targets = torch.stack(
            [y1 * float(np.sqrt(h)) + 0.004 * _t((n_assets,)) for h in (1, 2, 5, 10, 21, 63)],
            dim=-1,
        )
        sessions.append(
            dict(
                prices=prices,
                fundamentals=_t((n_assets, 8)),
                macro=_t((n_assets, 6)),
                news=_t((n_assets, 8)),
                charts=_t((n_assets, 8)),
                modality_mask=mask,
                physics_v4=physics,
                high_snr_context=_t((n_assets, 10)),
                exposures=_t((n_assets, 4)),
                rmt_loadings=_t((n_assets, 4)),
                regime_vec=_t((10,)),
                ach_signal=torch.abs(_t((n_assets, 1))) * 0.5,
                ne_signal=torch.tensor([0.2], dtype=torch.float32),
                target_returns=targets,
            )
        )
    return sessions


def test_harness_config_defaults_immutability_and_validation(tmp_path: Path):
    """`HarnessConfig` es inmutable, respeta los valores por defecto y valida invariantes."""
    cfg = HarnessConfig(
        run_dir=str(tmp_path / "run_00"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
    )
    assert cfg.checkpoint_every_sessions == 250
    assert cfg.compaction_interval == 63
    assert cfg.max_ood_energy == pytest.approx(4.0)
    assert cfg.max_load_imbalance == pytest.approx(2.5)
    assert cfg.cusum_alert_threshold == pytest.approx(0.25)
    assert cfg.auto_recover_on_nan is True
    assert cfg.sealed_test_start_year == 2024

    cfg_wf = HarnessConfig(
        run_dir=str(tmp_path / "run_wf"),
        variation="v07_moe_double_loop",
        split_mode="expanding_walk_forward",
    )
    assert cfg_wf.split_mode == "expanding_walk_forward"

    with pytest.raises(FrozenInstanceError):
        cfg.checkpoint_every_sessions = 10  # type: ignore[misc]

    with pytest.raises(ValueError):
        HarnessConfig(
            run_dir=str(tmp_path),
            variation="v99_unknown",
            split_mode="full_history_final",
        )
    with pytest.raises(ValueError):
        HarnessConfig(run_dir=str(tmp_path), variation="v13_mmarrs_titan_full", split_mode="random")
    with pytest.raises(ValueError):
        HarnessConfig(
            run_dir=str(tmp_path),
            variation="v13_mmarrs_titan_full",
            split_mode="full_history_final",
            checkpoint_every_sessions=0,
        )
    with pytest.raises(ValueError):
        HarnessConfig(
            run_dir=str(tmp_path),
            variation="v13_mmarrs_titan_full",
            split_mode="full_history_final",
            compaction_interval=0,
        )
    with pytest.raises(ValueError):
        HarnessConfig(
            run_dir=str(tmp_path),
            variation="v13_mmarrs_titan_full",
            split_mode="full_history_final",
            max_load_imbalance=0.5,
        )


def test_initializer_writes_sha256_manifest_telemetry_and_preserves_progress(tmp_path: Path):
    """Initializer escribe `manifest.json` SHA-256 + `progress_ledger.jsonl` sin sobrescribir."""
    run_dir = tmp_path / "init_run"
    years = [2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=11)

    cfg = HarnessConfig(
        run_dir=str(run_dir),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=2,
        compaction_interval=3,
        hidden_dim=16,
        seed=42,
        device="cpu",
    )
    harness = LongRunningTitansHarness(cfg)
    init_res = harness.initialize(sample_session=sessions[0])

    manifest_path = run_dir / "manifest.json"
    ledger_path = harness.ledger_path
    assert manifest_path.is_file()
    assert ledger_path.is_file()
    assert ledger_path.name == "progress_ledger.jsonl.gz"
    assert init_res["resumed"] is False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["architecture_fingerprint"]) == 64
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["variation"] == "v13_mmarrs_titan_full"
    assert manifest["seal_status"]["sealed_test_start_year"] == 2024
    assert manifest["seal_status"]["seal_enforced"] is True
    assert manifest["seal_status"]["allow_sealed_test"] is False
    # Telemetría de hardware RTX PRO 6000 sm_120 presente en el manifiesto
    specs = manifest["hardware_telemetry"]["target_specs"]
    assert specs["arch"] == "sm_120"
    assert specs["sms"] == 188
    assert specs["vram_gb"] == pytest.approx(96.0)
    assert specs["l2_cache_mb"] == 128
    # Contrato de variables presente
    assert manifest["feature_contract"]["dims"]["prices_dim"] == 16
    assert len(manifest["feature_contract"]["quantile_levels"]) == 9

    # Procesar 2 sesiones para poblar progreso en el ledger
    harness.run_incremental_sessions(sessions[:2], years[:2], total_planned_steps=4)
    lines_before = read_progress_ledger(ledger_path)
    completed_before = [e for e in lines_before if e["event"] == "session_completed"]
    assert len(completed_before) == 2

    # Reinicializar en el mismo `run_dir` NUNCA sobrescribe progreso completado
    harness_re = LongRunningTitansHarness(cfg)
    re_res = harness_re.initialize(sample_session=sessions[0])
    assert re_res["resumed"] is True
    assert re_res["session_cursor"] == 2
    lines_after = read_progress_ledger(ledger_path)
    assert len(lines_after) >= len(lines_before)
    completed_after = [e for e in lines_after if e["event"] == "session_completed"]
    assert len(completed_after) == 2

    # Intentar reinicializar con otra variación en el mismo `run_dir` falla explícitamente
    bad_cfg = HarnessConfig(
        run_dir=str(run_dir),
        variation="v00_dense_baseline",
        split_mode="full_history_final",
        hidden_dim=16,
        seed=42,
        device="cpu",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        LongRunningTitansHarness(bad_cfg).initialize(sample_session=sessions[0])


def test_incremental_run_stop_mid_run_and_resume_exact_continuity(tmp_path: Path):
    """Parar a mitad y reanudar desde checkpoint logra continuidad exacta sin duplicados."""
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=14, seed=77)
    n_total = len(sessions)

    # 1) Ejecución de referencia ininterrumpida
    ref_cfg = HarnessConfig(
        run_dir=str(tmp_path / "ref_run"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=3,
        compaction_interval=4,
        hidden_dim=16,
        lr=2e-3,
        seed=99,
        device="cpu",
    )
    h_ref = LongRunningTitansHarness(ref_cfg)
    h_ref.initialize(sample_session=sessions[0])
    ref_out = h_ref.run_incremental_sessions(sessions, years, total_planned_steps=n_total)
    assert ref_out["sessions_processed"] == n_total
    assert ref_out["compactions"] == 2  # pasos 4 y 8

    # 2) Tramos: 4 sesiones -> parada -> 3 sesiones -> parada -> resto + llamada idempotente
    res_dir = tmp_path / "resumed_run"
    res_cfg = HarnessConfig(
        run_dir=str(res_dir),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=3,
        compaction_interval=4,
        hidden_dim=16,
        lr=2e-3,
        seed=99,
        device="cpu",
    )

    h1 = LongRunningTitansHarness(res_cfg)
    h1.initialize(sample_session=sessions[0])
    out1 = h1.run_incremental_sessions(
        sessions, years, max_sessions=4, total_planned_steps=n_total
    )
    assert out1["sessions_processed"] == 4
    assert out1["session_cursor"] == 4

    h2 = LongRunningTitansHarness(res_cfg)
    init2 = h2.initialize(sample_session=sessions[0])
    assert init2["resumed"] is True
    assert init2["session_cursor"] == 4
    out2 = h2.run_incremental_sessions(
        sessions, years, max_sessions=3, total_planned_steps=n_total
    )
    assert out2["sessions_processed"] == 3
    assert out2["session_cursor"] == 7

    h3 = LongRunningTitansHarness(res_cfg)
    init3 = h3.initialize(sample_session=sessions[0])
    assert init3["resumed"] is True
    assert init3["session_cursor"] == 7
    out3 = h3.run_incremental_sessions(sessions, years, total_planned_steps=n_total)
    assert out3["sessions_processed"] == 3
    assert out3["session_cursor"] == n_total

    # Llamada repetida sobre sesiones ya completadas: cero trabajo duplicado
    out4 = h3.run_incremental_sessions(sessions, years, total_planned_steps=n_total)
    assert out4["sessions_processed"] == 0

    # Verificar paridad exacta de pesos, EMA y estado causal Titans respecto a referencia continua
    for k, ref_t in h_ref.model.state_dict().items():
        res_t = h3.model.state_dict()[k]
        torch.testing.assert_close(res_t, ref_t, atol=1e-6, rtol=1e-6)
    for k, ref_t in h_ref.ema_model.state_dict().items():
        res_t = h3.ema_model.state_dict()[k]
        torch.testing.assert_close(res_t, ref_t, atol=1e-6, rtol=1e-6)
    for bank in ("fast", "fast_mom", "medium", "slow"):
        torch.testing.assert_close(
            h3.causal_state["titans_state"][bank],
            h_ref.causal_state["titans_state"][bank],
            atol=1e-6,
            rtol=1e-6,
        )

    # Verificar ledger JSONL(.gz) válido y cero sesiones duplicadas
    ledger = read_progress_ledger(h3.ledger_path)
    completed = [e for e in ledger if e["event"] == "session_completed"]
    assert [e["session_idx"] for e in completed] == list(range(n_total))
    assert len({e["session_key"] for e in completed}) == n_total
    compactions = [e for e in ledger if e["event"] == "context_compaction"]
    assert len(compactions) == 2


def test_session_health_gate_catches_all_six_failure_and_warning_modes():
    """`SessionHealthGate` valida las 6 invariantes (NaN, cuantiles, PT1, MoE, OOD, CUSUM)."""
    torch.manual_seed(5)
    model = build_variation(
        "v13_mmarrs_titan_full",
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
    )
    sessions = _make_synthetic_sessions([2022], n_assets=12, seed=5)
    batch = sessions[0]
    model.eval()
    with torch.no_grad():
        healthy_out = model.forward_session(batch)

    gate = SessionHealthGate(
        max_ood_energy=4.0,
        max_load_imbalance=2.5,
        cusum_alert_threshold=0.25,
        max_memory_norm=25.0,
        auto_recover_on_nan=True,
    )

    # 1) Sesión sana pasa sin fallos duros ni advertencias
    rep_ok = gate.verify_session(
        model=model,
        outputs=healthy_out,
        state=healthy_out["state"],
        targets=batch["target_returns"],
        loss=torch.tensor(0.5),
    )
    assert isinstance(rep_ok, HealthReport)
    assert rep_ok.passed is True
    assert rep_ok.hard_failures == []
    assert rep_ok.metrics["weights_finite"] is True
    assert rep_ok.metrics["predictions_finite"] is True
    assert rep_ok.metrics["quantile_crossings"] == 0

    # 2) Predicción con NaN / Inf -> fallo duro + dispara callback de rollback
    bad_nan_out = dict(healthy_out)
    bad_pred = healthy_out["pred_return"].clone()
    bad_pred[0, 0] = float("nan")
    bad_nan_out["pred_return"] = bad_pred
    rollback_called = {"count": 0}

    def _on_rollback() -> None:
        rollback_called["count"] += 1

    rep_nan = gate.verify_session(
        model=model,
        outputs=bad_nan_out,
        state=healthy_out["state"],
        targets=batch["target_returns"],
        rollback_fn=_on_rollback,
    )
    assert rep_nan.passed is False
    assert any("non_finite_predictions" in f for f in rep_nan.hard_failures)
    assert rep_nan.rollback_triggered is True
    assert rollback_called["count"] == 1

    # 3) Cuantiles cruzados (intercambio de q3 y q4) -> fallo duro
    bad_q_out = dict(healthy_out)
    bad_q = healthy_out["quantiles"].clone()
    bad_q[0, 0, 3], bad_q[0, 0, 4] = bad_q[0, 0, 4].clone() + 0.5, bad_q[0, 0, 3].clone() - 0.5
    bad_q_out["quantiles"] = bad_q
    rep_q = gate.verify_session(model=model, outputs=bad_q_out, state=healthy_out["state"])
    assert rep_q.passed is False
    assert rep_q.metrics["quantile_crossings"] > 0
    assert any("crossing_quantiles" in f for f in rep_q.hard_failures)

    # 4) Violación de cota PT1 de Lyapunov en memoria -> fallo duro
    bad_state = dict(healthy_out["state"])
    bad_state["titans_state"] = dict(
        fast=torch.full((12, 16), 20.0),  # norm = 20 * 4 = 80 > 25
        fast_mom=torch.zeros(12, 16),
        medium=torch.zeros(12, 16),
        slow=torch.zeros(12, 16),
    )
    rep_pt1 = gate.verify_session(model=model, outputs=healthy_out, state=bad_state)
    assert rep_pt1.passed is False
    assert any("pt1_memory_norm" in f for f in rep_pt1.hard_failures)

    # 5) OOD moderado (advertencia) y extremo (fallo duro) + desbalanceo MoE extremo (fallo duro)
    mod_diag = dict(healthy_out["diagnostics"])
    mod_diag["ood_energy_mean"] = 5.5  # > 4.0 (advertencia), < 10.0 (duro)
    mod_diag["load_max_min_ratio"] = 3.2  # > 2.5 (advertencia), < 10.0 (duro)
    mod_out = dict(healthy_out, diagnostics=mod_diag)
    rep_warn = gate.verify_session(model=model, outputs=mod_out, state=healthy_out["state"])
    assert rep_warn.passed is True
    assert any("ood_energy" in w for w in rep_warn.warnings)
    assert any("moe_load_imbalance" in w for w in rep_warn.warnings)

    ext_diag = dict(healthy_out["diagnostics"])
    ext_diag["ood_energy_mean"] = 18.0
    ext_diag["load_max_min_ratio"] = 40.0
    ext_out = dict(healthy_out, diagnostics=ext_diag)
    rep_ext = gate.verify_session(model=model, outputs=ext_out, state=healthy_out["state"])
    assert rep_ext.passed is False
    assert any("ood_energy" in f for f in rep_ext.hard_failures)
    assert any("moe_load_imbalance" in f for f in rep_ext.hard_failures)

    # 6) Caída sostenida de IC dispara alerta CUSUM
    cusum_gate = SessionHealthGate(cusum_alert_threshold=0.25, cusum_drift=0.01)
    cusum_gate.verify_session(outputs=healthy_out, session_ic=0.15)
    rep_cusum = None
    for _ in range(5):
        rep_cusum = cusum_gate.verify_session(outputs=healthy_out, session_ic=-0.05)
    assert rep_cusum is not None
    assert rep_cusum.metrics["cusum_alert"] is True
    assert any("cusum" in w for w in rep_cusum.warnings)

    # 7) Pesos NaN en el modelo -> fallo duro
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    rep_w = gate.verify_session(model=model, outputs=healthy_out, state=healthy_out["state"])
    assert rep_w.passed is False
    assert rep_w.metrics["weights_finite"] is False
    assert any("non_finite_weights" in f for f in rep_w.hard_failures)


def test_auto_recover_on_nan_injection_rolls_back_to_last_verified_checkpoint(tmp_path: Path):
    """Ante inyección de NaN en medio del entrenamiento, revierte al último checkpoint sano."""
    years = [2019, 2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=33)

    cfg = HarnessConfig(
        run_dir=str(tmp_path / "nan_recovery"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=2,
        compaction_interval=10,
        auto_recover_on_nan=True,
        hidden_dim=16,
        seed=21,
        device="cpu",
    )
    harness = LongRunningTitansHarness(cfg)
    harness.initialize(sample_session=sessions[0])

    # Correr 2 sesiones limpias -> checkpoint verificado en cursor 2
    res_clean = harness.run_incremental_sessions(
        sessions, years, max_sessions=2, total_planned_steps=5
    )
    assert res_clean["sessions_processed"] == 2
    assert res_clean["rollbacks"] == 0
    snapshot_params = {k: v.detach().clone() for k, v in harness.model.state_dict().items()}

    # Inyectar NaN en los pesos durante la sesión 2 (tercera sesión) vía hook
    def _inject_nan_weight(idx: int, model: torch.nn.Module, _out: dict, _loss: dict) -> None:
        if idx == 2:
            with torch.no_grad():
                first_param = next(model.parameters())
                first_param.fill_(float("nan"))

    res_fail = harness.run_incremental_sessions(
        sessions,
        years,
        max_sessions=3,
        total_planned_steps=5,
        post_step_hook=_inject_nan_weight,
    )
    assert res_fail["rollbacks"] == 1
    assert res_fail["last_health_report"].passed is False
    assert res_fail["last_health_report"].rollback_triggered is True
    assert harness.session_cursor == 2

    # Todos los pesos y EMA deben estar 100% finitos e idénticos al último checkpoint sano
    for k, ref_t in snapshot_params.items():
        cur_t = harness.model.state_dict()[k]
        assert torch.isfinite(cur_t).all()
        torch.testing.assert_close(cur_t, ref_t, atol=0.0, rtol=0.0)
    for p in harness.ema_model.parameters():
        assert torch.isfinite(p).all()

    ledger = read_progress_ledger(harness.ledger_path)
    events = [e["event"] for e in ledger]
    assert "session_health_failure" in events
    assert "rollback_recovered" in events


def test_compact_working_state_damps_fast_drift_while_preserving_slow_and_hrr():
    """`compact_working_state` atenúa `fast`/`fast_mom` y preserva intactos `medium`/`slow`/HRR."""
    torch.manual_seed(17)
    model = build_variation(
        "v13_mmarrs_titan_full",
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
    )
    sessions = _make_synthetic_sessions([2020, 2021, 2022], n_assets=12, seed=17)
    model.eval()
    state = None
    with torch.no_grad():
        for b in sessions:
            out = model.forward_session(b, state=state)
            state = model.step_matured(
                out["state"],
                matured_batch=b,
                matured_targets=b["target_returns"],
                matured_preds=out["pred_return"][:, 0].detach(),
            )

    # Simular deriva acumulada en buffers rápidos tras cientos de sesiones
    state["titans_state"]["fast"] = state["titans_state"]["fast"] + 3.5
    state["titans_state"]["fast_mom"] = state["titans_state"]["fast_mom"] + 2.0

    fast_before = state["titans_state"]["fast"].clone()
    mom_before = state["titans_state"]["fast_mom"].clone()
    med_before = state["titans_state"]["medium"].clone()
    slow_before = state["titans_state"]["slow"].clone()
    hrr_keys_before = model.hrr_hopfield.slot_keys.clone()
    hrr_vals_before = model.hrr_hopfield.slot_vals.clone()
    hrr_filled_before = model.hrr_hopfield.slot_filled.clone()
    hrr_traces_before = model.hrr_hopfield.traces.clone()
    assert int(hrr_filled_before.sum().item()) > 0

    compacted_state, metrics = compact_working_state(
        state, fast_decay=0.65, momentum_decay=0.40, model=model
    )

    # 1) Buffers rápidos atenuados exactamente por los factores configurados
    torch.testing.assert_close(
        compacted_state["titans_state"]["fast"], fast_before * 0.65, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        compacted_state["titans_state"]["fast_mom"], mom_before * 0.40, atol=1e-6, rtol=1e-6
    )
    assert metrics["fast_norm_after"] < metrics["fast_norm_before"]
    assert metrics["fast_mom_norm_after"] < metrics["fast_mom_norm_before"]

    # 2) Bancos cristalizados Benna-Fusi `medium` y `slow` preservados bit-a-bit
    assert torch.equal(compacted_state["titans_state"]["medium"], med_before)
    assert torch.equal(compacted_state["titans_state"]["slow"], slow_before)
    assert metrics["medium_norm_after"] == pytest.approx(metrics["medium_norm_before"])
    assert metrics["slow_norm_after"] == pytest.approx(metrics["slow_norm_before"])

    # 3) Ranuras episódicas y trazas HRR-Hopfield preservadas bit-a-bit
    assert torch.equal(model.hrr_hopfield.slot_keys, hrr_keys_before)
    assert torch.equal(model.hrr_hopfield.slot_vals, hrr_vals_before)
    assert torch.equal(model.hrr_hopfield.slot_filled, hrr_filled_before)
    assert torch.equal(model.hrr_hopfield.traces, hrr_traces_before)
    assert metrics["hrr_slots_preserved"] == int(hrr_filled_before.sum().item())

    # 4) No muta los tensores originales del diccionario de entrada
    assert torch.equal(state["titans_state"]["fast"], fast_before)


def test_strict_2024_seal_enforcement_across_splits_and_training(tmp_path: Path):
    """Rechaza estrictamente cualquier intento de incluir `year >= 2024` en entrenamiento/tuning."""
    years = [y for y in (2020, 2021, 2022, 2023, 2024, 2025) for _ in range(3)]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=55)

    cfg = HarnessConfig(
        run_dir=str(tmp_path / "seal_check"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        sealed_test_start_year=2024,
        hidden_dim=16,
        device="cpu",
    )
    harness = LongRunningTitansHarness(cfg)
    harness.initialize(sample_session=sessions[0])

    # 1) Intento directo de entrenar sobre sesiones con 2024+ -> SealedTestViolationError
    with pytest.raises(SealedTestViolationError):
        harness.run_incremental_sessions(sessions, years)

    # 2) Intento de pasar FullHistorySplitConfig con allow_sealed_test=True -> rechazado
    unsealed_cfg = FullHistorySplitConfig(
        mode="full_history_final",
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        sealed_test_start_year=2024,
        purge_sessions=1,
        embargo_sessions=1,
        allow_sealed_test=True,
    )
    with pytest.raises(SealedTestViolationError):
        harness.prepare_splits(sessions, years, split_cfg=unsealed_cfg)
    with pytest.raises(SealedTestViolationError):
        LongRunningTitansHarness(
            HarnessConfig(
                run_dir=str(tmp_path / "seal_init"),
                variation="v13_mmarrs_titan_full",
                split_mode="full_history_final",
                hidden_dim=16,
                device="cpu",
            )
        ).initialize(sample_session=sessions[0], split_cfg=unsealed_cfg)

    # 3) Partición sellada válida (horizonte 1d -> purge>=1, embargo>=5) retiene 2024+
    cfg_h1 = HarnessConfig(
        run_dir=str(tmp_path / "seal_check_h1"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        sealed_test_start_year=2024,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h_h1 = LongRunningTitansHarness(cfg_h1)
    valid_split = h_h1.build_split_config(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=5,
    )
    parts = h_h1.prepare_splits(sessions, years, split_cfg=valid_split)
    assert len(parts["sealed_test"]) == 0
    assert parts["n_sealed_withheld"] == 6  # 3 de 2024 + 3 de 2025
    assert all(y < 2024 for y in parts["train_years"])
    assert all(y < 2024 for y in parts["val_years"])

    # 3b) Con horizonte 63 activo, purge=5 es RECHAZADO (P0-4 anti-fuga multi-horizonte)
    assert required_purge_embargo((1, 2, 5, 10, 21, 63)) == (63, 13)
    assert required_purge_embargo((1,)) == (1, 5)
    with pytest.raises(ValueError, match="purge_sessions"):
        harness.build_split_config(
            train_start_year=2000,
            train_end_year=2022,
            val_year=2023,
            purge_sessions=5,
            embargo_sessions=13,
        )
    with pytest.raises(ValueError, match="embargo_sessions"):
        harness.build_split_config(
            train_start_year=2000,
            train_end_year=2022,
            val_year=2023,
            purge_sessions=63,
            embargo_sessions=5,
        )

    # 4) Modo expanding_walk_forward: construye historia acumulada 2000..T-1 y rechaza T >= 2024
    wf_harness = LongRunningTitansHarness(
        HarnessConfig(
            run_dir=str(tmp_path / "wf_seal"),
            variation="v07_moe_double_loop",
            split_mode="expanding_walk_forward",
            sealed_test_start_year=2024,
            hidden_dim=16,
            device="cpu",
        )
    )
    wf_splits = wf_harness.build_walk_forward_splits(
        eval_years=[2008, 2020, 2022, 2023], train_start_year=2000
    )
    assert all(s.purge_sessions == 63 and s.embargo_sessions == 13 for s in wf_splits)
    assert [(s.train_start_year, s.train_end_year, s.val_year) for s in wf_splits] == [
        (2000, 2007, 2008),
        (2000, 2019, 2020),
        (2000, 2021, 2022),
        (2000, 2022, 2023),
    ]
    assert all(not s.allow_sealed_test for s in wf_splits)

    with pytest.raises(SealedTestViolationError):
        wf_harness.build_walk_forward_splits(eval_years=[2022, 2024], train_start_year=2000)


def _make_six_dim_report(
    *,
    rank_ic: float = 0.045,
    pure_alpha_ic: float = 0.038,
    rank_ic_t_nw: float = 3.2,
    crps: float = 0.0110,
    cov_80: float = 0.80,
    net_sharpe_1x: float = 1.60,
    net_sharpe_2x: float = 1.15,
    dsr: float = 0.85,
) -> dict[str, dict[str, float]]:
    """Construye un informe granular de 6 dimensiones compatible con `granular_evaluator`."""
    return dict(
        dim1_value=dict(mae_bps=85.0, rmse_bps=120.0, oos_r2=0.02),
        dim2_ranking=dict(
            rank_ic_mean=rank_ic,
            pure_alpha_ic_mean=pure_alpha_ic,
            rank_ic_t_nw=rank_ic_t_nw,
        ),
        dim3_trend=dict(sign_accuracy=0.53),
        dim4_uncertainty=dict(crps=crps, cov_80=cov_80),
        dim5_error_autopsy=dict(factor_var_share=0.25),
        dim6_economics=dict(
            net_sharpe_1x=net_sharpe_1x,
            net_sharpe_2x=net_sharpe_2x,
            dsr=dsr,
        ),
    )


def test_stage1_pretraining_vs_stage2_posttraining_separation_and_promotion(tmp_path: Path):
    """El manifiesto separa explícitamente Stage 1 (pre-training) y Stage 2 (post-training)."""
    assert PIPELINE_STAGES == ("stage1_pretraining", "stage2_posttraining")
    years = [2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=41)

    cfg_s1 = HarnessConfig(
        run_dir=str(tmp_path / "stage1_run"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        pipeline_stage="stage1_pretraining",
        hidden_dim=16,
        device="cpu",
    )
    h1 = LongRunningTitansHarness(cfg_s1)
    h1.initialize(sample_session=sessions[0])

    manifest = json.loads((tmp_path / "stage1_run" / "manifest.json").read_text(encoding="utf-8"))
    stages = manifest["pipeline_stages"]
    assert stages["active_stage"] == "stage1_pretraining"
    assert stages["stage1_pretraining"]["updates_model_weights"] is True
    assert stages["stage1_pretraining"]["post_training_layers_active"] is False
    assert stages["stage2_posttraining"]["updates_model_weights"] is False
    assert stages["stage2_posttraining"]["requires_stage1_gain_pass"] is True
    assert stages["stage2_posttraining"]["status"] == "locked_pending_stage1"
    assert set(stages["stage2_posttraining"]["post_training_layers"]) >= {
        "calibration",
        "convex_portfolio_p_plus",
        "metacognitive_a5_a8",
    }

    # Si Stage 1 es RECHAZADO por GainValidationGate, Stage 2 permanece bloqueado
    controls = {
        k: _make_six_dim_report(
            rank_ic=0.040,
            pure_alpha_ic=0.032,
            rank_ic_t_nw=2.5,
            crps=0.0120,
            cov_80=0.80,
            net_sharpe_1x=1.20,
            net_sharpe_2x=0.80,
            dsr=0.70,
        )
        for k in DEFAULT_MATCHED_CONTROLS
    }
    weak_candidate = _make_six_dim_report(
        rank_ic=0.035,
        pure_alpha_ic=0.025,
        rank_ic_t_nw=1.4,
        crps=0.0135,
        cov_80=0.65,
        net_sharpe_1x=0.90,
        net_sharpe_2x=0.40,
        dsr=0.40,
    )
    rej = h1.validate_and_record_gain(weak_candidate, controls)
    assert rej.verdict == "REJECT"
    assert rej.promoted is False
    m_after_rej = json.loads(
        (tmp_path / "stage1_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert m_after_rej["pipeline_stages"]["stage2_posttraining"]["status"] == (
        "blocked_gain_rejected"
    )

    # Cuando Stage 1 APRUEBA contra todos los controles, Stage 2 queda desbloqueado
    strong_candidate = _make_six_dim_report(
        rank_ic=0.055,
        pure_alpha_ic=0.046,
        rank_ic_t_nw=3.6,
        crps=0.0102,
        cov_80=0.81,
        net_sharpe_1x=1.85,
        net_sharpe_2x=1.35,
        dsr=0.92,
    )
    ok = h1.validate_and_record_gain(strong_candidate, controls)
    assert ok.verdict == "PASS"
    assert ok.promoted is True
    m_after_ok = json.loads(
        (tmp_path / "stage1_run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert m_after_ok["pipeline_stages"]["stage2_posttraining"]["status"] == "unlocked_ready"

    # En Stage 2 (post-training), intentar entrenar pesos del tronco (update_weights=True) falla
    cfg_s2 = HarnessConfig(
        run_dir=str(tmp_path / "stage2_run"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        pipeline_stage="stage2_posttraining",
        hidden_dim=16,
        device="cpu",
    )
    h2 = LongRunningTitansHarness(cfg_s2)
    h2.initialize(sample_session=sessions[0])
    with pytest.raises(ValueError, match="Stage 2"):
        h2.run_incremental_sessions(sessions[:2], years[:2], update_weights=True)
    # Con update_weights=False (inferencia/calibración congelada), Stage 2 corre limpiamente
    s2_out = h2.run_incremental_sessions(sessions[:2], years[:2], update_weights=False)
    assert s2_out["sessions_processed"] == 2


def test_gain_validation_gate_enforces_matched_controls_and_six_dim_verdict():
    """`GainValidationGate` exige superar controles emparejados en las 6 dimensiones."""
    gate = GainValidationGate(
        required_controls=tuple(DEFAULT_MATCHED_CONTROLS.keys()),
        min_rank_ic_gain=0.002,
        min_pure_alpha_ic_gain=0.002,
        max_crps_regression=0.0,
        max_cov80_error=0.06,
        min_net_sharpe_1x_gain=0.10,
        min_net_sharpe_2x_gain=0.05,
        min_dsr=0.60,
        min_newey_west_t=2.0,
    )
    controls = {
        "dense_iso_flop": _make_six_dim_report(
            rank_ic=0.030, pure_alpha_ic=0.022, crps=0.0125, net_sharpe_1x=1.00, net_sharpe_2x=0.65
        ),
        "dense_iso_param": _make_six_dim_report(
            rank_ic=0.034, pure_alpha_ic=0.026, crps=0.0120, net_sharpe_1x=1.15, net_sharpe_2x=0.75
        ),
        "uniform_1_over_e": _make_six_dim_report(
            rank_ic=0.032, pure_alpha_ic=0.024, crps=0.0122, net_sharpe_1x=1.08, net_sharpe_2x=0.70
        ),
        "shuffled_placebo": _make_six_dim_report(
            rank_ic=0.020, pure_alpha_ic=0.015, crps=0.0135, net_sharpe_1x=0.70, net_sharpe_2x=0.40
        ),
        "k1_fixed_depth": _make_six_dim_report(
            rank_ic=0.036, pure_alpha_ic=0.028, crps=0.0118, net_sharpe_1x=1.20, net_sharpe_2x=0.80
        ),
    }

    # 1) Candidato superior en todas las dimensiones y frente a los 5 controles -> PASS
    winner = _make_six_dim_report(
        rank_ic=0.048,
        pure_alpha_ic=0.040,
        rank_ic_t_nw=3.1,
        crps=0.0108,
        cov_80=0.81,
        net_sharpe_1x=1.55,
        net_sharpe_2x=1.10,
        dsr=0.88,
    )
    v_pass = gate.evaluate("v13_mmarrs_titan_full", winner, controls)
    assert isinstance(v_pass, GainVerdict)
    assert v_pass.verdict == "PASS"
    assert v_pass.promoted is True
    assert v_pass.failed_criteria == ()
    assert set(v_pass.controls_evaluated) == set(controls.keys())
    assert v_pass.deltas_vs_controls["k1_fixed_depth"]["delta_rank_ic"] == pytest.approx(0.012)
    assert v_pass.deltas_vs_controls["dense_iso_param"]["crps_improvement"] == pytest.approx(0.0012)

    # 2) Candidato que empeora Pure Alpha IC frente a `k1_fixed_depth` -> REJECT
    bad_alpha = _make_six_dim_report(
        rank_ic=0.048,
        pure_alpha_ic=0.027,  # < 0.028 + 0.002
        rank_ic_t_nw=3.1,
        crps=0.0108,
        cov_80=0.80,
        net_sharpe_1x=1.55,
        net_sharpe_2x=1.10,
        dsr=0.88,
    )
    v_alpha = gate.evaluate("v09_holographic_3l_moe_loop", bad_alpha, controls)
    assert v_alpha.verdict == "REJECT"
    assert v_alpha.promoted is False
    assert any("pure_alpha_ic" in c for c in v_alpha.failed_criteria)

    # 3) Candidato con mala calibración Cov80 o regresión CRPS o DSR bajo o t_NW bajo -> REJECT
    bad_cal = _make_six_dim_report(
        rank_ic=0.048,
        pure_alpha_ic=0.040,
        rank_ic_t_nw=1.5,  # < 2.0
        crps=0.0140,  # peor que todos los controles
        cov_80=0.62,  # fuera de [0.74, 0.86]
        net_sharpe_1x=1.55,
        net_sharpe_2x=1.10,
        dsr=0.45,  # < 0.60
    )
    v_cal = gate.evaluate("v07_moe_double_loop", bad_cal, controls)
    assert v_cal.verdict == "REJECT"
    assert any("cov_80" in c for c in v_cal.failed_criteria)
    assert any("crps" in c for c in v_cal.failed_criteria)
    assert any("dsr" in c for c in v_cal.failed_criteria)
    assert any("rank_ic_t_nw" in c for c in v_cal.failed_criteria)

    # 4) Sin controles emparejados -> REJECT categórico (jamás promover sin controles!)
    v_no_ctrl = gate.evaluate("v13_mmarrs_titan_full", winner, {})
    assert v_no_ctrl.verdict == "REJECT"
    assert v_no_ctrl.promoted is False

    # 4b) Por defecto GainValidationGate() exige los 5 controles emparejados (rechaza subconjunto)
    default_gate = GainValidationGate()
    assert default_gate.required_controls == tuple(DEFAULT_MATCHED_CONTROLS.keys())
    v_weak_only = default_gate.evaluate(
        "v13_mmarrs_titan_full", winner, {"shuffled_placebo": controls["shuffled_placebo"]}
    )
    assert v_weak_only.verdict == "REJECT"
    assert v_weak_only.promoted is False
    assert any("missing_required_controls" in c for c in v_weak_only.failed_criteria)

    partial_4 = {k: v for k, v in controls.items() if k != "k1_fixed_depth"}
    v_missing_k1 = default_gate.evaluate("v13_mmarrs_titan_full", winner, partial_4)
    assert v_missing_k1.verdict == "REJECT"
    assert any("k1_fixed_depth" in c for c in v_missing_k1.failed_criteria)

    # 5) Evaluación en lote contra variaciones de control estándar
    batch_reports = {
        "v00_dense_baseline": controls["dense_iso_flop"],
        "v01_dense_wide_iso_param": controls["dense_iso_param"],
        "v03_moe_only_k1": controls["k1_fixed_depth"],
        "v04_moe_uniform_control": controls["uniform_1_over_e"],
        "v05_moe_shuffled_ctx_placebo": controls["shuffled_placebo"],
        "v13_mmarrs_titan_full": winner,
        "v07_moe_double_loop": bad_alpha,
    }
    batch_verdicts = gate.evaluate_batch(batch_reports)
    assert batch_verdicts["v13_mmarrs_titan_full"].verdict == "PASS"
    assert batch_verdicts["v07_moe_double_loop"].verdict == "REJECT"


def test_varying_universe_nan_batch_and_disabled_auto_recover(tmp_path: Path):
    """Soporta universo cambiante `N_t`, rollback ante lote NaN y `auto_recover_on_nan=False`."""
    years = [2020, 2021, 2022, 2023]
    # Universo variable entre sesiones (simula altas/bajas 2000-2023)
    sessions = [
        _make_synthetic_sessions([yr], n_assets=n_a, seed=100 + i)[0]
        for i, (yr, n_a) in enumerate(zip(years, (10, 14, 12, 16), strict=True))
    ]

    cfg = HarnessConfig(
        run_dir=str(tmp_path / "var_univ"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=2,
        compaction_interval=2,
        auto_recover_on_nan=True,
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=sessions[0])
    out_ok = h.run_incremental_sessions(sessions[:2], years[:2], total_planned_steps=4)
    assert out_ok["sessions_processed"] == 2
    assert out_ok["compactions"] == 1

    # Inyectar NaN directamente en los datos de entrada de la sesión 2
    poisoned = [dict(s) for s in sessions]
    poisoned[2]["prices"] = poisoned[2]["prices"].clone()
    poisoned[2]["prices"][0, 0] = float("nan")
    out_poison = h.run_incremental_sessions(poisoned, years, total_planned_steps=4)
    assert out_poison["rollbacks"] == 1
    assert out_poison["last_health_report"].passed is False
    assert out_poison["last_health_report"].rollback_triggered is True
    assert h.session_cursor == 2
    assert all(torch.isfinite(p).all() for p in h.model.parameters())

    # Con auto_recover_on_nan=False no dispara rollback automático
    cfg_no_rec = HarnessConfig(
        run_dir=str(tmp_path / "no_recover"),
        variation="v00_dense_baseline",
        split_mode="full_history_final",
        auto_recover_on_nan=False,
        hidden_dim=16,
        device="cpu",
    )
    h_no = LongRunningTitansHarness(cfg_no_rec)
    h_no.initialize(sample_session=sessions[0])
    out_no = h_no.run_incremental_sessions(poisoned[2:3], [2022])
    assert out_no["rollbacks"] == 0
    assert out_no["last_health_report"].passed is False
    assert out_no["last_health_report"].rollback_triggered is False


def test_post_step_matured_corruption_rejected_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Un `step_matured` que viole la cota PT1 o produzca NaN revierte antes de guardar."""
    years = [2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=88)

    cfg = HarnessConfig(
        run_dir=str(tmp_path / "post_matured_guard"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=2,
        health_check_every_k=1,
        max_memory_norm=25.0,
        auto_recover_on_nan=True,
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=sessions[0])
    out_ok = h.run_incremental_sessions(sessions, years, max_sessions=2, total_planned_steps=4)
    assert out_ok["sessions_processed"] == 2
    assert h.session_cursor == 2

    # Inyectar violación PT1 en `step_matured` de la sesión 2 (después del forward sano)
    orig_step_matured = h.model.step_matured

    def _corrupt_step_matured(state, **kwargs):
        res = orig_step_matured(state, **kwargs)
        bad_titans = dict(res["titans_state"])
        bad_titans["fast"] = torch.full_like(bad_titans["fast"], 30.0)  # norm = 120 > 25
        res["titans_state"] = bad_titans
        return res

    monkeypatch.setattr(h.model, "step_matured", _corrupt_step_matured)
    out_bad = h.run_incremental_sessions(sessions, years, max_sessions=2, total_planned_steps=4)
    assert out_bad["rollbacks"] == 1
    assert out_bad["last_health_report"].passed is False
    assert out_bad["last_health_report"].rollback_triggered is True
    assert any("pt1_post_matured" in f for f in out_bad["last_health_report"].hard_failures)
    assert h.session_cursor == 2

    # El checkpoint en disco permanece en el cursor 2 sano y dentro de la cota PT1
    ckpt = torch.load(h.latest_checkpoint_path, weights_only=False)
    assert ckpt["session_cursor"] == 2
    for bank_t in ckpt["causal_state"]["titans_state"].values():
        assert torch.isfinite(bank_t).all()
        assert float(bank_t.float().norm(dim=-1).max().item()) <= 25.0


def test_p0_5_gain_gate_strict_dsr_champion_pbo_margin_liquid_crisis_and_survivorship():
    """P0-5: DSR>=0.90 (0.95 campeón), PBO<=0.50, margen>=2.5x, tercio líquido y crisis."""
    controls = {
        k: _make_six_dim_report(
            rank_ic=0.030,
            pure_alpha_ic=0.022,
            rank_ic_t_nw=2.2,
            crps=0.0120,
            cov_80=0.80,
            net_sharpe_1x=1.00,
            net_sharpe_2x=0.60,
            dsr=0.70,
        )
        for k in DEFAULT_MATCHED_CONTROLS
    }

    def _cand(
        *,
        dsr: float = 0.93,
        pbo: float = 0.20,
        safety_margin: float = 3.2,
        liq_ic: float = 0.025,
        crisis_2008_ic: float = 0.015,
    ) -> dict:
        rep = _make_six_dim_report(
            rank_ic=0.050,
            pure_alpha_ic=0.042,
            rank_ic_t_nw=3.4,
            crps=0.0105,
            cov_80=0.81,
            net_sharpe_1x=1.65,
            net_sharpe_2x=1.20,
            dsr=dsr,
        )
        rep["dim6_economics"]["pbo"] = pbo
        rep["dim6_economics"]["safety_margin"] = safety_margin
        rep["liquid_third"] = dict(rank_ic_mean=liq_ic, net_sharpe_1x=1.10)
        rep["crisis_metrics"] = {
            2008: dict(rank_ic_mean=crisis_2008_ic, net_sharpe_1x=0.40),
            2020: dict(rank_ic_mean=0.020, net_sharpe_1x=0.50),
            2022: dict(rank_ic_mean=0.018, net_sharpe_1x=0.30),
        }
        return rep

    default_gate = GainValidationGate()
    assert default_gate.min_dsr == pytest.approx(0.90)
    assert default_gate.champion_min_dsr == pytest.approx(0.95)
    assert default_gate.max_pbo == pytest.approx(0.50)
    assert default_gate.min_safety_margin == pytest.approx(2.5)

    # 1) DSR=0.85 rechazado por puerta por defecto (min_dsr=0.90)
    v_low_dsr = default_gate.evaluate("v13_mmarrs_titan_full", _cand(dsr=0.85), controls)
    assert v_low_dsr.verdict == "REJECT"
    assert v_low_dsr.survivorship_disclosure is True
    assert any("dsr" in c for c in v_low_dsr.failed_criteria)

    # 2) DSR=0.93 aprueba puerta estándar, pero falla en modo campeón (min_dsr=0.95)
    v_std = default_gate.evaluate("v13_mmarrs_titan_full", _cand(dsr=0.93), controls)
    assert v_std.verdict == "PASS"
    assert v_std.promoted is True
    assert v_std.survivorship_disclosure is True
    assert v_std.to_dict()["survivorship_disclosure"] is True

    champ_gate = GainValidationGate(champion_mode=True)
    v_champ_fail = champ_gate.evaluate("v13_mmarrs_titan_full", _cand(dsr=0.93), controls)
    assert v_champ_fail.verdict == "REJECT"
    v_champ_ok = champ_gate.evaluate("v13_mmarrs_titan_full", _cand(dsr=0.96), controls)
    assert v_champ_ok.verdict == "PASS"

    # 3) PBO > 0.50, safety_margin < 2.5, tercio líquido negativo y colapso crisis 2008 -> REJECT
    v_bad_pbo = default_gate.evaluate("v13_mmarrs_titan_full", _cand(pbo=0.62), controls)
    assert v_bad_pbo.verdict == "REJECT" and any("pbo" in c for c in v_bad_pbo.failed_criteria)

    v_bad_margin = default_gate.evaluate(
        "v13_mmarrs_titan_full", _cand(safety_margin=1.9), controls
    )
    assert v_bad_margin.verdict == "REJECT" and any(
        "safety_margin" in c for c in v_bad_margin.failed_criteria
    )

    v_bad_liq = default_gate.evaluate("v13_mmarrs_titan_full", _cand(liq_ic=-0.01), controls)
    assert v_bad_liq.verdict == "REJECT" and any(
        "liquid_third" in c for c in v_bad_liq.failed_criteria
    )

    v_bad_crisis = default_gate.evaluate(
        "v13_mmarrs_titan_full", _cand(crisis_2008_ic=-0.03), controls
    )
    assert v_bad_crisis.verdict == "REJECT" and any(
        "crisis_2008" in c for c in v_bad_crisis.failed_criteria
    )

    # 4) Regresión #73: informes anidados de 6 dimensiones (formato real de `granular_evaluator`)
    # para `liquid_third` y `crisis_metrics` NO pueden saltarse la puerta silenciosamente
    nested_bad_liq = _cand()
    nested_bad_liq["liquid_third"] = _make_six_dim_report(
        rank_ic=-0.2077,
        pure_alpha_ic=-0.18,
        net_sharpe_1x=-16.14,
        net_sharpe_2x=-18.0,
        dsr=0.05,
    )
    v_nested_liq = default_gate.evaluate("v13_mmarrs_titan_full", nested_bad_liq, controls)
    assert v_nested_liq.verdict == "REJECT"
    assert v_nested_liq.checks.get("liquid_third_rank_ic") is False
    assert v_nested_liq.checks.get("liquid_third_net_sharpe_1x") is False

    nested_bad_crisis = _cand()
    nested_bad_crisis["crisis_metrics"] = {
        2008: _make_six_dim_report(rank_ic=-0.04, net_sharpe_1x=-1.20),
        2020: _make_six_dim_report(rank_ic=0.02, net_sharpe_1x=0.45),
        2022: _make_six_dim_report(rank_ic=0.015, net_sharpe_1x=0.25),
    }
    v_nested_crisis = default_gate.evaluate("v13_mmarrs_titan_full", nested_bad_crisis, controls)
    assert v_nested_crisis.verdict == "REJECT"
    assert v_nested_crisis.checks.get("crisis_2008_rank_ic") is False
    assert v_nested_crisis.checks.get("crisis_2008_net_sharpe_1x") is False

    nested_ok = _cand()
    nested_ok["liquid_third"] = _make_six_dim_report(rank_ic=0.030, net_sharpe_1x=1.15)
    nested_ok["crisis_metrics"] = {
        2008: _make_six_dim_report(rank_ic=0.018, net_sharpe_1x=0.35),
        2020: _make_six_dim_report(rank_ic=0.022, net_sharpe_1x=0.50),
        2022: _make_six_dim_report(rank_ic=0.015, net_sharpe_1x=0.25),
    }
    v_nested_ok = default_gate.evaluate("v13_mmarrs_titan_full", nested_ok, controls)
    assert v_nested_ok.verdict == "PASS"
    assert v_nested_ok.checks.get("liquid_third_rank_ic") is True
    assert v_nested_ok.checks.get("liquid_third_net_sharpe_1x") is True
    assert v_nested_ok.checks.get("crisis_2008_rank_ic") is True
    assert v_nested_ok.checks.get("crisis_2008_net_sharpe_1x") is True

    # 5) Fallback `linear_cost_bps + impact_cost_bps` si `safety_margin` no viene explícito
    cost_fallback = _cand()
    del cost_fallback["dim6_economics"]["safety_margin"]
    cost_fallback["dim6_economics"]["gross_mean_bps"] = 6.0
    cost_fallback["dim6_economics"]["linear_cost_bps"] = 2.0
    cost_fallback["dim6_economics"]["impact_cost_bps"] = 2.0  # margin = 6/4 = 1.5 < 2.5
    v_cost_fb = default_gate.evaluate("v13_mmarrs_titan_full", cost_fallback, controls)
    assert v_cost_fb.verdict == "REJECT"
    assert v_cost_fb.checks.get("safety_margin") is False
    assert v_cost_fb.candidate_metrics["safety_margin"] == pytest.approx(1.5)


def test_p0_1_p0_7_p2_14_and_48_asset_ids_throttle_store_cuda_rng_and_gzip_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verifica `asset_ids`, throttling K, `ZeroCopySessionStore`, CUDA RNG y ledger `.jsonl.gz`."""
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    # Ragged sessions con asset_ids explícitos
    sessions = []
    for i, (yr, n_a) in enumerate(zip(years, (10, 12, 11, 13, 12, 14), strict=True)):
        b = _make_synthetic_sessions([yr], n_assets=n_a, seed=200 + i)[0]
        b["asset_ids"] = torch.arange(i, i + n_a, dtype=torch.int64)
        sessions.append(b)

    # 1) Throttling de SessionHealthGate (`health_check_every_k=5`) + propagación de `asset_ids`
    cfg = HarnessConfig(
        run_dir=str(tmp_path / "throttled_gz"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=10,
        compaction_interval=10,
        health_check_every_k=5,
        compress_ledger=True,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=sessions[0])

    verify_calls = {"count": 0}
    orig_verify = h.health_gate.verify_session

    def _spy_verify(*args, **kwargs):
        verify_calls["count"] += 1
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr(h.health_gate, "verify_session", _spy_verify)
    seen_asset_ids: list[torch.Tensor] = []
    orig_step_matured = h.model.step_matured

    def _spy_step_matured(state, *, matured_batch, **kwargs):
        assert "asset_ids" in matured_batch
        seen_asset_ids.append(matured_batch["asset_ids"].detach().clone())
        return orig_step_matured(state, matured_batch=matured_batch, **kwargs)

    monkeypatch.setattr(h.model, "step_matured", _spy_step_matured)

    res = h.run_incremental_sessions(sessions[:5], years[:5], total_planned_steps=6)
    assert res["sessions_processed"] == 5
    # Con health_check_every_k=5 y tramo de 5, verify_session corre SOLO 1 vez (en el paso 5)
    assert verify_calls["count"] == 1
    assert len(seen_asset_ids) == 5
    assert "last_asset_ids" in h.causal_state
    torch.testing.assert_close(h.causal_state["last_asset_ids"], sessions[4]["asset_ids"])

    # 2) Ledger comprimido `.jsonl.gz`: válido, sin pérdidas y legible vía `read_progress_ledger`
    assert h.ledger_path.suffix == ".gz"
    gz_entries = read_progress_ledger(h.ledger_path)
    assert len([e for e in gz_entries if e["event"] == "session_completed"]) == 5
    # Lectura transparente si se consulta con nombre base `.jsonl`
    assert read_progress_ledger(h.run_dir / "progress_ledger.jsonl") == gz_entries

    # Comparar tamaño en disco frente a ledger sin comprimir
    cfg_plain = HarnessConfig(
        run_dir=str(tmp_path / "plain_jsonl"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=10,
        health_check_every_k=5,
        compress_ledger=False,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h_plain = LongRunningTitansHarness(cfg_plain)
    h_plain.initialize(sample_session=sessions[0])
    h_plain.run_incremental_sessions(sessions[:5], years[:5], total_planned_steps=6)
    assert h_plain.ledger_path.name == "progress_ledger.jsonl"
    assert h.ledger_path.stat().st_size < h_plain.ledger_path.stat().st_size

    # 3) Integración directa con `ZeroCopySessionStore.iter_pinned_h2d` y `ZeroCopySessionView`
    store_dir = tmp_path / "zc_store"
    store_years = [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2023, 2023, 2023, 2023, 2024]
    raw_store_sessions = _make_synthetic_sessions(store_years, n_assets=12, seed=91)
    store_sessions = [
        {k: v for k, v in s.items() if v.shape[0] == 12} for s in raw_store_sessions
    ]
    store = ZeroCopySessionStore.from_sessions(store_dir, store_sessions, store_years)
    iter_spy = {"called": 0}
    orig_iter = store.iter_pinned_h2d

    def _spy_iter(*args, **kwargs):
        iter_spy["called"] += 1
        return orig_iter(*args, **kwargs)

    monkeypatch.setattr(store, "iter_pinned_h2d", _spy_iter)
    h_store = LongRunningTitansHarness(
        HarnessConfig(
            run_dir=str(tmp_path / "store_run"),
            variation="v00_dense_baseline",
            split_mode="full_history_final",
            active_horizons=(1,),
            hidden_dim=16,
            device="cpu",
        )
    )
    store_split = h_store.build_split_config(
        train_start_year=2020,
        train_end_year=2022,
        val_year=2023,
        purge_sessions=1,
        embargo_sessions=5,
    )
    zc_parts = h_store.prepare_splits(store, split_cfg=store_split)
    assert isinstance(zc_parts["train"], ZeroCopySessionView)
    assert zc_parts["n_sealed_withheld"] == 1
    out_store = h_store.run_incremental_sessions(store, split_cfg=store_split)
    assert out_store["sessions_processed"] == 5  # 6 train - 1 purged
    assert iter_spy["called"] == 1

    # 3b) Alineación vectorial de `w_prev` por `asset_ids` (altas, bajas y permutaciones)
    prev_ids = torch.tensor([30, 10, 20], dtype=torch.int64)
    prev_w = torch.tensor([0.3, 0.1, 0.2], dtype=torch.float32)
    cur_ids = torch.tensor([10, 99, 30, 5], dtype=torch.int64)
    aligned = _align_w_prev_by_asset_ids(cur_ids, prev_ids, prev_w)
    assert aligned is not None
    torch.testing.assert_close(aligned, torch.tensor([0.1, 0.0, 0.3, 0.0], dtype=torch.float32))

    # 4) P2-14: Guardado y restauración de `torch.cuda.get_rng_state_all()` (mock sin tocar GPU)
    fake_cuda_rng = [torch.tensor([1, 2, 3, 4], dtype=torch.uint8)]
    restored_rng: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_cuda_rng)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda states: restored_rng.append(list(states))
    )
    ckpt_p = h_store.save_checkpoint(milestone=False, log_event=False)
    ckpt_data = torch.load(ckpt_p, weights_only=False)
    assert ckpt_data["cuda_rng_state"] is not None
    h_store.load_checkpoint(ckpt_p)
    assert len(restored_rng) == 1
    torch.testing.assert_close(restored_rng[0][0], fake_cuda_rng[0])

    # 5) Diagnósticos 0D GPU/tensor en `SessionHealthGate.verify_session` en 1 sola sync D2H
    sync_calls = {"count": 0}
    orig_pull = lrh._batch_scalar_tensors_to_floats

    def _spy_pull(tensors):
        sync_calls["count"] += 1
        return orig_pull(tensors)

    monkeypatch.setattr(lrh, "_batch_scalar_tensors_to_floats", _spy_pull)
    gate = SessionHealthGate()
    rep_t = gate.verify_session(
        model=h_store.model,
        outputs=dict(
            pred_return=torch.zeros(4, 6),
            quantiles=torch.linspace(-0.1, 0.1, 9).view(1, 1, 9).expand(4, 6, 9),
            diagnostics=dict(
                load_max_min_ratio=torch.tensor(1.4),
                ood_energy_mean=torch.tensor(0.8),
            ),
        ),
        targets=torch.zeros(4, 6),
        loss=torch.tensor(0.25),
    )
    assert rep_t.passed is True
    assert sync_calls["count"] == 1
    assert rep_t.metrics["moe_load_max_min_ratio"] == pytest.approx(1.4)
    assert rep_t.metrics["ood_energy_mean"] == pytest.approx(0.8)


def test_p1_4_unchecked_nan_step_leaves_weights_opt_ema_finite_and_single_sync_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """P1-4: NaN en paso no muestreado no corrompe pesos/opt/EMA y paso checked hace 1 sola sync."""
    years = [2020, 2021, 2022, 2023]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=404)

    # 1) NaN en paso 1 (no muestreado con `health_check_every_k=10` y `checkpoint_every=10`)
    cfg = HarnessConfig(
        run_dir=str(tmp_path / "nan_unchecked_guard"),
        variation="v07_moe_double_loop",
        split_mode="full_history_final",
        checkpoint_every_sessions=10,
        compaction_interval=10,
        health_check_every_k=10,
        auto_recover_on_nan=False,  # SIN rollback: los pesos deben quedar intactos por sí solos!
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=sessions[0])
    # Paso 0 sano: poblar estado AdamW y EMA
    out0 = h.run_incremental_sessions(sessions[:1], years[:1], total_planned_steps=4)
    assert out0["sessions_processed"] == 1
    w_snap_before = [p.detach().clone() for p in h.model.parameters()]
    ema_snap_before = [p.detach().clone() for p in h.ema_model.parameters()]

    # Inyectar NaN en sesión 1 (no último del tramo ni múltiplo de 10 -> unchecked)
    poisoned = [dict(s) for s in sessions]
    poisoned[1]["prices"] = poisoned[1]["prices"].clone()
    poisoned[1]["prices"][0, 0] = float("nan")

    # Espiar inmediatamente después de sesión 1 antes de que se ejecute sesión 2
    after_s1_ok = {"checked": False}
    orig_step_matured = h.model.step_matured

    def _inspect_after_s1(state, *, matured_batch, **kwargs):
        res = orig_step_matured(state, matured_batch=matured_batch, **kwargs)
        return res

    # Ejecutar sesiones 1..3: sesión 1 es NaN (unchecked), sesiones 2 y 3 son sanas
    observed_w_after_s1: list[torch.Tensor] = []
    observed_ema_after_s1: list[torch.Tensor] = []
    observed_opt_finite_s1: list[bool] = []
    orig_fwd = h.model.forward_session
    call_idx = {"n": 0}

    def _spy_fwd(batch, **kwargs):
        if call_idx["n"] == 1:
            # Justo al entrar a sesión 2 -> inspeccionar estado tras paso 1 (NaN unchecked)!
            observed_w_after_s1.extend(p.detach().clone() for p in h.model.parameters())
            observed_ema_after_s1.extend(p.detach().clone() for p in h.ema_model.parameters())
            for st in h.optimizer.state.values():
                for v in st.values():
                    if isinstance(v, torch.Tensor):
                        observed_opt_finite_s1.append(bool(torch.isfinite(v).all().item()))
            after_s1_ok["checked"] = True
        call_idx["n"] += 1
        return orig_fwd(batch, **kwargs)

    monkeypatch.setattr(h.model, "forward_session", _spy_fwd)
    monkeypatch.setattr(h.model, "step_matured", _inspect_after_s1)

    h.run_incremental_sessions(poisoned, years, total_planned_steps=4)
    assert after_s1_ok["checked"] is True
    assert len(observed_w_after_s1) == len(w_snap_before)
    assert all(torch.isfinite(w).all() for w in observed_w_after_s1)
    assert all(torch.isfinite(w).all() for w in observed_ema_after_s1)
    assert observed_opt_finite_s1 and all(observed_opt_finite_s1)
    # Ni pesos ni EMA se alteraron en el paso NaN no muestreado!
    for w_a, w_b in zip(observed_w_after_s1, w_snap_before, strict=True):
        torch.testing.assert_close(w_a, w_b)
    for e_a, e_b in zip(observed_ema_after_s1, ema_snap_before, strict=True):
        torch.testing.assert_close(e_a, e_b)

    # 2) En paso muestreado (`should_check_health=True`), consolidar en 1 sola sync D2H
    cfg_sync = HarnessConfig(
        run_dir=str(tmp_path / "single_sync_checked"),
        variation="v07_moe_double_loop",
        split_mode="full_history_final",
        checkpoint_every_sessions=10,
        compaction_interval=10,
        health_check_every_k=1,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h_sync = LongRunningTitansHarness(cfg_sync)
    h_sync.initialize(sample_session=sessions[0])
    pull_calls = {"count": 0}
    orig_pull = lrh._batch_scalar_tensors_to_floats

    def _counting_pull(tensors):
        pull_calls["count"] += 1
        return orig_pull(tensors)

    monkeypatch.setattr(lrh, "_batch_scalar_tensors_to_floats", _counting_pull)
    out_s = h_sync.run_incremental_sessions(sessions[:2], years[:2], total_planned_steps=2)
    assert out_s["sessions_processed"] == 2
    # Exactamente 1 llamada agrupada `_batch_scalar_tensors_to_floats` por paso checked (2 total)
    assert pull_calls["count"] == 2


def test_p1_4_buffered_gzip_ledger_compacts_members_and_flushes_safely(tmp_path: Path):
    """P1-4: buffer JSONL.GZ reduce miembros gzip y tamaño, vaciando en checkpoint/close."""
    years = [2020] * 12
    sessions = _make_synthetic_sessions(years, n_assets=10, seed=505)

    cfg_buf = HarnessConfig(
        run_dir=str(tmp_path / "buffered_gz"),
        variation="v00_dense_baseline",
        split_mode="full_history_final",
        checkpoint_every_sessions=12,
        compaction_interval=20,
        health_check_every_k=12,
        compress_ledger=True,
        ledger_flush_every_k=6,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    cfg_per_line = HarnessConfig(
        run_dir=str(tmp_path / "per_line_gz"),
        variation="v00_dense_baseline",
        split_mode="full_history_final",
        checkpoint_every_sessions=12,
        compaction_interval=20,
        health_check_every_k=12,
        compress_ledger=True,
        ledger_flush_every_k=1,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )

    with LongRunningTitansHarness(cfg_buf) as h_buf:
        h_buf.initialize(sample_session=sessions[0])
        out_buf = h_buf.run_incremental_sessions(sessions, years, total_planned_steps=12)
        assert out_buf["sessions_processed"] == 12

    with LongRunningTitansHarness(cfg_per_line) as h_line:
        h_line.initialize(sample_session=sessions[0])
        h_line.run_incremental_sessions(sessions, years, total_planned_steps=12)

    recs_buf = read_progress_ledger(h_buf.ledger_path)
    recs_line = read_progress_ledger(h_line.ledger_path)
    assert len([r for r in recs_buf if r["event"] == "session_completed"]) == 12
    assert len(recs_buf) == len(recs_line)

    raw_buf = h_buf.ledger_path.read_bytes()
    raw_line = h_line.ledger_path.read_bytes()
    # Mucho menos miembros gzip RFC-1952 y archivo más compacto
    members_buf = raw_buf.count(b"\x1f\x8b\x08")
    members_line = raw_line.count(b"\x1f\x8b\x08")
    assert members_buf <= 5
    assert members_line >= 12
    assert len(raw_buf) < len(raw_line) * 0.70

    with pytest.raises(ValueError):
        HarnessConfig(run_dir=str(tmp_path / "bad_flush"), ledger_flush_every_k=0)


def test_p1_4_cuda_rng_resume_handles_multi_gpu_to_single_gpu_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """P1-4: reanudar checkpoint multi-GPU (4 GPUs) en 1× GPU no lanza error en RNG CUDA."""
    sessions = _make_synthetic_sessions([2021], n_assets=10, seed=606)
    cfg = HarnessConfig(
        run_dir=str(tmp_path / "multi_to_single_gpu"),
        variation="v00_dense_baseline",
        split_mode="full_history_final",
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=sessions[0])

    # Guardar checkpoint simulando 4 GPUs DGX
    fake_4gpu_rng = [
        torch.tensor([i, i + 1, i + 2, i + 3], dtype=torch.uint8) for i in range(4)
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_4gpu_rng)
    ckpt_path = h.save_checkpoint(milestone=False, log_event=False)

    # Reanudar simulando 1× Colab RTX PRO 6000: `set_rng_state_all` fallaría si se llamara!
    per_dev_restored: list[tuple[int, torch.Tensor]] = []

    def _forbid_all(states):
        raise RuntimeError("cuda_rng_state count mismatch (4 != 1)")

    def _capture_single(state: torch.Tensor, device: int = 0):
        per_dev_restored.append((int(device), state.clone()))

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", _forbid_all)
    monkeypatch.setattr(torch.cuda, "set_rng_state", _capture_single)

    h.load_checkpoint(ckpt_path)
    assert len(per_dev_restored) == 1
    assert per_dev_restored[0][0] == 0
    torch.testing.assert_close(per_dev_restored[0][1], fake_4gpu_rng[0])


def test_p1_4_harness_supports_sota_optimizers_muon_hybrid_and_schedule_free(tmp_path: Path):
    """P1-4: `HarnessConfig.optimizer_name` soporta `muon_hybrid` y `schedule_free_adamw`."""
    from mars_titan.models.mmarrs.autotuner_sota import MuonHybridOptimizer, ScheduleFreeAdamW

    with pytest.raises(ValueError, match="optimizer_name"):
        HarnessConfig(run_dir=str(tmp_path / "bad_opt"), optimizer_name="sgd_bogus")

    years = [2021, 2022]
    sessions = _make_synthetic_sessions(years, n_assets=12, seed=707)

    for opt_name, expected_cls in (
        ("muon_hybrid", MuonHybridOptimizer),
        ("schedule_free_adamw", ScheduleFreeAdamW),
    ):
        cfg = HarnessConfig(
            run_dir=str(tmp_path / f"opt_{opt_name}"),
            variation="v07_moe_double_loop",
            split_mode="full_history_final",
            optimizer_name=opt_name,
            checkpoint_every_sessions=2,
            health_check_every_k=1,
            active_horizons=(1,),
            hidden_dim=16,
            device="cpu",
        )
        h = LongRunningTitansHarness(cfg)
        h.initialize(sample_session=sessions[0])
        assert isinstance(h.optimizer, expected_cls)
        out = h.run_incremental_sessions(sessions, years, total_planned_steps=2)
        assert out["sessions_processed"] == 2
        assert all(torch.isfinite(p).all() for p in h.model.parameters())
        # Reanudación desde checkpoint conserva el optimizador SOTA sin errores
        h2 = LongRunningTitansHarness(cfg)
        res_info = h2.initialize(sample_session=sessions[0])
        assert res_info["resumed"] is True
        assert isinstance(h2.optimizer, expected_cls)
        assert h2.session_cursor == 2


def test_p1_4_update_ema_masked_nan_safe_and_syncs_non_episodic_buffers():
    """#109: `_update_ema_masked` inmune a `0.0 * NaN` y sincroniza buffers no episódicos."""
    import copy

    from mars_titan.models.mmarrs.ablation_runner import _is_episodic_buffer
    from mars_titan.models.mmarrs.long_running_harness import _update_ema_masked
    from mars_titan.models.mmarrs.model import build_variation

    sessions = _make_synthetic_sessions([2021], n_assets=8, seed=808)
    s0 = sessions[0]
    model = build_variation(
        "v13_mmarrs_titan_full",
        prices_dim=int(s0["prices"].shape[-1]),
        fundamentals_dim=int(s0["fundamentals"].shape[-1]),
        macro_dim=int(s0["macro"].shape[-1]),
        news_dim=int(s0["news"].shape[-1]),
        charts_dim=int(s0["charts"].shape[-1]),
        physics_dim=int(s0["physics_v4"].shape[-1]),
        context_dim=int(s0["high_snr_context"].shape[-1]),
        exposures_dim=int(s0["exposures"].shape[-1]),
        rmt_dim=int(s0["rmt_loadings"].shape[-1]),
        hidden_dim=16,
    )
    ema_model = copy.deepcopy(model)
    ema_params_before = [p.detach().clone() for p in ema_model.parameters()]
    ema_bufs_before = {k: v.detach().clone() for k, v in ema_model.named_buffers()}

    # 1) `step_finite=False` con NaN en parámetros y buffers no episódicos -> EMA intacta y finita
    nan_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in nan_model.parameters():
            p.fill_(float("nan"))
        for b_name, b in nan_model.named_buffers():
            if not _is_episodic_buffer(b_name) and b.dtype.is_floating_point:
                b.fill_(float("nan"))
    _update_ema_masked(ema_model, nan_model, decay=0.9, step_finite=torch.tensor(False))
    for p_after, p_before in zip(ema_model.parameters(), ema_params_before, strict=True):
        assert torch.isfinite(p_after).all()
        torch.testing.assert_close(p_after, p_before)
    for b_name, b_after in ema_model.named_buffers():
        assert torch.isfinite(b_after).all()
        torch.testing.assert_close(b_after, ema_bufs_before[b_name])

    # 2) `step_finite=True`: sincroniza parámetros (EMA) y buffers no episódicos, omite episódicos
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        non_ep_touched: list[str] = []
        ep_touched: list[str] = []
        for b_name, b in model.named_buffers():
            if not b.dtype.is_floating_point:
                continue
            if _is_episodic_buffer(b_name):
                b.fill_(7.5)
                ep_touched.append(b_name)
            else:
                b.add_(0.25)
                non_ep_touched.append(b_name)
    assert non_ep_touched and ep_touched

    _update_ema_masked(ema_model, model, decay=0.9, step_finite=torch.tensor(True))
    for p_ema, p_src, p_before in zip(
        ema_model.parameters(), model.parameters(), ema_params_before, strict=True
    ):
        torch.testing.assert_close(p_ema, p_before * 0.9 + p_src * 0.1)
    ema_buf_map = dict(ema_model.named_buffers())
    src_buf_map = dict(model.named_buffers())
    for b_name in non_ep_touched:
        torch.testing.assert_close(ema_buf_map[b_name], src_buf_map[b_name])
    for b_name in ep_touched:
        torch.testing.assert_close(ema_buf_map[b_name], ema_bufs_before[b_name])


def test_p1_4_sanitize_causal_state_masked_ragged_and_matured_preds_nan_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """#109: `_sanitize_causal_state_masked` en paneles ragged y `layer_precision` bajo NaN GPU."""
    import mars_titan.models.mmarrs.long_running_harness as lrh
    from mars_titan.models.mmarrs.long_running_harness import _sanitize_causal_state_masked

    # 1a) Ragged WITH asset_ids: activos retenidos conservan estado previo, IPOs -> 0, 0 NaNs
    prev_state = {
        "last_asset_ids": torch.tensor([10, 20, 30], dtype=torch.long),
        "titans_state": {
            "fast": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "medium": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            "asset_ids": torch.tensor([10, 20, 30], dtype=torch.long),
        },
    }
    next_state_ragged = {
        "last_asset_ids": torch.tensor([10, 30, 40, 50], dtype=torch.long),
        "titans_state": {
            "fast": torch.full((4, 2), float("nan")),
            "medium": torch.full((4, 2), float("nan")),
            "asset_ids": torch.tensor([10, 30, 40, 50], dtype=torch.long),
        },
    }
    san_ragged = _sanitize_causal_state_masked(
        prev_state, next_state_ragged, torch.tensor(False)
    )
    ts_r = san_ragged["titans_state"]
    assert isinstance(ts_r, dict)
    assert torch.isfinite(ts_r["fast"]).all()
    assert torch.isfinite(ts_r["medium"]).all()
    expected_fast = torch.tensor([[1.0, 2.0], [5.0, 6.0], [0.0, 0.0], [0.0, 0.0]])
    torch.testing.assert_close(ts_r["fast"], expected_fast)
    assert ts_r["asset_ids"].tolist() == [10, 30, 40, 50]

    # 1b) Ragged WITHOUT asset_ids: fallback a ceros finitos sin fugar NaN
    prev_no_ids = {"titans_state": {"fast": torch.ones((3, 2))}}
    next_no_ids = {"titans_state": {"fast": torch.full((5, 2), float("nan"))}}
    san_no_ids = _sanitize_causal_state_masked(prev_no_ids, next_no_ids, torch.tensor(False))
    ts_n = san_no_ids["titans_state"]
    assert isinstance(ts_n, dict)
    assert ts_n["fast"].shape == (5, 2)
    assert torch.isfinite(ts_n["fast"]).all()
    torch.testing.assert_close(ts_n["fast"], torch.zeros((5, 2)))

    # 1c) `prev_state=None` con NaN -> ceros finitos sin fugar NaN
    san_none = _sanitize_causal_state_masked(None, next_no_ids, torch.tensor(False))
    assert torch.isfinite(san_none["titans_state"]["fast"]).all()

    # 2) Ruta `use_gpu_found_inf=True` en sesión ragged NaN no muestreada:
    #    `layer_precision`, `balance_bias`, `ema_model` y `_causal_state` siguen 100% finitos
    years = [2021, 2021, 2021]
    s0 = _make_synthetic_sessions([2021], n_assets=10, seed=901)[0]
    s1 = _make_synthetic_sessions([2021], n_assets=13, seed=902)[0]
    s2 = _make_synthetic_sessions([2021], n_assets=11, seed=903)[0]
    s1["prices"] = s1["prices"].clone()
    s1["prices"][:, :] = float("nan")
    ragged_sessions = [s0, s1, s2]

    cfg = HarnessConfig(
        run_dir=str(tmp_path / "gpu_found_inf_ragged_nan"),
        variation="v13_mmarrs_titan_full",
        split_mode="full_history_final",
        checkpoint_every_sessions=10,
        compaction_interval=10,
        health_check_every_k=10,
        auto_recover_on_nan=False,
        active_horizons=(1,),
        hidden_dim=16,
        device="cpu",
    )
    h = LongRunningTitansHarness(cfg)
    h.initialize(sample_session=s0)

    # Simular `use_gpu_found_inf=True` en CPU: si `found_inf == 1`, saltar `opt.step()` como CUDA
    import types

    monkeypatch.setattr(lrh, "_supports_gpu_found_inf", lambda _opt, _dev: True)
    orig_opt_step = h.optimizer.step

    def _sim_fused_step(_self_opt, *args, **kwargs):
        fi = getattr(h.optimizer, "found_inf", None)
        if isinstance(fi, torch.Tensor) and float(fi.item()) > 0.0:
            return None
        h.optimizer.found_inf = None  # type: ignore[attr-defined]
        return orig_opt_step(*args, **kwargs)

    monkeypatch.setattr(h.optimizer, "step", types.MethodType(_sim_fused_step, h.optimizer))

    captured_after_s1: dict[str, bool] = {}
    orig_fwd = h.model.forward_session
    call_count = {"n": 0}

    def _spy_fwd(batch, **kwargs):
        if call_count["n"] == 2:
            # Justo antes de sesión 2 (después del paso ragged NaN unchecked s1!)
            lp = h.model.recursive_core.layer_precision
            captured_after_s1["lp_finite"] = bool(torch.isfinite(lp).all().item())
            bb_ok = all(
                bool(torch.isfinite(blk.balance_bias).all().item())
                for blk in h.model.recursive_core.step_blocks
                if hasattr(blk, "balance_bias")
            )
            captured_after_s1["bb_finite"] = bb_ok
            captured_after_s1["ema_p_finite"] = all(
                bool(torch.isfinite(p).all().item()) for p in h.ema_model.parameters()
            )
            captured_after_s1["ema_b_finite"] = all(
                bool(torch.isfinite(b).all().item()) for _, b in h.ema_model.named_buffers()
            )
            ts = (h._causal_state or {}).get("titans_state", {})
            captured_after_s1["ts_finite"] = isinstance(ts, dict) and all(
                bool(torch.isfinite(v).all().item())
                for v in ts.values()
                if isinstance(v, torch.Tensor)
            )
        call_count["n"] += 1
        return orig_fwd(batch, **kwargs)

    monkeypatch.setattr(h.model, "forward_session", _spy_fwd)
    h.run_incremental_sessions(ragged_sessions, years, total_planned_steps=3)
    assert captured_after_s1 == {
        "lp_finite": True,
        "bb_finite": True,
        "ema_p_finite": True,
        "ema_b_finite": True,
        "ts_finite": True,
    }



