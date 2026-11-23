"""Pruebas TDD para la calculadora de unidades de cómputo (CU) y política de parada."""

from __future__ import annotations

import json

import cu_budget_guard as cg
import pytest


@pytest.mark.parametrize(
    ("panel_text", "expected_rate", "expected_avail"),
    [
        ("8.42 compute units per hour", 8.42, None),
        ("7,8 CU/h", 7.8, None),
        ("Available: 812.4 compute units | Rate: 8.7/hr", 8.7, 812.4),
        ("Disponibles: 640,5 CU — Tasa: 11,2 unidades/hora", 11.2, 640.5),
        ('{"cu_per_hour": 6.95, "available_cu": 920.0}', 6.95, 920.0),
    ],
)
def test_parse_colab_cu_panel_supports_panel_formats_and_json(
    panel_text, expected_rate, expected_avail
):
    parsed = cg.parse_colab_cu_panel(panel_text)
    assert parsed["cu_per_hour"] == pytest.approx(expected_rate, rel=1e-9)
    assert parsed["available_cu"] == expected_avail


@pytest.mark.parametrize(
    "bad_text",
    [
        "",
        "sin tasa numerica",
        "0.0 CU/h",
        "-5.0 compute units per hour",
        '{"cu_per_hour": 0}',
        '{"cu_per_hour": 7.8, "available_cu": -1.0}',
    ],
)
def test_parse_colab_cu_panel_rejects_invalid_or_non_positive(bad_text):
    with pytest.raises(ValueError):
        cg.parse_colab_cu_panel(bad_text)


def test_origin_row_multiplier_anchors_and_strict_2024_seal():
    assert cg.origin_row_multiplier(2013, "US") == pytest.approx(1.0, rel=1e-9)
    assert cg.origin_row_multiplier(2023, "US") == pytest.approx(
        13_775_654.0 / 5_353_824.0, rel=1e-6
    )
    assert cg.origin_row_multiplier(2013, "US+CN") == pytest.approx(
        6_080_386.0 / 5_353_824.0, rel=1e-6
    )
    assert cg.origin_row_multiplier(2023, "US+CN") == pytest.approx(
        16_204_399.0 / 5_353_824.0, rel=1e-6
    )

    for sealed in (2024, 2025):
        with pytest.raises(ValueError, match="sellado"):
            cg.origin_row_multiplier(sealed, "US")
    for bad in (2004, True):
        with pytest.raises(ValueError):
            cg.origin_row_multiplier(bad, "US")


def test_estimate_remaining_job_wall_hours_from_real_steps_per_min():
    # 7200 pasos a 240 pasos/min = 30 min = 0.5 h; con 8% sobrecarga = 0.54 h
    hours = cg.estimate_remaining_job_wall_hours(
        remaining_steps=7200,
        steps_per_min_aggregate=240.0,
        relay_overhead_fraction=0.08,
    )
    assert hours == pytest.approx(0.54, rel=1e-9)

    assert (
        cg.estimate_remaining_job_wall_hours(remaining_steps=0, steps_per_min_aggregate=240.0)
        == 0.0
    )

    with pytest.raises(ValueError):
        cg.estimate_remaining_job_wall_hours(remaining_steps=100, steps_per_min_aggregate=0.0)
    with pytest.raises(ValueError):
        cg.estimate_remaining_job_wall_hours(remaining_steps=-1, steps_per_min_aggregate=100.0)


def test_cu_ledger_persists_prior_relays_without_losing_spent_cu(tmp_path):
    ledger = tmp_path / "cu_ledger.json"
    r1 = cg.load_and_update_cu_ledger(
        ledger,
        session_id="relay-01",
        session_elapsed_hours=20.0,
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
    )
    assert r1["total_cu_spent"] == pytest.approx(160.0)
    assert r1["prior_relays_cu_spent"] == pytest.approx(0.0)

    # Segundo relevo: NO pierde las 160 CU del relevo 1
    r2 = cg.load_and_update_cu_ledger(
        ledger,
        session_id="relay-02",
        session_elapsed_hours=10.0,
        cu_per_hour=8.5,
        total_budget_cu=1000.0,
    )
    assert r2["prior_relays_cu_spent"] == pytest.approx(160.0)
    assert r2["current_session_cu_spent"] == pytest.approx(85.0)
    assert r2["total_cu_spent"] == pytest.approx(245.0)

    # Actualizar el mismo relevo-02 no duplica sus CU previas
    r2_upd = cg.load_and_update_cu_ledger(
        ledger,
        session_id="relay-02",
        session_elapsed_hours=12.0,
        cu_per_hour=8.5,
        total_budget_cu=1000.0,
    )
    assert r2_upd["prior_relays_cu_spent"] == pytest.approx(160.0)
    assert r2_upd["total_cu_spent"] == pytest.approx(262.0)


def test_evaluate_launch_or_stop_all_decision_branches_and_boundaries():
    # 1. Caso LAUNCH limpio: 1000 CU, reserva 150 CU, gastadas 200 CU -> quedan 650 utilizables;
    # siguiente trabajo cuesta 10 h * 8 CU/h = 80 CU <= 650 y sesión lleva 5 h < 22.5 h
    d_launch = cg.evaluate_launch_or_stop(
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
        prior_relays_cu_spent=160.0,
        session_elapsed_hours=5.0,
        next_job_estimated_wall_hours=10.0,
        next_origin_year=2013,
        reserve_fraction=0.15,
    )
    assert d_launch["action"] == "launch"
    assert d_launch["exit_code"] == cg.EXIT_LAUNCH
    assert d_launch["total_cu_spent"] == pytest.approx(200.0)
    assert d_launch["usable_cu_remaining"] == pytest.approx(650.0)
    assert d_launch["next_job_cu"] == pytest.approx(80.0)
    assert d_launch["fits_current_relay"] is True

    # 2. Caso STOP por reserva CU alcanzada (850 gastadas con reserva 150 -> usable = 0)
    d_res = cg.evaluate_launch_or_stop(
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
        prior_relays_cu_spent=810.0,
        session_elapsed_hours=5.0,
        next_job_estimated_wall_hours=1.0,
        next_origin_year=2014,
        reserve_fraction=0.15,
    )
    assert d_res["action"] == "stop_and_sync"
    assert d_res["exit_code"] == cg.EXIT_STOP_CU_RESERVE
    assert d_res["reason"] == "cu_reserve_reached"
    assert any("colab_sync.py push" in step for step in d_res["shutdown_protocol"])

    # 3. Caso STOP por límite de sesión 22.5 h (margen 0.5 h -> a las 22.1 h quedan 0.4 h < 0.5 h)
    d_wall = cg.evaluate_launch_or_stop(
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
        prior_relays_cu_spent=100.0,
        session_elapsed_hours=22.1,
        next_job_estimated_wall_hours=2.0,
        next_origin_year=2015,
        reserve_fraction=0.15,
        min_session_margin_hours=0.5,
    )
    assert d_wall["action"] == "stop_and_sync"
    assert d_wall["exit_code"] == cg.EXIT_STOP_SESSION_WALL
    assert d_wall["reason"] == "session_wall_limit_reached"

    # 4. Caso STOP porque el próximo trabajo excede las CU utilizables (quedan 50 CU, trabajo
    # cuesta 64 CU)
    d_next = cg.evaluate_launch_or_stop(
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
        prior_relays_cu_spent=760.0,
        session_elapsed_hours=5.0,  # 760 + 40 = 800 spent; reserve 150 -> 50 usable
        next_job_estimated_wall_hours=8.0,  # 64 CU > 50
        next_origin_year=2016,
        reserve_fraction=0.15,
    )
    assert d_next["action"] == "stop_and_sync"
    assert d_next["exit_code"] == cg.EXIT_STOP_JOB_EXCEEDS_BUDGET
    assert d_next["reason"] == "next_job_exceeds_usable_cu"

    # 5. Bloqueo estricto de origen 2024
    with pytest.raises(ValueError, match="sellado"):
        cg.evaluate_launch_or_stop(
            cu_per_hour=8.0,
            total_budget_cu=1000.0,
            prior_relays_cu_spent=0.0,
            session_elapsed_hours=1.0,
            next_job_estimated_wall_hours=2.0,
            next_origin_year=2024,
        )


def test_standard_scenarios_consistent_baseline_and_cli(capsys):
    report = cg.build_campaign_cu_scenarios(cu_per_hour=7.8, workers=4, speedup_vs_dgx=1.5)
    assert report["rate_status"] == "[no verificado en GPU]"
    assert len(report["scenarios"]) == 4
    # Verificar coherencia exacta: ningún bloque usa base 1.5 desalineada (todos 1.6 h DGX)
    for sc in report["scenarios"]:
        assert sc["base_dgx_hours_2013"] == pytest.approx(1.6)
        assert sc["estimated_cu"] > 0

    code = cg.main(
        [
            "decide",
            "--panel-text",
            "8.0 CU/h",
            "--budget-cu",
            "1000",
            "--prior-cu-spent",
            "100",
            "--session-elapsed-hours",
            "4.0",
            "--remaining-steps",
            "12000",
            "--steps-per-min",
            "200.0",
            "--next-origin",
            "2013",
        ]
    )
    assert code == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision["action"] == "launch"


def test_pass3_adversarial_nan_inf_explicit_reserve_relay_warning_ledger_and_cli_codes(
    tmp_path, capsys
):
    # NaN, Inf y booleanos deben rechazarse siempre (nunca colarse a launch)
    for bad_val in (float("nan"), float("inf"), float("-inf"), True):
        with pytest.raises(ValueError, match="finito"):
            cg.evaluate_launch_or_stop(
                cu_per_hour=bad_val,
                total_budget_cu=1000.0,
                prior_relays_cu_spent=0.0,
                session_elapsed_hours=1.0,
                next_job_estimated_wall_hours=1.0,
                next_origin_year=2013,
            )
    with pytest.raises(ValueError, match="finito"):
        cg.load_and_update_cu_ledger(
            tmp_path / "nan_ledger.json",
            session_id="relay-01",
            session_elapsed_hours=float("nan"),
            cu_per_hour=8.0,
        )

    # Reserva explícita en CU + aviso fits_current_relay=False cuando trabajo > resto sesión 22.5 h
    dec = cg.evaluate_launch_or_stop(
        cu_per_hour=8.0,
        total_budget_cu=1000.0,
        prior_relays_cu_spent=100.0,
        session_elapsed_hours=18.0,  # quedan 4.5 h en sesión
        next_job_estimated_wall_hours=10.0,  # 80 CU cabe en presupuesto pero supera 4.5 h de relevo
        next_origin_year=2013,
        reserve_cu=200.0,
    )
    assert dec["action"] == "launch"
    assert dec["effective_reserve_cu"] == pytest.approx(200.0)
    assert dec["fits_current_relay"] is False
    assert dec["relay_warning"] is not None

    # Ledger corrupto y enlace simbólico rechazados
    bad_ledger = tmp_path / "bad.json"
    bad_ledger.write_text('{"kind": "otro"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Esquema"):
        cg.load_and_update_cu_ledger(
            bad_ledger, session_id="r1", session_elapsed_hours=1.0, cu_per_hour=8.0
        )

    sym_ledger = tmp_path / "sym.json"
    sym_ledger.symlink_to(bad_ledger)
    with pytest.raises(ValueError, match="enlace simbólico"):
        cg.load_and_update_cu_ledger(
            sym_ledger, session_id="r1", session_elapsed_hours=1.0, cu_per_hour=8.0
        )

    # Códigos de salida CLI: 10 (reserva), 11 (reloj 22.5 h), 12 (próximo trabajo excede CU), 2
    # (origen 2024)
    assert (
        cg.main(
            [
                "decide",
                "--panel-text",
                "8.0 CU/h",
                "--budget-cu",
                "1000",
                "--prior-cu-spent",
                "850",
                "--session-elapsed-hours",
                "1.0",
                "--next-job-hours",
                "1.0",
                "--next-origin",
                "2013",
            ]
        )
        == cg.EXIT_STOP_CU_RESERVE
    )
    capsys.readouterr()

    assert (
        cg.main(
            [
                "decide",
                "--panel-text",
                "8.0 CU/h",
                "--budget-cu",
                "1000",
                "--prior-cu-spent",
                "0",
                "--session-elapsed-hours",
                "22.2",
                "--next-job-hours",
                "1.0",
                "--next-origin",
                "2013",
            ]
        )
        == cg.EXIT_STOP_SESSION_WALL
    )
    capsys.readouterr()

    assert (
        cg.main(
            [
                "decide",
                "--panel-text",
                "8.0 CU/h",
                "--budget-cu",
                "1000",
                "--prior-cu-spent",
                "800",
                "--session-elapsed-hours",
                "1.0",
                "--next-job-hours",
                "10.0",
                "--next-origin",
                "2013",
            ]
        )
        == cg.EXIT_STOP_JOB_EXCEEDS_BUDGET
    )
    capsys.readouterr()

    assert (
        cg.main(
            [
                "decide",
                "--panel-text",
                "8.0 CU/h",
                "--budget-cu",
                "1000",
                "--prior-cu-spent",
                "0",
                "--session-elapsed-hours",
                "1.0",
                "--next-job-hours",
                "1.0",
                "--next-origin",
                "2024",
            ]
        )
        == cg.EXIT_ERROR
    )
    err_out = json.loads(capsys.readouterr().out)
    assert "sellado" in err_out["error"]
