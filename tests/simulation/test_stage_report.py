"""Informe financiero de la etapa de políticas sobre salidas confirmadas, sin aprender.

Las salidas proceden de la etapa reducida de `rl_stage_fixture`, con brazos aprendidos
sustituidos por políticas guionizadas. El informe solo lee recibos confirmados.
"""

import csv
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.simulation import campaign_stage, index_benchmark, stage_report
from tests.simulation import rl_stage_fixture as fixture

ARMS = ["klpo_terminal", "double_dqn", *fixture.REFERENCES]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return fixture.base_campaign(tmp_path_factory.mktemp("report"), "A")


@pytest.fixture
def stage_output(base, tmp_path, learning_doubles):
    summary = fixture.run(base, tmp_path / "stage", fixture.ScriptedLearner())
    assert summary["status"] == "completed"
    return tmp_path / "stage"


def report_of(base, outputs, folder, **options):
    result = stage_report.build_report(base.stage, outputs, folder, **options)
    return result, {
        (f["predictor"], f["cost_bps"]): f
        for section in result["sections"]
        for f in section["families"]
    }


def receipts(output):
    return {
        str(path.parent.relative_to(output / "jobs")): json.loads(path.read_text())
        for path in (output / "jobs").rglob("receipt.json")
    }


def test_every_family_pairs_klpo_with_its_controls_on_the_same_sessions(
    base, stage_output, tmp_path
):
    result, found = report_of(base, [stage_output], tmp_path / "report")
    assert result["kind"] == stage_report.REPORT_KIND and result["final_test_opened"] is False
    assert result["missing_benchmarks"] == ["CN"]
    assert set(found) == {(p, c) for p in ("gru", "lstm") for c in (0, 5, 10, 20)}
    for (predictor, _), family in found.items():
        arms = ARMS if predictor == "gru" else [a for a in ARMS if a != "double_dqn"]
        assert list(family["arms"]) == arms
        assert family["windows"] == ["fold-002", "fold-003"] and family["excluded"] == {}
        sessions = {values["sessions"] for values in family["arms"].values()}
        assert len(sessions) == 1 and sessions.pop() > 400
        bootstrap = family["bootstrap"]
        assert bootstrap["difference"] == "primary_minus_control"
        assert bootstrap["base"] == "klpo_terminal"
        assert set(bootstrap["differences"]["sharpe"]) == set(arms[1:])
        assert bootstrap["multiplicity"]["family_size"] == len(arms) - 1
        assert [entry["block_length"] for entry in bootstrap["sensitivity"]] == [5, 63]
        # Tres semillas de KLPO, con sus métricas por separado.
        assert family["arms"]["klpo_terminal"]["seeds"] == 3
        assert len(family["arms"]["klpo_terminal"]["per_seed"]) == 3
        cash = family["arms"]["cash"]
        assert cash["cumulative_return"] == 0 and cash["sharpe"] is None
    for name in ("report.json", "metrics.csv", "contrasts.csv", "equity.parquet"):
        assert (tmp_path / "report" / name).is_file()


def test_chained_windows_compound_the_liquidated_return_of_each_episode(
    base, stage_output, tmp_path
):
    _, found = report_of(base, [stage_output], tmp_path / "report")
    episodes = receipts(stage_output)
    for (predictor, cost), family in found.items():
        for arm in ("hold_initial", "equal_weight_monthly", "market_index"):
            growth = [
                1 + record["liquidated_net_return"]
                for window in family["windows"]
                for record in episodes[f"US/US/{window}/{predictor}/{arm}/reference"]["evaluation"]
                if record["cost_bps"] == cost
            ]
            expected = math.prod(growth) - 1
            assert family["arms"][arm]["cumulative_return"] == pytest.approx(expected, rel=1e-12)
        # KLPO reparte el capital entre sus semillas: su patrimonio final de una ventana es la
        # media de los patrimonios liquidados de cada semilla.
        window = family["windows"][0]
        liquidated = [
            1 + record["liquidated_net_return"]
            for seed in (42, 43, 44)
            for record in episodes[f"US/US/{window}/{predictor}/klpo_terminal/fit-s{seed}"][
                "evaluation"
            ]
            if record["cost_bps"] == cost
        ]
        table = pq.read_table(tmp_path / "report" / "equity.parquet").to_pylist()
        rows = [
            row
            for row in table
            if (row["predictor"], row["arm"], row["cost_bps"], row["window"])
            == (predictor, "klpo_terminal", cost, window)
        ]
        assert rows[-1]["nav"] == pytest.approx(1_000_000 * np.mean(liquidated), rel=1e-12)
        assert rows[0]["nav"] == 1_000_000


def test_contrasts_are_klpo_minus_each_control(base, stage_output, tmp_path):
    _, found = report_of(base, [stage_output], tmp_path / "report")
    family = found[("gru", 10)]
    table = pq.read_table(tmp_path / "report" / "equity.parquet").to_pylist()

    def returns(arm):
        series = {}
        for row in table:
            if (row["predictor"], row["arm"], row["cost_bps"]) == ("gru", arm, 10):
                series.setdefault(row["window"], []).append(row["nav"])
        return np.concatenate([np.diff(nav) / np.asarray(nav[:-1]) for nav in series.values()])

    klpo = returns("klpo_terminal")
    for control in ("cash", "market_index", "double_dqn"):
        row = family["bootstrap"]["differences"]["mean_session_return"][control]
        assert row["estimate"] == pytest.approx(klpo.mean() - returns(control).mean(), abs=1e-15)
        lower, upper = row["simultaneous_interval"]
        assert lower <= row["estimate"] <= upper
    with (tmp_path / "report" / "contrasts.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["primary"] for row in rows} == {"klpo_terminal"}
    assert {(row["predictor"], row["control"]) for row in rows} >= {("lstm", "market_index")}


def test_a_failed_window_is_excluded_with_its_reason_for_every_arm(
    base, tmp_path, learning_doubles, monkeypatch
):
    real = campaign_stage.campaign_source

    def source(base_, output, seed):
        read = real(base_, output, seed)

        def missing(scope, market, window, predictor):
            receipt, values = read(scope, market, window, predictor)
            return receipt, None if predictor == "lstm" and window == "fold-003" else values

        return missing

    monkeypatch.setattr(campaign_stage, "campaign_source", source)
    fixture.run(base, tmp_path / "stage", fixture.ScriptedLearner())
    _, found = report_of(base, [tmp_path / "stage"], tmp_path / "report")
    lstm = found[("lstm", 10)]
    assert lstm["windows"] == ["fold-002"]
    assert lstm["excluded"]["fold-003"]["reason"] == "failed_episodes"
    assert set(lstm["excluded"]["fold-003"]["arms"]) == {"klpo_terminal", *fixture.REFERENCES}
    assert found[("gru", 10)]["windows"] == ["fold-002", "fold-003"]


def test_a_paused_output_reports_its_pending_windows(base, tmp_path, learning_doubles):
    first = "US/US/fold-003/gru/klpo_terminal/fit-s42"
    summary = fixture.run(base, tmp_path / "stage", fixture.ScriptedLearner(pause={first}))
    assert summary["status"] == "paused"
    _, found = report_of(base, [tmp_path / "stage"], tmp_path / "report")
    family = found[("gru", 10)]
    assert family["windows"] == ["fold-002"]
    assert family["excluded"]["fold-003"] == dict(reason="pending_jobs", arms=ARMS)


def test_a_summary_that_disagrees_with_its_receipts_is_rejected(base, stage_output, tmp_path):
    path = stage_output / "summary.json"
    summary = json.loads(path.read_text())
    summary["metrics"]["gru"]["cash"]["10"]["completed"] += 1
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="no concuerdan"):
        stage_report.build_report(base.stage, [stage_output], tmp_path / "report")


def test_a_tampered_equity_series_is_rejected(base, stage_output, tmp_path):
    path = stage_output / "jobs/US/US/fold-002/gru/cash/reference/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["evaluation"][0]["equity"]["nav"][-1] *= 1.5
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="no concilia"):
        stage_report.build_report(base.stage, [stage_output], tmp_path / "report")


def test_only_declared_benchmarks_are_accepted(base, stage_output, tmp_path):
    with pytest.raises(ValueError, match="declarados"):
        stage_report.build_report(
            base.stage, [stage_output], tmp_path / "report", benchmarks={"US": ("x", "0" * 64)}
        )


def test_the_chinese_index_benchmark_joins_a_family_without_an_engine_instrument(tmp_path):
    day = 86_400_000_000
    start = int(np.datetime64("2023-01-03T07:05", "us").astype(np.int64))
    closes = [start + day * k for k in range(3)]

    def record(nav, status="completed"):
        return dict(
            cost_bps=10,
            status=status,
            reason=None,
            liquidated_net_return=nav[-1] / nav[0] - 1,
            turnover=0.0,
            costs=0.0,
            equity=dict(basis="close_valuation", close_times=closes, nav=nav),
        )

    found = {
        "fold-010": {
            "klpo_terminal": [record([100.0, 101.0, 102.0])] * 3,
            "cash": [record([100.0, 100.0, 100.0])],
        }
    }
    path = tmp_path / "levels.parquet"
    pq.write_table(
        pa.table(
            dict(
                session=["2023-01-03", "2023-01-04", "2023-01-05"],
                open=[10.0, 11.0, 12.0],
                close=[10.0, 11.0, 13.0],
            )
        ),
        path,
    )
    levels = index_benchmark.read_levels(path, sha256(path))
    family = stage_report._family(
        found,
        ["fold-010"],
        ["klpo_terminal", "cash", "market_index"],
        planned=lambda window, arm: dict(klpo_terminal=3, cash=1).get(arm, 0),
        benchmark=levels,
        market="CN",
        capital=100.0,
        cost=10,
    )
    index = family["arms"]["market_index"]
    units = 100 / (11.0 * 1.001)
    assert index["basis"] == index_benchmark.BASIS
    assert index["final_nav"] == pytest.approx(units * 13 * 0.999, rel=1e-14)
    assert family["equity"]["market_index"]["nav"][:2] == [100.0, pytest.approx(units * 11)]


def test_seeds_share_the_capital_and_a_ruined_seed_stays_at_zero():
    closes = [1, 2, 3]

    def record(nav, status, liquidated, turnover, costs):
        return dict(
            status=status,
            liquidated_net_return=liquidated,
            turnover=turnover,
            costs=costs,
            equity=dict(basis="close_valuation", close_times=closes[: len(nav)], nav=nav),
        )

    window = stage_report._arm_window(
        [
            record([100.0, 110.0, 120.0], "completed", 0.19, 2.0, 1.0),
            record([100.0, 0.0], "ruined", -1.0, 1.0, 3.0),
        ],
        closes,
        100.0,
    )
    # La semilla completa paga su salida en el último cierre y la arruinada vale cero.
    assert window["nav"].tolist() == [100.0, 55.0, 59.5]
    assert [nav.tolist() for nav in window["seeds"]] == [[100.0, 110.0, 119.0], [100.0, 0.0, 0.0]]
    assert (window["turnover"], window["costs"]) == (1.5, 0.02)
    with pytest.raises(ValueError, match="compartir sus cierres"):
        stage_report._arm_window([record([100.0, 0.0], "ruined", -1.0, 0, 0)], [1, 5, 6], 100.0)


def test_the_campaign_command_writes_the_report(base, stage_output, tmp_path, capsys):
    import runpy

    script = runpy.run_path("scripts/run_masked_campaign.py")
    arguments = ["rl-report", "--stage", str(base.stage), "--output", str(stage_output)]
    assert script["main"]([*arguments, "--report", str(tmp_path / "report")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["families"] == 8 and printed["with_bootstrap"] == 8
    assert printed["missing_benchmarks"] == ["CN"]


def test_a_universe_series_ending_in_evaluation_is_listed_for_the_survival_sensitivity(
    tmp_path_factory, tmp_path, learning_doubles
):
    # A0000 deja de cotizar en octubre de 2023, dentro de la evaluación de fold-003.
    base = fixture.base_campaign(tmp_path_factory.mktemp("ending"), "A", ending=200)
    fixture.run(base, tmp_path / "stage", fixture.ScriptedLearner())
    result, found = report_of(base, [tmp_path / "stage"], tmp_path / "report")
    survival = result["sections"][0]["survival"]
    assert survival["status"] == "secondary_evaluation_pending"
    assert survival["exit_returns"] == [0.0, -0.3, -1.0] and survival["role"] == "secondary"
    assert survival["affected"] == [
        dict(scope="US", market="US", window="fold-003", assets=["US/A0000"])
    ]
    assert found[("gru", 10)]["windows"] == ["fold-002"]
    assert found[("gru", 10)]["excluded"]["fold-003"]["reason"] == "failed_episodes"


def test_without_ending_series_the_survival_sensitivity_has_no_window(base, stage_output, tmp_path):
    result, _ = report_of(base, [stage_output], tmp_path / "report")
    survival = result["sections"][0]["survival"]
    assert survival["status"] == "no_affected_windows" and survival["affected"] == []


def test_survival_lists_only_universe_exclusions_caused_by_an_ending_series():
    # Una ventana fallida por filas sin verificar o por falta de predicciones no depende de
    # un retorno de salida. Solo cuentan los activos del universo cuya serie termina.
    policies = fixture.policies()

    def receipt(window, failure):
        job = dict(scope="US", market="US", window=window)
        return dict(job=job, identity=dict(tapes=dict(failure=failure)))

    excluded = "universe_assets_excluded"
    receipts = dict(
        a=receipt(
            "fold-005",
            dict(
                reason=excluded,
                excluded={"US/X": "series_ends_in_tape", "US/Y": "unverified_rows_in_tape"},
            ),
        ),
        b=receipt("fold-005", dict(reason=excluded, excluded={"US/Z": "series_ends_in_tape"})),
        c=receipt("fold-006", dict(reason=excluded, excluded={"US/Y": "unverified_rows_in_tape"})),
        d=receipt("fold-007", dict(reason="predictor_without_predictions")),
        e=receipt("fold-008", None),
    )
    result = stage_report.survival(policies, dict(receipts=receipts))
    assert result["status"] == "secondary_evaluation_pending"
    assert result["affected"] == [
        dict(scope="US", market="US", window="fold-005", assets=["US/X", "US/Z"])
    ]
