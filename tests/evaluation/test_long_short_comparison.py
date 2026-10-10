"""Cartera larga y corta de una comparación walk-forward sobre predicciones y precios sintéticos.

Las predicciones se escriben en cada prueba y los precios proceden de una edición sintética
con el formato real. Un brazo conoce de antemano el rendimiento de la sesión siguiente, otro
lo invierte y el control cero se abstiene, así que el resultado esperado se conoce a mano.
Nada procede de un modelo ajustado ni de datos de mercado.
"""

import copy
import json
import zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.data import prediction_files
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import long_short_comparison as comparison
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from tests.evaluation.test_walk_forward_comparison import ARMS, OFFSETS, Study
from tests.simulation.unadjusted_edition_fixture import Asset, write_edition

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
SYMBOLS = {"US": [f"S{i}" for i in range(8)], "CN": [f"60000{i}.SS" for i in range(8)]}
# Rendimiento de apertura a cierre de cada activo en las sesiones evaluadas.
GAINS = np.array([-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035])
WINDOWS = {"fold-000": ("2023-11-01", "2023-12-01"), "fold-001": ("2023-12-01", "2024-01-01")}


def decisions(market, start, end, count=3):
    clock = MarketClock(market, start, "2023-12-31")
    moments = [m for m, d in zip(clock.decisions, clock.days, strict=True) if d.isoformat() < end]
    return moments[:count]


def traded_days(market):
    """Posiciones en 2023 de las sesiones siguientes a las decisiones evaluadas."""
    year = [day.isoformat() for day in MarketClock(market, "2023-01-01", "2023-12-31").days]
    result = []
    for start, end in WINDOWS.values():
        for moment in decisions(market, start, end):
            day = moment.date().isoformat() if market == "US" else moment.date().isoformat()
            result.append(year.index(day) + 1)
    return result


def edition_assets():
    assets = {}
    for market, symbols in SYMBOLS.items():
        assets[market] = []
        for i, symbol in enumerate(symbols):
            overrides = {
                position: dict(open=20.0, close=round(20.0 * (1 + GAINS[i]), 2))
                for position in traded_days(market)
            }
            assets[market].append(Asset(symbol, base=20.0, overrides=overrides))
    return assets


class PortfolioStudy(Study):
    """Comparación sintética con decisiones reales del calendario y ocho activos por mercado."""

    def __init__(self, root, edition, **options):
        self.edition = edition
        super().__init__(root, **options)

    def declare(self, **changes):
        repository = json.loads(CONFIG.read_text())
        self.config["schema_version"] = 4
        for name in ("modality_strata", "modality_ablation", "long_short"):
            self.config[name] = copy.deepcopy(repository[name])
        self.config["long_short"].update(min_assets=8, **changes)
        self.publish()

    def table(self, arm, seed, fold, partition):
        start, end = fold[partition]
        keys = []
        for market in self.markets:
            for moment in decisions(market, start, end):
                for i, symbol in enumerate(SYMBOLS[market]):
                    keys.append((market, f"{market}/{symbol}", moment, GAINS[i]))
        rng = np.random.default_rng(zlib.crc32(f"{arm}/{seed}/{fold['id']}/{partition}".encode()))
        target = np.round(rng.normal(0, 1, len(keys)), 2)
        gain = np.array([key[3] for key in keys])
        prediction = dict(
            titans=gain + 0.001 * (seed - 42),
            gru=-gain,
            ridge=np.round(rng.normal(0, 1, len(keys)), 3),
        )[arm]
        values = dict(
            sample_id=[f"{asset}/{moment.isoformat()}" for _, asset, moment, _ in keys],
            asset_id=[asset for _, asset, _, _ in keys],
            market=[market for market, _, _, _ in keys],
            prediction_at=pa.array([key[2] for key in keys], pa.timestamp("us", tz="UTC")),
            target=target if partition == "calibration" else np.round(gain * 100, 2),
            prediction=prediction,
        )
        if ARMS[arm]["output"] == "quantile_head_v1":
            quantiles = prediction[:, None] + 0.2 * OFFSETS
            values.update(zip(QUANTILE_COLUMNS, quantiles.T, strict=True))
        for change in self.changes:
            change(arm, seed, fold["id"], partition, values)
        return pa.table(values)

    def run(self):
        return comparison.evaluate_long_short(
            self.config_path, self.sources_path, self.scope, self.edition
        )


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    root = tmp_path_factory.mktemp("edition")
    write_edition(root, edition_assets())
    return root


@pytest.fixture(scope="module")
def joint(tmp_path_factory, edition):
    study = PortfolioStudy(tmp_path_factory.mktemp("joint"), edition)
    study.declare()
    return study, *study.run()


def costs(report, market, arm, statistic="mean_net_return"):
    seed_mean = report["views"][market]["arms"][arm]["seed_mean"]
    return {cost: entry[statistic]["estimate"] for cost, entry in seed_mean.items()}


def test_foresight_inversion_and_abstention_give_the_returns_computed_by_hand(joint):
    _, report, sessions = joint
    assert report["final_test_opened"] is False and report["markets"] == ["US", "CN"]
    # Larga en +0,025 y +0,035 y corta en −0,035 y −0,025, con 0,25 cada una.
    gross, traded = 0.03, 2.0
    us = costs(report, "US", "titans")
    for cost in (0, 5, 10, 20):
        assert us[str(cost)] == pytest.approx(gross - cost / 10_000 * traded, abs=1e-12)
    inverse = costs(report, "US", "gru")
    assert inverse["10"] == pytest.approx(-gross - 0.001 * traded, abs=1e-12)
    assert set(costs(report, "US", "zero").values()) == {0.0}
    # China paga además el timbre de venta del 0,05 %: largas al cierre y cortas a la apertura.
    taxes = 0.25 * 0.0005 * (1.025 + 1.035) + 0.25 * 0.0005 * 2
    assert costs(report, "CN", "titans")["0"] == pytest.approx(gross - taxes, abs=1e-12)
    assert sessions.num_rows == (1 + 1 + 2 + 2) * 12


def test_statistics_follow_the_session_series_and_the_turnover(joint):
    _, report, _ = joint
    us = report["views"]["US"]
    titans = us["arms"]["titans"]
    assert titans["seed_mean"]["0"]["turnover"]["estimate"] == pytest.approx(2.0)
    expected = (1.03**6) - 1
    assert titans["seed_mean"]["0"]["cumulative_return"]["estimate"] == pytest.approx(expected)
    assert titans["seed_mean"]["0"]["max_drawdown"]["estimate"] == 0
    # Una serie constante no tiene volatilidad y su Sharpe queda sin definir.
    assert titans["seed_mean"]["0"]["sharpe"]["estimate"] is None
    execution = titans["seeds"]["42"]["execution"]
    assert execution["fill_rate"] == 1.0 and execution["mean_gross_exposure"] == 1.0
    assert execution["trades"] == 4 * 2 * 6
    zero = us["arms"]["zero"]["seeds"]["deterministic"]["execution"]
    assert zero["sessions_without_positions"] == 6 and zero["fill_rate"] is None
    assert us["resampling"]["sessions"] == 6 and us["sessions_per_year"] == 252
    assert report["views"]["CN"]["sessions_per_year"] == 243


def test_contrasts_use_the_declared_families_with_paired_sessions(joint):
    _, report, _ = joint
    family = report["views"]["US"]["contrasts"]["10"]["mean_net_return"]["quantile_models"]
    (row,) = family["contrasts"]
    assert row["name"] == "titans-gru"
    assert row["estimate"] == pytest.approx(0.06, abs=1e-12)
    # Las dos series son constantes: la diferencia no varía entre réplicas.
    assert row["interval"] == pytest.approx([0.06, 0.06], abs=1e-12)
    levels = report["views"]["US"]["contrasts"]["0"]["sharpe"]["levels"]
    assert all(
        entry["estimate"] is None for entry in levels["contrasts"] if entry["name"] != "ridge"
    )
    assert report["assumptions"]["china_t_plus_one"]


def test_a_market_outside_its_eligible_windows_stays_out_of_the_portfolio(joint, tmp_path):
    """Con el diseño conjunto, un mercado no elegible en una ventana no entra en sus libros.

    La elegibilidad se cambia en las fuentes ya validadas, como la dejaría `joint_design`
    para China antes de su historia mínima, sin declarar un diseño conjunto completo.
    """
    study = joint[0]
    config = walk.resolve_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    sources["eligible"]["CN"] = ["fold-001"]
    scoped = walk.scope_config(config, study.scope)
    declared = config["long_short"]
    books, _, sessions, record = comparison._window(
        sources, scoped, "fold-000", study.edition, declared
    )
    assert set(record["prices"]) == {"US"}
    assert set(sessions["market"].tolist()) == {sources["markets"].index("US")}
    assert all(len(book["gross_return"]) == len(sessions["time"]) for book in books.values())
    _, _, both, record = comparison._window(sources, scoped, "fold-001", study.edition, declared)
    assert set(record["prices"]) == {"US", "CN"} and len(set(both["market"].tolist())) == 2
    # Los agregados de la retención v2 guardan los mismos libros y solo los precios elegibles.
    folder = tmp_path / "aggregates"
    window_aggregates.write_long_short(folder, scoped, sources, "fold-000", study.edition)
    stored = window_aggregates.read_long_short(folder, scoped, sources, "fold-000", study.edition)
    assert window_aggregates.same(stored[0], books)
    identity = window_aggregates._long_short_identity(scoped, sources, "fold-000", study.edition)
    assert set(identity["prices"]) == {"US"}


def test_unexecutable_rows_leave_cash_and_are_counted(tmp_path, edition):
    study = PortfolioStudy(tmp_path, edition, scope="US")

    def unknown(arm, seed, fold, part, values):
        if part == "evaluation":
            values["asset_id"] = [
                asset.replace("US/S7", "US/MISSING") for asset in values["asset_id"]
            ]

    study.changes.append(unknown)
    study.declare()
    report, _ = study.run()
    titans = report["views"]["US"]["arms"]["titans"]
    assert titans["seed_mean"]["0"]["mean_net_return"]["estimate"] == pytest.approx(
        0.25 * 0.025 + 0.25 * 0.06
    )
    assert titans["seeds"]["42"]["execution"]["fill_rate"] == pytest.approx(3 / 4)
    assert report["unfilled_selected_rows"]["titans/42"]["no_row"] == 6
    assert report["windows"]["fold-000"]["rows_by_execution"]["no_row"] == 3


def test_arms_that_do_not_share_rows_are_rejected(tmp_path, edition):
    study = PortfolioStudy(tmp_path, edition, scope="US")

    def drift(arm, seed, fold, part, values):
        if arm == "ridge" and part == "evaluation":
            values["target"] = values["target"] + 1

    study.changes.append(drift)
    study.declare()
    with pytest.raises(ValueError, match="no evalúa las mismas filas"):
        study.run()


def test_configurations_without_the_portfolio_are_rejected(tmp_path, edition):
    study = PortfolioStudy(tmp_path, edition, scope="US")
    with pytest.raises(ValueError, match="cartera"):
        study.run()


def test_cli_writes_the_report_and_the_session_table(tmp_path, edition, capsys):
    study = PortfolioStudy(tmp_path / "study", edition, scope="CN")
    study.declare()
    output = tmp_path / "output"
    arguments = [
        "--config",
        str(study.config_path),
        "--sources",
        str(study.sources_path),
        "--scope",
        "CN",
        "--edition",
        str(edition),
        "--output",
        str(output),
    ]
    assert comparison.main(arguments) == 0
    assert "Reserva final cerrada" in capsys.readouterr().out
    report = json.loads((output / "long_short.json").read_text())
    assert report["artifacts"]["sessions.parquet"]
    assert report["prices"]["CN"]["market_rules"] == "cn_a_share_v1"
    with pytest.raises(ValueError, match="nueva"):
        comparison.main(arguments)


def test_the_portfolio_from_window_aggregates_is_identical_after_releasing_the_rows(
    tmp_path, edition, monkeypatch
):
    """Los libros por ventana bastan para el informe de la cartera, sin abrir filas."""
    study = PortfolioStudy(tmp_path / "study", edition)
    study.declare()
    expected, expected_sessions = study.run()
    config = walk.resolve_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    folder = tmp_path / "aggregates"
    for window in sources["windows"]:
        record = window_aggregates.write_long_short(folder, config, sources, window, edition)
        assert record["path"].name == f"{window}.long_short.npz"
    for entries in study.sources["arms"].values():
        for windows in entries.values():
            for entry in windows.values():
                for part in ("calibration", "evaluation"):
                    if part in entry:
                        path = study.sources_path.parent / entry[part]["path"]
                        prediction_files.release(path, entry[part]["sha256"], stage="fixture")
    with pytest.raises(prediction_files.PredictionsReleased):
        study.run()
    report, sessions = comparison.evaluate_long_short(
        study.config_path, study.sources_path, study.scope, edition, aggregates=folder
    )
    volatile = {"created_at_utc", "resources"}
    assert {k: v for k, v in report.items() if k not in volatile} == {
        k: v for k, v in expected.items() if k not in volatile
    }
    assert sessions.equals(expected_sessions)
    # Otra edición de precios, aunque las predicciones sean las mismas, no los acepta.
    from mars_titan.simulation.session_prices import SessionPrices

    original = SessionPrices.identity
    monkeypatch.setattr(
        SessionPrices, "identity", lambda self: dict(original(self), edition_id="other")
    )
    with pytest.raises(ValueError, match="prices"):
        comparison.evaluate_long_short(
            study.config_path, study.sources_path, study.scope, edition, aggregates=folder
        )


def test_seeds_are_averaged_session_by_session_before_the_statistics():
    sessions = 8
    rng = np.random.default_rng(2)

    def book(gross):
        zeros = np.zeros(sessions, dtype=np.int64)
        return dict(
            gross_return=gross,
            traded=np.full(sessions, 2.0),
            taxes=np.zeros(sessions),
            long_exposure=np.full(sessions, 0.5),
            short_exposure=np.full(sessions, 0.5),
            selected_long=zeros + 2,
            selected_short=zeros + 2,
            filled_long=zeros + 2,
            filled_short=zeros + 2,
            exits_at_limit=zeros,
        )

    first, second = rng.normal(0, 0.01, sessions), rng.normal(0.002, 0.01, sessions)
    books = {("a", 42): book(first), ("a", 43): book(second), ("b", 42): book(first * 0)}
    config = dict(
        comparison=dict(block_length=2, replicates=30, seed=5, confidence=0.9),
        resolved_families={"b_vs_a": {"b-a": {"b": 1.0, "a": -1.0}}},
    )
    declared = dict(cost_bps_per_side=[0], annualization_sessions={"US": 252})
    view = comparison._view(
        config, declared, list(books), books, np.ones(sessions, dtype=bool), "US"
    )
    mean = view["arms"]["a"]["seed_mean"]["0"]["mean_net_return"]["estimate"]
    assert mean == pytest.approx(np.mean((first + second) / 2), rel=1e-12)
    single = view["arms"]["a"]["seeds"]["43"]["costs"]["0"]["mean_net_return"]
    assert single == pytest.approx(np.mean(second), rel=1e-12)
    (row,) = view["contrasts"]["0"]["mean_net_return"]["b_vs_a"]["contrasts"]
    assert row["estimate"] == pytest.approx(-mean, rel=1e-12)
