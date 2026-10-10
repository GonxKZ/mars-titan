"""Estratos de liquidez causales y peso de las filas extremas en el MSE.

Las ediciones sin ajustar y las predicciones son sintéticas, con valores conocidos. No
proceden de ningún modelo ajustado ni de la edición real, no se lee el año sellado y no se
ejecuta ningún paso de optimizador.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation import comparison_matrix as matrix_module
from mars_titan.evaluation import liquidity_strata as ls
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates as aggregates
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import SessionScores, score_sessions
from tests.evaluation.liquidity_fixture import EXPECTED, THIN, section, study_edition
from tests.evaluation.test_walk_forward_comparison import ARMS, Study
from tests.simulation.unadjusted_edition_fixture import Asset, tape_days, write_edition

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "configs/evaluation"
CAMPAIGN = (
    "historical-masked-2000-comparison.json",
    "historical-masked-2000-joint-comparison.json",
)


def decision(market, day):
    """Instante de decisión (cierre más cinco minutos) de una sesión, en microsegundos UTC."""
    clock = MarketClock(market, "2023-01-01", "2023-12-31")
    return int(clock.decision(day).timestamp()) * 1_000_000


def assign(edition, market, symbols, moments, **changes):
    liquidity = ls.EditionLiquidity(edition, section(**changes))
    assets = [f"{market}/{symbol}" for symbol in symbols for _ in moments]
    instants = [moment for _ in symbols for moment in moments]
    codes, reasons = liquidity.assign([market] * len(assets), assets, instants)
    shape = (len(symbols), len(moments))
    names = np.array(ls.NAMES)[codes].reshape(shape)
    return names, np.array(ls.REASONS)[reasons].reshape(shape)


# Declaración


@pytest.mark.parametrize("name", CAMPAIGN)
def test_the_campaign_comparisons_declare_the_strata_with_their_quadratic_metric(name):
    config = walk.load_config(EVALUATION / name)
    declared = config[walk.LIQUIDITY_FIELD]
    assert "mse" in config["comparison"]["metrics"] and declared["metrics"] == ["mae", "mse"]
    assert declared["declared_at"] == "2026-10-10" and declared["source"] == ls.SOURCE
    assert (declared["price_below"], declared["volume_below"]) == (1.0, 1000)
    assert declared["volume_sessions"] == 20 and declared["extremes"] == [1, 10, 100]


def test_every_comparison_that_contrasts_a_quadratic_metric_declares_its_strata():
    checked = 0
    for path in sorted(EVALUATION.glob("*.json")):
        config = json.loads(path.read_text())
        if not isinstance(config, dict) or config.get("kind") != walk.CONFIG_KIND:
            continue
        checked += 1
        if set(ls.QUADRATIC) & set(config["comparison"]["metrics"]):
            assert walk.LIQUIDITY_FIELD in config, path.name
    assert checked >= 2


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda s: s.pop("listed_rows"), "exactamente sus campos"),
        (lambda s: s.update(status="primary"), "análisis secundario"),
        (lambda s: s.update(source="prepared_adjusted_close"), "análisis secundario"),
        (lambda s: s.update(declared_at="10/10/2026"), "análisis secundario"),
        (lambda s: s["strata"].pop("unclassified"), "cuatro de precio"),
        (lambda s: s["strata"]["liquid"].update(low_price=True), "cuatro de precio"),
        (lambda s: s.update(price_below=0), "umbral de precio"),
        (lambda s: s.update(price_below=True), "umbral de precio"),
        (lambda s: s.update(volume_below=1000.5), "enteros positivos"),
        (lambda s: s.update(volume_sessions=0), "enteros positivos"),
        (lambda s: s.update(volume_sessions=251), "enteros positivos"),
        (lambda s: s.update(metrics=["mae"]), "cada métrica cuadrática"),
        (lambda s: s.update(metrics=["mse", "mae"]), "cada métrica cuadrática"),
        (lambda s: s.update(metrics=["mae", "mse", "mse"]), "cada métrica cuadrática"),
        (lambda s: s.update(extremes=[10, 1]), "crecientes"),
        (lambda s: s.update(extremes=[0, 10]), "crecientes"),
        (lambda s: s.update(extremes=[1, 10_001]), "crecientes"),
        (lambda s: s.update(extremes=[1, 10], listed_rows=11), "listadas"),
        (lambda s: s.update(listed_rows=101), "listadas"),
        (lambda s: s.update(recalibrate=True), "calibrar"),
        (lambda s: s.update(min_rows=0), "min_rows"),
        (lambda s: s.update(min_sessions=1.5), "min_sessions"),
    ],
)
def test_the_declaration_is_fixed_before_reading_any_data(change, message):
    declared = copy.deepcopy(section())
    change(declared)
    with pytest.raises(ValueError, match=message):
        ls.declaration(declared, walk.SERIES_METRICS, ["mae", "mse"])


def test_only_the_mae_is_required_when_no_quadratic_metric_is_contrasted():
    declared = section(metrics=["mae"])
    assert ls.declaration(declared, walk.SERIES_METRICS, ["mae", "rank_ic"]) is declared


# Asignación causal


@pytest.fixture(scope="module")
def edition(tmp_path_factory):
    return study_edition(tmp_path_factory.mktemp("edition"))


@pytest.mark.parametrize("market", ["US", "CN"])
def test_each_asset_falls_in_its_declared_stratum(edition, market):
    days = tape_days(market)
    moments = [decision(market, days[index]) for index in (40, 120, 200)]
    symbols = sorted(EXPECTED)
    names, reasons = assign(edition, market, symbols, moments)
    for row, symbol in enumerate(symbols):
        assert set(names[row]) == {EXPECTED[symbol]}, symbol
    assert set(reasons[symbols.index("D")]) == {"no_verified_traded_close"}
    assert set(reasons[symbols.index("G")]) == {"asset_not_in_edition"}
    classified = [row for row, symbol in enumerate(symbols) if symbol not in "DG"]
    assert set(reasons[classified].ravel()) == {"classified"}


def perturbed(root, after):
    """El mismo activo con precio y volumen de una acción ilíquida desde la sesión `after`."""
    late = {index: dict(close=0.5, open=0.5, volume=THIN) for index in range(after, 250)}
    write_edition(root, {"US": [Asset("E", overrides=late)]})
    return root


def test_rows_after_the_published_session_never_change_the_stratum(tmp_path):
    days = tape_days("US")
    clean = tmp_path / "clean"
    write_edition(clean, {"US": [Asset("E")]})
    dirty = perturbed(tmp_path / "dirty", 121)
    moments = [decision("US", days[index]) for index in range(60, 121)]
    # Una decisión hasta la sesión 120 no ve nada de lo cambiado, ni el día de su etiqueta.
    assert np.array_equal(
        assign(clean, "US", ["E"], moments)[0], assign(dirty, "US", ["E"], moments)[0]
    )
    boundary = decision("US", days[121])
    before, at = assign(dirty, "US", ["E"], [boundary - 1, boundary])[0][0]
    # Un microsegundo antes de la decisión de la sesión 121 su barra aún no está publicada.
    assert (before, at) == ("liquid", "low_price")


def test_the_label_session_is_never_read(tmp_path):
    days = tape_days("US")
    # Solo cambia la sesión 121, que es la de la etiqueta de la decisión de la 120.
    label = {121: dict(close=0.5, open=0.5, volume=0.0)}
    write_edition(tmp_path / "label", {"US": [Asset("E", overrides=label)]})
    names, reasons = assign(tmp_path / "label", "US", ["E"], [decision("US", days[120])])
    assert names[0, 0] == "liquid" and reasons[0, 0] == "classified"


def test_price_and_volume_thresholds_are_strict_and_use_the_traded_grid(tmp_path):
    year = range(len(tape_days("US")))
    write_edition(
        tmp_path,
        {
            "US": [
                Asset("ONE", overrides={i: dict(close=1.0, open=1.0) for i in year}),
                Asset("SUB", overrides={i: dict(close=0.9999, open=0.9999) for i in year}),
                Asset("VOL", overrides={i: dict(volume=1000.0) for i in year}),
                Asset("LOW", overrides={i: dict(volume=999.0) for i in year}),
            ]
        },
    )
    moments = [decision("US", tape_days("US")[150])]
    names, _ = assign(tmp_path, "US", ["LOW", "ONE", "SUB", "VOL"], moments)
    assert names.ravel().tolist() == ["thin_volume", "liquid", "low_price", "liquid"]
    # Con otro umbral declarado cambia la asignación, sin tocar los datos.
    names, _ = assign(tmp_path, "US", ["ONE"], moments, price_below=1.01)
    assert names[0, 0] == "low_price"


def test_suspensions_count_as_zero_volume_and_their_filled_close_is_ignored(tmp_path):
    # Once sesiones rellenadas con volumen cero y un cierre que sube 0,37 cada vez.
    write_edition(tmp_path, {"US": [Asset("Z", base=0.5, zero_volume=tuple(range(100, 111)))]})
    days = tape_days("US")
    names, reasons = assign(tmp_path, "US", ["Z"], [decision("US", days[110])])
    # El cierre rellenado supera 1, pero el último cierre negociado sigue por debajo.
    assert names[0, 0] == "low_price_and_thin_volume" and reasons[0, 0] == "classified"
    names, _ = assign(tmp_path, "US", ["Z"], [decision("US", days[109])])
    assert names[0, 0] == "low_price"


def test_unverified_rows_and_short_histories_leave_rows_unclassified(tmp_path):
    write_edition(
        tmp_path,
        {"US": [Asset("U", unverified=(150,)), Asset("S", start=100)]},
    )
    days = tape_days("US")
    moments = [decision("US", days[index]) for index in (99, 110, 150, 169, 170)]
    names, reasons = assign(tmp_path, "US", ["S", "U"], moments)
    assert reasons[0].tolist() == [
        "no_published_row",
        "incomplete_volume_window",
        "classified",
        "classified",
        "classified",
    ]
    assert reasons[1].tolist() == [
        "classified",
        "classified",
        "incomplete_volume_window",
        "incomplete_volume_window",
        "classified",
    ]
    assert set(names[reasons != "classified"]) == {"unclassified"}


def test_assets_of_another_market_and_a_changed_edition_are_rejected(edition, tmp_path):
    liquidity = ls.EditionLiquidity(edition, section())
    moment = decision("US", tape_days("US")[40])
    with pytest.raises(ValueError, match="no es de su mercado"):
        liquidity.assign(["US"], ["CN/A"], [moment])
    copy_root = tmp_path / "copy"
    write_edition(copy_root, {"US": [Asset("E")]})
    target = copy_root / "assets/US/E/prices.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ValueError, match="ha cambiado"):
        ls.EditionLiquidity(copy_root, section()).assign(["US"], ["US/E"], [moment])


# Filas extremas


def panel(rows):
    """Panel con filas (mercado, activo, instante, error) y objetivo cero."""
    markets = [market for market, *_ in rows]
    ids = [f"{market}/{market}/{asset}/{at}" for market, asset, at, _ in rows]
    return ForecastPanel.from_columns(
        ids,
        markets,
        np.array([at for *_, at, _ in rows], dtype=np.int64),
        np.zeros(len(rows)),
        np.array([error for *_, error in rows], dtype=np.float64),
    )


DAY = 86_400_000_000
# US: sesión 1 con errores 3 y 1, sesión 2 con error 2. CN: una sesión con tres errores 1.
ROWS = [
    ("US", "A", DAY, 3.0),
    ("US", "B", DAY, 1.0),
    ("US", "A", 2 * DAY, 2.0),
    ("CN", "X", DAY, 1.0),
    ("CN", "Y", DAY, 1.0),
    ("CN", "Z", DAY, 1.0),
]


def extremes(parts, panels, weighting, markets=("US", "CN"), **changes):
    scores = SessionScores.concatenate([score_sessions(p, rank_ic_min_assets=3) for p in panels])
    views = walk._views(scores, list(markets))
    declared = section(**changes)
    return ls.extremes_report(parts, views, list(markets), weighting, declared)


@pytest.mark.parametrize(
    "weighting, session_shares",
    [
        # MSE por sesión de 10/3. Peso de cada fila: error al cuadrado / filas / sesiones.
        ("session", [1.5 / (10 / 3), (1.5 + 4 / 3) / (10 / 3), 1.0]),
        # Ponderación por mercado: (4,5 + 1) / 2 = 2,75 y una fila de US pesa e² / n / 4.
        ("market", [1.125 / 2.75, 2.125 / 2.75, 1.0]),
    ],
)
def test_extreme_shares_match_the_hand_computation(weighting, session_shares):
    whole = panel(ROWS)
    part = ls.window_extremes(whole, np.zeros(whole.rows, dtype=np.int8), 6)
    report = extremes([part], [whole], weighting, extremes=[1, 2, 6], listed_rows=2)
    scope = report["US+CN"]
    assert scope["row"]["total"] == pytest.approx(17.0, rel=1e-15)
    assert [scope["row"]["shares"][k] for k in ("1", "2", "6")] == pytest.approx(
        [9 / 17, 13 / 17, 1.0], rel=1e-14
    )
    assert [scope["session"]["shares"][k] for k in ("1", "2", "6")] == pytest.approx(
        session_shares, rel=1e-14
    )
    top = scope["row"]["rows"][0]
    assert (top["market"], top["asset_id"], top["squared_error"]) == ("US", "US/A", 9.0)
    assert top["session_rows"] == 2 and top["stratum"] == "liquid"
    # En la vista de un mercado solo cuentan sus filas y sus sesiones.
    assert report["CN"]["row"]["shares"]["1"] == pytest.approx(1 / 3, rel=1e-15)
    assert report["US"]["row"]["shares"]["2"] == pytest.approx(13 / 14, rel=1e-15)


def test_extremes_from_separate_windows_equal_one_window():
    whole = panel(ROWS)
    first, second = panel(ROWS[:2] + ROWS[3:]), panel(ROWS[2:3])
    count = 2
    parts = [
        ls.window_extremes(part, np.zeros(part.rows, dtype=np.int8), count)
        for part in (first, second)
    ]
    joined = extremes(parts, [first, second], "session", extremes=[1, 2], listed_rows=2)
    single = extremes(
        [ls.window_extremes(whole, np.zeros(whole.rows, dtype=np.int8), count)],
        [whole],
        "session",
        extremes=[1, 2],
        listed_rows=2,
    )
    assert joined == single


def test_ties_follow_the_canonical_order_and_strata_are_counted():
    tied = panel([("US", "B", DAY, 2.0), ("US", "A", DAY, 2.0), ("US", "C", DAY, 1.0)])
    codes = np.array([ls.NAMES.index("thin_volume"), ls.NAMES.index("low_price"), 0], np.int8)
    part = ls.window_extremes(tied, codes, 2)
    assert part["US"]["row"]["row_id"] == [tied.row_id[0].as_py(), tied.row_id[1].as_py()]
    report = extremes([part], [tied], "session", ["US"], extremes=[1, 2], listed_rows=2)
    report = report["US"]["row"]
    first = ls.NAMES[int(codes[0])]
    assert report["strata"]["1"][first] == 1
    assert report["strata"]["2"]["thin_volume"] + report["strata"]["2"]["low_price"] == 2


def test_a_zero_total_reports_its_reason():
    zero = panel([("US", "A", DAY, 0.0), ("US", "B", DAY, 0.0)])
    part = ls.window_extremes(zero, np.zeros(2, dtype=np.int8), 2)
    result = extremes([part], [zero], "session", ["US"], extremes=[1], listed_rows=1)["US"]
    assert result["row"]["shares"] is None and result["row"]["reason"] == "El total es cero"


def test_codes_must_follow_the_panel():
    with pytest.raises(ValueError, match="no siguen las filas"):
        ls.window_extremes(panel(ROWS), np.zeros(2, dtype=np.int8), 2)


# Comparación walk-forward


@pytest.fixture(scope="module")
def liquid(tmp_path_factory):
    root = tmp_path_factory.mktemp("liquid")
    study = Study(root / "study")
    study.config[walk.LIQUIDITY_FIELD] = section(min_rows=1, min_sessions=1)
    study.publish()
    edition = study_edition(root / "edition")
    report, _ = walk.evaluate_walk_forward(
        study.config_path, study.sources_path, "US+CN", edition=edition
    )
    return study, edition, report


def by_hand(study, arm, seed, stratum):
    """MAE y MSE por sesión de las filas de un estrato, sin el panel ni la edición.

    El control cero (``seed`` None) predice cero sobre los objetivos comunes, que se leen
    de las predicciones de otro modelo.
    """
    sessions = {}
    for fold in study.folds:
        source = ("ridge", 42) if seed is None else (arm, seed)
        table = study.table(*source, fold, "evaluation").to_pydict()
        for asset, market, moment, y, p in zip(
            table["asset_id"],
            table["market"],
            table["prediction_at"],
            table["target"],
            table["prediction"],
            strict=True,
        ):
            if EXPECTED[asset.split("/")[1]] == stratum:
                error = (0.0 if seed is None else p) - y
                sessions.setdefault((market, moment), []).append(error)
    mae = [np.mean(np.abs(errors)) for errors in sessions.values()]
    mse = [np.mean(np.square(errors)) for errors in sessions.values()]
    return np.mean(mae), np.mean(mse)


def test_every_model_is_scored_on_the_same_rows_of_each_stratum(liquid):
    study, _, report = liquid
    section_report = report[walk.LIQUIDITY_FIELD]
    assert section_report["status"] == "computed" and section_report["recalibrated"] is False
    for window, record in section_report["assignment"].items():
        assert record["rows"] == 42, window
        assert record["strata"] == dict(
            liquid=12, low_price=6, thin_volume=6, low_price_and_thin_volume=6, unclassified=12
        )
        assert record["unclassified"]["asset_not_in_edition"] == 6
        assert record["unclassified"]["no_verified_traded_close"] == 6
    for stratum in ls.NAMES:
        for arm, declared in ARMS.items():
            seeds = declared["seeds"] or [None]
            for seed in seeds:
                label = "deterministic" if seed is None else str(seed)
                cell = section_report["arms"][arm][label][stratum]["overall"]["US+CN"]
                mae, mse = by_hand(study, arm, seed, stratum)
                assert cell["session_mae"] == pytest.approx(mae, rel=1e-12), (arm, stratum)
                assert cell["session_mse"] == pytest.approx(mse, rel=1e-12), (arm, stratum)
    shares = section_report["population"]["liquid"]["overall"]["US+CN"]["row_share"]
    assert shares == pytest.approx(12 / 42)


def test_extremes_cover_every_model_and_keep_every_row(liquid):
    _, _, report = liquid
    extremes_report = report[walk.LIQUIDITY_FIELD]["extremes"]
    assert set(extremes_report) == set(ARMS)
    for arm, seeds in extremes_report.items():
        for label, views in seeds.items():
            assert set(views) == {"US+CN", "US", "CN"}
            for view in views.values():
                for criterion in ("row", "session"):
                    # 100 filas superan las 84 de cada ámbito: la parte es toda la suma.
                    assert view[criterion]["shares"]["100"] == pytest.approx(1.0, rel=1e-12)
                    assert sum(view[criterion]["strata"]["100"].values()) <= 84
                    assert len(view[criterion]["rows"]) == 10
            summary = report["arms"][arm]["seeds"][label]["overall"]["US+CN"]["summary"]
            assert views["US+CN"]["session"]["total"] == pytest.approx(
                summary["point"]["mse"], rel=1e-14
            )
            assert views["US+CN"]["row"]["total"] == pytest.approx(
                summary["row_weighted"]["mse"] * summary["rows"], rel=1e-12
            )


def test_without_the_edition_the_section_is_pending(liquid):
    study, _, _ = liquid
    report, _ = walk.evaluate_walk_forward(study.config_path, study.sources_path, "US+CN")
    assert report[walk.LIQUIDITY_FIELD]["status"] == "pending"
    assert "edición sin ajustar" in report[walk.LIQUIDITY_FIELD]["reason"]


def test_window_aggregates_keep_the_strata_and_their_edition(liquid, tmp_path):
    study, edition, report = liquid
    config = walk.scope_config(walk.load_config(study.config_path), "US+CN")
    sources = walk.load_sources(study.sources_path, config, "US+CN")
    liquidity = walk.liquidity_source(config, edition)
    folder = tmp_path / "aggregates"
    for window in sources["windows"]:
        aggregates.write(folder, config, sources, window, liquidity=liquidity)
        assert aggregates.same(
            aggregates.read(folder, config, sources, window, liquidity=liquidity),
            walk._score_window(sources, config, window, None, liquidity),
        )
        with pytest.raises(ValueError, match="liquidity"):
            aggregates.read(folder, config, sources, window)
    stored, _ = walk.evaluate_walk_forward(
        study.config_path, study.sources_path, "US+CN", aggregates=folder, edition=edition
    )
    assert stored[walk.LIQUIDITY_FIELD] == report[walk.LIQUIDITY_FIELD]


def test_the_matrix_refuses_a_quadratic_metric_without_its_strata(liquid, tmp_path):
    from tests.evaluation.test_comparison_matrix import sources, toy_matrix

    study, edition, _ = liquid
    matrix = toy_matrix(tmp_path, study.config_path)
    with_strata = tmp_path / "with"
    walk.write_walk_forward(
        study.config_path, study.sources_path, "US+CN", with_strata, edition=edition
    )
    manifest = sources(tmp_path, [(walk.REPORT_KIND, with_strata / "comparison.json")])
    report = matrix_module.evaluate(matrix, manifest, "US+CN")
    assert report["quadratic_metrics"]["metrics"] == ["mse"]
    without = tmp_path / "without"
    walk.write_walk_forward(study.config_path, study.sources_path, "US+CN", without)
    manifest = sources(
        tmp_path, [(walk.REPORT_KIND, without / "comparison.json")], name="without.json"
    )
    with pytest.raises(ValueError, match="no publica los estratos de liquidez"):
        matrix_module.evaluate(matrix, manifest, "US+CN")
