"""Calibración en línea de PT2 (#454) dentro de la comparación walk-forward.

Las predicciones son sintéticas y no proceden de ningún modelo ajustado. La madurez de cada
fila se fija en cada prueba sustituyendo la lectura de la vista, salvo en las pruebas con
vistas técnicas, que la leen de sus propias etiquetas. No hay pasos de optimizador.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS
from mars_titan.data.storage import sha256
from mars_titan.evaluation import online_calibration as online
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.models.quantile_head import LEVELS
from tests.evaluation.test_modality_strata import technical, write_technical_study  # noqa: F401
from tests.evaluation.test_walk_forward_comparison import ROOT, Study, rows

JOINT = ROOT / "configs/evaluation/historical-masked-2000-joint-comparison.json"
DECLARATION = ROOT / "configs/evaluation/online-conformal-comparison.json"
SECTION = json.loads(JOINT.read_text())[online.FIELD]
DAY = 86_400_000_000
NOMINALS = [0.8, 0.95]


def provider(study, delay, *, edit=None):
    """Madurez de cada fila de la vista, desordenada a propósito, con un retraso fijo."""

    def view_maturity(path, policy, expected):
        assert policy == HISTORICAL_MASKED and expected == sha256(Path(path))
        fold = next(fold for fold in study.folds if fold["id"] == Path(path).stem)
        tables = {}
        for partition in online.PARTITIONS:
            keys, target = rows(study.markets, int(fold[partition][0][5:7]), fold["id"])
            moments = pa.array([t for _, _, t in keys], pa.timestamp("us", tz="UTC"))
            table = pa.table(
                dict(
                    asset_id=[asset for _, asset, _ in keys],
                    market=[market for market, _, _ in keys],
                    prediction_at=moments,
                    target=target,
                    matures_at=moments.cast(pa.int64()).to_numpy() + delay,
                )
            )
            order = np.random.default_rng(len(keys)).permutation(len(keys))
            table = table.take(pa.array(order))
            tables[partition] = table if edit is None else edit(partition, table)
        return tables

    return view_maturity


def pseudo_years(scores):
    """Noviembre como 2022 y diciembre como 2023: la ventana de selección y la evaluada."""
    months = scores.session_time.astype("datetime64[us]").astype("datetime64[M]").astype(int)
    return np.where(months % 12 + 1 == 11, 2022, 2023)


def declare(study, **changes):
    study.config[online.FIELD] = dict(
        copy.deepcopy(SECTION),
        selection_years=[2022, 2022],
        assessment_years=[2023, 2023],
        periodic_refit=dict(SECTION["periodic_refit"], every_sessions=2, lookback_days=40),
        **changes,
    )
    study.publish()
    return study


def window_results(study, window_id):
    config = walk.resolve_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    config = walk.scope_config(config, study.scope)
    return walk._score_window(sources, config, window_id)[0], config, sources


@pytest.fixture(scope="module")
def studies(tmp_path_factory):
    """El mismo estudio US+CN sin la sección y con ella, con madurez a un día."""
    study = Study(tmp_path_factory.mktemp("online-calibration"))
    plain = study.run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(online, "view_maturity", provider(study, DAY))
        patch.setattr(online, "_session_years", pseudo_years)
        declared = declare(study).run()
    return SimpleNamespace(study=study, plain=plain, declared=declared)


def test_the_campaign_section_follows_the_pt2_declaration():
    declared = json.loads(DECLARATION.read_text())
    assert online.declaration(copy.deepcopy(SECTION)) == SECTION
    arm = declared["arms"]["cqr_online"]
    assert SECTION["method"] == arm["method"]
    assert SECTION["rate_fractions"] == arm["rate_fraction_candidates"]
    assert SECTION["max_pending"] == arm["max_pending"]
    trivial = declared["arms"]["cqr_monthly_refit"]
    assert SECTION["periodic_refit"]["every_sessions"] == trivial["refit_every_sessions"]
    assert SECTION["periodic_refit"]["lookback_days"] == trivial["window_days"]
    assert SECTION["assessment_years"] == declared["metric"]["assessment_years"]
    assert SECTION["selection_years"] == arm["selection_years"]
    assert (
        declared["base_comparison"]
        == "configs/evaluation/historical-masked-2000-joint-comparison.json"
    )


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(extra=1), "contrato"),
        (lambda s: s.update(status="executed"), "contrato"),
        (lambda s: s.update(method="cqr_symmetric_score_v1"), "contrato"),
        (lambda s: s.update(maturity_source="period_end"), "contrato"),
        (lambda s: s.update(rate_fractions=[0.02, 0.005]), "orden creciente"),
        (lambda s: s.update(rate_fractions=[0.0, 0.02]), "orden creciente"),
        (lambda s: s.update(rate_fractions=[1, 2]), "orden creciente"),
        (lambda s: s.update(max_pending=65), "cohortes pendientes"),
        (lambda s: s["periodic_refit"].update(method="annual"), "recalibración periódica"),
        (lambda s: s["periodic_refit"].update(lookback_days=0), "recalibración periódica"),
        (lambda s: s.update(selection_years=[2005, 2014]), "anteriores a los evaluados"),
        (lambda s: s.update(assessment_years=[2014, 2024]), "anteriores a 2024"),
        (lambda s: s["decision"].pop("min_relative_reduction"), "regla de decisión"),
        (lambda s: s["decision"].update(min_relative_reduction=1.0), "regla de decisión"),
    ],
)
def test_declaration_rejects_any_change_to_its_contract(edit, message):
    section = copy.deepcopy(SECTION)
    edit(section)
    with pytest.raises(ValueError, match=message):
        online.declaration(section)


def test_the_comparison_requires_the_masked_policy_for_the_section(tmp_path):
    study = Study(tmp_path, scope="US", policy=STRICT_INPUTS)
    study.config[online.FIELD] = copy.deepcopy(SECTION)
    study.publish()
    with pytest.raises(ValueError, match="política de entradas con máscaras"):
        walk.resolve_config(study.config_path)


def test_the_section_leaves_every_primary_output_unchanged(studies):
    (plain, plain_sessions), (report, sessions) = studies.plain, studies.declared
    assert online.FIELD not in plain and online.FIELD in report
    for key in ("arms", "contrasts", "interval_calibration", "sign_reliability"):
        assert json.dumps(report[key], sort_keys=True) == json.dumps(plain[key], sort_keys=True)
    assert sessions.equals(plain_sessions)
    assert "evaluation/online_calibration.py" in report["analysis_source_sha256"]
    assert "evaluation/online_calibration.py" not in plain["analysis_source_sha256"]


@pytest.mark.parametrize(("delay", "updates"), [(DAY, 2), (DAY + 1, 1)])
def test_a_cohort_matures_before_the_emission_at_its_label_instant(
    studies, monkeypatch, delay, updates
):
    """Tres sesiones por mercado: con madurez a un día justo, las dos primeras cohortes
    maduran antes de emitir la siguiente. Con un microsegundo más, solo la primera."""
    monkeypatch.setattr(online, "view_maturity", provider(studies.study, delay))
    results, config, _ = window_results(studies.study, "fold-000")
    for (arm, _), entry in results.items():
        if config["arms"][arm]["output"] != "quantile_head_v1":
            assert online.FIELD not in entry
            continue
        records = entry[online.FIELD]["records"]
        for fraction in SECTION["rate_fractions"]:
            record = records[online.online_name(fraction)]
            assert record["emissions"] == 3 * 2 and record["updates"] == updates * 2
        # El reajuste de la tercera sesión usa la calibración y las filas ya maduras.
        log = records[online.PERIODIC]["log"]
        assert [item["rows"] for item in log] == [21 + updates * 7] * 2
        assert all(item["used"] for item in log)


def test_labels_that_never_mature_leave_every_variant_equal_to_the_static_cqr(studies, monkeypatch):
    # Con 40 días, la calibración de octubre madura después de la última sesión evaluada de
    # noviembre y la evaluación antes de 2024, que el calibrador no admite.
    monkeypatch.setattr(online, "view_maturity", provider(studies.study, 40 * DAY))
    results, config, _ = window_results(studies.study, "fold-000")
    quantile = [key for key in results if config["arms"][key[0]]["output"] == "quantile_head_v1"]
    assert quantile
    for key in quantile:
        entry = results[key]
        static = entry["calibrated"]
        for name, scores in entry[online.FIELD]["scores"].items():
            assert np.array_equal(scores.coverage, static.coverage), name
            assert np.array_equal(scores.interval_score, static.interval_score), name
            assert np.array_equal(scores.width, static.width), name
        assert entry[online.FIELD]["records"][online.online_name(0.02)]["updates"] == 0


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda part, table: table.slice(1) if part == "evaluation" else table, "mismas filas"),
        (lambda part, table: table.slice(1) if part == "calibration" else table, "mismas filas"),
        (
            lambda part, table: table.set_column(
                3, "target", pa.array(np.asarray(table["target"]) + (part == "evaluation"))
            ),
            "otro objetivo",
        ),
    ],
    ids=["evaluation-row", "calibration-row", "target"],
)
def test_maturity_must_cover_the_same_rows_and_targets_as_the_panel(
    studies, monkeypatch, edit, message
):
    monkeypatch.setattr(online, "view_maturity", provider(studies.study, DAY, edit=edit))
    with pytest.raises(ValueError, match=message):
        window_results(studies.study, "fold-000")


def test_report_chooses_kappa_in_the_selection_years_and_applies_the_rule(studies):
    report = studies.declared[0][online.FIELD]
    assert report["declaration"]["selection_years"] == [2022, 2022]
    assert set(report["arms"]) == {"gru", "titans"}
    allowed = {"keep", "abandon", "inconclusive", "not_estimable"}
    for arm in report["arms"].values():
        selection, chosen = arm["selection"], arm["chosen"]
        assert set(selection) == {online.online_name(v) for v in SECTION["rate_fractions"]}
        assert chosen == min(
            selection, key=lambda name: (selection[name], name != "cqr_online@0.005")
        )
        assert list(arm["views"]) == ["US+CN", "US", "CN"]
        for view in arm["views"].values():
            deviation = view["coverage_deviation"]
            assert list(deviation) == [online.STATIC, chosen, online.PERIODIC]
            relative = view["relative_to_static"][chosen]
            assert relative["estimate"] == pytest.approx(
                deviation[chosen] / deviation[online.STATIC] - 1, rel=1e-12
            )
            assert relative["interval"][0] <= relative["interval"][1]
            assert view["decision"] in allowed
            assert set(view["interval_score"]) == {"0.8", "0.95"}
        totals = arm["totals"]
        # Dos semillas, dos ventanas y dos mercados con tres cohortes cada uno.
        assert totals[chosen]["emissions"] == 2 * 2 * 2 * 3
        assert totals[online.PERIODIC]["refits"] == 2 * 2 * 2


def panel(times, markets, target, quantiles):
    return ForecastPanel.from_columns(
        pa.array([f"{m}/{i}/{t}" for i, (m, t) in enumerate(zip(markets, times, strict=True))]),
        np.array(markets),
        np.array(times, dtype=np.int64),
        np.array(target, dtype=np.float64),
        np.array(quantiles)[:, 2],
        quantiles=np.array(quantiles, dtype=np.float64),
        levels=LEVELS,
        markets=("US", "CN"),
    )


def two_years():
    """Sesiones de US y CN a ambos lados del cambio de año con coberturas conocidas."""
    starts = {2014: 1_419_000_000_000_000, 2015: 1_421_000_000_000_000}  # diciembre y enero
    times, markets, target = [], [], []
    for year, start in starts.items():
        for market in ("US", "CN"):
            for session in range(2 + (year == 2015)):
                for row in range(4 + (market == "CN")):
                    times.append(start + session * DAY + (market == "CN") * 3_600_000_000)
                    markets.append(market)
                    target.append(row - 2.0 + 0.5 * session + (year == 2015))
    base = np.array([-2.5, -1.5, 0.0, 1.5, 2.5])
    narrow = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    return (
        times,
        markets,
        target,
        np.tile(base, (len(target), 1)),
        np.tile(narrow, (len(target), 1)),
    )


def by_hand(times, markets, target, quantiles, mask_years):
    """Desviación media por año y mercado calculada fila a fila."""
    cells = {}
    for t, m, y, q in zip(times, markets, target, quantiles, strict=True):
        year = int(np.datetime64(t, "us").astype("datetime64[Y]").astype(int)) + 1970
        if year not in mask_years:
            continue
        cell = cells.setdefault((year, m), dict(rows=0, covered=[0, 0]))
        cell["rows"] += 1
        cell["covered"][0] += q[1] <= y <= q[3]
        cell["covered"][1] += q[0] <= y <= q[4]
    errors = [
        abs(cell["covered"][j] / cell["rows"] - nominal)
        for cell in cells.values()
        for j, nominal in enumerate(NOMINALS)
    ]
    return sum(errors) / len(errors)


def test_coverage_deviation_matches_the_row_by_row_computation():
    times, markets, target, wide, narrow = two_years()
    scores = [score_sessions(panel(times, markets, target, q)) for q in (wide, narrow)]
    coverage = np.stack([online._coverages(item, NOMINALS) for item in scores], axis=-1)
    years = online._session_years(scores[0])
    for chosen in ({2014}, {2015}, {2014, 2015}):
        mask = np.isin(years, list(chosen))
        values = online.coverage_deviation(scores[0], coverage, mask, NOMINALS)
        for index, quantiles in enumerate((wide, narrow)):
            assert values[index] == pytest.approx(
                by_hand(times, markets, target, quantiles, chosen), rel=1e-12
            )


def test_bootstrap_with_each_day_once_gives_the_estimate(monkeypatch):
    times, markets, target, wide, narrow = two_years()
    scores = [score_sessions(panel(times, markets, target, q)) for q in (wide, narrow)]
    coverage = np.stack([online._coverages(item, NOMINALS) for item in scores], axis=-1)
    mask = np.ones(len(scores[0].samples), dtype=bool)

    def ones(rng, replicates, periods, block_length):
        return np.ones((replicates, periods), dtype=np.int64)

    monkeypatch.setattr(online, "circular_block_counts", ones)
    comparison = dict(seed=3, replicates=4, block_length=2)
    draws = online._relative_draws(scores[0], coverage, mask, NOMINALS, comparison)
    estimate = online.coverage_deviation(scores[0], coverage, mask, NOMINALS)
    assert draws.shape == (4, 2)
    assert np.allclose(draws, estimate[None], rtol=1e-12, atol=0)


def contrast(name, percent):
    return dict(contrasts=[dict(name=name, relative_improvement_percent=percent, interval=[-1, 1])])


@pytest.mark.parametrize(
    ("estimate", "interval", "percents", "expected"),
    [
        (-0.4, [-0.6, -0.1], [0.5, -0.5], "keep"),
        (-0.4, [-0.6, 0.1], [0.5, 0.5], "inconclusive"),
        (-0.1, [-0.2, -0.05], [0.5, 0.5], "inconclusive"),
        (-0.4, [-0.6, -0.1], [0.5, -1.5], "abandon"),
        (0.0, [-0.1, 0.1], [0.5, 0.5], "abandon"),
        (None, None, [0.5, 0.5], "not_estimable"),
    ],
)
def test_the_decision_applies_the_declared_thresholds(estimate, interval, percents, expected):
    chosen = "cqr_online@0.005"
    primary = {chosen: dict(estimate=estimate, interval=interval)}
    scores = {
        f"{n:g}": contrast(f"{chosen}-{online.STATIC}", p)
        for n, p in zip(NOMINALS, percents, strict=True)
    }
    assert online._decision(SECTION, primary, scores, chosen) == expected


def test_aggregates_keep_the_online_variants_bit_for_bit(studies, tmp_path, monkeypatch):
    monkeypatch.setattr(online, "view_maturity", provider(studies.study, DAY))
    config = walk.resolve_config(studies.study.config_path)
    sources = walk.load_sources(studies.study.sources_path, config, studies.study.scope)
    config = walk.scope_config(config, studies.study.scope)
    record = window_aggregates.write(tmp_path, config, sources, "fold-000")
    stored = window_aggregates.read(tmp_path, config, sources, "fold-000")
    assert window_aggregates.same(stored, walk._score_window(sources, config, "fold-000"))
    entry = stored[0]["gru", 42]
    assert record["sha256"] and set(entry[online.FIELD]["scores"]) == set(online.variants(SECTION))
    identity = window_aggregates.identity(config, sources, "fold-000")
    assert {"evaluation/online_calibration.py", "calibration/periodic_conformal.py"} <= set(
        identity["code"]
    )


def labels_maturity(view, partition):
    """Madurez de cada fila de un tramo leída directamente de las etiquetas de la vista."""
    manifest = json.loads(view.read_text())
    result = {}
    for asset in manifest["assets"]:
        path = Path(manifest["roots"]["labels"]) / "US" / asset["symbol"] / "labels.parquet"
        table = pq.read_table(path).to_pydict()
        for moment, available, part in zip(
            table["prediction_at"], table["target_available_at"], table["partition"], strict=True
        ):
            if part == partition:
                result[f"US/{asset['symbol']}", moment] = available
    return result


def test_view_maturity_reads_the_labels_of_each_row(technical):  # noqa: F811
    for fold in technical.folds:
        view = technical.root / "views" / fold["id"] / "manifest.json"
        partitions = ("validation", *online.PARTITIONS)
        tables = online.view_maturity(view, HISTORICAL_MASKED, sha256(view), partitions)
        assert set(tables) == set(partitions)
        for partition in partitions:
            table = tables[partition]
            expected = labels_maturity(view, partition)
            assert table.num_rows == len(expected) > 0
            observed = {
                (asset, moment): matures
                for asset, moment, matures in zip(
                    table["asset_id"].to_pylist(),
                    table["prediction_at"].to_pylist(),
                    table["matures_at"].to_pylist(),
                    strict=True,
                )
            }
            assert observed == {
                key: int(value.timestamp() * 1_000_000) for key, value in expected.items()
            }
    with pytest.raises(ValueError, match="cambió antes de leer"):
        online.view_maturity(view, HISTORICAL_MASKED, "0" * 64)


def test_technical_views_drive_the_online_updates_with_their_own_labels(technical, tmp_path):  # noqa: F811
    config_path, sources_path, _ = write_technical_study(tmp_path / "study", technical)
    config = json.loads(config_path.read_text())
    config[online.FIELD] = dict(
        copy.deepcopy(SECTION),
        selection_years=[2022, 2022],
        assessment_years=[2023, 2023],
        periodic_refit=dict(SECTION["periodic_refit"], every_sessions=1, lookback_days=60),
    )
    config_path.write_text(json.dumps(config))
    loaded = walk.resolve_config(config_path)
    sources = walk.load_sources(sources_path, loaded, "US")
    loaded = walk.scope_config(loaded, "US")
    for window_id in sources["windows"]:
        results = walk._score_window(sources, loaded, window_id)[0]
        records = results["gru", 42][online.FIELD]["records"]
        sessions = len({moment for *_, moment, _ in _view_keys(technical, window_id)})
        assert records[online.online_name(0.02)]["emissions"] == sessions
        assert 0 < records[online.online_name(0.02)]["updates"] < sessions
    report, _ = walk.evaluate_walk_forward(config_path, sources_path, "US")
    # Los datos técnicos son de 2023: sin años de selección, el informe lo dice.
    assert (
        report[online.FIELD]["arms"]["gru"]["reason"] == "No hay sesiones en los años de selección"
    )


def _view_keys(study, window_id):
    from tests.evaluation.test_modality_strata import view_rows

    view = study.root / "views" / window_id / "manifest.json"
    return view_rows(view, "evaluation")
