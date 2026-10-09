"""Estratos por presencia de noticias y fundamentales en la comparación walk-forward.

Las predicciones y los patrones de presencia se fijan en cada prueba. No proceden de
ningún modelo ajustado ni de la edición real, y no se ejecuta ningún paso de optimizador.
Las pruebas de integración leen la presencia de vistas técnicas preparadas con el corpus
de fixture.
"""

import copy
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED, STRICT_INPUTS
from mars_titan.data.storage import sha256
from mars_titan.evaluation import modality_strata as strata
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.splits import build_folds
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.evaluation.test_comparison_sources import masked_protocol, save
from tests.evaluation.test_walk_forward_comparison import CONFIG, OFFSETS, Study, rows
from tests.training.historical_temporal_fixture import historical_temporal_fixture

SECTION = json.loads(CONFIG.read_text())["modality_strata"]
NAMES = list(strata.STRATA)
# Estados Unidos: dos activos con todo, uno solo con noticias, dos solo con fundamentales
# y dos sin ninguno. China: noticias en A y fundamentales solo en D el primer día de cada mes.
US = dict(A=(1, 1), B=(1, 1), C=(1, 0), D=(0, 1), E=(0, 1), F=(0, 0), G=(0, 0))


def by_asset(market, asset, moment):
    letter = asset[-1]
    if market == "US":
        return tuple(bool(value) for value in US[letter])
    return letter == "A", letter == "D" and moment.day == 6


def stratum_of(rule, market, asset, moment):
    news, fundamentals = rule(market, asset, moment)
    return next(
        name
        for name, pattern in strata.STRATA.items()
        if (pattern["news"], pattern["fundamentals"]) == (news, fundamentals)
    )


def provider(study, rule, *, macro=None, edit=None):
    """Sustituir la lectura de la vista por filas conocidas, desordenadas a propósito."""

    def view_presence(path, policy, expected):
        assert policy == HISTORICAL_MASKED and expected == sha256(Path(path))
        fold = next(fold for fold in study.folds if fold["id"] == Path(path).stem)
        keys, target = rows(study.markets, int(fold["evaluation"][0][5:7]), fold["id"])
        bits = np.array(
            [
                [
                    True,
                    rule(m, a, t)[0],
                    True,
                    rule(m, a, t)[1],
                    True if macro is None else macro(fold["id"], m, a, t),
                ]
                for m, a, t in keys
            ]
        )
        table = pa.table(
            dict(
                asset_id=[a for _, a, _ in keys],
                market=[m for m, _, _ in keys],
                prediction_at=pa.array([t for _, _, t in keys], pa.timestamp("us", tz="UTC")),
                target=target,
            )
        )
        order = np.random.default_rng(zlib.crc32(fold["id"].encode())).permutation(len(keys))
        table, bits = table.take(pa.array(order)), bits[order]
        if edit is not None:
            table, bits = edit(fold["id"], table, bits)
        return table, bits

    return view_presence


def declare(study, **changes):
    study.config["schema_version"] = 2
    study.config["modality_strata"] = dict(copy.deepcopy(SECTION), **changes)
    study.publish()
    return study


def run_with(study, rule, monkeypatch, **options):
    monkeypatch.setattr(strata, "view_presence", provider(study, rule, **options))
    return study.run()


def by_hand(study, rule, arm, seed, stratum, *, market=None, windows=None):
    """MAE por sesión del estrato calculado sin panel: media por sesión y media simple."""
    sessions = {}
    for fold in study.folds:
        if windows is not None and fold["id"] not in windows:
            continue
        values = study.table("ridge" if arm == "zero" else arm, seed, fold, "evaluation")
        values = values.to_pydict()
        for m, a, t, y, p in zip(
            values["market"],
            values["asset_id"],
            values["prediction_at"],
            values["target"],
            values["prediction"],
            strict=True,
        ):
            if stratum_of(rule, m, a, t) != stratum or (market and m != market):
                continue
            sessions.setdefault((m, t), []).append(abs((0.0 if arm == "zero" else p) - y))
    means = [sum(errors) / len(errors) for errors in sessions.values()]
    return sum(means) / len(means), len(sessions), sum(map(len, sessions.values()))


@pytest.fixture(scope="module")
def joint(tmp_path_factory):
    """Mismo estudio US+CN sin estratos (versión 1) y con ellos (versión 2)."""
    calls = []
    original = walk.cqr.fit_conformal_quantiles

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(walk.cqr, "fit_conformal_quantiles", counted)
        study = Study(tmp_path_factory.mktemp("joint-strata"))
        first = study.run()
        primary_fits = len(calls)
        # Las mismas fuentes y predicciones, ahora con la configuración versión 2.
        second = run_with(declare(study, min_rows=6, min_sessions=3), by_asset, patch)
    return SimpleNamespace(
        plain=first,
        study=study,
        report=second[0],
        sessions=second[1],
        fits=(primary_fits, len(calls) - primary_fits),
    )


def test_declared_configuration_is_a_secondary_analysis_fixed_before_results():
    config = walk.load_config(CONFIG)
    section = config["modality_strata"]
    assert config["schema_version"] == 2
    assert section["status"] == "secondary_descriptive" and section["declared_at"] == "2026-10-09"
    assert section["use"] == strata.USE and "not_for_model_selection" in section["use"]
    assert section["strata"] == strata.STRATA and section["focus"] == "news_and_fundamentals"
    assert section["always_present"] == ["prices", "charts", "macro"]
    assert section["metrics"] == ["mae"] and section["recalibrate"] is False
    assert (section["min_rows"], section["min_sessions"]) == (1000, 50)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda s: s.pop("min_sessions"), "exactamente sus campos"),
        (lambda s: s.update(extra=1), "exactamente sus campos"),
        (lambda s: s.update(status="primary"), "secundario"),
        (lambda s: s.update(use="model_selection"), "secundario"),
        (lambda s: s.update(declared_at="2026-13-09"), "secundario"),
        (lambda s: s.update(presence_source="predictions"), "secundario"),
        (lambda s: s.update(always_present=["prices", "charts"]), "secundario"),
        (lambda s: s.update(multiplicity="none"), "secundario"),
        (
            lambda s: s["strata"].update(news_only=dict(news=False, fundamentals=True)),
            "cuatro patrones",
        ),
        (lambda s: s["strata"].pop("neither"), "cuatro patrones"),
        (lambda s: s.update(focus="all"), "cuatro patrones"),
        (lambda s: s.update(metrics=["mse"]), "empezar por el MAE"),
        (lambda s: s.update(metrics=["mae", "mae"]), "empezar por el MAE"),
        (lambda s: s.update(metrics=["mae", "coverage"]), "empezar por el MAE"),
        (lambda s: s.update(recalibrate=True), "volver a calibrar"),
        (lambda s: s.update(min_rows=0), "min_rows"),
        (lambda s: s.update(min_rows=True), "min_rows"),
        (lambda s: s.update(min_sessions=2.5), "min_sessions"),
    ],
)
def test_declaration_rejects_any_change_to_its_contract(edit, message):
    section = copy.deepcopy(SECTION)
    edit(section)
    with pytest.raises(ValueError, match=message):
        strata.declaration(section, walk.SERIES_METRICS)


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda c: c.pop("modality_strata"), "contrato"),
        (lambda c: c.update(schema_version=1), "contrato"),
        (lambda c: c.update(schema_version=True), "contrato"),
        (lambda c: c.update(input_policy=STRICT_INPUTS), "máscaras"),
    ],
)
def test_configuration_versions_require_their_own_fields(tmp_path, edit, message):
    study = declare(Study(tmp_path, scope="US"))
    edit(study.config)
    study.publish()
    with pytest.raises(ValueError, match=message):
        walk.load_config(study.config_path)


def test_codes_follow_the_declared_patterns_and_incomplete_rows_are_counted():
    bits = np.array(
        [
            [True, True, True, True, True],
            [True, True, True, False, True],
            [True, False, True, True, True],
            [True, False, True, False, True],
            [True, True, True, False, False],
        ]
    )
    assert [NAMES[code] for code in strata.codes(bits)] == [*NAMES, "news_only"]
    for code, pattern in zip(strata.codes(bits)[:4], strata.STRATA.values(), strict=True):
        assert (bool(bits[code, 1]), bool(bits[code, 3])) == (
            pattern["news"],
            pattern["fundamentals"],
        )
    assert strata.incomplete_rows(bits) == 1
    assert strata.incomplete_rows(bits[:4]) == 0


@pytest.mark.parametrize(
    "rows,sessions,estimable,fragment",
    [
        (0, 0, False, "no tiene filas"),
        (5, 3, False, "5 filas y 3 sesiones, por debajo del mínimo declarado de 6 filas y 3"),
        (6, 2, False, "6 filas y 2 sesiones"),
        (6, 3, True, None),
        (60, 30, True, None),
    ],
)
def test_estimability_uses_the_declared_thresholds_inclusively(rows, sessions, estimable, fragment):
    flag, reason = strata.estimability(rows, sessions, min_rows=6, min_sessions=3)
    assert flag is estimable
    assert (reason is None) if fragment is None else (fragment in reason)


def test_adjusted_confidence_divides_the_error_by_the_declared_cells():
    assert strata.adjusted_confidence(0.95, 12) == pytest.approx(1 - 0.05 / 12, abs=1e-15)
    assert strata.adjusted_confidence(0.95, 1) == pytest.approx(0.95, abs=1e-15)
    with pytest.raises(ValueError):
        strata.adjusted_confidence(0.95, 0)


def small_panel():
    """Dos sesiones US con cinco y tres filas, objetivos y cuantiles conocidos."""
    ids = [f"US/A{i}/1" for i in range(5)] + [f"US/A{i}/2" for i in range(3)]
    target = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.2, 0.1])
    prediction = np.array([0.0, 0.1, 0.1, 0.2, 0.2, -0.1, 0.0, 0.3])
    quantiles = prediction[:, None] + 0.25 * OFFSETS
    times = np.array([1] * 5 + [2] * 3, dtype=np.int64) * 86_400_000_000
    panel = ForecastPanel.from_columns(
        ids, ["US"] * 8, times, target, prediction, quantiles=quantiles, levels=LEVELS
    )
    codes = np.array([0, 0, 1, 3, 3, 1, 3, 3], dtype=np.int8)
    return panel, codes


def test_score_strata_partitions_every_session_and_keeps_the_calibrated_quantiles():
    panel, codes = small_panel()
    calibrated = panel.quantiles + np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    scored = strata.score_strata(panel, codes, rank_ic_min_assets=3, calibrated=calibrated)
    assert scored["fundamentals_only"] == dict(raw=None, calibrated=None)
    whole = walk.score_sessions(panel, rank_ic_min_assets=3)
    errors = np.abs(panel.prediction - panel.target)
    # Por sesión, el MAE completo es la media de los estratos ponderada por sus filas.
    for session, start in enumerate(panel.session_starts):
        members = panel.session == session
        total = 0.0
        for entry in scored.values():
            if entry["raw"] is None:
                continue
            index = np.flatnonzero(entry["raw"].session_time == panel.session_time[session])
            if index.size:
                total += entry["raw"].samples[index[0]] * entry["raw"].mae[index[0]]
        assert total == pytest.approx(whole.samples[session] * whole.mae[session], abs=1e-15)
        assert whole.mae[session] == pytest.approx(errors[members].mean(), abs=1e-15)
        assert start == np.flatnonzero(members)[0]
    # Con filas, el MAE por filas completo es la combinación con pesos N_k / N.
    row_weighted = sum(
        float(np.dot(entry["raw"].samples, entry["raw"].mae))
        for entry in scored.values()
        if entry["raw"] is not None
    )
    assert row_weighted / panel.rows == pytest.approx(errors.mean(), abs=1e-15)
    neither = scored["neither"]
    assert list(neither["raw"].samples) == [2, 2]
    rows = codes == 3
    low, high = calibrated[rows, 1], calibrated[rows, 3]
    covered = (low <= panel.target[rows]) & (panel.target[rows] <= high)
    assert list(neither["calibrated"].coverage[:, 0]) == [covered[:2].mean(), covered[2:].mean()]
    assert list(neither["raw"].mae) == list(neither["calibrated"].mae)


def test_score_strata_rejects_codes_that_do_not_follow_the_panel():
    panel, codes = small_panel()
    with pytest.raises(ValueError, match="no siguen las filas"):
        strata.score_strata(panel, codes[:-1], rank_ic_min_assets=3)


def test_population_counts_each_stratum_by_market_window_and_scope(joint):
    population = joint.report["modality_strata"]["population"]
    expected = {
        "news_and_fundamentals": {"US+CN": (12, 6), "US": (12, 6), "CN": (0, 0)},
        "news_only": {"US+CN": (12, 12), "US": (6, 6), "CN": (6, 6)},
        "fundamentals_only": {"US+CN": (14, 8), "US": (12, 6), "CN": (2, 2)},
        "neither": {"US+CN": (46, 12), "US": (12, 6), "CN": (34, 6)},
    }
    totals = {"US+CN": 84, "US": 42, "CN": 42}
    for name, views in expected.items():
        assert population[name]["pattern"] == strata.STRATA[name]
        for view, (count, sessions) in views.items():
            cell = population[name]["overall"][view]
            assert (cell["rows"], cell["sessions"]) == (count, sessions), (name, view)
            assert cell["row_share"] == count / totals[view]
            assert cell["estimable"] is (count >= 6 and sessions >= 3)
    assert sum(population[name]["overall"]["US+CN"]["rows"] for name in NAMES) == 84
    for window in ("fold-000", "fold-001"):
        cells = {name: population[name]["windows"][window] for name in NAMES}
        assert (cells["news_only"]["US"]["rows"], cells["news_only"]["US"]["sessions"]) == (3, 3)
        assert cells["news_only"]["US"]["estimable"] is False
        assert cells["neither"]["US"]["estimable"] is True
        assert cells["fundamentals_only"]["CN"]["rows"] == 1
        assert sum(cells[name]["US"]["rows"] for name in NAMES) == 21


def test_stratum_session_mae_matches_the_hand_computation(joint):
    arms = joint.report["modality_strata"]["arms"]
    for arm, seed in (("gru", 42), ("titans", 43), ("ridge", 42)):
        for name in NAMES:
            for view, market in (("US+CN", None), ("US", "US"), ("CN", "CN")):
                cell = arms[arm][str(seed)][name]["overall"][view]
                population = joint.report["modality_strata"]["population"][name]["overall"]
                if not population[view]["estimable"]:
                    assert cell["session_mae"] is None and cell["reason"]
                    continue
                expected, _, _ = by_hand(joint.study, by_asset, arm, seed, name, market=market)
                assert cell["session_mae"] == pytest.approx(expected, rel=1e-12), (arm, name)
    zero = arms["zero"]["deterministic"]["neither"]["overall"]["CN"]["session_mae"]
    expected, _, _ = by_hand(joint.study, by_asset, "zero", 42, "neither", market="CN")
    assert zero == pytest.approx(expected, rel=1e-12)
    window = arms["gru"]["42"]["neither"]["windows"]["fold-001"]
    expected, _, _ = by_hand(
        joint.study, by_asset, "gru", 42, "neither", market="US", windows={"fold-001"}
    )
    assert window["US"] == pytest.approx(expected, rel=1e-12)
    assert arms["gru"]["42"]["news_only"]["windows"]["fold-000"]["US"] is None


def test_small_or_empty_strata_are_declared_not_estimable_and_never_omitted(joint):
    section = joint.report["modality_strata"]
    empty = section["population"]["news_and_fundamentals"]["overall"]["CN"]
    assert empty["estimable"] is False and "no tiene filas" in empty["reason"]
    small = section["population"]["fundamentals_only"]["overall"]["CN"]
    assert "2 filas y 2 sesiones, por debajo del mínimo declarado de 6 filas y 3" in small["reason"]
    for name, reason in (
        ("news_and_fundamentals", empty["reason"]),
        ("fundamentals_only", small["reason"]),
    ):
        cell = section["arms"]["gru"]["43"][name]["overall"]["CN"]
        assert cell == dict(
            session_mae=None, reason=reason, calibrated_intervals=None, calibrated_reason=reason
        )
        assert section["contrasts"][name]["CN"] == dict(reason=reason)
        assert section["interval_calibration"][name]["gru"]["CN"] == dict(reason=reason)
        assert set(section["contrasts"][name]) == {"US+CN", "US", "CN"}
    assert set(section["arms"]["ridge"]["42"]) == set(NAMES)


def test_contrasts_use_the_declared_families_with_bonferroni_over_cells(joint):
    section = joint.report["modality_strata"]
    assert section["status"] == "computed" and section["recalibrated"] is False
    assert section["declaration"]["focus"] == "news_and_fundamentals"
    confidence = 1 - 0.05 / 12
    assert section["multiplicity"]["cells"] == 12
    assert section["multiplicity"]["confidence"] == pytest.approx(confidence, abs=1e-15)
    family = section["contrasts"]["news_and_fundamentals"]["US+CN"]["references_vs_zero"]
    assert list(family) == ["mae"]
    mae = family["mae"]
    assert mae["confidence"] == pytest.approx(confidence, abs=1e-15)
    assert mae["resampling"]["block_length"] == 1 and mae["resampling"]["seed"] == 7
    row = next(row for row in mae["contrasts"] if row["name"] == "gru-zero")
    gru = np.mean(
        [by_hand(joint.study, by_asset, "gru", s, "news_and_fundamentals")[0] for s in (42, 43)]
    )
    zero = by_hand(joint.study, by_asset, "zero", 42, "news_and_fundamentals")[0]
    assert row["estimate"] == pytest.approx(gru - zero, rel=1e-12)
    assert row["simultaneous_interval"] is not None
    calibration = section["interval_calibration"]["neither"]["titans"]["US+CN"]["0.8"]
    assert calibration["comparison"]["confidence"] == pytest.approx(confidence, abs=1e-15)


def test_strata_do_not_refit_the_common_calibrator(joint):
    assert joint.fits == (8, 8)
    report, study = joint.report, joint.study
    for seed in (42, 43):
        sessions = {}
        for fold in study.folds:
            record = report["arms"]["gru"]["seeds"][str(seed)]["windows"][fold["id"]]
            table = study.table("gru", seed, fold, "evaluation").to_pydict()
            quantiles = np.column_stack([table[column] for column in QUANTILE_COLUMNS])
            calibrated, _ = walk.cqr.apply_conformal_quantiles(
                record["calibrator"]["record"], quantiles, np.asarray(table["market"])
            )
            for i, (m, a, t, y) in enumerate(
                zip(
                    table["market"],
                    table["asset_id"],
                    table["prediction_at"],
                    table["target"],
                    strict=True,
                )
            ):
                if stratum_of(by_asset, m, a, t) == "neither":
                    covered = calibrated[i, 1] <= y <= calibrated[i, 3]
                    sessions.setdefault((m, t), []).append(covered)
        expected = np.mean([np.mean(values) for values in sessions.values()])
        cell = report["modality_strata"]["arms"]["gru"][str(seed)]["neither"]["overall"]["US+CN"]
        interval = cell["calibrated_intervals"][0]
        assert interval["nominal"] == 0.8
        assert interval["coverage"] == pytest.approx(expected, rel=1e-12)


def test_primary_outputs_are_identical_with_or_without_the_declared_strata(joint):
    plain, _ = joint.plain
    report = joint.report
    assert "modality_strata" not in plain
    assert set(plain["analysis_source_sha256"]) < set(report["analysis_source_sha256"])
    for name, digest in plain["analysis_source_sha256"].items():
        assert report["analysis_source_sha256"][name] == digest
    ignored = {"created_at_utc", "resources", "configuration", "analysis_source_sha256"}
    for key in set(plain) | set(report):
        if key in ignored or key == "modality_strata":
            continue
        assert plain[key] == report[key], key
    assert plain["configuration"]["name"] == report["configuration"]["name"]
    assert joint.plain[1].equals(joint.sessions)


def test_session_constant_presence_recombines_the_primary_session_mae(tmp_path, monkeypatch):
    # Cada sesión entera pertenece a un único estrato, así que el MAE principal es la media
    # de los estratos ponderada por sus sesiones.
    def by_session(market, asset, moment):
        return [(True, True), (True, False), (False, True), (False, False)][
            (moment.day - 6 + (market == "CN")) % 4
        ]

    study = declare(Study(tmp_path), min_rows=1, min_sessions=1)
    report, _ = run_with(study, by_session, monkeypatch)
    section = report["modality_strata"]
    for arm, seed in (("gru", "42"), ("ridge", "42"), ("zero", "deterministic")):
        for view in ("US+CN", "US", "CN"):
            primary = report["arms"][arm]["seeds"][seed]["overall"][view]["summary"]["point"]
            total = sum(section["population"][name]["overall"][view]["sessions"] for name in NAMES)
            combined = sum(
                section["population"][name]["overall"][view]["sessions"]
                * section["arms"][arm][seed][name]["overall"][view]["session_mae"]
                for name in NAMES
                if section["population"][name]["overall"][view]["sessions"]
            )
            assert combined / total == pytest.approx(primary["mae"], rel=1e-12), (arm, view)


@pytest.mark.parametrize(
    "edit,message",
    [
        (
            lambda fold, table, bits: (
                (table.slice(1), bits[1:])
                if fold == "fold-001"
                else (
                    table,
                    bits,
                )
            ),
            "0 filas solo en este brazo, 1 filas solo en la referencia y 0 filas",
        ),
        (
            lambda fold, table, bits: (
                table.set_column(3, "target", pa.array(table["target"].to_numpy() + 1)),
                bits,
            ),
            "0 filas solo en la referencia y 42 filas con otro objetivo",
        ),
    ],
    ids=["missing_row", "other_target"],
)
def test_presence_must_cover_exactly_the_evaluated_rows(tmp_path, monkeypatch, edit, message):
    study = declare(Study(tmp_path))
    with pytest.raises(ValueError, match="La presencia de la vista de fold-00") as error:
        run_with(study, by_asset, monkeypatch, edit=edit)
    assert message in str(error.value)


def test_rows_without_macro_leave_the_section_not_estimable_and_the_primary_intact(
    tmp_path, monkeypatch, joint
):
    def macro(fold, market, asset, moment):
        return not (fold == "fold-001" and market == "CN" and asset == "CN/G" and moment.day < 8)

    study = declare(Study(tmp_path), min_rows=6, min_sessions=3)
    report, sessions = run_with(study, by_asset, monkeypatch, macro=macro)
    section = report["modality_strata"]
    assert section["status"] == "not_estimable"
    assert section["reason"].startswith("2 filas de evaluación no tienen precios, gráficos y macro")
    assert section["presence"] == {
        "fold-000": dict(rows=42, incomplete=0),
        "fold-001": dict(rows=42, incomplete=2),
    }
    assert set(section) == {"declaration", "status", "reason", "presence"}
    assert report["arms"] == joint.plain[0]["arms"]
    assert report["contrasts"] == joint.plain[0]["contrasts"]
    assert sessions.equals(joint.plain[1])


def test_single_market_scope_corrects_over_four_cells(tmp_path, monkeypatch):
    study = declare(Study(tmp_path, scope="US"), min_rows=6, min_sessions=3)
    report, _ = run_with(study, by_asset, monkeypatch)
    section = report["modality_strata"]
    assert section["multiplicity"]["cells"] == 4
    assert section["multiplicity"]["confidence"] == pytest.approx(1 - 0.05 / 4, abs=1e-15)
    assert set(section["population"]["neither"]["overall"]) == {"US"}
    assert set(section["population"]["neither"]["windows"]["fold-000"]) == {"US"}
    expected, sessions, count = by_hand(study, by_asset, "gru", 42, "neither")
    assert section["arms"]["gru"]["42"]["neither"]["overall"]["US"]["session_mae"] == (
        pytest.approx(expected, rel=1e-12)
    )
    assert section["population"]["neither"]["overall"]["US"]["rows"] == count == 12


# Vistas técnicas reales: siete activos US y sesiones en todos los tramos de dos ventanas.
DAYS = (
    "2022-06-15",
    "2023-03-15",
    "2023-06-14",
    "2023-08-15",
    "2023-09-13",
    "2023-09-14",
    "2023-10-11",
    "2023-10-12",
    "2023-10-13",
    "2023-11-13",
    "2023-11-14",
    "2023-11-15",
    "2023-12-12",
    "2023-12-13",
    "2023-12-14",
)


def fixture_presence(market, symbol, row):
    """Patrón conocido por activo y fila, con el macro siempre presente."""
    number = int(symbol[1:])
    return (number + row) % 2 == 0, (number + row) % 3 != 0, True


@pytest.fixture(scope="module")
def technical(tmp_path_factory):
    root = tmp_path_factory.mktemp("strata-views")
    data = historical_temporal_fixture(
        root / "data", ("US",), {"US": DAYS}, assets=7, presence=fixture_presence
    )
    protocol = masked_protocol("US", "2023-09-01")
    save(root / "config" / "protocol-US.json", protocol)
    prepare_temporal_corpus(
        data.parent,
        root / "config" / "protocol-US.json",
        None,
        None,
        root / "views",
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    return SimpleNamespace(root=root, protocol=protocol, folds=build_folds(protocol))


def view_rows(view, partition):
    """Filas de un tramo leídas directamente de las etiquetas de la vista."""
    manifest = json.loads(view.read_text())
    keys = []
    for asset in manifest["assets"]:
        path = Path(manifest["roots"]["labels"]) / "US" / asset["symbol"] / "labels.parquet"
        table = pq.read_table(path).to_pydict()
        for row, moment, target, part in zip(
            table["sample_row"],
            table["prediction_at"],
            table["target"],
            table["partition"],
            strict=True,
        ):
            if part == partition:
                keys.append((asset["symbol"], row, moment, target))
    return keys


def test_view_presence_reads_the_samples_of_each_evaluation_row(technical):
    for fold in technical.folds:
        view = technical.root / "views" / fold["id"] / "manifest.json"
        table, bits = strata.view_presence(view, HISTORICAL_MASKED, sha256(view))
        expected = {
            (f"US/{symbol}", moment): (target, fixture_presence("US", symbol, row))
            for symbol, row, moment, target in view_rows(view, "evaluation")
        }
        assert table.num_rows == len(expected) == 21
        observed = {}
        for index, (asset, moment, target) in enumerate(
            zip(
                table["asset_id"].to_pylist(),
                table["prediction_at"].to_pylist(),
                table["target"].to_pylist(),
                strict=True,
            )
        ):
            observed[asset, moment] = (target, (bool(bits[index, 1]), bool(bits[index, 3]), True))
            assert bits[index, [0, 2, 4]].all()
        assert observed == expected
        assert set(table["market"].to_pylist()) == {"US"}
    with pytest.raises(ValueError, match="cambió antes de leer"):
        strata.view_presence(view, HISTORICAL_MASKED, "0" * 64)
    with pytest.raises(ValueError):
        strata.view_presence(view, STRICT_INPUTS, sha256(view))


def write_technical_study(root, technical):
    """Configuración versión 2 y predicciones sintéticas sobre las filas de las vistas."""
    arms = dict(
        zero=dict(family="control", output="zero_control", seeds=[]),
        ridge=dict(family="reference", output="point", seeds=[42]),
        gru=dict(family="reference", output="quantile_head_v1", seeds=[42, 43]),
    )
    config = json.loads(CONFIG.read_text())
    config.update(
        name="technical-strata",
        scopes={"US": dict(protocols={"US": "protocol-US.json"}, windows="all")},
        arms=arms,
        metrics=dict(
            primary="mae",
            market_weighting="session",
            rank_ic_min_assets=3,
            quantile_head="quantile_head_v1",
        ),
        modality_strata=dict(config["modality_strata"], min_rows=3, min_sessions=2),
    )
    config["calibration"]["min_rows"] = 10
    config["comparison"].update(
        block_length=1,
        sensitivity_block_lengths=[],
        replicates=50,
        seed=7,
        families=dict(
            references_vs_zero=dict(kind="delta", base="zero", variants=["ridge", "gru"])
        ),
    )
    config_path = technical.root / "config" / f"{root.name}.json"
    save(config_path, config)
    windows, entries, predictions = {}, {}, {}
    for fold in technical.folds:
        view = technical.root / "views" / fold["id"] / "manifest.json"
        windows[fold["id"]] = dict(view=dict(path=str(view), sha256=sha256(view)))
        for arm, declared in arms.items():
            for seed in declared["seeds"]:
                entry = dict(input_policy=HISTORICAL_MASKED, view_sha256=sha256(view))
                parts = ["evaluation"] + (["calibration"] if arm == "gru" else [])
                for part in parts:
                    keys = view_rows(view, part)
                    target = np.array([target for *_, target in keys])
                    rng = np.random.default_rng(
                        zlib.crc32(f"{arm}{seed}{fold['id']}{part}".encode())
                    )
                    prediction = np.round(0.5 * target + rng.normal(0, 0.05, len(keys)), 4)
                    values = dict(
                        asset_id=[f"US/{symbol}" for symbol, *_ in keys],
                        market=["US"] * len(keys),
                        prediction_at=pa.array(
                            [moment for _, _, moment, _ in keys], pa.timestamp("us", tz="UTC")
                        ),
                        target=target,
                        prediction=prediction,
                    )
                    if arm == "gru":
                        quantiles = prediction[:, None] + 0.02 * OFFSETS
                        values.update(zip(QUANTILE_COLUMNS, quantiles.T, strict=True))
                    path = root / arm / str(seed) / fold["id"] / f"{part}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.table(values), path)
                    entry[part] = dict(path=str(path), sha256=sha256(path))
                    if part == "evaluation":
                        predictions[arm, seed, fold["id"]] = (keys, prediction)
                entries.setdefault(arm, {}).setdefault(str(seed), {})[fold["id"]] = entry
    sources = dict(
        schema_version=1,
        kind=walk.SOURCES_KIND,
        scope="US",
        input_policy=HISTORICAL_MASKED,
        windows=windows,
        arms=entries,
    )
    sources_path = root / "sources.json"
    save(sources_path, sources)
    return config_path, sources_path, predictions


def test_technical_views_feed_the_strata_from_their_own_samples(technical, tmp_path):
    config, sources, predictions = write_technical_study(tmp_path / "study", technical)
    report, _ = walk.evaluate_walk_forward(config, sources, "US")
    section = report["modality_strata"]
    assert section["status"] == "computed"
    assert section["presence"] == {
        "fold-000": dict(rows=21, incomplete=0),
        "fold-001": dict(rows=21, incomplete=0),
    }
    sessions, counts = {}, dict.fromkeys(NAMES, 0)
    for fold in technical.folds:
        keys, prediction = predictions["ridge", 42, fold["id"]]
        for (symbol, row, moment, target), value in zip(keys, prediction, strict=True):
            news, fundamentals, _ = fixture_presence("US", symbol, row)
            name = next(
                n
                for n, p in strata.STRATA.items()
                if (p["news"], p["fundamentals"]) == (news, fundamentals)
            )
            counts[name] += 1
            sessions.setdefault(name, {}).setdefault(moment, []).append(abs(value - target))
    for name in NAMES:
        cell = section["population"][name]["overall"]["US"]
        assert cell["rows"] == counts[name] and cell["sessions"] == len(sessions.get(name, {}))
        means = [np.mean(errors) for errors in sessions.get(name, {}).values()]
        mae = section["arms"]["ridge"]["42"][name]["overall"]["US"]["session_mae"]
        if cell["estimable"]:
            assert mae == pytest.approx(np.mean(means), rel=1e-12)
        else:
            assert mae is None
    assert sum(counts.values()) == 42 and all(counts.values())
    for name in NAMES:
        gru = section["arms"]["gru"]["42"][name]["overall"]["US"]
        if section["population"][name]["overall"]["US"]["estimable"]:
            assert [row["nominal"] for row in gru["calibrated_intervals"]] == [0.8, 0.95]


def test_missing_common_calibrator_leaves_stratum_intervals_absent_with_reason(
    tmp_path, monkeypatch
):
    study = Study(tmp_path, scope="US")
    study.config["calibration"]["min_rows"] = 22
    report, _ = run_with(declare(study, min_rows=6, min_sessions=3), by_asset, monkeypatch)
    section = report["modality_strata"]
    cell = section["arms"]["gru"]["42"]["neither"]["overall"]["US"]
    assert cell["session_mae"] is not None and cell["calibrated_intervals"] is None
    assert cell["calibrated_reason"] == "Alguna ventana no tiene calibrador"
    calibration = section["interval_calibration"]["neither"]["gru"]["US"]["0.8"]
    assert calibration["calibrated_coverage_error"] is None
    assert calibration["raw_coverage_error"] is not None
    ridge = section["arms"]["ridge"]["42"]["neither"]["overall"]["US"]
    assert ridge["calibrated_reason"] == "El brazo no emite cuantiles"


def presence_table(presence, counts):
    return pa.table(
        dict(
            prediction_at=pa.array([0] * len(counts), pa.timestamp("us", tz="UTC")),
            presence=presence,
            news_count=pa.array(counts, pa.int64()),
        )
    )


@pytest.mark.parametrize(
    "presence,counts,message",
    [
        ([[True, True, True, False]], [1], "cinco booleanos"),
        ([[True, None, True, False, True]], [0], "cinco booleanos"),
        ([[True, True, True, False, True]], [0], "eventos admitidos"),
        ([[True, False, True, False, True]], [2], "eventos admitidos"),
        ([[True, False, True, False, True]], [-1], "eventos admitidos"),
    ],
)
def test_presence_bits_are_validated_like_the_corpus(presence, counts, message):
    with pytest.raises(ValueError, match=message):
        strata._bits(presence_table(presence, counts))
    good = presence_table([[True, True, True, False, True]], [3])
    assert strata._bits(good).tolist() == [[True, True, True, False, True]]
