"""Eventos declarados de D2 en la comparación walk-forward (#453 y #454).

Las series y las predicciones son sintéticas y no salen de ningún modelo ajustado. No hay
pasos de optimizador. Las pruebas comprueban qué sesiones forman cada evento, que el
percentil y su cambio relativo son los de esas sesiones, que el bootstrap remuestrea las
mismas sesiones para los dos modelos de un contraste y que la sección llega al informe sin
cambiar nada más.
"""

import copy
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from mars_titan.evaluation import declared_events as events
from mars_titan.evaluation import online_calibration as online
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.evaluation.paired_comparisons import delta, interaction
from tests.evaluation import test_walk_forward_comparison as base
from tests.evaluation.test_online_calibration import JOINT, NOMINALS, panel, two_years
from tests.evaluation.test_walk_forward_predictive_ability import long_decisions
from tests.evaluation.test_walk_forward_retention import session_mae

SECTION = json.loads(JOINT.read_text())[events.FIELD]
COMPARISON = dict(block_length=3, replicates=200, confidence=0.95, seed=7)
MARKETS = ("US", "CN")


def event(market, first, last, before=0, after=0):
    return dict(market=market, first=first, last=last, sessions_before=before, sessions_after=after)


def section(**declared):
    return dict(copy.deepcopy(SECTION), events=declared)


def calendar(days):
    """Una sesión diaria por mercado desde el 1 de junio de 2015, primero US y después CN."""
    start = datetime(2015, 6, 1, tzinfo=UTC)
    times = [
        int((start + timedelta(days=day, hours=hour)).timestamp()) * 1_000_000
        for hour in (20, 7)
        for day in range(days)
    ]
    return np.array(times, dtype=np.int64), np.repeat([0, 1], days)


def test_the_campaign_declares_the_d2_events_of_the_three_proposals():
    assert events.declaration(SECTION) is SECTION
    us = [item for item in SECTION["events"].values() if item["market"] == "US"]
    spans = sorted(
        item["first"] if item["first"] == item["last"] else f"{item['first']}/{item['last']}"
        for item in us
    )
    for name in ("titans-memory-stability", "online-conformal", "kalman-associative"):
        declared = json.loads((JOINT.parent / f"{name}-comparison.json").read_text())
        assert sorted(declared["decision_rule"]["declared_events"]) == spans
    assert all(item["sessions_before"] == item["sessions_after"] == 60 for item in us)
    # D2 fija para China las sesiones de junio a agosto de 2015, sin margen.
    china = [item for item in SECTION["events"].values() if item["market"] == "CN"]
    assert china == [event("CN", "2015-06-01", "2015-08-31")]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s.update(status="computed"), "contrato"),
        (lambda s: s.update(quantile=1.0), "percentil"),
        (lambda s: s.update(use=" "), "fuente y su uso"),
        (lambda s: s["events"]["lehman_2008"].update(market="EU"), "no está bien declarado"),
        (lambda s: s["events"]["lehman_2008"].update(sessions_before=-1), "no está bien"),
        (lambda s: s["events"]["march_2020"].update(last="2020-02-01"), "al revés"),
        (lambda s: s["events"]["rate_hikes_2022"].update(last="2024-01-02"), "anterior a 2024"),
        (lambda s: s["events"].update({"Otro": event("US", "2010-05-06", "2010-05-06")}), "Otro"),
    ],
)
def test_declaration_rejects_any_change_to_its_contract(edit, message):
    changed = copy.deepcopy(SECTION)
    edit(changed)
    with pytest.raises(ValueError, match=message):
        events.declaration(changed)


def test_an_event_takes_its_market_sessions_inside_the_span_and_the_declared_margins():
    times, codes = calendar(20)
    index, counts = events.event_sessions(
        event("US", "2015-06-10", "2015-06-12", before=3, after=2), times, codes, MARKETS
    )
    assert index.tolist() == list(range(6, 14))
    assert counts == dict(before=3, inside=3, after=2)
    # El margen se recorta donde empieza la serie y China usa sus propias sesiones.
    index, counts = events.event_sessions(
        event("CN", "2015-06-02", "2015-06-03", before=5), times, codes, MARKETS
    )
    assert index.tolist() == [20, 21, 22] and counts == dict(before=1, inside=2, after=0)
    # Las sesiones se ordenan por instante, no por su posición en la serie.
    order = np.random.default_rng(3).permutation(len(times))
    found, _ = events.event_sessions(
        event("US", "2015-06-10", "2015-06-10", 1, 1), times[order], codes[order], MARKETS
    )
    assert np.sort(order[found]).tolist() == [8, 9, 10]
    assert (
        events.event_sessions(event("US", "2016-01-04", "2016-01-08"), times, codes, MARKETS)
        is None
    )
    assert (
        events.event_sessions(event("CN", "2015-06-02", "2015-06-02"), times, codes, ("US",))
        is None
    )


def synthetic():
    """Tres modelos: b es 1,1 veces a, c es igual a a y d es otro, en US y CN."""
    times, codes = calendar(40)
    rng = np.random.default_rng(11)
    a = rng.gamma(2.0, 0.01, len(times))
    values = dict(a=a, b=1.1 * a, c=a.copy(), d=rng.gamma(2.0, 0.01, len(times)))
    views = {"US+CN": np.ones(len(times), dtype=bool), "US": codes == 0}

    def series_of(arm, view):
        mask = views[view]
        return values[arm][mask], np.ones(mask.sum(), dtype=bool), times[mask], codes[mask], MARKETS

    families = dict(
        scaled=dict(**{"b-a": delta("a", "b"), "c-a": delta("a", "c")}),
        factorial=dict(interaction=interaction("a", "b", "c", "d"), **{"d-a": delta("a", "d")}),
    )
    declared = section(
        june=event("US", "2015-06-10", "2015-06-12", before=4, after=4),
        july=event("US", "2015-07-01", "2015-07-03", before=2, after=6),
        china=event("CN", "2015-06-05", "2015-06-25"),
        later=event("US", "2016-03-01", "2016-03-02"),
    )
    return declared, families, values, views, series_of, times, codes


def test_each_model_gives_the_mean_and_percentile_of_its_event_sessions():
    declared, families, values, views, series_of, times, codes = synthetic()
    result = events.report(declared, COMPARISON, families, values, list(views), series_of)
    joint, us = result["views"]["US+CN"], result["views"]["US"]
    # La vista de US no lleva el evento de China y el evento sin sesiones lo dice.
    assert set(joint["events"]) == {"june", "july", "china", "later"}
    assert set(us["events"]) == {"june", "july", "later"}
    assert joint["events"]["later"]["reason"] == "Sin sesiones evaluadas en el tramo"
    assert joint["events"]["june"] == dict(market="US", before=4, inside=3, after=4, defined=11)
    pooled = []
    for name in ("june", "july", "china"):
        index, _ = events.event_sessions(declared["events"][name], times, codes, MARKETS)
        pooled.append(index)
        for arm, series in values.items():
            found = joint["arms"][arm]["events"][name]
            assert found == dict(
                sessions=len(index),
                mean=pytest.approx(series[index].mean(), rel=1e-12),
                quantile=pytest.approx(np.quantile(series[index], 0.95), rel=1e-12),
            )
    pooled = np.concatenate(pooled)
    assert joint["arms"]["d"]["pooled"]["quantile"] == pytest.approx(
        np.quantile(values["d"][pooled], 0.95), rel=1e-12
    )


def test_the_contrast_resamples_the_same_sessions_for_both_models():
    declared, families, values, views, series_of, _, _ = synthetic()
    result = events.report(declared, COMPARISON, families, values, list(views), series_of)
    contrasts = result["views"]["US+CN"]["quantile_relative_change"]
    # La interacción no es un contraste de dos modelos y no tiene percentil relativo.
    assert set(contrasts["factorial"]) == {"d-a"}
    scaled, equal = contrasts["scaled"]["b-a"], contrasts["scaled"]["c-a"]
    # Un modelo proporcional da el mismo cambio en cada réplica, y uno igual da cero, solo
    # si las dos series se remuestrean con las mismas sesiones.
    for row in (scaled["pooled"], *scaled["events"].values()):
        assert row["estimate"] == pytest.approx(0.1, rel=1e-12)
        assert row["interval"] == pytest.approx([0.1, 0.1], rel=1e-12)
    for row in (equal["pooled"], *equal["events"].values()):
        assert row["estimate"] == 0 and row["interval"] == [0, 0]
    other = contrasts["factorial"]["d-a"]["pooled"]
    assert other["interval"][0] <= other["estimate"] <= other["interval"][1]
    again = events.report(declared, COMPARISON, families, values, list(views), series_of)
    assert again == result


def test_models_with_other_defined_sessions_are_rejected():
    declared, families, values, views, series_of, _, _ = synthetic()

    def uneven(arm, view):
        found = list(series_of(arm, view))
        if arm == "d":
            found[1] = found[1].copy()
            found[1][0] = False
        return tuple(found)

    with pytest.raises(ValueError, match="mismas sesiones definidas"):
        events.report(declared, COMPARISON, families, values, list(views), uneven)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    declared = section(
        november=event("US", "2023-11-10", "2023-11-10", before=3, after=2),
        december_china=event("CN", "2023-12-05", "2023-12-08", before=1, after=1),
        lehman_2008=event("US", "2008-09-15", "2008-09-15", before=60, after=60),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "decisions", long_decisions)
        without = base.Study(tmp_path_factory.mktemp("plain")).run()[0]
        study = base.Study(tmp_path_factory.mktemp("events"))
        study.config[events.FIELD] = declared
        study.publish()
        report, _ = study.run()
        expected = {arm: session_mae(study, arm) for arm in ("ridge", "gru", "titans")}
    return study, report, without, expected, declared


def test_the_comparison_uses_the_seed_averaged_session_mae_of_each_event(runs):
    study, report, _, expected, declared = runs
    result = report[events.FIELD]
    assert result["declaration"] == declared
    assert result["views"]["US+CN"]["events"]["lehman_2008"]["reason"]
    for name, market, days in (
        ("november", "US", range(7, 13)),
        ("december_china", "CN", range(4, 10)),
    ):
        month = 11 if name == "november" else 12
        keys = sorted(
            key
            for key in expected["gru"]
            if key[0] == market and key[1].month == month and key[1].day in days
        )
        assert len(keys) == len(days)
        for arm in ("ridge", "gru", "titans"):
            values = np.array([expected[arm][key] for key in keys])
            for view in ("US+CN", market):
                found = result["views"][view]["arms"][arm]["events"][name]
                assert found["sessions"] == len(keys)
                assert found["mean"] == pytest.approx(values.mean(), rel=1e-12)
                assert found["quantile"] == pytest.approx(np.quantile(values, 0.95), rel=1e-12)
    contrasts = result["views"]["US+CN"]["quantile_relative_change"]
    # La familia de niveles no tiene contrastes de dos modelos.
    assert set(contrasts) == {"references_vs_zero", "quantile_models"}
    assert set(contrasts["references_vs_zero"]) == {"ridge-zero", "gru-zero", "titans-zero"}


def test_the_rest_of_the_report_does_not_change(runs):
    _, report, without, _, _ = runs
    for key in ("contrasts", "arms", "interval_calibration", "sign_reliability"):
        assert report[key] == without[key]
    assert set(report["analysis_source_sha256"]) - set(without["analysis_source_sha256"]) == {
        "evaluation/declared_events.py"
    }
    assert events.FIELD not in without


def test_window_aggregates_give_the_same_section_as_reading_the_rows(runs, tmp_path):
    study, report, _, _, _ = runs
    config = walk.load_config(study.config_path)
    sources = walk.load_sources(study.sources_path, config, study.scope)
    folder = tmp_path / "aggregates"
    for window in sources["windows"]:
        window_aggregates.write(folder, config, sources, window)
    from_aggregates, _ = walk.evaluate_walk_forward(
        study.config_path, study.sources_path, study.scope, aggregates=folder
    )
    assert from_aggregates[events.FIELD] == report[events.FIELD]


def test_online_calibration_gives_the_coverage_of_each_variant_in_each_event():
    times, markets, target, wide, narrow = two_years()
    declared = section(
        december=event("US", "2014-12-19", "2014-12-20"),
        january=event("CN", "2015-01-11", "2015-01-11", after=2),
        missing=event("US", "2015-06-01", "2015-06-02"),
    )
    scores = {
        name: score_sessions(panel(times, markets, target, quantiles), rank_ic_min_assets=3)
        for name, quantiles in (("wide", wide), ("narrow", narrow))
    }
    reference = scores["wide"]
    coverage = np.stack(
        [online._coverages(scores[name], NOMINALS) for name in ("wide", "narrow")], axis=-1
    )
    declaration = online.declaration(online_section())
    found = online._event_coverage(
        declared, declaration, reference, coverage, ["wide", "narrow"], NOMINALS
    )
    assert found["missing"] == dict(market="US", reason="Sin sesiones evaluadas en el tramo")
    for name, quantiles in (("wide", wide), ("narrow", narrow)):
        for label, market, year in (("december", "US", 2014), ("january", "CN", 2015)):
            years = np.array(times, dtype="datetime64[us]").astype("datetime64[Y]").astype(int)
            rows = np.flatnonzero((np.array(markets) == market) & (years + 1970 == year))
            for position, nominal in enumerate(NOMINALS):
                inner, outer = (1, 3) if position == 0 else (0, 4)
                covered = np.mean(
                    [quantiles[i][inner] <= target[i] <= quantiles[i][outer] for i in rows]
                )
                assert found[label]["coverage"][name][f"{nominal:g}"] == pytest.approx(covered)
            assert found[label]["rows"] == len(rows)
    assert found["december"]["years"] == [2014] and found["december"]["period"] == "selection"
    assert found["january"]["years"] == [2015] and found["january"]["period"] == "assessment"


def online_section():
    declared = json.loads(JOINT.read_text())[online.FIELD]
    return dict(declared, selection_years=[2005, 2014], assessment_years=[2015, 2023])
