"""Informe del techo de señal (#747) con informes de origen sintéticos, sin ajustar modelos."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation import signal_ceiling_report as scr
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS
from mars_titan.evaluation.variant_screening_analysis import evaluate_family

ROOT = Path(__file__).resolve().parents[2]
INFERENCE = json.loads((ROOT / "configs/experiments/signal-ceiling-v1.json").read_text())[
    "screening"
]["inference"]
FAMILIES = ("reversal", "momentum", "filings")
SESSIONS = 30
VALIDATION = {
    "residual": dict(ridge=0.01, lightgbm=0.02),
    "raw_open_close": dict(ridge=0.03, lightgbm=0.01),
}


def series(generator, mean, defined):
    values = generator.normal(mean, 0.05, len(defined))
    return dict(values=np.where(defined, values, 0.0).tolist(), defined=defined.tolist())


def origin(year, *, validation=VALIDATION):
    """Informe de un origen con una sesión sin definir distinta por objetivo y modelo."""
    generator = np.random.default_rng(year)
    # Sesiones diarias de marzo a abril, cada una en su día UTC.
    start = int(np.datetime64(f"{year}-03-01T20:05", "us").astype(np.int64))
    time = start + DAY_MICROSECONDS * np.arange(SESSIONS)
    targets = {}
    for t, target in enumerate(scr.TARGETS):
        zero = generator.uniform(0.01, 0.02, SESSIONS)
        models = {}
        for m, model in enumerate(scr.MODELS):
            defined = np.ones(SESSIONS, dtype=bool)
            defined[2 * t + m] = False
            models[model] = dict(
                session_ic=series(generator, 0.02, defined),
                session_rank_ic=series(generator, 0.01, defined),
                session_mae_raw=(zero + 0.001).tolist(),
                session_mae_calibrated=(zero - 0.0005 * (m + 1)).tolist(),
                validation_ic=validation[target][model],
                importance={family: series(generator, 0.005, defined) for family in FAMILIES},
            )
        targets[target] = dict(zero_mae=zero.tolist(), **models)
    return dict(
        kind="signal_ceiling_origin_v1",
        request=dict(origin=dict(year=year, fold=f"fold-{year - 2005:03d}")),
        sessions=dict(
            time=time.tolist(),
            day=(time // DAY_MICROSECONDS).tolist(),
            assets=[3] * SESSIONS,
        ),
        targets=targets,
    )


ORIGINS = [origin(year) for year in (2019, 2020, 2021)]


def test_the_origins_are_pooled_in_time_order_whatever_the_order_they_arrive_in():
    times, day, series = scr.pooled([ORIGINS[2], ORIGINS[0], ORIGINS[1]])
    assert times.tolist() == sum((r["sessions"]["time"] for r in ORIGINS), [])
    assert day.tolist() == sum((r["sessions"]["day"] for r in ORIGINS), [])
    assert set(series) == {(t, m) for t in scr.TARGETS for m in scr.MODELS}
    item = series["raw_open_close", "lightgbm"]
    parts = [r["targets"]["raw_open_close"]["lightgbm"] for r in ORIGINS]
    assert item["session_ic"][0].tolist() == sum((p["session_ic"]["values"] for p in parts), [])
    assert item["session_ic"][1].tolist() == sum((p["session_ic"]["defined"] for p in parts), [])
    assert item["session_mae_zero"].tolist() == sum(
        (r["targets"]["raw_open_close"]["zero_mae"] for r in ORIGINS), []
    )
    assert set(item["importance"]) == set(FAMILIES)
    assert item["validation_ic"] == [0.01, 0.01, 0.01]


def test_overlapping_missing_or_short_origins_are_refused():
    with pytest.raises(ValueError, match="al menos un origen"):
        scr.pooled([])
    repeated = copy.deepcopy(ORIGINS[1])
    repeated["request"]["origin"]["year"] = 2022
    with pytest.raises(ValueError, match="se solapan o se repiten"):
        scr.pooled([ORIGINS[0], ORIGINS[1], repeated])
    for path in (
        ("session_ic", "values"),
        ("session_rank_ic", "defined"),
        ("session_mae_calibrated",),
        ("importance", "momentum", "values"),
    ):
        short = copy.deepcopy(ORIGINS[0])
        values = short["targets"]["raw_open_close"]["lightgbm"]
        for key in path:
            values = values[key]
        values.pop()
        with pytest.raises(ValueError, match="una sesión por cada sesión"):
            scr.pooled([short, *ORIGINS[1:]])
    short = copy.deepcopy(ORIGINS[0])
    short["targets"]["residual"]["zero_mae"].pop()
    with pytest.raises(ValueError, match="una sesión por cada sesión"):
        scr.pooled([short, *ORIGINS[1:]])


def test_the_summary_averages_the_defined_sessions_of_each_series():
    result = scr.report(ORIGINS, INFERENCE)
    _, _, pooled = scr.pooled(ORIGINS)
    for (target, model), item in pooled.items():
        summary = result["summaries"][f"{target}/{model}"]
        values, defined = item["session_ic"]
        observed = values[defined]
        assert summary["mean_session_ic"] == pytest.approx(observed.mean(), rel=1e-12)
        assert summary["session_icir"] == pytest.approx(
            observed.mean() / observed.std(ddof=1), rel=1e-12
        )
        assert summary["defined_ic_sessions"] == 3 * (SESSIONS - 1)
        rank = item["session_rank_ic"]
        assert summary["mean_session_rank_ic"] == pytest.approx(rank[0][rank[1]].mean(), rel=1e-12)
        difference = item["session_mae_calibrated"] - item["session_mae_zero"]
        assert summary["mean_session_mae_calibrated_minus_zero"] == pytest.approx(
            difference.mean(), rel=1e-12
        )
        assert summary["mean_session_mae_raw"] == pytest.approx(
            np.mean(item["session_mae_raw"]), rel=1e-12
        )
        assert summary["validation_ic_by_origin"] == [VALIDATION[target][model]] * 3


def test_each_family_applies_the_screening_inference_to_the_pooled_sessions():
    result = scr.report(ORIGINS, INFERENCE)
    _, day, pooled = scr.pooled(ORIGINS)
    labels = [(t, m) for t in scr.TARGETS for m in scr.MODELS]
    names = ["/".join(label) for label in labels]
    families = result["families"]
    blocks = [INFERENCE["block_days"], *INFERENCE["sensitivity_block_days"]]
    assert set(families["session_ic"]) == {str(length) for length in blocks}
    assert set(families["session_rank_ic"]) == set(families["session_mae_calibrated_minus_zero"])
    assert set(families["session_rank_ic"]) == {str(INFERENCE["block_days"])}
    for length in blocks:
        members = [
            dict(name=n, values=v, defined=d, direction="greater")
            for n, (v, d) in zip(names, (pooled[k]["session_ic"] for k in labels), strict=True)
        ]
        expected = evaluate_family(members, day, block_length=length, inference=INFERENCE)
        assert families["session_ic"][str(length)] == expected
    # Las sesiones comunes de la familia son las que definen los cuatro miembros.
    main = families["session_ic"][str(INFERENCE["block_days"])]
    assert main["sessions"] == 3 * (SESSIONS - 4)
    mae = families["session_mae_calibrated_minus_zero"][str(INFERENCE["block_days"])]
    assert mae["sessions"] == 3 * SESSIONS
    for row, label in zip(mae["members"], labels, strict=True):
        difference = pooled[label]["session_mae_calibrated"] - pooled[label]["session_mae_zero"]
        assert row["estimate"] == pytest.approx(difference.mean(), rel=1e-12)
        # Un MAE menor que el del control cero es el lado favorable.
        assert row["favourable"] is (row["estimate"] < 0)
    assert set(result["importance"]) == set(names)
    for name, label in zip(names, labels, strict=True):
        family = result["importance"][name][str(INFERENCE["block_days"])]
        assert [row["name"] for row in family["members"]] == list(FAMILIES)
        values, defined = pooled[label]["importance"]["filings"]
        assert family["members"][2]["estimate"] == pytest.approx(values[defined].mean(), rel=1e-12)


def test_the_ceiling_follows_the_mean_validation_ic_and_a_tie_keeps_ridge():
    ceiling = scr.report(ORIGINS, INFERENCE)["ceiling"]
    assert ceiling["residual"]["model"] == "lightgbm"
    assert ceiling["raw_open_close"]["model"] == "ridge"
    assert ceiling["residual"]["validation_ic"] == pytest.approx(dict(ridge=0.01, lightgbm=0.02))
    tie = {target: dict(ridge=0.02, lightgbm=0.02) for target in scr.TARGETS}
    origins = [origin(year, validation=tie) for year in (2019, 2020, 2021)]
    ceiling = scr.report(origins, INFERENCE)["ceiling"]
    assert {target: ceiling[target]["model"] for target in scr.TARGETS} == dict.fromkeys(
        scr.TARGETS, "ridge"
    )
    # La media es la de los orígenes con el mismo peso: solo uno cambia la elección.
    leaning = copy.deepcopy(origins)
    leaning[1]["targets"]["residual"]["lightgbm"]["validation_ic"] = 0.03
    assert scr.report(leaning, INFERENCE)["ceiling"]["residual"]["model"] == "lightgbm"
    undefined = copy.deepcopy(ORIGINS)
    undefined[0]["targets"]["residual"]["ridge"]["validation_ic"] = None
    with pytest.raises(ValueError, match="no define el IC de validación"):
        scr.report(undefined, INFERENCE)


def test_the_report_is_plain_json_with_its_origins_sessions_and_inference():
    result = scr.report([ORIGINS[1], ORIGINS[0], ORIGINS[2]], INFERENCE)
    assert json.loads(json.dumps(result)) == result
    assert result["kind"] == scr.KIND and result["inference"] == INFERENCE
    assert result["sessions"] == result["days"] == 3 * SESSIONS
    # Los orígenes se publican en orden temporal, el de sus series unidas.
    assert result["origins"] == [r["request"]["origin"] for r in ORIGINS]
    assert scr.report(ORIGINS, INFERENCE) == result
