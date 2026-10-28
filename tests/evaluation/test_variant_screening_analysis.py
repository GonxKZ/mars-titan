"""Análisis de la criba de #746 con paneles sintéticos de respuesta conocida, sin PyTorch.

Las predicciones se generan en la prueba: la referencia B tiene señal y una amplitud tres
veces mayor de la que le corresponde, la memoria por áreas de B1 tiene señal y el resto de
modelos solo ruido. No se construye ningún modelo ni se ejecutan pasos de optimizador.
"""

import copy
import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pytest

from mars_titan.evaluation import variant_screening_analysis as analysis
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.training import variant_screening

ENTRIES = ("A1", "A2", "A3", "B1", "C4", "F1")
ASSETS = tuple(f"US/A{i:02d}" for i in range(16))
SPANS = dict(validation=60, calibration=70, evaluation=40)
LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
NORMAL = np.array([-1.959964, -1.281552, 0.0, 1.281552, 1.959964])
SECTORS = {asset: f"S{i % 4}" for i, asset in enumerate(ASSETS)}


def protocol_of(entries=ENTRIES):
    protocol = variant_screening.load_protocol()
    reduced = copy.deepcopy(protocol)
    reduced["catalog"] = {name: protocol["catalog"][name] for name in entries}
    return reduced


def partition_times(year):
    """Días consecutivos de validación, calibración y evaluación, que empieza el 2 de enero."""
    start = int(datetime(year, 1, 2, 21, tzinfo=UTC).timestamp() * 1_000_000)
    first = start - (SPANS["validation"] + SPANS["calibration"]) * DAY_MICROSECONDS
    result, day = {}, first
    for name, span in SPANS.items():
        days = day + DAY_MICROSECONDS * np.arange(span)
        result[name] = np.repeat(days, len(ASSETS))
        day += span * DAY_MICROSECONDS
    return result


def make_panel(times, target, prediction, quantiles=None):
    assets = np.tile(np.asarray(ASSETS), len(times) // len(ASSETS))
    return ForecastPanel.from_columns(
        [f"{asset}/{at}" for asset, at in zip(assets, times, strict=True)],
        np.full(len(times), "US"),
        times,
        target,
        prediction,
        quantiles=quantiles,
        levels=None if quantiles is None else LEVELS,
    )


class SyntheticOrigin:
    """Objetivos de un origen y paneles deterministas de cada trabajo."""

    def __init__(self, year, jobs):
        self.year = year
        self.times = partition_times(year)
        rng = np.random.default_rng(year)
        self.target = {name: rng.standard_normal(len(t)) for name, t in self.times.items()}
        self.jobs = {job["id"]: job for job in jobs if job["origin"] == year}
        self.cache = {}
        self.loads = []

    def prediction(self, job, name):
        digest = hashlib.sha256(f"{job['id']}/{name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        y = self.target[name]
        noise = rng.standard_normal(len(y))
        if "references/quantile_reference" in job["used_by"]:
            point = 3 * (0.4 * y + 0.6 * noise)
            scale = 3 * (0.5 + 0.5 * rng.random(len(y)))
            return point, point[:, None] + scale[:, None] * NORMAL
        if "B1/area_memory" in job["used_by"]:
            spread = 0.7 if job["definition"]["search_case"] == "eta5e-2" else 0.9
            return 0.5 * y + spread * noise, None
        return 0.8 * noise, None

    def load(self, job_id):
        self.loads.append(job_id)
        if job_id not in self.cache:
            job = self.jobs[job_id]
            panels = {}
            for name, times in self.times.items():
                point, quantiles = self.prediction(job, name)
                panels[name] = make_panel(times, self.target[name], point, quantiles)
            self.cache[job_id] = panels
        return self.cache[job_id]

    def origin(self):
        return analysis.Origin(
            year=self.year,
            load=self.load,
            cohorts={
                name: make_panel(times, self.target[name], np.zeros(len(times))).cohort_sha256
                for name, times in self.times.items()
            },
            maturity={name: times + DAY_MICROSECONDS for name, times in self.times.items()},
            sectors=SECTORS,
        )


@pytest.fixture(scope="module")
def screened():
    protocol = protocol_of()
    jobs = variant_screening.screening_jobs(protocol)
    sources = {o["year"]: SyntheticOrigin(o["year"], jobs) for o in protocol["origins"]}
    origins = {year: source.origin() for year, source in sources.items()}
    report, arrays = analysis.analyze(protocol, jobs, origins)
    return protocol, jobs, report, arrays


def brute_force_stepdown(estimate, draws):
    """Definición directa del paso descendente, hipótesis a hipótesis."""
    spread = draws.std(axis=0, ddof=1)
    statistic = np.abs(estimate) / spread
    centred = np.abs(draws - estimate) / spread
    order = np.argsort(-statistic, kind="stable")
    values, current = np.empty(len(estimate)), 0.0
    for j, k in enumerate(order):
        tail = centred[:, order[j:]].max(axis=1)
        current = max(current, (1 + np.count_nonzero(tail >= statistic[k])) / (1 + len(draws)))
        values[k] = current
    return values


def test_the_stepdown_matches_its_definition_and_bounds_the_marginal_pvalue():
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((500, 5)) @ np.diag([1.0, 2.0, 0.5, 1.0, 3.0])
    draws += 0.3 * rng.standard_normal((500, 1))
    estimate = np.array([3.5, 0.2, 1.4, -2.6, 0.0])
    tests = analysis.stepdown(estimate, draws)
    assert np.array_equal(tests["stepdown"], brute_force_stepdown(estimate, draws))
    assert np.all(tests["stepdown"] >= tests["marginal"])
    single = (
        1
        + np.count_nonzero(
            (np.abs(draws - estimate) / draws.std(axis=0, ddof=1)).max(axis=1)[:, None]
            >= np.abs(estimate) / draws.std(axis=0, ddof=1),
            axis=0,
        )
    ) / 501
    assert np.all(tests["stepdown"] <= single)


def test_a_member_without_variation_never_rejects():
    draws = np.column_stack([np.random.default_rng(1).standard_normal(200), np.full(200, 0.5)])
    tests = analysis.stepdown(np.array([2.0, 0.5]), draws)
    assert tests["marginal"][1] == tests["stepdown"][1] == 1.0
    alone = analysis.stepdown(np.array([2.0]), draws[:, :1])
    assert alone["marginal"][0] == alone["stepdown"][0]


def member(name, values, direction):
    return dict(name=name, values=values, defined=np.ones(len(values), bool), direction=direction)


def test_only_an_effect_in_the_declared_direction_passes():
    rng = np.random.default_rng(3)
    day = np.arange(300) // 2
    noise = rng.standard_normal((3, 300))
    members = [
        member("better_ic", 0.5 + noise[0], "greater"),
        member("worse_ic", -0.5 + noise[1], "greater"),
        member("lower_mae", -0.5 + noise[2], "less"),
    ]
    inference = variant_screening.load_protocol()["inference"]
    family = analysis.evaluate_family(members, day, block_length=16, inference=inference)
    result = {row["name"]: row for row in family["members"]}
    assert result["better_ic"]["passes"] and result["lower_mae"]["passes"]
    assert result["worse_ic"]["stepdown_pvalue"] <= 0.05 and not result["worse_ic"]["passes"]
    assert family["sessions"] == 300 and family["days"] == 150
    for row in result.values():
        assert row["simultaneous"][0] <= row["interval"][0] <= row["estimate"]
        assert row["stepdown_pvalue"] >= row["marginal_pvalue"]


def test_the_case_is_chosen_with_the_validation_and_never_with_s():
    year = 2019
    times = partition_times(year)
    rng = np.random.default_rng(0)
    target = {name: rng.standard_normal(len(t)) for name, t in times.items()}

    def panels(validation_noise, evaluation_sign):
        result = {}
        for name, t in times.items():
            y = target[name]
            point = y + validation_noise if name == "validation" else evaluation_sign * y
            result[name] = make_panel(t, y, point)
        return result

    loaded = {"good-validation": panels(0.1, -1.0), "good-s": panels(1.0, 1.0)}
    origin = analysis.Origin(
        year=year,
        load=loaded.__getitem__,
        cohorts={
            n: make_panel(t, target[n], np.zeros(len(t))).cohort_sha256 for n, t in times.items()
        },
        maturity={n: t + DAY_MICROSECONDS for n, t in times.items()},
    )
    candidates = [
        dict(id="good-s", definition=dict(search_case="eta25e-2")),
        dict(id="good-validation", definition=dict(search_case="eta5e-2")),
    ]
    chosen, scores = analysis.choose_case(origin, candidates)
    assert chosen["id"] == "good-validation"
    assert scores["good-validation"] < scores["good-s"]


def test_a_job_with_other_rows_or_in_the_final_test_is_rejected():
    times = partition_times(2019)
    target = {n: np.ones(len(t)) for n, t in times.items()}
    good = {n: make_panel(t, target[n], np.zeros(len(t))) for n, t in times.items()}
    other = dict(
        good,
        evaluation=make_panel(times["evaluation"], -target["evaluation"], target["evaluation"]),
    )
    origin = analysis.Origin(
        year=2019,
        load={"good": good, "other": other}.__getitem__,
        cohorts={n: p.cohort_sha256 for n, p in good.items()},
        maturity={n: t + DAY_MICROSECONDS for n, t in times.items()},
    )
    assert origin.panels("good") is good
    with pytest.raises(ValueError, match="no predice las filas de evaluation"):
        origin.panels("other")
    late = {n: t + (2024 - 2019) * 366 * DAY_MICROSECONDS for n, t in times.items()}
    sealed = {n: make_panel(t, target[n], np.zeros(len(t))) for n, t in late.items()}
    origin = analysis.Origin(
        year=2019,
        load=lambda _: sealed,
        cohorts={n: p.cohort_sha256 for n, p in sealed.items()},
        maturity={n: t + DAY_MICROSECONDS for n, t in late.items()},
    )
    with pytest.raises(ValueError, match="test final"):
        origin.panels("late")


def test_the_analysis_recovers_the_known_answer(screened):
    protocol, _, report, arrays = screened
    decisions = report["decisions"]
    assert set(decisions) == set(ENTRIES)
    assert decisions["B1"]["passes"] and all(decisions["B1"]["required"].values())
    assert set(decisions["B1"]["required"]) == {
        name for name in decisions["B1"]["required"] if name.startswith("B1/required/")
    }
    assert len(decisions["B1"]["required"]) == len(protocol["catalog"]["B1"]["required"])
    assert not decisions["F1"]["passes"]
    contrasts = {row["name"]: row for row in report["contrasts"]}
    shrink = contrasts[decisions["A1"]["contrast"]]
    assert decisions["A1"]["metric"] == "session_mae_calibrated" and shrink["estimate"] < 0
    assert report["sessions"] == len(protocol["origins"]) * SPANS["evaluation"]
    assert all(row["defined_sessions"] == report["sessions"] for row in report["contrasts"])
    assert {name: family["block_days"] for name, family in report["declared_family"].items()} == {
        "16": 16,
        "8": 8,
        "32": 32,
    }
    assert len(report["declared_family"]["16"]["members"]) == len(ENTRIES)
    assert np.all(np.diff(arrays["session_time"]) > 0)
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_each_correction_keeps_the_case_with_the_lower_validation_error(screened):
    protocol, _, report, _ = screened
    area = [row for row in report["cases"] if row["model"] == "B1/area_memory"]
    assert len(area) == len(protocol["origins"]) * len(protocol["seeds"])
    assert {row["case"] for row in area} == {"eta5e-2"}
    for row in report["cases"]:
        assert row["job"] == min(
            row["validation_session_mae"], key=row["validation_session_mae"].get
        )


def test_every_model_reports_the_secondary_metrics_of_the_protocol(screened):
    _, _, report, _ = screened
    summary = report["models"]["B1/area_memory"]
    assert set(summary) >= {
        "session_ic",
        "session_rank_ic",
        "session_mae_calibrated",
        "session_mae_raw",
        "session_mae_minus_zero",
    }
    assert summary["session_ic"]["mean"] > 0.3
    zero = report["models"]["common/zero"]
    assert zero["session_mae_minus_zero"]["mean"] == 0.0
    assert zero["session_ic"]["defined_sessions"] == 0
    # La referencia emite tres veces la amplitud que le corresponde y λ la reduce.
    parent = report["models"]["B1/zero_memory"]
    assert parent["session_mae_calibrated"]["mean"] < parent["session_mae_raw"]["mean"]


def test_a_variant_fails_when_a_required_contrast_does_not_favour_it():
    protocol = protocol_of(("B1",))
    rng = np.random.default_rng(11)
    day = np.arange(200)
    contrasts = []
    for row in variant_screening.contrasts(protocol):
        shift = dict(primary=0.4, required=-0.4, reported=0.0)[row["role"]]
        if row["role"] == "required" and row["control"][1] == "identical_areas":
            shift = 0.4
        contrasts.append(
            dict(
                row,
                name=analysis._name(row),
                direction="greater",
                values=shift + rng.standard_normal(200),
                defined=np.ones(200, bool),
            )
        )
    decision = analysis.families(protocol, contrasts, day)["decisions"]["B1"]
    assert decision["primary_passes"] and not decision["passes"]
    assert sorted(decision["required"].values()) == [False, True]
