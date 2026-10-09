"""Protocolo anual v2: cortes, purga por intervalo, reserva, parada y paridad del v1."""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation.splits import FoldPartitioner, build_folds, stopping_rule
from mars_titan.training.selection import advance_selection, initial_selection

CONFIGS = Path("configs/evaluation")
V2 = {
    "US": "historical-masked-us-walk-forward-v2.json",
    "CN": "historical-masked-cn-walk-forward-v2.json",
    "JOINT-US": "historical-masked-joint-us-walk-forward-v2.json",
}
DAY = 86_400_000_000
RULE = dict(
    metric="session_mae", stopping="fixed_budget", patience=5, min_delta=1e-5, max_epochs=30
)
# Huellas calculadas con el código anterior a la versión 2 sobre la misma rejilla.
V1_DIGESTS = {
    "chinese-real-walk-forward.json": (
        10,
        866,
        "395e67f473cd0b650af55ebda16052716386cd2864348d1523017ec6d3e39e18",
    ),
    "historical-masked-cn-walk-forward.json": (
        10,
        866,
        "b785650fb3bb8896ae2f5e91e1366769a2f18481566a31f05cb0b3185adb6564",
    ),
    "historical-masked-us-walk-forward.json": (
        10,
        897,
        "38966ee78fb8dd08e5a597932b0477d2bcdbbb3816a324df4f88883714f63d92",
    ),
    "real-expanded-walk-forward.json": (
        10,
        897,
        "cc1b3de1739fbbaa0b3c11e294e866ba5c9556327b28040268727692757b9647",
    ),
    "strict-macro-walk-forward.json": (
        4,
        778,
        "06d038112e142b2ad05d0d2bda76f7bd8a22f737e92864ab8f8515bb550d400c",
    ),
    "walk-forward.json": (
        41,
        3266,
        "f99babca4e1433532b40b50f4e94d37a70589123f8a9dd5aa8ac5efc957e41bc",
    ),
}


def protocol(name):
    return json.loads((CONFIGS / V2.get(name, name)).read_text())


def micros(moment):
    return int(moment.timestamp() * 1_000_000)


def midnight(day):
    return micros(datetime.fromisoformat(day).replace(tzinfo=UTC))


def protocol_digest(path):
    """Resumir cortes, límites y asignaciones de un protocolo sobre una rejilla fija."""
    config = json.loads(Path(path).read_text())
    folds = build_folds(config)
    digest = hashlib.sha256(json.dumps(folds, sort_keys=True).encode())
    clock = MarketClock(config["market"], "1999-12-01", "2025-01-10")
    moments = np.array([micros(m) for m in clock.decisions], dtype=np.int64)
    start = np.datetime64(config["first_validation_start"], "us").astype(np.int64) - 120 * DAY
    keep = (moments >= start) | (np.arange(len(moments)) % 20 == 0)
    index = np.flatnonzero(keep[:-2])
    prediction, step = moments[index], np.arange(len(index))
    month_end = (
        (prediction.astype("datetime64[us]").astype("datetime64[M]") + np.timedelta64(1, "M"))
        .astype("datetime64[us]")
        .astype(np.int64)
    )
    maturity = np.select(
        [step % 4 == 0, step % 4 == 1, step % 4 == 2],
        [moments[index + 1], moments[index + 2], prediction + 1],
        default=month_end,
    )
    available = np.where(step % 7 == 0, prediction + 1, prediction)
    eligible = step % 11 != 0
    for fold in folds:
        partitioner = FoldPartitioner(fold, clock, config)
        digest.update(json.dumps([[str(v) for v in b] for b in partitioner.bounds]).encode())
        result = partitioner.assign(prediction, available, maturity, eligible=eligible)
        for name in ("partition", "reason"):
            digest.update("\n".join(result[name].tolist()).encode())
    return len(folds), len(prediction), digest.hexdigest()


@pytest.mark.parametrize("name", sorted(V1_DIGESTS))
def test_v1_protocols_keep_exact_folds_bounds_and_assignments(name):
    assert protocol_digest(CONFIGS / name) == V1_DIGESTS[name]


@pytest.mark.parametrize(("name", "first", "count"), [("US", 2005, 19), ("CN", 2011, 13)])
def test_v2_windows_are_annual_expanding_and_stop_before_the_reserve(name, first, count):
    folds = build_folds(protocol(name))
    assert len(folds) == count
    for index, fold in enumerate(folds):
        year = first + index
        assert fold["id"] == f"fold-{index:03d}"
        assert fold["train"] == ["2000-01-01", f"{year - 1}-04-01"]
        assert fold["validation"] == [f"{year - 1}-04-01", f"{year - 1}-10-01"]
        assert fold["calibration"] == [f"{year - 1}-10-01", f"{year}-01-01"]
        assert fold["evaluation"] == [f"{year}-01-01", f"{year + 1}-01-01"]
    assert folds[-1]["evaluation"][1] == "2024-01-01"
    bounds = [day for fold in folds for part in fold.values() if part != fold["id"] for day in part]
    assert max(bounds) == "2024-01-01" and min(bounds) == "2000-01-01"


def test_joint_protocols_share_cuts_and_reuse_the_us_windows_by_date():
    us, cn, joint = protocol("US"), protocol("CN"), protocol("JOINT-US")
    assert {k: v for k, v in joint.items() if k != "market"} == {
        k: v for k, v in cn.items() if k != "market"
    }
    assert joint["market"] == "US" and cn["market"] == "CN"
    strip = [{k: v for k, v in fold.items() if k != "id"} for fold in build_folds(joint)]
    later = [
        {k: v for k, v in fold.items() if k != "id"}
        for fold in build_folds(us)
        if fold["evaluation"][0] >= "2011-01-01"
    ]
    assert strip == later


def _months(start, end):
    return (end.year - start.year) * 12 + end.month - start.month + (end.day >= start.day) - 1


@pytest.mark.parametrize(
    ("name", "market", "first_price"),
    [("US", "US", "2000-01-03"), ("CN", "CN", "2006-01-04"), ("JOINT-US", "CN", "2006-01-04")],
)
def test_first_window_is_the_first_with_three_years_of_mature_labels(name, market, first_price):
    clock = MarketClock(market, first_price, "2012-12-31")
    # El residual necesita 126 pares observados, así que la primera etiqueta es la sesión 126.
    first_label = clock.days[125]
    config = protocol(name)
    start = date.fromisoformat(build_folds(config)[0]["validation"][0])
    assert _months(first_label, start) >= 36
    earlier = date(start.year - 1, start.month, 1)
    assert _months(first_label, earlier) < 36


@pytest.mark.parametrize(
    "change",
    [
        dict(purge="fixed_sessions"),
        dict(gap_sessions=1),
        dict(step_months=18),
        dict(step_months=6),
        dict(schema_version=True),
        dict(schema_version=3),
        dict(selection=RULE | dict(stopping="whenever")),
        dict(selection=RULE | dict(metric="test_mae")),
        dict(selection=RULE | dict(patience=0)),
        dict(selection=RULE | dict(min_delta=-1)),
        dict(selection=RULE | dict(max_epochs=0)),
        dict(selection=RULE | dict(minimum_epochs=30)),
        dict(selection=RULE | dict(unknown=1)),
        dict(selection={k: v for k, v in RULE.items() if k != "stopping"}),
        dict(selection={k: v for k, v in RULE.items() if k != "max_epochs"}),
        dict(first_validation_start="2003-04-01"),
        dict(final_test_end="2024-01-01"),
    ],
)
def test_invalid_or_leaky_v2_protocols_are_rejected(change):
    config = protocol("US") | change
    if "gap_sessions" in change:
        config.pop("purge")
    with pytest.raises(ValueError):
        build_folds(config)


def test_v2_can_declare_a_subset_of_disjoint_annual_windows():
    folds = build_folds(protocol("US") | dict(step_months=36))
    assert [fold["evaluation"][0][:4] for fold in folds] == [
        "2005",
        "2008",
        "2011",
        "2014",
        "2017",
        "2020",
        "2023",
    ]
    assert all(fold["train"][0] == "2000-01-01" for fold in folds)


def _partitioner(name, index, clock):
    config = protocol(name)
    return FoldPartitioner(build_folds(config)[index], clock, config), config


def test_interval_purge_removes_each_label_that_matures_after_its_boundary():
    clock = MarketClock("US", "2021-12-01", "2025-01-10")
    partitioner, _ = _partitioner("US", 18, clock)
    days = [
        "2022-03-30",
        "2022-03-31",
        "2022-09-29",
        "2022-09-30",
        "2022-12-29",
        "2022-12-30",
        "2023-12-28",
        "2023-12-29",
        "2024-01-02",
    ]
    position = {day: clock.days.index(date.fromisoformat(day)) for day in days}
    prediction = np.array([micros(clock.decisions[position[d]]) for d in days])
    maturity = np.array([micros(clock.decisions[position[d] + 1]) for d in days])
    result = partitioner.assign(prediction, prediction.copy(), maturity)
    assert result["partition"].tolist() == [
        "train",
        "excluded",
        "validation",
        "excluded",
        "calibration",
        "excluded",
        "evaluation",
        "excluded",
        "test_reserved",
    ]
    purged = result["reason"] == "label_crosses_boundary"
    assert partitioner.nominal_partitions(prediction[purged]).tolist() == [
        "train",
        "validation",
        "calibration",
        "evaluation",
    ]
    assert "session_gap" not in result["reason"].tolist()


def test_interval_purge_keeps_an_already_mature_last_session_and_rejects_the_exact_boundary():
    clock = MarketClock("US", "2021-12-01", "2025-01-10")
    partitioner, _ = _partitioner("US", 18, clock)
    prediction = np.array([micros(clock.decision("2022-03-31"))] * 2)
    maturity = np.array([prediction[0] + 1, midnight("2022-04-01")])
    result = partitioner.assign(prediction, prediction.copy(), maturity)
    assert result["partition"].tolist() == ["train", "excluded"]
    assert result["reason"].tolist() == ["accepted", "label_crosses_boundary"]


def test_next_session_labels_purge_the_same_rows_as_the_previous_one_session_margin():
    v2 = protocol("US")
    v1 = {k: v for k, v in v2.items() if k not in {"purge", "selection"}}
    v1.update(schema_version=1, gap_sessions=1)
    clock = MarketClock("US", "1999-12-01", "2025-01-10")
    moments = np.array([micros(m) for m in clock.decisions], dtype=np.int64)
    prediction, maturity = moments[63:-1], moments[64:]
    for fold in build_folds(v2):
        new = FoldPartitioner(fold, clock, v2).assign(prediction, prediction, maturity)
        old = FoldPartitioner(fold, clock, v1).assign(prediction, prediction, maturity)
        assert np.array_equal(new["partition"], old["partition"])
        changed = new["reason"] != old["reason"]
        assert set(old["reason"][changed]) <= {"session_gap"}
        assert set(new["reason"][changed]) <= {"label_crosses_boundary"}


def test_no_reserved_session_enters_any_v2_window():
    for name in V2:
        config = protocol(name)
        clock = MarketClock(config["market"], "2023-12-01", "2025-01-10")
        moments = np.array([micros(m) for m in clock.decisions], dtype=np.int64)
        prediction, maturity = moments[:-1], moments[1:]
        later = prediction >= midnight("2024-01-01")
        reserved = later & (prediction < midnight("2025-01-01"))
        for fold in build_folds(config):
            result = FoldPartitioner(fold, clock, config).assign(prediction, prediction, maturity)
            assert set(result["partition"][reserved]) == {"test_reserved"}
            assert set(result["reason"][reserved]) == {"final_test_reserved"}
            assert set(result["partition"][later]) <= {"test_reserved", "excluded"}


def test_changing_the_future_suffix_does_not_move_earlier_rows():
    config = protocol("US")
    clock = MarketClock("US", "1999-12-01", "2025-01-10")
    moments = np.array([micros(m) for m in clock.decisions], dtype=np.int64)
    prediction, maturity = moments[63:-2], moments[64:-1]
    available = prediction.copy()
    rng = np.random.default_rng(42)
    for fold in build_folds(config):
        partitioner = FoldPartitioner(fold, clock, config)
        cut = midnight(fold["evaluation"][1])
        before = partitioner.assign(prediction, available, maturity)
        later = prediction >= cut
        changed_maturity = maturity.copy()
        changed_maturity[later] += rng.integers(1, 30 * DAY, later.sum())
        changed_available = available.copy()
        changed_available[later] += rng.integers(0, 2, later.sum())
        order = np.concatenate([np.flatnonzero(~later), rng.permutation(np.flatnonzero(later))])
        after = partitioner.assign(
            prediction[order], changed_available[order], changed_maturity[order]
        )
        kept = np.flatnonzero(~later)
        for name in ("partition", "reason"):
            assert np.array_equal(after[name][: len(kept)], before[name][kept])


def run_rule(rule, scores, *, parent=None, resume_at=None):
    """Recorrer evaluaciones completas fijadas, sin modelo ni pasos de optimizador."""
    options = {k: v for k, v in rule.items() if k != "max_epochs"}
    state = None if parent is None else initial_selection(parent, options)
    for epoch, score in enumerate(scores[: rule["max_epochs"]], start=1):
        state = advance_selection(state, score, epoch, options)
        if epoch == resume_at:
            state = json.loads(json.dumps(state))
        if state["should_stop"]:
            return dict(state, stop_reason="validation_plateau")
    return dict(state, stop_reason="budget_exhausted")


def test_v2_stopping_rule_is_common_and_matches_the_reference_plan():
    rules = [stopping_rule(protocol(name)) for name in V2]
    assert all(rule == RULE for rule in rules)
    plan = json.loads(
        Path("configs/baselines/historical-masked-reference-search-us.json").read_text()
    )
    assert {key: plan[key] for key in ("stopping", "patience", "min_delta", "max_epochs")} == {
        key: RULE[key] for key in ("stopping", "patience", "min_delta", "max_epochs")
    }


def test_fixed_budget_keeps_the_best_state_and_ignores_ties_and_small_gains():
    rule = stopping_rule(protocol("US"))
    flat = run_rule(rule, [1.0] * 40)
    assert flat["last_epoch"] == 30 and flat["best_epoch"] == 1
    assert flat["stop_reason"] == "budget_exhausted" and flat["plateau_epoch"] == 6
    tiny = run_rule(rule, [1.0, 0.999995, 0.999992, 0.999991] + [0.999991] * 40)
    assert tiny["best_epoch"] == 1 and tiny["last_epoch"] == 30
    gains = run_rule(rule, [1.0, 0.9, 0.8, 0.8, 0.7] + [0.75] * 20 + [0.69] + [0.8] * 10)
    assert gains["best_epoch"] == 26 and gains["plateau_epoch"] == 10
    assert gains["best_score"] == 0.69 and gains["last_epoch"] == 30


def test_a_declared_plateau_counts_patience_only_after_the_minimum():
    rule = dict(RULE, stopping="validation_plateau", minimum_epochs=3)
    stopping_rule(protocol("US") | dict(selection=rule))
    flat = run_rule(rule, [1.0] * 40)
    assert flat["last_epoch"] == 8 and flat["stop_reason"] == "validation_plateau"
    descent = run_rule(rule, [1.0 - 0.01 * epoch for epoch in range(40)])
    assert descent["stop_reason"] == "budget_exhausted" and descent["last_epoch"] == 30


@pytest.mark.parametrize("stopping", ["fixed_budget", "validation_plateau"])
def test_initial_state_is_eligible_and_recovery_mid_patience_keeps_the_decision(stopping):
    rule = dict(RULE, stopping=stopping)
    worse = run_rule(rule, [1.1, 1.2, 1.05, 1.3, 1.2, 1.4, 1.1, 1.2] * 4, parent=1.0)
    assert worse["best_epoch"] == 0 and worse["best_score"] == 1.0
    scores = [1.0, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.99, 0.99] * 4
    continuous = run_rule(rule, scores)
    resumed = run_rule(rule, scores, resume_at=4)
    assert continuous == resumed and continuous["best_epoch"] == 2
    assert continuous["last_epoch"] == (30 if stopping == "fixed_budget" else 7)
