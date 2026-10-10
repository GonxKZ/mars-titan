"""Regla, puntuación e identificadores de la cadena del walk-forward por etapas."""

import math

import numpy as np
import pyarrow as pa
import pytest

from mars_titan.posttraining import staged_chain


def table(rows):
    market, asset, moment, prediction, target = zip(*rows, strict=True)
    return pa.table(
        dict(
            market=list(market),
            asset_id=list(asset),
            prediction_at=pa.array(np.array(moment, dtype="datetime64[us]")),
            prediction=np.array(prediction, dtype=np.float32),
            target=np.array(target, dtype=np.float64),
        )
    )


ROWS = [
    ("US", "US/A", 10, 0.5, 0.0),
    ("US", "US/B", 10, -0.25, 0.25),
    ("US", "US/A", 20, 1.0, 0.0),
    ("CN", "CN/X", 10, 0.0, 2.0),
]


def test_the_score_is_the_mean_of_the_session_maes():
    # Sesiones (US, 10): (0.5 + 0.5) / 2, (US, 20): 1 y (CN, 10): 2. Cada una pesa igual.
    assert staged_chain.validation_score(table(ROWS)) == pytest.approx((0.5 + 1 + 2) / 3)


def test_the_score_does_not_depend_on_the_written_order():
    rng = np.random.default_rng(0)
    rows = [
        (
            str(rng.choice(["US", "CN"])),
            f"A{rng.integers(40)}",
            int(rng.integers(1, 60)),
            float(rng.normal()),
            float(rng.normal()),
        )
        for _ in range(9000)
    ]
    # Una fila por activo e instante, como en una partición de validación.
    rows = list({row[:3]: row for row in rows}.values())
    expected = staged_chain.validation_score(table(rows))
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(len(rows))
        assert staged_chain.validation_score(table([rows[i] for i in order])) == expected
    sessions = {}
    for market, _, moment, prediction, target in rows:
        error = abs(float(np.float32(prediction)) - target)
        sessions.setdefault((market, moment), []).append(error)
    manual = math.fsum(sum(errors) / len(errors) for errors in sessions.values()) / len(sessions)
    assert expected == pytest.approx(manual, rel=1e-12)


def test_each_session_is_summed_in_a_fixed_order():
    # 1e16 + 1 se redondea a 1e16 y 1 + 1 + 1e16 es exacto: el orden de la suma importa.
    rows = [
        ("US", "US/C", 10, 0.0, -1e16),
        ("US", "US/A", 10, 0.0, -1.0),
        ("US", "US/B", 10, 0.0, -1.0),
    ]
    assert staged_chain.validation_score(table(rows)) == (1e16 + 2) / 3
    assert staged_chain.validation_score(table(rows[::-1])) == (1e16 + 2) / 3


def test_the_score_rejects_missing_or_non_finite_errors():
    with pytest.raises(ValueError, match="finitos"):
        staged_chain.validation_score(table([("US", "US/A", 10, float("nan"), 0.0)]))
    with pytest.raises(KeyError):
        staged_chain.validation_score(table(ROWS).drop(["target"]))


def candidate(kind, job, score):
    return dict(kind=kind, arm=job, job=job, receipt_sha256="0" * 64, score=score)


def test_the_frozen_parent_is_only_replaced_by_a_strict_improvement():
    frozen = candidate("frozen_parent", "w/gru__frozen_parent/frozen-s42", 0.5)
    tie = candidate("adapter", "w/gru__head/fit-s42", 0.5)
    better = candidate("continuation", "w/gru__full/fit-s42", 0.5 - 1e-12)
    assert staged_chain.choose([frozen, tie]) is frozen
    assert staged_chain.choose([tie, frozen, better]) is better
    assert staged_chain.choose([frozen]) is frozen
    # Entre candidatos con la misma mejora, el identificador decide.
    first = candidate("adapter", "w/gru__a/fit-s42", 0.25)
    second = candidate("adapter", "w/gru__b/fit-s42", 0.25)
    assert staged_chain.choose([frozen, second, first]) is first
    assert staged_chain.choose([second, frozen, first]) is first


@pytest.mark.parametrize(
    "candidates",
    [
        [],
        [candidate("adapter", "a", 0.1)],
        [candidate("frozen_parent", "a", 0.1), candidate("frozen_parent", "b", 0.2)],
        [candidate("frozen_parent", "a", 0.1), candidate("adapter", "a", 0.2)],
        [candidate("frozen_parent", "a", 0.1), candidate("base", "b", 0.2)],
        [candidate("frozen_parent", "a", float("nan")), candidate("adapter", "b", 0.2)],
        [candidate("frozen_parent", "a", 0.1), candidate("adapter", "b", "0.2")],
    ],
)
def test_the_rule_needs_one_frozen_parent_and_distinct_finite_candidates(candidates):
    with pytest.raises(ValueError, match="padre congelado"):
        staged_chain.choose(candidates)


def test_identifiers_and_folders_follow_the_campaign_plan(tmp_path):
    assert staged_chain.chain_arm("gru") == "gru__chain"
    assert staged_chain.frozen_arm("gru") == "gru__frozen_parent"
    assert staged_chain.chain_job_id("US+CN", "fold-003", "gru", 7) == (
        "US+CN/fold-003/gru__chain/select-s7"
    )
    assert staged_chain.chain_folder(tmp_path, "US", "fold-001", "dlinear", 42) == (
        tmp_path / "windows/US/fold-001/dlinear__chain/seed-42"
    )
    assert staged_chain.read_selection(tmp_path, "US", "fold-001", "dlinear", 42) is None


def test_scope_windows_are_in_temporal_order():
    windows = {
        "fold-000": dict(evaluation=["2022-01-01", "2022-04-01"]),
        "fold-001": dict(evaluation=["2022-04-01", "2022-07-01"]),
    }
    campaign = dict(comparison_config=dict(resolved_scopes=dict(US=dict(windows=windows))))
    assert [name for name, _ in staged_chain.scope_windows(campaign, "US")] == [
        "fold-000",
        "fold-001",
    ]
    windows = dict(reversed(windows.items()))
    campaign["comparison_config"]["resolved_scopes"]["US"]["windows"] = windows
    with pytest.raises(ValueError, match="orden temporal"):
        staged_chain.scope_windows(campaign, "US")


def test_parent_jobs_prefer_the_carry_or_finalist_of_the_seed():
    def job(name, stage, seed):
        return dict(id=f"US/fold-000/gru/{name}", stage=stage, seed=seed)

    searches = [job(f"search-gru-{i}", "search", 42) for i in (3, 1, 2)]
    other = [job("search-gru-9", "search", 7), dict(id="US/fold-000/gru2/carry-s42")]
    assert staged_chain.parent_jobs(searches + other, "US", "fold-000", "gru", 42) == [
        "US/fold-000/gru/search-gru-1",
        "US/fold-000/gru/search-gru-2",
        "US/fold-000/gru/search-gru-3",
    ]
    finalist = job("finalist-s7", "finalist", 7)
    assert staged_chain.parent_jobs(searches + other + [finalist], "US", "fold-000", "gru", 7) == [
        "US/fold-000/gru/finalist-s7"
    ]
    carry = job("carry-s42", "carry", 42)
    assert staged_chain.parent_jobs([*searches, carry], "US", "fold-000", "gru", 42) == [
        "US/fold-000/gru/carry-s42"
    ]
    with pytest.raises(ValueError, match="no elige el estado"):
        staged_chain.parent_jobs(searches, "US", "fold-000", "gru", 5)
