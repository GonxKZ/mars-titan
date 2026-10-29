"""Escalera sin parámetros de Blum y Hardt para reutilizar S en la segunda ola (#794)."""

import json
from pathlib import Path

import numpy as np
import pytest

from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import reusable_holdout as rh
from mars_titan.evaluation.paired_comparisons import circular_block_counts

DECLARATION = "configs/experiments/reusable-holdout-v1.json"
SESSIONS = 240
DAYS = np.arange(SESSIONS, dtype=np.int64)
OPTIONS = dict(resolution=0.001, block_days=8, replicates=500, seed=7)


def series(identity, values, defined=None):
    values = np.asarray(values, dtype=np.float64)
    defined = np.ones(len(values), dtype=bool) if defined is None else np.asarray(defined)
    return rh.Submission(identity, values, defined)


def noise(seed, mean=0.0, scale=0.1):
    return np.random.default_rng(seed).normal(mean, scale, SESSIONS)


def test_the_released_value_is_the_nearest_multiple_of_the_resolution():
    assert rh.released_value(0.0124, 0.001) == pytest.approx(0.012)
    assert rh.released_value(0.0126, 0.001) == pytest.approx(0.013)
    assert rh.released_value(-0.0126, 0.001) == pytest.approx(-0.013)
    assert rh.released_value(0.37, 0.25) == 0.25


def test_the_standard_error_is_the_spread_of_the_block_bootstrap_means():
    differences = noise(1)
    error = rh.standard_error(differences, DAYS, block_days=8, replicates=500, seed=7)
    counts = circular_block_counts(np.random.default_rng(7), 500, SESSIONS, 8)
    means = counts @ differences / counts.sum(axis=1)
    assert error == pytest.approx(means.std(ddof=1), rel=1e-12)


def test_the_sessions_of_one_utc_day_are_resampled_together():
    # Días con una, tres y dos sesiones: el bootstrap sortea días, cada día arrastra todas sus
    # sesiones y la media de cada réplica se pondera por las sesiones que entran. Con un
    # periodo de tres días, un bloque de ocho cubre 15, 16 o 17 sesiones según dónde empiece,
    # así que las réplicas no tienen todas el tamaño de la muestra.
    day = np.repeat(np.arange(SESSIONS // 2), np.tile([1, 3, 2], SESSIONS // 6))
    differences = noise(17)
    error = rh.standard_error(differences, day, block_days=8, replicates=500, seed=7)
    counts = circular_block_counts(np.random.default_rng(7), 500, SESSIONS // 2, 8)[:, day]
    assert len(set(counts.sum(axis=1).tolist())) > 1
    means = counts @ differences / counts.sum(axis=1)
    assert error == pytest.approx(means.std(ddof=1), rel=1e-12)


def test_with_one_day_blocks_the_error_is_the_one_of_the_article():
    # Con sesiones independientes y bloques de un día, el bootstrap da std/√n.
    differences = noise(2)
    error = rh.standard_error(differences, DAYS, block_days=1, replicates=20_000, seed=3)
    assert error == pytest.approx(differences.std(ddof=1) / np.sqrt(SESSIONS), rel=0.05)


def test_the_first_submission_is_released_without_a_previous_best():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    public, sealed = ladder.submit(series("a", noise(3, mean=0.0123)))
    assert public == dict(
        identity="a", released=rh.released_value(sealed["score"], 0.001), improved=True
    )
    assert sealed["best_before"] is None and "standard_error" not in sealed


def test_only_a_clear_improvement_changes_the_board_and_the_best_is_kept():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    base = noise(4, mean=0.01)
    first, _ = ladder.submit(series("base", base))
    # Peor que el mejor: el tablero repite el valor y el mejor sigue siendo la base.
    worse, sealed = ladder.submit(series("worse", base - 0.02 + noise(5, scale=0.01)))
    assert worse == dict(identity="worse", released=first["released"], improved=False)
    assert sealed["best_before"] == "base" and sealed["paired_mean_difference"] < 0
    # Mejor por mucho más que su error típico: se publica redondeado.
    better, sealed = ladder.submit(series("better", base + 0.03 + noise(6, scale=0.01)))
    assert better["improved"] and sealed["best_before"] == "base"
    assert sealed["paired_mean_difference"] > sealed["standard_error"] > 0
    assert better["released"] == rh.released_value(sealed["score"], 0.001)
    assert ladder.best.identity == "better"


def test_an_improvement_within_one_standard_error_is_not_released():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    base = noise(7, mean=0.01)
    ladder.submit(series("base", base))
    # Una media algo mayor pero con mucho ruido en la diferencia.
    public, sealed = ladder.submit(series("noisy", base + 0.002 + noise(8, scale=0.2)))
    assert not public["improved"] and public["released"] == rh.released_value(base.mean(), 0.001)
    assert sealed["paired_mean_difference"] <= sealed["standard_error"]


def test_a_tie_with_the_best_is_not_an_improvement():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    base = noise(9)
    ladder.submit(series("base", base))
    # Una diferencia constante nula tiene error típico cero y no supera nada.
    tie, sealed = ladder.submit(series("tie", base.copy()))
    assert not tie["improved"] and sealed["standard_error"] == 0.0
    # La misma diferencia constante, pero positiva, sí lo supera.
    shifted, sealed = ladder.submit(series("shifted", base + 1e-4))
    assert shifted["improved"] and sealed["standard_error"] < 1e-12


def test_the_difference_uses_the_common_sessions_and_the_score_its_own():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    base = noise(10, mean=0.01)
    ladder.submit(series("base", base))
    defined = np.ones(SESSIONS, dtype=bool)
    defined[:40] = False
    values = base + 0.05 + noise(18, scale=0.01)
    values[:40] = -5.0
    public, sealed = ladder.submit(series("partial", values, defined))
    assert (
        sealed["common_sessions"] == SESSIONS - 40 and sealed["defined_sessions"] == SESSIONS - 40
    )
    expected = (values - base)[40:].mean()
    assert sealed["paired_mean_difference"] == pytest.approx(expected, rel=1e-12)
    assert sealed["score"] == pytest.approx(values[40:].mean())
    assert public["improved"]


def test_an_initial_best_is_published_and_beaten_like_any_other():
    base = series("first_wave", noise(11, mean=0.02))
    result = rh.run_ladder(
        dict(resolution=0.001, block_days=8, replicates=500, seed=7),
        DAYS * rh.DAY_MICROSECONDS,
        [series("same", base.values - 0.001), series("better", base.values + 0.05)],
        best=base,
    )
    initial = result["public"]["initial"]
    assert initial == dict(identity="first_wave", released=rh.released_value(base.score, 0.001))
    board = result["public"]["board"]
    assert [item["improved"] for item in board] == [False, True]
    assert board[0]["released"] == initial["released"]
    assert [item["best_before"] for item in result["sealed"]] == ["first_wave", "first_wave"]


def test_the_ladder_groups_the_decisions_by_utc_day_with_the_declared_bootstrap():
    # Dos decisiones por día, a las 13:30 y a las 20:00 UTC.
    day = np.repeat(np.arange(SESSIONS // 2), 2)
    hour = 3_600_000_000
    times = day * rh.DAY_MICROSECONDS + np.tile([27 * hour // 2, 20 * hour], SESSIONS // 2)
    base, other = series("base", noise(19)), series("other", noise(20, mean=0.01))
    result = rh.run_ladder(
        dict(resolution=0.001, block_days=8, replicates=500, seed=7), times, [base, other]
    )
    expected = rh.standard_error(
        other.values - base.values, day, block_days=8, replicates=500, seed=7
    )
    assert result["sealed"][1]["standard_error"] == pytest.approx(expected, rel=1e-12)


def test_the_public_board_reveals_only_the_declared_fields():
    result = rh.run_ladder(
        dict(resolution=0.001, block_days=8, replicates=500, seed=7),
        DAYS * rh.DAY_MICROSECONDS,
        [series("a", noise(12)), series("b", noise(13)), series("c", noise(14, mean=0.1))],
    )
    declared = json.loads(Path(DECLARATION).read_text(encoding="utf-8"))
    assert all(list(item) == declared["public"] for item in result["public"]["board"])
    hidden = {key for item in result["sealed"] for key in item} - {"identity", "improved"}
    assert hidden == set(declared["sealed"])


def test_the_same_inputs_give_the_same_board():
    submissions = [series("a", noise(15)), series("b", noise(16, mean=0.02))]
    options = dict(resolution=0.001, block_days=8, replicates=500, seed=7)
    first = rh.run_ladder(options, DAYS * rh.DAY_MICROSECONDS, submissions)
    again = rh.run_ladder(options, DAYS * rh.DAY_MICROSECONDS, submissions)
    assert first == again


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: series("", noise(1)), "nombre"),
        (lambda: series("a", noise(1), np.ones(SESSIONS)), "booleana"),
        (lambda: series("a", noise(1), np.ones(SESSIONS - 1, dtype=bool)), "booleana"),
        (lambda: series("a", np.full(SESSIONS, np.nan)), "finitos"),
        (lambda: series("a", noise(1), np.zeros(SESSIONS, dtype=bool)), "definidas"),
    ],
    ids=["name", "mask_type", "mask_shape", "nan", "undefined"],
)
def test_a_submission_needs_a_finite_series_and_its_mask(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_an_undefined_session_may_hold_any_value():
    values = noise(1)
    values[0] = np.nan
    defined = np.ones(SESSIONS, dtype=bool)
    defined[0] = False
    assert series("a", values, defined).score == pytest.approx(values[1:].mean())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (dict(resolution=0), "resolución"),
        (dict(resolution=0.0), "resolución"),
        (dict(resolution=-0.001), "resolución"),
        (dict(resolution=1), "resolución"),
        (dict(resolution=float("inf")), "resolución"),
        (dict(block_days=0), "bloque"),
        (dict(block_days=8.0), "bloque"),
        (dict(replicates=1), "réplicas"),
        (dict(seed="7"), "semilla"),
    ],
)
def test_the_ladder_refuses_options_outside_its_declaration(changes, message):
    with pytest.raises(ValueError, match=message):
        rh.Ladder(DAYS, **(OPTIONS | changes))


def test_the_ladder_refuses_days_out_of_order():
    with pytest.raises(ValueError, match="orden temporal"):
        rh.Ladder(DAYS[::-1], **OPTIONS)


def test_each_identity_is_submitted_once_with_one_value_per_session():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    ladder.submit(series("a", noise(1)))
    with pytest.raises(ValueError, match="ya se envió"):
        ladder.submit(series("a", noise(2)))
    with pytest.raises(ValueError, match="cada día"):
        ladder.submit(rh.Submission("short", noise(3)[:-1], np.ones(SESSIONS - 1, dtype=bool)))
    with pytest.raises(ValueError, match="Submission"):
        ladder.submit(dict(identity="raw"))
    with pytest.raises(ValueError, match="ya se envió"):
        rh.Ladder(DAYS, **OPTIONS, best=series("b", noise(4))).submit(series("b", noise(5)))


def test_a_comparison_needs_shared_sessions_and_more_days_than_a_block():
    ladder = rh.Ladder(DAYS, **OPTIONS)
    first = np.zeros(SESSIONS, dtype=bool)
    first[:120] = True
    ladder.submit(series("first", noise(1), first))
    with pytest.raises(ValueError, match="no comparten ninguna sesión"):
        ladder.submit(series("second", noise(2), ~first))
    # Con una sola sesión común, un solo día no admite ningún bloque.
    one = ~first
    one[0] = True
    with pytest.raises(ValueError, match="longitud del bloque"):
        ladder.submit(series("third", noise(3), one))
    short = rh.Ladder(DAYS, **(OPTIONS | dict(block_days=SESSIONS)))
    short.submit(series("a", noise(3)))
    with pytest.raises(ValueError, match="longitud del bloque"):
        short.submit(series("b", noise(4)))


def test_the_declaration_must_describe_this_ladder(tmp_path):
    declaration, digest = rh.load_declaration(Path(DECLARATION))
    assert declaration["resolution"] == 0.001 and len(digest) == 64
    document = json.loads(Path(DECLARATION).read_text(encoding="utf-8"))
    for changes in (dict(threshold="fixed_step"), dict(public=["identity", "score"])):
        path = tmp_path / f"{next(iter(changes))}.json"
        atomic_json(path, document | changes)
        with pytest.raises(ValueError, match="no coincide"):
            rh.load_declaration(path)
    atomic_json(tmp_path / "list.json", [document])
    with pytest.raises(ValueError, match="objeto JSON"):
        rh.load_declaration(tmp_path / "list.json")
