"""Cupos E3 con errores literales y estado agregado, sin ajustes de parámetros."""

import copy
import json
import math
import random
from dataclasses import replace

import pytest

from mars_titan.memory.salience_budget import (
    MatureWrite,
    MatureWriteBudget,
    SalienceBudgetConfig,
)


def row(flow="US/A", *, at=1, available=2, issued=0.0, target=1.0):
    return MatureWrite(flow, at, available, issued, target)


def budget(**options):
    return MatureWriteBudget(SalienceBudgetConfig(**options))


def history(run):
    for at in (1, 3):
        _, run = run.select(
            (
                row(target=10.0, at=at, available=at + 1),
                row("US/B", target=1.0, at=at, available=at + 1),
            ),
            cutoff=at + 1,
        )
    return run


def test_prior_asset_scale_changes_the_ranking_while_a_common_scale_does_not():
    run = history(budget(writes_per_market=1, min_history=2))
    cohort = (row(at=5, available=6, target=5.0), row("US/B", at=5, available=6, target=2.0))
    selected, updated = run.select(cohort, cutoff=6)
    assert selected.selected == (("US/B", 5),)
    assert dict(selected.scales) == {
        ("US/A", 5): pytest.approx(10.0),
        ("US/B", 5): pytest.approx(1.0),
    }
    assert dict(selected.scores) == {
        ("US/A", 5): pytest.approx(0.5),
        ("US/B", 5): pytest.approx(2.0),
    }
    assert updated.observed == 6 and updated.selected == 3
    absolute = history(budget(rule="absolute_error", writes_per_market=1, min_history=2))
    assert absolute.select(cohort, cutoff=6)[0].selected == (("US/A", 5),)


@pytest.mark.parametrize("rule", ("standardized_error", "absolute_error", "uniform"))
def test_every_rule_keeps_the_same_fixed_count_per_market_and_canonical_order(rule):
    run = budget(rule=rule, writes_per_market=2)
    cohort = tuple(row(f"US/{letter}") for letter in "ABCDE") + (row("CN/X"),)
    state_before = random.getstate()
    selected, updated = run.select(cohort, cutoff=2)
    assert len(selected.selected) == 3
    assert sum(flow.startswith("US/") for flow, _ in selected.selected) == 2
    assert ("CN/X", 1) in selected.selected
    assert selected.selected == tuple(sorted(selected.selected))
    assert run.observed == 0 and updated.observed == 6
    assert random.getstate() == state_before


@pytest.mark.parametrize("rule", ("standardized_error", "absolute_error", "uniform"))
def test_permuting_assets_keeps_selection_statistics_and_next_publication(rule):
    run = history(budget(rule=rule, writes_per_market=1, min_history=2))
    cohort = tuple(
        row(f"US/{letter}", at=5, available=6, target=index / 7)
        for index, letter in enumerate("ABCDEF")
    )
    left, a = run.select(cohort, cutoff=6)
    right, b = run.select(cohort[::-1], cutoff=6)
    assert left == right and a.export() == b.export()
    assert (
        a.select((row(at=7, available=8),), cutoff=8)[0]
        == b.select((row(at=7, available=8),), cutoff=8)[0]
    )


def test_cold_start_zero_scale_and_all_rejected_rows_still_update_statistics():
    run = budget(writes_per_market=0, min_history=1, initial_scale=0.02, scale_floor=0.001)
    selection, run = run.select((row(target=0.0),), cutoff=2)
    assert not selection.selected and dict(selection.scales)[("US/A", 1)] == 0.02
    selection, run = run.select((row(at=3, available=4, target=0.002),), cutoff=4)
    assert dict(selection.scales)[("US/A", 3)] == 0.001
    assert dict(selection.scores)[("US/A", 3)] == 2.0
    assert run.observed == 2 and run.selected == 0


def test_all_rows_of_one_asset_in_the_cohort_use_the_same_prior_scale():
    run = budget(writes_per_market=1, min_history=1, initial_scale=2.0)
    selection, updated = run.select(
        (row(target=100.0), row(at=2, available=3, target=1.0)), cutoff=3
    )
    assert set(dict(selection.scales).values()) == {2.0}
    assert updated.export()["records"]["US/A"][1] == pytest.approx(math.sqrt(5000.5))


def test_ties_and_large_quota_keep_real_ids_without_replacing_targets():
    cohort = (row("US/Z", target=-2.0), row("US/A", target=2.0))
    assert budget(writes_per_market=1).select(cohort, cutoff=2)[0].selected == (("US/A", 1),)
    assert budget(writes_per_market=100).select(cohort, cutoff=2)[0].selected == (
        ("US/A", 1),
        ("US/Z", 1),
    )
    assert cohort[0].target == -2.0


def test_uniform_selection_is_independent_of_errors_and_has_a_separate_seed():
    cohort = tuple(row(f"US/A{index}", target=float(index)) for index in range(32))
    altered = tuple(replace(item, target=1e4 - item.target) for item in cohort)
    first = budget(rule="uniform", writes_per_market=5, seed=7)
    assert first.select(cohort, cutoff=2)[0].selected == first.select(altered, cutoff=2)[0].selected
    assert (
        first.select(cohort, cutoff=2)[0].selected
        != budget(rule="uniform", writes_per_market=5, seed=8).select(cohort, cutoff=2)[0].selected
    )


def test_restore_roundtrip_preserves_the_next_selection_and_does_not_alias_payload():
    run = history(budget(writes_per_market=1, min_history=2))
    payload = json.loads(json.dumps(run.export(), allow_nan=False))
    restored = MatureWriteBudget.restore(run.config, payload)
    payload["records"]["US/A"][1] = 999.0
    assert restored.export() == run.export()
    cohort = (row(at=5, available=6), row("US/B", at=5, available=6, target=2.0))
    left, a = run.select(cohort, cutoff=6)
    right, b = restored.select(cohort, cutoff=6)
    assert left == right and a.export() == b.export()


def test_selection_does_not_mutate_or_expose_the_prior_state():
    run = history(budget())
    before = run.export()
    run.select((row(at=5, available=6),), cutoff=6)
    assert run.export() == before
    before["records"]["US/A"][1] = 999
    assert run.export()["records"]["US/A"][1] != 999


@pytest.mark.parametrize(
    "changes",
    [
        {"rule": "noise_detector"},
        {"writes_per_market": -1},
        {"writes_per_market": True},
        {"writes_per_market": 8193},
        {"min_history": 0},
        {"min_history": True},
        {"initial_scale": 0.0},
        {"scale_floor": math.nan},
        {"initial_scale": math.inf},
        {"scale_floor": 2.0, "initial_scale": 1.0},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        budget(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"flow_id": "UK/A"},
        {"flow_id": "US/../A"},
        {"prediction_at": True},
        {"prediction_at": -1},
        {"available_at": 1},
        {"available_at": 3},
        {"issued": math.nan},
        {"target": math.inf},
        {"issued": True},
        {"issued": -1e308, "target": 1e308},
    ],
)
def test_invalid_or_future_row_does_not_publish_state(changes):
    run = budget()
    before = run.export()
    with pytest.raises(ValueError):
        run.select((replace(row(), **changes),), cutoff=2)
    assert run.export() == before


def test_all_temporal_metadata_is_checked_before_accessing_any_outcome(monkeypatch):
    from mars_titan.memory import salience_budget

    run = budget()
    checked = salience_budget._finite

    def guarded(value, name):
        if name in ("La predicción emitida", "La etiqueta madura"):
            pytest.fail("No debe acceder a valores antes de comprobar toda la disponibilidad")
        return checked(value, name)

    monkeypatch.setattr(salience_budget, "_finite", guarded)
    rows = (row(), row("US/B", available=3))
    with pytest.raises(ValueError, match="La disponibilidad madura"):
        run.select(rows, cutoff=2)


def test_repeated_or_backward_publications_and_old_decisions_are_rejected():
    _, run = budget().select((row(),), cutoff=2)
    before = run.export()
    for at in (1, 2):
        with pytest.raises(ValueError):
            run.select((row("US/B"),), cutoff=at)
    with pytest.raises(ValueError):
        run.select((row(available=3),), cutoff=3)
    with pytest.raises(ValueError):
        budget().select((row(), row()), cutoff=2)
    assert run.export() == before


def test_empty_cohort_does_not_spend_quota_or_advance_the_cursor():
    run = budget()
    selection, updated = run.select((), cutoff=2)
    assert updated is run and not selection.selected and not selection.scores
    assert run.export()["cursor"] is None


def test_non_sequence_and_oversized_cohorts_are_rejected_before_iteration():
    def forbidden():
        pytest.fail("Una cohorte no acotada no debe consumirse")
        yield row()

    with pytest.raises(ValueError):
        budget().select(forbidden(), cutoff=2)
    with pytest.raises(ValueError):
        budget().select((row(),) * 8193, cutoff=2)


def test_rms_update_remains_finite_without_squaring_extreme_finite_errors():
    run = budget(rule="absolute_error", initial_scale=1.0)
    _, run = run.select((row(target=1e300),), cutoff=2)
    _, run = run.select((row(at=3, available=4, target=-1e300),), cutoff=4)
    assert run.export()["records"]["US/A"][1] == pytest.approx(1e300)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("cursor", 0),
        ("cohorts", True),
        ("selected", 999),
        ("sha256", "a" * 64),
        ("kind", "m2"),
        ("records", {}),
    ],
)
def test_restore_rejects_changed_or_incompatible_metadata(field, value):
    run = history(budget())
    payload = copy.deepcopy(run.export())
    payload[field] = value
    with pytest.raises(ValueError):
        MatureWriteBudget.restore(run.config, payload)
    with pytest.raises(ValueError):
        MatureWriteBudget.restore(replace(run.config, writes_per_market=1), run.export())


def test_restore_does_not_accept_boolean_configuration_as_an_equal_integer():
    run = budget(seed=1)
    payload = run.export()
    payload["configuration"]["seed"] = True
    with pytest.raises(ValueError):
        MatureWriteBudget.restore(run.config, payload)


def test_arbitrary_size_integers_are_rejected_as_financial_values():
    with pytest.raises(ValueError):
        budget().select((row(target=10**1000),), cutoff=2)
    with pytest.raises(ValueError):
        budget(initial_scale=10**1000)


def test_score_overflow_is_rejected_without_clipping_or_publishing():
    run = budget(initial_scale=1e-8)
    with pytest.raises(ValueError, match="desbord"):
        run.select((row(target=1e308),), cutoff=2)
    assert run.observed == 0


def test_flow_capacity_is_bounded_without_evicting_past_statistics():
    cohort = tuple(row(f"US/A{index}") for index in range(8192))
    _, run = budget().select(cohort, cutoff=2)
    before = run.export()
    with pytest.raises(ValueError, match="activos"):
        run.select((row("CN/NEW", at=3, available=4),), cutoff=4)
    assert run.export() == before


@pytest.mark.parametrize(
    "record",
    [
        [True, 1.0, 1, 2],
        [1, math.nan, 1, 2],
        [1, -1.0, 1, 2],
        [1, 1.0, 2, 2],
        [1, 1.0, 1, 99],
        [1, 1.0, 1],
    ],
)
def test_restore_checks_each_aggregate_record_even_with_an_edited_checksum(record):
    payload = history(budget()).export()
    payload["records"]["US/A"] = record
    with pytest.raises(ValueError):
        MatureWriteBudget.restore(budget().config, payload)
