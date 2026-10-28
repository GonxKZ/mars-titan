"""E3 por generaciones confirmadas, con el orden de madurez que entrega el ejecutor."""

import json
import math
from dataclasses import replace

import pytest

from mars_titan.memory import salience_budget as sb


def row(flow="US/A", at=1, available=2, issued=0.0, target=1.0):
    return sb.MatureWrite(flow, at, available, issued, target)


def budget(**options):
    return sb.CommittedWriteBudget(sb.SalienceBudgetConfig(**options))


def test_two_confirmed_generations_may_share_the_same_cutoff():
    _, prior = budget(writes_per_market=1).select((row(),), cutoff=3, generation=1)
    selection, following = prior.select(
        (row("US/B", target=2.0), row("US/C")), cutoff=3, generation=2
    )
    assert selection.selected == (("US/B", 1),)
    assert following.statistics.observed == 3 and following.statistics.selected == 2
    assert following.generation == 2 and following.cutoff == 3
    assert prior.statistics.observed == 1 and prior.generation == 1


def test_a_later_maturity_can_belong_to_an_earlier_decision():
    _, prior = budget(min_history=1).select(
        (row(at=3, available=4, target=3.0),), cutoff=4, generation=1
    )
    selection, following = prior.select(
        (row(at=1, available=5, target=4.0),), cutoff=5, generation=2
    )
    assert selection.selected == (("US/A", 1),)
    assert selection.scales == ((("US/A", 1), 3.0),)
    count, rms, decision, available = following.statistics.export()["records"]["US/A"]
    assert count == 2 and rms == pytest.approx(math.sqrt(12.5))
    assert (decision, available) == (1, 5)


def test_a_publication_scores_all_rows_with_the_prior_and_updates_rejected_rows():
    _, prior = budget(writes_per_market=1, min_history=1).select(
        (row(target=2.0), row("US/B", target=4.0)), cutoff=2, generation=1
    )
    selection, following = prior.select(
        (row(at=3, available=4, target=3.0), row("US/B", 3, 4, target=4.0)),
        cutoff=4,
        generation=2,
    )
    assert selection.selected == (("US/A", 3),)
    assert dict(selection.scores) == {("US/A", 3): 1.5, ("US/B", 3): 1.0}
    assert following.statistics.observed == 4 and following.statistics.selected == 2
    assert following.last_observed == 2 and following.last_selected == selection.selected


@pytest.mark.parametrize("quota", (0, 8192))
def test_empty_generations_do_not_create_errors_or_spend_write_quota(quota):
    selection, following = budget(writes_per_market=quota).select((), cutoff=4, generation=1)
    assert selection.selected == following.last_selected == ()
    assert following.generation == 1 and following.last_observed == 0
    assert following.statistics.observed == following.statistics.selected == 0
    assert following.statistics.cursor is None and following.cutoff == 4


def test_recovery_preserves_the_next_selection_and_does_not_alias_json():
    _, prior = budget(rule="uniform", writes_per_market=1).select(
        (row(), row("CN/B")), cutoff=2, generation=1
    )
    payload = json.loads(json.dumps(prior.export()))
    restored = sb.CommittedWriteBudget.restore(prior.config, payload)
    assert restored.export() == prior.export()
    payload["last"]["selected"].clear()
    payload["statistics"]["records"]["US/A"][1] = 999.0
    cohort = (row(at=3, available=4), row("US/B", 3, 4))
    a, next_a = prior.select(cohort, cutoff=4, generation=2)
    b, next_b = restored.select(cohort, cutoff=4, generation=2)
    assert a == b and next_a.export() == next_b.export()


@pytest.mark.parametrize("generation,cutoff", ((0, 3), (1, 3), (3, 3), (2, 1), (True, 3)))
def test_repeated_skipped_or_backward_generations_do_not_publish(generation, cutoff):
    _, prior = budget().select((row(),), cutoff=2, generation=1)
    before = prior.export()
    with pytest.raises(ValueError):
        prior.select((row("US/B"),), cutoff=cutoff, generation=generation)
    assert prior.export() == before


def test_future_metadata_is_checked_before_accessing_any_financial_value(monkeypatch):
    prior = budget()
    finite = sb._finite

    def guarded(value, name):
        if name in ("La predicción emitida", "La etiqueta madura"):
            pytest.fail("Se leyó un valor antes de validar toda la disponibilidad")
        return finite(value, name)

    monkeypatch.setattr(sb, "_finite", guarded)
    with pytest.raises(ValueError, match="disponibilidad"):
        prior.select((row(), row("US/B", available=4)), cutoff=3, generation=1)


@pytest.mark.parametrize("kind", ("duplicate", "future", "nonfinite", "unknown_market"))
def test_invalid_publication_does_not_mutate_the_confirmed_state(kind):
    prior = budget()
    bad = {
        "duplicate": (row(), row()),
        "future": (replace(row(), available_at=3),),
        "nonfinite": (replace(row(), target=math.nan),),
        "unknown_market": (replace(row(), flow_id="UK/A"),),
    }[kind]
    before = prior.export()
    with pytest.raises(ValueError):
        prior.select(bad, cutoff=2, generation=1)
    assert prior.export() == before


def test_restore_rejects_changed_bytes_and_inconsistent_counters_with_a_new_hash():
    _, prior = budget(writes_per_market=0).select((row(),), cutoff=2, generation=1)
    for reseal in (False, True):
        payload = prior.export()
        payload["last"]["observed"] = 2
        if reseal:
            payload["sha256"] = sb._digest({k: v for k, v in payload.items() if k != "sha256"})
        with pytest.raises(ValueError):
            sb.CommittedWriteBudget.restore(prior.config, payload)
