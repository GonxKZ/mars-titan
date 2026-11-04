"""Registro de ciclos de autoedición G1: N fijo, cobro, placebo y puerta entre ciclos (#748).

Las puntuaciones y los contrastes se escriben a mano. El recorrido completo sobre paneles
sintéticos está en `test_output_self_edit.py`. No hay pasos de optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.proposals import self_edit_cycles as cycles
from mars_titan.proposals.self_edit_cycles import SelfEditLedger

DAY = 86_400_000_000
START = 1_577_836_800_000_000  # 2020-01-01
DOMAIN = [dict(value=index) for index in range(20)]
DECISION = dict(block_length=4, replicates=200, seed=7, confidence=0.95)
PLAN = dict(candidates=3, placebo_seed=11, decision=DECISION)


def checked(item):
    if not (isinstance(item, dict) and set(item) == {"value"} and type(item["value"]) is int):
        raise ValueError("Configuración fuera de la gramática de prueba")
    return dict(item)


GRAMMARS = {"toy": checked}
SHA = "a" * 64


def ledger(plan=PLAN):
    return SelfEditLedger(plan, GRAMMARS)


def declare(book, *, current=None, generated=(1, 2, 3), at=START):
    return book.declare(
        "toy",
        current=dict(value=0) if current is None else current,
        generated=[dict(value=value) for value in generated],
        generator="manual",
        domain=DOMAIN,
        fit=dict(sha256=SHA, matured_until=at),
    )


def block(first, matured):
    return dict(sha256="b" * 64, first_emission=first, matured_until=matured)


def comparison(interval, *, loss=True, decision=DECISION, coefficients=None, name=None):
    return dict(
        kind="paired_session_contrasts",
        metric="mae",
        loss=loss,
        cohort_sha256="c" * 64,
        confidence=decision["confidence"],
        resampling=dict(
            block_length=decision["block_length"],
            replicates=decision["replicates"],
            seed=decision["seed"],
        ),
        multiplicity=dict(family_size=1),
        contrasts=[
            dict(
                name=name or cycles.CONTRAST,
                coefficients=coefficients or {"generated": 1.0, "placebo": -1.0},
                estimate=None if interval is None else sum(interval) / 2,
                interval=interval,
                sessions=40,
            )
        ],
    )


def run_cycle(book, interval, *, generated=(1, 2, 3), scores=None, at=START):
    index = declare(book, current=book.current("toy", dict(value=0)), generated=generated, at=at)
    scores = scores or dict(current=1.0, generated=[0.9, 0.8, 0.95], placebo=[0.99, 0.97, 1.2])
    book.select(index, partition=block(at + DAY, at + 30 * DAY), scores=scores, loss=True)
    book.assess(index, partition=block(at + 31 * DAY, at + 60 * DAY), comparison=interval)
    return index


def test_the_plan_is_declared_before_any_cycle():
    assert cycles.checked_plan(PLAN) == PLAN
    for wrong in (
        PLAN | dict(candidates=0),
        PLAN | dict(candidates=True),
        PLAN | dict(placebo_seed=-1),
        PLAN | dict(decision=DECISION | dict(confidence=1.0)),
        PLAN | dict(decision={"block_length": 4}),
        {key: value for key, value in PLAN.items() if key != "decision"},
    ):
        with pytest.raises(ValueError):
            cycles.checked_plan(wrong)


def test_the_placebo_samples_the_same_grammar_reproducibly_without_the_current():
    current = dict(value=0)
    first = cycles.placebo(DOMAIN, current, count=8, seed=11, cycle=1)
    assert first == cycles.placebo(DOMAIN, current, count=8, seed=11, cycle=1)
    assert first != cycles.placebo(DOMAIN, current, count=8, seed=11, cycle=2)
    assert len({item["value"] for item in first}) == 8 and current not in first
    assert all(item in DOMAIN for item in first)
    # Cada configuración distinta de la vigente aparece con probabilidad 8/19.
    counts = np.zeros(20)
    for cycle in range(3000):
        for item in cycles.placebo(DOMAIN, current, count=8, seed=11, cycle=cycle):
            counts[item["value"]] += 1
    frequency = counts[1:] / 3000
    assert counts[0] == 0 and np.all(np.abs(frequency - 8 / 19) < 0.05)
    with pytest.raises(ValueError, match="tantas"):
        cycles.placebo(DOMAIN, current, count=20, seed=11, cycle=1)


def test_each_list_declares_exactly_n_distinct_configurations_of_the_grammar():
    book = ledger()
    for generated in ((1, 2), (1, 2, 3, 4)):
        with pytest.raises(ValueError, match="exactamente N"):
            declare(book, generated=generated)
    with pytest.raises(ValueError, match="repite configuraciones"):
        declare(book, generated=(1, 1, 2))
    with pytest.raises(ValueError, match="repite la vigente"):
        declare(book, generated=(0, 1, 2))
    with pytest.raises(ValueError, match="no está registrada"):
        book.declare(
            "other",
            current=dict(value=0),
            generated=[],
            generator="g",
            domain=DOMAIN,
            fit=dict(sha256=SHA, matured_until=START),
        )
    with pytest.raises(ValueError, match="gramática de prueba"):
        declare(book, generated=(1, 2, 3.0))
    index = declare(book)
    cycle = book.cycle(index)
    assert cycle["status"] == "declared" and cycle["placebo"]["seed"] == [11, 1]
    assert cycle["placebo"]["items"] == cycles.placebo(
        DOMAIN, dict(value=0), count=3, seed=11, cycle=1
    )


def test_selection_charges_every_candidate_and_adopts_only_an_improvement():
    book = ledger()
    index = declare(book)
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.8], placebo=[0.99, 0.97, 0.97])
    selection = book.select(
        index, partition=block(START + DAY, START + 30 * DAY), scores=scores, loss=True
    )
    # Empate entre la segunda y la tercera: gana el orden declarado.
    assert selection["generated"] == 1 and selection["placebo"] == 1
    assert selection["adopted"] == dict(value=2) and selection["charged"] == 6
    assert [item["ordinal"] for item in selection["rejected"]] == [0, 2]
    assert book.charged() == 6 and book.current("toy", dict(value=0)) == dict(value=2)
    other = ledger()
    index = declare(other)
    worse = dict(current=0.5, generated=[0.9, 0.8, 0.8], placebo=[0.99, 0.97, 0.97])
    selection = other.select(
        index, partition=block(START + DAY, START + 30 * DAY), scores=worse, loss=True
    )
    assert selection["adopted"] is None and len(selection["rejected"]) == 3
    assert other.current("toy", dict(value=0)) == dict(value=0)


def test_selection_rejects_a_block_that_starts_before_g_matures_and_bad_scores():
    book = ledger()
    index = declare(book, at=START + 10 * DAY)
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.7], placebo=[0.9, 0.8, 0.7])
    with pytest.raises(ValueError, match="antes de que maduren"):
        book.select(index, partition=block(START, START + DAY), scores=scores, loss=True)
    with pytest.raises(ValueError, match="N candidatas"):
        book.select(
            index,
            partition=block(START + 10 * DAY, START + 30 * DAY),
            scores=scores | dict(placebo=[0.9]),
            loss=True,
        )
    with pytest.raises(ValueError, match="finitos"):
        book.select(
            index,
            partition=block(START + 10 * DAY, START + 30 * DAY),
            scores=scores | dict(current=float("nan")),
            loss=True,
        )
    with pytest.raises(ValueError, match="estado selected"):
        book.assess(index, partition=block(START, START), comparison=None)


def test_the_cycle_beats_the_placebo_only_with_a_favourable_interval():
    for interval, beats in (
        ([-0.3, -0.1], True),
        ([-0.3, 0.1], False),
        ([0.1, 0.3], False),
        ([-0.3, 0.0], False),
        (None, False),
    ):
        book = ledger()
        run_cycle(book, comparison(interval))
        assert book.cycle(1)["verdict"]["beats_placebo"] is beats, interval


def test_the_contrast_must_follow_the_declared_rule():
    for wrong in (
        comparison([-0.3, -0.1], decision=DECISION | dict(block_length=8)),
        comparison([-0.3, -0.1], decision=DECISION | dict(seed=8)),
        comparison([-0.3, -0.1], coefficients={"generated": -1.0, "placebo": 1.0}),
        comparison([-0.3, -0.1], name="other"),
        comparison([-0.3, -0.1], loss=False),
    ):
        with pytest.raises(ValueError):
            run_cycle(ledger(), wrong)


def test_the_assessment_block_starts_after_the_selection_block_matures():
    book = ledger()
    index = declare(book)
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.7], placebo=[0.95, 0.85, 0.75])
    book.select(index, partition=block(START + DAY, START + 30 * DAY), scores=scores, loss=True)
    with pytest.raises(ValueError, match="antes de que maduren"):
        book.assess(
            index,
            partition=block(START + 29 * DAY, START + 60 * DAY),
            comparison=comparison([-0.3, -0.1]),
        )


def test_the_same_choice_in_both_lists_cannot_beat_the_placebo():
    book = ledger()
    index = declare(book)
    placebo = book.cycle(index)["placebo"]["items"]
    # El generador propone la misma primera candidata que el placebo y ambas ganan.
    book = ledger()
    generated = [item["value"] for item in placebo]
    index = declare(book, generated=generated)
    scores = dict(current=1.0, generated=[0.5, 0.8, 0.9], placebo=[0.5, 0.8, 0.9])
    book.select(index, partition=block(START + DAY, START + 30 * DAY), scores=scores, loss=True)
    with pytest.raises(ValueError, match="no hay contraste"):
        book.assess(
            index,
            partition=block(START + 31 * DAY, START + 60 * DAY),
            comparison=comparison([-1.0, -0.5]),
        )
    verdict = book.assess(
        index, partition=block(START + 31 * DAY, START + 60 * DAY), comparison=None
    )
    assert verdict == dict(beats_placebo=False, reason="misma_configuracion", contrast=None)


def test_a_second_cycle_opens_only_after_the_first_beats_its_placebo():
    book = ledger()
    first = declare(book)
    with pytest.raises(ValueError, match="no está contrastado"):
        declare(book, current=dict(value=0), generated=(4, 5, 6))
    book.select(
        first,
        partition=block(START + DAY, START + 30 * DAY),
        scores=dict(current=1.0, generated=[0.9, 0.8, 0.95], placebo=[0.99, 0.97, 1.2]),
        loss=True,
    )
    book.assess(
        first,
        partition=block(START + 31 * DAY, START + 60 * DAY),
        comparison=comparison([-0.3, 0.1]),
    )
    with pytest.raises(ValueError, match="no batió al placebo"):
        declare(book, current=dict(value=2), generated=(4, 5, 6))
    winner = ledger()
    run_cycle(winner, comparison([-0.3, -0.1]))
    second = run_cycle(winner, comparison([-0.2, 0.2]), generated=(4, 5, 6), at=START + 61 * DAY)
    assert second == 2 and winner.cycle(2)["current"] == dict(value=2)
    with pytest.raises(ValueError, match="no batió al placebo"):
        declare(winner, current=winner.current("toy", dict(value=0)), generated=(7, 8, 9))


def test_the_state_survives_export_and_resumes_a_cycle_halfway():
    book = ledger()
    run_cycle(book, comparison([-0.3, -0.1]))
    index = declare(book, current=book.current("toy", dict(value=0)), generated=(4, 5, 6))
    payload = json.loads(json.dumps(book.export()))
    restored = SelfEditLedger.restore(payload, GRAMMARS)
    assert restored.export() == book.export()
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.95], placebo=[0.99, 0.97, 1.2])
    for item in (book, restored):
        item.select(
            index, partition=block(START + 61 * DAY, START + 90 * DAY), scores=scores, loss=True
        )
    assert restored.export() == book.export()
    tampered = json.loads(json.dumps(payload))
    tampered["cycles"][0]["verdict"]["beats_placebo"] = False
    with pytest.raises(ValueError, match="huella"):
        SelfEditLedger.restore(tampered, GRAMMARS)
    body = {key: value for key, value in tampered.items() if key != "state_sha256"}
    tampered["state_sha256"] = cycles._sha256(body)
    with pytest.raises(ValueError, match="secuencia"):
        SelfEditLedger.restore(tampered, GRAMMARS)
    with pytest.raises(ValueError, match="otras gramáticas"):
        SelfEditLedger.restore(payload, {"other": checked})


def test_a_block_may_start_when_the_previous_labels_mature():
    book = ledger()
    index = declare(book)
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.7], placebo=[0.95, 0.85, 0.75])
    book.select(index, partition=block(START, START + 30 * DAY), scores=scores, loss=True)
    book.assess(
        index,
        partition=block(START + 30 * DAY, START + 60 * DAY),
        comparison=comparison([-0.3, -0.1]),
    )
    assert book.cycle(index)["verdict"]["beats_placebo"] is True


def test_a_tie_with_the_current_configuration_is_not_adopted():
    book = ledger()
    index = declare(book)
    scores = dict(current=0.8, generated=[0.9, 0.8, 0.85], placebo=[0.99, 0.97, 0.97])
    selection = book.select(
        index, partition=block(START + DAY, START + 30 * DAY), scores=scores, loss=True
    )
    assert selection["generated"] == 1 and selection["adopted"] is None
    assert selection["rejected"][1] == dict(ordinal=1, reason="no_mejora_la_vigente")


def test_each_verdict_uses_a_block_no_previous_verdict_saw():
    book = ledger()
    run_cycle(book, comparison([-0.3, -0.1]))
    # El segundo ciclo puede seleccionar en el bloque contrastado por el primero, pero su
    # contraste necesita un bloque posterior a ese.
    index = declare(book, current=book.current("toy", dict(value=0)), generated=(4, 5, 6))
    scores = dict(current=1.0, generated=[0.9, 0.8, 0.95], placebo=[0.99, 0.97, 1.2])
    book.select(index, partition=block(START + DAY, START + 30 * DAY), scores=scores, loss=True)
    with pytest.raises(ValueError, match="antes de que maduren"):
        book.assess(
            index,
            partition=block(START + 31 * DAY, START + 59 * DAY),
            comparison=comparison([-0.3, -0.1]),
        )
    book.assess(
        index,
        partition=block(START + 60 * DAY, START + 90 * DAY),
        comparison=comparison([-0.3, -0.1]),
    )
    assert book.cycle(index)["status"] == "assessed"
