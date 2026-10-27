"""Ciclo de autoedición de la salida G1 sobre paneles sintéticos (#748).

Las semillas emiten medianas cinco veces más amplias que la señal. Solo fijan formas y
comprueban el recorrido: generador con G, selección y contraste en bloques posteriores de S.
Nada se ajusta con datos reales y no hay pasos de optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.evaluation import output_program
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.output_program import SeedPanels
from mars_titan.proposals import output_self_edit as edit
from mars_titan.proposals.self_edit_cycles import SelfEditLedger

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
DAY = 86_400_000_000
START = 1_577_836_800_000_000  # 2020-01-01
PLAN = dict(
    candidates=8,
    placebo_seed=20261023,
    decision=dict(block_length=4, replicates=200, seed=5, confidence=0.95),
)
MIN_ROWS = 50


def seed_panels(rng, start, sessions=24, assets=24, seeds=3):
    rows = sessions * assets
    at = start + np.repeat(np.arange(sessions), assets) * DAY
    ids = np.array([f"US/A{i % assets:03d}/{t}" for i, t in enumerate(at)])
    signal = rng.normal(0, 0.01, rows)
    target = signal + rng.normal(0, 0.02, rows)
    panels = []
    for _ in range(seeds):
        median = 5.0 * signal + rng.normal(0, 0.02, rows)
        spread = rng.uniform(0.01, 0.05, rows)
        panels.append(
            ForecastPanel.from_columns(
                ids,
                np.full(rows, "US"),
                at,
                target,
                median,
                quantiles=median[:, None] + spread[:, None] * OFFSETS,
                levels=LEVELS,
            )
        )
    return SeedPanels(tuple(panels), 0, panels[0].prediction_at + DAY)


@pytest.fixture(scope="module")
def partitions():
    rng = np.random.default_rng(42)
    names = ("validation", "calibration", "select", "assess")
    return {name: seed_panels(rng, START + 30 * DAY * k) for k, name in enumerate(names)}


@pytest.fixture(scope="module")
def generated(partitions):
    return edit.generate(
        output_program.REFERENCE,
        partitions["validation"],
        partitions["calibration"],
        count=8,
        min_rows=MIN_ROWS,
    )


def test_the_generator_proposes_the_best_programs_of_g_only(partitions, generated):
    scores = {
        json.dumps(program, sort_keys=True): edit.score(
            program, partitions["validation"], partitions["calibration"], min_rows=MIN_ROWS
        )
        for program in output_program.programs()
        if program != output_program.REFERENCE
    }
    ranked = sorted(scores.values())
    chosen = [scores[json.dumps(program, sort_keys=True)] for program in generated]
    assert chosen == ranked[:8] and output_program.REFERENCE not in generated
    # Con medianas cinco veces más amplias que la señal, ninguna elegida conserva λ = 1 sin
    # abstenerse ni combinar semillas.
    assert all(
        program["shrinkage"] != 1.0 or program["coverage"] is not None for program in generated
    )


def test_a_full_output_cycle_declares_before_selecting_and_assesses_later(partitions, generated):
    ledger = SelfEditLedger(PLAN, edit.GRAMMARS)
    index = edit.declare_output_cycle(
        ledger,
        inner_fit=partitions["validation"],
        inner_score=partitions["calibration"],
        min_rows=MIN_ROWS,
    )
    cycle = ledger.cycle(index)
    assert cycle["generated"] == generated and cycle["generator"] == edit.GENERATOR
    assert len(cycle["placebo"]["items"]) == 8 and cycle["status"] == "declared"
    selection = edit.select_output_cycle(
        ledger,
        index,
        fit_on=partitions["calibration"],
        select=partitions["select"],
        min_rows=MIN_ROWS,
    )
    expected = [
        edit.score(item, partitions["calibration"], partitions["select"], min_rows=MIN_ROWS)
        for item in cycle["generated"]
    ]
    assert selection["scores"]["generated"] == expected
    assert selection["generated"] == int(np.argmin(expected)) and selection["charged"] == 16
    verdict = edit.assess_output_cycle(
        ledger,
        index,
        fit_on=partitions["calibration"],
        assess=partitions["assess"],
        min_rows=MIN_ROWS,
    )
    assert isinstance(verdict["beats_placebo"], bool)
    final = ledger.cycle(index)
    assert final["status"] == "assessed"
    assert (
        final["partitions"]["select"]["first_emission"]
        >= cycle["partitions"]["fit"]["matured_until"]
    )
    if verdict["contrast"] is not None:
        assert verdict["contrast"]["sessions"] == partitions["assess"].reference.sessions
    restored = SelfEditLedger.restore(json.loads(json.dumps(ledger.export())), edit.GRAMMARS)
    assert restored.export() == ledger.export()


def test_the_stages_reject_blocks_out_of_order_or_another_calibration(partitions):
    ledger = SelfEditLedger(PLAN, edit.GRAMMARS)
    index = edit.declare_output_cycle(
        ledger,
        inner_fit=partitions["validation"],
        inner_score=partitions["calibration"],
        min_rows=MIN_ROWS,
    )
    with pytest.raises(ValueError, match="misma calibración"):
        edit.select_output_cycle(
            ledger,
            index,
            fit_on=partitions["validation"],
            select=partitions["select"],
            min_rows=MIN_ROWS,
        )
    # La calibración como bloque de S empieza antes de que maduren sus propias etiquetas.
    with pytest.raises(ValueError, match="maduran después|antes de que maduren"):
        edit.select_output_cycle(
            ledger,
            index,
            fit_on=partitions["calibration"],
            select=partitions["calibration"],
            min_rows=MIN_ROWS,
        )
    assert ledger.cycle(index)["status"] == "declared"


def test_the_score_gives_each_session_the_same_weight(partitions):
    select = partitions["select"].reference
    errors = np.abs(select.prediction - select.target)
    by_session = np.add.reduceat(errors, select.session_starts) / select.session_samples
    found = edit.score(
        output_program.REFERENCE,
        partitions["calibration"],
        partitions["select"],
        min_rows=MIN_ROWS,
    )
    assert found == pytest.approx(by_session.mean(), rel=1e-12)
    assert abs(np.median(by_session) - by_session.mean()) > 1e-6


def test_the_declared_g_maturity_is_the_latest_of_validation_and_calibration(partitions):
    ledger = SelfEditLedger(PLAN, edit.GRAMMARS)
    index = edit.declare_output_cycle(
        ledger,
        inner_fit=partitions["validation"],
        inner_score=partitions["calibration"],
        min_rows=MIN_ROWS,
    )
    fit = ledger.cycle(index)["partitions"]["fit"]
    assert fit["matured_until"] == partitions["calibration"].bounds()[1]
    assert fit["matured_until"] > partitions["validation"].bounds()[1]
    assert fit["sha256"] == partitions["calibration"].sha256()


def test_the_assessment_refits_with_the_calibration_the_generator_saw(partitions):
    ledger = SelfEditLedger(PLAN, edit.GRAMMARS)
    index = edit.declare_output_cycle(
        ledger,
        inner_fit=partitions["validation"],
        inner_score=partitions["calibration"],
        min_rows=MIN_ROWS,
    )
    edit.select_output_cycle(
        ledger,
        index,
        fit_on=partitions["calibration"],
        select=partitions["select"],
        min_rows=MIN_ROWS,
    )
    with pytest.raises(ValueError, match="misma calibración"):
        edit.assess_output_cycle(
            ledger,
            index,
            fit_on=partitions["validation"],
            assess=partitions["assess"],
            min_rows=MIN_ROWS,
        )
    assert ledger.cycle(index)["status"] == "selected"
