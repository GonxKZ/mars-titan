"""Ciclo de autoedición de la salida sobre predicciones guardadas (G1, #748).

Recorre un ciclo de `self_edit_cycles` con la gramática de `evaluation.output_program`, sin
reentrenar ningún peso. El generador no tiene conocimiento del mundo ni modelo de lenguaje:
ajusta cada programa de la gramática con la validación de G, lo puntúa con el MAE por sesión
en la calibración de G y propone los N mejores. Es el bucle interior de SEAL reducido a
aplicar una edición y medirla en datos que la edición no vio.

Las tres etapas reciben solo los tramos que pueden ver:

- `declare_output_cycle`: validación y calibración de G.
- `select_output_cycle`: la calibración, con la que se ajusta cada programa, y el primer
  bloque de S.
- `assess_output_cycle`: la calibración y el bloque posterior de S.

Cada programa que se aplica fuera de G se ajusta con la calibración, igual que λ en A1 y la
CQR en A2, y `output_program.apply` rechaza un tramo que empiece antes de que maduren sus
etiquetas.
"""

import numpy as np

from mars_titan.evaluation import output_program, scale_diagnostics
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.paired_comparisons import compare_series, delta
from mars_titan.proposals import self_edit_cycles

GRAMMAR = "output"
GENERATOR = "inner_validation_rank_v1"
GRAMMARS = {GRAMMAR: output_program.checked_program}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def session_mae(program, fit_on, score_on, *, min_rows=output_program.MIN_ROWS):
    """MAE de cada sesión de `score_on` con el programa ajustado en `fit_on`."""
    record = output_program.fit(program, fit_on, min_rows=min_rows)
    prediction = output_program.apply(record, score_on)
    return scale_diagnostics.session_mae(score_on.reference, prediction)


def score(program, fit_on, score_on, *, min_rows=output_program.MIN_ROWS):
    """MAE medio por sesión, con el mismo peso para cada sesión."""
    return float(np.mean(session_mae(program, fit_on, score_on, min_rows=min_rows)))


def generate(current, inner_fit, inner_score, *, count, min_rows=output_program.MIN_ROWS):
    """Los `count` programas distintos del vigente con menor MAE en la calibración de G.

    Los empates siguen el orden canónico de la gramática.
    """
    ranked = sorted(
        (score(program, inner_fit, inner_score, min_rows=min_rows), position, program)
        for position, program in enumerate(output_program.programs())
        if program != current
    )
    return [program for _, _, program in ranked[:count]]


def _partition(seeds):
    first, matured = seeds.bounds()
    return dict(sha256=seeds.sha256(), first_emission=first, matured_until=matured)


def declare_output_cycle(ledger, *, inner_fit, inner_score, min_rows=output_program.MIN_ROWS):
    """Generar y declarar las N candidatas y el placebo con la partición G solamente."""
    current = ledger.current(GRAMMAR, output_program.REFERENCE)
    generated = generate(
        current, inner_fit, inner_score, count=ledger.plan["candidates"], min_rows=min_rows
    )
    return ledger.declare(
        GRAMMAR,
        current=current,
        generated=generated,
        generator=GENERATOR,
        domain=output_program.programs(),
        fit=dict(
            sha256=inner_score.sha256(),
            matured_until=max(inner_fit.bounds()[1], inner_score.bounds()[1]),
        ),
    )


def select_output_cycle(ledger, index, *, fit_on, select, min_rows=output_program.MIN_ROWS):
    """Puntuar en el primer bloque de S la vigente y las 2N candidatas y registrar la selección."""
    cycle = ledger.cycle(index)
    _require(
        cycle["grammar"] == GRAMMAR and cycle["partitions"]["fit"]["sha256"] == fit_on.sha256(),
        "La selección debe ajustar con la misma calibración de G que vio el generador",
    )
    scores = dict(
        current=score(cycle["current"], fit_on, select, min_rows=min_rows),
        generated=[score(item, fit_on, select, min_rows=min_rows) for item in cycle["generated"]],
        placebo=[
            score(item, fit_on, select, min_rows=min_rows) for item in cycle["placebo"]["items"]
        ],
    )
    return ledger.select(index, partition=_partition(select), scores=scores, loss=True)


def assess_output_cycle(ledger, index, *, fit_on, assess, min_rows=output_program.MIN_ROWS):
    """Contrastar en el bloque posterior las dos elegidas con la regla declarada en el plan."""
    cycle = ledger.cycle(index)
    _require(
        cycle["grammar"] == GRAMMAR and cycle["partitions"]["fit"]["sha256"] == fit_on.sha256(),
        "El contraste debe ajustar con la misma calibración de G que vio el generador",
    )
    generated, rival = ledger.selected(index)
    comparison = None
    if generated != rival:
        panel = assess.reference
        series = {
            name: SessionSeries(
                "mae",
                True,
                panel.cohort_sha256,
                panel.markets,
                panel.session_market,
                panel.session_period,
                session_mae(program, fit_on, assess, min_rows=min_rows),
                np.ones(panel.sessions, dtype=bool),
            )
            for name, program in (("generated", generated), ("placebo", rival))
        }
        decision = ledger.plan["decision"]
        comparison = compare_series(
            series,
            {self_edit_cycles.CONTRAST: delta("placebo", "generated")},
            block_length=decision["block_length"],
            replicates=decision["replicates"],
            seed=decision["seed"],
            confidence=decision["confidence"],
        )
    return ledger.assess(index, partition=_partition(assess), comparison=comparison)
