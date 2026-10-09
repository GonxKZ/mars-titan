"""Mismo calentamiento de entradas en todas las familias con memoria, sin abrir datos.

Titans-MAC y los núcleos de CM-v1 leen la receta de Titans-MAC, MARS-TITAN hereda las fases
de su padre y la GRU candidata declara su propio calentamiento. Todas construyen las fases
con `walk_forward_phases.window_phases`. Se comprueba que las recetas declaran los mismos
meses y que, en cada ventana de los protocolos de la comparación, ninguna fase observa algo
anterior al origen del ajuste ni posterior al final de su tramo.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mars_titan.evaluation.splits import build_folds
from mars_titan.training import candidate_walk_forward as candidate
from mars_titan.training import cm_v1_factorial as cm
from mars_titan.training import mars_titan_walk_forward as mars
from mars_titan.training import titans_walk_forward as titans
from mars_titan.training import walk_forward_phases as phases

CONFIGS = Path("configs")
TITANS = CONFIGS / "titans/chronological-training-historical-masked.json"
CANDIDATE = CONFIGS / "candidate/chronological-training.json"
CM = CONFIGS / "titans/cm-v1-factorial.json"
COMPARISON = CONFIGS / "evaluation/historical-masked-2000-comparison.json"


def protocols():
    declared = json.loads(COMPARISON.read_text())
    names = {name for scope in declared["scopes"].values() for name in scope["protocols"].values()}
    return {name: json.loads((CONFIGS / "evaluation" / name).read_text()) for name in sorted(names)}


def day(value):
    return int(np.datetime64(value, "us").astype(np.int64))


def test_recipes_declare_the_same_warmup_for_every_family_with_memory():
    warmup = titans.walk_forward_options(json.loads(TITANS.read_text()))["warmup_months"]
    assert warmup == 12
    document = json.loads(CANDIDATE.read_text())
    assert document["walk_forward"]["warmup_months"] == warmup
    # Los núcleos de CM-v1 ajustan con la receta de Titans-MAC y MARS-TITAN hereda las fases
    # que registra su padre Titans-MAC.
    declaration = json.loads(CM.read_text())
    assert (CM.parent / declaration["base"]["core_recipe"]).resolve() == TITANS.resolve()
    for module in (titans, mars, candidate):
        assert module.window_phases is phases.window_phases
    assert cm.run_titans_window is titans.run_titans_window
    assert candidate.bank_policy(warmup)["warmup_months"] == warmup


@pytest.mark.parametrize("name", sorted(protocols()))
def test_phases_never_observe_before_the_origin_or_after_their_partition(name):
    folds = build_folds(protocols()[name])
    assert folds
    for fold in folds:
        result = phases.window_phases(fold, 12)
        origin, train_end = (day(value) for value in fold["train"])
        train = result["train"]
        assert (train.warmup_start, train.decision_start) == (origin, origin)
        assert (train.decision_end, train.close_at) == (train_end, train_end)
        for partition in phases.PREDICTED:
            start, end = (day(value) for value in fold[partition])
            phase = result[partition]
            assert (phase.decision_start, phase.decision_end, phase.close_at) == (start, end, end)
            # Doce meses antes del primer día del mes del tramo, nunca antes del origen.
            expected = max(origin, day(phases.months_before(fold[partition][0], 12)))
            assert phase.warmup_start == expected
            assert origin <= phase.warmup_start <= start
        assert result["evaluation"].close_at <= day("2024-01-01")


@pytest.mark.parametrize(
    ("value", "months", "expected"),
    [
        ("2006-01-01", 12, "2005-01-01"),
        ("2006-03-15", 12, "2005-03-01"),
        ("2006-01-01", 1, "2005-12-01"),
        ("2006-01-01", 0, "2006-01-01"),
        ("2006-12-31", 60, "2001-12-01"),
    ],
)
def test_months_before_counts_calendar_months(value, months, expected):
    assert phases.months_before(value, months) == expected


@pytest.mark.parametrize("months", [-1, 61, 12.0, True, None])
def test_warmup_outside_the_declared_range_is_rejected(months):
    fold = build_folds(protocols()["historical-masked-us-walk-forward-v2.json"])[0]
    with pytest.raises(ValueError, match="warmup_months"):
        phases.window_phases(fold, months)


def test_warmup_never_starts_before_the_origin_of_the_fit():
    # En los protocolos reales la validación empieza años después del origen. Con una
    # ventana corta, el calentamiento de 12 meses se recorta en el origen del ajuste.
    fold = dict(
        train=("2000-01-03", "2000-07-01"),
        validation=("2000-07-01", "2000-10-01"),
        calibration=("2000-10-01", "2001-01-01"),
        evaluation=("2001-01-01", "2002-01-01"),
    )
    result = phases.window_phases(fold, 12)
    assert {name: result[name].warmup_start for name in phases.PREDICTED} == {
        name: day("2000-01-03") for name in phases.PREDICTED
    }
    assert result["evaluation"].decision_start == day("2001-01-01")
