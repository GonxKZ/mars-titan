"""Gramática de salida de la autoedición G1 (#748).

Los paneles son sintéticos con semilla fija o escritos a mano. No hay modelos ajustados ni
pasos de optimizador.
"""

import json

import numpy as np
import pytest

from mars_titan.evaluation import neutral_abstention, output_program, scale_diagnostics
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.evaluation.output_program import REFERENCE, SeedPanels
from mars_titan.evaluation.output_stages import OUTPUT_STAGES, checked_output_stages

LEVELS = (0.025, 0.1, 0.5, 0.9, 0.975)
OFFSETS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
DAY = 86_400_000_000
START = 1_577_836_800_000_000  # 2020-01-01


def seed_panels(rng, *, start, sessions=20, assets=30, seeds=3, scale=5.0, own=0):
    """Semillas con la misma población: medianas demasiado amplias alrededor de la señal."""
    rows = sessions * assets
    at = start + np.repeat(np.arange(sessions), assets) * DAY
    ids = np.array([f"US/A{i % assets:03d}/{t}" for i, t in enumerate(at)])
    signal = rng.normal(0, 0.01, rows)
    target = signal + rng.normal(0, 0.02, rows)
    markets = np.where(np.arange(rows) % 3 == 0, "CN", "US")
    panels = []
    for _ in range(seeds):
        median = scale * signal + rng.normal(0, 0.02, rows)
        spread = rng.uniform(0.01, 0.05, rows)
        panels.append(
            ForecastPanel.from_columns(
                ids,
                markets,
                at,
                target,
                median,
                quantiles=median[:, None] + spread[:, None] * OFFSETS,
                levels=LEVELS,
            )
        )
    matures = panels[0].prediction_at + DAY
    return SeedPanels(tuple(panels), own, matures)


def pair(seed=1, **options):
    rng = np.random.default_rng(seed)
    fit_on = seed_panels(rng, start=START, **options)
    later = seed_panels(rng, start=START + 40 * DAY, **options)
    return fit_on, later


def program(**changes):
    return REFERENCE | changes


def test_the_grammar_is_closed_typed_and_contains_the_reference():
    programs = output_program.programs()
    assert len(programs) == 5 * 3 * 5 * 4
    assert len({json.dumps(item, sort_keys=True) for item in programs}) == len(programs)
    assert programs[0] == REFERENCE
    assert all(output_program.checked_program(item) == item for item in programs)
    for wrong in (
        program(seed_weight=1),
        program(seed_weight=True),
        program(coverage=0.9),
        program(shrinkage="fit_s"),
        program(threshold=0.1),
        {**REFERENCE, "extra": 1},
        {key: value for key, value in REFERENCE.items() if key != "threshold"},
    ):
        with pytest.raises(ValueError):
            output_program.checked_program(wrong)
    assert checked_output_stages({output_program.IDENTITY: program(coverage=0.8)})
    assert output_program.IDENTITY in OUTPUT_STAGES


def test_the_reference_program_emits_the_own_median_bit_for_bit():
    fit_on, later = pair(own=1)
    record = output_program.fit(REFERENCE, fit_on, min_rows=50)
    prediction = output_program.apply(record, later)
    assert prediction.tobytes() == later.reference.prediction.tobytes()
    assert later.reference.prediction.tobytes() != later.panels[0].prediction.tobytes()


def test_the_combination_averages_quantiles_level_by_level():
    fit_on, later = pair()
    record = output_program.fit(program(seed_weight=0.25), fit_on, min_rows=50)
    quantiles = np.stack([panel.quantiles for panel in later.panels])
    expected = 0.25 * quantiles[0] + 0.75 * quantiles.mean(axis=0)
    assert np.array_equal(output_program.apply(record, later), expected[:, 2])
    combined = output_program.combine(later, 0.0)
    assert np.all(np.diff(combined.quantiles, axis=1) >= 0)
    with pytest.raises(ValueError, match="dos"):
        output_program.combine(SeedPanels(later.panels[:1], 0, later.matures_at), 0.5)


def test_the_coverage_field_is_the_neutral_abstention_of_the_combined_quantiles():
    fit_on, later = pair(seed=3)
    record = output_program.fit(program(coverage=0.95), fit_on, min_rows=50)
    config = neutral_abstention.DEFAULT | dict(nominal=0.95, min_rows=50)
    stage = neutral_abstention.fit(fit_on.reference, fit_on.matures_at, config)
    expected = neutral_abstention.apply(stage, later.reference)[0]
    assert np.array_equal(output_program.apply(record, later), expected)


def test_the_threshold_uses_the_declared_quantile_of_each_market_in_g():
    fit_on, later = pair(seed=4)
    record = output_program.fit(program(threshold=0.5), fit_on, min_rows=50)
    reference = fit_on.reference
    for code, market in enumerate(reference.markets):
        magnitude = np.abs(reference.prediction[reference.market == code])
        assert record["thresholds"][market] == np.quantile(magnitude, 0.5)
    prediction = output_program.apply(record, later)
    panel = later.reference
    bound = np.array([record["thresholds"][name] for name in panel.markets])[panel.market]
    kept = np.abs(panel.prediction) >= bound
    assert np.array_equal(prediction, np.where(kept, panel.prediction, 0.0))
    assert 0 < kept.mean() < 1


def test_the_quantile_rule_matches_a_hand_computation():
    # |m| = 1, 2, 3, 4 en G: el cuantil lineal 0,5 es 2,5 y el 0,25 es 1,75.
    assert np.quantile([1.0, 2.0, 3.0, 4.0], 0.5, method=output_program.QUANTILE_METHOD) == 2.5
    assert np.quantile([1.0, 2.0, 3.0, 4.0], 0.25, method=output_program.QUANTILE_METHOD) == 1.75


def test_shrinkage_is_fixed_or_the_market_lambda_of_g():
    fit_on, later = pair(seed=5)
    fixed = output_program.fit(program(shrinkage=0.0625), fit_on, min_rows=50)
    assert np.array_equal(output_program.apply(fixed, later), 0.0625 * later.reference.prediction)
    fitted = output_program.fit(program(shrinkage="fit_g"), fit_on, min_rows=50)
    factors = scale_diagnostics.market_shrinkage(fit_on.reference, fit_on.reference.prediction)
    assert fitted["shrinkage"] == factors and all(0 <= value < 1 for value in factors.values())
    assert np.array_equal(
        output_program.apply(fitted, later),
        scale_diagnostics.apply_shrinkage(later.reference, later.reference.prediction, factors),
    )


def test_a_program_is_applied_only_after_its_labels_mature_and_never_reads_targets():
    fit_on, later = pair(seed=6)
    record = output_program.fit(program(coverage=0.8, shrinkage="fit_g"), fit_on, min_rows=50)
    with pytest.raises(ValueError, match="maduran después"):
        output_program.apply(record, fit_on)
    changed = SeedPanels(
        tuple(
            ForecastPanel.from_columns(
                panel.row_id,
                np.asarray(panel.markets)[panel.market],
                panel.prediction_at,
                -panel.target + 1.0,
                panel.prediction,
                quantiles=panel.quantiles,
                levels=LEVELS,
            )
            for panel in later.panels
        ),
        later.own,
        later.matures_at,
    )
    left, right = output_program.apply(record, later), output_program.apply(record, changed)
    assert left.tobytes() == right.tobytes()


def test_the_fitted_record_survives_json_and_keeps_its_contract():
    fit_on, later = pair(seed=7)
    record = output_program.fit(
        program(seed_weight=0.5, coverage=0.8, shrinkage="fit_g", threshold=0.25),
        fit_on,
        min_rows=50,
    )
    restored = json.loads(json.dumps(record))
    assert output_program.apply(restored, later).tobytes() == (
        output_program.apply(record, later).tobytes()
    )
    with pytest.raises(ValueError, match="contrato"):
        output_program.apply(restored | {"identity": "other"}, later)
    with pytest.raises(ValueError, match="otras semillas"):
        output_program.apply(restored | {"own": 2}, later)


def test_the_seed_panels_need_the_same_rows_and_a_valid_own_seed():
    first, _ = pair(seed=8)
    _, other = pair(seed=9, sessions=21)
    with pytest.raises(ValueError, match="mismas filas"):
        SeedPanels((first.panels[0], other.panels[0]), 0, first.matures_at)
    with pytest.raises(ValueError, match="semilla propia"):
        SeedPanels(first.panels, 3, first.matures_at)
    with pytest.raises(ValueError, match="madurez"):
        SeedPanels(first.panels, 0, first.matures_at[:-1])
    assert first.sha256() != SeedPanels(first.panels, 1, first.matures_at).sha256()


def test_the_reference_needs_no_other_seed_and_starts_when_g_matures():
    fit_on, later = pair(seed=10, seeds=1)
    record = output_program.fit(program(coverage=0.8), fit_on, min_rows=50)
    shifted = int(later.reference.prediction_at.min()) - record["fitted_until"]
    on_time = SeedPanels(
        tuple(
            ForecastPanel.from_columns(
                panel.row_id,
                np.asarray(panel.markets)[panel.market],
                panel.prediction_at - shifted,
                panel.target,
                panel.prediction,
                quantiles=panel.quantiles,
                levels=LEVELS,
            )
            for panel in later.panels
        ),
        0,
        later.matures_at - shifted,
    )
    assert int(on_time.reference.prediction_at.min()) == record["fitted_until"]
    output_program.apply(record, on_time)
    reference = output_program.fit(REFERENCE, fit_on, min_rows=50)
    assert output_program.apply(reference, later).tobytes() == (
        later.reference.prediction.tobytes()
    )


def hand_seeds(median, start):
    median = np.asarray(median, dtype=np.float64)
    rows = len(median)
    at = start + np.arange(rows, dtype=np.int64) * DAY
    panel = ForecastPanel.from_columns(
        np.array([f"r{i}" for i in range(rows)]),
        np.full(rows, "US"),
        at,
        np.zeros(rows),
        median,
        quantiles=median[:, None] + OFFSETS,
        levels=LEVELS,
    )
    return SeedPanels((panel,), 0, at + DAY)


def test_a_median_on_the_threshold_is_emitted():
    fit_on = hand_seeds([1.0, -2.0, 3.0, -4.0], START)
    record = output_program.fit(program(threshold=0.5), fit_on, min_rows=1)
    assert record["thresholds"] == {"US": 2.5}
    later = hand_seeds([2.5, -2.4, -3.0], START + 10 * DAY)
    assert output_program.apply(record, later).tolist() == [2.5, 0.0, -3.0]
