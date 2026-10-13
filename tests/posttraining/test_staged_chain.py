"""Puntuación de validación y brazo del padre congelado de la cadena por etapas.

La regla, los identificadores y las rutas de la cadena se prueban con
`training.campaign_chain` en `tests/training/test_campaign_chain.py`.
"""

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


def test_the_frozen_parent_arm_follows_the_campaign_plan():
    assert staged_chain.frozen_arm("gru") == "gru__frozen_parent"
