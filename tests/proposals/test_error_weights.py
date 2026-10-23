"""Pesos extremos sin perder un mercado por normalizar antes de multiplicar."""

from fractions import Fraction

from tests.proposals.test_error_rule import api, plan_for, synthetic_review
from tests.proposals.test_error_rule import registry as registry


def test_weighted_mae_keeps_a_representable_term_with_underflowing_normalized_weight(registry):
    plan = plan_for(registry, market_weights=(("CN", 1e308), ("US", 1e-308)))
    review = synthetic_review(
        [("CN/A", 1, 0.0, 0.0, "label"), ("US/A", 1, 1e308, 0.0, "label")], plan
    )
    result = api().parent_zero_statistics(review, plan=plan)
    expected = float(Fraction(1e308) * Fraction(1e-308) / (Fraction(1e308) + Fraction(1e-308)))
    assert expected > 0.0
    assert result["parent_mae"] == result["excess_mae"] == expected
    assert result["triggered"] is True
