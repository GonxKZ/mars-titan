"""Estimación por momentos de σ² y ϱ para la regla kalman de B6 (PT3, #455)."""

import numpy as np
import pytest

from mars_titan.memory.cohort_noise import METHOD, estimate_cohort_noise

CAP = 0.95


def reference(values, markets, sessions):
    """ANOVA de un factor con estratos escrito con bucles, sin vectorizar."""
    strata = {}
    for value, market, session in zip(values, markets, sessions, strict=True):
        strata.setdefault(market, {}).setdefault(session, []).append(float(value))
    between = within = concentration = 0.0
    rows = groups = 0
    for sessions_of_market in strata.values():
        members = [v for group in sessions_of_market.values() for v in group]
        center = sum(members) / len(members)
        for group in sessions_of_market.values():
            mean = sum(group) / len(group)
            between += len(group) * (mean - center) ** 2
            within += sum((v - mean) ** 2 for v in group)
            concentration += len(group) ** 2 / len(members)
        rows += len(members)
        groups += len(sessions_of_market)
    msb = between / (groups - len(strata))
    msw = within / (rows - groups)
    n0 = (rows - concentration) / (groups - len(strata))
    return msb, msw, n0


def panel(rng, sessions, size, rho, *, markets=("US",), sigma2=1.0, unbalanced=False):
    values, market_of, session_of = [], [], []
    for market in markets:
        for session in range(sessions):
            n = int(rng.integers(2, 2 * size)) if unbalanced else size
            common = rng.normal(0.0, np.sqrt(rho * sigma2))
            noise = rng.normal(0.0, np.sqrt((1 - rho) * sigma2), n)
            values.extend(common + noise)
            market_of.extend([market] * n)
            session_of.extend([1_000_000 * (session + 1)] * n)
    return np.asarray(values), np.asarray(market_of), np.asarray(session_of, dtype=np.int64)


def test_unbalanced_estimate_matches_the_loop_anova():
    rng = np.random.default_rng(7)
    values, markets, sessions = panel(rng, 40, 12, 0.2, markets=("US", "CN"), unbalanced=True)
    result = estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP)
    msb, msw, n0 = reference(values, markets, sessions)
    assert result["method"] == METHOD
    assert result["between_mean_square"] == pytest.approx(msb, rel=1e-12)
    assert result["within_mean_square"] == pytest.approx(msw, rel=1e-12)
    assert result["mean_session_size"] == pytest.approx(n0, rel=1e-12)
    common = (msb - msw) / n0
    assert result["observation_noise"] == pytest.approx(common + msw, rel=1e-12)
    assert result["cohort_correlation"] == pytest.approx(common / (common + msw), rel=1e-12)
    assert result["unclipped_correlation"] == pytest.approx(result["cohort_correlation"])
    assert (result["rows"], result["sessions"], result["markets"]) == (
        len(values),
        80,
        ["CN", "US"],
    )


def test_balanced_sessions_use_their_size_and_recover_the_simulated_noise():
    rng = np.random.default_rng(11)
    values, markets, sessions = panel(rng, 2000, 50, 0.05, sigma2=4.0)
    result = estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP)
    assert result["mean_session_size"] == 50.0
    # Con 2.000 sesiones el error típico de ϱ ronda 0,002 y el de σ² un 0,5 %.
    assert abs(result["cohort_correlation"] - 0.05) < 0.02
    assert result["observation_noise"] == pytest.approx(4.0, rel=0.05)


def test_a_market_offset_is_not_cohort_correlation():
    rng = np.random.default_rng(3)
    values, markets, sessions = panel(rng, 300, 20, 0.0, markets=("US", "CN"))
    plain = estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP)
    shifted = values + np.where(markets == "CN", 5.0, -2.0)
    moved = estimate_cohort_noise(shifted, markets, sessions, correlation_cap=CAP)
    for name in ("observation_noise", "cohort_correlation", "unclipped_correlation"):
        assert moved[name] == pytest.approx(plain[name], rel=1e-9, abs=1e-12)
    # Sin el estrato, el desplazamiento entre mercados parecería un efecto de sesión.
    pooled = estimate_cohort_noise(
        shifted, np.full(len(values), "US"), sessions * 2 + (markets == "CN"), correlation_cap=CAP
    )
    assert pooled["cohort_correlation"] > 0.5


def test_scale_changes_the_variance_and_keeps_the_correlation():
    rng = np.random.default_rng(5)
    values, markets, sessions = panel(rng, 200, 8, 0.3)
    one = estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP)
    ten = estimate_cohort_noise(10.0 * values, markets, sessions, correlation_cap=CAP)
    assert ten["observation_noise"] == pytest.approx(100.0 * one["observation_noise"], rel=1e-12)
    assert ten["cohort_correlation"] == pytest.approx(one["cohort_correlation"], rel=1e-12)


def test_negative_correlation_is_zero_and_a_large_one_is_capped():
    # Cada sesión tiene la misma media, así que la variación entre sesiones es nula.
    values = np.tile([-1.0, 1.0], 50)
    sessions = np.repeat(np.arange(1, 51, dtype=np.int64), 2)
    markets = np.full(100, "US")
    result = estimate_cohort_noise(values, markets, sessions, correlation_cap=CAP)
    assert result["cohort_correlation"] == 0.0 and result["unclipped_correlation"] < 0
    assert result["observation_noise"] == result["within_mean_square"]
    rng = np.random.default_rng(9)
    strong, markets, sessions = panel(rng, 400, 10, 0.99)
    capped = estimate_cohort_noise(strong, markets, sessions, correlation_cap=0.5)
    assert capped["cohort_correlation"] == 0.5 and capped["unclipped_correlation"] > 0.9


@pytest.mark.parametrize(
    ("values", "markets", "sessions", "cap", "message"),
    [
        ([1.0, 2.0, 3.0], ["US"] * 3, [1, 1, 1], CAP, "más sesiones que mercados"),
        ([1.0, 2.0, 3.0, 4.0], ["US"] * 4, [1, 2, 3, 4], CAP, "dos residuos"),
        ([1.0, 1.0, 2.0, 2.0], ["US"] * 4, [1, 1, 2, 2], CAP, "no varían"),
        ([1.0, float("nan"), 2.0, 3.0], ["US"] * 4, [1, 1, 2, 2], CAP, "finitos"),
        ([1.0, 2.0, 3.0, 4.0], ["US"] * 3, [1, 1, 2, 2], CAP, "alineados"),
        ([1.0, 2.0, 3.0, 4.0], ["US"] * 4, [1, 1, 2, 2], 1.0, "tope"),
        ([1.0, 2.0, 3.0, 4.0], ["US"] * 4, [1, 1, 2, 2], 0, "tope"),
    ],
)
def test_degenerate_panels_are_rejected(values, markets, sessions, cap, message):
    with pytest.raises(ValueError, match=message):
        estimate_cohort_noise(
            np.asarray(values),
            np.asarray(markets),
            np.asarray(sessions, dtype=np.int64),
            correlation_cap=cap,
        )
