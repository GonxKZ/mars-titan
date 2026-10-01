"""Residuales con fechas convertidas una vez y ventanas NumPy acotadas."""

from datetime import date

import numpy as np
import pandas as pd

from .budget_targets import _aligned_returns
from .temporal import MarketClock


def residual_targets_array(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    clock: MarketClock,
    *,
    cutoff: str,
    history: int = 252,
    minimum: int = 126,
) -> pd.DataFrame:
    """Conservar la referencia OLS sin crear Series de fechas por cada ventana."""
    if not 2 <= minimum <= history:
        raise ValueError("El historial para calcular el residual no es válido")
    end = date.fromisoformat(cutoff)
    sessions = [day.isoformat() for day in clock.days if day <= end]
    stock, stock_at = _aligned_returns(asset, sessions, cutoff)
    factor, factor_at = _aligned_returns(market, sessions, cutoff)
    decisions = [clock.decision(session) for session in sessions]
    decision_times = pd.to_datetime(decisions, utc=True).to_numpy(dtype="datetime64[ns]")
    stock_times = stock_at.to_numpy(dtype="datetime64[ns]")
    factor_times = factor_at.to_numpy(dtype="datetime64[ns]")
    stock_dates, factor_dates = stock_at.tolist(), factor_at.tolist()
    finite = np.isfinite(stock) & np.isfinite(factor)
    rows = []
    for index, decision in enumerate(decisions):
        start = max(0, index - history + 1)
        window = slice(start, index + 1)
        valid = (
            finite[window]
            & (stock_times[window] <= decision_times[index])
            & (factor_times[window] <= decision_times[index])
        )
        x, y = factor[window][valid], stock[window][valid]
        row = dict(
            prediction_at=decision,
            target_available_at=None,
            target=None,
            alpha=None,
            beta=None,
            history_pairs=len(x),
            reason="accepted",
        )
        if index + 1 == len(sessions):
            row["reason"] = "target_after_cutoff"
        elif not finite[index + 1]:
            row["reason"] = "missing_next_session"
        elif len(x) < minimum:
            row["reason"] = "insufficient_history"
        elif np.var(x) <= np.finfo(float).eps:
            row["reason"] = "zero_market_variance"
        else:
            availability = [stock_dates[index + 1], factor_dates[index + 1]]
            maturity = max(availability) if all(pd.notna(at) for at in availability) else None
            if maturity is None or maturity.date() > end:
                row["reason"] = "target_after_cutoff"
            else:
                centered = x - x.mean()
                beta = float(centered @ (y - y.mean()) / (centered @ centered))
                alpha = float(y.mean() - beta * x.mean())
                row.update(
                    alpha=alpha,
                    beta=beta,
                    target_available_at=maturity,
                    target=float(stock[index + 1] - alpha - beta * factor[index + 1]),
                )
        rows.append(row)
    return pd.DataFrame(rows)
