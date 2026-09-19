"""Etiquetas residuales causales para medir coste con sondas, no para selección."""

from datetime import date

import numpy as np
import pandas as pd

from .temporal import MarketClock


def _aligned_returns(frame, sessions, cutoff):
    frame = frame.loc[frame.session <= cutoff].copy()
    if frame.session.duplicated().any():
        raise ValueError("Duplicate target price session")
    frame = frame.set_index("session").reindex(sessions)
    values = frame[["open", "close"]].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    returns = np.full(len(values), np.nan)
    returns[valid] = values[valid, 1] / values[valid, 0] - 1
    available = pd.to_datetime(frame.available_at, utc=True)
    return returns, available


def residual_targets(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    clock: MarketClock,
    *,
    cutoff: str,
    history: int = 252,
    minimum: int = 126,
) -> pd.DataFrame:
    """OLS con intercepto, ventana de sesiones y maduración anterior al corte."""
    if not 2 <= minimum <= history:
        raise ValueError("Invalid residual history")
    end = date.fromisoformat(cutoff)
    sessions = [d.isoformat() for d in clock.days if d <= end]
    stock, stock_at = _aligned_returns(asset, sessions, cutoff)
    spy, spy_at = _aligned_returns(market, sessions, cutoff)
    rows = []
    for index, session in enumerate(sessions):
        decision = clock.decision(session)
        start = max(0, index - history + 1)
        x, y = spy[start : index + 1], stock[start : index + 1]
        valid = (
            np.isfinite(x)
            & np.isfinite(y)
            & (stock_at.iloc[start : index + 1] <= decision).to_numpy()
            & (spy_at.iloc[start : index + 1] <= decision).to_numpy()
        )
        x, y = x[valid], y[valid]
        row = {
            "prediction_at": decision,
            "target_available_at": None,
            "target": None,
            "alpha": None,
            "beta": None,
            "history_pairs": len(x),
            "reason": "accepted",
        }
        if index + 1 == len(sessions):
            row["reason"] = "target_after_cutoff"
        elif not np.isfinite([stock[index + 1], spy[index + 1]]).all():
            row["reason"] = "missing_next_session"
        elif len(x) < minimum:
            row["reason"] = "insufficient_history"
        elif np.var(x) <= np.finfo(float).eps:
            row["reason"] = "zero_market_variance"
        else:
            availability = [stock_at.iloc[index + 1], spy_at.iloc[index + 1]]
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
                    target=float(stock[index + 1] - alpha - beta * spy[index + 1]),
                )
        rows.append(row)
    return pd.DataFrame(rows)
