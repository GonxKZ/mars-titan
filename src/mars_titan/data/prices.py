"""Normalización por activo, sin ajustar ni reparar silenciosamente OHLCV."""

from pathlib import Path

import numpy as np
import pandas as pd

from .temporal import MarketClock


def read_prices(path: Path, clock: MarketClock) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, dtype={"Date": str})
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not {"Date", *required} <= set(frame.columns):
        raise ValueError(f"Missing OHLCV columns in {path.name}")
    frame["session"] = frame["Date"].str[:10]
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    duplicates = frame["session"].duplicated(keep=False)
    finite = np.isfinite(values).all(axis=1)
    o, h, lo, c, v = (values[name] for name in required)
    invalid = (
        ~finite
        | (o <= 0)
        | (lo <= 0)
        | (c <= 0)
        | (h < lo)
        | (h < o)
        | (h < c)
        | (lo > o)
        | (lo > c)
        | (v < 0)
    )
    session_values = {
        day.isoformat(): cutoff for day, cutoff in zip(clock.days, clock.decisions, strict=True)
    }
    not_session = ~frame["session"].isin(session_values)
    excluded = duplicates | invalid | not_session
    result = values.loc[~excluded].rename(columns=str.lower).copy()
    result["session"] = frame.loc[~excluded, "session"]
    result["available_at"] = result["session"].map(session_values)
    result = result.sort_values("session").reset_index(drop=True)
    audit = {
        "rows": len(frame),
        "accepted": len(result),
        "excluded": int(excluded.sum()),
        "invalid_ohlc": int(invalid.sum()),
        "duplicate_session_rows": int(duplicates.sum()),
        "non_session_rows": int(not_session.sum()),
        "first_session": frame["session"].min() if len(frame) else None,
        "last_session": frame["session"].max() if len(frame) else None,
        "adjustments": "as_distributed_retrospective_adjustments_not_reconstructed",
        "reason_counts_overlap": True,
    }
    return result, audit
