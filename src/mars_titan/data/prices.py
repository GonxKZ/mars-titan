"""Normalización por activo, sin ajustar ni reparar silenciosamente OHLCV."""

from pathlib import Path

import numpy as np
import pandas as pd

from .temporal import MarketClock


def read_prices(
    path: Path, clock: MarketClock, *, include_details: bool = False
) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, dtype=dict.fromkeys(["Date", "Dividends", "Stock Splits"], str))
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not {"Date", *required} <= set(frame.columns):
        raise ValueError(f"Faltan columnas OHLCV en {path.name}")
    frame["session"] = frame["Date"].str[:10]
    valid_date = (
        frame["session"].str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        & pd.to_datetime(frame["session"], format="%Y-%m-%d", errors="coerce").notna()
    )
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        values = values.astype(float)
    duplicates = frame["session"].duplicated(keep=False) & valid_date
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
    not_session = ~frame["session"].isin(session_values) & valid_date
    excluded = duplicates | invalid | not_session | ~valid_date
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
        "invalid_date_rows": int((~valid_date).sum()),
        "first_session": frame.loc[valid_date, "session"].min() if valid_date.any() else None,
        "last_session": frame.loc[valid_date, "session"].max() if valid_date.any() else None,
        "corporate_action_columns": [c for c in ("Dividends", "Stock Splits") if c in frame],
        "adjustments": "as_distributed_retrospective_adjustments_not_reconstructed",
        "reason_counts_overlap": True,
    }
    if include_details:
        reasons = {
            "duplicate_session": duplicates,
            "invalid_ohlc": invalid,
            "non_session": not_session & valid_date,
            "invalid_date": ~valid_date,
        }
        audit["details"] = {
            "exclusions": [
                {
                    "source_row": int(index) + 1,
                    "source_date": frame.at[index, "Date"]
                    if pd.notna(frame.at[index, "Date"])
                    else None,
                    "reasons": [name for name, flags in reasons.items() if flags.at[index]],
                }
                for index in frame.index[excluded]
            ],
            "corporate_actions": _corporate_actions(frame),
            "coverage": _coverage(frame, valid_date, excluded, session_values),
        }
    return result, audit


def _corporate_actions(frame: pd.DataFrame) -> list[dict]:
    columns = [name for name in ("Dividends", "Stock Splits") if name in frame]
    if not columns or frame.empty:
        return []
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(values) & (values >= 0)).all(axis=1)
    present = (values != 0).any(axis=1) | ~valid
    raw = frame.reindex(columns=["Date", "Dividends", "Stock Splits"]).astype(object)
    raw = raw.where(pd.notna(raw), None)
    return [
        {
            "source_row": int(index) + 1,
            "source_date": raw.at[index, "Date"],
            "dividends": raw.at[index, "Dividends"],
            "stock_splits": raw.at[index, "Stock Splits"],
            "values_valid": bool(valid.at[index]),
        }
        for index in frame.index[present]
    ]


def _coverage(frame, valid_date, excluded, sessions) -> list[dict]:
    dates = frame.loc[valid_date, "session"]
    if dates.empty:
        return []
    first, last = dates.min(), dates.max()
    result = []
    for year in range(int(first[:4]), int(last[:4]) + 1):
        observed = dates[dates.str[:4] == str(year)]
        expected = {day for day in sessions if first <= day <= last and day[:4] == str(year)}
        rejected = int(excluded.loc[observed.index].sum())
        result.append(
            {
                "year": year,
                "first_observed_session": observed.min() if len(observed) else None,
                "last_observed_session": observed.max() if len(observed) else None,
                "observed_rows": len(observed),
                "accepted_rows": len(observed) - rejected,
                "excluded_rows": rejected,
                "expected_sessions_within_observed_span": len(expected),
                "absent_sessions_within_observed_span": len(expected - set(observed)),
            }
        )
    return result
