"""Noticias originales con disponibilidad, vinculación declarada y deduplicación."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from .temporal import MarketClock, aware


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _publication(value: str, clock: MarketClock, lag: int):
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None, clock.date_available(value, lag=lag), None
    fractions = re.finditer(r"\d{2}:\d{2}:\d{2}[.,](\d+)", value)
    if any(len(part[1]) > 6 for part in fractions):
        return None, None, "unsupported_timestamp_precision"
    offset = re.search(r"[+-](\d{2}):(\d{2})$", value)
    if offset and (int(offset[1]) > 23 or int(offset[2]) > 59):
        return None, None, "unverified_timezone"
    event = datetime.fromisoformat(value)
    if (
        event.tzinfo is None
        or not re.search(r"(?:Z|[+-]\d{2}:\d{2})$", value)
        or value.endswith("-00:00")
    ):
        return None, None, "unverified_timezone"
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,6})?(?:Z|[+-]\d{2}:\d{2})",
        value,
    ):
        return None, None, "unsupported_timestamp_precision"
    published = aware(event)
    return published, published, None


def read_news(
    path: Path, symbol: str, clock: MarketClock, *, date_only_lag: int = 1
) -> tuple[list[dict], list[dict]]:
    if type(date_only_lag) is not int or date_only_lag < 1:
        raise ValueError("El retardo debe ser un número entero positivo de sesiones")
    symbol = _text(symbol).upper().replace(".SH", ".SS")
    if not symbol:
        raise ValueError("El símbolo del activo no puede estar vacío")
    accepted, rejected, seen = [], [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            provenance = {
                "source_file": path.as_posix(),
                "line": line_number,
                "source_record_hash": hashlib.sha256(line.rstrip("\r\n").encode()).hexdigest(),
            }
            reason = None
            try:
                raw = json.loads(line, parse_constant=lambda _: None)
                if not isinstance(raw, dict):
                    raise ValueError("Cada registro de noticias debe ser un objeto")
                title = _text(raw.get("Article_title", raw.get("title")))
                body = _text(raw.get("Article", raw.get("summary")))
                text = "\n".join(filter(None, [title, body]))
                date_value = _text(raw.get("Date", raw.get("datetime")))
                provenance["source_date"] = date_value or None
                source_symbol = _text(raw.get("Stock_symbol")).upper().replace(".SH", ".SS")
                if not source_symbol:
                    reason = "missing_symbol_evidence"
                elif source_symbol != symbol:
                    reason = "symbol_mismatch"
                elif not text:
                    reason = "missing_text"
                elif not date_value:
                    reason = "missing_publication"
                if reason:
                    rejected.append({**provenance, "reason": reason})
                    continue
                published, available, reason = _publication(date_value, clock, date_only_lag)
                if reason:
                    rejected.append({**provenance, "reason": reason})
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                url = _text(raw.get("Url", raw.get("url")))
                identity = (digest, url, published.isoformat() if published else date_value)
                if identity in seen:
                    rejected.append({**provenance, "reason": "duplicate"})
                    continue
                seen.add(identity)
                accepted.append(
                    {
                        **provenance,
                        "event_id": hashlib.sha256(
                            json.dumps([symbol, *identity], separators=(",", ":")).encode()
                        ).hexdigest(),
                        "text": text,
                        "content_hash": digest,
                        "url": url,
                        "language": _text(raw.get("language", raw.get("lang"))) or None,
                        "event_at": published,
                        "published_at": published,
                        "available_at": available,
                        "availability_rule": "next_session_close"
                        if published is None
                        else "source_timestamp",
                        "symbol": symbol,
                        "association_evidence": "source_field",
                    }
                )
            except (ValueError, TypeError, OverflowError) as error:
                rejected.append({**provenance, "reason": "invalid_record", "detail": str(error)})
    return sorted(accepted, key=lambda r: (r["available_at"], r["content_hash"])), rejected
