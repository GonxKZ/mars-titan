"""Noticias originales con disponibilidad, vinculación declarada y deduplicación."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .temporal import MarketClock, aware


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def read_news(path: Path, symbol: str, clock: MarketClock) -> tuple[list[dict], list[dict]]:
    accepted, rejected, seen = [], [], set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            reason = None
            try:
                raw = json.loads(line, parse_constant=lambda _: None)
                if not isinstance(raw, dict):
                    raise ValueError("News row must be an object")
                title = _text(raw.get("Article_title", raw.get("title")))
                body = _text(raw.get("Article", raw.get("summary")))
                text = "\n".join(filter(None, [title, body]))
                date_value = _text(raw.get("Date", raw.get("datetime")))
                source_symbol = _text(raw.get("Stock_symbol", symbol)).upper().replace(".SH", ".SS")
                if source_symbol != symbol.upper():
                    reason = "symbol_mismatch"
                elif not text:
                    reason = "missing_text"
                elif not date_value:
                    reason = "missing_publication"
                if reason:
                    rejected.append({"line": line_number, "reason": reason})
                    continue
                event = datetime.fromisoformat(date_value)
                date_only = len(date_value) == 10
                if date_only:
                    available = clock.date_available(date_value)
                    published = None
                else:
                    zone = "America/New_York" if clock.market == "US" else "Asia/Shanghai"
                    # A naive source timestamp retains the documented source timezone.
                    published = aware(
                        event.replace(tzinfo=ZoneInfo(zone)) if event.tzinfo is None else event
                    )
                    available = published
                digest = hashlib.sha256(text.encode()).hexdigest()
                url = _text(raw.get("Url", raw.get("url")))
                identity = (digest, url, date_value)
                if identity in seen:
                    rejected.append({"line": line_number, "reason": "duplicate"})
                    continue
                seen.add(identity)
                accepted.append(
                    {
                        "text": text,
                        "content_hash": digest,
                        "url": url,
                        "source_date": date_value,
                        "published_at": published,
                        "available_at": available,
                        "line": line_number,
                        "availability_rule": "next_session_close"
                        if date_only
                        else "source_timestamp",
                        "symbol": symbol,
                        "association_evidence": "source_field"
                        if "Stock_symbol" in raw
                        else "source_file_only",
                    }
                )
            except (ValueError, TypeError) as error:
                rejected.append(
                    {"line": line_number, "reason": "invalid_record", "detail": str(error)}
                )
    return sorted(accepted, key=lambda r: (r["available_at"], r["content_hash"])), rejected
