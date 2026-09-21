"""Revisiones editoriales ligadas al contenido original, sin completar texto."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from .storage import sha256
from .temporal import aware


def _validate_review(review: dict) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", review["source_record_hash"]):
        raise ValueError("La revisión necesita una huella de registro válida")
    if review["status"] not in {"verified_full_article", "rejected", "unverifiable"}:
        raise ValueError("El estado de revisión no es válido")
    if not all(
        isinstance(review[name], str) and review[name].strip()
        for name in ("symbol", "source_date", "source_url", "note")
    ):
        raise ValueError("La revisión necesita procedencia y justificación")
    aware(datetime.fromisoformat(review["checked_at"]))
    if review["status"] != "verified_full_article":
        return
    url = urlsplit(review["evidence_url"])
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("La revisión necesita una URL de evidencia pública")
    if not re.fullmatch(r"[0-9a-f]{64}", review["body_sha256"]):
        raise ValueError("La revisión necesita una huella de cuerpo válida")
    if type(review["historical_version_verified"]) is not bool:
        raise ValueError("La comprobación histórica debe declararse explícitamente")
    spans = review["body_spans"]
    if not isinstance(spans, list) or not 1 <= len(spans) <= 8:
        raise ValueError("La revisión necesita intervalos de texto acotados")
    previous = 0
    for span in spans:
        if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span):
            raise ValueError("Los intervalos deben contener dos enteros")
        start, end = span
        if not previous <= start < end:
            raise ValueError("Los intervalos se solapan o están desordenados")
        previous = end


def load_reviews(path: Path) -> dict[str, dict]:
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("El manifiesto de revisiones supera el límite de 16 MiB")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("reviews"), list):
        raise ValueError("El manifiesto de revisiones no tiene el formato esperado")
    result = {}
    for review in manifest["reviews"]:
        try:
            _validate_review(review)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Hay una revisión incompleta o inválida") from error
        key = review["source_record_hash"]
        if key in result:
            raise ValueError("Hay revisiones duplicadas del mismo registro")
        result[key] = review
    return result


def reviewed_body(raw: dict, record_hash: str, reviews: dict) -> tuple[str, dict, str | None]:
    body = raw.get("Article")
    if not isinstance(body, str) or not body.strip():
        return "", {}, "missing_full_article"
    review = reviews.get(record_hash)
    if review is None:
        return "", {}, "content_unreviewed"
    if review["status"] != "verified_full_article":
        return "", {"detail": review["note"]}, f"content_{review['status']}"
    body = body.strip()
    symbol = str(raw.get("Stock_symbol", "")).strip().upper().replace(".SH", ".SS")
    fields_match = (
        symbol == review["symbol"]
        and raw.get("Date", raw.get("datetime")) == review["source_date"]
        and raw.get("Url", raw.get("url")) == review["source_url"]
    )
    previous, pieces = 0, []
    for start, end in review["body_spans"]:
        if not previous <= start < end <= len(body):
            return "", {}, "content_review_mismatch"
        pieces.append(body[start:end])
        previous = end
    selected = "\n".join(pieces)
    if (
        not selected.strip()
        or not fields_match
        or hashlib.sha256(selected.encode()).hexdigest() != review["body_sha256"]
    ):
        return "", {}, "content_review_mismatch"
    return (
        selected,
        {
            "content_review": "verified_full_article",
            "reviewed_body_sha256": review["body_sha256"],
            "review_evidence_url": review["evidence_url"],
            "review_checked_at": review["checked_at"],
            "historical_body_version_verified": review["historical_version_verified"],
        },
        None,
    )


def check_review_sources(source: Path, reviews: dict[str, dict]) -> dict:
    """Contrasta localizadores, huellas y cuerpos sin alterar las fuentes."""
    grouped = defaultdict(dict)
    for key, review in reviews.items():
        _validate_review(review)
        relative, number = review["source_file"], review["source_row"]
        path = source / relative
        if (
            Path(relative).is_absolute()
            or path.is_symlink()
            or not path.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError("La revisión sale del directorio de origen")
        if key != review["source_record_hash"] or type(number) is not int or number < 1:
            raise ValueError("El localizador de la revisión no es válido")
        if number in grouped[relative]:
            raise ValueError("Hay dos revisiones para el mismo registro")
        grouped[relative][number] = key
    source_hashes, counts = {}, Counter()
    for relative, locations in grouped.items():
        path = source / relative
        digest = sha256(path)
        found = set()
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if number not in locations:
                    continue
                key = locations[number]
                if hashlib.sha256(line.rstrip("\r\n").encode()).hexdigest() != key:
                    raise ValueError("El registro original difiere de la revisión")
                raw, review = json.loads(line), reviews[key]
                if (
                    str(raw.get("Stock_symbol", "")).strip().upper().replace(".SH", ".SS")
                    != review["symbol"]
                    or raw.get("Date") != review["source_date"]
                    or raw.get("Url") != review["source_url"]
                ):
                    raise ValueError("La procedencia original difiere de la revisión")
                if (
                    review["status"] == "verified_full_article"
                    and reviewed_body(raw, key, reviews)[2]
                ):
                    raise ValueError("El cuerpo revisado no corresponde al original")
                counts[(review["symbol"], int(review["source_date"][:4]), review["status"])] += 1
                found.add(number)
        if found != set(locations) or sha256(path) != digest:
            raise ValueError("Falta un registro o la fuente cambió durante la comprobación")
        source_hashes[relative] = digest
    return {
        "reviews": len(reviews),
        "statuses": dict(Counter(review["status"] for review in reviews.values())),
        "by_asset_year": [
            dict(symbol=s, year=y, status=status, records=n)
            for (s, y, status), n in sorted(counts.items())
        ],
        "source_hashes": source_hashes,
    }
