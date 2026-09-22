"""Contraste literal de artículos completos, sin completar ni parafrasear cuerpos."""

import hashlib
import re
import unicodedata
from datetime import datetime

from .news_sources import MAX_ARTICLE_CHARS, ArticleEvidence


def _tokens(text):
    """Normalizar tipografía sin suprimir cifras, signos, palabras o su orden."""
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFC", text))


_PROMOTION = re.compile(
    r"^10 stocks we like better than [^\n]+\n.*?"
    r"^\*Stock Advisor returns as of [^\n]+(?:\n|$)",
    re.MULTILINE | re.DOTALL,
)
_NOTICE = (
    "The views and opinions expressed herein are the views and opinions of the author "
    "and do not necessarily reflect those of Nasdaq, Inc."
)
# Plantillas contrastadas en el manifiesto de revisiones, sin copiar su contenido.
PROMOTION_TEMPLATES = frozenset(
    {
        "7f05eb760e8958561cbb5b99747ab69c1e852883ff15de6b125e8ef9e2103a22",
        "bdf025267bf43349133ef381bb97c8848486bbff90ca37939fde73319c3b96a1",
        "b91a889d1fba19b0337c4d4fcb4b6b8c49d8a8f523c63e6ffa9624084d74f7c2",
        "9cde1eaa07f1ff3eda94b0ad7ee56f5d5d3a1e43286b4cc99577cb1ed8966ea2",
        "025a8e694765283432aa035df0f1fdca4893d51a8c0100e49d2a6b6c23cd0130",
    }
)


def _known_promotion(block):
    company = block.splitlines()[0].removeprefix("10 stocks we like better than ").strip()
    if not company:
        return False
    template = " ".join(unicodedata.normalize("NFC", block.replace(company, "{company}")).split())
    template = re.sub(
        r"\*Stock Advisor returns as of [A-Za-z]+ \d{1,2}, \d{4}$",
        "*Stock Advisor returns as of {date}",
        template,
    )
    return hashlib.sha256(template.encode()).hexdigest() in PROMOTION_TEMPLATES


def _editorial_spans(body):
    """Excluir únicamente bloques publicitarios delimitados y el aviso final conocido."""
    end = len(body)
    while body[:end].rstrip().endswith(_NOTICE):
        end = len(body[:end].rstrip()) - len(_NOTICE)
    spans = []
    start = 0
    for match in _PROMOTION.finditer(body, 0, end):
        if not _known_promotion(match[0]):
            continue
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, end))
    trimmed = []
    for start, end in spans:
        part = body[start:end]
        left = start + len(part) - len(part.lstrip())
        right = end - len(part) + len(part.rstrip())
        if left < right:
            trimmed.append([left, right])
    return trimmed


def _unverified(reason, note):
    return {"status": "unverifiable", "reason": reason, "note": note}


def compare_article(raw: dict, evidence: ArticleEvidence) -> dict:
    """Comprobar correspondencia editorial actual, sin certificar una versión histórica."""
    body = raw.get("Article")
    if not isinstance(body, str) or not body.strip():
        return _unverified("missing_full_article", "El original no contiene un cuerpo completo")
    body = body.strip()
    if len(body) > MAX_ARTICLE_CHARS:
        return _unverified("body_too_large", "El cuerpo supera el presupuesto de contraste")
    symbol = raw.get("Stock_symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return _unverified("missing_symbol_evidence", "Falta una identidad declarada del activo")
    symbol = symbol.strip().upper().replace(".SH", ".SS")
    if symbol not in evidence.symbols:
        return _unverified("unconfirmed_entity", "La fuente no acredita la relación con el activo")
    title = raw.get("Article_title", raw.get("title"))
    if not isinstance(title, str) or _tokens(title) != _tokens(evidence.title):
        return _unverified("title_mismatch", "El título no coincide con la evidencia consultada")
    declared = raw.get("Date", raw.get("datetime"))
    try:
        day = datetime.fromisoformat(declared).date()
    except (ValueError, TypeError):
        return _unverified("invalid_publication", "No hay una fecha declarada interpretable")
    if day != evidence.published_at.date():
        return _unverified(
            "publication_mismatch", "La publicación no coincide con la fecha declarada"
        )
    if (
        evidence.modified_at is not None
        and evidence.modified_at.astimezone(evidence.published_at.tzinfo).date() > day
    ):
        return _unverified("later_revision", "La fuente conserva una modificación posterior")
    spans = [[0, len(body)]]
    selected = body
    expected = _tokens(evidence.body)
    if _tokens(selected) != expected:
        spans = _editorial_spans(body)
        selected = "\n".join(body[start:end] for start, end in spans)
    if not 1 <= len(spans) <= 8 or _tokens(selected) != expected:
        return _unverified("body_mismatch", "No coincide todo el cuerpo editorial")
    return {
        "status": "verified_full_article",
        "symbol": symbol,
        "source_date": declared,
        "source_url": raw.get("Url", raw.get("url")),
        "evidence_url": evidence.url,
        "evidence_capture_sha256": evidence.capture_sha256,
        "evidence_published_at": evidence.published_at.isoformat(),
        "evidence_modified_at": evidence.modified_at.isoformat() if evidence.modified_at else None,
        "body_spans": spans,
        "body_sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "historical_version_verified": False,
        "note": (
            "Correspondencia del cuerpo completo tras normalizar Unicode y espacios. "
            "La captura no acredita una versión histórica inmutable."
        ),
    }
