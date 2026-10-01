"""Contraste contable con evidencia primaria, sin fechar los cierres como anuncios."""

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, localcontext
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .temporal import aware

CONCEPTS = {
    "total_assets": ("Assets", "stock"),
    "total_liab": ("Liabilities", "stock"),
    "total_hldr_eqy_exc_min_int": ("EquityAttributableToParent", "stock"),
    "total_hldr_eqy_inc_min_int": ("EquityIncludingNoncontrollingInterest", "stock"),
    "revenue": ("Revenue", "flow"),
    "n_income_attr_p": ("NetIncomeAttributableToParent", "flow"),
    "n_cashflow_act": ("OperatingCashFlow", "flow"),
}


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("La fecha debe tener formato de calendario ISO")
    return date.fromisoformat(value)


def _symbol(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{6}\.(SH|SS|SZ)", value):
        raise ValueError("El instrumento necesita código y mercado explícitos")
    return value.replace(".SH", ".SS")


def _number(value):
    if isinstance(value, bool) or value is None or len(str(value)) > 80:
        raise ValueError("La cifra no es un número contable acotado")
    result = Decimal(str(value))
    if not result.is_finite() or result.adjusted() > 24 or result.adjusted() < -12:
        raise ValueError("La cifra es no finita o queda fuera del rango contable admitido")
    return result


def _publication(evidence):
    if evidence.get("publication_kind") != "actual":
        raise ValueError("Una fecha prevista no acredita publicación")
    day = _day(evidence["publication_date"])
    timestamp = evidence.get("published_at")
    if timestamp is not None:
        if (
            not isinstance(timestamp, str)
            or timestamp.endswith("-00:00")
            or not re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
                r"(?:[.,][0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
                timestamp,
            )
        ):
            raise ValueError("La hora necesita zona acreditada y precisión representable")
        moment = datetime.fromisoformat(timestamp)
        utc = aware(moment)
        if utc.astimezone(ZoneInfo("Asia/Shanghai")).date() != day:
            raise ValueError("La fecha declarada y el instante de publicación no coinciden")
        timestamp = utc.isoformat()
    return day, timestamp


def _provenance(evidence):
    address = urlsplit(evidence["source_url"])
    if (
        address.scheme != "https"
        or address.netloc != "static.cninfo.com.cn"
        or not address.path.lower().endswith(".pdf")
        or address.query
        or address.fragment
    ):
        raise ValueError("El documento no tiene una dirección primaria admitida")
    for name in ("document_sha256", "publication_evidence_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", evidence[name]):
            raise ValueError("La evidencia necesita huellas del documento y de su anuncio")
    if type(evidence["source_page"]) is not int or evidence["source_page"] < 1:
        raise ValueError("La cifra necesita una página de origen válida")
    if not isinstance(evidence["report_id"], str) or not 1 <= len(evidence["report_id"]) <= 128:
        raise ValueError("Falta un identificador de presentación acotado")
    expected = f"/finalpage/{evidence['publication_date']}/{evidence['report_id']}.pdf"
    if address.path.lower() != expected.lower():
        raise ValueError("El documento no corresponde al anuncio y la fecha declarados")


def reconcile_chinese_fact(raw: dict, evidence: dict) -> dict:
    """Contrastar una cifra con una revisión explícita, sin certificar por sí solo el PDF.

    La revisión suministrada debe proceder de la lectura del documento y del
    anuncio. El resultado conserva el contexto de esa fuente primaria. No afirma
    recuperar metadatos que se eliminaron de la copia original.
    """
    if not evidence.get("publication_date"):
        return {"status": "missing_publication_evidence"}
    try:
        published, timestamp = _publication(evidence)
        _provenance(evidence)
        symbol = _symbol(raw["ts_code"])
        if symbol != _symbol(evidence["symbol"]):
            return {"status": "identity_mismatch"}
        period_end = _day(evidence["period_end"])
        source_end = datetime.strptime(raw["end_date"], "%Y%m%d").date()
        if period_end != source_end or period_end > published:
            return {"status": "period_mismatch"}
        if evidence["currency"] != "CNY" or raw.get("currency", "CNY") != "CNY":
            return {"status": "currency_mismatch"}
        if (
            evidence["accounting_standard"] != "CAS"
            or raw.get("accounting_standard", "CAS") != "CAS"
        ):
            return {"status": "accounting_standard_mismatch"}
        scope = evidence["statement_scope"]
        if scope not in {"consolidated", "separate"}:
            raise ValueError("Falta el perímetro de la presentación")
        if raw.get("statement_scope", scope) != scope:
            return {"status": "statement_scope_mismatch"}
        field = evidence["field"]
        concept, kind = CONCEPTS[field]
        start = evidence.get("period_start")
        if evidence["quantity_kind"] != kind or (kind == "stock" and start is not None):
            return {"status": "quantity_kind_mismatch"}
        if kind == "flow" and (start is None or _day(start) > period_end):
            return {"status": "missing_flow_interval"}
        if raw.get("period_start", start) != start:
            return {"status": "period_mismatch"}
        multiplier = _number(evidence["unit_multiplier"])
        if multiplier not in {
            Decimal(1),
            Decimal(1000),
            Decimal(10000),
            Decimal(10**6),
            Decimal(10**8),
        }:
            return {"status": "unsupported_unit"}
        reported = _number(evidence["value"])
        with localcontext() as context:
            context.prec = len(reported.as_tuple().digits) + len(multiplier.as_tuple().digits)
            value = reported * multiplier
            if _number(raw[field]) != value:
                return {"status": "value_mismatch"}
        fact = {
            key: evidence[key]
            for key in (
                "statement_scope",
                "accounting_standard",
                "currency",
                "period_end",
                "publication_date",
                "source_url",
                "document_sha256",
                "publication_evidence_sha256",
                "report_id",
                "source_page",
            )
        }
        fact.update(
            symbol=symbol,
            field=field,
            concept=f"cn-reported:{concept}:CNY",
            quantity_kind=kind,
            period_start=start,
            published_at=timestamp,
            value_cny=format(value, "f"),
        )
        return {"status": "reconciled", "fact": fact, "original_context_recovered": False}
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError) as error:
        return {"status": "invalid_evidence", "detail": str(error)}
