"""Identidades y disponibilidad de los seis conceptos chinos publicados en comunicados."""

import calendar
import re
from datetime import date
from decimal import Decimal
from urllib.parse import urlsplit

NBS_INDEX = "https://www.stats.gov.cn/english/PressRelease/"
PBOC_INDEX = "https://www.pbc.gov.cn/en/3688247/3688975/index.html"
RELEASE_CONCEPTS = {
    "cn_cpi_yoy_published": ("NBS", "percent_yoy", NBS_INDEX),
    "cn_industrial_yoy_published": ("NBS", "percent_yoy", NBS_INDEX),
    "cn_manufacturing_pmi": ("NBS/CFLP", "diffusion_index_50_neutral", NBS_INDEX),
    "cn_m2_stock": ("PBOC", "billion_CNY_normalized_from_release", PBOC_INDEX),
    "cn_tsf_flow": ("PBOC", "billion_CNY_per_month_normalized_from_release", PBOC_INDEX),
    "cn_tsf_stock": ("PBOC", "billion_CNY_normalized_from_release", PBOC_INDEX),
}


def release_source(source_url: str) -> tuple[str, date | None]:
    """Aceptar únicamente rutas de los archivos consultados, sin redirecciones implícitas."""
    url = urlsplit(source_url)
    if (
        url.scheme != "https"
        or url.username
        or url.password
        or url.port
        or url.query
        or url.fragment
    ):
        raise ValueError("La URL del comunicado no cumple el contrato HTTPS")
    if url.hostname == "www.stats.gov.cn":
        match = re.fullmatch(r"/english/PressRelease/(\d{6})/t(\d{8})_\d+\.html", url.path)
        if match and match.group(1) == match.group(2)[:6]:
            return "nbs", date.fromisoformat(match.group(2))
    if url.hostname == "www.pbc.gov.cn" and re.fullmatch(
        r"/diaochatongjisi/116219/116225/[a-z0-9]{1,40}/index\.html", url.path
    ):
        return "pboc", None
    if url.hostname == "www.gdjr.gov.cn" and re.fullmatch(
        r"/gdjr/(?:jrzx/jryw|zwgk/zdly/sjfb/tjsj)/content/post_[0-9]+\.html", url.path
    ):
        return "pboc_reprint", None
    raise ValueError("La URL no pertenece a un archivo oficial de comunicados admitido")


def release_exclusion(entry: dict) -> str | None:
    """Validar la identidad sin presentar el identificador interno como una API oficial."""
    identity = RELEASE_CONCEPTS.get(entry.get("id"))
    if identity is None:
        return "unverified_official_release_concept"
    provider, unit, source = identity
    expected = dict(
        kind="raw",
        provider=provider,
        unit=unit,
        source_url=source,
        series_id="no identifier verified",
        frequency="M",
        vintage_policy="DATED_OFFICIAL_RELEASES",
        availability_rule="OFFICIAL_RELEASE_BOUND_THEN_NEXT_SESSION",
        verification_status="verified_official_release_archive",
    )
    return (
        "unverified_official_release_concept"
        if any(entry.get(k) != v for k, v in expected.items())
        else None
    )


def validate_release_observation(entry: dict, row: dict, original_start: date) -> None:
    """Rechazar unidades, periodos y límites de publicación incompatibles con el documento."""
    if entry.get("vintage_policy") != "DATED_OFFICIAL_RELEASES":
        return
    if release_exclusion(entry):
        raise ValueError("El catálogo no acredita este concepto de comunicado oficial")
    family, path_day = release_source(row.get("source_url", ""))
    is_nbs = entry["provider"].startswith("NBS")
    if is_nbs != (family == "nbs"):
        raise ValueError("El comunicado no corresponde al proveedor del indicador")
    try:
        published = date.fromisoformat(row["declared_publication_date"])
        period = date.fromisoformat(row["period_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Faltan fechas válidas del periodo y de su publicación") from error
    last = date(period.year, period.month, calendar.monthrange(period.year, period.month)[1])
    expected_policy = (
        "latest_declared_or_archive_day_then_next_session"
        if is_nbs
        else "declared_publication_day_then_next_session"
    )
    if (
        row.get("source_timezone") != "Asia/Shanghai"
        or row.get("publication_timestamp_verified") is not False
        or row.get("availability_policy") != expected_policy
        or row.get("native_unit") != entry["unit"]
        or row.get("reference_period_end") != last.isoformat()
        or period.day != 1
        or last > published
        or original_start != max(published, path_day or published)
        or is_nbs
        and row.get("archive_path_date") != path_day.isoformat()
    ):
        raise ValueError("Los metadatos del comunicado no acreditan su disponibilidad y unidad")
    if not is_nbs:
        unit = row.get("source_unit")
        amount = row.get("source_value")
        if (
            unit not in {"万亿元", "亿元"}
            or not isinstance(amount, str)
            or not re.fullmatch(r"-?\d{1,16}(?:\.\d{1,10})?", amount)
        ):
            raise ValueError("Falta la cifra monetaria original con su unidad")
        normalized = float(
            Decimal(amount) * (Decimal(1000) if unit == "万亿元" else Decimal("0.1"))
        )
        if row["value"] != normalized or entry["id"] != "cn_tsf_flow" and normalized <= 0:
            raise ValueError("La cifra normalizada no conserva el valor monetario publicado")
