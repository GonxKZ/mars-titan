"""Evidencia editorial extraída de una captura identificada del editor."""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .temporal import aware

MAX_ARTICLE_CHARS = 1024**2
MAX_CAPTURE_BYTES = 4 * 1024**2


@dataclass(frozen=True)
class ArticleEvidence:
    url: str
    title: str
    body: str
    published_at: datetime
    modified_at: datetime | None
    symbols: tuple[str, ...]
    capture_sha256: str

    def __post_init__(self):
        address = urlsplit(self.url)
        if (
            address.scheme != "https"
            or not address.hostname
            or address.username
            or address.password
        ):
            raise ValueError("La evidencia necesita una URL HTTPS sin credenciales")
        if not isinstance(self.title, str) or not self.title.strip() or len(self.title) > 8192:
            raise ValueError("La evidencia necesita un título acotado")
        if (
            not isinstance(self.body, str)
            or not self.body.strip()
            or len(self.body) > MAX_ARTICLE_CHARS
        ):
            raise ValueError("La evidencia necesita un cuerpo completo y acotado")
        if not isinstance(self.published_at, datetime):
            raise ValueError("La publicación de la evidencia debe incluir fecha y zona")
        published = aware(self.published_at)
        if self.modified_at is not None:
            if not isinstance(self.modified_at, datetime) or aware(self.modified_at) < published:
                raise ValueError("La modificación no puede preceder a la publicación")
        if not re.fullmatch(r"[0-9a-f]{64}", self.capture_sha256):
            raise ValueError("La captura necesita una huella SHA-256 válida")
        if not isinstance(self.symbols, tuple) or len(self.symbols) > 512:
            raise ValueError("Las identidades de la evidencia deben formar una tupla acotada")
        if any(
            not isinstance(symbol, str)
            or symbol in {".", ".."}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
            for symbol in self.symbols
        ):
            raise ValueError("Una identidad de la evidencia no es válida")


def _article_metadata(soup):
    articles = []
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            queue = [json.loads(tag.string or "")]
        except (ValueError, RecursionError) as error:
            raise ValueError("No se pueden interpretar los metadatos editoriales") from error
        while queue:
            item = queue.pop()
            if isinstance(item, list):
                queue.extend(item)
            elif isinstance(item, dict):
                if item.get("@type") == "NewsArticle":
                    articles.append(item)
                if "@graph" in item:
                    queue.append(item["@graph"])
    if len(articles) != 1:
        raise ValueError("La captura no identifica un único artículo editorial")
    return articles[0]


def _quote_identity(href):
    address = urlsplit(href)
    if address.netloc not in {"", "www.fool.com"}:
        return None
    match = re.fullmatch(r"/quote/(nasdaq|nyse|nysemkt|otc)/(\w[\w.\-]*)/", address.path)
    return (match[1].upper(), match[2].upper()) if match else None


def _publisher_zone(soup, metadata, published, modified):
    """Conservar el instante y acreditar la zona con la fecha visible del editor."""
    zones = []
    identity = metadata.get("author", {})
    author_path = urlsplit(identity.get("url", "")).path if isinstance(identity, dict) else ""
    for author in soup.select('a[href*="/author/"]'):
        address = urlsplit(author["href"])
        if address.netloc not in {"", "www.fool.com"} or address.path != author_path:
            continue
        box = author.find_parent("div")
        label = box.get_text(" ", strip=True) if box else ""
        match = re.search(
            r"\b(Updated|Published) ([A-Za-z]{3} \d{1,2}, \d{4} at \d{1,2}:\d{2}[AP]M) (EST|EDT)$",
            label,
        )
        if not match:
            continue
        declared = datetime.strptime(match[2], "%b %d, %Y at %I:%M%p")
        moment = modified if match[1] == "Updated" else published
        if moment is None:
            raise ValueError("La hora visible no tiene un instante acreditado en los metadatos")
        local = aware(moment).astimezone(ZoneInfo("America/New_York"))
        if (
            local.tzname() != match[3]
            or local.replace(tzinfo=None, second=0, microsecond=0) != declared
        ):
            raise ValueError("La hora visible contradice los metadatos del editor")
        zones.append(local.tzinfo)
    if zones:
        return published.astimezone(zones[0]), modified.astimezone(zones[0]) if modified else None
    return published, modified


def parse_fool(html: bytes, url: str) -> ArticleEvidence:
    """Leer el artículo, sin navegación, promociones ni cotizaciones actuales."""
    from bs4 import BeautifulSoup

    if len(html) > MAX_CAPTURE_BYTES or urlsplit(url).netloc != "www.fool.com":
        raise ValueError("El tamaño o el editor de la captura no están admitidos")
    soup = BeautifulSoup(html.decode("utf-8", errors="strict"), "html.parser")
    metadata = _article_metadata(soup)
    canonical = soup.select('link[rel="canonical"]')
    identity = metadata.get("mainEntityOfPage", {})
    if isinstance(identity, dict):
        identity = identity.get("@id")
    if (
        len(canonical) != 1
        or canonical[0].get("href") != url
        or identity != url
        or metadata.get("isAccessibleForFree") in (False, "false")
    ):
        raise ValueError("La captura no acredita una página editorial pública y única")
    bodies = soup.select("main #article-body")
    if len(bodies) != 1:
        raise ValueError("La captura no contiene un cuerpo único")
    body = bodies[0]
    if body.select("video, iframe"):
        raise ValueError("El contenido audiovisual necesita una transcripción verificada")
    for tag in body.select("script, style"):
        tag.decompose()
    for card in body.select("section.shadow-card"):
        if "Current Price" in card.get_text(" ", strip=True) and card.select('a[href*="/quote/"]'):
            card.decompose()
    symbols = set()
    for anchor in body.select('a[href*="/quote/"]'):
        identity = _quote_identity(anchor["href"])
        if identity:
            exchange, symbol = identity
            symbols.add(symbol)
            mention = anchor.find_parent("span", class_="ticker-mention")
            if mention is not None:
                quote = mention.get_text(" ", strip=True)
                if not re.fullmatch(
                    rf"\(\s*{re.escape(symbol)}\s+[+\-]?\d+(?:\.\d+)?%\s*\)", quote
                ):
                    raise ValueError("El componente de cotización contiene texto desconocido")
                mention.replace_with(f"({exchange}: {symbol})")
    disclosures = [
        paragraph
        for paragraph in soup.select("main p")
        if any(
            urlsplit(anchor["href"]).netloc in {"", "www.fool.com"}
            and urlsplit(anchor["href"]).path.lower()
            in {
                "/legal/fool-disclosure-policy/",
                "/legal/fool-disclosure-policy.aspx",
            }
            for anchor in paragraph.select("a[href]")
        )
    ]
    if len(disclosures) != 1:
        raise ValueError("No se identifica una declaración editorial completa y única")
    text = body.get_text(" ", strip=True)
    if body not in disclosures[0].parents:
        text += "\n" + disclosures[0].get_text(" ", strip=True)
    try:
        published = datetime.fromisoformat(metadata["datePublished"])
        modified = metadata.get("dateModified")
        modified = datetime.fromisoformat(modified) if modified is not None else None
        published, modified = _publisher_zone(soup, metadata, aware(published), modified)
        return ArticleEvidence(
            url,
            metadata["headline"],
            text,
            published,
            modified,
            tuple(sorted(symbols)),
            hashlib.sha256(html).hexdigest(),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("La captura contiene metadatos editoriales incompletos") from error
