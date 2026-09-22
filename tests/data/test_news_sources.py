import hashlib
import importlib
import json
from datetime import UTC, datetime

import pytest

URL = "https://www.fool.com/investing/2022/06/01/results/"


def page(*, body=None, metadata=None, disclosure=True):
    article = {
        "@type": "NewsArticle",
        "headline": "Resultados de AAA",
        "mainEntityOfPage": {"@id": URL},
        "author": {"url": "https://www.fool.com/author/1/"},
        "datePublished": "2022-06-01T12:00:00Z",
        "dateModified": "2022-06-01T12:01:00Z",
        "about": [{"@type": "Corporation", "tickerSymbol": "NASDAQ AAA"}],
        **(metadata or {}),
    }
    if body is None:
        body = (
            '<p>AAA <span class="ticker-mention">(<a href="/quote/nasdaq/aaa/">AAA</a> '
            '<span>+90.1%</span>)</span> ganó 2 %.</p><div class="table-responsive">'
            "<table><tr><th>Ingresos</th><td>1,5 millones</td></tr></table></div>"
        )
    note = (
        "<p><em>El autor no mantiene posiciones. "
        '<a href="/legal/fool-disclosure-policy/">disclosure policy</a>.</em></p>'
        if disclosure
        else ""
    )
    return (
        f'<html><head><link rel="canonical" href="{URL}">'
        f'<script type="application/ld+json">{json.dumps(article)}</script></head><body>'
        '<nav><a href="/quote/nasdaq/WRONG/">WRONG</a></nav><main>'
        f'<div id="article-body">{body}</div>{note}</main></body></html>'
    ).encode()


def parse(html=None, url=URL):
    module = importlib.import_module("mars_titan.data.news_sources")
    function = getattr(module, "parse_fool", None)
    assert function is not None, "Falta el adaptador de capturas editoriales"
    return function(page() if html is None else html, url)


def test_parser_preserves_table_disclosure_and_original_capture_hash():
    html = page()
    actual = parse(html)
    assert actual.title == "Resultados de AAA"
    assert "1,5 millones" in actual.body
    assert "no mantiene posiciones" in actual.body
    assert "NASDAQ: AAA" in actual.body
    assert "90.1" not in actual.body
    assert actual.symbols == ("AAA",)
    assert actual.published_at == datetime(2022, 6, 1, 12, tzinfo=UTC)
    assert actual.capture_sha256 == hashlib.sha256(html).hexdigest()


def test_live_quote_card_does_not_become_editorial_text_or_entity():
    html = page(
        body='<p>Texto conservado.</p><section class="shadow-card">'
        '<a href="/quote/nyse/ZZZ/">ZZZ</a>Current Price $999</section>'
    )
    actual = parse(html)
    assert "999" not in actual.body
    assert "Texto conservado" in actual.body
    assert actual.symbols == ()


@pytest.mark.parametrize(
    "body",
    ['<video src="movie.mp4"></video>', '<iframe src="https://example.test/video"></iframe>'],
)
def test_media_without_verified_transcript_is_not_complete(body):
    with pytest.raises(ValueError, match="transcripción"):
        parse(page(body=body))


@pytest.mark.parametrize(
    "changes",
    [
        {"datePublished": "2022-06-01T12:00:00"},
        {"datePublished": None},
        {"mainEntityOfPage": {"@id": "https://www.fool.com/other/"}},
        {"isAccessibleForFree": False},
    ],
)
def test_ambiguous_or_inaccessible_metadata_is_not_evidence(changes):
    with pytest.raises(ValueError):
        parse(page(metadata=changes))


def test_missing_disclosure_is_not_silently_omitted():
    with pytest.raises(ValueError, match="declaración"):
        parse(page(disclosure=False))


def test_duplicate_article_body_is_not_guessed():
    with pytest.raises(ValueError):
        parse(page().replace(b"</main>", b'<div id="article-body">Otro</div></main>'))


def test_unknown_host_and_oversized_capture_are_rejected():
    with pytest.raises(ValueError):
        parse(url="https://example.test/article")
    with pytest.raises(ValueError):
        parse(b" " * (4 * 1024**2 + 1))


def test_script_and_comment_text_never_enter_article():
    actual = parse(page(body="<p>Texto.<script>robar()</script><!-- secreto --></p>"))
    assert "robar" not in actual.body and "secreto" not in actual.body


def test_unknown_card_style_does_not_authorize_deleting_editorial_text():
    actual = parse(page(body='<section class="shadow-card">El ingreso cayó un 40 %.</section>'))
    assert "cayó un 40 %" in actual.body


def test_foreign_quote_link_does_not_accredit_a_company():
    actual = parse(page(body='<a href="https://example.test/quote/nasdaq/AAA/">AAA</a>'))
    assert actual.symbols == ()


def test_unknown_text_inside_ticker_component_is_not_discarded():
    html = page(
        body='<span class="ticker-mention">(<a href="/quote/nasdaq/aaa/">AAA</a>) '
        "Perdió 90 %.</span>"
    )
    with pytest.raises(ValueError, match="cotización"):
        parse(html)


def test_observed_legacy_disclosure_link_keeps_the_complete_declaration():
    html = page().replace(
        b"/legal/fool-disclosure-policy/", b"http://www.fool.com/Legal/fool-disclosure-policy.aspx"
    )
    actual = parse(html)
    assert "El autor no mantiene posiciones." in actual.body


def test_publisher_timezone_is_used_only_when_the_byline_confirms_metadata():
    html = page(
        metadata={"datePublished": "2022-06-02T03:30:00Z", "dateModified": "2022-06-02T03:31:00Z"}
    )
    html = html.replace(
        b"</body>",
        b'<div hidden id="S:1">By <a href="/author/1/">Autor</a> '
        b"Updated Jun 1, 2022 at 11:31PM EDT</div></body>",
    )
    actual = parse(html)
    assert actual.published_at.date().isoformat() == "2022-06-01"
    assert actual.published_at.astimezone(UTC).isoformat() == "2022-06-02T03:30:00+00:00"


def test_byline_cannot_override_a_conflicting_instant():
    html = page().replace(
        b"<main>",
        b'<main><div>By <a href="/author/1/">Autor</a> Updated Jun 1, 2022 at 11:31PM EDT</div>',
    )
    with pytest.raises(ValueError, match="hora"):
        parse(html)
