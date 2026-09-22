import hashlib
import importlib
from datetime import UTC, datetime, timedelta

import pytest


def modules():
    try:
        return (
            importlib.import_module("mars_titan.data.news_sources"),
            importlib.import_module("mars_titan.data.news_verification"),
        )
    except ModuleNotFoundError:
        pytest.fail("Falta el contraste reproducible de evidencia editorial")


def raw_record(**changes):
    return {
        "Stock_symbol": "AAA",
        "Date": "2022-06-01",
        "Article_title": "Resultados de AAA",
        "Article": "El ingreso fue de 1,5 millones.\n\nEl margen fue del 2 %.",
        "Url": "https://www.nasdaq.com/articles/aaa-results",
        **changes,
    }


def evidence(**changes):
    source, _ = modules()
    values = {
        "url": "https://www.fool.com/investing/2022/06/01/aaa-results/",
        "title": "Resultados de AAA",
        "body": "El ingreso fue de 1,5 millones.\n\nEl margen fue del 2 %.",
        "published_at": datetime(2022, 6, 1, 12, tzinfo=UTC),
        "modified_at": datetime(2022, 6, 1, 13, tzinfo=UTC),
        "symbols": ("AAA",),
        "capture_sha256": hashlib.sha256(b"captura de prueba").hexdigest(),
        **changes,
    }
    return source.ArticleEvidence(**values)


def compare(raw=None, proof=None):
    _, verifier = modules()
    return verifier.compare_article(raw or raw_record(), proof or evidence())


def test_complete_body_produces_original_spans_and_hash():
    raw = raw_record()
    result = compare(raw)
    assert result["status"] == "verified_full_article"
    assert result["body_spans"] == [[0, len(raw["Article"])]]
    assert result["body_sha256"] == hashlib.sha256(raw["Article"].encode()).hexdigest()
    assert result["historical_version_verified"] is False


@pytest.mark.parametrize(
    "body",
    [
        "El ingreso fue de 15 millones.\n\nEl margen fue del 2 %.",
        "El ingreso no fue de 1,5 millones.\n\nEl margen fue del 2 %.",
        "El ingreso fue de -1,5 millones.\n\nEl margen fue del 2 %.",
        "El ingreso fue de 1,5 miles.\n\nEl margen fue del 2 %.",
        "El ingreso fue de 1,5 millones.",
        "El margen fue del 2 %.\n\nEl ingreso fue de 1,5 millones.",
    ],
)
def test_altered_numbers_negation_units_or_missing_body_are_not_verified(body):
    assert compare(raw_record(Article=body))["status"] != "verified_full_article"


def test_spacing_does_not_rewrite_selected_text():
    raw = raw_record(Article="El ingreso   fue de 1,5 millones.\nEl margen fue del 2 %.")
    result = compare(raw)
    assert result["status"] == "verified_full_article"
    assert result["body_sha256"] == hashlib.sha256(raw["Article"].encode()).hexdigest()


def test_word_boundaries_are_preserved():
    raw = raw_record(Article="El cambio fue notable.")
    proof = evidence(body="El cambio fue not able.")
    assert compare(raw, proof)["status"] != "verified_full_article"


def test_unknown_inserted_paragraph_is_not_silently_removed():
    raw = raw_record(
        Article="El ingreso fue de 1,5 millones.\nTodo lo anterior es falso.\n"
        "El margen fue del 2 %."
    )
    assert compare(raw)["status"] != "verified_full_article"


@pytest.mark.parametrize(
    "changes",
    [
        {"title": "Resultados de otra empresa"},
        {"symbols": ("OTHER",)},
        {"symbols": ()},
        {"published_at": datetime(2021, 6, 1, 12, tzinfo=UTC)},
        {"modified_at": datetime(2022, 6, 2, 12, tzinfo=UTC)},
    ],
)
def test_identity_or_version_mismatch_is_not_verified(changes):
    assert compare(proof=evidence(**changes))["status"] != "verified_full_article"


def test_summary_is_not_a_complete_original_article():
    raw = raw_record()
    raw["summary"] = raw.pop("Article")
    assert compare(raw)["status"] != "verified_full_article"


@pytest.mark.parametrize(
    "changes",
    [
        {"capture_sha256": "incorrecta"},
        {"url": "http://example.test/article"},
        {"url": "https://user:password@example.test/article"},
        {"published_at": datetime(2022, 6, 1)},
        {"modified_at": datetime(2022, 6, 1, 12, tzinfo=UTC) - timedelta(days=1)},
        {"body": ""},
    ],
)
def test_incomplete_evidence_fails_before_a_verification(changes):
    with pytest.raises(ValueError):
        evidence(**changes)


def test_canonical_unicode_equivalence_keeps_original_bytes():
    raw = raw_record(Article="E\u0301xito comercial.")
    result = compare(raw, evidence(body="Éxito comercial."))
    assert result["status"] == "verified_full_article"
    assert result["body_sha256"] == hashlib.sha256(raw["Article"].encode()).hexdigest()


def test_equivalent_timezone_does_not_create_a_later_calendar_revision():
    published = datetime.fromisoformat("2022-06-01T23:30:00-04:00")
    modified = datetime.fromisoformat("2022-06-02T03:31:00+00:00")
    assert (
        compare(proof=evidence(published_at=published, modified_at=modified))["status"]
        == "verified_full_article"
    )


def test_known_promotion_is_removed_but_disclosure_is_kept(monkeypatch):
    main = "El ingreso fue de 1,5 millones."
    disclosure = "El autor no mantiene posiciones."
    promotion = (
        "\n10 stocks we like better than AAA\nContenido promocional.\n"
        "*Stock Advisor returns as of June 1, 2026\n"
    )
    notice = (
        "The views and opinions expressed herein are the views and opinions of the author "
        "and do not necessarily reflect those of Nasdaq, Inc."
    )
    _, verifier = modules()
    template = (
        "10 stocks we like better than {company} Contenido promocional. "
        "*Stock Advisor returns as of {date}"
    )
    monkeypatch.setattr(
        verifier,
        "PROMOTION_TEMPLATES",
        {hashlib.sha256(template.encode()).hexdigest()},
        raising=False,
    )
    raw = raw_record(Article=main + promotion + disclosure + "\n" + notice)
    proof = evidence(body=main + "\n\n" + disclosure)
    result = compare(raw, proof)
    assert result["status"] == "verified_full_article"
    expected_start = len(main) + len(promotion)
    assert result["body_spans"] == [
        [0, len(main)],
        [expected_start, expected_start + len(disclosure)],
    ]
    assert result["body_sha256"] == hashlib.sha256((main + "\n" + disclosure).encode()).hexdigest()


def test_unknown_promotion_shape_does_not_allow_arbitrary_removal():
    raw = raw_record(
        Article="El ingreso fue de 1,5 millones.\n10 stocks we like better than AAA\n"
        "La cifra anterior era incorrecta.\nEl margen fue del 2 %."
    )
    assert compare(raw)["status"] != "verified_full_article"


def test_case_sensitive_units_are_not_equivalent():
    assert (
        compare(
            raw_record(Article="La potencia fue de 1 MW."),
            evidence(body="La potencia fue de 1 mW."),
        )["status"]
        != "verified_full_article"
    )


def test_complete_promotion_markers_do_not_hide_an_unknown_correction():
    raw = raw_record(
        Article="El ingreso fue de 1,5 millones.\n"
        "10 stocks we like better than AAA\n"
        "Corrección editorial: el ingreso anterior es falso.\n"
        "*Stock Advisor returns as of June 1, 2026\n"
        "El autor no mantiene posiciones."
    )
    proof = evidence(body="El ingreso fue de 1,5 millones.\nEl autor no mantiene posiciones.")
    assert compare(raw, proof)["status"] != "verified_full_article"
