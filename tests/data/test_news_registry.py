import hashlib
import importlib
import json
import sqlite3

import pytest

from mars_titan.data.corpus_catalog import corpus_candidates, index_news
from mars_titan.data.news_reviews import reviewed_body
from tests.data.test_corpus_catalog import source_fixture
from tests.data.test_news_sources import URL, page


def module():
    try:
        return importlib.import_module("mars_titan.data.news_registry")
    except ModuleNotFoundError:
        pytest.fail("Falta el registro recuperable de verificación")


def raw(**changes):
    return {
        "Stock_symbol": "AAA",
        "Date": "2022-06-01",
        "Url": URL,
        "Article_title": "Resultados de AAA",
        "Article": (
            "AAA (NASDAQ: AAA) ganó 2 %. Ingresos 1,5 millones\n"
            "El autor no mantiene posiciones. disclosure policy."
        ),
        **changes,
    }


def setup(tmp_path, records=None):
    records = records or [raw()]
    payload = "".join(json.dumps(row) + "\n" for row in records).encode()
    source, inventory, paths = source_fixture(tmp_path, news=payload)
    catalog = tmp_path / "catalog.sqlite"
    index_news(source, corpus_candidates(source, inventory, "US"), catalog, cutoff="2023-12-31")
    database = tmp_path / "reviews.sqlite"
    module().initialize_verification(catalog, database)
    return source, catalog, database, paths


def cached(tmp_path, monkeypatch):
    from mars_titan.data import news_fetch

    def request(url, **kwargs):
        return (
            (200, {"content-type": "text/html"}, page())
            if url == URL
            else (
                200,
                {},
                b"User-agent: *\nAllow: /\n",
            )
        )

    monkeypatch.setattr(news_fetch, "_request", request)
    root = tmp_path / "captures"
    news_fetch.ArticleFetcher(root, min_interval=0).fetch(URL)
    return root


def test_full_queue_has_no_company_or_row_sampling_and_reserves_final_test(tmp_path):
    source, catalog, database, _ = setup(tmp_path, [raw(), raw(Date="2024-06-01")])
    result = module().verification_status(database)
    assert result["records"] == 2
    assert result["states"] == {"pending": 1, "reserved": 1}
    assert result["training_ready"] is False
    assert source.exists() and catalog.exists()


def test_cached_verification_matches_legacy_body_reader(tmp_path, monkeypatch):
    _, _, database, _ = setup(tmp_path)
    root = cached(tmp_path, monkeypatch)
    report = module().verify_pending(database, root)
    assert report["states"] == {"verified_full_article": 1}
    reviews = module().reviews_for_asset(database, "AAA")
    key = hashlib.sha256(json.dumps(raw()).encode()).hexdigest()
    body, metadata, reason = reviewed_body(raw(), key, reviews)
    assert reason is None
    assert body == raw()["Article"]
    assert metadata["historical_body_version_verified"] is False


def test_operational_stop_resumes_remaining_records_without_duplicates(tmp_path, monkeypatch):
    _, _, database, _ = setup(tmp_path, [raw(), raw(Article=raw()["Article"] + " Distinto.")])
    root = cached(tmp_path, monkeypatch)
    first = module().verify_pending(database, root, stop_after=1)
    assert first["processed"] == 1 and first["states"]["pending"] == 1
    second = module().verify_pending(database, root)
    assert second["processed"] == 1
    assert second["states"] == {"unverifiable": 1, "verified_full_article": 1}
    assert module().verify_pending(database, root)["processed"] == 0


def test_shared_url_does_not_transfer_admission_to_a_different_body(tmp_path, monkeypatch):
    _, _, database, _ = setup(tmp_path, [raw(), raw(Article="AAA perdió todo.")])
    report = module().verify_pending(database, cached(tmp_path, monkeypatch))
    assert report["states"]["verified_full_article"] == 1
    assert report["states"]["unverifiable"] == 1


def test_missing_capture_remains_retryable_without_network(tmp_path, monkeypatch):
    from mars_titan.data import news_fetch

    _, _, database, _ = setup(tmp_path)
    monkeypatch.setattr(news_fetch, "_request", lambda *a, **k: pytest.fail("Red no autorizada"))
    result = module().verify_pending(database, tmp_path / "captures")
    assert result["states"] == {"retry": 1}
    assert result["reasons"] == {"missing_capture": 1}
    assert result["training_ready"] is False


def test_changed_raw_bytes_abort_instead_of_becoming_a_rejection(tmp_path, monkeypatch):
    _, _, database, paths = setup(tmp_path)
    paths["AAA", "news"].write_text(json.dumps(raw(Article="Alterado")) + "\n")
    with pytest.raises(ValueError, match="original"):
        module().verify_pending(database, cached(tmp_path, monkeypatch))
    assert module().verification_status(database)["states"] == {"pending": 1}


def test_index_change_invalidates_continuation(tmp_path):
    _, catalog, database, _ = setup(tmp_path)
    with sqlite3.connect(catalog) as db:
        db.execute("UPDATE records SET title='Alterado'")
    with pytest.raises(ValueError, match="índice"):
        module().verify_pending(database, tmp_path / "captures")


def test_manual_review_is_preserved_without_overwriting_it(tmp_path, monkeypatch):
    _, catalog, database, _ = setup(tmp_path)
    module().verify_pending(database, cached(tmp_path, monkeypatch))
    review = next(iter(module().reviews_for_asset(database, "AAA").values()))
    review["status"] = "rejected"
    review["note"] = "Rechazo manual conservado"
    manifest = tmp_path / "manual.json"
    manifest.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    second = tmp_path / "manual.sqlite"
    module().initialize_verification(catalog, second, manual_reviews=manifest)
    report = module().verify_pending(second, cached(tmp_path, monkeypatch))
    assert report["processed"] == 0
    assert next(iter(module().reviews_for_asset(second, "AAA").values())) == review


def test_registry_cannot_replace_index_or_write_inside_sources(tmp_path):
    source, catalog, _, _ = setup(tmp_path)
    for target in (catalog, source / "reviews.sqlite"):
        with pytest.raises(ValueError):
            module().initialize_verification(catalog, target)


def test_invalid_operational_limit_fails_before_processing(tmp_path):
    _, _, database, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        module().verify_pending(database, tmp_path / "captures", stop_after=0)


def test_cli_creates_checks_and_resumes_queue_without_network(tmp_path):
    import subprocess

    _, catalog, _, _ = setup(tmp_path)
    database = tmp_path / "cli.sqlite"
    commands = [
        ["news-queue", "--catalog", str(catalog), "--database", str(database)],
        ["news-status", "--database", str(database)],
        ["news-verify", "--database", str(database), "--evidence", str(tmp_path / "captures")],
    ]
    reports = []
    for arguments in commands:
        completed = subprocess.run(
            ["uv", "run", "--no-sync", "mars-data", *arguments], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        reports.append(json.loads(completed.stdout))
    assert reports[0]["states"] == reports[1]["states"] == {"pending": 1}
    assert reports[2]["states"] == {"retry": 1}


def test_chinese_summary_remains_visible_as_unresolved_provenance(tmp_path):
    payload = b'{"datetime":"2022-06-01 09:00:00","title":"Noticia","summary":"Breve"}\n'
    source, inventory, _ = source_fixture(
        tmp_path, symbols=("000001.SZ",), market="CN", news=payload
    )
    catalog, database = tmp_path / "catalog.sqlite", tmp_path / "registry.sqlite"
    index_news(source, corpus_candidates(source, inventory, "CN"), catalog, cutoff="2023-12-31")
    module().initialize_verification(catalog, database)
    result = module().verify_pending(database, tmp_path / "captures")
    assert result["states"] == {"needs_provenance": 1}
    assert result["markets"] == {"CN": 1}
    assert result["processed"] == 0


def test_publisher_hypotheses_do_not_claim_verified_urls():
    record = raw(
        Url="https://www.nasdaq.com/articles/results-2018-12-10",
        Date="2018-12-10",
        Article="The Motley Fool",
    )
    assert module().candidate_urls(record) == (
        "https://www.fool.com/investing/2018/12/10/results.aspx",
        "https://www.fool.com/investing/2018/12/10/results/",
    )
    assert module().candidate_urls({**record, "Article": "Otro editor"}) == ()


def test_invalid_url_does_not_prevent_the_next_record_from_being_verified(tmp_path, monkeypatch):
    _, _, database, _ = setup(tmp_path, [raw(Url=URL + "#comment"), raw()])
    result = module().verify_pending(database, cached(tmp_path, monkeypatch))
    assert result["processed"] == 2
    assert result["states"] == {"unverifiable": 1, "verified_full_article": 1}
    assert result["reasons"] == {"unsupported_url": 1}


def test_changed_line_ending_fails_raw_hash_even_when_text_identity_is_equal(tmp_path, monkeypatch):
    _, _, database, paths = setup(tmp_path)
    path = paths["AAA", "news"]
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r"))
    with pytest.raises(ValueError, match="original"):
        module().verify_pending(database, cached(tmp_path, monkeypatch))


def test_manual_review_does_not_authorize_same_record_under_another_company(tmp_path):
    source, inventory, _ = source_fixture(
        tmp_path, symbols=("AAA", "BBB"), news=(json.dumps(raw()) + "\n").encode()
    )
    catalog, database = tmp_path / "index.sqlite", tmp_path / "reviews.sqlite"
    index_news(source, corpus_candidates(source, inventory, "US"), catalog, cutoff="2023-12-31")
    review = {
        "source_record_hash": hashlib.sha256(json.dumps(raw()).encode()).hexdigest(),
        "status": "rejected",
        "symbol": "AAA",
        "source_date": "2022-06-01",
        "source_url": URL,
        "source_file": "text/sp500_news/AAA.jsonl",
        "source_row": 1,
        "note": "Revisión ligada al activo y localizador",
        "checked_at": "2026-09-22T00:00:00Z",
    }
    manifest = tmp_path / "manual.json"
    manifest.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    module().initialize_verification(catalog, database, manual_reviews=manifest)
    assert module().reviews_for_asset(database, "AAA")
    assert module().reviews_for_asset(database, "BBB") == {}
    review["source_row"] = 2
    manifest.write_text(json.dumps({"schema_version": 1, "reviews": [review]}))
    with pytest.raises(ValueError, match="revisión manual"):
        module().initialize_verification(
            catalog, tmp_path / "wrong-location.sqlite", manual_reviews=manifest
        )
