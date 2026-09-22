import hashlib
import importlib
import json
import os
import sqlite3
import subprocess

import pytest

from mars_titan.data.inventory import inventory


def module():
    try:
        return importlib.import_module("mars_titan.data.corpus_catalog")
    except ModuleNotFoundError:
        pytest.fail("Falta el catálogo completo y recuperable del corpus")


def article(symbol="AAA", **changes):
    return {
        "Stock_symbol": symbol,
        "Date": "2022-06-01",
        "Article_title": "Resultados publicados",
        "Article": "La empresa publicó sus resultados.",
        "Url": "https://www.nasdaq.com/articles/results",
        **changes,
    }


def source_fixture(tmp_path, *, symbols=("AAA",), market="US", news=None, missing=()):
    source = tmp_path / "dataset"
    source.mkdir(exist_ok=True)
    paths = {}
    for symbol in symbols:
        if market == "US":
            locations = {
                "prices": f"time_series/S&P500_time_series/{symbol.lower()}.csv",
                "news": f"text/sp500_news/{symbol}.jsonl",
                "fundamentals": f"table/financial_reports/{symbol.lower()}/facts.json",
                "charts": f"image/S&P500_image/{symbol.lower()}/{symbol}_2022_H1_candlestick.png",
            }
        else:
            locations = {
                "prices": f"time_series/HS300_time_series/{symbol}.csv",
                "news": f"text/hs300news_summary/{symbol}_empresa.jsonl",
                "fundamentals": f"table/hs300_tabular/{symbol}/facts.json",
                "charts": f"image/HS300_image/{symbol}/{symbol}_2022_H1_candlestick.png",
            }
        for modality, relative in locations.items():
            if (symbol, modality) in missing:
                continue
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = {
                "prices": b"Date,Open,High,Low,Close,Volume\n2022-06-01,1,2,1,2,10\n",
                "fundamentals": b"[]",
                "charts": b"\x89PNG\r\n\x1a\n" + bytes(24),
                "news": news
                if news is not None
                else (json.dumps(article(symbol), ensure_ascii=False) + "\n").encode(),
            }[modality]
            path.write_bytes(content)
            paths[symbol, modality] = path
    database = tmp_path / "inventory.sqlite"
    inventory(source, database)
    return source, database, paths


def rows(database):
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM records ORDER BY file_id, ordinal")]


def test_catalog_keeps_unclassified_and_incomplete_assets(tmp_path):
    source, catalog, _ = source_fixture(
        tmp_path, symbols=("AAA", "BBB", "CCC"), missing=(("CCC", "fundamentals"),)
    )
    assets = module().corpus_candidates(source, catalog, "US")
    assert [asset["symbol"] for asset in assets] == ["AAA", "BBB", "CCC"]
    assert [asset["has_all_sources"] for asset in assets] == [True, True, False]
    assert assets[2]["missing_modalities"] == ["fundamentals"]


def test_catalog_has_no_64_asset_limit(tmp_path):
    source, catalog, _ = source_fixture(tmp_path, symbols=tuple(f"A{n:03}" for n in range(130)))
    assert len(module().corpus_candidates(source, catalog, "US")) == 130


def test_missing_inventory_is_not_created(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises((ValueError, FileNotFoundError)):
        module().corpus_candidates(tmp_path, missing, "US")
    assert not missing.exists()


def test_catalog_rejects_another_source_root(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="origen"):
        module().corpus_candidates(source.parent, catalog, "US")


def test_catalog_rejects_inconsistent_inventory_identity(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    with sqlite3.connect(catalog) as db:
        key, payload = db.execute("SELECT path,record FROM files LIMIT 1").fetchone()
        record = json.loads(payload)
        record["symbol"] = "WRONG"
        db.execute("UPDATE files SET record=? WHERE path=?", (json.dumps(record), key))
    with pytest.raises(ValueError, match="identidad"):
        module().corpus_candidates(source, catalog, "US")


def test_index_keeps_distinct_bodies_and_exact_byte_locations(tmp_path):
    lines = [
        json.dumps(article(Article=body), ensure_ascii=False)
        for body in ("Texto uno", "Texto dos", "Texto uno")
    ]
    encoded = [line.encode() + b"\r\n" for line in lines]
    raw = b"\xef\xbb\xbf" + b"".join(encoded)
    source, catalog, _ = source_fixture(tmp_path, news=raw)
    destination = tmp_path / "news.sqlite"
    report = module().index_news(
        source, module().corpus_candidates(source, catalog, "US"), destination, cutoff="2023-12-31"
    )
    actual = rows(destination)
    assert len(actual) == 3
    assert len({row["body_sha256"] for row in actual}) == 2
    assert actual[0]["source_record_hash"] == actual[2]["source_record_hash"]
    offset = 0
    for number, row in enumerate(actual):
        payload = (b"\xef\xbb\xbf" if number == 0 else b"") + encoded[number]
        assert row["byte_offset"] == offset
        assert row["byte_length"] == len(payload)
        assert row["raw_sha256"] == hashlib.sha256(payload).digest()
        assert row["source_record_hash"] == hashlib.sha256(lines[number].encode()).digest()
        assert row["state"] == "pending_verification"
        offset += len(payload)
    assert report["index_complete"] is True
    assert report["training_ready"] is False
    assert report["records"] == 3


def test_chinese_summary_is_not_promoted_to_verified_article(tmp_path):
    raw = b'{"datetime":"2022-06-01 09:00:00","title":NaN,"summary":"Breve"}\n'
    source, catalog, _ = source_fixture(tmp_path, symbols=("000001.SZ",), market="CN", news=raw)
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source, module().corpus_candidates(source, catalog, "CN"), destination, cutoff="2023-12-31"
    )
    row = rows(destination)[0]
    assert row["content_kind"] == "summary"
    assert row["title"] is None and row["url"] is None
    assert row["state"] == "needs_provenance"
    assert row["declared_day"] == "2022-06-01"


def test_reserved_records_are_not_queued_for_verification(tmp_path):
    source, catalog, _ = source_fixture(
        tmp_path, news=(json.dumps(article(Date="2024-01-01")) + "\n").encode()
    )
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source, module().corpus_candidates(source, catalog, "US"), destination, cutoff="2023-12-31"
    )
    assert rows(destination)[0]["state"] == "reserved"


def test_bad_records_do_not_hide_following_records(tmp_path):
    raw = b"not json\n\xff\n" + (json.dumps(article()) + "\n").encode()
    source, catalog, _ = source_fixture(tmp_path, news=raw)
    destination = tmp_path / "news.sqlite"
    result = module().index_news(
        source, module().corpus_candidates(source, catalog, "US"), destination, cutoff="2023-12-31"
    )
    assert [row["state"] for row in rows(destination)] == [
        "invalid_json",
        "invalid_utf8",
        "pending_verification",
    ]
    assert result["records"] == 3


def test_large_record_is_consumed_without_losing_next_offset(tmp_path):
    large = b'{"Article":"' + b"x" * 3000 + b'"}\n'
    small = (json.dumps(article()) + "\n").encode()
    source, catalog, _ = source_fixture(tmp_path, news=large + small)
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source,
        module().corpus_candidates(source, catalog, "US"),
        destination,
        cutoff="2023-12-31",
        max_record_bytes=512,
    )
    first, second = rows(destination)
    assert first["state"] == "record_too_large"
    assert first["raw_sha256"] == hashlib.sha256(large).digest()
    assert first["source_record_hash"] is None
    assert second["state"] == "pending_verification"
    assert second["byte_offset"] == len(large)


def test_failure_rolls_back_only_the_unconfirmed_file(tmp_path):
    source, catalog, paths = source_fixture(tmp_path, symbols=("AAA", "BBB"))
    assets = module().corpus_candidates(source, catalog, "US")
    path = paths["BBB", "news"]
    original = path.read_bytes()
    path.write_bytes(original + b"{}\n")
    destination = tmp_path / "news.sqlite"
    with pytest.raises(ValueError, match="fuente"):
        module().index_news(source, assets, destination, cutoff="2023-12-31")
    state = module().corpus_status(destination)
    assert state["completed_files"] == 1 and state["records"] == 1
    assert state["index_complete"] is False
    path.write_bytes(original)
    resumed = module().index_news(source, assets, destination, cutoff="2023-12-31")
    assert resumed["records"] == 2 and resumed["reused_files"] == 1
    assert resumed["index_complete"] is True


def test_resume_checks_completed_source_content(tmp_path):
    source, catalog, paths = source_fixture(tmp_path)
    assets = module().corpus_candidates(source, catalog, "US")
    destination = tmp_path / "news.sqlite"
    module().index_news(source, assets, destination, cutoff="2023-12-31")
    paths["AAA", "news"].write_text("{}\n")
    with pytest.raises(ValueError, match="fuente"):
        module().index_news(source, assets, destination, cutoff="2023-12-31")
    assert len(rows(destination)) == 1


def test_resume_rejects_changed_cutoff(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    assets = module().corpus_candidates(source, catalog, "US")
    destination = tmp_path / "news.sqlite"
    module().index_news(source, assets, destination, cutoff="2023-12-31")
    with pytest.raises(ValueError, match="configuración"):
        module().index_news(source, assets, destination, cutoff="2022-12-31")
    assert len(rows(destination)) == 1


def test_existing_unrelated_database_is_preserved(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    destination = tmp_path / "news.sqlite"
    with sqlite3.connect(destination) as db:
        db.execute("CREATE TABLE original(value TEXT)")
        db.execute("INSERT INTO original VALUES ('conservar')")
    with pytest.raises(ValueError):
        module().index_news(
            source,
            module().corpus_candidates(source, catalog, "US"),
            destination,
            cutoff="2023-12-31",
        )
    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT value FROM original").fetchone()[0] == "conservar"


def test_index_cannot_write_under_source(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="origen"):
        module().index_news(
            source,
            module().corpus_candidates(source, catalog, "US"),
            source / "new.sqlite",
            cutoff="2023-12-31",
        )
    assert not (source / "new.sqlite").exists()


def test_cli_indexes_both_markets_and_returns_status(tmp_path):
    source, catalog, _ = source_fixture(tmp_path)
    source_fixture(tmp_path, symbols=("000001.SZ",), market="CN")
    destination = tmp_path / "news.sqlite"
    import sys

    command = [
        sys.executable,
        "-m",
        "mars_titan.data.cli",
        "corpus-index",
        "--source",
        str(source),
        "--inventory",
        str(catalog),
        "--database",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["records"] == 2
    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "mars_titan.data.cli",
            "corpus-status",
            "--database",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert set(json.loads(status.stdout)["markets"]) == {"US", "CN"}


def test_duplicate_json_fields_are_not_interpreted_as_unambiguous(tmp_path):
    raw = b'{"Date":"2024-01-01","Date":"2022-01-01","Article":"Texto"}\n'
    source, catalog, _ = source_fixture(tmp_path, news=raw)
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source, module().corpus_candidates(source, catalog, "US"), destination, cutoff="2023-12-31"
    )
    assert rows(destination)[0]["state"] == "invalid_json"


def test_reserved_or_missing_symbol_identity_cannot_be_a_directory(tmp_path):
    source = tmp_path / "dataset"
    folder = source / "time_series" / "S&P500_time_series"
    folder.mkdir(parents=True)
    (folder / "..csv").write_text("Date,Open,Close\n2022-01-01,1,1\n")
    catalog = tmp_path / "inventory.sqlite"
    inventory(source, catalog)
    with pytest.raises(ValueError, match="identidad"):
        module().corpus_candidates(source, catalog, "US")


def test_resume_rejects_a_different_indexer_policy(tmp_path, monkeypatch):
    source, catalog, _ = source_fixture(tmp_path)
    assets = module().corpus_candidates(source, catalog, "US")
    destination = tmp_path / "news.sqlite"
    module().index_news(source, assets, destination, cutoff="2023-12-31")
    substitute = tmp_path / "other_parser.py"
    substitute.write_text("Otro intérprete de los registros")
    monkeypatch.setattr(module(), "__file__", str(substitute))
    with pytest.raises(ValueError, match="configuración"):
        module().index_news(source, assets, destination, cutoff="2023-12-31")
    assert len(rows(destination)) == 1


def test_interruption_does_not_confirm_partial_file(tmp_path, monkeypatch):
    raw = ((json.dumps(article()) + "\n") * 2).encode()
    source, catalog, _ = source_fixture(tmp_path, news=raw)
    assets = module().corpus_candidates(source, catalog, "US")
    destination = tmp_path / "news.sqlite"
    real = module()._read_source

    def interrupted(*args):
        yield next(real(*args))
        raise KeyboardInterrupt

    monkeypatch.setattr(module(), "_read_source", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module().index_news(source, assets, destination, cutoff="2023-12-31")
    assert module().corpus_status(destination)["records"] == 0
    assert module().corpus_status(destination)["completed_files"] == 0
    assert rows(destination) == []
    monkeypatch.setattr(module(), "_read_source", real)
    assert module().index_news(source, assets, destination, cutoff="2023-12-31")["records"] == 2


@pytest.mark.parametrize("value", [0, -1, True, 17 * 1024**2, "100"])
def test_invalid_record_budget_does_not_create_database(tmp_path, value):
    source, catalog, _ = source_fixture(tmp_path)
    destination = tmp_path / "news.sqlite"
    with pytest.raises(ValueError, match="presupuesto"):
        module().index_news(
            source,
            module().corpus_candidates(source, catalog, "US"),
            destination,
            cutoff="2023-12-31",
            max_record_bytes=value,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"Article": ""}, "missing_content"),
        ({"Stock_symbol": "OTHER"}, "symbol_mismatch"),
        ({"Date": "fecha desconocida"}, "needs_provenance"),
        ({"Url": "https://user:password@example.test/story"}, "needs_provenance"),
        ({"Article_title": "x" * 8193}, "metadata_too_large"),
        ({"Article": "\ud800"}, "invalid_unicode"),
    ],
)
def test_incomplete_or_ambiguous_fields_remain_outside_verification_queue(
    tmp_path, changes, expected
):
    raw = (json.dumps(article(**changes)) + "\n").encode()
    source, catalog, _ = source_fixture(tmp_path, news=raw)
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source, module().corpus_candidates(source, catalog, "US"), destination, cutoff="2023-12-31"
    )
    row = rows(destination)[0]
    assert row["state"] == expected
    if "Url" in changes:
        assert row["url"] is None


def test_non_utf8_source_filename_keeps_its_exact_identity(tmp_path):
    source, catalog, paths = source_fixture(tmp_path, symbols=("000001.SZ",), market="CN")
    original = paths["000001.SZ", "news"]
    renamed = original.with_name(os.fsdecode(b"000001.SZ_\xff.jsonl"))
    original.rename(renamed)
    inventory(source, catalog)
    destination = tmp_path / "news.sqlite"
    module().index_news(
        source,
        module().corpus_candidates(source, catalog, "CN"),
        destination,
        cutoff="2023-12-31",
    )
    with sqlite3.connect(destination) as db:
        recorded = db.execute("SELECT path FROM source_files").fetchone()[0]
    assert recorded == os.fsencode(renamed.relative_to(source))
    assert len(rows(destination)) == 1


@pytest.mark.parametrize("failed_modality", ["news", "fundamentals"])
def test_inventory_errors_remain_visible_in_persistent_status(tmp_path, failed_modality):
    source, catalog, _ = source_fixture(tmp_path)
    with sqlite3.connect(catalog) as db:
        for key, payload in db.execute("SELECT path,record FROM files").fetchall():
            entry = json.loads(payload)
            if entry["modality"] == failed_modality:
                entry.update(state="error", sha256=None, error="No se pudo leer la fuente")
                db.execute("UPDATE files SET record=? WHERE path=?", (json.dumps(entry), key))
    destination = tmp_path / "news.sqlite"
    result = module().index_news(
        source,
        module().corpus_candidates(source, catalog, "US"),
        destination,
        cutoff="2023-12-31",
    )
    assert result["errors"] == 1
    assert result["source_errors_by_modality"] == {failed_modality: 1}
    assert result["source_files"] == 1
    assert result["completed_files"] == (0 if failed_modality == "news" else 1)
    assert result["index_complete"] is (failed_modality != "news")
    assert result["training_ready"] is False
    with sqlite3.connect(destination) as db:
        stored = db.execute("SELECT modality,reason FROM source_errors").fetchone()
    assert stored == (failed_modality, "No se pudo leer la fuente")
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mars_titan.data.cli",
            "corpus-status",
            "--database",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["errors"] == 1
