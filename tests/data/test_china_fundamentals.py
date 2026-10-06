"""Importación contable desde revisiones explícitas y fuentes identificadas."""

import importlib
import json
from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock


def materialize(*args, **kwargs):
    try:
        module = importlib.import_module("mars_titan.data.china_fundamentals")
    except ModuleNotFoundError:
        pytest.fail("Falta la materialización de hechos chinos reconciliados")
    return module.materialize_chinese_facts(*args, **kwargs)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    raw = source / "balance.jsonl"
    raw.write_text(
        json.dumps(
            [
                {"ts_code": "000001.SZ", "end_date": "20221231", "total_assets": 100_000_000},
                {"ts_code": "000001.SZ", "end_date": "20221231", "total_assets": 100_000_000},
                {"ts_code": "000001.SZ", "end_date": "20211231", "total_assets": 90_000_000},
            ]
        )
    )
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4\nDocumento sintetico de prueba\n")
    publication = tmp_path / "announcement.json"
    publication.write_text(
        json.dumps(
            {
                "announcements": [
                    {
                        "secCode": "000001",
                        "announcementId": "prueba",
                        "announcementTime": 1678291200000,
                        "adjunctUrl": "finalpage/2023-03-09/prueba.PDF",
                        "adjunctType": "PDF",
                        "pageColumn": "SZZB",
                    }
                ]
            }
        )
    )
    evidence = {
        "symbol": "000001.SZ",
        "field": "total_assets",
        "value": "100",
        "currency": "CNY",
        "unit_multiplier": "1000000",
        "quantity_kind": "stock",
        "period_start": None,
        "period_end": "2022-12-31",
        "statement_scope": "consolidated",
        "accounting_standard": "CAS",
        "publication_kind": "actual",
        "publication_date": "2023-03-09",
        "published_at": None,
        "report_id": "prueba",
        "source_page": 1,
        "source_url": "https://static.cninfo.com.cn/finalpage/2023-03-09/prueba.PDF",
        "document_sha256": sha256(document),
        "publication_evidence_sha256": sha256(publication),
    }
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "symbol": "000001.SZ",
                "source_file": "balance.jsonl",
                "source_file_sha256": sha256(raw),
                "results": [
                    {
                        "evidence": evidence,
                        "original_matches": [
                            {"array_record_1_based": 1},
                            {"array_record_1_based": 2},
                        ],
                    },
                    {
                        "evidence": {**evidence, "value": "90", "period_end": "2021-12-31"},
                        "original_matches": [{"array_record_1_based": 3}],
                    },
                ],
            }
        )
    )
    return dict(
        source=source,
        review=review,
        document=document,
        publication=publication,
        output=tmp_path / "prepared",
        clock=MarketClock("CN", "2023-03-01", "2024-01-31"),
    )


def change_review(inputs, change):
    path = inputs["review"]
    content = json.loads(path.read_text())
    change(content)
    path.write_text(json.dumps(content))


def test_unique_facts_keep_exact_values_and_actual_publication(inputs):
    report = materialize(**inputs)
    rows = pq.read_table(inputs["output"] / "fundamentals.parquet").to_pylist()
    assert report["facts"] == 2
    assert report["matched_records"] == 3
    assert report["unique_source_records"] == 3
    assert report["duplicate_matches"] == 1
    assert report["training_ready"] is False
    assert [r["value"] for r in rows] == [90_000_000, 100_000_000]
    assert [r["value_exact"] for r in rows] == ["90000000", "100000000"]
    assert [r["source_records"] for r in rows] == [[3], [1, 2]]
    for row in rows:
        assert row["concept"] == "cn-reported:Assets:CNY"
        assert row["unit"] == "CNY"
        assert row["accounting_standard"] == "CAS"
        assert row["statement_scope"] == "consolidated"
        assert row["filed"] == "2023-03-09"
        assert row["published_at"] is None
        assert row["available_at"] == datetime(2023, 3, 10, 7, 5, tzinfo=UTC)


def test_repeated_execution_reuses_only_verified_artifacts(inputs):
    first = materialize(**inputs)
    path = inputs["output"] / "fundamentals.parquet"
    before = path.stat()
    second = materialize(**inputs)
    assert first["reused"] is False
    assert second["reused"] is True
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert second["facts"] == 2
    path.write_bytes(b"PAR1corrupto")
    with pytest.raises(ValueError):
        materialize(**inputs)


@pytest.mark.parametrize("name", ["document", "publication"])
def test_changed_evidence_is_rejected_before_publication(inputs, name):
    inputs[name].write_bytes(b"contenido cambiado")
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_declared_match_does_not_replace_reading_the_original_value(inputs):
    raw = inputs["source"] / "balance.jsonl"
    rows = json.loads(raw.read_text())
    rows[0]["total_assets"] += 1
    raw.write_text(json.dumps(rows))
    change_review(inputs, lambda r: r.update(source_file_sha256=sha256(raw)))
    with pytest.raises(ValueError, match="value_mismatch"):
        materialize(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("secCode", "601318"),
        ("announcementId", "otro"),
        ("adjunctUrl", "finalpage/2023-03-10/prueba.PDF"),
        ("announcementTime", 1678377600000),
        ("adjunctType", "HTML"),
        ("pageColumn", "SHZB"),
    ],
)
def test_publication_must_match_issuer_document_and_day(inputs, field, value):
    notice = json.loads(inputs["publication"].read_text())
    notice["announcements"][0][field] = value
    inputs["publication"].write_text(json.dumps(notice))
    digest = sha256(inputs["publication"])
    change_review(
        inputs,
        lambda r: [
            item["evidence"].update(publication_evidence_sha256=digest) for item in r["results"]
        ],
    )
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("ordinal", [0, True, -1, 4, 100_001])
def test_missing_or_invalid_source_record_is_rejected(inputs, ordinal):
    change_review(
        inputs,
        lambda r: r["results"][0]["original_matches"][0].update(array_record_1_based=ordinal),
    )
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_clock_from_another_market_is_rejected(inputs):
    inputs["clock"] = MarketClock("US", "2023-03-01", "2024-01-31")
    with pytest.raises(ValueError):
        materialize(**inputs)


def test_cutoff_does_not_admit_a_later_availability(inputs):
    with pytest.raises(ValueError):
        materialize(**inputs, cutoff="2023-03-09")
    assert not inputs["output"].exists()


def test_final_reserve_cannot_be_opened_by_configuration(inputs):
    with pytest.raises(ValueError):
        materialize(**inputs, cutoff="2024-01-01")


def test_output_cannot_replace_a_source(inputs):
    inputs["output"] = inputs["source"] / "prepared"
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_forged_summary_is_not_reused(inputs):
    materialize(**inputs)
    path = inputs["output"] / "report.json"
    report = json.loads(path.read_text())
    report["facts"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        materialize(**inputs)


def test_interrupted_write_can_be_repeated_without_partial_final_output(inputs, monkeypatch):
    try:
        module = importlib.import_module("mars_titan.data.china_fundamentals")
    except ModuleNotFoundError:
        pytest.fail("Falta la materialización de hechos chinos reconciliados")
    original = module.atomic_parquet

    def fail(path, table):
        original(path, table)
        raise OSError("Corte reproducido antes de publicar")

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_parquet", fail)
        with pytest.raises(OSError):
            materialize(**inputs)
    assert not inputs["output"].exists()
    assert materialize(**inputs)["facts"] == 2


def test_equivalent_decimal_formats_do_not_create_a_false_conflict(inputs):
    def split_matches(review):
        first = review["results"][0]
        first["original_matches"] = [{"array_record_1_based": 1}]
        review["results"].append(
            {
                "evidence": {**first["evidence"], "value": "100.0"},
                "original_matches": [{"array_record_1_based": 2}],
            }
        )

    change_review(inputs, split_matches)
    report = materialize(**inputs)
    assert report["facts"] == 2
    assert report["duplicate_matches"] == 1


def test_recovery_checks_population_before_decoding_parquet(inputs, monkeypatch):
    materialize(**inputs)
    path = inputs["output"] / "fundamentals.parquet"
    table = pq.read_table(path)
    import pyarrow as pa

    pq.write_table(pa.concat_tables([table, table.slice(0, 1)]), path)
    report_path = inputs["output"] / "report.json"
    report = json.loads(report_path.read_text())
    report["sha256"] = sha256(path)
    report_path.write_text(json.dumps(report))

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Se intentó decodificar una población ya incompatible")

    monkeypatch.setattr(pq.ParquetFile, "read", forbidden_read)
    with pytest.raises(ValueError):
        materialize(**inputs)


def test_code_change_during_write_does_not_publish_a_stale_identity(inputs, monkeypatch):
    module = importlib.import_module("mars_titan.data.china_fundamentals")
    original_write = module.atomic_parquet
    real_hash = module.sha256
    written = False

    def write(path, table):
        nonlocal written
        original_write(path, table)
        written = True

    def changed_hash(path):
        if str(path) == module.__file__ and written:
            return "f" * 64
        return real_hash(path)

    monkeypatch.setattr(module, "atomic_parquet", write)
    monkeypatch.setattr(module, "sha256", changed_hash)
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_duplicate_original_json_keys_are_not_silently_overwritten(inputs):
    raw = inputs["source"] / "balance.jsonl"
    content = raw.read_text().replace(
        '"total_assets": 100000000', '"total_assets": 1, "total_assets": 100000000', 1
    )
    raw.write_text(content)
    change_review(inputs, lambda r: r.update(source_file_sha256=sha256(raw)))
    with pytest.raises(ValueError):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_sources_changed_while_reusing_are_rejected(inputs, monkeypatch):
    materialize(**inputs)
    close = pq.ParquetFile.__exit__

    def change_after_read(file, *args):
        result = close(file, *args)
        with (inputs["source"] / "balance.jsonl").open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(pq.ParquetFile, "__exit__", change_after_read)
    with pytest.raises(ValueError):
        materialize(**inputs)


def test_conflicting_reconciled_values_are_not_arbitrarily_selected(inputs):
    raw = inputs["source"] / "balance.jsonl"
    content = json.loads(raw.read_text())
    content[1]["total_assets"] = 101_000_000
    raw.write_text(json.dumps(content))

    def split(review):
        review["source_file_sha256"] = sha256(raw)
        first = review["results"][0]
        first["original_matches"] = [{"array_record_1_based": 1}]
        review["results"].append(
            {
                "evidence": {**first["evidence"], "value": "101"},
                "original_matches": [{"array_record_1_based": 2}],
            }
        )

    change_review(inputs, split)
    with pytest.raises(ValueError, match="contradictorios"):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_cli_materializes_the_same_source_contract(inputs, capsys):
    module = importlib.import_module("mars_titan.data.china_fundamentals")
    args = []
    for name in ("source", "review", "document", "publication", "output"):
        args += ["--" + name, str(inputs[name])]
    module.main(args)
    report = json.loads((inputs["output"] / "report.json").read_text())
    assert report["facts"] == 2
    assert report["training_ready"] is False
    assert "Admisión multimodal pendiente" in capsys.readouterr().out


def test_original_changed_without_a_new_review_is_rejected(inputs):
    with (inputs["source"] / "balance.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="huella del original"):
        materialize(**inputs)
    assert not inputs["output"].exists()


def test_forged_parquet_content_is_not_accepted_by_its_updated_hash(inputs):
    import pyarrow as pa

    materialize(**inputs)
    path = inputs["output"] / "fundamentals.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["value"] += 1
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report_path = inputs["output"] / "report.json"
    report = json.loads(report_path.read_text())
    report["sha256"] = sha256(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="contenido"):
        materialize(**inputs)


def test_oversized_document_is_rejected_before_loading_it(inputs):
    with inputs["document"].open("wb") as stream:
        stream.truncate(65 * 1024**2)
    with pytest.raises(ValueError, match="presupuesto"):
        materialize(**inputs)
    assert not inputs["output"].exists()
