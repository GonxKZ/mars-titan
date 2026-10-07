"""Fixtures técnicos de H.15, sin documentos ni cifras de terceros."""

import hashlib
import importlib
import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

POLICY = "FED_H15_REVIEWED_WEEKLY_ARCHIVE_V1"
AVAILABILITY = "SOURCE_DAY_END_TARGET_NEXT_SESSION_V1"
REVIEW = "PDF_HTML_REVIEWED_CELLS_V1"
URL = "https://www.federalreserve.gov/releases/h15/20000110/h15.htm"
DATES = [f"2000-01-{day:02d}" for day in range(3, 8)]
VALUES = {
    "us_fed_funds": ("Federal funds (effective)", ["0.00", "1.01", "1.02", "1.03", "1.04"]),
    "us_treasury_3m": ("3-month", ["-0.50", "1.11", "1.12", "1.13", "1.14"]),
    "us_treasury_2y": ("2-year", ["2.00", "2.01", "2.02", "2.03", "2.04"]),
    "us_treasury_5y": ("5-year", ["3.00", "3.01", "3.02", "3.03", "3.04"]),
    "us_treasury_10y": ("10-year", ["4.0000000000000001", "4.01", "4.02", "4.03", "4.04"]),
    "us_treasury_30y": ("30-year", ["5.00", "5.01", "5.02", "5.03", "5.04"]),
}


def module():
    return importlib.import_module("mars_titan.data.macro_h15_archive")


def row(label, values):
    return "<tr><th>" + label + "</th>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"


def document():
    dates = [f"2000 Jan {day}" for day in range(3, 8)]
    rows = [
        row("Instruments", dates + ["Week ending Jan 7", "Week ending Dec 31", "1999 Dec"]),
        row(*("Federal funds (effective) 1 2 3", VALUES["us_fed_funds"][1] + ["90", "91", "92"])),
        row("Treasury bills", [""] * 8),
        row("3-month", ["99"] * 8),
        row("Treasury constant maturities 13", [""] * 8),
    ]
    rows += [
        row(label, values + ["90", "91", "92"])
        for key, (label, values) in VALUES.items()
        if key != "us_fed_funds"
    ]
    rows += [row("Composite", [""] * 8), row("10-year", ["88"] * 8)]
    return (
        "<html><p>FEDERAL RESERVE STATISTICAL RELEASE</p><p>H.15 (519)</p>"
        "<p>Release Date: January 10, 2000</p><p>For immediate release January 10, 2000</p>"
        '<p>Yields in percent per annum</p><a href="h15.pdf">PDF</a>'
        '<table summary="Selected Interest Rates">' + "".join(rows) + "</table></html>"
    ).encode()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    html, pdf = root / "release.html", root / "release.pdf"
    html.write_bytes(document())
    pdf.write_bytes(b"%PDF-1.4\nFixture tecnico revisado, no documento real\n%%EOF\n")
    html_receipt, pdf_receipt = root / "html-receipt.json", root / "pdf-receipt.json"
    common = {"status": 200, "complete": True}
    dump(
        html_receipt,
        {
            **common,
            "url": URL,
            "final_url": URL,
            "body_sha256": digest(html),
            "body_bytes": html.stat().st_size,
        },
    )
    pdf_url = URL.replace("h15.htm", "h15.pdf")
    dump(
        pdf_receipt,
        {
            **common,
            "url": pdf_url,
            "final_url": pdf_url,
            "sha256": digest(pdf),
            "bytes": pdf.stat().st_size,
            "valid_pdf": True,
        },
    )
    review = root / "review.json"
    cells = [
        {
            "indicator_id": key,
            "period_start": day,
            "value_exact": value,
            "source_page": 1,
            "source_column": column,
            "source_label": label,
        }
        for key, (label, values) in VALUES.items()
        for column, (day, value) in enumerate(zip(DATES, values, strict=True), 1)
    ]
    dump(
        review,
        {
            "schema_version": 1,
            "policy": REVIEW,
            "status": "document_reviewed_not_admitted",
            "publication_date": "2000-01-10",
            "html_sha256": digest(html),
            "pdf_sha256": digest(pdf),
            "publication_date_page": 1,
            "reviewed_pages": [1, 2],
            "unit": "percent_per_annum",
            "correction_notice": False,
            "observations": cells,
        },
    )
    manifest = root / "manifest.json"
    meta = {
        "schema_version": 1,
        "policy": POLICY,
        "availability_rule": AVAILABILITY,
        "review_policy": REVIEW,
        "documents": [
            {
                "publication_date": "2000-01-10",
                "html": {
                    "path": html.name,
                    "sha256": digest(html),
                    "url": URL,
                    "receipt": {"path": html_receipt.name, "sha256": digest(html_receipt)},
                },
                "pdf": {
                    "path": pdf.name,
                    "sha256": digest(pdf),
                    "url": pdf_url,
                    "receipt": {"path": pdf_receipt.name, "sha256": digest(pdf_receipt)},
                },
                "review": {"path": review.name, "sha256": digest(review)},
            }
        ],
    }
    dump(manifest, meta)
    return {
        "manifest": manifest,
        "html": html,
        "pdf": pdf,
        "review": review,
        "output": tmp_path / "edition",
        "html_receipt": html_receipt,
    }


def prepare(source, **kwargs):
    return module().prepare_h15_archive(source["manifest"], source["output"], **kwargs)


def update_review(source, change):
    review = json.loads(source["review"].read_text())
    change(review)
    dump(source["review"], review)
    manifest = json.loads(source["manifest"].read_text())
    manifest["documents"][0]["review"]["sha256"] = digest(source["review"])
    dump(source["manifest"], manifest)


def test_candidates_select_constant_maturities_and_preserve_exact_decimals():
    result = module().extract_h15_release(document(), URL)
    assert result["status"] == "candidate_extracted"
    assert result["admission_required"] is True
    assert len(result["observations"]) == 30
    selected = {
        r["indicator_id"]: r for r in result["observations"] if r["period_start"] == "2000-01-03"
    }
    assert selected["us_treasury_3m"]["value_exact"] == "-0.50"
    assert selected["us_treasury_10y"]["value_exact"] == "4.0000000000000001"
    assert selected["us_fed_funds"]["value_exact"] == "0.00"
    assert {r["source_column"] for r in result["observations"]} == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("token", ["n.a.", "ND", ""])
def test_explicit_missing_cell_remains_missing(token):
    body = document().replace(b">-0.50<", f">{token}<".encode())
    rows = module().extract_h15_release(body, URL)["observations"]
    item = next(
        r for r in rows if r["indicator_id"] == "us_treasury_3m" and r["source_column"] == 1
    )
    assert item["value_exact"] is None
    assert item["missing_reason"] == "missing_source_value"


@pytest.mark.parametrize(
    "before,after",
    [
        (b"2000 Jan 3", b"2000 Jan 4"),
        (b"2000 Jan 7", b"2000 Jan 31"),
        (b"2000 Jan 3", b"Jan 3"),
        (b"2000 Jan 3", b"1999 Dec 32"),
        (b"percent per annum", b"basis points"),
        (b"January 10, 2000", b"January 11, 2000"),
        (b">-0.50<", b">NaN<"),
        (b">-0.50<", b">Infinity<"),
        (b">-0.50<", b">1e9999<"),
        (b"Treasury constant maturities 13", b"Treasury bills"),
        (b"<td>4.04</td>", b""),
        (b"<td>4.04</td>", b'<td colspan="2">4.04</td>'),
    ],
)
def test_ambiguous_or_corrupt_tables_are_rejected(before, after):
    with pytest.raises(ValueError):
        module().extract_h15_release(document().replace(before, after), URL)


@pytest.mark.parametrize(
    "url",
    [
        URL.replace("https", "http"),
        URL.replace("www.federalreserve.gov", "example.org"),
        URL + "?date=2000",
        URL.replace("20000110", "20000111"),
    ],
)
def test_only_exact_official_issue_urls_are_supported(url):
    with pytest.raises(ValueError):
        module().extract_h15_release(document(), url)


def test_cross_year_and_tuesday_issue_are_valid():
    body = document().replace(b"January 10, 2000", b"January 4, 2000")
    for old, new in zip(
        range(3, 8),
        ["1999 Dec 27", "1999 Dec 28", "1999 Dec 29", "1999 Dec 30", "1999 Dec 31"],
        strict=True,
    ):
        body = body.replace(f"2000 Jan {old}".encode(), new.encode())
    result = module().extract_h15_release(body, URL.replace("20000110", "20000104"))
    assert result["observations"][0]["period_start"] == "1999-12-27"
    assert result["publication_date"] == "2000-01-04"


def test_bytes_budget_is_checked_before_html_parsing():
    with pytest.raises(ValueError, match="presupuesto|MiB"):
        module().extract_h15_release(b" " * (8 * 1024**2 + 1), URL)


def test_cell_budget_is_checked_before_extracting_candidates():
    body = document().replace(b"</table>", b"<tr>" + b"<td>x</td>" * 5001 + b"</tr></table>")
    with pytest.raises(ValueError, match="presupuesto|celdas"):
        module().extract_h15_release(body, URL)


def test_reviewed_edition_preserves_values_and_calendar_availability(source):
    result = prepare(source)
    assert result["observations"] == 30
    assert result["rows"] == 60
    assert result["admission_required"] is True and result["training_ready"] is False
    rows = pq.read_table(source["output"] / "observations.parquet").to_pylist()
    first = next(
        r
        for r in rows
        if r["market"] == "US"
        and r["indicator_id"] == "us_treasury_10y"
        and r["period_start"] == "2000-01-03"
    )
    assert first["value_exact"] == "4.0000000000000001"
    assert first["available_at"] == datetime(2000, 1, 11, 21, 5, tzinfo=UTC)
    assert first["publication_date"] == "2000-01-10"
    assert first["review_status"] == "document_reviewed_not_admitted"
    china = next(r for r in rows if r["market"] == "CN")
    assert china["available_at"] == datetime(2000, 1, 12, 7, 5, tzinfo=UTC)
    assert all(r["source_page"] == 1 for r in rows)


def test_recovery_verifies_and_does_not_rewrite_confirmed_files(source):
    prepare(source)
    old = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source["output"].iterdir()}
    result = prepare(source)
    assert result["reused"] is True
    assert old == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source["output"].iterdir()
    }


@pytest.mark.parametrize("file", ["html", "pdf", "review", "html_receipt"])
def test_changed_sources_reject_recovery(source, file):
    prepare(source)
    source[file].write_bytes(source[file].read_bytes() + b" ")
    with pytest.raises(ValueError):
        prepare(source)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="candidate_extracted"),
        lambda r: r.update(publication_date="2000-01-03"),
        lambda r: r.update(correction_notice=True),
        lambda r: r.update(schema_version=True),
        lambda r: r.update(pdf_sha256=None),
        lambda r: r["observations"][0].update(value_exact="7.0"),
        lambda r: r["observations"][0].update(source_page=3),
        lambda r: r["observations"][0].update(source_column=True),
        lambda r: r["observations"].pop(),
    ],
)
def test_review_must_match_source_cells_and_declared_pages(source, change):
    update_review(source, change)
    with pytest.raises(ValueError):
        prepare(source)
    assert not source["output"].exists()


def test_publication_after_cutoff_is_rejected_before_source_payload(source, monkeypatch):
    m = json.loads(source["manifest"].read_text())
    m["documents"][0]["publication_date"] = "2024-01-01"
    dump(source["manifest"], m)
    with pytest.raises(ValueError, match="corte|reserva"):
        prepare(source)


@pytest.mark.parametrize("field", ["observations", "rows"])
def test_corrupt_report_counts_are_not_accepted_on_recovery(source, field):
    prepare(source)
    p = source["output"] / "report.json"
    report = json.loads(p.read_text())
    report[field] += 1
    dump(p, report)
    with pytest.raises(ValueError):
        prepare(source)


def test_output_corruption_is_not_silently_rebuilt(source):
    prepare(source)
    p = source["output"] / "observations.parquet"
    p.write_bytes(b"corrupto")
    with pytest.raises(ValueError):
        prepare(source)
    assert p.read_bytes() == b"corrupto"


def test_interrupted_publication_leaves_no_confirmed_edition(source, monkeypatch):
    original = module().atomic_parquet

    def fail(path, table):
        original(path, table)
        raise OSError("interrupción técnica")

    monkeypatch.setattr(module(), "atomic_parquet", fail)
    with pytest.raises(OSError, match="interrupción"):
        prepare(source)
    assert not source["output"].exists()
    monkeypatch.setattr(module(), "atomic_parquet", original)
    assert prepare(source)["status"] == "completed"


def test_new_section_cannot_be_read_as_constant_maturities():
    body = document().replace(
        row("Treasury constant maturities 13", [""] * 8).encode(),
        (
            row("Treasury constant maturities 13", [""] * 8) + row("Treasury bills", [""] * 8)
        ).encode(),
    )
    with pytest.raises(ValueError):
        module().extract_h15_release(body, URL)


def test_nonnumeric_review_marker_is_not_a_numeric_pdf_match(source):
    update_review(source, lambda review: review["observations"][0].update(value_exact="n.a."))
    with pytest.raises(ValueError):
        prepare(source)


def test_symlink_substitution_during_publication_is_rejected(source, monkeypatch):
    real_write = module().atomic_parquet

    def replace_source(path, table):
        real_write(path, table)
        outside = source["output"].parent / "other.html"
        outside.write_bytes(source["html"].read_bytes())
        source["html"].unlink()
        source["html"].symlink_to(outside)

    monkeypatch.setattr(module(), "atomic_parquet", replace_source)
    with pytest.raises(ValueError):
        prepare(source)
    assert not source["output"].exists()


def test_file_budget_is_checked_before_reading_pdf(source, monkeypatch):
    with source["pdf"].open("wb") as stream:
        stream.truncate(8 * 1024**2 + 1)
    from pathlib import Path

    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == source["pdf"]:
            raise AssertionError("El PDF sobredimensionado no debe abrirse")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="presupuesto"):
        prepare(source)


def test_source_change_before_publication_prevents_confirmation(source, monkeypatch):
    real_write = module().atomic_parquet

    def change_source(path, table):
        real_write(path, table)
        source["html"].write_bytes(source["html"].read_bytes() + b" ")

    monkeypatch.setattr(module(), "atomic_parquet", change_source)
    with pytest.raises(ValueError, match="fuente|recibo"):
        prepare(source)
    assert not source["output"].exists()


def test_code_change_before_publication_prevents_confirmation(source, monkeypatch):
    original = module()._code_identity
    called = 0

    def changed():
        nonlocal called
        called += 1
        result = original()
        if called > 1:
            result["macro_h15_archive.py"] = "0" * 64
        return result

    monkeypatch.setattr(module(), "_code_identity", changed)
    with pytest.raises(ValueError, match="código"):
        prepare(source)
    assert not source["output"].exists()


def test_recovery_rejects_changed_boolean_type(source):
    prepare(source)
    p = source["output"] / "report.json"
    report = json.loads(p.read_text())
    report["training_ready"] = 0
    dump(p, report)
    with pytest.raises(ValueError):
        prepare(source)


def test_final_day_is_retained_without_a_future_available_decision(source):
    report = prepare(source, cutoff="2000-01-10")
    rows = pq.read_table(source["output"] / "observations.parquet").to_pylist()
    assert report["unavailable_before_cutoff_rows"] == 60
    assert all(row["available_at"] is None for row in rows)
    assert all(row["publication_date"] == "2000-01-10" for row in rows)


def test_duplicate_editions_never_overwrite_observations(source):
    manifest = json.loads(source["manifest"].read_text())
    manifest["documents"].append(manifest["documents"][0])
    dump(source["manifest"], manifest)
    with pytest.raises(ValueError, match="repite|duplic"):
        prepare(source)


def test_relative_path_cannot_escape_manifest_root(source):
    manifest = json.loads(source["manifest"].read_text())
    manifest["documents"][0]["html"]["path"] = "../source/release.html"
    dump(source["manifest"], manifest)
    with pytest.raises(ValueError, match="directorio"):
        prepare(source)


@pytest.mark.parametrize(
    "field,value", [("admission_granted", True), ("reused", 0), ("output_bytes", 1)]
)
def test_recovery_rejects_altered_receipt_contract(source, field, value):
    prepare(source)
    path = source["output"] / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    dump(path, report)
    with pytest.raises(ValueError):
        prepare(source)


def test_rehashed_parquet_cannot_change_availability(source):
    prepare(source)
    path = source["output"] / "observations.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["available_at"] = datetime(2000, 1, 3, 21, 5, tzinfo=UTC)
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    report_path = source["output"] / "report.json"
    report = json.loads(report_path.read_text())
    report["artifacts"]["observations.parquet"] = digest(path)
    report["output_bytes"] = path.stat().st_size
    dump(report_path, report)
    with pytest.raises(ValueError, match="observaciones"):
        prepare(source)


def test_recovery_confirms_sources_again_after_reading_report(source, monkeypatch):
    prepare(source)
    original = module().read_manifest

    def change_after_read(path, maximum):
        result = original(path, maximum)
        if path == source["output"] / "report.json":
            source["html"].write_bytes(source["html"].read_bytes() + b" ")
        return result

    monkeypatch.setattr(module(), "read_manifest", change_after_read)
    with pytest.raises(ValueError, match="fuente|recibo"):
        prepare(source)


def test_columns_can_cross_month_boundary():
    body = document().replace(b"January 10, 2000", b"February 7, 2000")
    replacements = ["2000 Jan 28", "2000 Jan 31", "2000 Feb 1", "2000 Feb 2", "2000 Feb 3"]
    for day, target in zip(range(3, 8), replacements, strict=True):
        body = body.replace(f">2000 Jan {day}<".encode(), f">{target}<".encode())
    rows = module().extract_h15_release(body, URL.replace("20000110", "20000207"))["observations"]
    assert [r["period_start"] for r in rows[:5]] == [
        "2000-01-28",
        "2000-01-31",
        "2000-02-01",
        "2000-02-02",
        "2000-02-03",
    ]


def test_tuesday_january_18_is_not_forced_to_a_monday():
    body = document().replace(b"January 10, 2000", b"January 18, 2000")
    for day in range(3, 8):
        body = body.replace(f">2000 Jan {day}<".encode(), f">2000 Jan {day + 7}<".encode())
    result = module().extract_h15_release(body, URL.replace("20000110", "20000118"))
    assert result["publication_date"] == "2000-01-18"
    assert {r["period_start"] for r in result["observations"]} == {
        f"2000-01-{day}" for day in range(10, 15)
    }


def combined_source(source, *, same_pdf=False):
    root = source["manifest"].parent
    second = root / "second"
    second.mkdir()
    old = json.loads(source["manifest"].read_text())
    item = json.loads(json.dumps(old["documents"][0]))
    item["publication_date"] = "2000-01-18"
    for kind in ["html", "pdf"]:
        ref = item[kind]
        original = root / ref["path"]
        target = second / original.name
        content = original.read_bytes()
        if kind == "html":
            content = content.replace(b"January 10, 2000", b"January 18, 2000").replace(
                b"4.0000000000000001", b"4.50"
            )
        elif not same_pdf:
            content = content.replace(b"Fixture tecnico", b"Fixture tecnico de otra edicion")
        target.write_bytes(content)
        ref["path"] = str(target.relative_to(root))
        ref["sha256"] = digest(target)
        ref["url"] = ref["url"].replace("20000110", "20000118")
        receipt = json.loads((root / ref["receipt"]["path"]).read_text())
        receipt.update(url=ref["url"], final_url=ref["url"])
        receipt["body_sha256" if kind == "html" else "sha256"] = digest(target)
        receipt["body_bytes" if kind == "html" else "bytes"] = target.stat().st_size
        receipt_path = second / (kind + "-receipt.json")
        dump(receipt_path, receipt)
        ref["receipt"] = {
            "path": str(receipt_path.relative_to(root)),
            "sha256": digest(receipt_path),
        }
    review = json.loads(source["review"].read_text())
    review.update(
        publication_date="2000-01-18",
        html_sha256=item["html"]["sha256"],
        pdf_sha256=item["pdf"]["sha256"],
    )
    for cell in review["observations"]:
        if cell["indicator_id"] == "us_treasury_10y" and cell["period_start"] == "2000-01-03":
            cell["value_exact"] = "4.50"
    review_path = second / "review.json"
    dump(review_path, review)
    item["review"] = {"path": str(review_path.relative_to(root)), "sha256": digest(review_path)}
    old["documents"].append(item)
    combined = root / "combined.json"
    dump(combined, old)
    return combined


def test_later_edition_keeps_the_previous_published_values(source):
    prepare(source)
    old_table = pq.read_table(source["output"] / "observations.parquet")
    combined = combined_source(source)
    output = source["output"].parent / "history"
    result = module().prepare_h15_archive(combined, output)
    rows = pq.read_table(output / "observations.parquet").to_pylist()
    assert result["observations"] == 60 and result["rows"] == 120
    assert [r for r in rows if r["publication_date"] == "2000-01-10"] == old_table.to_pylist()
    assert pq.read_table(source["output"] / "observations.parquet").equals(old_table)
    assert {
        r["value_exact"]
        for r in rows
        if r["indicator_id"] == "us_treasury_10y" and r["period_start"] == "2000-01-03"
    } == {"4.0000000000000001", "4.50"}


def test_one_pdf_cannot_accredit_different_publication_dates(source):
    combined = combined_source(source, same_pdf=True)
    with pytest.raises(ValueError, match="PDF|publicación|edición"):
        module().prepare_h15_archive(combined, source["output"])


def rehash_parquet(source):
    path = source["output"] / "observations.parquet"
    report_path = source["output"] / "report.json"
    report = json.loads(report_path.read_text())
    report["artifacts"]["observations.parquet"] = digest(path)
    report["output_bytes"] = path.stat().st_size
    dump(report_path, report)


def test_recovery_bounds_dictionary_expansion_before_dense_read(source, monkeypatch):
    prepare(source)
    path = source["output"] / "observations.parquet"
    table = pq.read_table(path)
    column = pa.DictionaryArray.from_arrays(
        pa.array([0] * table.num_rows, type=pa.int32()), pa.array(["x" * (2 * 1024**2)])
    )
    changed = table.set_column(table.schema.get_field_index("source_label"), "source_label", column)
    pq.write_table(changed, path, compression="zstd", store_schema=False, write_statistics=False)
    rehash_parquet(source)
    original = module().pq.ParquetFile
    with original(path) as file:
        assert file.schema_arrow == table.schema
        encoded = sum(
            file.metadata.row_group(i).column(j).total_uncompressed_size
            for i in range(file.num_row_groups)
            for j in range(file.metadata.num_columns)
        )
        assert encoded < module()._MAX_PARQUET_BYTES
    assert table.num_rows * len(column.dictionary[0].as_py()) > module()._MAX_PARQUET_BYTES
    old = {p.name: (digest(p), p.stat().st_mtime_ns) for p in source["output"].iterdir()}

    class BoundedRead:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def read(self, *args, **kwargs):
            raise AssertionError("No debe expandirse el archivo entero antes de acotarlo")

        def iter_batches(self, *args, **kwargs):
            for batch in self.file.iter_batches(*args, **kwargs):
                assert batch.nbytes < module()._MAX_PARQUET_BYTES
                if "source_label" in batch.schema.names:
                    assert pa.types.is_dictionary(batch["source_label"].type)
                yield batch

    monkeypatch.setattr(module().pq, "ParquetFile", BoundedRead)
    with pytest.raises(ValueError, match="presupuesto"):
        prepare(source)
    assert old == {p.name: (digest(p), p.stat().st_mtime_ns) for p in source["output"].iterdir()}


@pytest.mark.parametrize("dictionary", [False, True])
@pytest.mark.parametrize("value", ["otra etiqueta", None])
def test_recovery_compares_text_and_nulls_exactly(source, dictionary, value):
    prepare(source)
    path = source["output"] / "observations.parquet"
    table = pq.read_table(path)
    labels = table["source_label"].to_pylist()
    labels[-1] = value
    changed = table.set_column(
        table.schema.get_field_index("source_label"), "source_label", pa.array(labels)
    )
    pq.write_table(changed, path, use_dictionary=dictionary, row_group_size=13)
    rehash_parquet(source)
    with pytest.raises(ValueError, match="observaciones"):
        prepare(source)


@pytest.mark.parametrize("dictionary", [False, True])
def test_recovery_accepts_equivalent_parquet_encodings(source, dictionary):
    prepare(source)
    path = source["output"] / "observations.parquet"
    table = pq.read_table(path)
    pq.write_table(table, path, use_dictionary=dictionary, row_group_size=13)
    rehash_parquet(source)
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert prepare(source)["reused"] is True
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
