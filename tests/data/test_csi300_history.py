"""Ampliación del factor con tablas históricas revisadas y fuentes inmutables."""

import csv
import importlib
import json
from datetime import UTC, datetime
from urllib.parse import urlencode

import pyarrow.parquet as pq
import pytest

from mars_titan.data.csi300_factor import materialize_csi300_factor
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock

COLUMNS = [
    "session",
    "open",
    "high",
    "low",
    "close",
    "source_pdf_sha256",
    "pdf_page_1_based",
    "printed_page",
]


def api():
    try:
        return importlib.import_module("mars_titan.data.csi300_history")
    except ModuleNotFoundError:
        pytest.fail("Falta la ampliación revisada del historial CSI 300")


def base_factor(root, clock):
    source = root / "acquisition"
    source.mkdir()
    days = MarketClock("CN", "2022-01-01", "2022-01-31").days
    body = source / "202201.jsonp"
    quotes = [
        dict(
            MDATE=d.strftime("%Y%m%d"), OPEN="4000.00", HIGH="4010.00", LOW="3990.00", CLS="4001.25"
        )
        for d in days
    ]
    body.write_text("monthlyData(" + json.dumps({"result": quotes}) + ")")
    url = "https://query.sse.com.cn/commonQuery.do?" + urlencode(
        dict(
            sqlId="COMMON_SSE_ZQZS_M_CSI300_INDEX_C",
            isPagination="false",
            MDATE="202201",
            jsonCallBack="monthlyData",
        )
    )
    atomic_json(
        source / "202201.json",
        dict(
            status=200,
            url=url,
            final_url=url,
            requested_at_utc="2026-10-07T00:00:00Z",
            bytes=body.stat().st_size,
            sha256=sha256(body),
        ),
    )
    acquisition = source / "acquisition.json"
    atomic_json(
        acquisition,
        dict(
            schema_version=1,
            status="completed",
            source_url="https://www.sse.com.cn/aboutus/publication/monthly/index/",
            query_id="COMMON_SSE_ZQZS_M_CSI300_INDEX_C",
            training_ready=False,
            final_test_opened=False,
            acquired_at_utc="2026-10-07T01:00:00Z",
            months=[
                dict(
                    month="202201",
                    rows=len(quotes),
                    sha256=sha256(body),
                    receipt_sha256=sha256(source / "202201.json"),
                )
            ],
            rows=len(quotes),
            first="20220104",
            last="20220128",
        ),
    )
    base = root / "base"
    materialize_csi300_factor(acquisition, base, clock=clock)
    return base


def reviewed_month(root, month, *, missing=None):
    source = root / "reviewed"
    source.mkdir(exist_ok=True)
    pdf = source / f"{month}.pdf"
    pdf.write_bytes(b"%PDF-1.4\n% Documento de prueba revisado externamente.\n%%EOF\n")
    signature = sha256(pdf)
    days = MarketClock("CN", "2021-12-01", "2022-01-31").days
    rows = [
        dict(
            zip(
                COLUMNS,
                [
                    d.isoformat(),
                    "4000.00",
                    "4010.00",
                    "3990.00",
                    "4001.25",
                    signature,
                    "19",
                    "13",
                ],
                strict=True,
            )
        )
        for d in days
        if d.strftime("%Y%m") == month and d.isoformat() != missing
    ]
    path = source / f"{month}.csv"
    write_csv(path, rows)
    return dict(
        month=month,
        source_url="https://www.sse.com.cn/history/example.pdf",
        pdf_path=pdf.name,
        pdf_sha256=signature,
        csv_path=path.name,
        csv_sha256=sha256(path),
        pdf_page_1_based=19,
        printed_page=13,
        title="沪深 300 指数",
    ), rows


def write_csv(path, rows, columns=COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    clock = MarketClock("CN", "2021-12-01", "2022-01-31")
    base = base_factor(tmp_path, clock)
    entry, rows = reviewed_month(tmp_path, "202112")
    manifest = tmp_path / "reviewed/manifest.json"
    atomic_json(
        manifest,
        dict(
            schema_version=1,
            kind="reviewed_csi300_history",
            market="CN",
            symbol="000300",
            sources=[entry],
        ),
    )
    return base, manifest, tmp_path / "extended", clock, rows


def run(inputs):
    base, manifest, output, clock, _ = inputs
    return api().extend_csi300_history(base, manifest, output, clock=clock)


def refresh(inputs, rows):
    manifest = inputs[1]
    meta = json.loads(manifest.read_text())
    entry = meta["sources"][0]
    csv_path = manifest.parent / entry["csv_path"]
    write_csv(csv_path, rows)
    entry["csv_sha256"] = sha256(csv_path)
    atomic_json(manifest, meta)


def test_extension_preserves_base_rows_and_publishes_a_reusable_factor(inputs):
    base, manifest, output, _, _ = inputs
    before = {p.name: sha256(p) for p in base.iterdir()}
    base_table = pq.read_table(base / "prices.parquet")
    report = run(inputs)
    table = pq.read_table(output / "prices.parquet")
    assert table.num_rows == 42
    assert table.slice(23).equals(base_table)
    assert table["session"][0].as_py() == "2021-12-01"
    assert table["session"][-1].as_py() == "2022-01-28"
    assert table["available_at"][0].as_py() == datetime(2021, 12, 1, 7, 5, tzinfo=UTC)
    assert table["source_sha256"][0].as_py() == sha256(manifest.parent / "202112.pdf")
    assert report["base_rows"] == 19 and report["added_rows"] == 23
    assert report["identical_overlap_rows"] == 0 and report["missing_sessions"] == []
    assert report["point_in_time_verified"] is False
    assert report["final_test_opened"] is False
    assert before == {p.name: sha256(p) for p in base.iterdir()}
    times = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
    assert run(inputs) == {**report, "reused": True}
    assert times == {p.name: p.stat().st_mtime_ns for p in output.iterdir()}


def test_identical_overlap_keeps_base_provenance_and_reports_its_count(inputs):
    base, manifest, output, _, _ = inputs
    entry, _ = reviewed_month(manifest.parent.parent, "202201")
    meta = json.loads(manifest.read_text())
    meta["sources"].append(entry)
    atomic_json(manifest, meta)
    report = run(inputs)
    assert report["rows"] == 42 and report["identical_overlap_rows"] == 19
    assert (
        pq.read_table(output / "prices.parquet")
        .slice(23)
        .equals(pq.read_table(base / "prices.parquet"))
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", "4000.001"),
        ("close", "NaN"),
        ("high", "3999.00"),
        ("session", "2021-12-04"),
        ("session", "2024-12-02"),
        ("source_pdf_sha256", "0" * 64),
        ("pdf_page_1_based", "18"),
        ("printed_page", "12"),
    ],
)
def test_invalid_row_or_false_provenance_blocks_publication(inputs, field, value):
    rows = inputs[4]
    rows[0][field] = value
    refresh(inputs, rows)
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs[2].exists()


def test_duplicate_reviewed_dates_are_rejected_even_when_identical(inputs):
    rows = inputs[4]
    refresh(inputs, rows + [rows[0]])
    with pytest.raises(ValueError, match="repetid|duplicad"):
        run(inputs)


def test_missing_sessions_remain_explicit_without_filling_prices(inputs):
    rows = inputs[4]
    refresh(inputs, rows[1:])
    report = run(inputs)
    assert report["missing_sessions"] == ["2021-12-01"]
    assert report["rows"] == 41


@pytest.mark.parametrize(
    "field,value",
    [
        ("month", "202401"),
        ("month", "202113"),
        ("title", "SSE Composite Index"),
        ("source_url", "https://unrelated.example/history.pdf"),
        ("pdf_sha256", "0" * 64),
        ("csv_sha256", None),
        ("pdf_page_1_based", True),
        ("printed_page", -1),
        ("pdf_path", "../unreviewed.pdf"),
        ("csv_path", "/tmp/unreviewed.csv"),
    ],
)
def test_invalid_manifest_source_is_rejected(inputs, field, value):
    manifest = inputs[1]
    meta = json.loads(manifest.read_text())
    meta["sources"][0][field] = value
    atomic_json(manifest, meta)
    with pytest.raises(ValueError):
        run(inputs)
    assert not inputs[2].exists()


def test_conflicting_overlap_does_not_replace_the_base(inputs):
    base, manifest, output, _, _ = inputs
    entry, rows = reviewed_month(manifest.parent.parent, "202201")
    rows[0]["close"] = "4002.25"
    path = manifest.parent / entry["csv_path"]
    write_csv(path, rows)
    entry["csv_sha256"] = sha256(path)
    meta = json.loads(manifest.read_text())
    meta["sources"].append(entry)
    atomic_json(manifest, meta)
    before = sha256(base / "prices.parquet")
    with pytest.raises(ValueError, match="solap|difer|contradic"):
        run(inputs)
    assert not output.exists() and sha256(base / "prices.parquet") == before


@pytest.mark.parametrize("field", ["pdf_path", "csv_path"])
def test_symlinked_reviewed_source_is_rejected(inputs, field):
    manifest = inputs[1]
    entry = json.loads(manifest.read_text())["sources"][0]
    path = manifest.parent / entry[field]
    other = path.with_name("other" + path.suffix)
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(ValueError, match="enlace"):
        run(inputs)


@pytest.mark.parametrize("budget", ["_MAX_PDF", "_MAX_CSV"])
def test_size_limits_reject_sources_before_parsing(inputs, monkeypatch, budget):
    monkeypatch.setattr(api(), budget, 8)
    with pytest.raises(ValueError, match="presupuesto|límite"):
        run(inputs)


def test_month_limit_is_checked_before_reading_the_sources(inputs, monkeypatch):
    monkeypatch.setattr(api(), "_MAX_MONTHS", 0)
    with pytest.raises(ValueError):
        run(inputs)


@pytest.mark.parametrize("target", ["base", "output"])
@pytest.mark.parametrize("artifact", ["prices.parquet", "market-factors.json", "report.json"])
def test_changed_base_or_cached_artifact_is_rejected(inputs, target, artifact):
    if target == "output":
        run(inputs)
    root = inputs[0] if target == "base" else inputs[2]
    (root / artifact).write_bytes(b"{}")
    with pytest.raises(ValueError):
        run(inputs)


def test_changed_cached_quotes_are_rejected_even_with_consistent_artifact_hashes(inputs):
    run(inputs)
    output = inputs[2]
    table = pq.read_table(output / "prices.parquet")
    rows = table.to_pylist()
    rows[0]["close"] = 4002.25
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), output / "prices.parquet")
    digest = sha256(output / "prices.parquet")
    spec = json.loads((output / "market-factors.json").read_text())
    spec["CN"]["prices_sha256"] = digest
    atomic_json(output / "market-factors.json", spec)
    report = json.loads((output / "report.json").read_text())
    report["artifacts"] = {name: sha256(output / name) for name in report["artifacts"]}
    atomic_json(output / "report.json", report)
    with pytest.raises(ValueError, match="contenido|cotizaciones|coincide"):
        run(inputs)


def test_partial_or_wrong_market_calendar_is_rejected(inputs):
    for clock in (
        MarketClock("CN", "2021-12-02", "2022-01-31"),
        MarketClock("US", "2021-12-01", "2022-01-31"),
    ):
        with pytest.raises(ValueError):
            api().extend_csi300_history(*inputs[:3], clock=clock)


def test_output_cannot_contain_or_overwrite_its_sources(inputs):
    for output in (inputs[0], inputs[0] / "new", inputs[1].parent / "new", inputs[1].parent.parent):
        with pytest.raises(ValueError):
            api().extend_csi300_history(inputs[0], inputs[1], output, clock=inputs[3])


def test_interruption_does_not_publish_partial_results_and_can_recover(inputs, monkeypatch):
    with monkeypatch.context() as patch:

        def interrupted(*args):
            raise OSError("Corte de prueba")

        patch.setattr(api(), "_publish_directory", interrupted)
        with pytest.raises(OSError):
            run(inputs)
    assert not inputs[2].exists()
    assert run(inputs)["rows"] == 42


def test_source_change_after_writing_is_rejected_before_publication(inputs, monkeypatch):
    module = api()
    original = module.atomic_parquet

    def changed(path, table):
        original(path, table)
        pdf = inputs[1].parent / "202112.pdf"
        pdf.write_bytes(pdf.read_bytes() + b" ")

    monkeypatch.setattr(module, "atomic_parquet", changed)
    with pytest.raises(ValueError, match="cambi"):
        run(inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("target", ["base", "output"])
def test_null_artifact_hash_cannot_disable_verification(inputs, target):
    if target == "output":
        run(inputs)
    root = inputs[0] if target == "base" else inputs[2]
    spec = json.loads((root / "market-factors.json").read_text())
    spec["CN"]["prices_sha256"] = None
    atomic_json(root / "market-factors.json", spec)
    report = json.loads((root / "report.json").read_text())
    report["artifacts"]["prices.parquet"] = None
    report["artifacts"]["market-factors.json"] = sha256(root / "market-factors.json")
    atomic_json(root / "report.json", report)
    with pytest.raises(ValueError, match="huella"):
        run(inputs)


def test_malformed_csv_is_a_validation_error(inputs):
    manifest = inputs[1]
    meta = json.loads(manifest.read_text())
    path = manifest.parent / meta["sources"][0]["csv_path"]
    path.write_text(",".join(COLUMNS) + '\n"2021-12-01')
    meta["sources"][0]["csv_sha256"] = sha256(path)
    atomic_json(manifest, meta)
    with pytest.raises(ValueError, match="CSV"):
        run(inputs)


def test_base_policy_flags_must_keep_their_boolean_type(inputs):
    path = inputs[0] / "report.json"
    report = json.loads(path.read_text())
    report["final_test_opened"] = 0
    atomic_json(path, report)
    with pytest.raises(ValueError):
        run(inputs)


def test_official_bilingual_csi300_title_is_admitted_as_an_explicit_variant(inputs):
    path = inputs[1]
    meta = json.loads(path.read_text())
    meta["sources"][0]["title"] = "沪深 300 指数 SHSE-SZSE 300 Index"
    atomic_json(path, meta)
    assert run(inputs)["added_rows"] == 23


@pytest.mark.parametrize(
    "field,value", [("final_test_opened", 0), ("schema_version", True), ("rows", 42.0)]
)
def test_reuse_rejects_report_type_changes_without_changing_values(inputs, field, value):
    run(inputs)
    path = inputs[2] / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    atomic_json(path, report)
    with pytest.raises(ValueError):
        run(inputs)


@pytest.mark.parametrize("field", ["pdf_page_1_based", "printed_page"])
def test_reuse_rejects_nested_provenance_type_changes(inputs, field):
    run(inputs)
    path = inputs[2] / "report.json"
    report = json.loads(path.read_text())
    entry = report["identity"]["reviewed_sources"][0]
    entry[field] = float(entry[field])
    atomic_json(path, report)
    with pytest.raises(ValueError):
        run(inputs)


@pytest.mark.parametrize("target", ["base", "output"])
def test_factor_descriptor_types_cannot_change_even_after_rehashing(inputs, target):
    if target == "output":
        run(inputs)
    root = inputs[0] if target == "base" else inputs[2]
    path = root / "market-factors.json"
    specification = json.loads(path.read_text())
    specification["CN"]["point_in_time_verified"] = 0
    atomic_json(path, specification)
    report = json.loads((root / "report.json").read_text())
    report["artifacts"]["market-factors.json"] = sha256(path)
    atomic_json(root / "report.json", report)
    with pytest.raises(ValueError):
        run(inputs)
