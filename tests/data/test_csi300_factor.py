"""Factor retrospectivo CSI 300 con origen, calendario y recuperación explícitos."""

import importlib
import json
from datetime import UTC, datetime
from urllib.parse import urlencode

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock

PAGE = "https://www.sse.com.cn/aboutus/publication/monthly/index/"
QUERY = "COMMON_SSE_ZQZS_M_CSI300_INDEX_C"


def api():
    try:
        return importlib.import_module("mars_titan.data.csi300_factor")
    except ModuleNotFoundError:
        pytest.fail("Falta la materialización del factor CSI 300")


def refresh(source, rows, *, month="202303"):
    body = source / f"{month}.jsonp"
    body.write_text("monthlyData(" + json.dumps({"result": rows, "actionErrors": []}) + ")")
    url = "https://query.sse.com.cn/commonQuery.do?" + urlencode(
        dict(sqlId=QUERY, isPagination="false", MDATE=month, jsonCallBack="monthlyData")
    )
    request = dict(
        status=200,
        url=url,
        final_url=url,
        requested_at_utc="2026-10-06T10:00:00+00:00",
        bytes=body.stat().st_size,
        sha256=sha256(body),
    )
    atomic_json(source / f"{month}.json", request)
    manifest = dict(
        schema_version=1,
        status="completed",
        query_id=QUERY,
        source_url=PAGE,
        acquired_at_utc="2026-10-06T11:00:00+00:00",
        months=[
            dict(
                month=month,
                rows=len(rows),
                sha256=sha256(body),
                receipt_sha256=sha256(source / f"{month}.json"),
            )
        ],
        rows=len(rows),
        first=min(row["MDATE"] for row in rows),
        last=max(row["MDATE"] for row in rows),
        final_test_opened=False,
        training_ready=False,
    )
    atomic_json(source / "acquisition.json", manifest)


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    clock = MarketClock("CN", "2023-03-01", "2023-03-31")
    rows = [
        dict(
            MDATE=d.strftime("%Y%m%d"),
            OPEN="4000.00",
            HIGH="4010.00",
            LOW="3990.00",
            CLS=f"{4000 + i % 5}.25",
        )
        for i, d in enumerate(clock.days)
    ]
    refresh(source, rows)
    return source, rows, tmp_path / "factor", clock


def run(inputs):
    source, _, output, clock = inputs
    return api().materialize_csi300_factor(source / "acquisition.json", output, clock=clock)


def test_official_columns_and_retrospective_availability_reach_the_factor_contract(inputs):
    source, rows, output, clock = inputs
    before = {p.name: sha256(p) for p in source.iterdir()}
    report = run(inputs)
    table = pq.read_table(output / "prices.parquet")
    assert table.num_rows == len(clock.days) == 23
    assert table["session"].to_pylist() == [d.isoformat() for d in clock.days]
    assert table["available_at"].to_pylist() == clock.decisions
    assert table["open"].to_pylist() == [float(row["OPEN"]) for row in rows]
    assert table["close"].to_pylist() == [float(row["CLS"]) for row in rows]
    assert report["missing_sessions"] == []
    assert report["point_in_time_verified"] is False
    assert report["financial_simulation_ready"] is False
    assert report["availability_policy"] == "retrospective_session_close_plus_5_minutes"
    spec = json.loads((output / "market-factors.json").read_text())["CN"]
    assert spec["market"] == "CN" and spec["symbol"] == "000300"
    assert spec["prices_path"] == str((output / "prices.parquet").resolve())
    assert spec["prices_sha256"] == sha256(output / "prices.parquet")
    assert before == {p.name: sha256(p) for p in source.iterdir()}
    times = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
    assert run(inputs) == {**report, "reused": True}
    assert times == {p.name: p.stat().st_mtime_ns for p in output.iterdir()}


def test_missing_sessions_stay_absent_and_are_recorded(inputs):
    source, rows, output, clock = inputs
    missing = rows.pop(3)["MDATE"]
    refresh(source, rows)
    report = run(inputs)
    assert report["missing_sessions"] == [datetime.strptime(missing, "%Y%m%d").date().isoformat()]
    assert pq.read_table(output / "prices.parquet").num_rows == len(clock.days) - 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("OPEN", "0.00"),
        ("CLS", "NaN"),
        ("HIGH", "3999.00"),
        ("LOW", "4011.00"),
        ("OPEN", True),
        ("MDATE", "20230304"),
        ("MDATE", "20240401"),
    ],
)
def test_invalid_quote_or_date_blocks_publication(inputs, field, value):
    source, rows, output, _ = inputs
    rows[0][field] = value
    refresh(source, rows)
    with pytest.raises(ValueError):
        run(inputs)
    assert not output.exists()


def test_duplicate_dates_block_publication_even_when_values_match(inputs):
    source, rows, _, _ = inputs
    rows.append(dict(rows[0]))
    refresh(source, rows)
    with pytest.raises(ValueError, match="repetida|duplicada"):
        run(inputs)


@pytest.mark.parametrize(
    "change", ["payload", "request", "callback", "url", "duplicate_key", "future_month"]
)
def test_changed_or_misidentified_sources_are_rejected(inputs, change):
    source, _, output, _ = inputs
    acquisition = source / "acquisition.json"
    meta = json.loads(acquisition.read_text())
    body = source / "202303.jsonp"
    request_path = source / "202303.json"
    request = json.loads(request_path.read_text())
    if change == "payload":
        body.write_bytes(body.read_bytes() + b" ")
    elif change == "request":
        atomic_json(request_path, {**request, "status": 500})
    elif change == "callback":
        body.write_text(body.read_text().replace("monthlyData(", "execute("))
    elif change == "duplicate_key":
        body.write_text(
            body.read_text().replace('"OPEN": "4000.00"', '"OPEN": "4000.00", "OPEN": "1.00"', 1)
        )
    elif change == "url":
        request["url"] = request["url"].replace("MDATE=202303", "MDATE=202304")
    else:
        meta["months"][0]["month"] = "202401"
    if change in {"callback", "duplicate_key", "url"}:
        request.update(sha256=sha256(body), bytes=body.stat().st_size)
        atomic_json(request_path, request)
        meta["months"][0].update(sha256=sha256(body), receipt_sha256=sha256(request_path))
    atomic_json(acquisition, meta)
    with pytest.raises(ValueError):
        run(inputs)
    assert not output.exists()


def test_oversized_payload_is_rejected_before_parsing(inputs, monkeypatch):
    monkeypatch.setattr(api(), "_MAX_PAYLOAD", 64)
    with pytest.raises(ValueError, match="presupuesto|límite"):
        run(inputs)


@pytest.mark.parametrize("artifact", ["prices.parquet", "market-factors.json", "report.json"])
def test_reuse_rejects_changed_artifacts(inputs, artifact):
    run(inputs)
    path = inputs[2] / artifact
    path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        run(inputs)


def test_output_must_be_independent_of_the_sources(inputs):
    source, _, _, clock = inputs
    with pytest.raises(ValueError):
        api().materialize_csi300_factor(source / "acquisition.json", source / "new", clock=clock)


def test_failed_publication_is_recoverable(inputs, monkeypatch):
    module = api()
    with monkeypatch.context() as patch:

        def interrupted(*args):
            raise OSError("Corte de prueba")

        patch.setattr(module, "_publish_directory", interrupted)
        with pytest.raises(OSError):
            run(inputs)
    assert not inputs[2].exists()
    assert run(inputs)["rows"] == 23


def test_wrong_market_cannot_assign_another_close_to_the_same_dates(inputs):
    source, _, output, _ = inputs
    with pytest.raises(ValueError, match="CN|chino"):
        api().materialize_csi300_factor(
            source / "acquisition.json", output, clock=MarketClock("US", "2023-03-01", "2023-03-31")
        )


def test_retrieval_time_is_not_used_as_historical_availability(inputs):
    run(inputs)
    table = pq.read_table(inputs[2] / "prices.parquet")
    assert max(table["available_at"].to_pylist()) < datetime(2024, 1, 1, tzinfo=UTC)


def test_partial_calendar_cannot_hide_missing_sessions_from_a_declared_month(inputs):
    source, rows, output, _ = inputs
    refresh(source, rows[:8])
    with pytest.raises(ValueError, match="íntegramente"):
        api().materialize_csi300_factor(
            source / "acquisition.json", output, clock=MarketClock("CN", "2023-03-01", "2023-03-10")
        )


def test_consistent_future_fixture_is_rejected_before_reading_its_month(tmp_path):
    source = tmp_path / "future-fixture"
    source.mkdir()
    clock = MarketClock("CN", "2024-01-01", "2024-01-31")
    rows = [
        dict(MDATE=day.strftime("%Y%m%d"), OPEN="10.00", HIGH="11.00", LOW="9.00", CLS="10.50")
        for day in clock.days
    ]
    refresh(source, rows, month="202401")
    with pytest.raises(ValueError, match="reserva final"):
        api().materialize_csi300_factor(
            source / "acquisition.json", tmp_path / "output", clock=clock
        )


def test_source_changed_after_parquet_write_cannot_be_published(inputs, monkeypatch):
    module = api()
    original = module.atomic_parquet

    def changed(path, table):
        original(path, table)
        raw = inputs[0] / "202303.jsonp"
        raw.write_bytes(raw.read_bytes() + b" ")

    monkeypatch.setattr(module, "atomic_parquet", changed)
    with pytest.raises(ValueError, match="cambió"):
        run(inputs)
    assert not inputs[2].exists()


def test_cli_writes_a_factor_descriptor_for_the_existing_consumer(inputs, capsys):
    source, _, output, _ = inputs
    api().main(
        [
            "--acquisition",
            str(source / "acquisition.json"),
            "--output",
            str(output),
            "--calendar-start",
            "2023-03-01",
            "--calendar-end",
            "2023-03-31",
        ]
    )
    assert "23 sesiones" in capsys.readouterr().out
    assert json.loads((output / "report.json").read_text())["market"] == "CN"
