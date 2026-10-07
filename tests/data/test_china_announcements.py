"""Catálogo por páginas, con transporte simulado y respuestas recuperables."""

import importlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def api():
    try:
        return importlib.import_module("mars_titan.data.china_announcements")
    except ModuleNotFoundError:
        pytest.fail("Falta el catálogo recuperable de anuncios chinos")


def notice(number, day="2022-03-31", code="000001", board="SZZB"):
    identifier = f"{day.replace('-', '')}{number:06d}"
    return dict(
        secCode=code,
        secName="Emisor de prueba",
        orgId=f"org{code}",
        pageColumn=board,
        announcementId=identifier,
        announcementTime=int(
            datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() * 1000
        ),
        announcementTitle="2021年年度报告（更正后）",
        adjunctUrl=f"finalpage/{day}/{identifier}.PDF",
        adjunctType="PDF",
    )


def response(rows, total=None, *, more=None, status=200):
    total = len(rows) if total is None else total
    value = dict(
        totalAnnouncement=total,
        totalRecordNum=total,
        totalpages=total // 30,
        hasMore=total > len(rows) if more is None else more,
        announcements=rows,
    )
    return status, {}, json.dumps(value).encode()


class Transport:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def __call__(self, parameters):
        self.calls.append(dict(parameters))
        result = next(self.replies)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "inventory"
    source.mkdir()
    queue = source / "reconciliation-queue.parquet"
    pq.write_table(
        pa.table(
            {
                "symbol": ["000001.SZ", "600079.SS", "000001.SZ"],
                "market": ["CN"] * 3,
                "period_end": ["2021-12-31", "2022-12-31", "2022-03-31"],
            }
        ),
        queue,
    )
    return dict(
        queue=queue,
        output=tmp_path / "catalogue",
        publication_start="2022-01-01",
        publication_end="2022-12-31",
    )


def collect(inputs, replies, budget=24):
    transport = Transport(replies)
    delays = []
    result = api().collect_chinese_announcements(
        **inputs, max_requests=budget, transport=transport, sleep=delays.append
    )
    return result, transport.calls, delays


def test_page_ceiling_and_all_markets_reach_a_reusable_catalogue(inputs):
    rows = [notice(i) for i in range(31)]
    rows[1] = notice(1, code="600079", board="SHZB")
    rows[2] = notice(2, code="831906", board="BJS")
    result, calls, delays = collect(
        inputs, [response(rows[:30], 31), response(rows[30:], 31, more=False)]
    )
    assert result["status"] == "completed"
    assert result["announcements"] == 31 and result["in_census"] == 30
    assert [call["pageNum"] for call in calls] == ["1", "2"]
    assert all(call["pageSize"] == "30" and call["searchkey"] == "" for call in calls)
    assert all(call["category"] == "category_ndbg_szsh" for call in calls)
    assert delays == [2, 2]
    table = pq.read_table(inputs["output"] / "announcements.parquet")
    assert len(table) == 31 and set(table["page_column"].to_pylist()) == {"SZZB", "SHZB", "BJS"}
    assert result["period_coverage_verified"] is False
    assert result["financial_values_admitted"] is False
    times = {p.name: p.stat().st_mtime_ns for p in inputs["output"].iterdir() if p.is_file()}
    again, calls, delays = collect(inputs, [])
    assert again["announcements"] == 31 and calls == delays == []
    assert times == {
        p.name: p.stat().st_mtime_ns for p in inputs["output"].iterdir() if p.is_file()
    }


def test_budget_resumes_without_repeating_a_confirmed_page(inputs):
    rows = [notice(i) for i in range(31)]
    partial, calls, _ = collect(inputs, [response(rows[:30], 31)], budget=1)
    assert partial["status"] == "partial" and len(calls) == 1
    assert not (inputs["output"] / "announcements.parquet").exists()
    done, calls, _ = collect(inputs, [response(rows[30:], 31, more=False)], budget=1)
    assert done["status"] == "completed" and calls[0]["pageNum"] == "2"


def test_transport_interruption_keeps_the_confirmed_cursor(inputs):
    rows = [notice(i) for i in range(31)]
    with pytest.raises(OSError, match="Corte"):
        collect(inputs, [response(rows[:30], 31), OSError("Corte simulado")])
    result, calls, _ = collect(inputs, [response(rows[30:], 31, more=False)])
    assert result["status"] == "completed" and calls[0]["pageNum"] == "2"


def test_saved_response_can_recover_a_cut_before_cursor_confirmation(inputs, monkeypatch):
    module = api()
    original = module.atomic_json

    def interrupted(path, value):
        if path.name == "cursor.json" and len(value["responses"]) == 1:
            raise OSError("Corte tras conservar la respuesta")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", interrupted)
        with pytest.raises(OSError):
            collect(inputs, [response([notice(1)])])
    result, calls, delays = collect(inputs, [])
    assert result["status"] == "completed" and calls == delays == []


def test_large_interval_splits_dates_and_checks_the_parent_population(inputs, monkeypatch):
    monkeypatch.setattr(api(), "_MAX_PAGES", 1)
    inputs.update(publication_start="2022-03-30", publication_end="2022-03-31")
    left = [notice(i, "2022-03-30") for i in range(15)]
    right = [notice(i) for i in range(16)]
    result, calls, _ = collect(
        inputs, [response((left + right)[:30], 31), response(left), response(right)]
    )
    assert result["status"] == "completed" and result["announcements"] == 31
    assert [c["seDate"] for c in calls] == [
        "2022-03-30~2022-03-31",
        "2022-03-30~2022-03-30",
        "2022-03-31~2022-03-31",
    ]
    assert all(c["pageNum"] == "1" for c in calls)


def test_single_day_above_the_interface_limit_stops_with_evidence(inputs):
    inputs.update(publication_start="2022-03-31", publication_end="2022-03-31")
    result, calls, _ = collect(inputs, [response([notice(i) for i in range(30)], 3001)])
    assert result["status"] == "blocked" and len(calls) == 1
    assert "día" in result["failure"]
    assert not (inputs["output"] / "announcements.parquet").exists()


@pytest.mark.parametrize("status", [403, 429])
def test_access_rejection_stops_immediately_and_is_not_retried_on_resume(inputs, status):
    result, calls, delays = collect(inputs, [(status, {}, b"Denied")])
    assert result["status"] == "blocked" and str(status) in result["failure"]
    assert len(calls) == 1 and delays == [2]
    again, calls, delays = collect(inputs, [])
    assert again["status"] == "blocked" and calls == delays == []


@pytest.mark.parametrize("change", ["total", "duplicate", "short", "has_more"])
def test_changed_pagination_stops_without_publishing_a_partial_catalogue(inputs, change):
    first = [notice(i) for i in range(30)]
    second = response([notice(30)], 31, more=False)
    if change == "total":
        second = response([notice(30)], 32, more=False)
    elif change == "duplicate":
        second = response([notice(0)], 31, more=False)
    elif change == "short":
        first = first[:-1]
    else:
        second = response([notice(30)], 31, more=True)
    result, _, _ = collect(inputs, [response(first, 31), second])
    assert result["status"] == "blocked"
    assert not (inputs["output"] / "announcements.parquet").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("secCode", "1"),
        ("announcementId", "../other"),
        ("announcementTime", 1704067200000),
        ("adjunctUrl", "https://foreign.example/document.pdf"),
        ("adjunctUrl", 7),
        ("adjunctUrl", "finalpage/2022-03-30/20220331000001.PDF"),
        ("adjunctType", "XLS"),
    ],
)
def test_invalid_document_identity_is_preserved_but_not_confirmed(inputs, field, value):
    row = notice(1)
    row[field] = value
    result, calls, _ = collect(inputs, [response([row])])
    assert result["status"] == "blocked" and len(calls) == 1


def test_response_size_budget_is_enforced(inputs, monkeypatch):
    monkeypatch.setattr(api(), "_MAX_RESPONSE", 64)
    result, _, _ = collect(inputs, [response([notice(1)])])
    assert result["status"] == "blocked"


@pytest.mark.parametrize("source", ["queue", "response", "cursor", "configuration", "catalogue"])
def test_altered_sources_and_confirmed_outputs_cannot_be_adopted(inputs, source):
    collect(inputs, [response([notice(1)])])
    output = inputs["output"]
    if source == "queue":
        path = inputs["queue"]
    elif source == "response":
        path = next((output / "responses").glob("*/body.json"))
    else:
        path = (
            output
            / {
                "cursor": "cursor.json",
                "configuration": "configuration.json",
                "catalogue": "announcements.parquet",
            }[source]
        )
    path.write_bytes(b"alterado")
    with pytest.raises((ValueError, OSError)):
        collect(inputs, [])


def test_reserved_publication_dates_fail_before_transport(inputs):
    inputs["publication_end"] = "2024-01-01"
    with pytest.raises(ValueError):
        collect(inputs, [])
    assert not inputs["output"].exists()


def test_empty_interval_is_a_complete_empty_catalogue(inputs):
    result, calls, _ = collect(inputs, [response([], 0, more=False)])
    assert result["status"] == "completed" and result["announcements"] == 0
    assert (
        len(calls) == 1 and pq.read_table(inputs["output"] / "announcements.parquet").num_rows == 0
    )


def test_completed_collection_cannot_recreate_a_missing_confirmed_cursor(inputs):
    collect(inputs, [response([notice(1)])])
    (inputs["output"] / "cursor.json").unlink()
    with pytest.raises(ValueError):
        collect(inputs, [])


def test_completed_collection_rejects_rollback_to_an_earlier_valid_cursor(inputs):
    with pytest.raises(OSError):
        collect(inputs, [OSError("Corte antes de responder")])
    path = inputs["output"] / "cursor.json"
    previous = path.read_bytes()
    collect(inputs, [response([notice(1)])])
    path.write_bytes(previous)
    with pytest.raises(ValueError):
        collect(inputs, [])


def test_previous_response_cannot_change_during_a_later_request(inputs):
    rows = [notice(i) for i in range(31)]
    collect(inputs, [response(rows[:30], 31)], budget=1)
    saved = next((inputs["output"] / "responses").glob("*/body.json"))

    def changed(parameters):
        saved.write_bytes(saved.read_bytes() + b" ")
        return response(rows[30:], 31, more=False)

    with pytest.raises(ValueError, match="cambi"):
        api().collect_chinese_announcements(
            **inputs,
            max_requests=1,
            transport=changed,
            sleep=lambda seconds: None,
        )
    assert not (inputs["output"] / "report.json").exists()


def test_queue_change_during_catalogue_write_prevents_final_confirmation(inputs, monkeypatch):
    module = api()
    original = module.atomic_parquet

    def changed(path, table):
        original(path, table)
        queue = inputs["queue"]
        queue.write_bytes(queue.read_bytes() + b" ")

    monkeypatch.setattr(module, "atomic_parquet", changed)
    with pytest.raises(ValueError, match="cambi"):
        collect(inputs, [response([notice(1)])])
    assert not (inputs["output"] / "report.json").exists()


def test_cursor_change_during_transport_is_not_overwritten(inputs):
    def changed(parameters):
        (inputs["output"] / "cursor.json").write_text("{}")
        return response([notice(1)])

    with pytest.raises(ValueError, match="cambi"):
        api().collect_chinese_announcements(
            **inputs,
            max_requests=1,
            transport=changed,
            sleep=lambda seconds: None,
        )
    assert not (inputs["output"] / "report.json").exists()


@pytest.mark.parametrize("padding", ["", " ", "\t"])
def test_incomplete_http_body_is_preserved_and_blocks_confirmation(inputs, padding):
    status, _, body = response([notice(1)])
    declared = f"{padding}{len(body) + 10}{padding}"
    result, _, _ = collect(inputs, [(status, {"Content-Length": declared}, body)])
    assert result["status"] == "blocked"
    assert next((inputs["output"] / "responses").glob("*/body.json")).read_bytes() == body


@pytest.mark.parametrize("clipped", [False, True])
def test_orphan_flag_cannot_override_header_or_read_length(inputs, monkeypatch, clipped):
    module = api()
    status, _, body = response([notice(1)])
    if clipped:
        monkeypatch.setattr(module, "_MAX_RESPONSE", len(body))
    declared = len(body) if clipped else len(body) + 10
    received = body + b" " if clipped else body
    original = module.atomic_json

    def interrupted(path, value):
        if path.name == "cursor.json" and len(value["responses"]) == 1:
            raise OSError("Corte antes de confirmar el cursor")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", interrupted)
        with pytest.raises(OSError):
            collect(inputs, [(status, {"Content-Length": str(declared)}, received)])
    path = next((inputs["output"] / "responses").glob("*/receipt.json"))
    receipt = json.loads(path.read_text())
    assert receipt["truncated"] is True
    receipt["truncated"] = False
    path.write_text(json.dumps(receipt))
    retained = path.read_bytes()
    result, calls, delays = collect(inputs, [])
    assert result["status"] == "blocked" and calls == delays == []
    assert "completa" in result["failure"]
    assert path.read_bytes() == retained
    assert not (inputs["output"] / "announcements.parquet").exists()


@pytest.mark.parametrize("padding", [" ", "\t"])
def test_complete_http_body_accepts_optional_content_length_whitespace(inputs, padding):
    status, _, body = response([notice(1)])
    declared = f"{padding}{len(body)}{padding}"
    result, _, _ = collect(inputs, [(status, {"Content-Length": declared}, body)])
    assert result["status"] == "completed"


def test_body_at_the_limit_is_complete_with_a_matching_declared_length(inputs, monkeypatch):
    status, _, body = response([notice(1)])
    monkeypatch.setattr(api(), "_MAX_RESPONSE", len(body))
    result, _, _ = collect(inputs, [(status, {"Content-Length": str(len(body))}, body)])
    assert result["status"] == "completed"


@pytest.mark.parametrize("change", ["total", "document"])
def test_split_children_must_preserve_parent_totals_and_documents(inputs, monkeypatch, change):
    monkeypatch.setattr(api(), "_MAX_PAGES", 1)
    inputs.update(publication_start="2022-03-30", publication_end="2022-03-31")
    left = [notice(i, "2022-03-30") for i in range(15)]
    right = [notice(i) for i in range(16)]
    parent = response((left + right)[:30], 31)
    if change == "total":
        right.pop()
    else:
        right[0]["announcementTitle"] += " cambiado"
    result, calls, _ = collect(inputs, [parent, response(left), response(right)])
    assert result["status"] == "blocked" and len(calls) == 3
    assert not (inputs["output"] / "announcements.parquet").exists()


def test_only_one_writer_can_issue_requests(inputs):
    import fcntl

    output = inputs["output"]
    output.mkdir()
    with (output / ".catalogue.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            collect(inputs, [])
    assert not (output / "configuration.json").exists()


def test_public_transport_uses_post_and_bounded_reads_without_retry(monkeypatch):
    module = api()
    observed = {}

    class Reply:
        status = 200
        headers = {"Content-Length": "2"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            observed["read_limit"] = count
            return b"{}"

    class Opener:
        def open(self, request, *, timeout):
            observed.update(
                method=request.get_method(),
                url=request.full_url,
                parameters=module.urllib.parse.parse_qs(request.data.decode()),
                timeout=timeout,
            )
            return Reply()

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    status, _, body = module._request({"pageNum": "1", "searchkey": ""})
    assert status == 200 and body == b"{}"
    assert observed == {
        "method": "POST",
        "url": "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        "parameters": {"pageNum": ["1"]},
        "timeout": 30,
        "read_limit": 1024**2,
    }


def test_cli_keeps_annual_scope_explicit_for_a_queue_with_quarterly_closes(inputs, capsys):
    module = api()
    transport = Transport([response([notice(1)])])
    code = module.main(
        [
            "--queue",
            str(inputs["queue"]),
            "--output",
            str(inputs["output"]),
            "--publication-start",
            inputs["publication_start"],
            "--publication-end",
            inputs["publication_end"],
            "--max-requests",
            "1",
        ],
        transport=transport,
        sleep=lambda seconds: None,
    )
    assert code == 0 and "completed" in capsys.readouterr().out
    report = json.loads((inputs["output"] / "report.json").read_text())
    config = json.loads((inputs["output"] / "configuration.json").read_text())
    assert "2022-03-31" in report["queue_periods"]
    assert report["scope"] == config["scope"] == "annual_category_only"
    assert report["category"] == config["category"] == "category_ndbg_szsh"
    assert report["period_coverage_verified"] is False
