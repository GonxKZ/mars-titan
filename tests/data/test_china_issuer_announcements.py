"""El ámbito por emisor conserva su evidencia y las comprobaciones del catálogo."""

import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256
from tests.data.test_china_announcements import Transport, api, collect, notice, response
from tests.data.test_china_announcements import inputs as inputs


@pytest.fixture
def issuer(tmp_path):
    module = api()
    folder = tmp_path / "issuer-evidence"
    module._save_response(
        folder,
        module._task("2022-01-01", "2022-12-31"),
        "a" * 64,
        response([notice(7)]),
        "2026-10-07T00:00:00+00:00",
        False,
    )
    return dict(
        issuer_symbol="000001.SZ",
        issuer_receipt=folder / "receipt.json",
        issuer_receipt_sha256=sha256(folder / "receipt.json"),
        issuer_announcement_id=notice(7)["announcementId"],
    )


def scoped(inputs, issuer, replies, budget=1):
    transport, delays = Transport(replies), []
    result = api().collect_chinese_announcements(
        **inputs, **issuer, max_requests=budget, transport=transport, sleep=delays.append
    )
    return result, transport.calls, delays


def test_issuer_scope_is_derived_from_a_frozen_announcement(inputs, issuer):
    result, calls, delays = scoped(inputs, issuer, [response([notice(7)])])
    assert result["status"] == "completed"
    assert result["schema_version"] == 2
    assert result["scope"] == "issuer_annual_category_only"
    assert result["issuer"]["symbol"] == "000001.SZ"
    assert result["issuer"]["org_id"] == "org000001"
    assert result["issuer"]["receipt_sha256"] == issuer["issuer_receipt_sha256"]
    assert result["known_announcement_required"] is True
    assert result["known_announcement_present"] is True
    assert result["issuer_history_complete"] is False
    assert result["period_coverage_verified"] is False
    assert calls[0]["stock"] == "000001,org000001" and delays == [5]
    config = json.loads((inputs["output"] / "configuration.json").read_text())
    assert config["schema_version"] == 2 and config["issuer"] == result["issuer"]
    assert config["limits"]["request_spacing_seconds"] == 5
    saved = next((inputs["output"] / "responses").glob("*/receipt.json"))
    assert json.loads(saved.read_text())["parameters"] == calls[0]


@pytest.mark.parametrize("fault", ["symbol", "not_in_census", "announcement", "hash", "partial"])
def test_wrong_or_partial_issuer_selection_fails_before_request(inputs, issuer, fault):
    if fault == "symbol":
        issuer["issuer_symbol"] = "600079.SS"
    elif fault == "not_in_census":
        issuer["issuer_symbol"] = "000002.SZ"
    elif fault == "announcement":
        issuer["issuer_announcement_id"] = "999"
    elif fault == "hash":
        issuer["issuer_receipt_sha256"] = "b" * 64
    else:
        issuer.pop("issuer_receipt_sha256")
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [])
    assert not inputs["output"].exists()


@pytest.mark.parametrize("fault", ["body", "receipt", "injected", "url", "org_delimiter"])
def test_altered_or_non_public_identity_evidence_is_rejected(inputs, issuer, fault):
    path = issuer["issuer_receipt"]
    receipt = json.loads(path.read_text())
    if fault == "body":
        body = path.with_name("body.json")
        body.write_bytes(body.read_bytes() + b" ")
    elif fault == "receipt":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        if fault == "injected":
            receipt["transport"] = "injected"
        elif fault == "url":
            receipt["url"] = "https://example.invalid/query"
        else:
            body = path.with_name("body.json")
            value = json.loads(body.read_text())
            value["announcements"][0]["orgId"] = "org000001,600079"
            body.write_text(json.dumps(value))
            receipt.update(
                sha256=sha256(body), bytes=body.stat().st_size, read_bytes=body.stat().st_size
            )
        path.write_text(json.dumps(receipt))
        issuer["issuer_receipt_sha256"] = sha256(path)
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [])
    assert not inputs["output"].exists()


@pytest.mark.parametrize("fault", ["code", "org", "market"])
def test_ignored_or_crossed_issuer_filter_stops_and_keeps_the_response(inputs, issuer, fault):
    other = notice(8)
    other[{"code": "secCode", "org": "orgId", "market": "pageColumn"}[fault]] = {
        "code": "600079",
        "org": "another",
        "market": "SHZB",
    }[fault]
    result, calls, _ = scoped(inputs, issuer, [response([notice(7), other])])
    assert result["status"] == "blocked" and len(calls) == 1
    assert not (inputs["output"] / "announcements.parquet").exists()
    assert len(list((inputs["output"] / "responses").glob("*/body.json"))) == 1
    again, calls, _ = scoped(inputs, issuer, [])
    assert again["status"] == "blocked" and calls == []


@pytest.mark.parametrize("rows", [[], [notice(8)]], ids=["empty", "known_notice_missing"])
def test_missing_known_announcement_cannot_complete_its_publication_window(inputs, issuer, rows):
    result, _, _ = scoped(inputs, issuer, [response(rows)])
    assert result["status"] == "blocked"
    assert result["known_announcement_required"] is True
    assert result["known_announcement_present"] is False
    assert not (inputs["output"] / "announcements.parquet").exists()


def test_empty_other_window_only_records_no_returned_results(inputs, issuer):
    inputs.update(publication_start="2023-01-01", publication_end="2023-12-31")
    result, _, _ = scoped(inputs, issuer, [response([])])
    assert result["status"] == "completed" and result["announcements"] == 0
    assert result["known_announcement_required"] is False
    assert result["known_announcement_present"] is None
    assert result["issuer_history_complete"] is False
    assert len(pq.read_table(inputs["output"] / "announcements.parquet")) == 0


def test_issuer_resume_keeps_the_exact_scope_and_confirmed_page(inputs, issuer):
    rows = [notice(i) for i in range(31)]
    partial, _, _ = scoped(inputs, issuer, [response(rows[:30], 31)])
    assert partial["status"] == "partial"
    first = next((inputs["output"] / "responses").glob("*/body.json"))
    before = sha256(first), first.stat().st_mtime_ns
    done, calls, _ = scoped(inputs, issuer, [response(rows[30:], 31, more=False)])
    assert done["status"] == "completed" and calls[0]["pageNum"] == "2"
    assert calls[0]["stock"] == "000001,org000001"
    assert (sha256(first), first.stat().st_mtime_ns) == before
    again, calls, delays = scoped(inputs, issuer, [])
    assert again["status"] == "completed" and calls == delays == []


def test_issuer_evidence_is_rechecked_when_resuming(inputs, issuer):
    rows = [notice(i) for i in range(31)]
    scoped(inputs, issuer, [response(rows[:30], 31)])
    cursor = (inputs["output"] / "cursor.json").read_bytes()
    path = issuer["issuer_receipt"].with_name("body.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [])
    assert (inputs["output"] / "cursor.json").read_bytes() == cursor


@pytest.mark.parametrize("fault", ["none_notices", "out_of_range_page"])
def test_identity_evidence_needs_a_real_announcement_on_a_valid_page(inputs, issuer, fault):
    receipt_path = issuer["issuer_receipt"]
    receipt = json.loads(receipt_path.read_text())
    body_path = receipt_path.with_name("body.json")
    body = json.loads(body_path.read_text())
    if fault == "none_notices":
        body.update(announcements=None, totalRecordNum=0, totalAnnouncement=0)
    else:
        receipt["parameters"]["pageNum"] = "101"
        body.update(totalRecordNum=3001, totalAnnouncement=3001, totalpages=100)
    body_path.write_text(json.dumps(body))
    receipt.update(
        sha256=sha256(body_path),
        bytes=body_path.stat().st_size,
        read_bytes=body_path.stat().st_size,
    )
    receipt_path.write_text(json.dumps(receipt))
    issuer["issuer_receipt_sha256"] = sha256(receipt_path)
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [response([notice(7)])])
    assert not inputs["output"].exists()


def test_evidence_changed_during_transport_stops_then_recovers_the_orphan_response(inputs, issuer):
    path = issuer["issuer_receipt"].with_name("body.json")
    original = path.read_bytes()

    def changed(_):
        path.write_bytes(original + b" ")
        return response([notice(7)])

    with pytest.raises(ValueError):
        api().collect_chinese_announcements(
            **inputs, **issuer, max_requests=1, transport=changed, sleep=lambda _: None
        )
    assert not (inputs["output"] / "report.json").exists()
    path.write_bytes(original)
    done, calls, delays = scoped(inputs, issuer, [])
    assert done["status"] == "completed" and calls == delays == []


def test_saved_issuer_response_cannot_hide_a_different_stock_filter(inputs, issuer, monkeypatch):
    module = api()
    original = module.atomic_json

    def cut(path, value):
        if path.name == "cursor.json" and value["responses"]:
            raise OSError("Corte después de conservar la respuesta")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", cut)
        with pytest.raises(OSError):
            scoped(inputs, issuer, [response([notice(7)])])
    path = next((inputs["output"] / "responses").glob("*/receipt.json"))
    value = json.loads(path.read_text())
    value["parameters"]["stock"] = ""
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [])


@pytest.mark.parametrize("fault", ["duplicate", "total", "has_more"])
def test_issuer_scope_keeps_pagination_integrity_checks(inputs, issuer, fault):
    status, headers, raw = response([notice(7), notice(8)])
    body = json.loads(raw)
    if fault == "duplicate":
        body["announcements"][1] = body["announcements"][0]
    elif fault == "total":
        body["totalRecordNum"] = 3
    else:
        body["hasMore"] = True
    result, _, _ = scoped(inputs, issuer, [(status, headers, json.dumps(body).encode())])
    assert result["status"] == "blocked"
    assert not (inputs["output"] / "announcements.parquet").exists()


def test_general_collection_cannot_be_reinterpreted_as_an_issuer_scope(inputs, issuer):
    collect(inputs, [response([notice(7)])])
    previous = {
        p.relative_to(inputs["output"]): p.read_bytes()
        for p in inputs["output"].rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError):
        scoped(inputs, issuer, [])
    assert previous == {name: (inputs["output"] / name).read_bytes() for name in previous}
    result, calls, _ = collect(inputs, [])
    assert result["schema_version"] == 1 and "issuer" not in result and calls == []


def test_issuer_cli_requires_the_same_frozen_evidence(inputs, issuer, capsys):
    options = {**inputs, **issuer, "max_requests": 1}
    arguments = [
        part
        for name, value in options.items()
        for part in ("--" + name.replace("_", "-"), str(value))
    ]
    transport = Transport([response([notice(7)])])
    assert api().main(arguments, transport=transport, sleep=lambda _: None) == 0
    assert transport.calls[0]["stock"] == "000001,org000001"
    assert "completed" in capsys.readouterr().out
