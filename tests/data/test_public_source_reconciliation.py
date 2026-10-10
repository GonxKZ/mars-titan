"""Reconciliación del catálogo público con las capturas confirmadas, sin red."""

import hashlib
import json
from pathlib import Path

import pytest

from mars_titan.data import public_snapshots
from mars_titan.data.public_source_reconciliation import (
    CATALOG,
    DECISIONS,
    decision_problems,
    main,
    reconcile,
)

BODY = b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,15,16,14,15.5\n"
PDF = b"%PDF-1.7 documento fijo"
DECISION = dict(use="Contraste local", gap="Carencia comprobada", decision="pending", evidence="Ok")
TERMS = dict(
    reviewed_on="2026-10-10",
    status="verified",
    local_use="permitted",
    redistribution="not_granted",
    evidence_urls=["https://example.org/terms"],
    conditions="Uso local con cita, sin permiso para redistribuir.",
)


def source(identifier, status, validator="vix_csv", **changes):
    return (
        dict(
            id=identifier,
            provider="Proveedor",
            status=status,
            validator=validator,
            license_unknown=True,
            benchmark_eligible=False,
            use_decisions=[DECISION],
            terms_review=TERMS,
        )
        | changes
    )


def valid(identifier, path, body):
    return dict(
        source_id=identifier,
        status="downloaded_validated",
        http_status=200,
        sha256=hashlib.sha256(body).hexdigest(),
        bytes=len(body),
        local_path=path,
    )


def store(tmp_path, records, diagnostics=()):
    """Captura inicial con los registros dados, y sus archivos en disco."""
    for record in records:
        if record.get("local_path") and record.get("_body") is not None:
            target = tmp_path / record["local_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(record.pop("_body"))
        record.pop("_body", None)
    manifest = tmp_path / public_snapshots.INITIAL
    manifest.parent.mkdir(parents=True, exist_ok=True)
    document = dict(schema_version=1, records=records, diagnostics=list(diagnostics))
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return public_snapshots.build_index(tmp_path, tmp_path / "data/external")


def rows(report):
    return {row["source_id"]: row for row in report["sources"]}


def test_each_catalog_status_needs_matching_capture_evidence(tmp_path):
    index = store(
        tmp_path,
        [
            valid("vix", "data/external/a/vix.csv", BODY) | {"_body": BODY},
            valid("pdf", "data/external/a/doc.pdf", PDF) | {"_body": PDF},
            valid("lost", "data/external/a/lost.csv", BODY + b"x"),
            dict(source_id="sec", status="failed", http_status=403),
            dict(source_id="gdelt", status="failed", http_status=200),
            dict(source_id="stooq", status="blocked_at_source_discovery", http_status=200),
        ],
        diagnostics=[dict(source_id="gdelt", http_status=429, retained_payload=False)],
    )
    catalog = dict(
        sources=[
            source("vix", "downloaded_validated"),
            source("pdf", "downloaded_validated", validator="financial_pdf"),
            source("lost", "downloaded_validated"),
            source("sec", "failed"),
            source("gdelt", "failed"),
            source("stooq", "blocked_at_source_discovery", validator=None),
            source("never", "failed"),
        ]
    )
    report = reconcile(catalog, index)
    found = rows(report)
    assert {name: row["evidence"] for name, row in found.items()} == {
        "vix": "acquired_validated",
        "pdf": "fixed_document",
        "lost": "content_missing",
        "sec": "failed_attempt",
        "gdelt": "failed_attempt",
        "stooq": "blocked_at_source_discovery",
        "never": "not_attempted",
    }
    assert found["sec"]["failure_http_statuses"] == [403]
    assert found["gdelt"]["diagnostic_http_statuses"] == [429]
    # Un manifiesto válido sin bytes y una fuente nunca intentada no cuadran con el catálogo.
    assert report["problems"] == [
        "lost: estado downloaded_validated sin evidencia coherente (content_missing)",
        "never: estado failed sin evidencia coherente (not_attempted)",
    ]
    assert report["benchmark_eligible"] is False


def test_a_failed_source_with_a_valid_capture_is_inconsistent(tmp_path):
    index = store(tmp_path, [valid("vix", "data/vix.csv", BODY) | {"_body": BODY}])
    report = reconcile(dict(sources=[source("vix", "failed")]), index)
    assert report["problems"] == ["vix: estado failed sin evidencia coherente (acquired_validated)"]


def test_captures_without_a_catalog_source_are_reported(tmp_path):
    index = store(tmp_path, [valid("vix", "data/vix.csv", BODY) | {"_body": BODY}])
    assert reconcile(dict(sources=[]), index)["problems"] == [
        "vix: captura sin fuente en el catálogo"
    ]


@pytest.mark.parametrize(
    "changes, problem",
    [
        (dict(use_decisions=[]), "sin decisiones por uso"),
        (dict(use_decisions=[DECISION | {"decision": "admitted"}]), "valor no admitido"),
        (dict(use_decisions=[DECISION | {"gap": " "}]), "incompleta"),
        (dict(benchmark_eligible=True), "benchmark_eligible"),
        (dict(benchmark_eligible=None), "benchmark_eligible"),
    ],
)
def test_use_decisions_are_complete_and_never_benchmark_eligible(changes, problem):
    problems = decision_problems(source("vix", "downloaded_validated", **changes))
    assert len(problems) == 1 and problem in problems[0]


@pytest.mark.parametrize(
    "changes, problem",
    [
        (dict(terms_review=None), "revisión de condiciones incompleta"),
        (dict(terms_review=TERMS | {"reviewed_on": "10/10/2026"}), "incompleta"),
        (dict(terms_review=TERMS | {"local_use": "free"}), "incompleta"),
        (dict(terms_review=TERMS | {"conditions": " "}), "incompleta"),
        (dict(terms_review=TERMS | {"evidence_urls": []}), "incompleta"),
        (dict(terms_review=TERMS | {"evidence_urls": ["http://example.org"]}), "incompleta"),
        (
            dict(
                use_decisions=[DECISION | {"decision": "admitted_exploratory"}],
                terms_review=TERMS | {"local_use": "not_stated"},
            ),
            "uso local admitido",
        ),
        (dict(license_unknown=False), "license_unknown no coincide"),
        (
            dict(
                license_unknown=False,
                terms_review=TERMS
                | {"status": "not_retrievable", "redistribution": "permitted_with_attribution"},
            ),
            "sin condiciones verificadas",
        ),
    ],
)
def test_terms_review_supports_each_decision_and_the_license_flag(changes, problem):
    problems = decision_problems(source("vix", "downloaded_validated", **changes))
    assert len(problems) == 1 and problem in problems[0]


def test_a_source_blocked_before_its_terms_needs_no_reviewed_page():
    review = TERMS | dict(
        status="blocked_at_source_discovery",
        local_use="not_verified",
        redistribution="not_verified",
        evidence_urls=[],
    )
    blocked = source("stooq", "blocked_at_source_discovery", terms_review=review)
    assert decision_problems(blocked) == []
    unread = source("sec", "failed", terms_review=review | {"status": "not_retrievable"})
    assert decision_problems(unread) == ["revisión de condiciones incompleta"]


def test_committed_catalog_declares_traceable_decisions_for_every_source():
    catalog = json.loads(Path(CATALOG).read_text(encoding="utf-8"))
    assert all(decision_problems(item) == [] for item in catalog["sources"])
    decided = {d["decision"] for item in catalog["sources"] for d in item["use_decisions"]}
    assert decided <= DECISIONS
    # Cada fuente tiene su revisión de condiciones del 10 de octubre de 2026.
    assert {item["terms_review"]["reviewed_on"] for item in catalog["sources"]} == {"2026-10-10"}
    # Los fallos de SEC y GDELT y el bloqueo de Stooq siguen registrados como tales.
    status = {item["id"]: item["status"] for item in catalog["sources"]}
    assert status["sec_aapl_submissions"] == status["sec_aapl_companyfacts"] == "failed"
    assert status["gdelt_article_metadata"] == "failed"
    assert status["stooq_aapl_eod"] == "blocked_at_source_discovery"


def test_command_writes_the_report_and_fails_on_problems(tmp_path, capsys):
    store(tmp_path, [valid("vix", "data/vix.csv", BODY) | {"_body": BODY}])
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(dict(sources=[source("vix", "failed")])), encoding="utf-8")
    output = tmp_path / "report.json"
    code = main(["--root", str(tmp_path), "--catalog", str(catalog), "--output", str(output)])
    assert code == 1
    report = json.loads(output.read_text())
    assert report["catalog_sha256"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    assert report["captures"][0]["manifest"] == public_snapshots.INITIAL
    assert json.loads(capsys.readouterr().out)["problems"] == report["problems"]


def test_published_reconciliation_matches_the_committed_catalog():
    report = json.loads(Path("reports/data/public-source-validation.json").read_text())
    raw = Path(CATALOG).read_bytes()
    # Si cambia el catálogo, el informe debe regenerarse con la misma orden.
    assert report["catalog_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["problems"] == [] and report["benchmark_eligible"] is False
    assert len(report["sources"]) == len(json.loads(raw)["sources"])
