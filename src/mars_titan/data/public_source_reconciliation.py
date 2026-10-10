"""Reconciliación del catálogo de fuentes públicas con sus capturas confirmadas.

El estado que declara el catálogo debe coincidir con la evidencia de los manifiestos: una
fuente `downloaded_validated` necesita un registro válido cuyo archivo conserve su huella,
una fuente `failed` no puede tener ninguno y un bloqueo en el descubrimiento sigue siendo
un bloqueo. Cada fuente declara además sus decisiones por uso, con la carencia que cubriría
y la evidencia, y ninguna es elegible para el benchmark. La reconciliación no descarga nada
ni cambia la decisión: solo comprueba que catálogo y capturas cuentan lo mismo.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from mars_titan.data import public_snapshots
from mars_titan.data.storage import atomic_json

CATALOG = "data/catalogs/public-sources.json"
# Evidencia admisible para cada estado del catálogo.
EXPECTED = {
    "downloaded_validated": frozenset({"acquired_validated", "fixed_document"}),
    "failed": frozenset({"failed_attempt"}),
    "blocked_at_source_discovery": frozenset({"blocked_at_source_discovery"}),
}
DECISIONS = frozenset({"admitted_exploratory", "pending", "excluded"})
DECISION_TEXT = ("use", "gap", "evidence")


def evidence(source, attempts, contents):
    """Clase de evidencia de una fuente según sus intentos registrados."""
    valid = [record for record in attempts if record["status"] in public_snapshots.VALID]
    if any(contents.get(record["sha256"], {}).get("available") for record in valid):
        fixed = source.get("validator") == "financial_pdf"
        return "fixed_document" if fixed else "acquired_validated"
    if valid:
        # Hay manifiesto válido pero no los bytes, como en un clon sin `data/external`.
        return "content_missing"
    if any(record["status"] == "blocked_at_source_discovery" for record in attempts):
        return "blocked_at_source_discovery"
    return "failed_attempt" if attempts else "not_attempted"


def decision_problems(source):
    """Decisiones por uso ausentes o incompletas y elegibilidad indebida."""
    problems = []
    decisions = source.get("use_decisions")
    if not isinstance(decisions, list) or not decisions:
        problems.append("sin decisiones por uso")
    for item in decisions if isinstance(decisions, list) else []:
        complete = isinstance(item, dict) and all(
            isinstance(item.get(key), str) and item[key].strip() for key in DECISION_TEXT
        )
        if not complete or item.get("decision") not in DECISIONS:
            problems.append("decisión por uso incompleta o con un valor no admitido")
    if source.get("benchmark_eligible") is not False:
        problems.append("benchmark_eligible debe ser false")
    return problems


def reconcile(catalog, index, *, catalog_sha256=None):
    """Tabla por fuente con su evidencia y la lista de incoherencias encontradas."""
    attempts, diagnostics = defaultdict(list), defaultdict(set)
    for capture in index["captures"]:
        for record in capture["records"]:
            attempts[record["source_id"]].append(record | {"run_id": capture["run_id"]})
        for item in capture.get("diagnostics", []):
            if item["http_status"] is not None:
                diagnostics[item["source_id"]].add(item["http_status"])
    rows, problems = [], []
    for source in catalog["sources"]:
        found = attempts.get(source["id"], [])
        valid = [record for record in found if record["status"] in public_snapshots.VALID]
        failed = [record for record in found if record["status"] not in public_snapshots.VALID]
        kind = evidence(source, found, index["contents"])
        issues = decision_problems(source)
        if kind not in EXPECTED.get(source.get("status"), ()):
            issues.append(f"estado {source.get('status')} sin evidencia coherente ({kind})")
        rows.append(
            dict(
                source_id=source["id"],
                provider=source.get("provider"),
                catalog_status=source.get("status"),
                evidence=kind,
                attempts=len(found),
                valid_records=len(valid),
                distinct_contents=len({record["sha256"] for record in valid}),
                first_valid_run_id=valid[0]["run_id"] if valid else None,
                latest_valid_run_id=valid[-1]["run_id"] if valid else None,
                failure_http_statuses=sorted(
                    {r["http_status"] for r in failed if r["http_status"] is not None}
                ),
                diagnostic_http_statuses=sorted(diagnostics[source["id"]]),
                license_unknown=source.get("license_unknown"),
                decisions=[
                    dict(use=item.get("use"), decision=item.get("decision"))
                    for item in source.get("use_decisions") or []
                    if isinstance(item, dict)
                ],
                problems=issues,
            )
        )
        problems.extend(f"{source['id']}: {issue}" for issue in issues)
    undeclared = sorted(set(attempts) - {source["id"] for source in catalog["sources"]})
    problems.extend(f"{name}: captura sin fuente en el catálogo" for name in undeclared)
    return dict(
        schema_version=1,
        project="MARS-TITAN",
        kind="public_source_reconciliation",
        catalog_sha256=catalog_sha256,
        captures=[
            dict(run_id=c["run_id"], manifest=c["manifest"], manifest_sha256=c["manifest_sha256"])
            for c in index["captures"]
        ],
        interrupted_runs=index["interrupted_runs"],
        sources=rows,
        problems=problems,
        benchmark_eligible=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reconciliar catálogo público y capturas.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Raíz del almacén")
    parser.add_argument("--catalog", type=Path, help="Catálogo, por defecto el de la raíz")
    parser.add_argument("--output", type=Path, help="Informe JSON que se escribe")
    args = parser.parse_args(argv)
    raw = (args.catalog or args.root / CATALOG).read_bytes()
    index = public_snapshots.build_index(args.root, args.root / "data/external")
    report = reconcile(json.loads(raw), index, catalog_sha256=hashlib.sha256(raw).hexdigest())
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(dict(sources=len(report["sources"]), problems=report["problems"])))
    return int(bool(report["problems"]))


if __name__ == "__main__":
    raise SystemExit(main())
