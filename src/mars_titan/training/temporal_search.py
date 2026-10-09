"""Ejecutar la búsqueda predefinida en todas las ventanas con una única carga CUDA."""

import argparse
import fcntl
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import (
    HISTORICAL_MASKED,
    INPUT_POLICIES,
    STRICT_INPUTS,
    masked_inputs,
)
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation.splits import PARTITIONS, build_folds, stopping_rule

from .checkpoints import StopRequest
from .cohort_contract import input_identity
from .experiment_resources import GpuLease
from .learning_hold import require_learning_allowed
from .reference_search import DEFAULT_MAX_RUNS, _configuration, run_search
from .selection import VALIDATION_PLATEAU
from .temporal_contract import temporal_contracts, temporal_fold

# Versión del informe de vistas según la política de entradas y si une los dos mercados.
_REPORT_VERSIONS = {
    (STRICT_INPUTS, False): 1,
    (STRICT_INPUTS, True): 2,
    (HISTORICAL_MASKED, False): 2,
    (HISTORICAL_MASKED, True): 3,
}


def _stopping(rule):
    """Comparar reglas de planes y protocolos con los valores implícitos de cada versión."""
    return dict(
        stopping=rule.get("stopping", VALIDATION_PLATEAU),
        minimum_epochs=rule.get("minimum_epochs", 0),
        **{key: rule[key] for key in ("max_epochs", "patience", "min_delta")},
    )


def _source_reports(report, views, markets, policy, joint):
    masked = masked_inputs(policy)
    if not joint:
        if len(markets) != 1:
            raise ValueError("La campaña conjunta necesita un informe temporal de versión 2")
        result = {next(iter(markets)): report}
    else:
        sources = report.get("sources")
        if markets != {"US", "CN"} or not isinstance(sources, dict) or set(sources) != markets:
            raise ValueError("La unión no identifica los informes de sus dos mercados")
        result = {}
        for market, record in sources.items():
            relative = f"markets/{market}/report.json"
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256"}
                or record["path"] != relative
            ):
                raise ValueError("El informe local queda fuera de la unión temporal")
            local, digest = read_manifest(views / relative, 1024**2)
            if (
                digest != record["sha256"]
                or local.get("schema_version") != (2 if masked else 1)
                or local.get("status") != "temporal_views_prepared"
                or local.get("final_test_opened") is not False
            ):
                raise ValueError("Un informe local no conserva su identidad temporal")
            input_identity(local, input_policy=policy)
            result[market] = local
    # La política histórica conserva el macro del padre y no tiene panel ni admisión propios.
    keys = ("parent_sha256", "protocol_sha256")
    keys += () if masked else ("macro_sha256", "admission_sha256")
    for local in result.values():
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(local.get(key))) for key in keys):
            raise ValueError("Faltan huellas de procedencia de la preparación")
    return result


def _inputs(config, views):
    plan, cases, config_hash = _configuration(config)
    if (
        plan["schema_version"] not in {2, 3, 4}
        or len(plan["arms"]) != 1
        or plan["arms"][0] not in {"US", "CN", "US+CN"}
        or plan["pooled_weightings"] != ["natural"]
    ):
        raise ValueError("La campaña temporal requiere un único brazo y peso natural")
    markets = {"US", "CN"} if plan["arms"] == ["US+CN"] else set(plan["arms"])
    # La versión 4 del plan declara la lectura. Las anteriores conservan la política estricta.
    policy = plan.get("input_policy", STRICT_INPUTS)
    report, report_hash = read_manifest(views / "report.json", 1024**2)
    joint = isinstance(report, dict) and report.get("kind") == "joint_temporal_views"
    if isinstance(report, dict) and report.get("input_policy", STRICT_INPUTS) != policy:
        raise ValueError("La política de entradas del plan no coincide con la de las vistas")
    if (
        not isinstance(report, dict)
        or policy not in INPUT_POLICIES
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != _REPORT_VERSIONS[(policy, joint)]
        or report.get("status") != "temporal_views_prepared"
        or report.get("final_test_opened") is not False
        or not isinstance(report.get("folds"), list)
        or not 1 <= len(report["folds"]) <= 128
    ):
        raise ValueError("Las vistas no tienen un informe de preparación admisible")
    input_identity(report, input_policy=policy)
    if not re.fullmatch(r"[0-9a-f]{64}", str(report.get("parent_sha256"))):
        raise ValueError("Faltan huellas de procedencia de la preparación")
    evidence = _source_reports(report, views, markets, policy, joint)
    if any(len(local.get("folds", [])) != len(report["folds"]) for local in evidence.values()):
        raise ValueError("Los informes locales no contienen todas las ventanas conjuntas")
    records, protocols = [], None
    for index, row in enumerate(report["folds"]):
        name = f"fold-{index:03}"
        if row.get("id") != name or row.get("has_all_partitions") is not True:
            raise ValueError("Las ventanas deben ser consecutivas y tener todas sus particiones")
        path = views / name / "manifest.json"
        meta, digest = read_manifest(path, 8 * 1024**2)
        if not isinstance(meta, dict):
            raise ValueError("El manifiesto de la ventana no es un objeto")
        contracts = temporal_contracts(meta, input_policy=policy)
        current = {market: contract["protocol"] for market, contract in contracts.items()}
        if protocols is None:
            protocols = current
        assets = meta.get("assets")
        if (
            set(contracts) != markets
            or current != protocols
            or meta.get("markets", sorted(markets))
            not in (sorted(markets), sorted(markets, reverse=True))
            or not isinstance(assets, list)
            or not assets
            or any(
                not isinstance(asset, dict) or asset.get("market") not in markets
                for asset in assets
            )
            or meta.get("scope") != plan["scope"]
            or (plan["scope"] == "full_corpus" and meta.get("cohort_complete") is not True)
        ):
            raise ValueError(
                "La ventana no conserva el mercado, la población y el alcance del diseño"
            )
        if (
            digest != row.get("manifest_sha256")
            or set(meta.get("counts", {})) != set(PARTITIONS)
            or any(type(v) is not int or v <= 0 for v in meta["counts"].values())
            or meta["counts"] != row.get("counts")
            or meta.get("final_test_opened") is not False
        ):
            raise ValueError("Una ventana no conserva su identidad y población declaradas")
        for market, contract in contracts.items():
            local = evidence[market]
            if any(
                contract.get(key) != local.get(key)
                for key in ("macro_sha256", "parent_sha256", "admission_sha256")
            ):
                raise ValueError("Una ventana no conserva las fuentes de su mercado")
            if joint:
                local_record = local["folds"][index]
                local_path = views / "markets" / market / name / "manifest.json"
                local_view, signature = read_manifest(local_path, 8 * 1024**2)
                if (
                    local_record.get("id") != name
                    or signature != local_record.get("manifest_sha256")
                    or local_view.get("temporal_view") != contract
                    or local_view.get("counts") != local_record.get("counts")
                    or local_view.get("assets")
                    != [asset for asset in assets if asset["market"] == market]
                    or row.get("market_counts", {}).get(market) != local_view.get("counts")
                ):
                    raise ValueError("La unión no conserva exactamente su ventana local")
        if joint and meta["counts"] != {
            part: sum(local["folds"][index]["counts"][part] for local in evidence.values())
            for part in PARTITIONS
        }:
            raise ValueError("Los recuentos conjuntos no suman las poblaciones locales")
        records.append(
            dict(
                id=name,
                manifest=path,
                manifest_sha256=digest,
                fold=temporal_fold(meta, input_policy=policy),
                counts=meta["counts"],
            )
        )
    if any(
        [r["fold"] for r in records] != build_folds(protocol) for protocol in protocols.values()
    ):
        raise ValueError("La campaña no contiene todas las ventanas del protocolo")
    protocol = next(iter(protocols.values()))
    # Todas las familias deben seleccionar y detenerse con la regla registrada en el protocolo.
    if protocol["schema_version"] == 2 and _stopping(plan) != _stopping(stopping_rule(protocol)):
        raise ValueError("El plan no aplica la regla de parada declarada por el protocolo")
    per_fold = len(cases) + len(plan["models"]) * (
        len(plan["finalist_seeds"]) - 1 + 2 * len(plan["finalist_seeds"])
    )
    limit = plan.get("max_runs", DEFAULT_MAX_RUNS)
    if per_fold * len(records) > limit:
        raise ValueError(
            f"La campaña temporal prevé {per_fold * len(records)} ejecuciones "
            f"({per_fold} por ventana en {len(records)} ventanas) y supera el límite "
            f"declarado de {limit}. Declara max_runs en el plan antes de ejecutar"
        )
    identity = dict(
        config_sha256=config_hash,
        views_report_sha256=report_hash,
        manifests={r["id"]: r["manifest_sha256"] for r in records},
        code_sha256=sha256(Path(__file__)),
    )
    identity.update(
        {"protocol_sha256": report["protocol_sha256"]}
        if not joint
        else {
            "protocols_sha256": {
                market: local["protocol_sha256"] for market, local in evidence.items()
            }
        }
    )
    if masked_inputs(policy):
        identity["input_policy"] = policy
    return records, identity, per_fold


def check_temporal_search(config, views):
    """Validar ventanas, población y presupuesto sin reservar la GPU ni entrenar."""
    records, identity, per_fold = _inputs(Path(config), Path(views))
    return dict(
        status="checked",
        identity=identity,
        runs_per_fold=per_fold,
        planned_runs=per_fold * len(records),
        folds=[dict(id=r["id"], fold=r["fold"], counts=r["counts"]) for r in records],
        scientific_training_started=False,
        final_test_opened=False,
    )


def run_temporal_search(config, views, output, *, resume=False):
    require_learning_allowed("la búsqueda temporal de referencias")
    config, views, output = map(Path, (config, views, output))
    records, identity, per_fold = _inputs(config, views)
    safe_destination(output)
    outside_source(views, output)
    outside_source(Path("dataset"), output)
    if (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa un destino nuevo o reanuda una campaña existente")
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    accepted_summary = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "summary.json"
        if resume and path.exists():
            summary, _ = read_manifest(path, 1024**2)
            if summary.get("identity") != identity or summary.get("planned_runs") != per_fold * len(
                records
            ):
                raise ValueError("La identidad de la campaña temporal ha cambiado")
        else:
            if any(
                p.name != ".lock"
                and not (p.name.startswith(".summary.json.") and p.is_file() and not p.is_symlink())
                for p in output.iterdir()
            ):
                raise ValueError("No se puede recuperar un inicio con artefactos desconocidos")
            summary = dict(
                schema_version=1,
                kind="temporal_reference_search",
                identity=identity,
                status="running",
                started_at_utc=datetime.now(UTC).isoformat(),
                final_test_opened=False,
                planned_runs=per_fold * len(records),
                completed_runs=0,
                folds=[
                    dict(id=r["id"], status="pending", completed_runs=0, planned_runs=per_fold)
                    for r in records
                ],
            )
        expected_ids = [r["id"] for r in records]
        if (
            [f.get("id") for f in summary.get("folds", [])] != expected_ids
            or summary.get("final_test_opened") is not False
            or summary.get("schema_version") != 1
            or summary.get("kind") != "temporal_reference_search"
            or summary.get("status") not in {"running", "paused", "failed", "completed"}
            or any(
                row.get("planned_runs") != per_fold
                or type(row.get("completed_runs")) is not int
                or not 0 <= row["completed_runs"] <= per_fold
                or row.get("status") not in {"pending", "running", "paused", "failed", "completed"}
                or row["status"] == "completed"
                and row["completed_runs"] != per_fold
                for row in summary["folds"]
            )
            or summary.get("completed_runs") != sum(r["completed_runs"] for r in summary["folds"])
        ):
            raise ValueError("El resumen no conserva las ventanas y la reserva final")
        accepted_summary = True

        def save():
            summary["completed_runs"] = sum(f["completed_runs"] for f in summary["folds"])
            summary["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(path, summary)

        summary["status"] = "running"
        save()
        with GpuLease() as resources, StopRequest() as stop:
            summary["resources"] = resources.record
            for record, progress in zip(records, summary["folds"], strict=True):
                if stop.requested:
                    summary["status"] = "paused"
                    save()
                    return summary
                if (
                    sha256(config) != identity["config_sha256"]
                    or sha256(views / "report.json") != identity["views_report_sha256"]
                ):
                    raise ValueError("La configuración o la preparación han cambiado")

                def observe(child, entry=progress):
                    if (
                        child["planned_runs"] != per_fold
                        or not 0 <= child["completed_runs"] <= per_fold
                    ):
                        raise ValueError("El progreso de la ventana no concilia con su presupuesto")
                    entry.update(status=child["status"], completed_runs=child["completed_runs"])
                    save()

                folder = output / record["id"]
                result = run_search(
                    config, record["manifest"], folder, resume=folder.exists(), progress=observe
                )
                resources.check()
                observe(result)
                if result["status"] != "completed":
                    summary["status"] = result["status"]
                    save()
                    return summary
            if summary["completed_runs"] != summary["planned_runs"]:
                raise ValueError("La campaña no completó todas las ejecuciones declaradas")
            summary["status"] = "completed"
            save()
        return summary
    except BaseException as error:
        if accepted_summary:
            summary.update(status="failed", error_type=type(error).__name__, error=str(error))
            atomic_json(output / "summary.json", summary)
        raise
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "views"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Validar y presupuestar sin reservar la GPU"
    )
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_temporal_search(args.config, args.views), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("La ejecución necesita --output")
    report = run_temporal_search(args.config, args.views, args.output, resume=args.resume)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
