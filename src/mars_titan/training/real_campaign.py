"""Encadenar referencias reales, continuaciones y fiabilidad bajo el supervisor existente."""

import argparse
import fcntl
import os
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.posttraining.completion import _receipt, run_child, stage_plan
from mars_titan.posttraining.preparation import encoder_contract
from mars_titan.posttraining.queue import read_design

from .checkpoints import StopRequest
from .learning_hold import require_learning_allowed
from .reference_search import _configuration as neural_configuration
from .run_receipts import initialize_receipt
from .tabular_search import _configuration as tabular_configuration
from .temporal_contract import temporal_contracts
from .temporal_search import _inputs as temporal_inputs

_FAMILIES = ("rnn", "lstm", "gru", "dlinear", "ridge", "xgboost")
_PATHS = (
    "views",
    "encoded",
    "neural_config",
    "tabular_config",
    "post_config",
    "references",
    "completion",
    "state_dir",
)
_STATUSES = {"pending", "running", "paused", "blocked", "failed", "completed"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _code():
    root = Path(__file__).parents[1]
    files = sorted(root.rglob("*.py"))
    _require(0 < len(files) <= 1024, "El código excede el presupuesto de identidad")
    return {path.relative_to(root).as_posix(): sha256(path) for path in files}


def prepare_campaign(args):
    """Fijar identidad y recuentos antes de ejecutar procesos o adquirir CUDA."""
    for name in _PATHS:
        safe_destination(getattr(args, name))
    outputs = (args.references, args.completion, args.state_dir)
    for first, second in combinations(outputs, 2):
        outside_source(first, second)
        outside_source(second, first)
    for source in (
        args.views,
        args.encoded.parent,
        args.neural_config,
        args.tabular_config,
        args.post_config,
    ):
        for output in outputs:
            outside_source(source, output)
            outside_source(output, source)
    records, neural_identity, reference_runs = temporal_inputs(args.neural_config, args.views)
    neural, _, _ = neural_configuration(args.neural_config)
    tabular, tabular_cases, tabular_hash = tabular_configuration(args.tabular_config)
    post, post_cases, post_hash = read_design(args.post_config)
    encoded, encoded_hash = read_manifest(args.encoded, 8 * 1024**2)
    first, _ = read_manifest(records[0]["manifest"], 8 * 1024**2)
    _require(
        all(
            contract["protocol"]["final_test_start"] == "2024-01-01"
            for contract in temporal_contracts(first).values()
        )
        and all(row["fold"]["evaluation"][1] <= "2024-01-01" for row in records)
        and encoded.get("final_test_opened") is False
        and neural["scope"] == "full_corpus"
        and set(neural["models"]) == set(_FAMILIES[:4])
        and post["schema_version"] == 2
        and post["parent_seed_policy"] == "matching"
        and post["conditions"] == ["real"],
        "La campaña requiere datos reales, seis referencias, padres emparejados y test cerrado",
    )
    _require(
        set(neural["finalist_seeds"]) == set(tabular["finalist_seeds"]) == set(post["seeds"]),
        "Las semillas de las referencias y continuaciones no coinciden",
    )
    admissions = {}
    for record in records:
        binding = encoder_contract(record["manifest"], args.encoded)
        _require(
            binding["supervision_sha256"] == record["manifest_sha256"]
            and binding["encoded_sha256"] == encoded_hash,
            "La representación cambió durante la admisión",
        )
        metadata, metadata_hash = read_manifest(record["manifest"], 8 * 1024**2)
        _require(metadata_hash == record["manifest_sha256"], "La vista cambió durante la admisión")
        for contract in temporal_contracts(metadata).values():
            path = Path(contract["admission_path"])
            safe_destination(path)
            admission, signature = read_manifest(path, 8 * 1024**2)
            indicators = admission.get("required_indicator_ids")
            _require(
                signature == contract["admission_sha256"]
                and isinstance(indicators, list)
                and len(indicators) == 140
                and all(isinstance(name, str) and name for name in indicators)
                and len(set(indicators)) == 140,
                "La admisión debe conservar los 140 indicadores y su huella",
            )
            admissions[str(path.resolve())] = signature
    stages = stage_plan(
        [row["id"] for row in records],
        reference_runs=reference_runs,
        tabular_runs=len(tabular_cases) + len(tabular["finalist_seeds"]) - 1,
        adjustment_runs=sum(len(post_cases(family)) for family in _FAMILIES),
    )
    models = sum(row["planned_runs"] for row in stages if row["stage"] == "evaluation")
    _require(models * 2 <= 8192, "La campaña supera el presupuesto del análisis final")
    identity = dict(
        neural=neural_identity,
        encoded_sha256=encoded_hash,
        tabular_config_sha256=tabular_hash,
        post_config_sha256=post_hash,
        admissions=admissions,
        paths={name: str(getattr(args, name).resolve()) for name in _PATHS},
        code=_code(),
        resources={"MARS_TITAN_INPUT_CACHE_MIB": os.environ.get("MARS_TITAN_INPUT_CACHE_MIB")},
        final_test_opened=False,
    )
    return identity, [
        dict(name="neural", planned=reference_runs * len(records), unit="run"),
        dict(name="completion", planned=sum(row["planned_runs"] for row in stages), unit="run"),
        dict(name="reliability", planned=models, unit="model"),
    ]


def _stage_receipt(name, path, expected, args, identity):
    safe_destination(path)
    if name == "neural":
        report = _receipt(path, expected)
        _require(
            report.get("kind") == "temporal_reference_search"
            and report.get("identity") == identity["neural"],
            "La etapa neuronal pertenece a otra identidad",
        )
        return report
    if name == "completion":
        report = _receipt(path, expected)
        inherited = report.get("identity", {})
        _require(
            report.get("kind") == "temporal_posttraining_completion"
            and inherited.get("reference_sha256") == sha256(args.references / "summary.json")
            and inherited.get("dependency") is None
            and all(
                inherited.get(key) == identity[key]
                for key in ("tabular_config_sha256", "post_config_sha256", "encoded_sha256")
            ),
            "La continuación pertenece a otra identidad",
        )
        code = inherited.get("code")
    else:
        report, _ = read_manifest(path, 64 * 1024**2)
        provenance = report.get("provenance", {})
        _require(
            report.get("kind") == "frozen_campaign_reliability"
            and report.get("status") == "completed"
            and report.get("final_test_opened") is False
            and report.get("target_kind") == "residual_return"
            and report.get("coverage_guaranteed") is False
            and report.get("counts", {}).get("models") == expected
            and report["counts"].get("prediction_files") == 2 * expected
            and provenance.get("reference_sha256") == sha256(args.references / "summary.json")
            and provenance.get("completion_sha256") == sha256(args.completion / "summary.json"),
            "La fiabilidad no corresponde a la campaña cerrada",
        )
        artifact = path.parent / "cases.csv"
        safe_destination(artifact)
        _require(
            artifact.stat().st_size <= 128 * 1024**2
            and report.get("artifacts") == {"cases.csv": sha256(artifact)},
            "El artefacto de fiabilidad ha cambiado",
        )
        report = report | dict(planned_runs=expected, completed_runs=expected)
        code = report.get("analysis_source_sha256")
    _require(
        isinstance(code, dict)
        and code
        and all(identity["code"].get(key) == value for key, value in code.items()),
        "La etapa cambió la identidad de su código",
    )
    return report


def _command(name, args):
    if name == "neural":
        module = "mars_titan.training.temporal_search"
        options = dict(config=args.neural_config, views=args.views, output=args.references)
    elif name == "completion":
        module = "mars_titan.posttraining.completion"
        options = dict(
            reference=args.references,
            encoded=args.encoded,
            tabular_config=args.tabular_config,
            post_config=args.post_config,
            output=args.completion,
        )
    else:
        module = "mars_titan.evaluation.campaign_reliability"
        options = dict(
            reference=args.references,
            completion=args.completion,
            output=args.state_dir / "reliability",
        )
    command = [sys.executable, "-m", module]
    for name, value in options.items():
        command.extend(("--" + name.replace("_", "-"), str(value)))
    if module.endswith("temporal_search") and args.references.exists():
        command.append("--resume")
    return command


def run_campaign(args, stop):
    """Reanudar etapas confirmadas sin duplicar entrenamiento ni asumir una parada completa."""
    require_learning_allowed("la campaña real encadenada")
    identity, specifications = prepare_campaign(args)
    receipts = dict(
        neural=args.references / "summary.json",
        completion=args.completion / "summary.json",
        reliability=args.state_dir / "reliability/reliability.json",
    )
    total = sum(row["planned"] for row in specifications if row["unit"] == "run")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.state_dir / ".campaign.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(
            args.state_dir, identity, record="summary.json", lock=".campaign.lock"
        )
        if not confirmed:
            _require(
                not args.references.exists() and not args.completion.exists(),
                "Una campaña nueva no puede adoptar salidas ajenas",
            )
        state = (
            _receipt(args.state_dir / "summary.json", total)
            if confirmed
            else dict(
                schema_version=1,
                kind="real_retraining_campaign",
                domain="real",
                identity=identity,
                status="pending",
                planned_runs=total,
                completed_runs=0,
                planned_reliability_models=specifications[-1]["planned"],
                completed_reliability_models=0,
                stages=[dict(row, status="pending", completed=0) for row in specifications],
                final_test_opened=False,
            )
        )
        _require(
            state.get("identity") == identity
            and len(state.get("stages", [])) == len(specifications)
            and state.get("schema_version") == 1
            and state.get("kind") == "real_retraining_campaign"
            and state.get("domain") == "real"
            and state.get("planned_reliability_models") == specifications[-1]["planned"],
            "El estado pertenece a otra identidad",
        )
        for specification, progress in zip(specifications, state["stages"], strict=True):
            _require(
                all(progress.get(key) == value for key, value in specification.items())
                and progress.get("status") in _STATUSES
                and type(progress.get("completed")) is int
                and 0 <= progress["completed"] <= progress["planned"],
                "El estado no conserva las etapas previstas",
            )
            if progress["status"] == "completed":
                path = receipts[progress["name"]]
                report = _stage_receipt(progress["name"], path, progress["planned"], args, identity)
                _require(
                    report["status"] == "completed"
                    and progress["completed"] == progress["planned"]
                    and sha256(path) == progress.get("sha256"),
                    "Ha cambiado una etapa confirmada",
                )
        _require(
            state["completed_runs"]
            == sum(row["completed"] for row in state["stages"] if row["unit"] == "run")
            and state["completed_reliability_models"] == state["stages"][-1]["completed"],
            "El progreso de la campaña no concilia",
        )
        if state["status"] == "completed":
            _require(
                all(row["status"] == "completed" for row in state["stages"]),
                "La campaña conserva etapas pendientes",
            )
            return state

        def save():
            state["completed_runs"] = sum(
                row["completed"] for row in state["stages"] if row["unit"] == "run"
            )
            state["completed_reliability_models"] = state["stages"][-1]["completed"]
            state["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(args.state_dir / "summary.json", state)

        state.update(status="running")
        state.pop("error", None)
        save()
        try:
            for progress in state["stages"]:
                if stop.requested:
                    raise InterruptedError("Campaña pausada antes de la siguiente etapa")
                _require(
                    prepare_campaign(args)[0] == identity, "Las entradas o el código han cambiado"
                )
                if progress["status"] == "completed":
                    continue
                name, expected = progress["name"], progress["planned"]
                path = receipts[name]

                def reader(receipt, count, stage=name):
                    return _stage_receipt(stage, receipt, count, args, identity)

                if path.is_file() and reader(path, expected)["status"] == "completed":
                    progress.update(status="completed", completed=expected, sha256=sha256(path))
                    save()
                    continue
                progress["status"] = "running"
                save()

                def observe(result, entry=progress):
                    entry.update(
                        completed=result["completed_runs"],
                        status="running" if result["status"] == "completed" else result["status"],
                    )
                    save()

                run_child(
                    _command(name, args), path, expected, stop, observe, receipt_reader=reader
                )
                _require(
                    prepare_campaign(args)[0] == identity, "Las entradas o el código han cambiado"
                )
                result = reader(path, expected)
                if result["status"] != "completed":
                    raise InterruptedError("La etapa conserva trabajo pendiente")
                progress.update(status="completed", completed=expected, sha256=sha256(path))
                save()
            state["status"] = "completed"
        except InterruptedError:
            state["status"] = "paused"
        except BaseException as error:
            state.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
            raise
        finally:
            save()
        return state
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("views", "encoded", "neural-config", "tabular-config", "post-config", "state-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--references-output", dest="references", type=Path, required=True)
    parser.add_argument("--completion-output", dest="completion", type=Path, required=True)
    args = parser.parse_args(argv)
    with StopRequest() as stop:
        result = run_campaign(args, stop)
    print(
        f"Estado: {result['status']}. "
        f"Ejecuciones: {result['completed_runs']}/{result['planned_runs']}. "
        f"Modelos analizados: {result['completed_reliability_models']}."
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
