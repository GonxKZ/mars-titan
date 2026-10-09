"""Completar referencias, ajustes y evaluación temporal mediante procesos recuperables."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.run_receipts import initialize_receipt


def _receipt(path, expected=None):
    if not path.is_file():
        raise ValueError("Falta el resultado confirmado del proceso")
    report, _ = read_manifest(path, 8 * 1024**2)
    if (
        not isinstance(report, dict)
        or not isinstance(report.get("status"), str)
        or report["status"]
        not in {
            "running",
            "pending",
            "queued",
            "waiting",
            "paused",
            "blocked",
            "failed",
            "completed",
        }
    ):
        raise ValueError("El resultado contiene un estado desconocido")
    count, planned = report.get("completed_runs"), report.get("planned_runs")
    if (
        type(count) is not int
        or type(planned) is not int
        or not 0 <= count <= planned
        or planned < 1
        or report.get("final_test_opened") is not False
        or expected is not None
        and planned != expected
        or report.get("status") == "completed"
        and count != planned
    ):
        raise ValueError("El recuento o la reserva del resultado no concilian")
    return report


def wait_dependency(path, stop, observe, *, poll_seconds=15):
    """Esperar sin reservar CUDA ni interpretar una pausa como una finalización."""
    while not stop.requested:
        report = _receipt(path)
        status = report.get("status")
        if status == "completed":
            return dict(path=str(path.resolve()), sha256=sha256(path))
        if status not in {"running", "pending", "queued"}:
            raise RuntimeError(f"La campaña previa necesita revisión: {status}")
        observe("waiting_dependency")
        if not stop.requested:
            time.sleep(poll_seconds)
    raise InterruptedError("Espera interrumpida antes de adquirir CUDA")


def stage_plan(folds, *, tabular_runs, adjustment_runs, reference_runs=40):
    stages = [
        dict(fold=fold, stage=stage, planned_runs=count)
        for fold in folds
        for stage, count in (("tabular", tabular_runs), ("posttraining", adjustment_runs))
    ]
    stages.extend(
        dict(
            fold=fold,
            stage="evaluation",
            planned_runs=reference_runs + tabular_runs + adjustment_runs,
        )
        for fold in folds
    )
    return stages


def run_child(
    command,
    receipt,
    expected,
    stop,
    observe,
    *,
    poll_seconds=5,
    grace_seconds=620,
    receipt_reader=None,
):
    """Propagar la parada y exigir un recibo completo antes de abrir la etapa siguiente."""
    if stop.requested:
        raise InterruptedError("Campaña pausada antes de iniciar otra etapa")
    reader = _receipt if receipt_reader is None else receipt_reader
    if not callable(reader):
        raise ValueError("El lector de recibos debe ser invocable")
    process = subprocess.Popen(command)
    deadline = None
    try:
        while True:
            if stop.requested and deadline is None:
                process.terminate()
                deadline = time.monotonic() + grace_seconds
            if deadline is not None and time.monotonic() >= deadline:
                process.kill()
            try:
                timeout = (
                    poll_seconds
                    if deadline is None
                    else min(poll_seconds, max(0.001, deadline - time.monotonic()))
                )
                code = process.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                if receipt.exists():
                    try:
                        result = reader(receipt, expected)
                    except BlockingIOError:
                        continue
                    observe(result)
        if code != 0:
            if code == 2 and receipt.exists():
                result = reader(receipt, expected)
                if result["status"] in {"blocked", "failed", "completed"}:
                    raise RuntimeError("El código de pausa no corresponde al estado del recibo")
                observe(result)
            if code == 2 or stop.requested and code < 0:
                raise InterruptedError("La etapa se ha detenido sin declararse completa")
            raise RuntimeError(f"La etapa terminó con código {code}")
        result = reader(receipt, expected)
        observe(result)
        if result["status"] in {"blocked", "failed"}:
            raise RuntimeError("La etapa conserva un fallo confirmado en su recibo")
        if result["status"] != "completed":
            raise InterruptedError("La etapa conserva trabajo pendiente")
        return result
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _inputs(args):
    from mars_titan.posttraining.preparation import encoder_contract
    from mars_titan.posttraining.queue import _queue_code, read_design
    from mars_titan.training.baseline_queue import reference_market, reference_view
    from mars_titan.training.partition_contract import supervision_bounds
    from mars_titan.training.tabular_search import _code, _configuration
    from mars_titan.training.temporal_contract import temporal_contracts

    root, digest = read_manifest(args.reference / "summary.json", 8 * 1024**2)
    if root.get("kind") != "temporal_reference_search" or root.get("status") != "completed":
        raise ValueError("Falta una campaña temporal neuronal completa")
    _receipt(args.reference / "summary.json")
    folds = root["folds"]
    if not 1 <= len(folds) <= 128 or [r["id"] for r in folds] != [
        f"fold-{i:03}" for i in range(len(folds))
    ]:
        raise ValueError("Las ventanas no forman una secuencia completa")
    if sum(row["completed_runs"] for row in folds) != root["completed_runs"]:
        raise ValueError("El total neuronal no concilia las ventanas")
    references, market = {}, None
    for row in folds:
        if row["status"] != "completed" or row["completed_runs"] != row["planned_runs"]:
            raise ValueError("Hay una ventana neuronal pendiente")
        reference = args.reference / row["id"] / "summary.json"
        arm = reference_market(reference)
        if market is not None and arm != market:
            raise ValueError("Las ventanas deben conservar el mismo mercado")
        market = arm
        proof = reference_view(reference, arm)
        source, _ = read_manifest(Path(proof["manifest"]), 8 * 1024**2)
        contracts = temporal_contracts(source)
        expected_markets = {"US", "CN"} if arm == "US+CN" else {arm}
        if set(contracts) != expected_markets:
            raise ValueError("El mercado de la referencia no coincide con su protocolo")
        for source_market, contract in contracts.items():
            if set(supervision_bounds(source, market=source_market)) != {
                "train",
                "validation",
                "calibration",
                "evaluation",
            }:
                raise ValueError("Faltan las cuatro particiones temporales")
            admission, _ = read_manifest(Path(contract["admission_path"]), 8 * 1024**2)
            if len(admission["required_indicator_ids"]) != 140:
                raise ValueError("La nueva campaña requiere los 140 indicadores")
        encoder_contract(Path(proof["manifest"]), args.encoded)
        references[row["id"]] = proof
    if len({row["planned_runs"] for row in folds}) != 1:
        raise ValueError("Las ventanas deben compartir el presupuesto neuronal")
    tabular, cases, tabular_hash = _configuration(args.tabular_config)
    post, post_cases, post_hash = read_design(args.post_config)
    if post["conditions"] != ["real"]:
        raise ValueError("Esta continuación estricta solo admite el diseño real")
    stages = stage_plan(
        list(references),
        tabular_runs=len(cases) + len(tabular["finalist_seeds"]) - 1,
        adjustment_runs=sum(
            len(post_cases(kind)) for kind in ("rnn", "lstm", "gru", "dlinear", "ridge", "xgboost")
        ),
        reference_runs=folds[0]["planned_runs"],
    )
    identity = dict(
        market=market,
        reference_sha256=digest,
        references=references,
        tabular_config_sha256=tabular_hash,
        post_config_sha256=post_hash,
        encoded_sha256=sha256(args.encoded),
        code=_queue_code()
        | _code()
        | {
            "training/baseline_queue.py": sha256(
                Path(__file__).parents[1] / "training/baseline_queue.py"
            ),
            "posttraining/completion.py": sha256(Path(__file__)),
            "posttraining/heldout.py": sha256(Path(__file__).with_name("heldout.py")),
            "posttraining/analysis.py": sha256(Path(__file__).with_name("analysis.py")),
        },
    )
    return identity, stages


def _stage(args):
    if args.stage in {"tabular", "posttraining"}:
        require_learning_allowed(f"la etapa {args.stage}")
    from mars_titan.training.baseline_queue import reference_market

    folder = args.output / args.fold
    reference = args.reference / args.fold / "summary.json"
    arm = reference_market(reference)
    with StopRequest() as stop:
        if args.stage == "tabular":
            import torch

            from mars_titan.training.baseline_queue import run_queue
            from mars_titan.training.experiment_resources import GpuLease

            with GpuLease() as lease:
                torch.set_num_threads(4)
                result = run_queue(
                    args.tabular_config, reference, folder / "tabular", arm=arm, stop=stop
                )
                lease.check()
        elif args.stage == "posttraining":
            from mars_titan.posttraining.queue import run_queue

            result = run_queue(
                args.post_config,
                reference,
                folder / "tabular/summary.json",
                args.encoded,
                folder / "posttraining",
                arm=arm,
                stop=stop,
            )
        else:
            from mars_titan.posttraining.heldout import run_evaluation

            result = run_evaluation(
                reference,
                folder / "tabular/summary.json",
                folder / "posttraining/summary.json",
                folder / "evaluation",
                arm=arm,
                stop=stop,
            )
    print(
        f"{args.fold}/{args.stage}: {result['status']}, "
        f"{result['completed_runs']}/{result['planned_runs']}",
        flush=True,
    )
    return 0 if result["status"] == "completed" else 2


def run_completion(args, stop):
    require_learning_allowed("la campaña de compleción")
    for protected in (args.reference, args.encoded.parent, args.tabular_config, args.post_config):
        outside_source(protected, args.output)
        outside_source(args.output, protected)
    safe_destination(args.output)
    dependency = None
    if args.after:
        status_path = args.output.with_name(args.output.name + "-dependency.json")
        safe_destination(status_path)

        def observe(status):
            atomic_json(
                status_path,
                dict(
                    status=status,
                    observed_at_utc=datetime.now(UTC).isoformat(),
                    dependency=str(args.after),
                    final_test_opened=False,
                ),
            )

        dependency = wait_dependency(args.after, stop, observe)
        observe("dependency_completed")
    identity, stages = _inputs(args)
    identity["dependency"] = dependency
    args.output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        args.output / ".completion.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(
            args.output, identity, record="summary.json", lock=".completion.lock"
        )
        summary = (
            read_manifest(args.output / "summary.json", 8 * 1024**2)[0]
            if confirmed
            else dict(
                schema_version=1,
                kind="temporal_posttraining_completion",
                identity=identity,
                status="running",
                stages=[dict(row, status="pending", completed_runs=0) for row in stages],
                planned_runs=sum(row["planned_runs"] for row in stages),
                completed_runs=0,
                final_test_opened=False,
            )
        )
        if summary["identity"] != identity or len(summary["stages"]) != len(stages):
            raise ValueError("La continuación pertenece a otra identidad")
        if (
            summary.get("final_test_opened") is not False
            or summary.get("planned_runs") != sum(row["planned_runs"] for row in stages)
            or summary.get("completed_runs")
            != sum(row["completed_runs"] for row in summary["stages"])
        ):
            raise ValueError("El progreso de las etapas no concilia")
        for specification, progress in zip(stages, summary["stages"], strict=True):
            if (
                any(progress.get(key) != value for key, value in specification.items())
                or type(progress.get("completed_runs")) is not int
                or not 0 <= progress["completed_runs"] <= specification["planned_runs"]
            ):
                raise ValueError("La etapa no conserva el diseño y su recuento")
            if progress["status"] == "completed":
                receipt = args.output / progress["fold"] / progress["stage"] / "summary.json"
                _receipt(receipt, progress["planned_runs"])
                if (
                    sha256(receipt) != progress.get("sha256")
                    or progress["completed_runs"] != progress["planned_runs"]
                ):
                    raise ValueError("Ha cambiado una etapa confirmada")
        if summary["status"] == "completed":
            from mars_titan.training.predictive_parents import _verified_file

            if summary["completed_runs"] != summary["planned_runs"]:
                raise ValueError("La campaña completa conserva etapas pendientes")
            for record in summary["analysis"].values():
                _verified_file(args.output, record)
            return summary

        def save():
            summary["completed_runs"] = sum(row["completed_runs"] for row in summary["stages"])
            summary["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(args.output / "summary.json", summary)

        summary["status"] = "running"
        summary.pop("error", None)
        save()
        try:
            for specification, progress in zip(stages, summary["stages"], strict=True):
                if any(progress.get(key) != value for key, value in specification.items()):
                    raise ValueError("La etapa no conserva el diseño")
                receipt = (
                    args.output / specification["fold"] / specification["stage"] / "summary.json"
                )
                if progress["status"] == "completed":
                    _receipt(receipt, specification["planned_runs"])
                    if sha256(receipt) != progress["sha256"]:
                        raise ValueError("Ha cambiado una etapa confirmada")
                    continue
                command = [sys.executable, "-m", "mars_titan.posttraining.completion"]
                for name in ("reference", "encoded", "tabular_config", "post_config", "output"):
                    command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
                command.extend(("--stage", specification["stage"], "--fold", specification["fold"]))
                progress["status"] = "running"
                save()

                def observe(result, entry=progress):
                    entry.update(
                        status="running" if result["status"] == "completed" else result["status"],
                        completed_runs=result["completed_runs"],
                    )
                    save()

                run_child(command, receipt, specification["planned_runs"], stop, observe)
                progress.update(status="completed", sha256=sha256(receipt))
                save()
            from mars_titan.posttraining.analysis import analyse_completion

            analyse_completion(args.output)
            summary["analysis"] = {
                name: dict(path=name, sha256=sha256(args.output / name))
                for name in ("analysis.json", "analysis.md")
            }
            summary["status"] = "completed"
        except InterruptedError:
            summary["status"] = "paused"
        except BaseException as error:
            summary.update(
                status="failed", error=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            save()
        return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "encoded", "tabular-config", "post-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--stage", choices=("tabular", "posttraining", "evaluation"))
    parser.add_argument("--fold")
    args = parser.parse_args()
    if args.stage:
        if args.fold is None or args.fold not in {f"fold-{i:03}" for i in range(128)}:
            parser.error("La etapa requiere una ventana válida")
        return _stage(args)
    with StopRequest() as stop:
        try:
            result = run_completion(args, stop)
        except InterruptedError:
            return 2
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
