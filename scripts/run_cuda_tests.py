"""Ejecutar la selección CUDA secuencialmente, con un proceso pytest por caso."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _environment(receipt):
    env = dict(os.environ)
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH"))))
    env["MARS_TITAN_PYTEST_RECEIPT"] = str(receipt)
    return env


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def _process(directory, arguments, timeout):
    directory.mkdir()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-rs",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-p",
        "scripts.pytest_case_receipt",
        "--junitxml=" + str(directory / "junit.xml"),
        *arguments,
    ]
    record = dict(command=command, state="running", exit_code=None)
    started = time.monotonic()
    with (directory / "pytest.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=_environment(directory / "receipt.json"),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        record["pid"] = process.pid
        try:
            _write(directory / "process.json", record)
            record["exit_code"] = process.wait(timeout=timeout)
            record["state"] = "finished"
        except subprocess.TimeoutExpired:
            record.update(state="timed_out", exit_code=124)
        except KeyboardInterrupt:
            record.update(state="interrupted", exit_code=2)
        finally:
            _stop(process)
    record["seconds"] = time.monotonic() - started
    _write(directory / "process.json", record)
    return record


def _receipt(directory, process):
    receipt = json.loads((directory / "receipt.json").read_text())
    if receipt.get("pid") != process["pid"] or receipt.get("exit_code") != process["exit_code"]:
        raise ValueError("El recibo no corresponde al proceso terminado")
    selected = receipt["selected"]
    if not isinstance(selected, list) or any(
        not isinstance(node, str) or not node for node in selected
    ):
        raise ValueError("La selección debe contener identificadores no vacíos")
    if len(set(selected)) != len(selected):
        raise ValueError("La selección contiene identificadores duplicados")
    return receipt


def _case_state(receipt, node, exit_code):
    if receipt["selected"] != [node] or any(phase["nodeid"] != node for phase in receipt["phases"]):
        raise ValueError("El proceso no ejecutó exclusivamente el caso seleccionado")
    phases = {phase["when"]: phase["outcome"] for phase in receipt["phases"]}
    if len(phases) != len(receipt["phases"]):
        raise ValueError("El recibo contiene fases duplicadas")
    if exit_code or "failed" in phases.values():
        return "failed"
    if "skipped" in phases.values():
        return "skipped"
    return (
        "passed"
        if phases == dict(setup="passed", call="passed", teardown="passed")
        else "incomplete"
    )


def _suite(output, paths, timeout, summary):
    collection = _process(
        output / "collection", ["--collect-only", "-m", "cuda", "--", *paths], timeout
    )
    summary["collection"] = collection
    if collection["exit_code"]:
        return collection["exit_code"]
    selected = _receipt(output / "collection", collection)["selected"]
    if not selected:
        return 5
    summary.update(selected=selected, pending=selected.copy())
    _write(output / "summary.json", summary)
    for index, node in enumerate(selected):
        directory = output / f"{index:05d}"
        case = _process(directory, ["--", node], timeout)
        case["nodeid"] = node
        summary["cases"].append(case)
        summary["pending"] = selected[index + 1 :]
        if case["exit_code"] == 2 or case["exit_code"] < 0:
            case["state"] = "interrupted"
        if case["state"] in ("timed_out", "interrupted"):
            return case["exit_code"] if case["exit_code"] >= 0 else 2
        case["state"] = "invalid_receipt"
        case["state"] = _case_state(_receipt(directory, case), node, case["exit_code"])
        _write(output / "summary.json", summary)
        print(f"Caso {index + 1}/{len(selected)}: {node} ({case['state']})", flush=True)
    return int(any(case["state"] not in ("passed", "skipped") for case in summary["cases"]))


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("paths", nargs="*", default=["tests"])
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("El plazo por proceso debe estar entre 0 y 3600 segundos")
    output = args.output.resolve()
    output.mkdir(parents=True)
    summary = dict(
        schema_version=1,
        started_at_utc=datetime.now(UTC).isoformat(),
        supervisor_pid=os.getpid(),
        selected=[],
        cases=[],
        pending=[],
        exit_code=2,
    )
    started = time.monotonic()
    _write(output / "summary.json", summary)
    try:
        summary["exit_code"] = _suite(output, args.paths, args.timeout_seconds, summary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        summary.update(exit_code=3, error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        summary["exit_code"] = 2
    finally:
        summary.update(
            seconds=time.monotonic() - started,
            finished_at_utc=datetime.now(UTC).isoformat(),
            supervisor_imported_torch="torch" in sys.modules,
        )
        _write(output / "summary.json", summary)
        print(output / "summary.json", flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    raise SystemExit(main())
