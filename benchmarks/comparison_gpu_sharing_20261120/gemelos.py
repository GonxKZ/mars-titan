"""Repetir trabajos de la campaña de comparación en estados de prueba y cotejarlos con el original.

Sirve para medir si varios trabajos pueden compartir la GPU, con o sin MPS, con un tope de
VRAM o con otra configuración del asignador de PyTorch, sin cambiar ni un bit de lo que
calculan. No toca el estado de la campaña: `prepare` copia cohortes e índices a una carpeta
propia y registra allí su permiso, `run` lanza el mismo proceso `job` del runtime fijo en uno
o varios estados a la vez y `compare` coteja las trazas y el historial de épocas con los del
trabajo original.

Las rutas se pasan siempre de forma explícita. `run` conserva del entorno las variables de la
campaña y solo ajusta las que cambia cada ensayo.
"""

import argparse
import importlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MPS_VARIABLES = ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")
TRIAL_VARIABLES = (*MPS_VARIABLES, "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", "PYTORCH_CUDA_ALLOC_CONF")
COPIED = ("learning-traces.json", "trace-qualification.json", "file-digests.json")


def trial_environment(base, runtime, state, *, mps, vram_limit_mib=None, alloc_conf=None):
    """Entorno de un trabajo de prueba: el de la campaña con las opciones del ensayo."""
    if mps and not all(base.get(name) for name in MPS_VARIABLES):
        raise ValueError("El ensayo con MPS necesita las carpetas del demonio en el entorno")
    if vram_limit_mib is not None and not mps:
        raise ValueError("El tope de VRAM por proceso solo existe con MPS")
    environment = {name: value for name, value in base.items() if name not in TRIAL_VARIABLES}
    environment.update(
        PYTHONPATH=str(Path(runtime) / "src"),
        MARS_TITAN_DIGEST_CACHE=str(Path(state) / "file-digests.json"),
    )
    if mps:
        environment.update({name: base[name] for name in MPS_VARIABLES})
        if vram_limit_mib is not None:
            environment["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={vram_limit_mib}M"
    if alloc_conf is not None:
        environment["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    return environment


def campaign_module(runtime):
    root = Path(runtime).resolve()
    sys.path.insert(0, str(root / "src"))
    module = importlib.import_module("mars_titan.training.comparison_campaign")
    if Path(module.__file__).resolve() != root / "src/mars_titan/training/comparison_campaign.py":
        raise ValueError("Se ha cargado un controlador distinto del runtime solicitado")
    return module


def prepare(args):
    state, year = args.state.resolve(), args.key.split("/")[1]
    state.mkdir(parents=True)
    shutil.copytree(args.campaign / "cohorts" / year, state / "cohorts" / year)
    shutil.copytree(args.campaign / "indices", state / "indices")
    for name in COPIED:
        shutil.copy2(args.campaign / name, state / name)
    command = [
        sys.executable,
        "-m",
        "mars_titan.training.comparison_campaign",
        "authorize",
        "--state",
        str(state),
        "--note",
        args.note,
    ]
    environment = trial_environment(os.environ, args.runtime, state, mps=False)
    subprocess.run(command, cwd=args.runtime, env=environment, check=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def query(*fields):
    result = subprocess.run(
        ["nvidia-smi", "--id=0", *fields, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def device_sample(zone):
    """Uso, potencia, temperatura y memoria de la GPU, VRAM por proceso y estado del equipo."""
    names = "utilization.gpu,power.draw,temperature.gpu,clocks.current.graphics,memory.used"
    try:
        gpu = [float(value) for value in query(f"--query-gpu={names}")[0].split(",")]
        rows = (line.split(",") for line in query("--query-compute-apps=pid,used_memory"))
        used = {int(pid): int(amount) for pid, amount in rows}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        gpu, used = None, {}
    cores = Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
    frequencies = sorted((int(path.read_text()) / 1e6 for path in cores), reverse=True)
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = round(int(line.split()[1]) / 1024**2, 2)
    sample = dict(
        gpu=gpu,
        board_celsius=None if zone is None else int(Path(zone).read_text()) / 1000,
        frequency_top4_ghz=[round(value, 2) for value in frequencies[:4]],
        frequency_mean_ghz=round(sum(frequencies) / len(frequencies), 3) if frequencies else None,
        available_gib=available,
    )
    return sample, used


def group_anonymous(pid, *, proc=Path("/proc"), groups=Path("/sys/fs/cgroup")):
    """Memoria anónima del grupo de recursos del proceso, sin la caché de archivos recuperable."""
    try:
        group = (proc / str(pid) / "cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
        lines = (groups / group / "memory.stat").read_text().splitlines()
        stat = dict(line.split() for line in lines)
        return int(stat["anon"]) + int(stat.get("shmem", 0))
    except (OSError, IndexError, KeyError, ValueError):
        return None


def scope_command(key, memory_max):
    """Orden del trabajo dentro de un grupo de recursos propio con la RAM acotada."""
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        f"MemoryMax={memory_max}",
        "-p",
        "MemorySwapMax=256M",
        "--",
        # Si falta RAM, el sistema cierra antes la prueba que la sesión de trabajo.
        "choom",
        "-n",
        "900",
        "--",
        sys.executable,
        "-m",
        "mars_titan.training.comparison_campaign",
        "job",
        "--machine",
        "references",
        "--key",
        key,
    ]


def advance(twin, fit, now, until_step):
    """Anotar el avance de una prueba y decir si hay que pedirle ya que se detenga."""
    step, cursor = fit.get("global_step"), fit.get("cursor")
    if step is None:
        return False
    mark = (step, json.dumps(cursor, sort_keys=True))
    if mark != twin.get("mark"):
        twin["mark"] = mark
        twin["timeline"].append((round(now, 2), step, cursor))
        print(f"{now:8.1f} s  {twin['state'].name}: paso {step} {cursor}", flush=True)
    if step < until_step or twin["stopped"]:
        return False
    twin["stopped"] = True
    twin["timeline"].append((round(now, 2), step, "stop_requested"))
    return True


def outcome(twin, record):
    """Resultado de una prueba: su registro en el estado propio y los picos observados."""
    attempt = record["attempts"][-1] if record["attempts"] else {}
    return dict(
        state=twin["state"].name,
        exit_code=twin["process"].returncode,
        record_status=record["status"],
        attempt_status=attempt.get("status"),
        attempt_seconds=attempt.get("seconds"),
        error=attempt.get("error"),
        resources=attempt.get("resources"),
        timeline=twin["timeline"],
        group_anonymous_peak_gib=round(twin["anonymous"] / 2**30, 3),
        process_vram_peak_mib=twin["vram"],
    )


def run(args):
    campaign = campaign_module(args.runtime)
    declaration, _ = campaign.load_declaration(None)
    job = {job.key: job for job in campaign.plan(declaration, "references")}[args.key]
    twins, saved = [], dict(os.environ)
    options = dict(mps=args.mps, vram_limit_mib=args.vram_limit_mib, alloc_conf=args.alloc_conf)
    for state in args.state:
        state = state.resolve()
        layout = campaign.Layout(state, None)
        os.environ.clear()
        os.environ.update(trial_environment(saved, args.runtime, state, **options))
        try:
            environment = campaign._environment(layout, declaration)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        twins.append(dict(state=state, layout=layout, environment=environment))
    result = dict(
        key=args.key,
        **options,
        until_step=args.until_step,
        states=[twin["state"].name for twin in twins],
        started_at_utc=datetime.now(UTC).isoformat(),
        twins=[],
        samples=[],
    )
    started = time.monotonic()
    for twin in twins:
        (twin["state"] / "logs").mkdir(exist_ok=True)
        with (twin["state"] / "logs/job.log").open("a") as stream:
            twin["process"] = subprocess.Popen(
                scope_command(args.key, args.memory_max),
                cwd=args.runtime,
                env=twin["environment"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        twin.update(timeline=[(0.0, None, "launched")], stopped=False, anonymous=0, vram=0)
        if args.stagger and twin is not twins[-1]:
            time.sleep(args.stagger)
    sampled = float("-inf")
    while any(twin["process"].poll() is None for twin in twins):
        now, used = time.monotonic() - started, {}
        if now - sampled >= args.sample_seconds:
            sampled = now
            sample, used = device_sample(args.thermal_zone)
            result["samples"].append(dict(t=round(now, 2), **sample))
        for twin in twins:
            process = twin["process"]
            if process.poll() is not None:
                continue
            fit = read_json(twin["layout"].output(job.key) / "fit/run.json") or {}
            twin["anonymous"] = max(twin["anonymous"], group_anonymous(process.pid) or 0)
            twin["vram"] = max(twin["vram"], used.get(process.pid, 0))
            if advance(twin, fit, now, args.until_step):
                # El trabajo se detiene en su siguiente barrera confirmada, como en la cola.
                process.send_signal(signal.SIGTERM)
        time.sleep(args.poll_seconds)
    for twin in twins:
        result["twins"].append(outcome(twin, campaign.read_record(twin["layout"], job.key)))
    result["elapsed_seconds"] = time.monotonic() - started
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")


def bits(value):
    """Patrón de bits de un número en doble precisión, para comparar sin tolerancia."""
    return None if value is None else struct.pack("<d", float(value)).hex()


def trace_rows(folder, limit):
    """Valores de las trazas de aprendizaje hasta el paso `limit`, agrupados por su clave."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = sorted((Path(folder) / "traces/parts").glob("part-*.parquet"))
    if not parts:
        return {}
    rows = {}
    for row in pa.concat_tables([pq.read_table(part) for part in parts]).to_pylist():
        if row["step"] > limit:
            continue
        key = (row["step"], row["phase"], row["metric"], row["group"], row["stat"])
        rows.setdefault(key, []).append(bits(row["value"]))
    return rows


def flat(value, prefix=""):
    """Hojas de un documento JSON con su ruta. Los números reales se comparan por sus bits."""
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flat(value[key], f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flat(item, f"{prefix}[{index}]")
    else:
        yield prefix, bits(value) if isinstance(value, float) else value


def compare_run(folder, original):
    """Cotejar un trabajo de prueba con el original hasta el paso que alcanzó la prueba."""
    fit, reference = read_json(Path(folder) / "fit/run.json"), read_json(original / "fit/run.json")
    if fit is None or reference is None:
        raise ValueError("Falta el estado del ajuste de la prueba o del original")
    step = fit["global_step"]
    mine, theirs = trace_rows(folder, step), trace_rows(original, step)
    shared = mine.keys() & theirs.keys()
    different = sorted(key for key in shared if mine[key] != theirs[key])
    epochs = {item["epoch"]: item for item in reference["history"]}
    history, missing = [], []
    for item in sorted(fit["history"], key=lambda item: item["epoch"]):
        if item["epoch"] not in epochs:
            missing.append(item["epoch"])
            continue
        a, b = dict(flat(item)), dict(flat(epochs[item["epoch"]]))
        history += [
            (item["epoch"], key, a.get(key), b.get(key))
            for key in sorted(a.keys() | b.keys())
            if a.get(key) != b.get(key)
        ]
    entry = dict(
        status=fit["status"],
        global_step=step,
        cursor=fit["cursor"],
        identity_equal=fit["identity"] == reference["identity"],
        run_id_equal=fit["run_id"] == reference["run_id"],
        trace_rows=len(mine),
        trace_rows_reference_until_step=len(theirs),
        trace_rows_compared=len(shared),
        trace_rows_only_twin=len(mine.keys() - theirs.keys()),
        trace_rows_only_reference=len(theirs.keys() - mine.keys()),
        trace_rows_different=len(different),
        first_trace_differences=[[*key, mine[key], theirs[key]] for key in different[:5]],
        history_epochs_compared=sorted(item["epoch"] for item in fit["history"]),
        history_epochs_missing_in_reference=missing,
        history_differences=len(history),
        first_history_differences=history[:10],
    )
    entry["identical"] = bool(
        entry["identity_equal"]
        and entry["run_id_equal"]
        and shared
        and not different
        and not entry["trace_rows_only_twin"]
        and not missing
        and not history
    )
    if fit["status"] == "completed":
        equal = fit["best_checkpoint"]["sha256"] == reference["best_checkpoint"]["sha256"]
        entry.update(best_checkpoint_equal=equal)
        entry["identical"] = entry["identical"] and equal
    return entry


def compare(args):
    original = args.campaign / "runs" / args.key
    report = dict(key=args.key, twins=[])
    for state in args.state:
        entry = compare_run(state.resolve() / "runs" / args.key, original)
        report["twins"].append(dict(state=state.resolve().name, **entry))
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if all(twin["identical"] for twin in report["twins"]) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prepare", help="Crear un estado de prueba con su permiso")
    one.add_argument("--state", type=Path, required=True)
    one.add_argument("--note", required=True)
    two = commands.add_parser("run", help="Lanzar el trabajo en uno o varios estados a la vez")
    two.add_argument("--state", type=Path, action="append", required=True)
    two.add_argument("--until-step", type=int, required=True)
    two.add_argument("--out", type=Path, required=True)
    two.add_argument("--mps", action="store_true")
    two.add_argument("--vram-limit-mib", type=int)
    two.add_argument("--alloc-conf")
    two.add_argument("--memory-max", default="8G")
    two.add_argument("--stagger", type=float, default=0.0)
    two.add_argument("--sample-seconds", type=float, default=2.0)
    two.add_argument("--poll-seconds", type=float, default=0.5)
    two.add_argument("--thermal-zone", type=Path)
    three = commands.add_parser("compare", help="Cotejar las pruebas con el trabajo original")
    three.add_argument("--state", type=Path, action="append", required=True)
    three.add_argument("--out", type=Path)
    for command in (one, two, three):
        command.add_argument("--key", required=True, help="Trabajo, como modelo/año/seed-N")
    for command in (one, two):
        command.add_argument("--runtime", type=Path, required=True, help="Copia científica fija")
    for command in (one, three):
        command.add_argument("--campaign", type=Path, required=True, help="Estado de la campaña")
    args = parser.parse_args(argv)
    return {"prepare": prepare, "run": run, "compare": compare}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
