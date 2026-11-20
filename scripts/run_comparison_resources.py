"""Ejecutar una cola congelada con un presupuesto de RAM por trabajo.

Con varios trabajadores y los presupuestos medidos de un equipo, los trabajos que caben a la
vez comparten la GPU. La admisión solo decide cuándo empieza cada trabajo: no cambia sus
datos, su receta ni ninguna operación numérica.
"""

import argparse
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

GIB = 1024**3
RESERVE_GIB = 4
ALLOC_CONF = re.compile(r"[A-Za-z0-9_.]+:[A-Za-z0-9_.]+(,[A-Za-z0-9_.]+:[A-Za-z0-9_.]+)*")


def memory_limit(runner):
    if runner == "tabular":
        return 12
    if runner in {"fit", "reader", "ridge", "rule"}:
        return 8
    raise ValueError("El ejecutor no tiene un presupuesto de RAM declarado")


def available_memory(path=Path("/proc/meminfo")):
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields[:1] == ["MemAvailable:"]:
            if len(fields) != 3 or fields[2] != "kB" or not fields[1].isdecimal():
                break
            return int(fields[1]) * 1024
    raise ValueError("No se puede comprobar la RAM disponible")


def memory_ready(limit_gib, available):
    if (
        type(limit_gib) is not int
        or not 1 <= limit_gib <= 12
        or type(available) is not int
        or available < 0
    ):
        raise ValueError("El límite o la memoria disponible no son válidos")
    return available >= (limit_gib + RESERVE_GIB) * GIB


def _whole(value, low):
    return type(value) is int and value >= low


class Packing:
    """Presupuestos medidos en un equipo para decidir qué trabajos caben a la vez en su GPU.

    `vram_mib` es el tope de VRAM de cada modelo ya medido y `vram_default_mib` el de los que
    aún no lo están, por ejecutor. Solo los ejecutores de `admit_gib` comparten la GPU, con esa
    RAM propia de admisión y hasta `max_jobs` trabajos. Los demás se ejecutan solos.

    `cuda_alloc_conf` es la configuración del asignador de PyTorch con la que se midieron los
    topes. El pico de reserva depende de ella, así que solo la reciben los modelos con tope.
    """

    def __init__(self, document):
        valid = isinstance(document, dict) and document.get("schema_version") == 1
        if valid:
            budgets, defaults = document.get("vram_mib"), document.get("vram_default_mib")
            admit, hot = document.get("admit_gib"), document.get("max_celsius")
            settle, conf = document.get("settle_seconds"), document.get("cuda_alloc_conf")
            valid = (
                _whole(document.get("device_budget_mib"), 1)
                and _whole(document.get("max_jobs"), 1)
                and _whole(document.get("reserve_mib"), 0)
                and type(settle) in (int, float)
                and settle >= 0
                and (hot is None or type(hot) in (int, float))
                and (conf is None or (type(conf) is str and bool(ALLOC_CONF.fullmatch(conf))))
                and all(
                    isinstance(table, dict)
                    and all(isinstance(k, str) and _whole(v, 1) for k, v in table.items())
                    for table in (budgets, defaults, admit)
                )
                and bool(admit)
                and admit.keys() <= defaults.keys()
                and all(value <= 12 for value in admit.values())
            )
        if not valid:
            raise ValueError("Los presupuestos del equipo no son válidos")
        self.device_budget_mib, self.max_jobs = document["device_budget_mib"], document["max_jobs"]
        self.reserve_mib, self.settle_seconds = document["reserve_mib"], settle
        self.max_celsius, self.vram_mib = hot, dict(budgets)
        self.vram_default_mib, self.admit_gib = dict(defaults), dict(admit)
        self.cuda_alloc_conf = conf

    def current(self):
        return self

    def shares(self, job):
        return job.runner in self.admit_gib

    def vram(self, job):
        return self.vram_mib.get(job.model, self.vram_default_mib[job.runner])

    def cap(self, job):
        """Tope de VRAM del proceso, solo para modelos cuyo pico ya se ha medido."""
        return self.vram_mib.get(job.model) if self.shares(job) else None

    def alloc_conf(self, job):
        """Configuración del asignador del proceso, la misma con la que se midió su tope."""
        return None if self.cap(job) is None else self.cuda_alloc_conf

    def admits(self, job, running, seen):
        """Si `job` puede empezar mientras siguen los trabajos `running`.

        `seen` es lo observado en ese momento: segundos desde el último arranque, temperatura,
        RAM disponible, VRAM libre y VRAM que ocupa ya cada trabajo en marcha.
        """
        if not running:
            return True
        if len(running) >= self.max_jobs or not all(self.shares(item) for item in (job, *running)):
            return False
        if seen.since_launch < self.settle_seconds:
            return False
        if self.max_celsius is not None and not (
            seen.celsius is not None and seen.celsius <= self.max_celsius
        ):
            return False
        # Cada archivo de entradas preparado guarda un solo origen por tamaño de bloque.
        if any(item.year != job.year for item in running):
            return False
        if not memory_ready(self.admit_gib[job.runner], seen.available):
            return False
        need = self.vram(job)
        if need + sum(self.vram(item) for item in running) > self.device_budget_mib:
            return False
        # Un trabajo recién arrancado aún no ocupa su pico: se descuenta lo que le falta.
        owed = sum(max(0, self.vram(item) - seen.used.get(item.key, 0)) for item in running)
        return seen.free_mib - owed >= need + self.reserve_mib


def read_packing(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError("Los presupuestos del equipo exceden el límite de lectura")
    return Packing(json.loads(path.read_text(encoding="utf-8")))


class PackingFile:
    """Presupuestos leídos de un archivo que se puede corregir con la cola en marcha.

    Cada medida nueva (el pico de un modelo, un trabajo más a la vez) se aplica al siguiente
    trabajo sin parar los que están en curso. Un archivo que deja de ser válido no cambia nada:
    siguen los últimos presupuestos válidos.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.stamp, self.packing = self._stamp(), read_packing(self.path)

    def _stamp(self):
        status = self.path.stat()
        return status.st_mtime_ns, status.st_size

    def current(self):
        try:
            stamp = self._stamp()
            if stamp != self.stamp:
                self.packing, self.stamp = read_packing(self.path), stamp
                print(f"Presupuestos del equipo releídos de {self.path.name}", flush=True)
        except (OSError, ValueError) as error:
            print(f"Se conservan los presupuestos anteriores: {error}", flush=True)
        return self.packing


def device_memory(run=subprocess.run):
    """VRAM libre de la GPU 0 y VRAM que ocupa cada proceso, en MiB."""
    values = []
    for query in ("--query-gpu=memory.free", "--query-compute-apps=pid,used_memory"):
        command = ["nvidia-smi", "--id=0", query, "--format=csv,noheader,nounits"]
        result = run(command, capture_output=True, text=True, timeout=10, check=True)
        values.append([line for line in result.stdout.splitlines() if line.strip()])
    free, processes = values
    if len(free) != 1:
        raise ValueError("No se puede leer la VRAM libre")
    used = {}
    for line in processes:
        pid, amount = (int(field) for field in line.split(","))
        used[pid] = used.get(pid, 0) + amount
    return int(free[0]), used


def celsius(zone):
    return None if zone is None else int(Path(zone).read_text()) / 1000


def mps_ready(environment=os.environ):
    """Si el demonio MPS de la carpeta declarada sigue vivo para atender a los trabajos.

    Un demonio que muere deja su socket en la carpeta, así que también se comprueba su proceso.
    """
    folder = environment.get("CUDA_MPS_PIPE_DIRECTORY")
    if not folder or not (Path(folder) / "control").is_socket():
        return False
    try:
        pid = int((Path(folder) / "nvidia-cuda-mps-control.pid").read_text())
    except (OSError, ValueError):
        return False
    return Path(f"/proc/{pid}").is_dir()


def runtime_path(runtime):
    root = Path(runtime).resolve()
    if not (root / "src/mars_titan/training/comparison_campaign.py").is_file():
        raise ValueError("El runtime no contiene el controlador de la comparación")
    source = str(root / "src")
    sys.path.insert(0, source)
    os.environ["PYTHONPATH"] = source
    return root


def load_campaign(runtime):
    root = runtime_path(runtime)
    module = importlib.import_module("mars_titan.training.comparison_campaign")
    expected = root / "src/mars_titan/training/comparison_campaign.py"
    if Path(module.__file__).resolve() != expected:
        raise ValueError("Se ha cargado un controlador distinto del runtime solicitado")
    return module


class Status:
    def __init__(self, path, job, limit):
        self.path, self.job, self.limit = Path(path), job, limit
        self.key, self.saved = None, float("-inf")

    def save(self, phase, *, available=None, pid=None, returncode=None):
        key, now = (phase, pid, returncode), time.monotonic()
        if key == self.key and now - self.saved < 15:
            return
        record = dict(
            observed_at_utc=datetime.now(UTC).isoformat(),
            job=self.job,
            phase=phase,
            limit_gib=self.limit,
            reserve_gib=RESERVE_GIB,
            available_bytes=available,
            pid=pid,
            returncode=returncode,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".resource-status-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        if key != self.key:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        self.key, self.saved = key, now


def running_pid(record, owned):
    path = Path(record)
    if not path.exists():
        return None
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("El registro del trabajo excede el límite de lectura")
    data = json.loads(path.read_text())
    attempts = data.get("attempts", [])
    pid = attempts[-1].get("pid") if attempts else None
    return pid if data.get("status") == "running" and pid in owned else None


def run_worker(
    launcher,
    limit_gib,
    command,
    *,
    status,
    record,
    job,
    stop,
    probe=available_memory,
    poll_seconds=1,
    stop_timeout=600,
    admit_gib=None,
    vram_cap_mib=None,
    cuda_alloc_conf=None,
):
    from mars_titan.training import gpu_supervisor as processes

    if not command or not Path(launcher).is_file() or not os.access(launcher, os.X_OK):
        raise ValueError("Falta el comando o el lanzador de recursos")
    state = Status(status, job, limit_gib)
    # El límite acota el grupo de recursos del trabajo. Al compartir la GPU, la admisión usa
    # la RAM propia medida del trabajo, porque el resto del límite es caché recuperable.
    admit = limit_gib if admit_gib is None else admit_gib
    child, owned, generation = None, {}, None
    parent = os.getppid()
    try:
        while not stop.requested and os.getppid() == parent:
            available = probe()
            if memory_ready(admit, available):
                break
            state.save("waiting_memory", available=available)
            time.sleep(poll_seconds)
        if stop.requested or os.getppid() != parent:
            state.save("paused")
            return 3
        environment = os.environ.copy()
        environment.setdefault(
            "MARS_TITAN_DIGEST_CACHE", str(state.path.absolute().parent / "file-digests.json")
        )
        if vram_cap_mib is not None:
            # MPS devuelve falta de memoria a este proceso antes de que invada a los demás.
            environment["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={vram_cap_mib}M"
        if cuda_alloc_conf is not None:
            # Solo cambia dónde reserva PyTorch la memoria de GPU, no las operaciones.
            environment["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
        # memslot vuelve a comprobar la memoria después de adquirir el cerrojo de GPU.
        child = subprocess.Popen(
            [str(launcher), "gpu", "--max", f"{limit_gib}G", "--", *command],
            start_new_session=True,
            env=environment,
        )
        root = processes._process_info(child.pid)
        if root is None:
            raise RuntimeError("No se puede acreditar el proceso de recursos")
        generation = root[2]
        owned = {child.pid: generation}
        while not stop.requested and os.getppid() == parent:
            owned = processes._owned_processes(child.pid, owned, generation)
            if processes._exited(child):
                code, forced, _ = processes._stop_child(child, stop_timeout, owned, generation)
                child = None
                code = 75 if forced else code if code >= 0 else 128 - code
                phase = "completed" if code == 0 else "paused" if code == 3 else "failed"
                state.save(phase, returncode=code)
                return code
            pid = running_pid(record, owned)
            available = probe()
            phase = "running" if pid else "queued_resources"
            if not pid and not memory_ready(admit, available):
                phase = "waiting_memory"
            state.save(phase, available=available, pid=pid)
            time.sleep(poll_seconds)
        code, forced, _ = processes._stop_child(child, stop_timeout, owned, generation)
        child = None
        state.save("failed" if forced else "paused", returncode=code)
        return 75 if forced else 3
    except BaseException:
        state.save("failed")
        raise
    finally:
        if child is not None:
            processes._stop_child(child, stop_timeout, owned, generation)


def make_queue(campaign, args):
    declaration, digest = campaign.load_declaration(args.declaration)
    layout = campaign.Layout(args.state, args.views)
    if not campaign.authorized(
        layout, digest, commit=campaign.code_commit(), changes=campaign.code_changes()
    ):
        raise ValueError("El runtime y la declaración no tienen un permiso coincidente")

    workers, budgets = getattr(args, "workers", 1), getattr(args, "packing", None)
    zone = getattr(args, "thermal_zone", None)
    if workers != 1 and budgets is None:
        raise ValueError("Varios trabajadores necesitan los presupuestos medidos del equipo")
    if budgets is not None and not mps_ready():
        raise ValueError("Compartir la GPU necesita el demonio MPS de CUDA_MPS_PIPE_DIRECTORY")

    class ResourceQueue(campaign.Queue):
        launched, warned = float("-inf"), None

        def _command(self, job, *, preflight=False):
            original = super()._command(job, preflight=preflight)
            shared, packing = [], None if budgets is None else budgets.current()
            if packing is not None and packing.shares(job):
                shared = ["--admit-gib", str(packing.admit_gib[job.runner])]
                if packing.cap(job) is not None:
                    shared += ["--vram-cap-mib", str(packing.cap(job))]
                if packing.alloc_conf(job) is not None:
                    shared += ["--cuda-alloc-conf", packing.alloc_conf(job)]
            return [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--runtime",
                str(args.runtime),
                "--resource-launcher",
                str(args.resource_launcher),
                "--limit-gib",
                str(memory_limit(job.runner)),
                *shared,
                "--status",
                str(layout.state / "resource-status.json"),
                "--record",
                str(layout.record(job.key)),
                "--job",
                job.key,
                "--",
                *original,
            ]

        def _pid(self, job):
            record = campaign.read_record(self.layout, job.key)
            attempts = record.get("attempts", [])
            return attempts[-1].get("pid") if attempts and record["status"] == "running" else None

        def _admissible(self):
            """Trabajos que pueden empezar ahora, en el orden del plan. Se llama con el cerrojo."""
            if budgets is None:
                return self.jobs
            if not mps_ready():
                # Con su carpeta exportada, un trabajo sin demonio fallaría al iniciar CUDA.
                self._warn("El demonio MPS no responde: no se lanza ningún trabajo")
                return []
            running = [job for job in self.jobs if job.key in self.taken]
            if not running:
                self._warn(None)
                return self.jobs
            packing = budgets.current()
            try:
                free, used = device_memory()
                seen = SimpleNamespace(
                    since_launch=time.monotonic() - self.launched,
                    celsius=celsius(zone),
                    available=available_memory(),
                    free_mib=free,
                    used={job.key: used.get(self._pid(job), 0) for job in running},
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self._warn(f"Sin medidas del equipo no se añade otro trabajo: {error}")
                return []
            self._warn(None)
            return [job for job in self.jobs if packing.admits(job, running, seen)]

        def _warn(self, message):
            """Avisar una vez de cada impedimento, sin repetirlo en cada consulta."""
            if message is not None and message != self.warned:
                print(message, flush=True)
            self.warned = message

        def _work(self):
            while not self.stop.is_set():
                if self._deadline_reached() or self._disk_exhausted():
                    return
                with self.guard:
                    job, handle = campaign._claim(self.layout, self._admissible(), self.taken)
                    if job is not None:
                        together = sorted(self.taken)
                        self.taken.add(job.key)
                        self.launched = time.monotonic()
                if job is None:
                    with self.guard:
                        busy = bool(self.taken)
                    if not busy or self._finished():
                        return
                    # Queda trabajo que ahora no cabe o que espera a su padre.
                    self.stop.wait(self.poll_seconds)
                    continue
                try:
                    self._note("start", job, together)
                    self._launch(job)
                finally:
                    with self.guard:
                        self.taken.discard(job.key)
                    handle.close()
                    self._note("end", job, None)

        def _note(self, event, job, together):
            """Anotar los solapes. Un tiempo medido con solape no es el coste en solitario."""
            if budgets is None:
                return
            line = dict(at_utc=datetime.now(UTC).isoformat(), event=event, job=job.key)
            if together is not None:
                packing = budgets.current()
                line.update(
                    together=together,
                    vram_cap_mib=packing.cap(job),
                    cuda_alloc_conf=packing.alloc_conf(job),
                )
            with self.guard, (self.layout.state / "packing-log.jsonl").open("a") as stream:
                stream.write(json.dumps(line, ensure_ascii=False) + "\n")

    return ResourceQueue(
        declaration, layout, args.machine, workers=workers, declaration_path=args.declaration
    )


def main(argv=None):
    values = list(sys.argv[1:] if argv is None else argv)
    worker = values[:1] == ["worker"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--resource-launcher", type=Path, required=True)
    if worker:
        parser.add_argument("--limit-gib", type=int, choices=(8, 12), required=True)
        parser.add_argument("--admit-gib", type=int)
        parser.add_argument("--vram-cap-mib", type=int)
        parser.add_argument("--cuda-alloc-conf")
        parser.add_argument("--status", type=Path, required=True)
        parser.add_argument("--record", type=Path, required=True)
        parser.add_argument("--job", required=True)
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(values[1:])
        runtime_path(args.runtime)
        from mars_titan.training.gpu_supervisor import StopFlag

        stop = StopFlag()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop.request)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return run_worker(
            args.resource_launcher,
            args.limit_gib,
            command,
            status=args.status,
            record=args.record,
            job=args.job,
            stop=stop,
            admit_gib=args.admit_gib,
            vram_cap_mib=args.vram_cap_mib,
            cuda_alloc_conf=args.cuda_alloc_conf,
        )
    parser.add_argument("--machine", choices=("references",), required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Trabajos a la vez, si caben")
    parser.add_argument("--packing", type=Path, help="Presupuestos medidos del equipo")
    parser.add_argument("--thermal-zone", type=Path, help="Archivo con la temperatura en m°C")
    args = parser.parse_args(values)
    args.runtime = args.runtime.resolve()
    args.resource_launcher = args.resource_launcher.resolve(strict=True)
    if not args.resource_launcher.is_file() or not os.access(args.resource_launcher, os.X_OK):
        raise ValueError("El lanzador de recursos debe ser un archivo ejecutable")
    for field in ("state", "views", "declaration", "results"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if args.packing is not None:
        args.packing = PackingFile(args.packing)
    campaign = load_campaign(args.runtime)
    queue = make_queue(campaign, args)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, queue.request_stop)
    result = queue.run()
    campaign._print_summary(result)
    pending = any(campaign.job_state(queue.layout, job) != campaign.DONE for job in queue.jobs)
    if pending:
        return 3
    if args.results:
        command = [
            sys.executable,
            "-m",
            "mars_titan.training.comparison_campaign",
            "export-results",
            "--machine",
            args.machine,
            "--state",
            str(queue.layout.state),
            "--destination",
            str(args.results),
        ]
        if args.declaration:
            command += ["--declaration", str(args.declaration)]
        return subprocess.run(command, cwd=args.runtime, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
