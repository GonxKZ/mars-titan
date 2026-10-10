"""Procesos de las ranuras GPU de la campaña: un trabajo por proceso, con VRAM acotada.

Cada trabajo GPU concurrente se ejecuta en un proceso nuevo creado con `spawn`. El proceso
fija sus variables (tubería de lectura, MPS y el límite de VRAM del cliente) antes de
importar PyTorch, limita el asignador a la VRAM declarada sin que el trabajo pueda ampliarla
y toma un cerrojo exclusivo del trabajo, de modo que dos procesos nunca escriben el mismo
punto de control. Si la campaña termina de forma abrupta, el núcleo mata al proceso hijo
(`PR_SET_PDEATHSIG`), y una nueva ejecución reanuda cada trabajo desde su propio intento.

El proceso ignora SIGINT y SIGTERM: la parada llega desde la campaña por un evento
compartido y el trabajo la atiende en su siguiente barrera confirmada. El resultado vuelve
por una tubería: el informe, la pausa o el error con su traza. Este módulo no importa
PyTorch en el proceso de la campaña.
"""

import ctypes
import fcntl
import multiprocessing
import os
import signal
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .campaign_resources import CONTEXT_MIB, MIB

# Opción PR_SET_PDEATHSIG de prctl: el núcleo avisa al hijo cuando muere su padre.
PDEATHSIG = 1
_CONTEXT = multiprocessing.get_context("spawn")
MPS_SERVER = "nvidia-cuda-mps-server"


class EventStop:
    """Parada de un trabajo hijo, pedida por la campaña con un evento compartido."""

    def __init__(self, event):
        self.event = event

    @property
    def requested(self):
        return self.event.is_set()


@dataclass(frozen=True)
class SlotTask:
    """Lo que el proceso necesita: ejecutor, `JobRun` sin parada, entorno y VRAM."""

    executor: str
    run: dict
    environment: dict
    vram_bytes: int
    lock: str


def _die_with_parent():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:  # pragma: no cover (solo existe en Linux)
        return
    libc.prctl(PDEATHSIG, signal.SIGKILL)


def strict_fp32():
    """FP32 estricto y cuDNN determinista en el proceso que ejecuta un trabajo CUDA.

    Sin TF32 en matmul ni en cuDNN, con la precisión más alta de matmul y sin la búsqueda
    de algoritmos de cuDNN, de modo que el mismo trabajo da los mismos bits solo, en otra
    ranura o junto a otros. Se aplica igual en la ejecución en serie y en las ranuras.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def _cuda_available():
    import torch

    return torch.cuda.is_available()


def bound_vram(limit):
    """Limitar el asignador de PyTorch a `limit` bytes en cuda:0 sin permitir ampliarlo."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, limit / total)
    setter = torch.cuda.set_per_process_memory_fraction

    def bounded(value, device=None):
        nonlocal fraction
        fraction = min(fraction, value)
        setter(fraction, device)

    torch.cuda.set_per_process_memory_fraction = bounded
    setter(fraction, 0)
    return fraction


def _resolve(name):
    module, _, attribute = name.partition(":")
    from importlib import import_module

    return getattr(import_module(module), attribute)


def slot_main(connection, task, event):
    """Entrada del proceso de una ranura. Nunca lanza: devuelve el resultado por la tubería."""
    _die_with_parent()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.environ.update(task.environment)
    handle = None
    try:
        handle = Path(task.lock).open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("El trabajo ya se ejecuta en otro proceso") from None
        from .masked_campaign import JobRun, Paused

        run = JobRun(**task.run, stop=EventStop(event))
        strict_fp32()
        if task.vram_bytes and _cuda_available():
            # El asignador recibe la VRAM del trabajo sin la del contexto de CUDA. Sin CUDA
            # no hay nada que acotar: el ejecutor exige `cuda:0` y falla por su cuenta.
            bound_vram(max(task.vram_bytes - CONTEXT_MIB * MIB, 256 * MIB))
        try:
            report = _resolve(task.executor)(run)
        except Paused:
            connection.send(("paused", None, _usage()))
            return
        connection.send(("completed", report, _usage()))
    except BaseException as error:  # Cualquier fallo se informa a la campaña.
        connection.send(
            ("failed", dict(type=type(error).__name__, message=str(error)), _usage())
            + (traceback.format_exc(),)
        )
    finally:
        connection.close()
        if handle is not None:
            handle.close()


def _usage():
    import resource

    usage = dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    try:
        import torch

        if torch.cuda.is_initialized():
            usage["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            usage["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    except ImportError:  # pragma: no cover (PyTorch siempre está en el entorno)
        pass
    return usage


def executor_name(function):
    """Nombre importable de un ejecutor, para reconstruirlo en el proceso hijo."""
    name = f"{function.__module__}:{function.__qualname__}"
    if "<" in name or _resolve(name) is not function:
        raise ValueError(f"El ejecutor {name} no se puede importar desde un proceso nuevo")
    return name


def run_fields(run):
    """Campos de un `JobRun` sin su parada, que el hijo sustituye por la de la campaña."""
    values = {item.name: getattr(run, item.name) for item in fields(run) if item.name != "stop"}
    return values


class SlotProcess:
    """Un trabajo GPU en su proceso. `poll` devuelve el resultado cuando termina."""

    def __init__(self, task, event):
        # El evento se conserva mientras viva el proceso: el hijo lo abre al arrancar.
        self.task, self.event = task, event
        self.receiver, sender = _CONTEXT.Pipe(duplex=False)
        self.process = _CONTEXT.Process(
            target=slot_main, args=(sender, task, event), name="mars-titan-slot", daemon=False
        )
        self.process.start()
        sender.close()
        self.result = None

    @property
    def sentinel(self):
        return self.receiver

    def poll(self):
        """`None` mientras corre. Después, `(estado, informe o error, uso, [traza])`."""
        if self.result is not None:
            return self.result
        # El estado del proceso se lee antes que la tubería. Si el hijo envía su resultado y
        # termina entre las dos lecturas, sigue vivo en esta vuelta y el resultado se recoge
        # en la siguiente, en lugar de darlo por perdido.
        alive = self.process.is_alive()
        if self.receiver.poll():
            try:
                self.result = self.receiver.recv()
            except EOFError:
                self.result = None
        if self.result is None and not alive:
            self.process.join()
            self.result = (
                "failed",
                dict(
                    type="ProcessExit",
                    message=f"El proceso del trabajo terminó sin resultado "
                    f"(código {self.process.exitcode})",
                ),
                {},
            )
        if self.result is not None:
            self.process.join()
            self.receiver.close()
        return self.result


def wait(handles, timeout):
    """Esperar a que termine alguno de `handles` o pase `timeout` segundos."""
    from multiprocessing.connection import wait as connection_wait

    return connection_wait([handle.sentinel for handle in handles], timeout)


def task_payload(task):
    return asdict(task)


def new_event():
    """Evento de parada compartido con los procesos de las ranuras."""
    return _CONTEXT.Event()


def slot_environment(execution, resources):
    """Tubería, hilos y límite de VRAM del proceso de un trabajo.

    `vram_bytes` es la VRAM de todo el proceso, con su contexto. Con MPS es también el
    límite del cliente, así que un trabajo que se pasa falla solo, sin tocar a los demás.
    El resto del entorno, como `CUBLAS_WORKSPACE_CONFIG`, se hereda de la campaña.
    """
    from .campaign_resources import environment

    values = environment(execution)
    values["MARS_TITAN_SLOT_VRAM_MIB"] = str(resources.vram_bytes // MIB)
    if execution.mps:
        values["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = f"0={resources.vram_bytes // MIB}M"
    return values


DESKTOP_TYPES = frozenset({"G", "C+G"})
COMPUTE_TYPES = frozenset({"C", "M", "M+C"})


def parse_gpu_processes(xml, mps):
    """VRAM libre y procesos de cómputo ajenos de la GPU 0 según `nvidia-smi -q -x`.

    Los procesos gráficos del escritorio («G» y «C+G») no cuentan como cargas de cómputo,
    igual que en `GpuLease`, aunque su memoria sí resta de la libre. Con MPS declarado se
    admite su servidor. Cualquier otro proceso «C», «M» o «M+C» es ajeno.
    """
    import re

    try:
        devices = ET.fromstring(xml).findall("gpu")
    except ET.ParseError as error:
        raise ValueError("El controlador no ha devuelto XML válido") from error
    if len(devices) != 1:
        raise ValueError("La consulta debe identificar una única GPU")
    free = devices[0].findtext("fb_memory_usage/free", "").strip()
    if not re.fullmatch(r"\d+ MiB", free):
        raise ValueError("La memoria libre de la GPU no está disponible en MiB")
    processes = devices[0].find("processes")
    if processes is None:
        raise ValueError("El controlador no informa de los procesos de GPU")
    foreign = []
    for process in processes.findall("process_info"):
        pid = process.findtext("pid", "").strip()
        kind = process.findtext("type", "").strip()
        name = process.findtext("process_name", "").strip()
        if not pid.isdecimal() or kind not in DESKTOP_TYPES | COMPUTE_TYPES:
            raise ValueError("El controlador no identifica un proceso de GPU")
        if kind in DESKTOP_TYPES or (mps and Path(name.split(" ")[0]).name == MPS_SERVER):
            continue
        foreign.append((int(pid), kind, name))
    return int(free.split()[0]) * MIB, foreign


def gpu_processes(mps):
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "-q", "-x", "-i", "0"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return parse_gpu_processes(result.stdout, mps)


def mps_ready(directory):
    """Comprobar que el demonio de control MPS responde en `directory`."""
    import subprocess

    result = subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="get_server_list\n",
        env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": directory},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and "Cannot find" not in result.stdout + result.stderr


class SlotLease:
    """Reserva de la GPU para una campaña con ranuras, sin abrir CUDA en su proceso.

    Usa el mismo cerrojo que `GpuLease`, así que excluye cualquier otra carga científica.
    Al empezar no admite procesos de cómputo ajenos, salvo el servidor MPS si se declara, y
    exige VRAM libre para el presupuesto declarado. Con MPS, exige el demonio de control.
    """

    def __init__(self, execution):
        self.execution, self.handle, self.record = execution, None, None

    def __enter__(self):
        from .campaign_resources import environment

        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/mars-titan-{os.getuid()}"))
        runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.handle = (runtime / "mars-titan-scientific-gpu.lock").open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free, foreign = gpu_processes(self.execution.mps)
            if foreign:
                raise RuntimeError(f"Hay otras cargas de cómputo activas en CUDA: {foreign}")
            budget = self.execution.vram_budget_bytes
            if budget is not None and free < budget:
                raise RuntimeError(
                    f"La GPU tiene {free // MIB} MiB libres y la ejecución declara {budget // MIB}"
                )
            if self.execution.mps:
                directory = environment(self.execution)["CUDA_MPS_PIPE_DIRECTORY"]
                if not mps_ready(directory):
                    raise RuntimeError(f"El demonio de control MPS no responde en {directory}")
            self.record = dict(
                device="cuda:0",
                slots=self.execution.gpu_slots,
                mps=self.execution.mps,
                free_vram_bytes=free,
                max_vram_bytes=budget,
            )
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_args):
        if self.handle:
            self.handle.close()
            self.handle = None
