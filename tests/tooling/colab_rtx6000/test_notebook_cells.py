"""Las celdas del cuaderno de Colab se ejecutan contra dobles y el kit admite sus órdenes.

Cada celda de bash entra en bash por la entrada estándar, como hace IPython con %%bash. Los
guiones del kit y las herramientas del sistema son dobles que anotan lo que reciben, y lo
anotado se compara con los pasos y las opciones que declaran los guiones reales. La parada se
prueba con una cola de mentira que lanza el colab_origin.sh real.

En esta máquina puede estar corriendo la campaña real: ningún doble reenvía pkill o pgrep sin
una sesión acotada, y esa sesión es siempre la que crea la prueba.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import notebook_validator as nv
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
ADDON = SCRIPTS / "colab_rtx6000"
SEP = "\x1f"
SEEN = "MARS_TITAN_SITE CUBLAS_WORKSPACE_CONFIG TRITON_F32_DEFAULT CUDA_MPS_PIPE_DIRECTORY"
PATTERN = "mars_titan.training.comparison_campaig[n] run"

# Anota su nombre, sus argumentos y las variables que le llegan.
RECORD = r"""#!/bin/bash
{ printf '%s' "$(basename "$0")"; for a in "$@"; do printf '\x1f%s' "$a"; done
  for name in @SEEN@; do printf '\x1f%s=%s' "env:$name" "${!name:-}"; done; echo; } >> "@LOG@"
"""
# Lanza la orden con un entorno vacío, como un servidor tmux que ya existía antes del cuaderno.
TMUX = r"""
[ "$1" = new-session ] || exit 1
while [ "$#" -gt 1 ]; do shift; done
@LAUNCH@
"""
# Solo reenvía una orden acotada a una sesión: aquí puede estar corriendo la campaña real.
BOUND = r"""#!/bin/bash
case " $* " in
  *" -s "[1-9]*) exec @REAL@ "$@" ;;
esac
echo "@NAME@ sin sesion acotada: $*" >> "@LOG@"
exit 99
"""
# El intérprete del entorno: anota, deja pasar lo que va a Python y responde por dos guiones.
PYTHON = r"""#!@REAL@
import json
import os
import sys
import time

args = sys.argv[1:]
with open("@LOG@", "a", encoding="utf-8") as log:
    log.write("\x1f".join(["python", *args]) + "\n")
if args[:1] in (["-"], ["-m"]):
    os.execv("@REAL@", ["@REAL@", *args])
name = os.path.basename(args[0]) if args else ""
if name == "mps_probe.py":
    on = os.environ.get("DOBLE_MPS") == "1"
    mps = {
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
        "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": "0=12288M",
    }
    strict = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TRITON_F32_DEFAULT": "ieee"}
    report = {
        "mps_operational": on,
        "recommended_workers": 4,
        "active_environment": {**strict, **(mps if on else {})},
    }
    with open(args[args.index("--output") + 1], "w", encoding="utf-8") as out:
        json.dump(report, out)
if name == "colab_sync.py" and "push" in args:
    with open("@LOG@", "a", encoding="utf-8") as log:
        log.write("\x1f".join(["copia", repr(time.time())]) + "\n")
    print(json.dumps({"held": [], "retried_later": []}))
"""
# Una cola de mentira: arranca, y al recibir la parada tarda en llegar a su barrera.
QUEUE = r"""import signal
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args[:1] != ["run"]:
    raise SystemExit(0)
state = Path(args[args.index("--state") + 1])
state.mkdir(parents=True, exist_ok=True)


def stop(*_):
    if (state / "sorda").exists():
        return
    time.sleep(0.6)
    (state / "barrera").write_text(repr(time.time()))
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
(state / "arrancada").write_text("")
while True:
    time.sleep(0.05)
"""


def cells():
    notebook = nv.build_bootstrap_notebook()
    return {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def record(log):
    return RECORD.replace("@SEEN@", SEEN).replace("@LOG@", str(log))


class Box:
    """Una máquina de mentira: carpetas con un espacio en el nombre y dobles delante en PATH."""

    def __init__(self, root, *, real_kit, background):
        self.log = root / "calls.log"
        self.work = root / "mars titan"
        self.kit = SCRIPTS if real_kit else root / "kit" / "scripts"
        self.state = self.work / "estado" / "v11"
        stubs = root / "dobles"
        for folder in (self.work, root / "espejo", root / "paquete", stubs):
            folder.mkdir(parents=True, exist_ok=True)
        launch = 'env -i PATH="$PATH" HOME="$HOME" @SETSID@ bash -c "$1"'
        launch = launch.replace("@SETSID@", "setsid" if background else "")
        launch += " >/dev/null 2>&1 &" if background else ""
        executable(stubs / "tmux", record(self.log) + TMUX.replace("@LAUNCH@", launch))
        for name in ("git", "nvidia-cuda-mps-control"):
            executable(stubs / name, record(self.log))
        for name, code in (("pkill", 0), ("pgrep", 1)):
            real = shutil.which(name)
            body = (
                BOUND.replace("@REAL@", real)
                .replace("@NAME@", name)
                .replace("@LOG@", str(self.log))
            )
            executable(stubs / name, body if background else record(self.log) + f"exit {code}\n")
        python = PYTHON.replace("@REAL@", sys.executable).replace("@LOG@", str(self.log))
        executable(self.work / "repo" / ".venv" / "bin" / "python", python)
        if not real_kit:
            for name in ("colab_bootstrap.sh", "colab_origin.sh"):
                executable(self.kit / name, record(self.log))
        declared = dict(
            re.findall(r"^os\.environ\['([A-Z_]+)'\] = '([^']*)'", cells()["step-00-config"], re.M)
        )
        keep = {k: v for k, v in os.environ.items() if not k.startswith(("MARS_TITAN", "CUDA_MPS"))}
        self.env = {
            **keep,
            **declared,
            "T": str(self.work),
            "K": str(self.kit),
            "ADDON": str(self.kit / "colab_rtx6000"),
            "P": str(root / "paquete"),
            "B": str(root / "paquete" / "mars-titan-colab.bundle"),
            "ESTADO": str(self.state),
            "VISTAS": "/vistas/US",
            "MIRROR_DRIVE": str(root / "espejo"),
            "PATH": f"{stubs}:{os.environ['PATH']}",
        }

    def run(self, name, **changes):
        source = cells()[name]
        assert source.startswith("%%bash\n"), name
        return subprocess.run(
            ["bash"],
            input=source.split("\n", 1)[1],
            env={**self.env, **changes},
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self, name=None):
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        found = [line.split(SEP) for line in lines]
        return [call for call in found if name is None or call[0] == name]


def arguments(call):
    return [item for item in call[1:] if not item.startswith("env:")]


def environment(call):
    return dict(item[4:].split("=", 1) for item in call[1:] if item.startswith("env:"))


def shell_surface(path):
    text = path.read_text(encoding="utf-8")
    steps = set(re.findall(r"^  ([a-z-]+)\) paso_", text, re.M))
    return steps, set(re.findall(r"^\s*(--[a-z-]+)\)", text, re.M))


def python_surface(path):
    text = path.read_text(encoding="utf-8")
    commands = set(re.findall(r"add_parser\(\s*\"([a-z_-]+)\"", text))
    return commands, set(re.findall(r"add_argument\(\s*\"(--[a-z0-9-]+)\"", text))


@pytest.mark.parametrize("mps", ["1", "0"])
def test_every_bash_cell_runs_and_gives_orders_the_kit_accepts(tmp_path, mps):
    box = Box(tmp_path, real_kit=False, background=False)
    ran = []
    for name, source in cells().items():
        if source.startswith("%%bash"):
            result = box.run(name, DOBLE_MPS=mps)
            assert result.returncode == 0, (name, result.stderr[-1500:])
            ran.append(name)
    assert len(ran) >= 7
    checked = set()
    for call in box.calls():
        name, args = call[0], arguments(call)
        if name in ("colab_bootstrap.sh", "colab_origin.sh"):
            steps, options = shell_surface(SCRIPTS / name)
            assert args[0] in steps, call
            assert {item for item in args[1:] if item.startswith("--")} <= options, call
            checked.add(name)
        if name == "python" and args[0] not in ("-", "-m"):
            script = Path(args[0]).name
            real = SCRIPTS / script if (SCRIPTS / script).is_file() else ADDON / script
            commands, options = python_surface(real)
            assert not commands or args[1] in commands, call
            assert {item for item in args[1:] if item.startswith("--")} <= options, call
            checked.add(script)
    assert checked >= {
        "colab_bootstrap.sh",
        "colab_origin.sh",
        "colab_site.py",
        "colab_sync.py",
        "kernel_sm120_plan.py",
        "mps_probe.py",
        "concurrency_probe.py",
        "cu_budget_guard.py",
    }


@pytest.mark.parametrize(("mps", "pipe"), [("1", "/tmp/nvidia-mps"), ("0", "")])
def test_the_queue_gets_whole_arguments_and_its_own_environment(tmp_path, mps, pipe):
    box = Box(tmp_path, real_kit=False, background=False)
    for name in ("step-03-mps-and-probe", "step-04-origin-and-sync", "step-04b-launch"):
        result = box.run(name, DOBLE_MPS=mps)
        assert result.returncode == 0, (name, result.stderr[-1500:])
    (launch,) = [call for call in box.calls("colab_origin.sh") if call[1] == "lanza"]
    args = arguments(launch)
    assert args[args.index("--repo") + 1] == str(box.work / "repo")
    assert args[args.index("--estado") + 1] == str(box.state)
    assert args[args.index("--semillas") + 1] == "42 43 44"
    assert args[args.index("--trabajadores") + 1] == "4"
    # El tmux de mentira no hereda nada: lo que llega lo escribió el lanzador.
    assert environment(launch) == {
        "MARS_TITAN_SITE": "google-colab-rtx-pro-6000",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TRITON_F32_DEFAULT": "ieee",
        "CUDA_MPS_PIPE_DIRECTORY": pipe,
    }
    script = (box.work / "lanza_campana.sh").read_text(encoding="utf-8")
    assert "PINNED_DEVICE_MEM_LIMIT" not in script
    assert (box.work / "campana.sid").read_text().strip().isdecimal()


@pytest.mark.parametrize("zip_path", ["", "/content/drive/MyDrive/factor cn.zip"])
def test_the_factor_of_china_is_restored_only_when_its_zip_is_given(tmp_path, zip_path):
    box = Box(tmp_path, real_kit=False, background=False)
    result = box.run("step-01-bootstrap", FACTOR_CN_ZIP=zip_path)
    assert result.returncode == 0, result.stderr[-1500:]
    (data,) = [call for call in box.calls("colab_bootstrap.sh") if call[1] == "datos"]
    args = arguments(data)
    assert ("--factor-cn" in args) == bool(zip_path)
    if zip_path:
        assert args[args.index("--factor-cn") + 1] == zip_path
    (clone,) = [call for call in box.calls("git") if call[1] == "clone"]
    assert arguments(clone)[arguments(clone).index("--branch") + 1] == "dgx-trainning"


@pytest.mark.parametrize("mps", ["1", "0"])
def test_the_stop_is_bound_to_the_session_of_the_queue(tmp_path, mps):
    box = Box(tmp_path, real_kit=False, background=False)
    for name in ("step-03-mps-and-probe", "step-04b-launch", "step-06-safe-shutdown"):
        result = box.run(name, DOBLE_MPS=mps)
        assert result.returncode == 0, (name, result.stderr[-1500:])
    session = str(os.getsid(0))
    (stop,) = box.calls("pkill")
    assert arguments(stop) == ["-TERM", "-s", session, "-f", PATTERN]
    assert all(arguments(call) == ["-s", session, "-f", PATTERN] for call in box.calls("pgrep"))
    assert json.loads((box.work / "final_sync.json").read_text()) == {
        "held": [],
        "retried_later": [],
    }
    quits = box.calls("nvidia-cuda-mps-control")
    assert len(quits) == (1 if mps == "1" else 0)
    assert all(environment(call)["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/nvidia-mps" for call in quits)


def test_without_a_queue_of_this_notebook_nothing_is_signalled(tmp_path):
    box = Box(tmp_path, real_kit=False, background=False)
    result = box.run("step-06-safe-shutdown")
    assert result.returncode == 0, result.stderr[-1500:]
    assert box.calls("pkill") == [] and box.calls("pgrep") == []
    (box.work / "campana.sid").write_text("0\n")
    result = box.run("step-06-safe-shutdown")
    assert result.returncode == 1 and "no válida" in result.stderr
    assert box.calls("pkill") == [] and not (box.work / "final_sync.json").exists()


@pytest.fixture
def queue(tmp_path):
    """La cola de mentira, lanzada por el colab_origin.sh real desde la celda del cuaderno."""
    box = Box(tmp_path, real_kit=True, background=True)
    repo = box.work / "repo"
    package = repo / "src" / "mars_titan" / "training"
    package.mkdir(parents=True)
    (repo / "src" / "mars_titan" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "comparison_campaign.py").write_text(QUEUE)
    (repo / "scripts").mkdir()
    shutil.copy(SCRIPTS / "colab_campaign.sh", repo / "scripts" / "colab_campaign.sh")
    (repo / "configs").mkdir()
    (repo / "configs" / "campaign.json").write_text("{}")
    box.env.update(DECLARACION="configs/campaign.json", VISTAS="/vistas/US", SEMILLAS="42 43")
    box.env.update(ESPERA_PARADA_S="20")
    result = box.run("step-04b-launch")
    assert result.returncode == 0, result.stderr[-1500:]
    deadline = time.time() + 20
    while not (box.state / "arrancada").exists() and time.time() < deadline:
        time.sleep(0.05)
    assert (box.state / "arrancada").exists(), box.calls()
    session = (box.work / "campana.sid").read_text().strip()
    assert session.isdecimal() and int(session) not in (0, os.getsid(0))
    try:
        yield box, session
    finally:
        subprocess.run([shutil.which("pkill"), "-KILL", "-s", session], check=False)


def alive(session):
    found = subprocess.run(
        [shutil.which("pgrep"), "-s", session, "-f", PATTERN], capture_output=True, check=False
    )
    return found.returncode == 0


def test_the_stop_reaches_the_queue_and_the_copy_waits_for_its_barrier(queue):
    box, session = queue
    assert alive(session)
    result = box.run("step-06-safe-shutdown")
    assert result.returncode == 0, result.stderr[-1500:]
    assert not alive(session)
    barrier = float((box.state / "barrera").read_text())
    (copied,) = box.calls("copia")
    assert barrier <= float(copied[1])
    assert json.loads((box.work / "final_sync.json").read_text())["held"] == []
    assert not (box.work / "campana.sid").exists()
    assert "sin sesion acotada" not in box.log.read_text()


def test_a_queue_that_does_not_stop_is_neither_copied_nor_released(queue):
    box, session = queue
    (box.state / "sorda").write_text("")
    (box.work / "final_sync.json").write_text('{"held": [], "retried_later": []}')
    result = box.run("step-06-safe-shutdown", ESPERA_PARADA_S="2")
    assert result.returncode == 1 and "sigue viva" in result.stderr
    assert alive(session)
    assert box.calls("copia") == []
    assert not (box.work / "final_sync.json").exists()
    assert (box.work / "campana.sid").read_text().strip() == session


@pytest.mark.parametrize(
    ("report", "released"),
    [
        ({"held": [], "retried_later": []}, True),
        ({"held": ["fit/run.json"], "retried_later": []}, False),
        ({"held": [], "retried_later": ["fit/run.json"]}, False),
        (None, False),
    ],
)
def test_the_machine_is_released_only_after_a_clean_copy(tmp_path, monkeypatch, report, released):
    work = tmp_path / "otro sitio"
    work.mkdir()
    if report is not None:
        (work / "final_sync.json").write_text(json.dumps(report))
    calls = []
    colab = types.ModuleType("google.colab")
    colab.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("T", str(work))
    source = cells()["step-06b-unassign"]
    exec(compile(source, "step-06b-unassign", "exec"), {})
    assert calls == [], "sin pedirlo no se suelta la máquina"
    assert "AUTO_UNASSIGN = False" in source
    exec(compile(source.replace("AUTO_UNASSIGN = False", "AUTO_UNASSIGN = True"), "x", "exec"), {})
    assert calls == (["unassign"] if released else [])
