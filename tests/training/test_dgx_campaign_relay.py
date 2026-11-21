"""Relevo de campañas de la DGX con campañas sustitutas: orden, avance, pausa y parada."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

RELAY = Path(__file__).resolve().parents[2] / "scripts/dgx_campaign_relay.sh"
STUB = """#!/usr/bin/env bash
# Campaña sustituta: anota sus argumentos y obedece a su archivo de conducta.
here="$(cd "$(dirname "$0")/.." && pwd)"
# La señal puede llegar en cuanto se anota la llamada: la trampa va antes.
trap 'echo parada >> "$here/llamadas.txt"; exit 0' TERM
echo "$*" >> "$here/llamadas.txt"
conducta="$(cat "$here/conducta" 2>/dev/null || echo bien)"
case "$conducta" in
  bien) exit 0 ;;
  parcial) exit 3 ;;
  rota) exit 1 ;;
  larga)
    for _ in $(seq 1 300); do sleep 0.1 & wait $!; [ -e "$here/termina" ] && exit 0; done
    exit 0 ;;
esac
"""


def copy(root, name, behaviour="bien"):
    folder = root / name
    (folder / "scripts").mkdir(parents=True)
    script = folder / "scripts/dgx_campaign.sh"
    script.write_text(STUB)
    script.chmod(0o755)
    (folder / "conducta").write_text(behaviour)
    return folder


def line(folder, phase):
    return f"{folder}|{folder}/estado|{folder}/vistas|declaracion.json|{phase}"


def calls(folder):
    path = folder / "llamadas.txt"
    return path.read_text().splitlines() if path.exists() else []


def environment(tmp_path, **changes):
    values = dict(
        os.environ,
        MARS_TITAN_RELAY_STATE=str(tmp_path / "relevo"),
        MARS_TITAN_RELAY_POLL="0.1",
        MARS_TITAN_RELAY_MIN_SECONDS="5",
    )
    values.update(changes)
    return values


def run(tmp_path, plan, **changes):
    return subprocess.run(
        ["bash", str(RELAY), str(plan)],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment(tmp_path, **changes),
    )


def wait_for(condition, seconds=20):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_the_relay_runs_each_phase_with_its_own_campaign_in_order_and_continues_later(tmp_path):
    us, joint = copy(tmp_path, "us"), copy(tmp_path, "conjunta", "parcial")
    plan = tmp_path / "plan.txt"
    plan.write_text(
        "# Bloque A\n\n"
        + line(us, "4 --seed 42 --origin 2013 --no-preflight")
        + "\n"
        + line(joint, "2 --seed 42 --origin 2013 --tier core --tier controls")
        + "\n"
    )
    done = run(tmp_path, plan)
    assert done.returncode == 0 and "plan terminado tras 2 fases" in done.stdout
    assert calls(us) == [
        f"run --machine mars_titan --declaration declaracion.json --state {us}/estado "
        f"--views {us}/vistas --workers 4 --seed 42 --origin 2013 --no-preflight"
    ]
    # Una fase con trabajos fallidos (código 3) también se anota y el plan sigue.
    assert calls(joint) == [
        f"run --machine mars_titan --declaration declaracion.json --state {joint}/estado "
        f"--views {joint}/vistas --workers 2 --seed 42 --origin 2013 --tier core --tier controls"
    ]
    assert Path(f"{plan}.hecho").read_text().strip() == "2"
    # El plan se amplía por el final: el relevo no repite lo terminado.
    with plan.open("a") as handle:
        handle.write(line(us, "4 --seed 42 --origin 2014 --no-preflight") + "\n")
    again = run(tmp_path, plan)
    assert again.returncode == 0 and len(calls(us)) == 2 and len(calls(joint)) == 1
    assert calls(us)[1].endswith("--workers 4 --seed 42 --origin 2014 --no-preflight")
    assert Path(f"{plan}.hecho").read_text().strip() == "3"


def test_a_phase_that_fails_at_once_stops_the_relay_without_consuming_the_plan(tmp_path):
    broken, good = copy(tmp_path, "rota", "rota"), copy(tmp_path, "bien")
    plan = tmp_path / "plan.txt"
    plan.write_text(line(broken, "4 --origin 2013") + "\n" + line(good, "4 --origin 2013") + "\n")
    done = run(tmp_path, plan)
    assert done.returncode == 1 and "falló al arrancar" in done.stdout
    assert not Path(f"{plan}.hecho").exists() and calls(good) == []
    # Una fase larga que acaba con error sí se anota: el plan sigue con la siguiente.
    late = run(tmp_path, plan, MARS_TITAN_RELAY_MIN_SECONDS="0")
    assert late.returncode == 0 and "se anota y el plan sigue" in late.stdout
    assert calls(good) != [] and Path(f"{plan}.hecho").read_text().strip() == "2"


@pytest.mark.parametrize(
    "text",
    [
        "{copia}|estado|vistas|declaracion.json|0 --origin 2013",
        "{copia}|estado|vistas|declaracion.json|9 --origin 2013",
        "{copia}|estado|vistas|declaracion.json|4",
        "{copia}|estado|vistas||4 --origin 2013",
        "{copia}|estado|4 --origin 2013",
        "/no/existe|estado|vistas|declaracion.json|4 --origin 2013",
    ],
)
def test_a_badly_written_phase_stops_the_relay(tmp_path, text):
    folder = copy(tmp_path, "copia")
    plan = tmp_path / "plan.txt"
    plan.write_text(text.format(copia=folder) + "\n")
    done = run(tmp_path, plan)
    assert done.returncode == 2 and "mal escrita" in done.stdout
    assert calls(folder) == [] and not Path(f"{plan}.hecho").exists()


def test_the_pause_stops_the_running_phase_and_repeats_it_when_lifted(tmp_path):
    folder = copy(tmp_path, "larga", "larga")
    plan = tmp_path / "plan.txt"
    plan.write_text(line(folder, "4 --origin 2013") + "\n")
    pause = tmp_path / "relevo/pausa"
    process = subprocess.Popen(
        ["bash", str(RELAY), str(plan)],
        stdout=subprocess.PIPE,
        text=True,
        env=environment(tmp_path),
    )
    try:
        assert wait_for(lambda: len(calls(folder)) == 1)
        pause.touch()
        assert wait_for(lambda: "parada" in calls(folder))
        # En pausa no se anota la fase ni se lanza otra.
        time.sleep(0.5)
        assert not Path(f"{plan}.hecho").exists() and len(calls(folder)) == 2
        pause.unlink()
        assert wait_for(lambda: len(calls(folder)) == 3)
        (folder / "termina").touch()
        assert process.wait(timeout=30) == 0
    finally:
        process.kill()
    output = process.stdout.read()
    assert "pausa pedida" in output and "en pausa" in output and "plan terminado" in output
    assert Path(f"{plan}.hecho").read_text().strip() == "1"


def test_a_stop_request_reaches_the_phase_and_keeps_it_for_the_next_start(tmp_path):
    folder = copy(tmp_path, "larga", "larga")
    plan = tmp_path / "plan.txt"
    plan.write_text(line(folder, "4 --origin 2013") + "\n")
    process = subprocess.Popen(
        ["bash", str(RELAY), str(plan)],
        stdout=subprocess.PIPE,
        text=True,
        env=environment(tmp_path),
    )
    try:
        assert wait_for(lambda: len(calls(folder)) == 1)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 0
    finally:
        process.kill()
    assert "parada pedida" in process.stdout.read()
    assert calls(folder)[-1] == "parada" and not Path(f"{plan}.hecho").exists()


def test_the_relay_needs_its_plan(tmp_path):
    missing = run(tmp_path, tmp_path / "no-existe.txt")
    assert missing.returncode == 2 and "Falta el plan" in missing.stderr
    usage = subprocess.run(["bash", str(RELAY)], capture_output=True, text=True, timeout=30)
    assert usage.returncode == 2 and "Uso:" in usage.stderr


def test_a_cleaning_line_removes_only_the_packed_copies_of_one_origin(tmp_path):
    folder = copy(tmp_path, "us")
    copies = tmp_path / "estado/packed-inputs"
    for year in ("2013", "2014"):
        (copies / year / "bloques").mkdir(parents=True)
        (copies / year / "bloques/a.bin").write_bytes(b"copia")
    (tmp_path / "estado/runs").mkdir()
    plan = tmp_path / "plan.txt"
    plan.write_text(
        line(folder, "4 --origin 2013")
        + f"\nlimpia|{copies}/2013\nlimpia|{copies}/2013\n"
        + line(folder, "4 --origin 2014")
        + "\n"
    )
    done = run(tmp_path, plan)
    assert done.returncode == 0 and "copias retiradas" in done.stdout
    # La segunda vez no hay nada que retirar y el plan sigue.
    assert "no hay copias" in done.stdout and len(calls(folder)) == 2
    assert not (copies / "2013").exists() and (copies / "2014/bloques/a.bin").exists()
    assert (tmp_path / "estado/runs").is_dir()
    assert Path(f"{plan}.hecho").read_text().strip() == "4"


@pytest.mark.parametrize(
    "target",
    [
        "{root}/estado/packed-inputs",
        "{root}/estado/runs",
        "{root}/estado/packed-inputs/2013/bloques",
        "{root}/estado/packed-inputs/../runs",
        "estado/packed-inputs/2013",
        "{root}/enlace",
        "",
    ],
)
def test_a_cleaning_line_refuses_any_other_folder(tmp_path, target):
    kept = tmp_path / "estado/packed-inputs/2013"
    kept.mkdir(parents=True)
    (kept / "a.bin").write_bytes(b"copia")
    (tmp_path / "estado/runs").mkdir()
    (tmp_path / "estado/runs/resultado.json").write_text("{}")
    (tmp_path / "enlace").symlink_to(kept, target_is_directory=True)
    plan = tmp_path / "plan.txt"
    plan.write_text("limpia|" + target.format(root=tmp_path) + "\n")
    done = run(tmp_path, plan)
    assert done.returncode == 2 and "mal escrita" in done.stdout
    assert (kept / "a.bin").exists() and (tmp_path / "estado/runs/resultado.json").exists()
    assert not Path(f"{plan}.hecho").exists()


PREPARER = """#!/usr/bin/env bash
# Intérprete sustituto: anota la orden de preparación y las rutas que recibe por el entorno.
trap 'echo parada >> llamadas.txt; exit 143' TERM
rutas="$MARS_TITAN_SHARED_INDICES | $MARS_TITAN_TRAIN_ARCHIVE | $MARS_TITAN_DIGEST_CACHE"
echo "$* | $rutas | $PYTHONPATH" >> llamadas.txt
conducta="$(cat conducta 2>/dev/null || echo bien)"
case "$conducta" in
  bien) exit 0 ;;
  rota) exit 1 ;;
  larga)
    for _ in $(seq 1 300); do sleep 0.1 & wait $!; [ -e termina ] && exit 0; done
    exit 0 ;;
esac
"""


def preparer(tmp_path, folder):
    """Copia con el preparador de entradas y un intérprete sustituto que anota su llamada."""
    (folder / "scripts/dgx_prepare_inputs.py").write_text("")
    interpreter = tmp_path / "python-sustituto"
    interpreter.write_text(PREPARER)
    interpreter.chmod(0o755)
    return dict(MARS_TITAN_DGX_PYTHON=str(interpreter), PYTHONPATH="")


def test_a_preparation_line_builds_the_inputs_of_one_origin_before_its_phase(tmp_path):
    folder = copy(tmp_path, "us")
    plan = tmp_path / "plan.txt"
    plan.write_text(
        f"prepara|{folder}|{folder}/estado|configs/receta.json|2023\n"
        + line(folder, "4 --origin 2023")
        + "\n"
    )
    done = run(tmp_path, plan, **preparer(tmp_path, folder))
    assert done.returncode == 0 and "plan terminado tras 2 fases" in done.stdout
    state = f"{folder}/estado"
    # Un solo proceso, con las rutas de índices, copias y huellas que usa la cola de ese estado.
    assert calls(folder) == [
        f"scripts/dgx_prepare_inputs.py origin --view {state}/cohorts/2023/manifest.json "
        f"--fold fold-018 --recipe configs/receta.json --archive | {state}/indices | "
        f"{state}/packed-inputs/2023 | {state}/file-digests.json | {folder}/src",
        f"run --machine mars_titan --declaration declaracion.json --state {state} "
        f"--views {folder}/vistas --workers 4 --origin 2023",
    ]
    assert Path(f"{plan}.hecho").read_text().strip() == "2"


def test_a_preparation_that_fails_at_once_stops_the_relay_and_a_late_failure_does_not(tmp_path):
    folder = copy(tmp_path, "us", "rota")
    plan = tmp_path / "plan.txt"
    plan.write_text(f"prepara|{folder}|{folder}/estado|configs/receta.json|2013\n")
    settings = preparer(tmp_path, folder)
    done = run(tmp_path, plan, **settings)
    assert done.returncode == 1 and "falló al arrancar" in done.stdout
    assert not Path(f"{plan}.hecho").exists()
    # Pasado el margen de arranque se anota: la fase siguiente construye lo que falte.
    late = run(tmp_path, plan, MARS_TITAN_RELAY_MIN_SECONDS="0", **settings)
    assert late.returncode == 0 and "se anota y el plan sigue" in late.stdout
    assert Path(f"{plan}.hecho").read_text().strip() == "1"


@pytest.mark.parametrize(
    "text",
    [
        "prepara|{copia}|{copia}/estado|configs/receta.json|23",
        "prepara|{copia}|{copia}/estado|configs/receta.json|2004",
        "prepara|{copia}|{copia}/estado|configs/receta.json|2013|otra",
        "prepara|{copia}|{copia}/estado||2013",
        "prepara|{copia}||configs/receta.json|2013",
        "prepara|{copia}/no-existe|{copia}/estado|configs/receta.json|2013",
        "prepara|{copia}|{copia}/estado|configs/receta.json",
    ],
)
def test_a_badly_written_preparation_stops_the_relay(tmp_path, text):
    folder = copy(tmp_path, "copia")
    plan = tmp_path / "plan.txt"
    plan.write_text(text.format(copia=folder) + "\n")
    done = run(tmp_path, plan, **preparer(tmp_path, folder))
    assert done.returncode == 2 and "mal escrita" in done.stdout
    assert calls(folder) == [] and not Path(f"{plan}.hecho").exists()


def test_the_pause_stops_a_preparation_and_repeats_it_when_lifted(tmp_path):
    folder = copy(tmp_path, "larga", "larga")
    plan = tmp_path / "plan.txt"
    plan.write_text(f"prepara|{folder}|{folder}/estado|configs/receta.json|2013\n")
    pause = tmp_path / "relevo/pausa"
    process = subprocess.Popen(
        ["bash", str(RELAY), str(plan)],
        stdout=subprocess.PIPE,
        text=True,
        env=environment(tmp_path, **preparer(tmp_path, folder)),
    )
    try:
        assert wait_for(lambda: len(calls(folder)) == 1)
        pause.touch()
        assert wait_for(lambda: "parada" in calls(folder))
        time.sleep(0.5)
        assert not Path(f"{plan}.hecho").exists() and len(calls(folder)) == 2
        pause.unlink()
        assert wait_for(lambda: len(calls(folder)) == 3)
        (folder / "termina").touch()
        assert process.wait(timeout=30) == 0
    finally:
        process.kill()
    assert "plan terminado tras 1 fases" in process.stdout.read()
    assert Path(f"{plan}.hecho").read_text().strip() == "1"
