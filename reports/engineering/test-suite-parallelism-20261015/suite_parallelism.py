"""Tiempo, memoria y resultados de la suite rápida de CPU en serie y con pytest-xdist (#491).

Uso: suite_parallelism.py SALIDA RONDAS CONFIGURACIÓN..., desde la raíz del árbol medido.

Las configuraciones son `serie` o `n<trabajadores>-<reparto>`, por ejemplo `n2-loadfile`.
Cada ejecución corre en su propio trabajo `memslot suite` con la selección y el entorno de
`make test-fast` (sin las pruebas lentas, nativas ni de CUDA, con CUDA_VISIBLE_DEVICES=-1 y
dos hilos), y el orden de las configuraciones rota en cada ronda. Dentro del scope de systemd
se mide el tiempo de reloj de pytest y, cada segundo, la memoria anónima del scope. Su máximo es
lo que ocupan los procesos. `memory.peak` incluye además la caché de página, que el núcleo deja
crecer hasta el límite de la plaza, así que se guarda solo como referencia. Al terminar se leen
los contadores de memory.events, que registran si el núcleo mató algún proceso por memoria. De
cada ejecución se guardan también el resultado de cada prueba según JUnit y las entradas de /tmp
que no existían al empezar. Otros programas del usuario también escriben allí, y pytest y
tempfile usan el TMPDIR propio de la plaza. SALIDA es un directorio con un JSON por ejecución y
el resumen.
"""

import json
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

MEMSLOT = "/home/gonzalo/.local/state/mars-titan/bin/memslot"
SELECTION = "not slow and not native and not cuda"
# Carpetas o archivos que recorre cada ejecución. Por defecto, toda la suite.
PATHS = os.environ.get("SUITE_PATHS", "tests").split()
# Plaza de memslot. Las medidas usan una de la suite, con su límite de 6 GiB.
SLOT = os.environ.get("SUITE_SLOT", "suite --max 6G").split()
# Mide dentro del scope: el tiempo de pytest sin la espera de plaza, el máximo de la memoria
# anónima del cgroup muestreada cada segundo, su pico con la caché y los avisos de memoria.
# Si el núcleo mata un proceso por memoria, systemd para el scope entero y no se llega al
# final, así que el máximo se guarda en MEASURE_FILE.anon cada vez que crece y el nombre del
# scope en MEASURE_FILE.scope, para leer después en el diario de systemd por qué se paró. El
# final de pytest se anota en MEASURE_FILE.end para no contar la espera del muestreo.
INNER = """
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
basename $cg > "$MEASURE_FILE.scope"
start=$(date +%s.%N)
( "$@"; code=$?; date +%s.%N > "$MEASURE_FILE.end"; exit $code ) &
pid=$!
anon=0
while kill -0 $pid 2>/dev/null; do
  now=$(awk '$1 == "anon" {print $2}' $cg/memory.stat)
  if ((now > anon)); then anon=$now; echo $anon > "$MEASURE_FILE.anon"; fi
  sleep 1
done
wait $pid
code=$?
end=$(cat "$MEASURE_FILE.end")
peak=$(cat $cg/memory.peak)
oom_kill=$(awk '$1 == "oom_kill" {print $2}' $cg/memory.events)
at_max=$(awk '$1 == "max" {print $2}' $cg/memory.events)
format='{"start": %s, "end": %s, "anon_peak_bytes": %s, "peak_bytes": %s,'
format+=' "oom_kill": %s, "max_events": %s, "exit": %s}\\n'
printf "$format" "$start" "$end" "$anon" "$peak" "$oom_kill" "$at_max" "$code" > "$MEASURE_FILE"
exit $code
"""


def command(config, junit):
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", SELECTION]
    base += [f"--junitxml={junit}", *PATHS]
    if config == "serie":
        return base
    workers, dist = config.removeprefix("n").split("-")
    return [*base, "-n", workers, "--dist", dist]


def outcomes(junit):
    """Resultado de cada prueba por clase y nombre, como lo registra JUnit."""
    result = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        key = f"{case.get('classname')}::{case.get('name')}"
        kinds = [child.tag for child in case if child.tag in {"failure", "error", "skipped"}]
        result[key] = kinds[0] if kinds else "passed"
    return result


def tmp_entries():
    """Entradas de primer nivel de /tmp del usuario. Otras sesiones también escriben allí."""
    names = set()
    for entry in os.scandir("/tmp"):
        try:
            if entry.stat(follow_symlinks=False).st_uid == os.getuid():
                names.add(entry.name)
        except FileNotFoundError:
            continue
    return names


def run(out, config, round_):
    name = f"{round_}-{config}"
    junit, measure = out / f"{name}.xml", out / f"{name}.measure.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env["MEASURE_FILE"] = str(measure)
    before = tmp_entries()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    load = Path("/proc/loadavg").read_text().split()[:3]
    inner = ["bash", "-c", INNER, "medida", *command(config, junit)]
    completed = subprocess.run(
        [MEMSLOT, *SLOT, "--", *inner],
        env=env,
        capture_output=True,
        text=True,
    )
    record = dict(
        config=config,
        round=round_,
        started=started,
        load_before=load,
        load_after=Path("/proc/loadavg").read_text().split()[:3],
        # Sin la ruta del intérprete, que es local.
        command=shlex.join(["python", *command(config, junit.name)[1:]]),
        returncode=completed.returncode,
        tail=(completed.stdout + completed.stderr).strip().splitlines()[-4:],
        new_tmp_entries=sorted(tmp_entries() - before),
    )
    partial, scope = (measure.with_name(measure.name + suffix) for suffix in (".anon", ".scope"))
    if measure.exists():
        measured = json.loads(measure.read_text())
        record["seconds"] = measured["end"] - measured["start"]
        for key in ("anon_peak_bytes", "peak_bytes", "oom_kill", "max_events", "exit"):
            record[key] = measured[key]
    elif partial.exists():
        # El scope se paró antes de terminar: queda el máximo anónimo hasta ese momento.
        record["anon_peak_bytes"] = int(partial.read_text())
        record["stopped_before_measure"] = True
        if scope.exists():
            journal = subprocess.run(
                ["journalctl", "--user", "-o", "cat", "-u", scope.read_text().strip()],
                capture_output=True,
                text=True,
            ).stdout
            record["stop_reason"] = [line for line in journal.splitlines() if "result" in line]
    for leftover in (partial, scope, measure.with_name(measure.name + ".end")):
        leftover.unlink(missing_ok=True)
    if junit.exists():
        record["outcomes"] = outcomes(junit)
    (out / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    counts = {}
    for kind in record.get("outcomes", {}).values():
        counts[kind] = counts.get(kind, 0) + 1
    seconds, peak = record.get("seconds"), record.get("anon_peak_bytes")
    print(name, seconds, peak, record["returncode"], counts, flush=True)
    return record


def main():
    out, rounds, configs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for round_ in range(rounds):
        # Rota el orden para que ninguna configuración vaya siempre primera tras una pausa.
        shift = round_ % len(configs)
        for config in configs[shift:] + configs[:shift]:
            records.append(run(out, config, round_))
    reference = records[0].get("outcomes", {})
    summary = dict(
        configs=configs,
        rounds=rounds,
        python=sys.version.split()[0],
        cpus=os.cpu_count(),
        runs=[{k: v for k, v in r.items() if k != "outcomes"} for r in records],
        same_outcomes=[r.get("outcomes") == reference for r in records],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
