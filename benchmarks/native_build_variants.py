"""Compilación y recorrido de las variantes nativas de la etapa RL, sin pasos de optimizador.

Cinco órdenes, todas con salida JSON:

- `build` configura y compila un preset en un directorio propio y registra los tiempos, el pico
  de memoria de los procesos hijos, las estadísticas de ccache si se usa, la identidad
  compilada y la huella de cada ejecutable, biblioteca y objeto.
- `relink` borra solo las salidas enlazadas de un directorio ya compilado y vuelve a enlazar,
  de modo que el tiempo medido es el del enlazador.
- `touch` actualiza la fecha de una cabecera propia sin cambiar su contenido y recompila, como
  al volver a una rama.
- `run` recorre `mars-titan-policy-benchmark` de cada variante sobre las mismas cintas, en
  rondas con el orden rotado, y exige que todas den las mismas huellas de contenido.
- `compare` clasifica los objetos y salidas enlazadas que difieren entre dos compilaciones:
  solo por la huella de identidad compilada o por su contenido.

La medida no cambia la configuración de seguridad ni el perfil de energía, y el binario del
recorrido comprueba al terminar que los parámetros conservan su huella y que no hubo pasos de
optimizador.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mars_titan.hardware.platform_identity import cpu_name

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native"
# Fases del recorrido que se resumen. Cada ruta baja por el informe de
# `mars-titan-policy-benchmark`. Un entero elige un elemento de una lista y una pareja
# (campo, valor) elige el elemento de la lista con ese valor.
PHASES = {
    "environment_step_p50_us": ("environment_step", "p50_us"),
    "ppo_collection_tick_p50_us": ("ppo_trainer_collection", "p50_us"),
    "ppo_first_tick_p50_us": ("ppo_trainer_first_tick", "p50_us"),
    "ppo_forward_backward_p50_us": ("ppo_minibatch", "forward_backward", "p50_us"),
    "ppo_act_p50_us": ("inference", ("architecture", "ppo_mlp64"), "act_and_copy_back", "p50_us"),
    "klpo_tick_p50_us": ("klpo_wave", "collection_tick", "p50_us"),
    "klpo_wave_seconds": ("klpo_wave", "collection_wave_seconds"),
    "klpo_objective_seconds": ("klpo_wave", "objective_forward_backward_seconds"),
    "gae_fp64_p50_us": ("gae_cpu_fp64", "p50_us"),
    "grpo_block_p50_us": ("group_objectives", "grpo_outcome_v1", "p50_us"),
    "klpo_terminal_block_p50_us": ("group_objectives", "klpo_terminal_token_full_v1", "p50_us"),
    "qr_dqn_loss_p50_us": (
        "value_minibatch",
        ("architecture", "qr_dqn_mlp64"),
        "loss_forward_backward",
        "p50_us",
    ),
    "evaluation_episode_p50_us": ("evaluation_single_lane", "p50_us"),
    "total_seconds": ("total_seconds",),
    "peak_rss_bytes": ("executable_peak_rss_bytes",),
}
LINK_RULE = re.compile(r"_(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)_LINKER_")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, *, cwd=None, env=None, capture=False):
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    seconds = time.monotonic() - started
    if result.returncode != 0:
        tail = (result.stdout or "")[-4000:] if capture else ""
        raise SystemExit(f"Falló {' '.join(map(str, command))} ({result.returncode})\n{tail}")
    return seconds, result.stdout or ""


def children_peak_kib():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def ccache_stats(env):
    if shutil.which("ccache") is None:
        return None
    _, text = run(["ccache", "--print-stats"], env=env, capture=True)
    stats = {}
    for line in text.splitlines():
        key, _, value = line.partition("\t")
        if value.isdigit():
            stats[key] = int(value)
    return {
        key: stats.get(key, 0)
        for key in (
            "direct_cache_hit",
            "preprocessed_cache_hit",
            "cache_miss",
            "files_in_cache",
            "cache_size_kibibyte",
            "uncacheable",
            "compiler_check_failed",
        )
    }


def ninja_targets(build_dir):
    _, text = run(["ninja", "-C", str(build_dir), "-t", "targets", "all"], capture=True)
    targets = {}
    for line in text.splitlines():
        name, _, rule = line.rpartition(": ")
        if name:
            targets[name] = rule
    return targets


def artifacts(build_dir):
    """Huellas de las salidas enlazadas y de los objetos, con rutas relativas al directorio."""
    linked, objects = {}, {}
    for name, rule in sorted(ninja_targets(build_dir).items()):
        path = build_dir / name
        if not path.is_file():
            continue
        if LINK_RULE.search(rule):
            linked[name] = sha256(path)
        elif name.endswith(".o"):
            objects[name] = sha256(path)
    return linked, objects


def identity(build_dir):
    files = sorted(build_dir.glob("native-build-identity-*.txt"))
    if not files:
        return None
    lines = files[0].read_text(encoding="utf-8").splitlines()
    fields = {}
    for line in lines[1:]:
        key, _, value = line.partition("=")
        fields[re.sub(r"\[\d+\]$", "", key)] = value
    return {
        "sha256": lines[0].removeprefix("sha256="),
        "compiler_launcher": fields.get("CMAKE_CXX_COMPILER_LAUNCHER", ""),
        "linker_type": fields.get("CMAKE_LINKER_TYPE", ""),
        "ipo": fields.get("MARS_TITAN_ENABLE_IPO", ""),
        "pgo": fields.get("MARS_TITAN_PGO", ""),
    }


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def temperature():
    value = read_text("/sys/class/thermal/thermal_zone0/temp")
    return None if value is None else int(value) / 1000


def memory_record():
    """Memoria total y disponible del equipo según /proc/meminfo, en bytes."""
    text = read_text("/proc/meminfo") or ""
    values = {
        key: int(value) * 1024
        for key, value in re.findall(r"^(MemTotal|MemAvailable):\s+(\d+) kB$", text, flags=re.M)
    }
    return {"total_bytes": values.get("MemTotal"), "available_bytes": values.get("MemAvailable")}


GPU_FIELDS = ("name", "driver_version", "memory.total", "memory.used", "temperature.gpu")


def gpu_record():
    """Estado de la GPU con nvidia-smi, o None si no está disponible."""
    if not shutil.which("nvidia-smi"):
        return None
    query = ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader"]
    result = subprocess.run(query, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    values = [value.strip() for value in result.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(GPU_FIELDS, values, strict=False))


def environment_record():
    tools = {}
    for name, args in (
        ("clang++", ["--version"]),
        ("cmake", ["--version"]),
        ("ninja", ["--version"]),
        ("ccache", ["--version"]),
        ("mold", ["--version"]),
        ("llvm-profdata-21", ["--version"]),
    ):
        if shutil.which(name):
            _, text = run([name, *args], capture=True)
            tools[name] = text.strip().splitlines()[0] if text.strip() else ""
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform_profile": read_text("/sys/firmware/acpi/platform_profile"),
        "thermal_zone0_celsius": temperature(),
        "machine": platform.machine(),
        "cpu_model": cpu_name(),
        "kernel": platform.release(),
        "cpus": os.cpu_count(),
        "load_average": os.getloadavg(),
        "memory": memory_record(),
        "gpu": gpu_record(),
        "tools": tools,
        "ccache_dir": os.environ.get("CCACHE_DIR", ""),
        "ccache_basedir": os.environ.get("CCACHE_BASEDIR", ""),
    }


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def command_build(args):
    build_dir = args.build_dir.resolve()
    if build_dir.exists() and not args.keep:
        shutil.rmtree(build_dir)
    defines = [f"-D{item}" for item in args.define]
    if args.launcher:
        defines += [
            f"-DCMAKE_C_COMPILER_LAUNCHER={args.launcher}",
            f"-DCMAKE_CXX_COMPILER_LAUNCHER={args.launcher}",
        ]
    if args.linker:
        defines.append(f"-DCMAKE_LINKER_TYPE={args.linker}")
    if args.ipo:
        defines.append("-DMARS_TITAN_ENABLE_IPO=ON")
    if args.pgo != "off":
        defines += [f"-DMARS_TITAN_PGO={args.pgo}", f"-DMARS_TITAN_PGO_DATA={args.pgo_data}"]
    env = dict(os.environ)
    if args.launcher:
        run(["ccache", "--zero-stats"], env=env, capture=True)
    started = environment_record()
    before = children_peak_kib()
    configure, _ = run(
        ["cmake", "--preset", args.preset, "-B", str(build_dir), *defines],
        cwd=NATIVE,
        env=env,
        capture=True,
    )
    build, log = run(
        ["cmake", "--build", str(build_dir), "-j", str(args.jobs)],
        cwd=NATIVE,
        env=env,
        capture=True,
    )
    linked, objects = artifacts(build_dir)
    payload = {
        "kind": "native_build_variant",
        "label": args.label,
        "preset": args.preset,
        "defines": defines,
        "jobs": args.jobs,
        "configure_seconds": configure,
        "build_seconds": build,
        "children_peak_rss_kib": max(children_peak_kib(), before),
        "edges": len(re.findall(r"^\[\d+/\d+\]", log, flags=re.M)),
        "ccache": ccache_stats(env) if args.launcher else None,
        "identity": identity(build_dir),
        "linked_sha256": linked,
        "objects_sha256": objects,
        "started": started,
        "environment": environment_record(),
    }
    write(args.output, payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("label", "configure_seconds", "build_seconds", "edges", "ccache")
            },
            ensure_ascii=False,
        )
    )


def command_relink(args):
    build_dir = args.build_dir.resolve()
    # Si cambió un archivo de configuración, ninja regenera build.ninja antes de nada. Se
    # hace fuera de la medida y después el directorio tiene que estar al día.
    run(["ninja", "-C", str(build_dir), "build.ninja"], capture=True)
    _, pending = run(["ninja", "-C", str(build_dir), "-n"], capture=True)
    if re.search(r"^\[\d+/\d+\]", pending, flags=re.M):
        raise SystemExit("El directorio tiene pasos pendientes antes del reenlace")
    targets = ninja_targets(build_dir)
    removed = [
        name
        for name, rule in targets.items()
        if LINK_RULE.search(rule) and (build_dir / name).is_file()
    ]
    started = environment_record()
    for name in removed:
        (build_dir / name).unlink()
    _, plan = run(["ninja", "-C", str(build_dir), "-n"], capture=True)
    planned = len(re.findall(r"^\[\d+/\d+\]", plan, flags=re.M))
    seconds, log = run(["ninja", "-C", str(build_dir), "-j", str(args.jobs)], capture=True)
    linked, _ = artifacts(build_dir)
    payload = {
        "kind": "native_relink",
        "label": args.label,
        "removed": len(removed),
        "planned_edges": planned,
        "link_seconds": seconds,
        "identity": identity(build_dir),
        "linked_sha256": linked,
        "started": started,
        "environment": environment_record(),
    }
    if planned != len(removed):
        raise SystemExit(f"El reenlace planificó {planned} pasos para {len(removed)} salidas")
    write(args.output, payload)
    print(json.dumps({"label": args.label, "removed": len(removed), "link_seconds": seconds}))


def command_touch(args):
    build_dir = args.build_dir.resolve()
    header = (ROOT / args.header).resolve()
    content = header.read_bytes()
    os.utime(header)
    if header.read_bytes() != content:
        raise SystemExit("La cabecera cambió de contenido")
    if args.launcher:
        run(["ccache", "--zero-stats"], capture=True)
    started = environment_record()
    seconds, log = run(["ninja", "-C", str(build_dir), "-j", str(args.jobs)], capture=True)
    payload = {
        "kind": "native_touch_rebuild",
        "started": started,
        "label": args.label,
        "header": args.header,
        "edges": len(re.findall(r"^\[\d+/\d+\]", log, flags=re.M)),
        "seconds": seconds,
        "ccache": ccache_stats(dict(os.environ)) if args.launcher else None,
        "environment": environment_record(),
    }
    write(args.output, payload)
    print(json.dumps({k: payload[k] for k in ("label", "edges", "seconds", "ccache")}))


def tape_arguments(tapes, market):
    folders = sorted(path for path in (tapes / market).iterdir() if path.is_dir())
    train = [path for path in folders if "-train-" in path.name]
    validation = [path for path in folders if "-validation-" in path.name]
    if len(train) != 3 or len(validation) != 1:
        raise SystemExit(f"Las cintas de {market} no tienen tres tramos de ajuste y una validación")
    arguments = []
    for path in train:
        arguments += ["--train-tape", str(path)]
    return [*arguments, "--validation-tape", str(validation[0])]


def digests(report):
    """Huellas de contenido del informe del recorrido, sin la identidad del binario."""
    found = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    key.endswith("_sha256")
                    and isinstance(value, str)
                    and "build" not in key
                    and "source" not in key
                ):
                    found[f"{path}{key}"] = value
                else:
                    walk(value, f"{path}{key}.")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}{index}.")

    walk(report, "")
    return found


def lookup(report, path):
    node = report
    for step in path:
        if isinstance(step, tuple) and isinstance(node, list):
            field, value = step
            node = next((item for item in node if item.get(field) == value), None)
        elif isinstance(step, str) and isinstance(node, dict):
            node = node.get(step)
        else:
            return None
    return node if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def phases(report):
    return {name: lookup(report, path) for name, path in PHASES.items()}


def command_run(args):
    variants = {}
    for item in args.variant:
        label, _, folder = item.partition("=")
        binary = Path(folder).resolve() / "mars-titan-policy-benchmark"
        if not binary.is_file():
            raise SystemExit(f"Falta {binary}")
        variants[label] = binary
    tapes = tape_arguments(args.tapes.resolve(), args.market)
    work = args.output.with_suffix("")
    rounds = []
    labels = list(variants)
    for round_index in range(args.rounds):
        order = labels[round_index % len(labels) :] + labels[: round_index % len(labels)]
        for label in order:
            output = work / f"{label}-{round_index}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(variants[label]),
                *tapes,
                "--device",
                args.device,
                "--output",
                str(output),
                "--ticks",
                str(args.ticks),
                "--warmup",
                str(args.warmup),
            ]
            for phase in args.skip:
                command += ["--skip", phase]
            started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            seconds, _ = run(command, capture=True)
            report = json.loads(output.read_text(encoding="utf-8"))
            rounds.append(
                {
                    "round": round_index,
                    "label": label,
                    "started": started,
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "thermal_zone0_celsius": temperature(),
                    "process_seconds": seconds,
                    "load_average": os.getloadavg(),
                    "phases": phases(report),
                    "digests": digests(report),
                }
            )
    reference = {label: None for label in labels}
    for entry in rounds:
        if reference[entry["label"]] is None:
            reference[entry["label"]] = entry["digests"]
        elif reference[entry["label"]] != entry["digests"]:
            raise SystemExit(f"Las rondas de {entry['label']} no dan las mismas huellas")
    first = reference[labels[0]]
    parity = {label: digests_ == first for label, digests_ in reference.items()}
    summary = {}
    for label in labels:
        entries = [entry for entry in rounds if entry["label"] == label]
        keys = sorted(
            {
                key
                for entry in entries
                for key, value in entry["phases"].items()
                if isinstance(value, (int, float))
            }
        )
        summary[label] = {
            key: {
                "median": statistics.median(entry["phases"][key] for entry in entries),
                "min": min(entry["phases"][key] for entry in entries),
                "max": max(entry["phases"][key] for entry in entries),
            }
            for key in keys
        }
        summary[label]["process_seconds"] = statistics.median(e["process_seconds"] for e in entries)
    payload = {
        "kind": "native_variant_runtime",
        "market": args.market,
        "device": args.device,
        "ticks": args.ticks,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "skip": args.skip,
        "variants": {label: str(path) for label, path in variants.items()},
        "digest_parity": parity,
        "digests": first,
        "summary": summary,
        "runs": rounds,
        "environment": environment_record(),
    }
    write(args.output, payload)
    print(json.dumps({"parity": parity, "summary": summary}, ensure_ascii=False))
    if not all(parity.values()):
        raise SystemExit("Las variantes no dan las mismas huellas de contenido")


def masked(path, digest):
    """Bytes del archivo con la huella de identidad compilada sustituida por ceros."""
    data = path.read_bytes()
    return data.replace(digest.encode(), b"0" * len(digest)), data.count(digest.encode())


SECTION = re.compile(r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)")
# Secciones que describen el enlace dinámico y no el cálculo: la RUNPATH lleva el directorio
# de compilación y build-id resume todo el archivo.
LINK_METADATA = {".note.gnu.build-id", ".dynstr", ".dynamic", ".gnu.version", ".gnu.version_r"}


def differing_sections(left_path, left, right_path, right):
    """Secciones con contenido distinto entre dos ELF ya enmascarados, o None si no casan.

    Con LTO los objetos son bitcode de LLVM y no tienen tabla de secciones, así que tampoco
    se pueden comparar por secciones.
    """
    if not (left.startswith(b"\x7fELF") and right.startswith(b"\x7fELF")):
        return None

    def table(path):
        text = subprocess.run(
            ["readelf", "-S", "-W", str(path)], capture_output=True, text=True, check=True
        ).stdout
        rows = {}
        for line in text.splitlines():
            match = SECTION.match(line)
            if match and match.group(2) != "NOBITS":
                rows[match.group(1)] = (int(match.group(3), 16), int(match.group(4), 16))
        return rows

    first, second = table(left_path), table(right_path)
    if set(first) != set(second):
        return None
    return sorted(
        name
        for name in first
        if left[first[name][0] : first[name][0] + first[name][1]]
        != right[second[name][0] : second[name][0] + second[name][1]]
    )


def command_compare(args):
    """Clasificar cada objeto y salida enlazada que difiere entre dos compilaciones.

    `identity_only` significa que los bytes coinciden al sustituir en cada lado la huella de
    su identidad (`MARS_TITAN_NATIVE_BUILD_SHA256`) por ceros, así que el código es el mismo.
    `identity_and_link_metadata` añade diferencias solo en las secciones del enlace dinámico
    (`LINK_METADATA`), como la RUNPATH con otro directorio de compilación o el build-id.
    `content` es cualquier otra diferencia, como la disposición de otro enlazador.
    """
    receipts, folders = [], []
    for item in (args.first, args.second):
        receipt_path, _, folder = item.partition("=")
        receipts.append(json.loads(Path(receipt_path).read_text(encoding="utf-8")))
        folders.append(Path(folder).resolve())
    digests_ = [receipt["identity"]["sha256"] for receipt in receipts]
    result = {}
    for kind in ("objects_sha256", "linked_sha256"):
        first, second = receipts[0][kind], receipts[1][kind]
        if set(first) != set(second):
            raise SystemExit(f"Las dos compilaciones no tienen las mismas salidas en {kind}")
        rows = {}
        for name in sorted(first):
            if first[name] == second[name]:
                continue
            left, left_count = masked(folders[0] / name, digests_[0])
            right, right_count = masked(folders[1] / name, digests_[1])
            sections = differing_sections(folders[0] / name, left, folders[1] / name, right)
            if left == right:
                kind_ = "identity_only"
            elif sections is not None and set(sections) <= LINK_METADATA:
                kind_ = "identity_and_link_metadata"
            else:
                kind_ = "content"
            rows[name] = {
                "class": kind_,
                "identity_occurrences": [left_count, right_count],
                "differing_sections": sections,
            }
        result[kind] = {
            "total": len(first),
            "equal": len(first) - len(rows),
            **{
                name: sum(row["class"] == name for row in rows.values())
                for name in ("identity_only", "identity_and_link_metadata", "content")
            },
            "differences": rows,
        }
    payload = {
        "kind": "native_build_comparison",
        "first": receipts[0]["label"],
        "second": receipts[1]["label"],
        "identity_sha256": digests_,
        **result,
    }
    write(args.output, payload)
    print(
        json.dumps(
            {kind: {k: v for k, v in result[kind].items() if k != "differences"} for kind in result}
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--preset", required=True)
    build.add_argument("--build-dir", type=Path, required=True)
    build.add_argument("--label", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--launcher", default="")
    build.add_argument("--linker", default="")
    build.add_argument("--ipo", action="store_true")
    build.add_argument("--pgo", choices=["off", "generate", "use"], default="off")
    build.add_argument("--pgo-data", default="")
    build.add_argument("--define", action="append", default=[])
    build.add_argument("--keep", action="store_true", help="No borrar el directorio antes")
    relink = commands.add_parser("relink")
    relink.add_argument("--build-dir", type=Path, required=True)
    relink.add_argument("--label", required=True)
    relink.add_argument("--output", type=Path, required=True)
    relink.add_argument("--jobs", type=int, default=2)
    touch = commands.add_parser("touch")
    touch.add_argument("--build-dir", type=Path, required=True)
    touch.add_argument("--header", required=True)
    touch.add_argument("--label", required=True)
    touch.add_argument("--output", type=Path, required=True)
    touch.add_argument("--jobs", type=int, default=2)
    touch.add_argument("--launcher", action="store_true")
    runtime = commands.add_parser("run")
    runtime.add_argument("--variant", action="append", required=True, help="etiqueta=directorio")
    runtime.add_argument("--tapes", type=Path, required=True)
    runtime.add_argument("--market", choices=["US", "CN"], default="US")
    runtime.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    runtime.add_argument("--rounds", type=int, default=3)
    runtime.add_argument("--ticks", type=int, default=512)
    runtime.add_argument("--warmup", type=int, default=32)
    runtime.add_argument("--skip", action="append", default=[])
    runtime.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--first", required=True, help="recibo=directorio")
    compare.add_argument("--second", required=True, help="recibo=directorio")
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    commands_ = {
        "build": command_build,
        "relink": command_relink,
        "touch": command_touch,
        "run": command_run,
        "compare": command_compare,
    }
    commands_[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
