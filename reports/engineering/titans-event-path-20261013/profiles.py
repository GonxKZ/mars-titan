"""Resumen de los perfiles del recorrido por evento de `mac_online` y del CUDA Graph de la emisión.

Uso: profiles.py DIRECTORIO SALIDA.json, desde la raíz del repositorio.

DIRECTORIO contiene la salida de las órdenes de [Perfiles](README.md#perfiles): `titans.pyspy`
(py-spy en formato raw), `titans.sqlite` (`nsys export --type sqlite`), `results.jsonl` de
`benchmarks/chronological_replay.py`, `graph-N.json` de `benchmarks/titans_emission_graph.py`
y, por ejecución, `ETIQUETA.pmon` con el uso de la SM por proceso cada segundo.

Del perfil de py-spy se cuentan las muestras de `_train_pass` y la parte inclusiva de cada
fase. De Nsight Systems se toma la ventana que va del primer rango del recorrido al último y
se suman los núcleos, las copias y los lanzamientos que empiezan dentro de ella. Los rangos
NVTX se anidan (`titans.prepare` dentro de `titans.observe`, `titans.backward` dentro de
`titans.update`), así que sus tiempos no se suman entre sí.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "benchmarks")
from collapsed_profile_summary import summary  # noqa: E402

ROOT = "_train_pass@training/financial_run.py"
PHASES = (
    "_observe@training/financial_run.py",
    "_update@training/financial_run.py",
    "_replay@training/financial_run.py",
    "_emit@training/financial_run.py",
    "prepare@models/titans/financial.py",
    "forward@models/titans/mac.py",
    "update@models/titans/neural_memory.py",
    "read@models/titans/neural_memory.py",
    "validate_state@models/titans/neural_memory.py",
    "encode_last@models/baselines/transformer.py",
    "validated_cpu_batch@models/titans/financial_inputs.py",
    "_digest@models/titans/financial_inputs.py",
    "validate_historical_vectors@data/input_policy.py",
    "_engine_run_backward@torch/autograd/graph.py",
    "clip_grad_norm_@torch/nn/utils/clip_grad.py",
    "iterencode@json/encoder.py",
)
PASS_RANGES = ("titans.observe", "titans.update", "titans.labels")


def nsys(path):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    ranges = db.execute(
        "SELECT COALESCE(e.text, s.value), e.start, e.end FROM NVTX_EVENTS e "
        "LEFT JOIN StringIds s ON s.id = e.textId WHERE e.end IS NOT NULL"
    ).fetchall()
    named = {}
    for name, start, end in ranges:
        if name and name.startswith("titans."):
            count, total = named.get(name, (0, 0))
            named[name] = (count + 1, total + end - start)
    spans = [(start, end) for name, start, end in ranges if name in PASS_RANGES]
    begin, finish = min(s for s, _ in spans), max(e for _, e in spans)

    def inside(table, value):
        return db.execute(
            f"SELECT COUNT(*), COALESCE(SUM({value}), 0) FROM {table} WHERE start BETWEEN ? AND ?",
            (begin, finish),
        ).fetchone()

    kernels, kernel_ns = inside("CUPTI_ACTIVITY_KIND_KERNEL", "end - start")
    copies, copy_ns = inside("CUPTI_ACTIVITY_KIND_MEMCPY", "end - start")
    _, copy_bytes = inside("CUPTI_ACTIVITY_KIND_MEMCPY", "bytes")
    calls = dict(
        db.execute(
            "SELECT s.value, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId WHERE r.start BETWEEN ? AND ? GROUP BY s.value",
            (begin, finish),
        ).fetchall()
    )
    wall = finish - begin
    return dict(
        pass_window_seconds=wall / 1e9,
        kernels=kernels,
        kernel_seconds=kernel_ns / 1e9,
        gpu_busy_fraction=kernel_ns / wall,
        copies=copies,
        copy_seconds=copy_ns / 1e9,
        copy_bytes=copy_bytes,
        runtime_calls={
            name.split("_v")[0]: count
            for name, count in sorted(calls.items(), key=lambda item: -item[1])[:8]
        },
        ranges={
            name: dict(instances=count, seconds=total / 1e9)
            for name, (count, total) in sorted(named.items(), key=lambda item: -item[1][1])
        },
    )


def foreign_samples(path):
    """Muestras en que el llama-server de Kaizten usó la SM y muestras en que apareció."""
    busy = listed = 0
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[-1] == "llama-server":
            listed += 1
            busy += fields[3].isdigit() and int(fields[3]) > 0
    return dict(llama_server_busy_samples=busy, llama_server_listed_samples=listed)


def main():
    directory, output = Path(sys.argv[1]), Path(sys.argv[2])
    replay = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines()]
    keep = (
        "label",
        "rows",
        "labels",
        "seconds",
        "rows_per_s",
        "optimizer_calls",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "loadavg",
        "tf32_matmul",
        "tf32_cudnn",
        "matmul_precision",
        "torch",
    )
    graphs = []
    for path in sorted(directory.glob("graph-*.json")):
        graphs.append(
            dict(
                label=path.stem,
                **json.loads(path.read_text()),
                **foreign_samples(directory / f"{path.stem}.pmon"),
            )
        )
    receipt = dict(
        kind="titans_event_path_profiles",
        issue=549,
        replay=[
            dict(
                {key: run[key] for key in keep},
                **foreign_samples(directory / f"{run['label']}.pmon"),
            )
            for run in replay
        ],
        pyspy=summary(directory / "titans.pyspy", ROOT, PHASES),
        nsys=nsys(directory / "titans.sqlite"),
        emission_graph=graphs,
        optimizer_steps=0,
        training_executed=False,
    )
    output.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
