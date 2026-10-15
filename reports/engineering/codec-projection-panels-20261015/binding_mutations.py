"""Mutaciones dirigidas del reparto de la proyección en el enlace de Python (#628).

Uso: binding_mutations.py ÁRBOL COMPILACIÓN SALIDA.json, sobre una copia del árbol.

COMPILACIÓN es un directorio ya configurado con el preset `native-candidate-cuda` y
`-DMARS_TITAN_BUILD_EPISODIC_PYTHON=ON`. Cada mutación cambia un único fragmento de
`native/src/episodic_python.cpp` (el reparto con `at::launch`, la espera, los tramos, la regla
de hilos y el aviso de fork), compila `_episodic_native` y ejecuta las pruebas de la
proyección de `tests/memory/test_episodic_codec.py` con ese enlace. Una mutación cuenta como
detectada si la compilación o las pruebas fallan o si se agota el tiempo. Al terminar se
restaura el archivo y se vuelve a compilar.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT, BUILD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
BINDING = ROOT / "native/src/episodic_python.cpp"
LIBRARY = "_episodic_native.cpython-312-x86_64-linux-gnu.so"
SELECTION = (
    "threads or pool or fork or keeps_every_encoded_bit or declared_order or campaign_widths"
)
MUTATIONS = {
    # El fallo que tuvo la rama: todo en el hilo que llama, con los mismos bits.
    "launch_inline": ("            at::launch(task);\n", "            task();\n"),
    # Las tareas nunca empiezan en el grupo 0, así que no calculan nada, y la mutación compila.
    "tasks_skip_their_range": (
        "            auto failure = range(first);\n",
        "            auto failure = first == 0 ? range(first) : std::exception_ptr{};\n",
    ),
    "caller_skips_first_range": (
        "    auto failure = range(0);\n",
        "    std::exception_ptr failure;\n",
    ),
    "last_range_past_the_end": (
        "first, std::min(groups, first + chunk));\n",
        "first, first + chunk);\n",
    ),
    "no_wait": ("    pending->done.wait(lock, [&] { return pending->left == 0; });\n", ""),
    "pool_never_used": (
        "    if (!inter_op_pool_usable.load(std::memory_order_relaxed)) {\n",
        "    if (true) {\n",
    ),
    "fork_handler_does_nothing": (
        "pthread_atfork(nullptr, nullptr, [] { inter_op_pool_usable.store(false); }) == 0);\n",
        "pthread_atfork(nullptr, nullptr, [] {}) == 0);\n",
    ),
}


def run(copy):
    compiled = subprocess.run(
        ["cmake", "--build", str(BUILD), "-j", "2", "--target", "_episodic_native"],
        capture_output=True,
        text=True,
    )
    if compiled.returncode:
        return dict(compiled=False, detected=True, tail=compiled.stdout.splitlines()[-3:])
    # Una copia con el mismo nombre, para que cada prueba cargue el enlace de esta mutación.
    shutil.copyfile(BUILD / LIBRARY, copy / LIBRARY)
    environment = dict(os.environ, MARS_TITAN_EPISODIC_NATIVE=str(copy / LIBRARY))
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p"]
    command += ["no:randomly", "tests/memory/test_episodic_codec.py", "-k", SELECTION]
    try:
        tested = subprocess.run(
            command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return dict(compiled=True, detected=True, tail="tiempo agotado")
    lines = (tested.stdout + tested.stderr).strip().splitlines()
    return dict(compiled=True, detected=tested.returncode != 0, tail=lines[-3:])


source = BINDING.read_text(encoding="utf-8")
results = {}
with tempfile.TemporaryDirectory(dir=ROOT) as directory:
    copy = Path(directory)
    baseline = run(copy)
    if baseline["detected"]:
        raise SystemExit(f"Las pruebas no pasan sin mutar: {baseline}")
    for name, (old, new) in MUTATIONS.items():
        if source.count(old) != 1:
            raise SystemExit(f"{name}: el fragmento no aparece una sola vez")
        BINDING.write_text(source.replace(old, new), encoding="utf-8")
        try:
            results[name] = run(copy)
        finally:
            BINDING.write_text(source, encoding="utf-8")
        print(name, results[name], flush=True)
    restored = run(copy)
    if restored["detected"]:
        raise SystemExit(f"Las pruebas no pasan tras restaurar: {restored}")
OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("detectadas", sum(bool(r["detected"]) for r in results.values()), "de", len(results))
