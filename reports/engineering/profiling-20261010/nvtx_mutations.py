"""Mutaciones dirigidas de `src/mars_titan/nvtx_ranges.py`.

Cada mutación introduce un fallo concreto (tragarse una excepción, no cerrar el lector, no
cerrar el rango, tocar el modo de gradiente o el generador aleatorio, consumir dentro del
rango, dejar los rangos siempre encendidos o aceptar cualquier valor del interruptor) y debe
hacer fallar alguna prueba de `tests/tooling/test_nvtx_ranges.py` o de
`tests/training/test_nvtx_parity.py`. El archivo original se restaura siempre al terminar.

Uso, desde la raíz del repositorio: nvtx_mutations.py SALIDA.json
"""

import json
import subprocess
import sys
from pathlib import Path

TARGET = Path("src/mars_titan/nvtx_ranges.py")
TESTS = ("tests/tooling/test_nvtx_ranges.py", "tests/training/test_nvtx_parity.py")
PUSH = "        _nvtx().range_push(self.name)\n        return self"
MUTATIONS = {
    "swallow": (
        "        _nvtx().range_pop()\n        return False",
        "        _nvtx().range_pop()\n        return True",
    ),
    "no_close": ("            close()\n", "            pass\n"),
    "no_pop": ("        _nvtx().range_pop()\n", "        pass\n"),
    "grad_off": (
        PUSH,
        "        _nvtx().range_push(self.name)\n        import torch\n"
        "        torch.set_grad_enabled(False)\n        return self",
    ),
    "rng": (
        PUSH,
        "        _nvtx().range_push(self.name)\n        import torch\n"
        "        torch.rand(1)\n        return self",
    ),
    "consumer_inside": (
        "                    return\n            yield item\n",
        "                    return\n                yield item\n",
    ),
    "always_on": ("    return _Range(name) if ENABLED else _OFF", "    return _Range(name)"),
    "env_lenient": (
        '    if value not in {"0", "1"}:\n        raise',
        "    if False:\n        raise",
    ),
}


def main():
    output = Path(sys.argv[1])
    original = TARGET.read_text()
    results = {}
    try:
        for name, (old, new) in MUTATIONS.items():
            if original.count(old) != 1:
                raise SystemExit(f"La mutación {name} no encuentra un único punto de inserción")
            TARGET.write_text(original.replace(old, new))
            process = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = process.stdout.strip().splitlines()
            results[name] = dict(
                killed=process.returncode != 0,
                summary=lines[-1] if lines else process.stderr[-200:],
            )
            print(name, results[name], flush=True)
    finally:
        TARGET.write_text(original)
    output.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
