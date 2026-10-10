"""Lista las funciones de prueba lentas a partir de los informes JUnit de pytest.

Una función es lenta si la suma de los tiempos de todos sus casos parametrizados alcanza el
umbral. La salida sustituye a `tests/slow_tests.txt`, que `tests/conftest.py` lee para añadir
la marca `slow`. Los tiempos dependen del equipo y de la carga, así que el archivo se
regenera con una ejecución completa y se revisa el diff antes de publicarlo.

Uso:
    uv run pytest --junitxml=suite.xml ...
    uv run python scripts/list_slow_tests.py --threshold 2 suite.xml > tests/slow_tests.txt
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def function_id(classname: str, name: str) -> str:
    """Identificador de pytest de la función, sin la parte de los parámetros."""
    parts = classname.split(".")
    modules = [index for index, part in enumerate(parts) if part.startswith("test_")]
    if not modules:
        raise ValueError(f"Clase de JUnit sin módulo de prueba: {classname}")
    module = modules[-1]
    node = "/".join(parts[: module + 1]) + ".py"
    for owner in parts[module + 1 :]:
        node += f"::{owner}"
    return f"{node}::{name.split('[', 1)[0]}"


def totals(reports: list[Path]) -> dict[str, float]:
    """Segundos de cada función, sumando sus casos de todos los informes."""
    seconds: dict[str, float] = defaultdict(float)
    for report in reports:
        for case in ET.parse(report).iter("testcase"):
            seconds[function_id(case.get("classname", ""), case.get("name", ""))] += float(
                case.get("time", 0)
            )
    return dict(seconds)


def render(seconds: dict[str, float], threshold: float) -> str:
    slow = sorted(node for node, value in seconds.items() if value >= threshold)
    header = [
        "# Funciones de prueba lentas: la suma de sus casos alcanza "
        f"{threshold:g} s en la medida de referencia.",
        "# Se regenera con scripts/list_slow_tests.py. Ver tests/README.md.",
    ]
    return "\n".join(header + slow) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="informes JUnit de pytest")
    parser.add_argument("--threshold", type=float, default=2.0, help="segundos por función")
    args = parser.parse_args(argv)
    if args.threshold <= 0:
        parser.error("El umbral debe ser positivo")
    sys.stdout.write(render(totals(args.reports), args.threshold))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
