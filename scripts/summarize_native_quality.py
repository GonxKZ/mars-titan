# /// script
# requires-python = ">=3.11"
# dependencies = ["lizard==1.17.31"]
# ///
"""Relacionar cobertura LCOV y complejidad de las funciones nativas propias."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import lizard

LCOV_FIELDS = {"LF": "lines", "LH": "covered_lines", "BRF": "branches", "BRH": "covered_branches"}


def read_lcov(path):
    files, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("SF:"):
            name = Path(line[3:])
            if name.is_absolute():
                name = name.relative_to(Path.cwd())
            current = files.setdefault(str(name), {"lines": {}, "totals": {}})
        elif current is not None and line.startswith("DA:"):
            number, count, *_ = line[3:].split(",")
            current["lines"][int(number)] = (
                current["lines"].get(int(number), False) or int(count) > 0
            )
        elif current is not None and line.split(":", 1)[0] in LCOV_FIELDS:
            field, count = line.split(":", 1)
            current["totals"][LCOV_FIELDS[field]] = int(count)
        elif line == "end_of_record":
            current = None
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcov", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coverage = read_lcov(args.lcov)
    sources = subprocess.check_output(
        [
            "rg",
            "--files",
            "native/src",
            "native/include",
            "-g",
            "*.cpp",
            "-g",
            "*.hpp",
            "-g",
            "*.h",
        ],
        text=True,
    ).splitlines()
    files, functions = [], []
    for name in sorted(sources):
        item = coverage.get(name, {"lines": {}, "totals": {}})
        counts = item["lines"]
        files.append(
            dict(
                path=name,
                source_sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest(),
                **{key: item["totals"].get(key, 0) for key in LCOV_FIELDS.values()},
            )
        )
        for function in lizard.analyze_file(name).function_list:
            lines = [
                covered
                for number, covered in counts.items()
                if function.start_line <= number <= function.end_line
            ]
            fraction = sum(lines) / len(lines) if lines else 0.0
            complexity = function.cyclomatic_complexity
            functions.append(
                dict(
                    path=name,
                    function=function.long_name,
                    start_line=function.start_line,
                    end_line=function.end_line,
                    cyclomatic_complexity=complexity,
                    executable_lines=len(lines),
                    covered_lines=sum(lines),
                    coverage_fraction=fraction,
                    crap=complexity**2 * (1 - fraction) ** 3 + complexity,
                )
            )
    if not any(item["lines"] for item in files):
        raise ValueError("El informe no contiene cobertura de las fuentes nativas propias")
    totals = {
        key: sum(item[key] for item in files)
        for key in ("lines", "covered_lines", "branches", "covered_branches")
    }
    report = dict(
        schema_version=1,
        measured_at=datetime.now(UTC).isoformat(),
        tools={
            "coverage": "LLVM 21 LCOV export",
            "complexity": f"lizard {importlib.metadata.version('lizard')}",
        },
        coverage_convention="LF/LH y BRF/BRH de LLVM LCOV, solo fuentes y cabeceras propias.",
        crap_convention="CCN² × (1 − cobertura de líneas de la función)³ + CCN.",
        limits=[
            "CRAP une las líneas DA ejecutadas de las instancias en el intervalo de Lizard.",
            "Las lambdas pueden compartir líneas con la función que las contiene.",
            "Una función sin líneas instrumentadas conserva cobertura cero en este diagnóstico.",
            "CRAP combina complejidad y recorridos observados, no demuestra ausencia de errores.",
        ],
        totals=totals,
        files=files,
        functions=sorted(functions, key=lambda item: item["crap"], reverse=True),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
