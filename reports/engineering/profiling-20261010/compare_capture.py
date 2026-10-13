"""Comparación bit a bit de dos capturas de `benchmarks/chronological_replay.py --capture`.

Cada captura guarda los gradientes que recibe el optimizador sustituto en cada paso, los
nombres de los parámetros y las predicciones pendientes al terminar. Dos capturas coinciden
si los nombres, las predicciones pendientes y todos los gradientes son iguales, con el mismo
tipo y `torch.equal`.

Uso: compare_capture.py DIRECTORIO SALIDA.json A:B [C:D ...]
     (compara DIRECTORIO/A.pt con DIRECTORIO/B.pt, y así cada par)
"""

import json
import sys
from pathlib import Path

import torch


def count(value):
    if isinstance(value, dict):
        return sum(count(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(count(item) for item in value)
    return int(isinstance(value, torch.Tensor))


def same(first, second):
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(same(first[key], second[key]) for key in first)
        )
    if isinstance(first, list | tuple):
        return (
            isinstance(second, list | tuple)
            and len(first) == len(second)
            and all(same(a, b) for a, b in zip(first, second, strict=True))
        )
    if isinstance(first, torch.Tensor):
        return (
            isinstance(second, torch.Tensor)
            and first.dtype == second.dtype
            and torch.equal(first, second)
        )
    return first == second


def main():
    folder, output, pairs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    report = []
    for pair in pairs:
        base, other = pair.split(":")
        first = torch.load(folder / f"{base}.pt", weights_only=False)
        second = torch.load(folder / f"{other}.pt", weights_only=False)
        entry = dict(
            first=base,
            second=other,
            steps=len(first["grads"]),
            gradient_tensors=count(first["grads"]),
            gradients_equal=same(first["grads"], second["grads"]),
            names_equal=first["names"] == second["names"],
            pending=len(first["pending"]),
            pending_equal=first["pending"] == second["pending"],
        )
        report.append(entry)
        print(json.dumps(entry))
    output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
