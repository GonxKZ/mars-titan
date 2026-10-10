"""Paquetes de trazas de prueba para las pruebas de navegador del observatorio.

Los valores son funciones deterministas sin relación con ningún entrenamiento. Cada paquete
se declara con procedencia `fixture` y la página lo rotula como datos de prueba en toda la
vista. Sirven para comprobar la lectura binaria, la reducción M4 y el coste de dibujo con
un millón de puntos, nunca para ilustrar resultados.

Uso: python make_traces.py CARPETA [--points N]
"""

import argparse
import math

from mars_titan.observatory.traces import write_trace_bundle


def wave(step, period, phase=0.0):
    return math.sin(2 * math.pi * step / period + phase)


def build(folder, points, cadence=None, truncated_at=None):
    x = list(range(1, points + 1))
    # Hueco declarado: un tramo sin observaciones que la página debe dejar en blanco.
    gap = range(points // 3, points // 3 + max(1, points // 50))
    loss = [0.02 + 0.01 * math.exp(-s / (points / 6)) + 0.002 * wave(s, 997) for s in x]
    for i in gap:
        loss[i] = math.nan
    layers = [f"Capa {i}" for i in range(1, 5)]
    columns = list(range(1, min(points, 200_000) + 1))
    gates = [
        0.5 + 0.4 * wave(s, 5000 + 700 * layer, layer) * math.exp(-s / (len(columns) * 2))
        for layer in range(len(layers))
        for s in columns
    ]
    write_trace_bundle(
        folder,
        "prueba-titans",
        run_id="fixture-titans-run",
        attempt_id="attempt-0001",
        model_id="fixture",
        provenance="fixture",
        x_unit="optimizer_step",
        cadence=cadence,
        truncated_at=truncated_at,
        series=[
            dict(
                id="optimization.loss",
                group="optimization",
                label="Pérdida de entrenamiento",
                unit="pérdida",
                x=x,
                y=loss,
            ),
            dict(
                id="titans.surprise",
                group="titans",
                label="Sorpresa asociativa",
                unit="pérdida asociativa",
                x=x,
                y=[abs(0.3 * wave(s, 1301)) + 0.05 * wave(s, 37) ** 2 for s in x],
            ),
            dict(
                id="episodic.size",
                group="episodic",
                label="Episodios en el banco",
                unit="episodios",
                x=x,
                y=[min(512.0, float(s // 97)) for s in x],
            ),
        ],
        matrices=[
            dict(
                id="titans.alpha",
                group="titans",
                label="Puerta de olvido α por capa",
                unit="fracción",
                rows=layers,
                x=columns,
                values=gates,
                range=[0.0, 1.0],
            ),
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--points", type=int, default=20_000)
    parser.add_argument("--cadence", type=int)
    parser.add_argument("--truncated-at", type=int)
    arguments = parser.parse_args()
    build(arguments.folder, arguments.points, arguments.cadence, arguments.truncated_at)
