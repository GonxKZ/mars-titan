"""Coste de anfitrión de la vista CPU y de los metadatos del estado con eventos reales (#638).

Uso: host_costs.py ÁRBOL EVENTOS VISTA SALIDA.json [REPETICIONES]

Recorre en CPU, con un hilo, todos los bloques de los eventos guardados por
`benchmarks/chronological_events.py`, troceados como en `benchmarks/chronological_replay.py`
con los `block_rows` de la receta, y mide con el código de ÁRBOL cada pieza por separado:

- `validated_cpu_batch` completo,
- el SHA-256 de la vista (`_digest`), con los bytes que resume,
- `validate_historical_vectors` sobre los mismos arreglos congelados,
- `_state_size` y `_usage` del predictor `mac_online` de la receta de la campaña, con el
  estado inicial de los flujos del bloque, y la parte de cada uno que es `json.dumps`.

Cada pieza se repite REPETICIONES veces (3 por defecto) y se guarda el mínimo, que es la
estimación menos afectada por otras cargas. No construye grafos ni ejecuta pasos.
"""

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

TREE = sys.argv[1]
sys.path[:0] = [f"{TREE}/src"]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mars_titan.data.input_policy import validate_historical_vectors  # noqa: E402
from mars_titan.models.baselines.inputs import MODALITIES  # noqa: E402
from mars_titan.models.titans.config import canonical  # noqa: E402
from mars_titan.models.titans.financial_inputs import (  # noqa: E402
    FinancialInputSpec,
    validated_cpu_batch,
)
from mars_titan.training import titans_walk_forward  # noqa: E402

RECIPE = Path(TREE) / "configs/titans/chronological-training-historical-masked.json"


def rows(raw, first, last):
    """Las filas [first, last) de un bloque guardado, con la misma estructura."""
    result = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            result[key] = rows(value, first, last)
        elif isinstance(value, (np.ndarray, list, tuple)):
            result[key] = value[first:last]
        else:
            result[key] = value
    return result


def blocks(event, size):
    for raw in event.inputs:
        total = len(raw["market"])
        for first in range(0, total, size):
            yield rows(raw, first, min(first + size, total))


def best(function, repeats):
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - began)
    return min(times), result


def main():
    events, view, output = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    torch.set_num_threads(1)
    with events.open("rb") as handle:
        replay = pickle.load(handle)
    first = replay["events"][0].inputs[0]
    manifest = json.loads(view.read_text())
    spec = FinancialInputSpec(
        source_sha256="0" * 64,
        view_sha256="1" * 64,
        representation=manifest["representation"],
        dimensions={name: first["inputs"][name].shape[-1] for name in first["inputs"]},
        input_policy=manifest["input_policy"],
    )
    document = json.loads(RECIPE.read_text())
    predictor, _ = titans_walk_forward._predictor(
        document, spec, "mac_online", 42, torch.device("cpu")
    )
    totals = dict(
        events=len(replay["events"]),
        block_rows=document["recipe"]["block_rows"],
        blocks=0,
        rows=0,
        digest_bytes=0,
        metadata_bytes=0,
        seconds=dict(
            validated_cpu_batch=0.0,
            digest=0.0,
            validate_historical_vectors=0.0,
            state_size=0.0,
            state_size_json=0.0,
            usage=0.0,
            usage_json=0.0,
        ),
    )
    seconds = totals["seconds"]
    size = document["recipe"]["block_rows"]
    for event in replay["events"]:
        for raw in blocks(event, size):
            spent, cpu = best(lambda raw=raw: validated_cpu_batch(raw, spec), repeats)
            seconds["validated_cpu_batch"] += spent
            seconds["digest"] += best(cpu._digest, repeats)[0]
            seconds["validate_historical_vectors"] += best(
                lambda cpu=cpu: validate_historical_vectors(
                    {name: cpu.inputs[name] for name in MODALITIES[1:]},
                    cpu.presence,
                    spec.representation,
                ),
                repeats,
            )[0]
            flows, ids, moments = cpu.flow_ids, cpu.sample_ids, cpu.prediction_at
            seconds["state_size"] += best(
                lambda cpu=cpu: predictor._state_size(
                    cpu.flow_ids, cpu.sample_ids, cpu.prediction_at
                ),
                repeats,
            )[0]
            metadata = predictor._metadata(flows, ids, moments)
            spent, text = best(lambda metadata=metadata: canonical(metadata), repeats)
            seconds["state_size_json"] += spent
            # El estado tras emitir el bloque, con sus IDs y su corte, como en `_validate_state`.
            state = replace(
                predictor.initial_state(flows), last_sample_ids=ids, last_prediction_at=moments
            )
            seconds["usage"] += best(lambda state=state: predictor._usage(state), repeats)[0]
            metadata = predictor._metadata(
                state.flow_ids, state.last_sample_ids, state.last_prediction_at
            )
            seconds["usage_json"] += best(lambda metadata=metadata: canonical(metadata), repeats)[0]
            totals["blocks"] += 1
            totals["rows"] += len(flows)
            totals["digest_bytes"] += sum(value.nbytes for value in cpu._arrays())
            totals["metadata_bytes"] += len(text.encode())
    totals.update(
        repeats=repeats,
        torch=torch.__version__,
        python=sys.version.split()[0],
        threads=torch.get_num_threads(),
        optimizer_steps=0,
    )
    output.write_text(json.dumps(totals, indent=1) + "\n")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
