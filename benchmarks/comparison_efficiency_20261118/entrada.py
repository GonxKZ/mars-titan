"""Fuentes de ajuste y validación de los checkpoints de la medición de noviembre."""

import json
import os
import sys
from pathlib import Path

from configuracion import workdir

from benchmarks.comparison_input_archive import replay, write
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.input_pipeline import background as background

STATE = Path(os.environ["MARS_TITAN_COMPARISON_STATE"]).resolve(strict=True)
OUT = workdir()
YEAR, SEED = 2021, 44


def sources():
    view = STATE / "cohorts" / str(YEAR) / "manifest.json"
    data = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    anchor = STATE / "runs/mars_titan_m2" / str(YEAR) / f"seed-{SEED}" / "run.json"
    run = json.loads(anchor.read_text())
    if run["status"] != "completed" or run["request"]["view_sha256"] != sha256(view):
        raise ValueError("El lector necesita su cohorte y un ajuste terminado")
    manifests = (STATE / "indices" / data.identity[:16]).glob("*/manifest.json")
    by_hash = {sha256(path): path for path in manifests}
    return {
        k: FinancialObservationSource(data, by_hash[run["identity"]["indices"][k]])
        for k in ("train", "validation")
    }


def packed_stream(source, spec, path, start, events, block, offset=0, output_block=None):
    stamp = json.loads(path.with_suffix(".json").read_text())
    if (stamp["start"], stamp["events"], stamp["block"]) != (start, events, block):
        raise ValueError("La copia contiene otro intervalo")
    yield from replay(source, spec, path, stamp, cursor=start + offset, block=output_block or block)


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise ValueError("La preparación de los archivos se ejecuta con CUDA oculta")
    src = sources()
    for name, width in (("train", 1024), ("validation", 512)):
        source = src[name]
        total = len(source.metadata["groups"])
        start, count, label = (
            (int(0.9 * total), 144, "train-late") if name == "train" else (0, total, name)
        )
        path = OUT / f"{label}-{width}.arrow"
        stamp = write(
            source,
            source.specification(),
            path,
            start=start,
            count=count,
            block=width,
            max_bytes=6 * 1024**3,
        )
        atomic_json(path.with_suffix(".json"), stamp)
        print(f"{label}: {count} instantes, {stamp['bytes']} bytes", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Las rutas se toman del entorno descrito en el informe")
    prepare()
