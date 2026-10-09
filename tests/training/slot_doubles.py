"""Ejecutores sustitutos importables desde los procesos de las ranuras de la campaña.

Escriben las filas nulas de la vista como `test_masked_campaign.Recorder` y anotan cada
llamada en un registro de líneas JSON que indica `SLOT_LOG`. No usan CUDA ni ajustan nada.
`SLOT_DELAY` alarga cada trabajo y `SLOT_CRASH` hace que ese trabajo muera sin resultado.
Con `SLOT_PAUSE_AFTER`, el primer trabajo que empieza tras ese número de trabajos terminados
crea el archivo de parada y espera a que la campaña se la transmita, así que la parada
siempre llega con al menos un trabajo en curso.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import QUANTILE_COLUMNS
from mars_titan.training import masked_campaign as engine
from mars_titan.training.corpus_inputs import CorpusDataset


def _log(**entry):
    path = os.environ.get("SLOT_LOG")
    if path:
        line = json.dumps(entry, sort_keys=True) + "\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)


def _completed():
    path = Path(os.environ["SLOT_LOG"])
    if not path.exists():
        return 0
    return sum(json.loads(line)["event"] == "completed" for line in path.read_text().splitlines())


def _rows(view, partition):
    dataset = CorpusDataset(view, input_policy=HISTORICAL_MASKED)
    parts = dict(sample_id=[], market=[], prediction_at=[], target=[])
    for batch in dataset.batches(partition=partition, batch_size=64, epoch=0, seed=0):
        parts["sample_id"] += list(batch["sample_ids"])
        parts["market"] += list(batch["market"])
        parts["prediction_at"].append(np.asarray(batch["prediction_at"]))
        parts["target"].append(np.asarray(batch["target"], dtype=np.float64))
    for name in ("prediction_at", "target"):
        parts[name] = np.concatenate(parts[name])
    return parts


def _table(run, partition):
    rows = _rows(run.view, partition)
    count = len(rows["target"])
    columns = dict(
        sample_id=rows["sample_id"],
        asset_id=["/".join(key.split("/")[:2]) for key in rows["sample_id"]],
        market=rows["market"],
        prediction_at=pa.array(rows["prediction_at"], type=pa.timestamp("us", tz="UTC")),
        target=rows["target"],
        prediction=np.zeros(count, dtype=np.float32),
        zero=np.zeros(count, dtype=np.float64),
    )
    if run.job["family"] == "neural_reference":
        columns.update({name: np.zeros(count, dtype=np.float32) for name in QUANTILE_COLUMNS})
    return pa.table(columns)


def _score(job):
    if job["stage"] != "search":
        return 0.5
    return {"gru-00": 0.02, "gru-10": 0.01}.get(job["candidate"], 0.03)


def record_job(run):
    """Escribir el resultado del trabajo y anotar proceso, inicio y fin."""
    started = time.time()
    if os.environ.get("SLOT_CRASH") == run.job["id"]:
        os._exit(3)
    delay = float(os.environ.get("SLOT_DELAY", "0"))
    run.folder.mkdir(parents=True, exist_ok=True)
    resumed = (run.folder / "partial.bin").exists()
    pause_after = os.environ.get("SLOT_PAUSE_AFTER")
    if pause_after and _completed() >= int(pause_after):
        Path(os.environ["SLOT_STOP_FILE"]).touch()
        delay = 60.0
    end = time.time() + delay
    while time.time() < end:
        if run.stop.requested:
            (run.folder / "partial.bin").write_bytes(b"incompleto")
            _log(id=run.job["id"], pid=os.getpid(), event="paused", start=started)
            raise engine.Paused
        time.sleep(0.01)
    predictions = {}
    for partition in ("calibration", "evaluation"):
        path = run.folder / f"{partition}-predictions.parquet"
        pq.write_table(_table(run, partition), path)
        predictions[partition] = dict(path=path.name, sha256=sha256(path))
    predictions["validation"] = dict(metrics=dict(session_mae=_score(run.job)))
    report = dict(status="completed", final_test_opened=False, predictions=predictions)
    if run.job["kind"] == "carry":
        anchor = json.loads((run.anchor["folder"] / "run.json").read_text())
        report["anchor"] = dict(checkpoint_sha256=anchor["checkpoint"]["sha256"])
        atomic_json(run.folder / "carry.json", report)
    else:
        (run.folder / "model.bin").write_text(run.job["id"])
        report["checkpoint"] = dict(path="model.bin", sha256=sha256(run.folder / "model.bin"))
        atomic_json(run.folder / "run.json", report)
    _log(
        id=run.job["id"],
        pid=os.getpid(),
        event="completed",
        start=started,
        end=time.time(),
        resumed=resumed,
        environment={k: v for k, v in os.environ.items() if k.startswith("MARS_TITAN_")},
    )
    return report


class FileStop:
    """Parada de la campaña que se pide creando un archivo."""

    def __init__(self, path):
        self.path = Path(path)

    @property
    def requested(self):
        return self.path.exists()
