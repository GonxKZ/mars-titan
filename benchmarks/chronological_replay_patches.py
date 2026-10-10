"""Ajustes de medición: caché de huellas de artefactos y tramo de ajuste acotado.

Solo para medir caudal, nunca para ajustar. Con `BENCH_SHA_CACHE`, la comprobación SHA-256
de cada artefacto del corpus se calcula de verdad una vez y se reutiliza en otros procesos
mientras el archivo conserve dispositivo, inodo, tamaño y fechas. Con `BENCH_TRAIN_RANGE`
(dos fechas ISO), el tramo de ajuste se acota a ese intervalo y la validación a su primer
mes, para que el índice de observaciones no recorra 22 años. El caudal se mide igual,
sobre el periodo más poblado de la ventana.
"""

import fcntl
import json
import os
from datetime import UTC, datetime

from mars_titan.memory.financial_session import FinancialPhase
from mars_titan.training import candidate_walk_forward as candidate
from mars_titan.training import corpus_inputs
from mars_titan.training import titans_walk_forward as titans

CACHE = os.environ.get("BENCH_SHA_CACHE")
RANGE = os.environ.get("BENCH_TRAIN_RANGE")
_real_sha256 = corpus_inputs.sha256
_known = {}
if CACHE and os.path.exists(CACHE):
    with open(CACHE) as handle:
        for line in handle:
            path, signature, digest = json.loads(line)
            _known[path] = (tuple(signature), digest)


def _cached_sha256(path):
    stat = os.stat(path)
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    key = os.path.realpath(path)
    hit = _known.get(key)
    if hit is not None and hit[0] == signature:
        return hit[1]
    digest = _real_sha256(path)
    _known[key] = (signature, digest)
    with open(CACHE, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps([key, list(signature), digest]) + "\n")
    return digest


if CACHE:
    corpus_inputs.sha256 = _cached_sha256


def _micros(day):
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1_000_000)


def _bounded(phases):
    start, end = (_micros(day) for day in RANGE.split(","))
    month = 31 * 86_400 * 1_000_000
    result = dict(phases)
    result["train"] = FinancialPhase("train", start, start, end, end)
    if "validation" in result:
        first = result["validation"].decision_start
        result["validation"] = FinancialPhase(
            "validation", first, first, first + month, first + month
        )
    return result


if RANGE:
    _window_phases, _candidate_phases = titans.window_phases, candidate._phases

    def window_phases(fold, warmup_months):
        return _bounded(_window_phases(fold, warmup_months))

    def candidate_phases(dataset):
        return _bounded(_candidate_phases(dataset))

    titans.window_phases = window_phases
    candidate._phases = candidate_phases
