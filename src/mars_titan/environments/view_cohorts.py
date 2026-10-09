"""Cohortes de ajuste y validación leídas por bloques desde la vista, sin copia ordenada.

`prepare_cohort_index` recorre una vez cada tramo con el lector que también alimenta el
corpus ordenado (`CorpusDataset.batches` con época y semilla cero) y guarda solo el
índice de sesiones, la población de cada mercado y la rejilla de acciones. No escribe
filas. `ViewCohortSource` entrega después las cohortes en el orden que pide su
consumidor. Agrupa posiciones consecutivas en bloques que caben en un presupuesto de
memoria, recorre la vista una vez por bloque y conserva solo las filas de las sesiones
del bloque. Cada cohorte pasa las mismas comprobaciones que `ParquetCohortSource` sobre
el corpus ordenado de la misma vista y conserva sus bits, así que los lotes, su orden y
los cursores no cambian.

Dentro de un bloque, un vector nulo o idéntico al primero guardado de su sesión no se
copia (el contexto macro coincide en todos los activos de una sesión y las noticias
ausentes son ceros). Si el bloque supera el presupuesto mientras se lee, se descartan
sus últimas sesiones, que pasan al bloque siguiente. El coste es una lectura completa
del tramo por bloque, a cambio de no ocupar disco con filas.
"""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import INPUT_POLICIES, masked_inputs, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.partition_contract import ordered_bounds

from .actions import ActionGrid
from .cohorts import shapes_contract
from .corpus_source import (
    MAX_BLOCK_BYTES,
    _check_cohorts,
    _code,
    check_presence,
    check_provenance,
    checked_cohort,
    checked_index,
)

INDEX_KIND = "causal_prediction_index"
PARTITIONS = ("train", "validation")
MAX_BUDGET_BYTES = 64 * 1024**3
# Bytes por fila fuera de las modalidades: clave, tres fechas u objetivos, bits y códigos.
ROW_OVERHEAD = 128
ZERO, REFERENCE, STORED = 0, 1, 2
# Filas por lote de cada lectura. El contenido de las filas no depende del lote.
PASS_ROWS = 4096


def _rows(dataset, partition, stop):
    """Lotes del lector de la vista con su instante entero, como al preparar el corpus."""
    for batch in dataset.batches(partition=partition, batch_size=PASS_ROWS, epoch=0, seed=0):
        if stop is not None and stop.requested:
            raise InterruptedError("Lectura de la vista interrumpida antes de cerrar el bloque")
        if np.isnat(batch["input_available_at"]).any():
            raise ValueError("Falta evidencia de disponibilidad para el entorno causal")
        yield batch, batch["prediction_at"].astype(np.int64)


def _record_digest(source_sha256, partition, record):
    content = dict(source_sha256=source_sha256, partition=partition, **record)
    content.pop("sha256", None)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _sessions(times, *columns):
    """Inicio, filas y mínimo y máximo de cada columna por sesión, con filas ordenadas."""
    starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    counts = np.diff(np.r_[starts, len(times)])
    extremes = [
        (np.minimum.reduceat(column, starts), np.maximum.reduceat(column, starts))
        for column in columns
    ]
    return times[starts], counts, extremes


def _partition_index(dataset, partition, report, assets, input_policy, stop):
    """Recorrer el tramo una vez y fijar índice, mercados y comprobaciones de cada cohorte."""
    times, codes, available, mature, targets, shapes = [], [], [], [], [], None
    for batch, at in _rows(dataset, partition, stop):
        if shapes is None:
            shapes = {name: list(value.shape[1:]) for name, value in batch["inputs"].items()}
        times.append(at)
        codes.append(
            np.fromiter(
                (assets[key.rsplit("/", 1)[0]] for key in batch["sample_ids"]), np.int64, len(at)
            )
        )
        available.append(batch["input_available_at"].astype(np.int64))
        mature.append(batch["target_available_at"].astype(np.int64))
        targets.append(batch["target"])
    if shapes is None:
        raise ValueError("El tramo de la vista no contiene filas")
    times, codes, available, mature, targets = map(
        np.concatenate, (times, codes, available, mature, targets)
    )
    count = len(times)
    if count != report["counts"][partition]:
        raise ValueError("La partición no conserva el número de muestras")
    # Mismo orden que el corpus ordenado: instante y después identificador del activo.
    order = np.lexsort((codes, times))
    times, codes, available, mature, targets = (
        array[order] for array in (times, codes, available, mature, targets)
    )
    repeated = np.r_[False, (times[1:] == times[:-1]) & (codes[1:] == codes[:-1])]
    names = sorted(assets, key=assets.get)
    market_of = np.array([name.split("/", 1)[0] for name in names])[codes]
    labels, sizes = np.unique(market_of, return_counts=True)
    markets = {str(market): int(n) for market, n in zip(labels, sizes, strict=True)}
    bounds = {
        market: ordered_bounds(report, market=market, input_policy=input_policy)[partition]
        for market in markets
    }
    market_index = []
    for market in markets:
        chosen = market_of == market
        moments, rows, (available_range, mature_range) = _sessions(
            times[chosen], available[chosen], mature[chosen]
        )
        starts = np.r_[0, np.cumsum(rows)[:-1]]
        repeats = np.add.reduceat(repeated[chosen].astype(np.int64), starts)
        market_index.extend(
            (market, int(at), int(n), int(n - dup), int(a0), int(a1), int(m0), int(m1))
            for at, n, dup, a0, a1, m0, m1 in zip(
                moments, rows, repeats, *available_range, *mature_range, strict=True
            )
        )
    if len(market_index) > 200_000:
        raise ValueError("El índice por mercado no concilia o supera su presupuesto")
    _check_cohorts(market_index, bounds)
    moments, rows, _ = _sessions(times)
    if not 1 <= len(moments) <= 100_000:
        raise ValueError("El índice temporal no concilia o supera el presupuesto de cohortes")
    maximum = int(rows.max())
    shapes_contract(shapes, maximum, MAX_BLOCK_BYTES)
    record = dict(
        rows=count,
        cohorts=[[int(at), int(n)] for at, n in zip(moments, rows, strict=True)],
        max_assets=maximum,
        market_rows=markets,
    )
    record["sha256"] = _record_digest(report["source_sha256"], partition, record)
    return record, shapes, targets


def prepare_cohort_index(dataset, output, *, input_policy, stop=None):
    """Confirmar el índice de ajuste y validación de una vista, o reutilizar el confirmado.

    El recibo conserva los campos del corpus ordenado que leen el padre y los lectores
    (recuentos, formas, mercados, límites y rejilla), con otra clase y sin archivos.
    """
    if input_policy not in INPUT_POLICIES or masked_inputs(input_policy) != dataset.masked:
        raise ValueError("La política de entradas no corresponde a la vista")
    started = time.perf_counter()
    output = Path(output)
    for protected in (*dataset.roots.values(), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    identity = dict(
        source_sha256=dataset.identity,
        code=dict(
            _code(masked=dataset.masked),
            **{"environments/view_cohorts.py": sha256(Path(__file__))},
        ),
        pyarrow=pa.__version__,
        numpy=np.__version__,
        cohort_id=dataset.cohort,
        news_content_policy=dataset.manifest.get("news_content_policy"),
        **policy_identity(input_policy),
    )
    if getattr(dataset, "temporal", None) is not None:
        identity["source_manifest"] = dict(
            path=str(dataset.path.resolve()), sha256=dataset.identity
        )
    path = output / "manifest.json"
    if path.is_file():
        report = read_manifest(path)[0]
        if report.get("identity") != identity or report.get("status") != "completed":
            raise ValueError("El índice confirmado pertenece a otra vista, política o código")
        return report
    report = dict(
        schema_version=1,
        kind=INDEX_KIND,
        status="completed",
        identity=identity,
        source_sha256=dataset.identity,
        counts=dataset.manifest["counts"],
        cohort_id=dataset.manifest.get("cohort_id"),
        news_content_policy=dataset.manifest.get("news_content_policy"),
        scope=dataset.manifest["scope"],
        cohort_complete=dataset.manifest["cohort_complete"],
        final_test_opened=False,
        shapes=None,
        partitions={},
        **policy_identity(input_policy),
    )
    if "source_manifest" in identity:
        report["source_manifest"] = identity["source_manifest"]
    check_provenance(report)
    names = sorted(f"{asset['market']}/{asset['symbol']}" for asset in dataset.assets)
    assets = {name: code for code, name in enumerate(names)}
    grid = None
    for partition in PARTITIONS:
        record, shapes, targets = _partition_index(
            dataset, partition, report, assets, input_policy, stop
        )
        if report["shapes"] not in (None, shapes):
            raise ValueError("Las dimensiones de las modalidades cambian entre tramos")
        report["shapes"], report["partitions"][partition] = shapes, record
        if partition == "train":
            grid = ActionGrid.fit(targets, source_sha256=dataset.identity, partition="train")
        del targets
    if sha256(dataset.path) != dataset.identity:
        raise ValueError("El manifiesto de origen ha cambiado durante la preparación")
    report.update(grid=grid.to_dict(), attempt_seconds=time.perf_counter() - started)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    return report


class _SessionRows:
    """Filas de una sesión con los vectores nulos o repetidos guardados una sola vez."""

    def __init__(self, shapes, masked):
        self.shapes, self.masked = shapes, masked
        self.chunks, self.reference, self.rows, self.bytes, self.uses = [], {}, 0, 0, 0

    def add(self, batch, rows):
        count = len(rows)
        chunk = dict(
            ids=[batch["sample_ids"][i].rsplit("/", 1)[0] for i in rows],
            available=batch["input_available_at"][rows].astype(np.int64),
            mature=batch["target_available_at"][rows].astype(np.int64),
            target=batch["target"][rows],
            presence=batch["presence"][rows] if self.masked else None,
            inputs={},
        )
        size = count * ROW_OVERHEAD
        for name in self.shapes:
            values = batch["inputs"][name][rows].reshape(count, -1)
            bits = values.view(np.uint32)
            kind = np.full(count, STORED, dtype=np.int8)
            kind[~bits.any(axis=1)] = ZERO
            reference = self.reference.get(name)
            if reference is None and (kind == STORED).any():
                reference = self.reference[name] = bits[np.argmax(kind == STORED)].copy()
                size += reference.nbytes
            if reference is not None:
                kind[(kind == STORED) & (bits == reference).all(axis=1)] = REFERENCE
            stored = values[kind == STORED]
            chunk["inputs"][name] = (kind, stored)
            size += kind.nbytes + stored.nbytes
        self.chunks.append(chunk)
        self.rows += count
        self.bytes += size
        return size

    def raw(self, at):
        """Reconstruir la cohorte con los mismos bits que el lector de la vista."""
        inputs = {}
        for name, shape in self.shapes.items():
            values = np.zeros((self.rows, math.prod(shape)), dtype=np.float32)
            reference, offset = self.reference.get(name), 0
            for chunk in self.chunks:
                kind, stored = chunk["inputs"][name]
                part = values[offset : offset + len(kind)]
                if reference is not None:
                    part[kind == REFERENCE] = reference.view(np.float32)
                part[kind == STORED] = stored
                offset += len(kind)
            inputs[name] = values.reshape((self.rows, *shape))
        raw = dict(
            prediction_at=at,
            asset_ids=[asset for chunk in self.chunks for asset in chunk["ids"]],
            inputs=inputs,
            available_at=np.concatenate([chunk["available"] for chunk in self.chunks]),
            target_available_at=np.concatenate([chunk["mature"] for chunk in self.chunks]),
            target=np.concatenate([chunk["target"] for chunk in self.chunks]),
        )
        presence = (
            np.concatenate([chunk["presence"] for chunk in self.chunks]) if self.masked else None
        )
        return raw, presence


class ViewCohortSource:
    """Cohortes de un tramo leídas desde la vista por bloques, con memoria acotada.

    Expone la interfaz de `ParquetCohortSource` que usan `PairedInputs` y el padre, más
    `cohorts(posiciones)`, que lee por bloques una secuencia completa. Una posición suelta
    cuesta una lectura del tramo, así que los consumidores piden la secuencia entera.
    """

    def __init__(self, manifest, dataset, *, partition, max_block_bytes, input_policy, stop=None):
        self.manifest_path = Path(manifest)
        meta, self.manifest_sha256 = read_manifest(self.manifest_path)
        expected = policy_identity(input_policy) if input_policy in INPUT_POLICIES else None
        if expected is None or any(
            {key: value[key] for key in ("input_policy", "mask_contract") if key in value}
            != expected
            for value in (meta, meta.get("identity", {}))
        ):
            raise ValueError("La política de entradas no coincide con el índice de cohortes")
        self.input_policy, self.masked = input_policy, masked_inputs(input_policy)
        if (
            meta.get("schema_version") != 1
            or meta.get("kind") != INDEX_KIND
            or meta.get("status") != "completed"
            or meta.get("final_test_opened") is not False
            or partition not in PARTITIONS
            or type(max_block_bytes) is not int
            or not 1 <= max_block_bytes <= MAX_BUDGET_BYTES
        ):
            raise ValueError("El índice o el presupuesto de lectura no es válido")
        if (
            dataset.identity != meta["source_sha256"]
            or dataset.masked != self.masked
            or meta["identity"]["source_sha256"] != meta["source_sha256"]
        ):
            raise ValueError("La vista abierta no es la que describe el índice")
        record = meta["partitions"][partition]
        if record.get("sha256") != _record_digest(meta["source_sha256"], partition, record):
            raise ValueError("El índice del tramo no conserva su huella")
        self.dataset, self.partition, self.stop = dataset, partition, stop
        self.population_counts = dict(meta["counts"])
        self.cohort_id, self.news_content_policy = check_provenance(meta)
        if dataset.cohort != self.cohort_id:
            raise ValueError("La vista no conserva la cohorte del índice")
        self.partition_sha256 = record["sha256"]
        self.shapes = shapes_contract(meta["shapes"], record["max_assets"], MAX_BLOCK_BYTES)
        self.max_assets, self.source_sha256 = record["max_assets"], meta["source_sha256"]
        self.market_bounds, self.bounds, self.index = checked_index(
            meta, record, partition, input_policy, self.max_assets
        )
        self.offsets = np.cumsum([0] + [row[1] for row in self.index])
        self.max_block_bytes = max_block_bytes
        # El plan del primer bloque supone filas completas. Los siguientes usan lo medido.
        self.row_bytes = 4 * sum(math.prod(shape) for shape in self.shapes.values())
        self.row_bytes += ROW_OVERHEAD + len(self.shapes)
        self.estimate = self.row_bytes
        self.statistics = dict(passes=0, cohorts=0, evicted=0, peak_block_bytes=0)
        self.closed = False

    def __len__(self):
        return len(self.index)

    def __call__(self, position):
        if self.closed or type(position) is not int or not 0 <= position <= len(self):
            raise ValueError("El cursor no pertenece a una fuente abierta")
        if position == len(self):
            return None
        return next(self.cohorts([position]))

    def _plan(self, positions, start):
        """Final del siguiente bloque según las filas estimadas de sus sesiones."""
        seen, total, end = set(), 0, start
        while end < len(positions):
            position = positions[end]
            if position not in seen:
                cost = self.index[position][1] * self.estimate
                if total + cost > self.max_block_bytes and end > start:
                    break
                seen.add(position)
                total += cost
            end += 1
        return end

    def _collect(self, block):
        """Leer el tramo una vez y conservar las filas de las sesiones del bloque.

        Devuelve cuántas posiciones iniciales del bloque quedan completas y sus filas.
        """
        rank, expected = {}, {}
        for i, position in enumerate(block):
            at, rows = self.index[position]
            rank.setdefault(at, i)
            expected[at] = rows
        sessions = {at: _SessionRows(self.shapes, self.masked) for at in rank}
        for at, uses in Counter(self.index[position][0] for position in block).items():
            sessions[at].uses = uses
        active, used, peak, evicted = np.array(sorted(rank), dtype=np.int64), 0, 0, []
        for batch, at in _rows(self.dataset, self.partition, self.stop):
            selected = np.flatnonzero(np.isin(at, active))
            if not len(selected):
                continue
            selected = selected[np.argsort(at[selected], kind="stable")]
            moments = at[selected]
            starts = np.flatnonzero(np.r_[True, moments[1:] != moments[:-1]])
            for begin, end in zip(starts, np.r_[starts[1:], len(selected)], strict=True):
                used += sessions[int(moments[begin])].add(batch, selected[begin:end])
            while used > self.max_block_bytes:
                # Se descarta la última sesión del bloque. Las anteriores siguen completas.
                victim = max((int(at) for at in active), key=rank.get)
                if rank[victim] == 0:
                    raise ValueError("Una sesión no cabe en el presupuesto del bloque")
                used -= sessions.pop(victim).bytes
                active = active[active != victim]
                evicted.append(victim)
            peak = max(peak, used)
        for at, session in sessions.items():
            if session.rows != expected[at]:
                raise ValueError("Una sesión no conserva su número de filas")
        done = min((rank[at] for at in evicted), default=len(block))
        rows = sum(session.rows for session in sessions.values())
        if rows:
            self.estimate = max(1, math.ceil(used / rows))
        self.statistics.update(
            passes=self.statistics["passes"] + 1,
            evicted=self.statistics["evicted"] + len(evicted),
            peak_block_bytes=max(self.statistics["peak_block_bytes"], peak),
        )
        return done, sessions

    def cohorts(self, positions):
        """Entregar las cohortes de las posiciones en su orden, leyendo por bloques."""
        positions = list(positions)
        if self.closed or any(
            type(position) is not int or not 0 <= position < len(self) for position in positions
        ):
            raise ValueError("Las posiciones no pertenecen a una fuente abierta")
        start = 0
        while start < len(positions):
            done, sessions = self._collect(positions[start : self._plan(positions, start)])
            for position in positions[start : start + done]:
                at = self.index[position][0]
                session = sessions[at]
                raw, presence = session.raw(at)
                session.uses -= 1
                if not session.uses:
                    del sessions[at]
                if presence is not None:
                    check_presence(presence, raw["inputs"])
                checked = checked_cohort(
                    raw,
                    presence,
                    shapes=self.shapes,
                    max_assets=self.max_assets,
                    market_bounds=self.market_bounds,
                )
                self.statistics["cohorts"] += 1
                yield {key: value for key, value in checked.items() if key != "sha256"}
            start += done

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
