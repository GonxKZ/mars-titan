"""Índice temporal de posiciones del corpus, sin duplicar modalidades ni calcular labels."""

import fcntl
import os
import tempfile
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan import nvtx_ranges
from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments import corpus_source
from mars_titan.environments.corpus_source import _ordered_parquet
from mars_titan.models.titans.config import MAX_BLOCK_ROWS
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch
from mars_titan.training import corpus_inputs
from mars_titan.training.corpus_inputs import (
    CorpusDataset,
    _fill_batch,
    _historical_times,
    _new_batch,
    _populated_groups,
    _window_contexts,
)
from mars_titan.training.input_pipeline import background, drain, submit
from mars_titan.training.partition_contract import LEGACY_BOUNDS

from .financial_session import FinancialPhase

_FIELDS = ("event_at", "kind", "asset_id", "source_group", "source_row", "sample_at")
_SCHEMA = pa.schema([(name, pa.int64()) for name in _FIELDS])
_MAX_ROWS = 100_000_000
_MAX_BYTES = 4 * 1024**3
# Instantes cuyos grupos se piden antes de montarlos cuando hay hilos de decodificación.
EVENTS_AHEAD = 4


def _identity(dataset, phase):
    if (
        type(dataset) is not CorpusDataset
        or not dataset.masked
        or type(phase) is not FinancialPhase
    ):
        raise ValueError("El índice necesita un corpus histórico y una fase explícitos")
    for market in {a["market"] for a in dataset.assets}:
        temporal = dataset.temporals.get(market)
        bounds = (
            {
                name: (int(start), int(end), int(cutoff))
                for name, start, end, cutoff in temporal.partitioner.bounds
            }
            if temporal
            else LEGACY_BOUNDS
        )
        if phase.partition not in bounds:
            raise ValueError("La partición no existe en esta supervisión")
        lower, upper, _ = bounds[phase.partition]
        if (
            not lower <= phase.decision_start < phase.decision_end <= upper
            or phase.close_at > upper
        ):
            raise ValueError("La fase no conserva los límites de la supervisión")
    if phase.warmup_start < 946_684_800_000_000:
        raise ValueError("La edición histórica comienza en 2000")
    return dict(
        schema_version=1,
        recipe="financial_observation_positions_v2",
        source_sha256=dataset.identity,
        phase=asdict(phase),
        threads=2,
        memory_limit="256MiB",
        max_spill_bytes=32 * 1024**3,
        max_index_bytes=_MAX_BYTES,
        max_records=_MAX_ROWS,
        pyarrow=pa.__version__,
        numpy=np.__version__,
        duckdb=duckdb.__version__,
        code={
            "financial_observations": sha256(Path(__file__)),
            "corpus_inputs": sha256(Path(corpus_inputs.__file__)),
            "external_order": sha256(Path(corpus_source.__file__)),
        },
    )


def _assets(dataset):
    return sorted(dataset.assets, key=lambda row: (row["market"], row["symbol"]))


def _table(columns):
    return pa.Table.from_pydict(columns, schema=_SCHEMA)


def _records(dataset, phase):
    total = 0
    for identity, asset in enumerate(_assets(dataset)):
        path = dataset._file(asset, "samples")
        with pq.ParquetFile(path) as file:
            if file.metadata.num_rows > 1_000_000:
                raise ValueError("Un activo supera el presupuesto de muestras")
            previous = None
            for group in _populated_groups(file):
                stamps = _historical_times(
                    file.read_row_group(group, columns=["prediction_at"], use_threads=False)
                )
                if len(stamps) and (
                    np.any(np.diff(stamps) <= 0) or (previous is not None and stamps[0] <= previous)
                ):
                    raise ValueError("Las muestras necesitan orden temporal único")
                if len(stamps):
                    previous = stamps[-1]
                rows = np.flatnonzero(
                    (stamps >= phase.warmup_start) & (stamps < phase.decision_end)
                )
                total += len(rows)
                if total >= _MAX_ROWS:
                    raise ValueError("El índice supera su número máximo de registros")
                yield _table(
                    dict(
                        event_at=stamps[rows],
                        kind=np.zeros(len(rows), np.int64),
                        asset_id=np.full(len(rows), identity, np.int64),
                        source_group=np.full(len(rows), group, np.int64),
                        source_row=rows.astype(np.int64),
                        sample_at=stamps[rows],
                    )
                )
            positions, stamps, _, maturity = dataset._labels(
                asset, phase.partition, file.metadata.num_rows
            )
            selected = np.flatnonzero(
                (stamps >= phase.decision_start)
                & (stamps < phase.decision_end)
                & (maturity < phase.decision_end)
            )
            total += len(selected)
            if total >= _MAX_ROWS:
                raise ValueError("El índice supera su número máximo de registros")
            yield _table(
                dict(
                    event_at=maturity[selected],
                    kind=np.ones(len(selected), np.int64),
                    asset_id=np.full(len(selected), identity, np.int64),
                    source_group=np.full(len(selected), -1, np.int64),
                    source_row=positions[selected],
                    sample_at=stamps[selected],
                )
            )
        dataset._file(asset, "samples")
        dataset._file(asset, "labels")
    # Esta fila es una instrucción administrativa, sin activo, input ni etiqueta.
    yield _table(
        dict(
            event_at=[phase.close_at],
            kind=[2],
            asset_id=[-1],
            source_group=[-1],
            source_row=[-1],
            sample_at=[-1],
        )
    )


def prepare_observation_index(dataset, output, *, phase, resume=False):
    identity, output = _identity(dataset, phase), Path(output)
    if type(resume) is not bool or output.exists() != resume:
        raise ValueError("El índice requiere salida nueva o recuperación explícita")
    for protected in (*dataset.roots.values(), dataset.path.parent):
        outside_source(protected, output)
        outside_source(output, protected)
    safe_destination(output)
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker, manifest = output / "identity.json", output / "manifest.json"
        if marker.exists():
            if read_manifest(marker)[0] != identity:
                raise ValueError("El índice corresponde a otra fuente, código o fase")
        elif resume:
            raise ValueError("Falta la identidad del índice recuperable")
        else:
            atomic_json(marker, identity)
        if manifest.exists():
            FinancialObservationSource(dataset, manifest)
            return manifest
        with tempfile.TemporaryDirectory(prefix="index-pending-", dir=output) as folder:
            temporary = Path(folder)
            raw, ordered = temporary / "raw.parquet", temporary / "ordered.parquet"
            count = atomic_parquet_batches(raw, _records(dataset, phase))
            if raw.stat().st_size > _MAX_BYTES:
                raise ValueError("El índice sin ordenar supera 4 GiB")
            with _ordered_parquet(
                raw,
                ordered,
                temporary,
                columns=("event_at", "kind", "asset_id", "sample_at"),
                threads=2,
                memory_limit="256MiB",
            ) as connection:
                groups = connection.execute(
                    "SELECT event_at, count(*) FROM read_parquet(?) "
                    "GROUP BY event_at ORDER BY event_at",
                    [str(ordered)],
                ).fetchmany(100_001)
            if (
                len(groups) > 100_000
                or sum(row[1] for row in groups) != count
                or ordered.stat().st_size > _MAX_BYTES
            ):
                raise ValueError("Los eventos no concilian o exceden el presupuesto del índice")
            digest = sha256(ordered)
            destination = output / f"events-{digest}.parquet"
            safe_destination(destination)
            if destination.exists():
                if sha256(destination) != digest:
                    raise ValueError("El índice existente conserva otros bytes")
            else:
                with ordered.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.link(ordered, destination)
            if sha256(dataset.path) != dataset.identity or _identity(dataset, phase) != identity:
                raise ValueError("El origen o el código cambiaron durante la ordenación")
            report = dict(
                kind="financial_observation_index",
                identity=identity,
                source_manifest=str(dataset.path.resolve()),
                events_path=destination.name,
                events_sha256=digest,
                bytes=destination.stat().st_size,
                rows=count,
                groups=groups,
                final_test_opened=False,
            )
            atomic_json(manifest, report)
        return manifest
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ObservationEvent:
    at: int
    inputs: tuple
    labels: tuple
    close_phase: bool


def _observation(dataset, asset, decoded, row, at, prices):
    """Seleccionar una fila decodificada con las mismas operaciones en ambas lecturas."""
    stamps, ends, vectors, presence, known, valid = decoded
    if not 0 <= row < len(stamps) or stamps[row] != at:
        raise ValueError("La posición no corresponde a la observación indexada")
    prices, price_at = prices
    rows = np.array([row], dtype=np.int64)
    return dict(
        vectors=vectors,
        rows=rows,
        prices=dataset.price_windows(asset, prices, price_at, ends[rows]),
        key=f"{asset['market']}/{asset['symbol']}",
        prediction_at=stamps[rows],
        sample_at=stamps[rows],
        price_available_at=price_at[ends[rows]],
        input_available_at=known[rows],
        availability_valid=valid[rows],
        presence=presence[rows],
    )


# Ventanas que `_window_contexts` transforma de una vez, el mismo tope que en `corpus_inputs`.
_WINDOW_ROWS = 256


def _block_window_contexts(windows, ends, context):
    """Ventanas de un bloque de hasta MAX_BLOCK_ROWS filas, por tramos de `_WINDOW_ROWS`.

    El resultado de `_window_contexts` por fila no depende de las demás filas, así que
    transformar por tramos da los mismos bits que una sola llamada, y un bloque de la receta
    mayor que 256 filas ya no choca con su límite de memoria.
    """
    ends = np.asarray(ends)
    return np.concatenate(
        [
            _window_contexts(
                windows[start : start + _WINDOW_ROWS], ends[start : start + _WINDOW_ROWS], context
            )
            for start in range(0, len(ends), _WINDOW_ROWS)
        ]
    )


class _BlockReader:
    """Conservar el último grupo decodificado de cada activo y montar bloques por instante.

    Cada fila se selecciona con las mismas comprobaciones que `_observation` y `_fill_batch`
    en la lectura por observación, y las ventanas de precios de un bloque se transforman
    juntas con `_price_contexts`, cuyo resultado por fila no depende del bloque. Si la
    edición declara ventanas por sesión del calendario, se forman activo a activo con
    `price_windows`, como en `_observation`. Con
    `executor`, los grupos que piden los instantes siguientes se decodifican antes en
    hilos, y cada grupo sigue siendo el resultado de `_sample_group` sobre el mismo archivo.

    Cada instante recorre los activos de su mercado en el mismo orden, un acceso cíclico en
    el que descartar el grupo usado hace más tiempo descarta justo el siguiente que se va a
    pedir, y sin sitio para todos ningún grupo llegaría a reutilizarse. Al superar el
    presupuesto se descartan primero los grupos de activos que no aparecen desde hace
    `STALE_EVENTS` instantes y después los usados más recientemente. Así, con un
    presupuesto menor que el conjunto activo, se sigue reutilizando la parte que cabe.
    """

    # Instantes sin aparecer tras los que un activo se considera fuera del recorrido. Cubre
    # la alternancia de mercados y los cierres de varios días de uno de ellos.
    STALE_EVENTS = 16

    def __init__(self, source, block_rows, max_cached_bytes, executor=None):
        if type(block_rows) is not int or not 1 <= block_rows <= MAX_BLOCK_ROWS:
            raise ValueError(f"El bloque de activos debe tener entre 1 y {MAX_BLOCK_ROWS} filas")
        if type(max_cached_bytes) is not int or not 1024**2 <= max_cached_bytes <= 8 * 1024**3:
            raise ValueError("La caché de grupos debe estar entre 1 MiB y 8 GiB")
        self.source, self.block_rows, self.limit = source, block_rows, max_cached_bytes
        self.groups, self.cached_bytes, self.decoded_groups = OrderedDict(), 0, 0
        self.peak_cached_bytes, self.redecoded_groups = 0, 0
        self.event, self._used = 0, {}
        self._labels, self._prices, self._group_counts = {}, {}, {}
        self._executor, self._pending, self._last = executor, {}, {}

    def labels(self, identity):
        """Las etiquetas y los precios se leen una vez por activo y recorrido.

        El índice confirma de nuevo todos los archivos al terminar el recorrido.
        """
        if identity not in self._labels:
            self._labels[identity] = self.source._label_arrays(identity)
        return self._labels[identity]

    def _file(self, identity):
        """Ruta comprobada del activo y su número de grupos, leído una vez por recorrido.

        No se guardan los metadatos Parquet completos. En memoria ocupan unos 0,9 MiB por
        archivo de muestras, ocho veces su tamaño serializado, y retenerlos para los cerca de
        5.000 activos de la ventana conjunta superaba los 4 GiB. Volver a leer el pie del
        archivo en cada decodificación cuesta en torno a 1 ms.
        """
        asset = self.source._assets[identity]
        path = self.source.dataset._file(asset, "samples")
        if identity not in self._group_counts:
            with pq.ParquetFile(path) as file:
                self._group_counts[identity] = file.metadata.num_row_groups
        return asset, path, self._group_counts[identity]

    def _decode(self, asset, path, group):
        """Decodificar un grupo. No usa estado del lector, así que puede ir en un hilo."""
        with nvtx_ranges.phase("reader.decode"), pq.ParquetFile(path) as file:
            _, *arrays = self.source.dataset._sample_group(asset, file, group)
        stamps, ends, vectors, *rest = arrays
        size = sum(v.nbytes for v in (stamps, ends, *vectors.values(), *rest) if v is not None)
        return arrays, size

    def schedule(self, rows):
        """Adelantar la decodificación de los grupos que piden estas filas del índice.

        Las filas inválidas se omiten aquí: el recorrido las rechaza en su instante.
        """
        if self._executor is None:
            return
        for _, kind, identity, group, *_ in rows:
            if kind != 0 or not 0 <= identity < len(self.source._assets):
                continue
            cached = self.groups.get(identity)
            if (cached is not None and cached[0] == group) or (identity, group) in self._pending:
                continue
            asset, path, count = self._file(identity)
            if 0 <= group < count:
                self._pending[identity, group] = submit(
                    self._executor, self._decode, asset, path, group
                )

    def cancel(self):
        """Cancelar las decodificaciones adelantadas y esperar a las que están en curso."""
        pending, self._pending = list(self._pending.values()), {}
        drain(pending)

    def _decoded(self, identity, group):
        self._used[identity] = self.event
        cached = self.groups.get(identity)
        if cached is not None and cached[0] == group:
            self.groups.move_to_end(identity)
            return cached[1]
        asset, path, count = self._file(identity)
        if not 0 <= group < count:
            raise ValueError("El grupo de origen no existe")
        future = self._pending.pop((identity, group), None)
        arrays, size = future.result() if future is not None else self._decode(asset, path, group)
        if cached is not None:
            self.cached_bytes -= self.groups.pop(identity)[2]
        self._evict(size)
        self.groups[identity] = (group, arrays, size)
        self.cached_bytes += size
        self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)
        self.decoded_groups += 1
        # El recorrido pide los grupos de cada activo en orden, así que repetir el último
        # grupo decodificado de un activo solo ocurre si la caché lo había descartado.
        if self._last.get(identity) == group:
            self.redecoded_groups += 1
        self._last[identity] = group
        return arrays

    def _evict(self, size):
        """Dejar sitio para `size` bytes: primero activos ausentes y después los recientes."""
        horizon = self.event - self.STALE_EVENTS
        while self.groups and self.cached_bytes + size > self.limit:
            oldest = next(iter(self.groups))
            stale = self._used[oldest] < horizon
            self.cached_bytes -= self.groups.popitem(last=not stale)[1][2]

    def _asset_prices(self, identity):
        if identity not in self._prices:
            self._prices[identity] = self.source.dataset._prices(self.source._assets[identity])
        return self._prices[identity]

    def blocks(self, positions):
        self.event += 1
        result = []
        for start in range(0, len(positions), self.block_rows):
            result.append(self._block(positions[start : start + self.block_rows]))
        return result

    def _block(self, chunk):
        """Un lote con las filas de `chunk`, igual al de `_observation` y `_fill_batch` por fila.

        Las comprobaciones de cada fila son las mismas. Las de valores no finitos, fechas y
        disponibilidad se aplican al lote, sobre los mismos valores copiados.
        """
        dataset, batch, widths = self.source.dataset, None, None
        context = dataset.context
        ends, windows, available, valid = [], [], [], []
        for filled, (identity, group, row, at) in enumerate(chunk):
            decoded = self._decoded(identity, group)
            stamps, group_ends, vectors, presence, known, known_valid = decoded
            shape = {name: value.shape[1] for name, value in vectors.items()}
            if widths is not None and shape != widths:
                raise ValueError("Las dimensiones cambian entre activos")
            widths = shape
            prices, price_at = self._asset_prices(identity)
            if not 0 <= row < len(stamps) or stamps[row] != at:
                raise ValueError("La posición no corresponde a la observación indexada")
            if batch is None:
                batch = _new_batch(
                    vectors,
                    context,
                    len(chunk),
                    masked=True,
                    supervised=False,
                    channels=dataset.price_channels,
                )
            end = group_ends[row]
            ends.append(end)
            windows.append(prices)
            available.append(price_at[end])
            valid.append(known_valid[row])
            for name, values in vectors.items():
                batch["inputs"][name][filled] = values[row]
            batch["prediction_at"][filled : filled + 1] = stamps[row : row + 1]
            batch["input_available_at"][filled : filled + 1] = known[row : row + 1]
            batch["presence"][filled] = presence[row]
            asset = self.source._assets[identity]
            key = f"{asset['market']}/{asset['symbol']}"
            batch["sample_ids"].append(f"{key}/{stamps[row]}")
            batch["market"].append(key.split("/", 1)[0])
        moments = batch["prediction_at"].astype(np.int64)
        if (np.asarray(available) > moments).any():
            raise ValueError("Las modalidades y la etiqueta no coinciden temporalmente")
        if not np.all(valid) or (batch["input_available_at"].astype(np.int64) > moments).any():
            raise ValueError("La disponibilidad de una modalidad es ausente o futura")
        for values in batch["inputs"].values():
            if values.ndim == 2 and not np.isfinite(values).all():
                raise ValueError("Una modalidad contiene valores no finitos")
        if dataset.price_window is None:
            batch["inputs"]["prices"][:] = _block_window_contexts(windows, ends, context)
            return batch
        # Con el contrato de ventanas por sesión del calendario, cada activo forma sus ventanas
        # con su calendario y sus ausencias de mercado, igual que en `_observation`.
        rows = {}
        for filled, (identity, *_) in enumerate(chunk):
            rows.setdefault(identity, []).append(filled)
        for identity, filled in rows.items():
            prices, price_at = self._asset_prices(identity)
            batch["inputs"]["prices"][filled] = dataset.price_windows(
                self.source._assets[identity], prices, price_at, np.asarray(ends)[filled]
            )
        return batch


class FinancialObservationSource:
    def __init__(self, dataset, manifest):
        self.dataset, self.path = dataset, Path(manifest)
        self.metadata, self.identity = read_manifest(self.path, 8 * 1024**2)
        meta = self.metadata
        self.phase = FinancialPhase(**meta["identity"]["phase"])
        if (
            meta["kind"] != "financial_observation_index"
            or meta["identity"] != _identity(dataset, self.phase)
            or meta["final_test_opened"] is not False
            or meta["events_path"] != f"events-{meta['events_sha256']}.parquet"
            or type(meta["rows"]) is not int
            or not 1 <= meta["rows"] <= _MAX_ROWS
            or type(meta["bytes"]) is not int
            or not 0 < meta["bytes"] <= _MAX_BYTES
        ):
            raise ValueError("El índice no conserva su identidad y presupuesto")
        self._assets = _assets(dataset)
        self._validate_groups(self._confirm())

    def _validate_groups(self, path):
        """Conciliar el índice completo antes de ofrecer la primera observación."""
        groups = []
        with pq.ParquetFile(path) as file:
            if file.schema_arrow != _SCHEMA or file.metadata.num_rows != self.metadata["rows"]:
                raise ValueError("El índice cambió su esquema o recuento")
            for batch in file.iter_batches(
                batch_size=8192, columns=["event_at"], use_threads=False
            ):
                column = batch.column(0)
                if column.null_count:
                    raise ValueError("El índice contiene eventos sin fecha")
                stamps = column.to_numpy()
                if len(stamps) and np.any(np.diff(stamps) < 0):
                    raise ValueError("Los eventos no conservan el orden temporal")
                unique, counts = np.unique(stamps, return_counts=True)
                for at, count in zip(unique, counts, strict=True):
                    at, count = int(at), int(count)
                    if groups and at < groups[-1][0]:
                        raise ValueError("Un grupo retrocede respecto al índice")
                    if groups and at == groups[-1][0]:
                        groups[-1][1] += count
                    else:
                        groups.append([at, count])
                    if len(groups) > 100_000 or groups[-1][1] > 16385:
                        raise ValueError("El índice supera su presupuesto de eventos")
        if groups != self.metadata["groups"] or not groups or groups[-1][0] != self.phase.close_at:
            raise ValueError("El resumen no corresponde a todas las filas del índice")

    def _confirm(self):
        path = self.path.parent / self.metadata["events_path"]
        safe_destination(path)
        if (
            path.stat().st_size != self.metadata["bytes"]
            or sha256(path) != self.metadata["events_sha256"]
            or sha256(self.path) != self.identity
            or sha256(self.dataset.path) != self.dataset.identity
            or _identity(self.dataset, self.phase) != self.metadata["identity"]
        ):
            raise ValueError("El índice o el origen cambiaron desde su confirmación")
        for asset in self._assets:
            for kind in ("prices", "samples", "labels"):
                self.dataset._file(asset, kind)
        return path

    def _input(self, identity, group, row, at):
        asset = self._assets[identity]
        path = self.dataset._file(asset, "samples")
        with pq.ParquetFile(path) as file:
            if not 0 <= group < file.num_row_groups:
                raise ValueError("El grupo de origen no existe")
            _, *decoded = self.dataset._sample_group(asset, file, group)
            block = _observation(self.dataset, asset, decoded, row, at, self.dataset._prices(asset))
            batch = _new_batch(
                decoded[2],
                self.dataset.context,
                1,
                masked=True,
                supervised=False,
                channels=self.dataset.price_channels,
            )
            _fill_batch(batch, 0, block, 0, 1)
            return batch

    def _label_arrays(self, identity):
        asset = self._assets[identity]
        path = self.dataset._file(asset, "samples")
        with pq.ParquetFile(path) as file:
            return self.dataset._labels(asset, self.phase.partition, file.metadata.num_rows)

    def _label(self, identity, row, at, sample_at, arrays=None):
        asset = self._assets[identity]
        positions, stamps, values, maturity = (
            self._label_arrays(identity) if arrays is None else arrays
        )
        index = int(np.searchsorted(positions, row))
        if (
            index == len(positions)
            or positions[index] != row
            or stamps[index] != sample_at
            or maturity[index] != at
        ):
            raise ValueError("La maduración indexada no corresponde al label verificado")
        return f"{asset['market']}/{asset['symbol']}", sample_at, float(values[index])

    def _event(self, at, rows, reader=None):
        inputs, labels, close, seen = [], [], False, set()
        for _, kind, identity, group, row, sample_at in rows:
            if kind == 2:
                if (
                    at != self.phase.close_at
                    or close
                    or (identity, group, row, sample_at) != (-1, -1, -1, -1)
                ):
                    raise ValueError(
                        "El cierre indexado no es la instrucción administrativa declarada"
                    )
                close = True
                continue
            if not 0 <= identity < len(self._assets) or (kind, identity, sample_at) in seen:
                raise ValueError("El índice repite o no identifica un evento")
            seen.add((kind, identity, sample_at))
            if kind == 0:
                if (
                    not self.phase.warmup_start <= at == sample_at < self.phase.decision_end
                    or len(inputs) >= 8192
                ):
                    raise ValueError("La observación indexada queda fuera del tramo o capacidad")
                if reader is None:
                    inputs.append(self._input(identity, group, row, at))
                else:
                    inputs.append((identity, group, row, at))
            elif kind == 1:
                if (
                    not self.phase.decision_start <= sample_at < at < self.phase.decision_end
                    or group != -1
                    or len(labels) >= 8192
                ):
                    raise ValueError("La maduración indexada queda fuera del tramo o capacidad")
                arrays = None if reader is None else reader.labels(identity)
                labels.append(self._label(identity, row, at, sample_at, arrays))
            else:
                raise ValueError("El índice contiene un tipo de evento desconocido")
        if reader is not None:
            inputs = reader.blocks(inputs)
        return ObservationEvent(at, tuple(inputs), tuple(labels), close)

    def events(self, *, start_cursor=0):
        """Entregar una observación por lote, como la referencia de paridad del lector."""
        for at, rows in self._logical_events(start_cursor):
            yield self._event(at, rows)

    def batched_events(self, *, start_cursor=0, block_rows=256, max_cached_bytes=None):
        """Entregar el mismo conjunto y orden canónico en bloques de activos por instante.

        Cada grupo Parquet se decodifica una vez mientras sus filas siguen en uso, si la
        caché alcanza un grupo por activo activo en el instante. Sin `max_cached_bytes`, el
        presupuesto es el de la tubería de la edición (`PipelineOptions.group_cache_bytes`).
        Con hilos de decodificación, los grupos de los instantes siguientes se decodifican
        antes, y con prefetch los instantes se preparan en un hilo productor. El conjunto,
        el orden y los errores no cambian.
        """
        pipeline = self.dataset.pipeline
        if max_cached_bytes is None:
            max_cached_bytes = pipeline.group_cache_bytes
        executor = self.dataset._executor() if pipeline.decode_workers else None
        reader = _BlockReader(self, block_rows, max_cached_bytes, executor)
        self.last_reader = reader
        stream = self._prepared_events(
            self._logical_events(start_cursor), reader, ahead=EVENTS_AHEAD if executor else 0
        )
        if pipeline.prefetch_batches:
            stream = background(stream, pipeline.prefetch_batches)
        yield from stream

    def _prepared_events(self, logical, reader, *, ahead):
        """Montar cada instante y adelantar la decodificación de los `ahead` siguientes.

        Un error del índice se lanza después de entregar los instantes anteriores.
        """
        window, failure = deque(), None
        try:
            while True:
                try:
                    at, rows = next(logical)
                except StopIteration:
                    break
                except BaseException as error:  # Se lanza en su posición, tras los anteriores.
                    failure = error
                    break
                reader.schedule(rows)
                window.append((at, rows))
                if len(window) > ahead:
                    with nvtx_ranges.phase("reader.event"):
                        event = self._event(*window.popleft(), reader)
                    yield event
            while window:
                with nvtx_ranges.phase("reader.event"):
                    event = self._event(*window.popleft(), reader)
                yield event
            if failure is not None:
                raise failure
        finally:
            reader.cancel()
            logical.close()

    def _logical_events(self, start_cursor):
        """Agrupar filas del índice por instante y conciliar su resumen sin decodificar."""
        if type(start_cursor) is not int or not 0 <= start_cursor <= len(self.metadata["groups"]):
            raise ValueError("El cursor de eventos no pertenece al índice")
        path = self._confirm()
        groups, rows, previous, count = [], [], None, 0
        with pq.ParquetFile(path) as file:
            if file.schema_arrow != _SCHEMA or file.metadata.num_rows != self.metadata["rows"]:
                raise ValueError("El índice cambió su esquema o recuento")
            for batch in file.iter_batches(batch_size=8192, use_threads=False):
                if any(column.null_count for column in batch.columns):
                    raise ValueError("El índice contiene posiciones ausentes")
                columns = [column.to_numpy(zero_copy_only=False) for column in batch.columns]
                for values in zip(*columns, strict=True):
                    item = tuple(int(value) for value in values)
                    at = item[0]
                    if previous is not None and at < previous:
                        raise ValueError("El índice ha perdido su orden temporal")
                    if previous is not None and at != previous:
                        groups.append([previous, len(rows)])
                        if len(groups) > start_cursor:
                            yield previous, rows
                        rows = []
                    rows.append(item)
                    if len(rows) > 16385:
                        raise ValueError("El evento lógico supera su presupuesto")
                    previous, count = at, count + 1
        if rows:
            groups.append([previous, len(rows)])
            if len(groups) > start_cursor:
                yield previous, rows
        if (
            groups != self.metadata["groups"]
            or count != self.metadata["rows"]
            or previous != self.phase.close_at
        ):
            raise ValueError("La cronología no completa el índice y su cierre")
        self._confirm()

    def label_decisions(self):
        """Devuelve, por activo, los instantes de decisión cuya etiqueta madura en la fase.

        Se leen las maduraciones del propio índice, sin decodificar entradas ni etiquetas.
        Devuelve `{"mercado/símbolo": instantes ordenados}` con las mismas claves que
        las etiquetas de los eventos.
        """
        path = self._confirm()
        table = pq.read_table(
            path, columns=["kind", "asset_id", "sample_at"], filters=[("kind", "=", 1)]
        )
        identities = table["asset_id"].to_numpy()
        stamps = table["sample_at"].to_numpy()
        result = {}
        for identity in np.unique(identities):
            asset = self._assets[int(identity)]
            moments = np.sort(stamps[identities == identity])
            if np.any(np.diff(moments) == 0):
                raise ValueError("El índice repite la maduración de una decisión")
            result[f"{asset['market']}/{asset['symbol']}"] = moments
        return result

    def specification(self):
        for event in self.events():
            if event.inputs:
                widths = {
                    name: values.shape[-1] for name, values in event.inputs[0]["inputs"].items()
                }
                return FinancialInputSpec(
                    source_sha256=self.dataset.identity,
                    view_sha256=self.identity,
                    representation=self.dataset.manifest["representation"],
                    dimensions=widths,
                    input_policy=self.dataset.manifest["input_policy"],
                )
        raise ValueError("La fase no contiene observaciones")


def run_observation_source(source, session):
    """Preparar una vez cada instante. La sesión conserva la única publicación."""
    if (
        type(source) is not FinancialObservationSource
        or session._input_spec.source_sha256 != source.dataset.identity
        or session._input_spec.view_sha256 != source.identity
        or session.phase != source.phase
    ):
        raise ValueError("La sesión no está vinculada al índice de observaciones y fase")
    cursor = session._executor.cursor
    if cursor > len(source.metadata["groups"]):
        raise ValueError("El cursor confirmado excede la cronología")
    if cursor and session.snapshot()["last_at"] != source.metadata["groups"][cursor - 1][0]:
        raise ValueError("El cursor no corresponde al instante del índice")
    pending = {(p.asset, p.decision_at): p.id for p in session._executor.pending()}
    for event in source.events(start_cursor=cursor):
        feedback = []
        for flow, at, label in event.labels:
            if (flow, at) not in pending:
                raise ValueError("El label no tiene una predicción emitida pendiente")
            feedback.append(session.native.Feedback(pending[flow, at], 0, event.at, label))
        batches = [validated_cpu_batch(raw, session._input_spec) for raw in event.inputs]
        kind = (
            ("warmup" if event.at < source.phase.decision_start else "decision")
            if batches
            else "settlement"
        )
        result = session.step(
            batches, feedback, kind=kind, cutoff=event.at, close_phase=event.close_phase
        )
        for prediction in result.predictions:
            pending[prediction.asset, prediction.decision_at] = prediction.id
        for item in (*result.applied, *result.excluded, *result.finalized):
            pending.pop((item.prediction.asset, item.prediction.decision_at))
    return session.diagnostics()
