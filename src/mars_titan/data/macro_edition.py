"""Componer ediciones macro por meses con sustituciones explícitas y fuentes inmutables."""

import argparse
import json
import os
import resource
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .macro_coverage import (
    _check_schema,
    _publish_directory,
    _read_catalog,
    _selected_batches,
)
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock


class _PanelCursor:
    """Consumir cada grupo una vez y conservar solo el lote pendiente de su mes."""

    def __init__(self, file, first, stop, allowed, owned):
        self.batches = _selected_batches(file, first, stop)
        self.allowed, self.owned = pa.array(allowed), pa.array(owned, type=pa.string())
        self.previous = None
        self.pending = None
        self.decoded_rows = 0
        self._advance()

    def _advance(self):
        self.pending = None
        for batch, _, _ in self.batches:
            indices = pc.index_in(batch["indicator_id"], value_set=self.allowed)
            if indices.null_count:
                raise ValueError("El panel contiene indicadores ajenos a su sustitución declarada")
            moments = batch["prediction_at"].to_numpy(zero_copy_only=False)
            if (
                np.any(moments[1:] < moments[:-1])
                or self.previous is not None
                and moments[0] < self.previous
            ):
                raise ValueError("El panel de entrada no está ordenado por decisión")
            self.previous = moments[-1]
            self.decoded_rows += len(batch)
            batch = batch.filter(pc.is_in(batch["indicator_id"], value_set=self.owned))
            if len(batch):
                self.pending = batch
                return

    def through(self, stop):
        while self.pending is not None:
            before = pc.less(self.pending["prediction_at"], stop)
            count = pc.sum(pc.cast(before, pa.int32())).as_py()
            if not count:
                return
            batch = self.pending.slice(0, count)
            self.pending = self.pending.slice(count)
            yield batch
            if len(self.pending):
                return
            self._advance()


def compose_macro_edition(
    base: Path,
    replacements: list[tuple[Path, list[str]]],
    catalog_path: Path,
    output: Path,
    *,
    market: str,
    start: str,
    end: str,
) -> dict:
    """Sustituir columnas completas sin rellenar ausencias ni tocar la edición anterior."""
    began = time.perf_counter()
    base, catalog_path, output = map(Path, (base, catalog_path, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("La edición macro ya existe")
    if not 1 <= len(replacements) <= 8:
        raise ValueError("La composición requiere entre uno y ocho paneles de sustitución")
    entries = _read_catalog(catalog_path)
    identifiers = sorted(entries)
    claimed, sources = set(), [base]
    groups = []
    for path, names in replacements:
        path = Path(path)
        if (
            not names
            or len(names) != len(set(names))
            or set(names) - set(entries)
            or claimed.intersection(names)
        ):
            raise ValueError("Las sustituciones se solapan o no pertenecen al catálogo")
        claimed.update(names)
        sources.append(path)
        groups.append(sorted(names))
    if len({path.resolve() for path in sources}) != len(sources):
        raise ValueError("Un archivo no puede ocupar dos posiciones de la composición")
    for source in sources + [catalog_path, Path("dataset")]:
        outside_source(source, output)
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise ValueError("El panel debe ser un archivo regular")
        if source.stat().st_size < 12:
            raise ValueError("El panel no tiene un pie Parquet completo")
        with source.open("rb") as stream:
            stream.seek(-8, 2)
            footer = stream.read(8)
        if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > min(
            8 * 1024**2, source.stat().st_size - 12
        ):
            raise ValueError("Los metadatos Parquet son inválidos o superan ocho MiB")
    first_day, last_day = date.fromisoformat(start), date.fromisoformat(end)
    if first_day > last_day or (last_day - first_day).days > 366 * 50:
        raise ValueError("El periodo está invertido o supera cincuenta años")
    first = datetime.combine(first_day, datetime.min.time(), UTC)
    stop = datetime.combine(last_day + timedelta(days=1), datetime.min.time(), UTC)
    clock = MarketClock(
        market,
        (first_day - timedelta(days=7)).isoformat(),
        (last_day + timedelta(days=7)).isoformat(),
    )
    decisions = [t for t in clock.decisions if first <= t < stop]
    if not decisions or len(decisions) * len(entries) > 8_000_000:
        raise ValueError("El calendario está vacío o supera el presupuesto de composición")
    moments = pa.array(decisions, type=pa.timestamp("us", tz="UTC"))
    ids = pa.array(identifiers)
    counts = np.zeros((len(decisions), len(entries)), dtype=np.uint16)
    catalog_hash = sha256(catalog_path)
    hashes = [sha256(path) for path in sources]
    groups.insert(0, sorted(set(identifiers) - claimed))
    output.parent.mkdir(parents=True, exist_ok=True)
    written = peak_block_bytes = 0
    with (
        tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary,
        ExitStack() as stack,
    ):
        stage = Path(temporary) / "edition"
        stage.mkdir()
        cursors = []
        for index, source in enumerate(sources):
            file = stack.enter_context(pq.ParquetFile(source))
            _check_schema(file.schema_arrow)
            if file.metadata.num_rows > 8_000_000:
                raise ValueError("El panel excede el presupuesto de filas")
            cursors.append(
                _PanelCursor(
                    file, first, stop, identifiers if index == 0 else groups[index], groups[index]
                )
            )
        writer, schema = None, None
        month_number = first_day.year * 12 + first_day.month - 1
        while True:
            month_number += 1
            bound = min(stop, datetime(month_number // 12, month_number % 12 + 1, 1, tzinfo=UTC))
            batches, block_bytes = [], 0
            for cursor in cursors:
                for batch in cursor.through(bound):
                    schema = schema or batch.schema
                    if batch.schema != schema:
                        raise ValueError("Los paneles deben conservar el mismo esquema y precisión")
                    block_bytes += batch.nbytes
                    if block_bytes > 64 * 1024**2:
                        raise ValueError("El bloque mensual excede 64 MiB")
                    d = pc.index_in(batch["prediction_at"], value_set=moments)
                    i = pc.index_in(batch["indicator_id"], value_set=ids)
                    if d.null_count or i.null_count:
                        raise ValueError("La decisión o el indicador no pertenece al contrato")
                    np.add.at(counts, (d.to_numpy(), i.to_numpy()), 1)
                    if counts.max() > 1:
                        raise ValueError(
                            "La composición contiene un indicador duplicado por decisión"
                        )
                    batches.append(batch)
            if batches:
                table = pa.Table.from_batches(batches).sort_by(
                    [("prediction_at", "ascending"), ("indicator_id", "ascending")]
                )
                if writer is None:
                    writer = stack.enter_context(
                        pq.ParquetWriter(stage / "macro.parquet", table.schema, compression="zstd")
                    )
                writer.write_table(table, row_group_size=8192)
                written += len(table)
                peak_block_bytes = max(peak_block_bytes, block_bytes)
            if bound == stop:
                break
        if np.any(counts != 1):
            raise ValueError("Faltan filas de indicadores en la composición")
        stack.close()
        with (stage / "macro.parquet").open("rb") as stream:
            os.fsync(stream.fileno())
        if [sha256(path) for path in sources] != hashes or sha256(catalog_path) != catalog_hash:
            raise ValueError("Un panel o el catálogo cambió durante la composición")
        report = dict(
            schema_version=1,
            market=market,
            start=start,
            end=end,
            rows=written,
            decisions=len(decisions),
            indicators=len(entries),
            catalog_sha256=catalog_hash,
            source_hashes=hashes,
            replaced_indicators=sorted(claimed),
            inputs=[
                dict(sha256=digest, indicators=group, decoded_rows=cursor.decoded_rows)
                for digest, group, cursor in zip(hashes, groups, cursors, strict=True)
            ],
            sha256=sha256(stage / "macro.parquet"),
            largest_month_bytes=peak_block_bytes,
            output_bytes=(stage / "macro.parquet").stat().st_size,
            input_file_bytes=sum(path.stat().st_size for path in sources),
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            elapsed_seconds=time.perf_counter() - began,
            population_ready=False,
            admission_required=True,
        )
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "catalog", "replacements", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args(argv)
    if args.replacements.stat().st_size > 1024**2:
        raise ValueError("La lista de sustituciones supera un MiB")
    replacements = json.loads(args.replacements.read_text())
    if not isinstance(replacements, list) or any(
        not isinstance(item, dict) or set(item) != {"path", "indicator_ids"}
        for item in replacements
    ):
        raise ValueError("La lista de sustituciones no cumple el contrato")
    report = compose_macro_edition(
        args.base,
        [(Path(r["path"]), r["indicator_ids"]) for r in replacements],
        args.catalog,
        args.output,
        market=args.market,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
