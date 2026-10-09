"""Vistas por calendario con entradas estrictas o máscaras históricas, sin copiar modalidades."""

import argparse
import copy
import json
import os
import resource
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.batches import read_bounded_table
from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.input_policy import (
    INPUT_POLICIES,
    STRICT_INPUTS,
    masked_inputs,
    policy_identity,
)
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.preparation import atomic_parquet
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.evaluation.split_readiness import _admission, _complete_dates
from mars_titan.evaluation.splits import (
    LABEL_INTERVAL_PURGE,
    PARTITIONS,
    FoldPartitioner,
    build_folds,
)

from .cohort_contract import cohort_identity, representation_identity, validate_cohort_rows
from .temporal_contract import validate_temporal_view

VIEW_REASONS = {
    "outside_window",
    "incomplete_inputs",
    "future_inputs",
    "label_crosses_boundary",
    "session_gap",
}


class TemporalInputs:
    """Compartir vectores por sesión y comprobar las identidades del contrato temporal."""

    def __init__(self, contract, *, input_policy=STRICT_INPUTS):
        self.masked = validate_temporal_view(contract, input_policy=input_policy)
        self.contract = contract
        self.protocol, self.fold = contract["protocol"], contract["fold"]
        folds = build_folds(self.protocol)
        if self.fold not in folds:
            raise ValueError("La ventana no pertenece al protocolo declarado")
        self.clock = MarketClock(
            self.protocol["market"], self.protocol["train_start"], self.protocol["final_test_end"]
        )
        self.partitioner = FoldPartitioner(self.fold, self.clock, self.protocol)
        if self.masked:
            parent, parent_hash = read_manifest(Path(contract["parent_manifest"]), 8 * 1024**2)
            cohort_identity(parent, input_policy=input_policy)
            if parent_hash != contract["parent_sha256"] or any(
                name in parent for name in ("temporal_view", "temporal_views")
            ):
                raise ValueError("El padre histórico no conserva su identidad anual")
            self.representation = representation_identity(
                parent["representation"], input_policy=input_policy
            )
            self.times = np.array(
                [
                    int(moment.timestamp() * 1_000_000)
                    for moment in self.clock.decisions
                    if moment.year < 2024
                ],
                dtype=np.int64,
            )
            self.macro = None
            self.verify()
            return
        self.admission_path = Path(contract["admission_path"])
        admission, identity = read_manifest(self.admission_path, 8 * 1024**2)
        if identity != contract["admission_sha256"]:
            raise ValueError("Ha cambiado la admisión de la vista temporal")
        _admission(admission, self.protocol)
        if admission["source_sha256"] != contract["macro_sha256"]:
            raise ValueError("El panel macro no corresponde a la admisión")
        self.times, _ = _complete_dates(self.admission_path, admission, self.protocol)
        if not len(self.times):
            raise ValueError("La vista necesita sesiones con todos los indicadores")
        self.macro = MacroVectors(Path(contract["macro_path"]), cutoff_year=2023)
        parent, parent_hash = read_manifest(Path(contract["parent_manifest"]), 8 * 1024**2)
        if parent_hash != contract["parent_sha256"]:
            raise ValueError("Ha cambiado el corpus de origen de la vista")
        declared = parent.get("representation", {}).get("macro_indicators")
        if declared is not None and self.macro.indicators != sorted(declared):
            raise ValueError("La vista no puede cambiar los conceptos macro del corpus de origen")
        if (
            self.macro.sha256 != contract["macro_sha256"]
            or self.macro.indicators != admission["required_indicator_ids"]
        ):
            raise ValueError("Los vectores macro no conservan el archivo y catálogo admitidos")
        self.values = np.empty((len(self.times), 3 * len(self.macro.indicators)), dtype=np.float32)
        self.available = np.empty(len(self.times), dtype=np.int64)
        lookup = {
            int(moment.timestamp() * 1_000_000): index for moment, index in self.macro.index.items()
        }
        for index, moment in enumerate(self.times):
            if int(moment) not in lookup:
                raise ValueError("Una sesión admitida no tiene vector macro")
            slot = lookup[int(moment)]
            self.values[index] = self.macro.values[slot]
            self.available[index] = int(self.macro.available[slot].timestamp() * 1_000_000)
        count = len(self.macro.indicators)
        if not np.all(self.values[:, count : 2 * count] == 1) or not np.isfinite(self.values).all():
            raise ValueError("Los vectores admitidos no contienen todos sus indicadores observados")
        if np.any(self.available > self.times):
            raise ValueError("El contexto macro utiliza información futura")
        self.verify()

    def verify(self):
        if not self.masked:
            self.macro.verify()
            if sha256(self.admission_path) != self.contract["admission_sha256"]:
                raise ValueError("Una fuente de la vista temporal ha cambiado")
        if sha256(Path(self.contract["parent_manifest"])) != self.contract["parent_sha256"]:
            raise ValueError("Una fuente de la vista temporal ha cambiado")

    def lookup(self, prediction):
        indices = np.searchsorted(self.times, prediction)
        safe = np.minimum(indices, len(self.times) - 1)
        valid = (indices < len(self.times)) & (self.times[safe] == prediction)
        if self.masked:
            return None, np.zeros(len(prediction), dtype=np.int64), valid
        values = np.zeros((len(prediction), self.values.shape[1]), dtype=np.float32)
        available = np.zeros(len(prediction), dtype=np.int64)
        values[valid], available[valid] = self.values[safe[valid]], self.available[safe[valid]]
        return values, available, valid

    def assign(self, prediction, available, maturity, eligible):
        result = dict(
            partition=np.full(len(prediction), "excluded", dtype="U13"),
            reason=np.full(len(prediction), "outside_window", dtype=object),
        )
        lower = np.datetime64(self.protocol["train_start"], "us").astype(np.int64)
        upper = np.datetime64(self.protocol["final_test_start"], "us").astype(np.int64)
        selected = (prediction >= lower) & (prediction < upper) & eligible
        if selected.any():
            _, macro_available, valid = self.lookup(prediction[selected])
            assigned = self.partitioner.assign(
                prediction[selected],
                np.maximum(available[selected], macro_available),
                maturity[selected],
                eligible=valid,
            )
            for name in result:
                result[name][selected] = assigned[name]
        reserved = prediction >= upper
        result["reason"][reserved] = "final_test_reserved"
        result["partition"][reserved] = "test_reserved"
        return result


def _sample_state(parent, asset, temporal):
    from .corpus_inputs import MAX_TABLE_BYTES, VECTORS, _availability, _times

    parts = []
    if parent.masked:
        return _masked_sample_state(parent, asset)
    with pq.ParquetFile(parent._file(asset, "samples")) as file:
        required = {"prediction_at", "price_end_index", *VECTORS, "input_availability"}
        if not required <= set(file.schema_arrow.names):
            raise ValueError("La muestra necesita las cuatro modalidades y su disponibilidad")
        metadata_columns = ["prediction_at", "input_availability", "price_end_index"]
        if "macro_available_at" in file.schema_arrow.names:
            metadata_columns.append("macro_available_at")
        for group in range(file.num_row_groups):
            size = file.metadata.row_group(group).total_byte_size
            if size > MAX_TABLE_BYTES:
                raise ValueError("Un grupo de muestras supera el presupuesto")
            table = file.read_row_group(group, columns=metadata_columns, use_threads=False)
            prediction = _times(table["prediction_at"])
            _, macro_available, macro_valid = temporal.lookup(prediction)
            available, valid = _availability(table, macro_override=(macro_available, macro_valid))
            if available is None:
                raise ValueError("La muestra no acredita disponibilidad de sus modalidades")
            selected = np.flatnonzero(valid)
            if len(selected):
                last = int(selected[-1]) + 1
                test_start = np.datetime64(temporal.protocol["final_test_start"], "us").astype(
                    np.int64
                )
                if (prediction[:last] >= test_start).any():
                    raise ValueError("El orden de muestras mezclaría valores del test reservado")
                batch = next(
                    file.iter_batches(
                        batch_size=last,
                        row_groups=[group],
                        columns=list(VECTORS),
                        use_threads=False,
                    )
                )
                if batch.nbytes > MAX_TABLE_BYTES:
                    raise ValueError("Las modalidades decodificadas superan el presupuesto")
                for name in ("news", "charts", "fundamentals"):
                    column = batch.column(name).take(pa.array(selected))
                    if (
                        not (
                            pa.types.is_list(column.type)
                            or pa.types.is_fixed_size_list(column.type)
                        )
                        or column.null_count
                    ):
                        raise ValueError("Falta el vector de una modalidad admitida")
                    lengths = pc.list_value_length(column).to_numpy()
                    if (
                        not np.all(lengths == lengths[0])
                        or not 1 <= lengths[0] <= 2048
                        or not np.isfinite(column.flatten().to_numpy()).all()
                    ):
                        raise ValueError("Una modalidad tiene dimensiones o valores inválidos")
            parts.append((prediction, available, valid))
    if not parts:
        raise ValueError("El activo no contiene muestras")
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(3))


def _masked_sample_state(parent, asset):
    """Conservar todas las entradas causales del padre, incluidas las ausencias explícitas."""
    from .corpus_inputs import (
        MAX_TABLE_BYTES,
        VECTORS,
        _availability,
        _historical_times,
        _presence,
        _vectors,
    )

    parts = []
    prices, price_available = parent._prices(asset)
    with pq.ParquetFile(parent._file(asset, "samples")) as file:
        if not 1 <= file.metadata.num_rows <= 1_000_000:
            raise ValueError("Las muestras históricas superan el presupuesto de filas")
        metadata = [
            "prediction_at",
            "price_end_index",
            "input_availability",
            "presence",
            "news_count",
            "cohort_id",
        ]
        columns = [*metadata, *VECTORS]
        if not set(columns) <= set(file.schema_arrow.names):
            raise ValueError("Falta el contrato de las muestras históricas")
        for group in range(file.num_row_groups):
            if file.metadata.row_group(group).total_byte_size > MAX_TABLE_BYTES:
                raise ValueError("Un grupo de muestras supera el presupuesto")
            table = file.read_row_group(group, columns=metadata, use_threads=False)
            prediction = _historical_times(table)
            validate_cohort_rows(table, parent.cohort)
            payload = file.read_row_group(group, columns=list(VECTORS), use_threads=False)
            if table.nbytes + payload.nbytes > MAX_TABLE_BYTES:
                raise ValueError("Las modalidades decodificadas superan el presupuesto")
            vectors = _vectors(payload, historical=True)
            presence = _presence(table, vectors, parent.manifest["representation"])
            available, valid = _availability(table, presence=presence)
            ends = table["price_end_index"]
            if not pa.types.is_integer(ends.type) or ends.null_count:
                raise ValueError("La ventana histórica necesita un índice de precios entero")
            ends = ends.to_numpy()
            if (ends < parent.context - 1).any() or (ends >= len(prices)).any():
                raise ValueError("La ventana histórica no tiene suficientes precios")
            available = np.maximum(available, price_available[ends])
            if not valid.all() or (available > prediction).any():
                raise ValueError("Una entrada observada tiene disponibilidad ausente o futura")
            parts.append((prediction, available, valid))
    if not parts:
        raise ValueError("El activo no contiene muestras")
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(3))


def _annual_candidates(labels, prediction, maturity, reasons, boundary):
    """Comprobar el único corte anual recuperable del contrato supervisado original."""
    candidates = reasons == "target_crosses_partition_boundary"
    if candidates.any():
        target = labels["target"].to_numpy()[candidates]
        if (
            not np.isfinite(target).all()
            or np.any(prediction[candidates] != boundary[0])
            or np.any(maturity[candidates] != boundary[1])
        ):
            raise ValueError("La exclusión anual no conserva un objetivo y una maduración válidos")
    return candidates


def prepare_temporal_corpus(
    parent_path,
    protocol_path,
    macro_path,
    admission_path,
    output,
    *,
    recover_annual_boundaries=False,
    input_policy=STRICT_INPUTS,
):
    """Publicar etiquetas por ventana y referencias a las modalidades originales."""
    from .corpus_inputs import CorpusDataset, _times

    if type(recover_annual_boundaries) is not bool:
        raise ValueError("La recuperación anual debe indicarse mediante un booleano")
    masked = masked_inputs(input_policy)
    if masked and not recover_annual_boundaries:
        raise ValueError("La edición histórica requiere recuperación anual explícita")
    if masked and (macro_path is not None or admission_path is not None):
        raise ValueError("La vista histórica conserva el macro del padre, sin otra admisión")
    began = time.perf_counter()
    parent_path, protocol_path, output = map(Path, (parent_path, protocol_path, output))
    if not masked:
        if macro_path is None or admission_path is None:
            raise ValueError("La política estricta requiere panel macro y admisión")
        macro_path, admission_path = Path(macro_path), Path(admission_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Las vistas temporales ya existen")
    parent = CorpusDataset(parent_path, input_policy=input_policy)
    if parent.temporal is not None or parent.manifest.get("final_test_opened") is True:
        raise ValueError("Se necesita un corpus original con el test reservado")
    for source in [
        *parent.roots.values(),
        protocol_path,
        Path("dataset"),
        *([] if masked else [macro_path, admission_path]),
    ]:
        outside_source(source, output)
    protocol, protocol_hash = read_manifest(protocol_path)
    folds = build_folds(protocol)
    interval_purge = protocol.get("purge") == LABEL_INTERVAL_PURGE
    annual_boundary = None
    if recover_annual_boundaries:
        clock = MarketClock(protocol["market"], "2022-12-01", "2023-01-31")
        last = max(moment for moment in clock.decisions if moment.year == 2022)
        following = min(moment for moment in clock.decisions if moment.year == 2023)
        annual_boundary = tuple(int(moment.timestamp() * 1_000_000) for moment in (last, following))
    contract = dict(
        schema_version=2 if masked else 1,
        **policy_identity(input_policy),
        protocol=protocol,
        fold=folds[0],
    )
    if masked:
        contract.update(selection_partition="validation", recover_annual_boundaries=True)
    else:
        contract.update(
            macro_path=str(macro_path.resolve()),
            macro_sha256=sha256(macro_path),
            admission_path=str(admission_path.resolve()),
            admission_sha256=sha256(admission_path),
        )
    contract.update(parent_manifest=str(parent_path.resolve()), parent_sha256=parent.identity)
    temporal = TemporalInputs(contract, input_policy=input_policy)
    views = {fold["id"]: copy.deepcopy(parent.manifest) for fold in folds}
    for fold in folds:
        view = views[fold["id"]]
        view.update(
            temporal_view={**contract, "fold": fold},
            counts={key: 0 for key in PARTITIONS},
            final_test_opened=False,
        )
        view["roots"]["labels"] = str((output / fold["id"] / "labels").resolve())
        if interval_purge:
            view["purged_by_boundary"] = {key: 0 for key in PARTITIONS}
        if recover_annual_boundaries:
            view["label_admission"] = dict(
                schema_version=1,
                recover_annual_boundaries=True,
                source_boundary="2023-01-01",
                target_horizon="next_market_session",
            )
            view["recovered_annual_labels"] = 0
    annual_candidates = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "views"
        stage.mkdir()
        for position, asset in enumerate(parent.assets):
            if asset["market"] != protocol["market"]:
                raise ValueError(
                    "La vista temporal requiere un único mercado con calendario propio"
                )
            prediction, available, valid = _sample_state(parent, asset, temporal)
            parent._labels(asset, "train", len(prediction))
            labels = read_bounded_table(parent._file(asset, "labels"), max_rows=1_000_000)
            indices = labels["sample_row"].to_numpy()
            label_prediction = _times(labels["prediction_at"])
            if not np.array_equal(prediction[indices], label_prediction):
                raise ValueError("Las etiquetas no corresponden a las decisiones de las muestras")
            source_reasons = np.asarray(
                labels["reason"].to_pylist()
                if "reason" in labels.column_names
                else ["accepted"] * len(labels),
                dtype="U40",
            )
            accepted = source_reasons == "accepted"
            maturity_column = labels["target_available_at"].fill_null(
                pa.scalar(0, type=labels["target_available_at"].type)
            )
            maturity = _times(maturity_column)
            recovered = np.zeros(len(labels), dtype=bool)
            if recover_annual_boundaries:
                recovered = _annual_candidates(
                    labels, label_prediction, maturity, source_reasons, annual_boundary
                )
                accepted |= recovered
                annual_candidates += int(recovered.sum())
            for fold in folds:
                temporal.fold = fold
                temporal.partitioner = FoldPartitioner(fold, temporal.clock, protocol)
                assigned = temporal.assign(label_prediction, available[indices], maturity, accepted)
                assigned["partition"][~valid[indices] & accepted] = "excluded"
                future = accepted & (available[indices] > label_prediction)
                assigned["reason"][~valid[indices] & accepted] = "incomplete_inputs"
                assigned["reason"][future] = "future_inputs"
                assigned["partition"][future] = "excluded"
                assigned["reason"][~accepted] = source_reasons[~accepted]
                reasons = assigned["reason"]
                used = reasons == "accepted"
                partition = pa.array(
                    [
                        value if keep else None
                        for value, keep in zip(assigned["partition"], used, strict=True)
                    ],
                    type=pa.string(),
                )
                updated = labels.set_column(
                    labels.schema.get_field_index("target"),
                    "target",
                    pc.if_else(pa.array(used), labels["target"], None),
                )
                updated = updated.set_column(
                    updated.schema.get_field_index("partition"), "partition", partition
                )
                if "reason" in updated.column_names:
                    updated = updated.set_column(
                        updated.schema.get_field_index("reason"), "reason", pa.array(reasons)
                    )
                else:
                    updated = updated.append_column("reason", pa.array(reasons))
                if recover_annual_boundaries:
                    updated = updated.append_column("source_reason", pa.array(source_reasons))
                folder = stage / fold["id"] / "labels" / asset["market"] / asset["symbol"]
                atomic_parquet(folder / "labels.parquet", updated)
                counts = {name: int(np.sum(assigned["partition"] == name)) for name in PARTITIONS}
                view = views[fold["id"]]
                view["assets"][position].update(
                    labels_sha256=sha256(folder / "labels.parquet"),
                    parent_labels_sha256=asset["labels_sha256"],
                    counts=counts,
                    excluded_reasons=dict(Counter(reasons[~used].tolist())),
                )
                for name in PARTITIONS:
                    view["counts"][name] += counts[name]
                if interval_purge:
                    # La frontera purgada es el final del tramo que contiene la decisión.
                    crossed = temporal.partitioner.nominal_partitions(
                        label_prediction[reasons == "label_crosses_boundary"]
                    )
                    purged = {name: int(np.sum(crossed == name)) for name in PARTITIONS}
                    view["assets"][position]["purged_by_boundary"] = purged
                    for name in PARTITIONS:
                        view["purged_by_boundary"][name] += purged[name]
                if recover_annual_boundaries:
                    view["recovered_annual_labels"] += int(np.sum(used & recovered))
            parent._file(asset, "samples")
            parent._file(asset, "labels")
            parent._file(asset, "prices")
        temporal.verify()
        if sha256(protocol_path) != protocol_hash:
            raise ValueError("El protocolo cambió durante la preparación")
        summaries = []
        for fold in folds:
            view = views[fold["id"]]
            atomic_json(stage / fold["id"] / "manifest.json", view)
            summaries.append(
                dict(
                    id=fold["id"],
                    counts=view["counts"],
                    manifest_sha256=sha256(stage / fold["id"] / "manifest.json"),
                    has_all_partitions=all(view["counts"].values()),
                )
            )
            if recover_annual_boundaries:
                summaries[-1]["recovered_annual_labels"] = view["recovered_annual_labels"]
            if interval_purge:
                summaries[-1]["purged_by_boundary"] = view["purged_by_boundary"]
        empty = [
            f"{row['id']}:{name}"
            for row in summaries
            for name, count in row["counts"].items()
            if interval_purge and not count
        ]
        if empty:
            # La versión 2 no cambia la población de una ventana dejando un tramo vacío.
            raise ValueError(
                f"El protocolo v2 no admite tramos sin filas en {protocol['market']}: "
                + ", ".join(empty[:8])
            )
        report = dict(
            schema_version=2 if masked else 1,
            **policy_identity(input_policy),
            folds=summaries,
            parent_sha256=parent.identity,
            **(
                {}
                if masked
                else {key: contract[key] for key in ("macro_sha256", "admission_sha256")}
            ),
            protocol_sha256=protocol_hash,
            status="temporal_views_prepared",
            scientific_training_started=False,
            final_test_opened=False,
            elapsed_seconds=time.perf_counter() - began,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
        if recover_annual_boundaries:
            report["label_admission"] = view["label_admission"]
            report["recovered_annual_candidates"] = annual_candidates
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
    for name in ("parent", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("macro", "admission"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--input-policy", choices=INPUT_POLICIES, default=STRICT_INPUTS)
    parser.add_argument("--recover-annual-boundaries", action="store_true")
    args = parser.parse_args(argv)
    if args.input_policy == STRICT_INPUTS and (args.macro is None or args.admission is None):
        parser.error("La política estricta necesita --macro y --admission")
    report = prepare_temporal_corpus(
        args.parent,
        args.protocol,
        args.macro,
        args.admission,
        args.output,
        recover_annual_boundaries=args.recover_annual_boundaries,
        input_policy=args.input_policy,
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
