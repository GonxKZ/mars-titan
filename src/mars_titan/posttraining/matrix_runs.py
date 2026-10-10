"""Ejecutar los casos de la matriz de adaptadores sobre los padres de una ventana.

La ventana prepara una vez su corpus ordenado de ajuste y validación o, con un
presupuesto de bloque, solo el índice de cohortes de la vista, que después se lee por
bloques sin copiar filas (`environments.view_cohorts`). Cada padre abre
su caché de predicciones, ajusta el normalizador solo con el tramo de ajuste y fija el
plan de la matriz, con las actualizaciones de cada caso, antes del primer ajuste. Cada
caso se ajusta o se recupera con `run_case`, que aplica la selección común con el padre
elegible en la época cero. Las predicciones de calibración y evaluación se escriben
después con el estado seleccionado y las filas completas de la vista de la ventana.

La cola (`posttraining.queue`) y la etapa de la campaña (`posttraining.campaign_stage`)
comparten estas piezas. Ninguna de ellas decide qué padres o ventanas se recorren.
"""

import contextlib
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.environments.view_cohorts import ViewCohortSource, prepare_cohort_index
from mars_titan.episodes.parents import ParentCache
from mars_titan.training.corpus_inputs import CorpusDataset

from . import adapter_matrix
from .heldout import PARTITIONS, _adjustment, evaluate_partition
from .inputs import PairedInputs, fit_normalization
from .parents import load_parent
from .run import run_case


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def index_manifest(folder):
    """Manifiesto del índice de cohortes de una ventana leída por bloques."""
    return Path(folder) / "cohorts" / "manifest.json"


class MatrixWindow:
    """Lectores de ajuste y validación de una vista, ordenados o por bloques.

    Sin `max_block_bytes`, la ventana prepara el corpus ordenado de siempre. Con él,
    prepara solo el índice y lee cada tramo desde la vista con ese presupuesto de memoria.
    Las cohortes, la rejilla y los lotes son los mismos en las dos lecturas.

    Con `since` (microsegundos, solo por bloques) el ajuste usa las sesiones de `train`
    desde ese instante y la rejilla se ajusta con sus objetivos. El índice sigue
    describiendo la población completa, que es la que identifica a un padre de la vista.
    """

    def __init__(
        self,
        view,
        folder,
        *,
        encoding,
        input_policy,
        batch_size,
        stop,
        max_block_bytes=None,
        since=None,
    ):
        self.view, self.folder = Path(view), Path(folder)
        self.encoding, self.input_policy, self.batch_size = encoding, input_policy, batch_size
        _require(
            since is None or max_block_bytes is not None,
            "El ajuste desde un instante solo se lee por bloques",
        )
        self.since = since
        if max_block_bytes is None:
            ordered = self.folder / "ordered"
            prepared = prepare_causal_corpus(
                self.view,
                ordered,
                batch_size=batch_size,
                resume=ordered.exists(),
                stop=stop,
                input_policy=input_policy,
            )
            if prepared["status"] != "completed":
                raise InterruptedError("La preparación del corpus ordenado quedó pendiente")
            self.manifest = ordered / "manifest.json"

            def source(name):
                return ParquetCohortSource(self.manifest, partition=name, input_policy=input_policy)

        else:
            dataset = CorpusDataset(self.view, input_policy=input_policy)
            self.manifest = index_manifest(self.folder)
            prepared = prepare_cohort_index(
                dataset, self.manifest.parent, input_policy=input_policy, stop=stop
            )

            def source(name):
                return ViewCohortSource(
                    self.manifest,
                    dataset,
                    partition=name,
                    max_block_bytes=max_block_bytes,
                    input_policy=input_policy,
                    stop=stop,
                    since=since if name == "train" else None,
                )

        self.source_sha256 = prepared["source_sha256"]
        self.grid = ActionGrid.from_dict(prepared["grid"])
        with contextlib.ExitStack() as sources:
            self.train, self.validation = (
                sources.enter_context(source(name)) for name in ("train", "validation")
            )
            if since is not None:
                # La rejilla solo ve los objetivos de las filas que se ajustan.
                positions = list(range(len(self.train)))
                targets = np.concatenate([c["target"] for c in self.train.cohorts(positions)])
                self.grid = ActionGrid.fit(
                    targets, source_sha256=self.source_sha256, partition="train"
                )
            self.sources = sources.pop_all()

    def close(self):
        self.sources.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def release_ordered(manifest):
    """Retirar las copias Parquet de ajuste y validación de un corpus ordenado.

    Conserva el manifiesto, que es lo único que leen la carga del padre y la
    reconstrucción de un estado seleccionado. Solo borra los dos archivos que el
    manifiesto declara con su huella, dentro de su carpeta.
    """
    manifest = Path(manifest)
    report, _ = read_manifest(manifest)
    _require(
        report.get("kind") == "causal_prediction_corpus" and report.get("status") == "completed",
        "Solo se retira la copia de un corpus ordenado confirmado",
    )
    released = []
    for partition in ("train", "validation"):
        record = report["partitions"][partition]
        _require(
            record.get("path") == f"{partition}-{record['sha256']}.parquet",
            "La copia declarada no corresponde a su huella",
        )
        path = manifest.parent / record["path"]
        if path.is_file() and not path.is_symlink():
            _require(sha256(path) == record["sha256"], "La copia ordenada ha cambiado")
            path.unlink()
        released.append(
            dict(path=record["path"], sha256=record["sha256"], bytes=record["size_bytes"])
        )
    return released


def predict_heldout(
    parent, run_path, report, dataset, folder, *, device, batch_size, stop, partitions=PARTITIONS
):
    """Escribir calibración y evaluación (o también validación) con el estado seleccionado."""
    model, grid, neural = _adjustment(Path(run_path), report, parent, device)
    predictions = {}
    for partition in partitions:
        path = Path(folder) / f"{partition}-predictions.parquet"
        metrics = evaluate_partition(
            dataset,
            parent,
            partition,
            path,
            stop=stop,
            device=device,
            model=model,
            grid=grid,
            neural=neural,
            batch_size=batch_size,
        )
        predictions[partition] = dict(path=path.name, sha256=sha256(path), metrics=metrics)
    return predictions


class MatrixParent:
    """Padre congelado de una ventana, con su caché, normalizador y plan de la matriz."""

    def __init__(
        self,
        window,
        report,
        folder,
        *,
        matrix,
        digest,
        seed,
        device,
        lease,
        stop,
        population=None,
    ):
        self.window, self.device, self.lease = window, device, lease
        self.diagnostic = device == "cpu"
        # Un padre de otra ventana se identifica con la población de la vista que lo ajustó.
        self.parent = load_parent(
            population or window.manifest,
            Path(report),
            device=device,
            diagnostic=self.diagnostic,
            lease=lease,
        )
        kind = self.parent.kind
        _require(kind in adapter_matrix.FAMILIES, "La matriz solo admite padres neuronales")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.cache = ParentCache(
            folder / "predictions.sqlite",
            self.parent.identity["checkpoint_sha256"],
            window.encoding,
            self.parent.predict,
        )
        try:
            self.data = PairedInputs(window.train, window.validation, self.cache)
            path = folder / "normalization.json"
            if path.exists():
                self.normalization = read_manifest(path, 4 * 1024**2)[0]
            else:
                # Solo el tramo de ajuste de la ventana. Validación y evaluación no se abren.
                self.normalization = fit_normalization(
                    self.data, batch_size=window.batch_size, stop=stop
                )
                atomic_json(path, self.normalization)
            self.head = adapter_matrix.model_head(self.parent.model)
            updates = self.data.budget("real", window.batch_size)["updates"]
            rows = adapter_matrix.plan(
                matrix,
                digest,
                kind,
                self.parent.model,
                updates_per_epoch=updates,
                linear_features=self.data.features,
            )
            # El plan cubre todas las semillas. Este padre solo ajusta los casos de la suya.
            self.rows = [row for row in rows[1:] if row["case"]["seed"] == seed]
            _require(self.rows, "La matriz no declara casos para la semilla del padre")
            self.budget = dict(
                matrix_sha256=digest,
                head=self.head,
                updates_per_epoch=updates,
                rows=[
                    {k: v for k, v in row.items() if k != "case"} for row in [rows[0], *self.rows]
                ],
                excluded=adapter_matrix.excluded_controls(matrix, self.head),
            )
            # El plan queda registrado antes del primer ajuste y no cambia al reanudar.
            path = folder / "plan.json"
            if path.exists():
                _require(
                    read_manifest(path, 4 * 1024**2)[0] == self.budget,
                    "El plan de la matriz de este padre ha cambiado",
                )
            else:
                atomic_json(path, self.budget)
        except BaseException:
            self.cache.close()
            raise

    def run(self, row, destination, *, checkpoint_seconds, stop):
        """Ajustar o recuperar un caso y exigir exactamente las actualizaciones del plan."""
        destination = Path(destination)
        report = run_case(
            self.data,
            destination,
            row["case"],
            self.window.grid,
            self.normalization,
            parent=self.parent,
            batch_size=self.window.batch_size,
            device=self.device,
            diagnostic=self.diagnostic,
            lease=self.lease,
            resume=destination.exists(),
            stop=stop,
            checkpoint_seconds=checkpoint_seconds,
        )
        _require(
            report["status"] != "completed" or report["global_step"] == row["updates"],
            "El caso no aplicó las actualizaciones fijadas en el plan de la matriz",
        )
        return report

    def close(self):
        self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
