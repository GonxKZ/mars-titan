"""Ejecutar los casos de la matriz de adaptadores sobre los padres de una ventana.

La ventana prepara una vez su corpus ordenado de ajuste y validación. Cada padre abre
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

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.corpus_source import ParquetCohortSource, prepare_causal_corpus
from mars_titan.episodes.parents import ParentCache

from . import adapter_matrix
from .heldout import PARTITIONS, _adjustment, evaluate_partition
from .inputs import PairedInputs, fit_normalization
from .parents import load_parent
from .run import run_case


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class MatrixWindow:
    """Corpus ordenado de una vista con sus lectores de ajuste y validación."""

    def __init__(self, view, folder, *, encoding, input_policy, batch_size, stop):
        self.view, self.folder = Path(view), Path(folder)
        self.encoding, self.input_policy, self.batch_size = encoding, input_policy, batch_size
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
        self.source_sha256 = prepared["source_sha256"]
        self.grid = ActionGrid.from_dict(prepared["grid"])
        with contextlib.ExitStack() as sources:
            self.train, self.validation = (
                sources.enter_context(
                    ParquetCohortSource(self.manifest, partition=name, input_policy=input_policy)
                )
                for name in ("train", "validation")
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


def predict_heldout(parent, run_path, report, dataset, folder, *, device, batch_size, stop):
    """Escribir calibración y evaluación con el estado seleccionado de un ajuste."""
    model, grid, neural = _adjustment(Path(run_path), report, parent, device)
    predictions = {}
    for partition in PARTITIONS:
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

    def __init__(self, window, report, folder, *, matrix, digest, seed, device, lease, stop):
        self.window, self.device, self.lease = window, device, lease
        self.diagnostic = device == "cpu"
        self.parent = load_parent(
            window.manifest, Path(report), device=device, diagnostic=self.diagnostic, lease=lease
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
                head=self.head,
                updates_per_epoch=updates,
                rows=[
                    {k: v for k, v in row.items() if k != "case"} for row in [rows[0], *self.rows]
                ],
                excluded=adapter_matrix.excluded_controls(matrix, self.head),
            )
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
