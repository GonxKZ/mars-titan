"""Contrastar claves, etiquetas y cinco entradas con una edición supervisada de referencia."""

import argparse
import time
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.environments.corpus_source import ParquetCohortSource
from mars_titan.training.corpus_inputs import CorpusDataset


def check(manifest, ordered, maximum_bytes=64 * 1024**2):
    dataset = CorpusDataset(manifest)
    started, partitions = time.perf_counter(), {}
    for partition in ("train", "validation"):
        expected, used = {}, 0
        for batch in dataset.batches(partition=partition, batch_size=256, epoch=0, seed=0):
            for i, key in enumerate(batch["sample_ids"]):
                inputs = {name: value[i].copy() for name, value in batch["inputs"].items()}
                used += sum(value.nbytes for value in inputs.values()) + 512 + len(key)
                if used > maximum_bytes:
                    raise ValueError(
                        "La comprobación directa supera su presupuesto, no se recortan filas"
                    )
                if key in expected:
                    raise ValueError("Hay una clave duplicada en la referencia")
                expected[key] = (
                    inputs,
                    batch["target"][i],
                    int(batch["target_available_at"][i].astype(np.int64)),
                )
        visited, maximum_difference = set(), 0.0
        with ParquetCohortSource(ordered, partition=partition) as source:
            if source.source_sha256 != dataset.identity:
                raise ValueError("El origen ordenado no corresponde a la edición supervisada")
            for index in range(len(source)):
                block = source(index)
                for i, asset in enumerate(block["asset_ids"]):
                    key = f"{asset}/{block['prediction_at']}"
                    if key in visited or key not in expected:
                        raise ValueError("Hay una clave ajena o duplicada en el origen ordenado")
                    inputs, target, maturity = expected[key]
                    if target != block["target"][i] or maturity != block["target_available_at"][i]:
                        raise ValueError("La etiqueta o su maduración ha cambiado")
                    for name, values in inputs.items():
                        if not np.array_equal(values, block["inputs"][name][i]):
                            raise ValueError("Una modalidad ha cambiado al ordenar los archivos")
                        maximum_difference = max(
                            maximum_difference,
                            float(np.max(np.abs(values - block["inputs"][name][i]))),
                        )
                    visited.add(key)
            if visited != set(expected):
                raise ValueError("El recorrido ha perdido muestras")
            partitions[partition] = dict(
                samples=len(visited),
                cohorts=len(source),
                max_assets_per_cohort=source.max_assets,
                comparison_buffer_bytes_estimate=used,
                maximum_feature_difference=maximum_difference,
                targets_equal=True,
                maturities_equal=True,
            )
    return dict(
        kind="causal_corpus_parity_audit",
        scope=dataset.manifest["scope"],
        cohort_complete=dataset.manifest["cohort_complete"],
        source_manifest_sha256=dataset.identity,
        ordered_manifest_sha256=sha256(ordered),
        assets=len(dataset.assets),
        partitions=partitions,
        total_seconds=time.perf_counter() - started,
        final_test_opened=False,
        limits=[
            "Contraste de la edición indicada, no una campaña de entrenamiento.",
            "El comparador directo tiene un presupuesto de 64 MiB y falla si se supera.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "ordered", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Conserva el informe anterior y elige otra salida")
    report = check(args.manifest, args.ordered)
    atomic_json(args.output, report)
    print(report["partitions"])


if __name__ == "__main__":
    main()
