"""Límites de ajuste derivados de una supervisión temporal verificada."""

from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.environments.cohorts import FINAL_TEST_START_US, VALIDATION_START_US
from mars_titan.evaluation.splits import PARTITIONS

LEGACY_BOUNDS = {
    "train": (0, VALIDATION_START_US, VALIDATION_START_US),
    "validation": (VALIDATION_START_US, FINAL_TEST_START_US, FINAL_TEST_START_US),
}


def supervision_bounds(source):
    """Validar las particiones sin utilizar sus etiquetas para definir los cortes."""
    temporal = source.get("temporal_view")
    counts = source.get("counts", {})
    expected = set(PARTITIONS) if temporal is not None else set(LEGACY_BOUNDS)
    if (
        source.get("final_test_opened", False) is not False
        or not isinstance(counts, dict)
        or set(counts) != expected
        or any(type(value) is not int or value < 1 for value in counts.values())
    ):
        raise ValueError("Las particiones de supervisión no conservan su población o reserva")
    if temporal is None:
        return dict(LEGACY_BOUNDS)
    from .temporal_corpus import TemporalInputs

    verified = TemporalInputs(temporal)
    if verified.partitioner.test_start != FINAL_TEST_START_US:
        raise ValueError("El postentrenamiento no puede cambiar el test final reservado")
    return {
        name: (int(start), int(end), int(cutoff))
        for name, start, end, cutoff in verified.partitioner.bounds
    }


def ordered_bounds(metadata):
    """Conservar el vínculo con la supervisión que produjo los Parquet ordenados."""
    record = metadata.get("source_manifest")
    if record is None:
        return supervision_bounds(metadata)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Falta el vínculo con la supervisión de origen")
    path = Path(record["path"])
    safe_destination(path)
    source, digest = read_manifest(path, 8 * 1024**2)
    if (
        digest != record["sha256"]
        or digest != metadata["source_sha256"]
        or source.get("counts") != metadata["counts"]
        or "temporal_view" not in source
    ):
        raise ValueError("La supervisión de origen no conserva su huella y población")
    return supervision_bounds(source)
