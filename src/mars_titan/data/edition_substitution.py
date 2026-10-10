"""Sustituir la edición anterior activo a activo, solo después de verificar el nuevo.

Cada activo confirmado de la edición nueva se compara con el de la anterior. Solo se admiten los
cambios declarados de la v3.1: sesiones nuevas, ventanas de precio distintas, gráficos nuevos y
vectores recalculados con otro codificador. Después se guarda el registro de la comparación con
las huellas de ambos activos y, solo entonces, se borra el `samples.parquet` anterior. El
manifiesto, la configuración y los factores del activo anterior se conservan como constancia.
Cualquier fallo de verificación detiene el recorrido sin borrar nada.
"""

from datetime import UTC, datetime
from pathlib import Path

from .cohort_files import read_manifest
from .edition_comparison import compare_asset
from .storage import atomic_json, outside_source, sha256


class SubstitutionError(RuntimeError):
    """La verificación de un activo ha fallado y no se ha borrado nada."""


def _confirmed(root, market, symbol):
    folder = Path(root) / "samples" / market / symbol
    receipt, receipt_hash = read_manifest(folder / "manifest.json")
    return receipt, receipt_hash, folder / "samples.parquet"


def substitute_asset(previous_root, current_root, records, market, symbol):
    """Verificar un activo nuevo, guardar su registro y borrar las muestras que sustituye."""
    previous_root, current_root, records = Path(previous_root), Path(current_root), Path(records)
    record_path = records / market / f"{symbol}.json"
    try:
        outside_source(previous_root, current_root)
        outside_source(current_root, previous_root)
        outside_source(previous_root, records)
        if not (previous_root / "samples" / market / symbol / "manifest.json").exists():
            return None  # El activo es nuevo en esta edición y no sustituye nada.
        before, before_hash, old = _confirmed(previous_root, market, symbol)
        if not old.exists():
            if not record_path.exists():
                raise ValueError("Faltan las muestras anteriores y no hay registro")
            return read_manifest(record_path)[0]
        after, after_hash, new = _confirmed(current_root, market, symbol)
        current = sha256(new)
        if current != after.get("samples_sha256"):
            raise ValueError("Las muestras nuevas no coinciden con su recibo")
        replaced = sha256(old)
        if replaced != before.get("samples_sha256"):
            raise ValueError("Las muestras anteriores no coinciden con su recibo")
        comparison = compare_asset(previous_root, current_root, market, symbol)
        if comparison["dropped_sessions"] or comparison["other_differences"]:
            raise ValueError("El activo tiene cambios no declarados respecto a la edición anterior")
    except (OSError, ValueError, KeyError) as error:
        raise SubstitutionError(f"{market}/{symbol}: {error}") from error
    record = dict(
        market=market,
        symbol=symbol,
        verified_at=datetime.now(UTC).isoformat(),
        previous=dict(
            manifest_sha256=before_hash,
            samples_sha256=replaced,
            samples_bytes=old.stat().st_size,
            removed=str(old),
        ),
        current=dict(manifest_sha256=after_hash, samples_sha256=current),
        comparison=comparison,
    )
    # El registro queda sincronizado en disco antes de borrar las muestras que sustituye.
    atomic_json(record_path, record)
    old.unlink()
    return record
