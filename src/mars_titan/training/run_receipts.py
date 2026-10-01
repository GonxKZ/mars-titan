"""Confirmar la identidad antes de producir artefactos recuperables."""

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json


def initialize_receipt(output, identity, *, record, lock, initial_files=()):
    """Usar con el bloqueo del directorio adquirido y sin sobrescribir archivos ajenos."""
    marker, receipt = output / "initialization.json", output / record
    safe_destination(marker)
    safe_destination(receipt)
    if marker.exists():
        if read_manifest(marker, 8 * 1024**2)[0] != identity:
            raise ValueError("La inicialización pertenece a otra identidad")
    else:
        if any(path.name != lock for path in output.iterdir()):
            raise ValueError("La salida sin identidad contiene archivos ajenos o previos")
        atomic_json(marker, identity)
    confirmed = receipt.is_file()
    if not confirmed and any(
        path.name not in {lock, marker.name, *initial_files} for path in output.iterdir()
    ):
        raise ValueError("Falta el recibo de una ejecución que ya tiene artefactos")
    return confirmed
