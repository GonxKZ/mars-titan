"""Proyectar contextos numéricos sin convertir unidades ni crear observaciones."""

import numpy as np
import pyarrow as pa

_MAX_BYTES = 64 * 1024**2
_BLOCK_BYTES = 1024**2
_MAX_CONCEPTS = 512
_MAX_CHUNKS = 1024


def _concepts(names):
    if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= _MAX_CONCEPTS:
        raise ValueError("El vocabulario debe contener entre 1 y 512 conceptos")
    if any(
        not isinstance(name, str) or not 1 <= len(name) <= 256 or not name.strip() for name in names
    ):
        raise ValueError("Cada identificador debe ser un texto de entre 1 y 256 caracteres")
    if len(set(names)) != len(names):
        raise ValueError("El vocabulario contiene identificadores duplicados")
    return tuple(names)


def _validate(block):
    if not np.isfinite(block).all():
        raise ValueError("El contexto numérico contiene NaN o infinitos")
    values, masks, ages = block[:, 0, :], block[:, 1, :], block[:, 2, :]
    if np.any((masks != 0) & (masks != 1)):
        raise ValueError("Las máscaras deben ser cero o uno")
    if np.any(ages < 0):
        raise ValueError("Las edades no pueden ser negativas")
    if np.any((masks == 0) & ((values != 0) | (ages != 0))):
        raise ValueError("Un concepto ausente debe tener valor y edad cero")


def project_numeric_context(array, source_concepts, target_concepts):
    """Reordenar bloques de valores, máscaras y edades por identificador exacto.

    Admite FixedSizeList<float32> y ChunkedArray de ese tipo, con hasta 1024
    fragmentos. Devuelve un FixedSizeListArray nuevo, sin compartir el buffer de
    entrada. Los conceptos añadidos tienen ceros estructurales en los tres canales.

    Entrada y salida suman como máximo 64 MiB. La entrada contabiliza también los
    buffers que conserva una vista Arrow. La validación y copia usan bloques cuya
    entrada y salida suman hasta 1 MiB, con temporales booleanos acotados.
    """
    source, target = _concepts(source_concepts), _concepts(target_concepts)
    positions = {name: i for i, name in enumerate(target)}
    if not set(source) <= positions.keys():
        raise ValueError("Todos los conceptos del origen deben existir exactamente en el destino")
    if (
        not isinstance(array, (pa.Array, pa.ChunkedArray))
        or not pa.types.is_fixed_size_list(array.type)
        or array.type.value_type != pa.float32()
        or array.type.list_size != 3 * len(source)
    ):
        raise ValueError("El contexto debe ser FixedSizeList<float32> con anchura 3 × conceptos")
    if isinstance(array, pa.ChunkedArray) and array.num_chunks > _MAX_CHUNKS:
        raise ValueError("La columna supera el límite de 1024 fragmentos")
    output_bytes = len(array) * 3 * len(target) * np.dtype(np.float32).itemsize
    input_bytes = max(array.nbytes, array.get_total_buffer_size())
    if input_bytes + output_bytes > _MAX_BYTES:
        raise ValueError("La entrada y la salida del contexto superan 64 MiB")

    output = np.zeros((len(array), 3, len(target)), dtype=np.float32)
    mapping = np.asarray([positions[name] for name in source], dtype=np.intp)
    width = 3 * len(source)
    block_rows = max(1, _BLOCK_BYTES // ((width + 3 * len(target)) * np.dtype(np.float32).itemsize))
    chunks = array.iterchunks() if isinstance(array, pa.ChunkedArray) else (array,)
    offset = 0
    for chunk in chunks:
        if chunk.null_count:
            raise ValueError("El contexto contiene filas nulas")
        for start in range(0, len(chunk), block_rows):
            count = min(block_rows, len(chunk) - start)
            values = chunk.values.slice((chunk.offset + start) * width, count * width)
            if values.null_count:
                raise ValueError("El contexto contiene componentes nulos")
            block = values.to_numpy(zero_copy_only=True).reshape(count, 3, len(source))
            _validate(block)
            output[offset + start : offset + start + count, :, mapping] = block
        offset += len(chunk)
    return pa.FixedSizeListArray.from_arrays(pa.array(output.reshape(-1)), 3 * len(target))
