"""Inspeccionar archivos nativos sin cargar código TorchScript ni tensores."""

import io
import pickle
import pickletools
from dataclasses import dataclass

from torch._C import PyTorchFileReader

_ARCHIVE_BYTES = 2 * 1024**2
_METADATA_BYTES = 16 * 1024
_TENSORS = {
    "keys": ("float32", 64),
    "values": ("float32", 64),
    "metadata": ("int64", 6),
    "outcomes": ("float64", 2),
}
_INTEGERS = ("schema_version", "capacity", "seed", "lane", "seen", "last_id", "confirmed_at")
_TEXTS = ("world", "partition", "fold", "representation", "reservoir_rng")
_RECORDS = {
    "data/0",
    "data/1",
    "data/2",
    "data/3",
    "data.pkl",
    "code/__torch__.py",
    "code/__torch__.py.debug_pkl",
    "constants.pkl",
    "version",
    "byteorder",
    ".data/serialization_id",
}
_OPCODES = frozenset(
    (
        "PROTO GLOBAL BINPUT LONG_BINPUT BINGET LONG_BINGET EMPTY_TUPLE EMPTY_DICT MARK "
        "BINUNICODE BININT1 BININT2 BININT LONG1 LONG4 TUPLE TUPLE1 TUPLE2 TUPLE3 NEWOBJ "
        "NEWFALSE NEWTRUE REDUCE SETITEMS BINPERSID BUILD STOP"
    ).split()
)


class _Metadata:
    pass


@dataclass(frozen=True)
class _Storage:
    dtype: str
    key: str
    elements: int


def _empty_hooks(*args):
    if args:
        raise ValueError("El descriptor no admite hooks ni argumentos de construcción")
    return ()


def _tensor(storage, offset, shape, strides, requires_grad, hooks):
    if (
        type(storage) is not _Storage
        or type(offset) is not int
        or offset != 0
        or type(shape) is not tuple
        or len(shape) != 2
        or any(type(value) is not int or not 0 <= value <= 1024 for value in shape)
        or type(strides) is not tuple
        or any(type(value) is not int for value in strides)
        or strides != (shape[1], 1)
        or requires_grad is not False
        or hooks != ()
    ):
        raise ValueError("La forma, offset, stride o gradiente del tensor no son válidos")
    return storage, shape


class _MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("__torch__", "Module"): _Metadata,
            ("torch._utils", "_rebuild_tensor_v2"): _tensor,
            ("collections", "OrderedDict"): _empty_hooks,
            ("torch", "FloatStorage"): "float32",
            ("torch", "LongStorage"): "int64",
            ("torch", "DoubleStorage"): "float64",
        }
        if (module, name) not in allowed:
            raise ValueError("El descriptor contiene un GLOBAL no admitido")
        return allowed[module, name]

    def persistent_load(self, value):
        if (
            type(value) is not tuple
            or len(value) != 5
            or value[0] != "storage"
            or value[1] not in {"float32", "int64", "float64"}
            or value[2] not in {"0", "1", "2", "3"}
            or value[3] != "cpu"
            or type(value[4]) is not int
            or not 0 <= value[4] <= 1024 * 64
        ):
            raise ValueError("El storage del descriptor no está admitido")
        return _Storage(value[1], value[2], value[4])


def _metadata(content):
    if not 0 < len(content) <= _METADATA_BYTES:
        raise ValueError("Los metadatos nativos superan el presupuesto")
    last_position = -1
    for index, (opcode, argument, position) in enumerate(pickletools.genops(content)):
        last_position = position
        if (
            index >= 512
            or opcode.name not in _OPCODES
            or (opcode.name == "PROTO" and argument != 2)
            or (isinstance(argument, int) and argument.bit_length() > 64)
            or (isinstance(argument, str) and len(argument) > 8192)
            or (
                opcode.name in {"BINPUT", "LONG_BINPUT", "BINGET", "LONG_BINGET"}
                and not 0 <= argument < 512
            )
        ):
            raise ValueError("La gramática del descriptor nativo no está admitida")
    if last_position != len(content) - 1:
        raise ValueError("El descriptor tiene datos adicionales")
    result = _MetadataReader(io.BytesIO(content)).load()
    if type(result) is not _Metadata:
        raise ValueError("El archivo no describe un módulo nativo")
    return vars(result)


def _verify_descriptors(reader, sizes, capacity):
    # OutputArchive escribe una clase declarativa sin métodos ejecutables.
    declaration = (
        "class Module(Module):\n  __parameters__ = []\n"
        '  __buffers__ = ["keys", "values", "metadata", "outcomes", ]\n'
    )
    declaration += "".join(f"  {name} : int\n" for name in _INTEGERS)
    declaration += "".join(f"  {name} : str\n" for name in _TEXTS)
    declaration += "".join(f"  {name} : Tensor\n" for name in _TENSORS)
    if (
        reader.get_record("code/__torch__.py") != declaration.encode()
        or reader.get_record("constants.pkl") != b"\x80\x02)."
        or reader.get_record("version") != b"3\n"
        or reader.get_record("byteorder") != b"little"
    ):
        raise ValueError("El archivo contiene código o formato no acreditado")
    metadata = _metadata(reader.get_record("data.pkl"))
    if (
        set(metadata) != set(_INTEGERS) | set(_TEXTS) | set(_TENSORS)
        or any(type(metadata[name]) is not int for name in _INTEGERS)
        or any(type(metadata[name]) is not str for name in _TEXTS)
        or metadata["schema_version"] not in {1, 2}
        or metadata["capacity"] != capacity
        or not 0 <= metadata["seen"] <= 2**32
    ):
        raise ValueError("Los campos del descriptor nativo no conservan su contrato")
    rows = min(metadata["seen"], capacity)
    for index, (name, (dtype, width)) in enumerate(_TENSORS.items()):
        value = metadata[name]
        expected = _Storage(dtype, str(index), rows * width)
        if type(value) is not tuple or value != (expected, (rows, width)):
            raise ValueError("La forma o los alias del tensor no corresponden al cupo")
        if sizes[f"data/{index}"] != rows * width * (4 if dtype == "float32" else 8):
            raise ValueError("El tamaño expandido del storage no coincide con su forma")


def verify_memory_archives(archives, capacities, *, max_expanded_bytes=3 * _ARCHIVE_BYTES):
    """Verificar el conjunto antes de que LibTorch materialice sus cuatro tensores."""
    if (
        type(archives) is not dict
        or type(capacities) is not dict
        or archives.keys() != capacities.keys()
        or not 1 <= len(archives) <= 3
        or type(max_expanded_bytes) is not int
        or max_expanded_bytes < 0
        or any(type(value) is not int or not 1 <= value <= 1024 for value in capacities.values())
        or any(
            type(value) is not bytes or not 1 <= len(value) <= _ARCHIVE_BYTES
            for value in archives.values()
        )
    ):
        raise ValueError("Los archivos nativos exceden el contrato acotado")
    readers, total = [], 0
    try:
        for role, content in archives.items():
            reader = PyTorchFileReader(io.BytesIO(content))
            names = reader.get_all_records()
            if len(names) != len(_RECORDS) or set(names) != _RECORDS:
                raise ValueError("El archivo contiene miembros adicionales o faltantes")
            sizes = {name: reader.get_record_size(name) for name in names}
            expanded = sum(sizes.values())
            total += expanded
            if (
                total > max_expanded_bytes
                or expanded > _ARCHIVE_BYTES
                or any(
                    size > _METADATA_BYTES
                    for name, size in sizes.items()
                    if not name.startswith("data/")
                )
            ):
                raise ValueError("Los archivos expandidos superan el presupuesto conjunto")
            readers.append((reader, sizes, capacities[role]))
        for reader, sizes, capacity in readers:
            _verify_descriptors(reader, sizes, capacity)
    except (
        RuntimeError,
        pickle.UnpicklingError,
        EOFError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as error:
        raise ValueError("El archivo no admite la inspección restringida de metadatos") from error
    return total
