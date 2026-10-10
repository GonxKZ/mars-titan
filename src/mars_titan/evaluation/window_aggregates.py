"""Agregados por sesión de una ventana de la comparación, guardados y releídos bit a bit.

La comparación walk-forward puntúa cada ventana por separado (``_score_window``) y solo
después une las sesiones de todas para resumir, contrastar y calibrar intervalos. Lo que
sale de una ventana (puntuaciones por sesión de cada brazo y semilla, calibrador, estratos,
variantes de la ablación y presencia) basta para calcular el informe completo sin volver a
leer predicciones por fila. Este módulo guarda esa salida en cuanto la ventana termina y la
devuelve idéntica, con los mismos tipos, formas y bits, para que la retención v2 pueda
liberar las filas de la ventana.

Cada archivo declara su identidad: comparación, ámbito, ventana, vista, huella de cada
archivo de predicciones leído (también los enmascarados) y huella del código que puntúa.
La lectura recalcula esa identidad con las fuentes finales y exige que coincida, así que
unos agregados no se pueden mezclar con otras predicciones, otra configuración u otro
código. Las estructuras se codifican sin pickle: el esqueleto es JSON, los decimales de
Python se guardan con sus bits y las matrices de NumPy van en un ``.npz`` sin objetos.
Los agregados se calculan en float64 y se guardan sin redondear ni comprimir con pérdida:
una matriz o un escalar decimal de otra precisión se rechaza.

La cartera larga y corta (``long_short_comparison``) también trabaja ventana a ventana: cada
una produce los libros por sesión de todos los brazos, las órdenes sin ejecutar y la
identidad de los precios. Se guardan aparte, con el código de la cartera y la identidad de
la edición de precios en su identidad, para que el informe de la cartera tampoco necesite
las filas.
"""

import dataclasses
import importlib
import io
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from mars_titan.data.storage import sha256

from . import walk_forward_comparison as walk

KIND = "walk_forward_window_aggregates"
LONG_SHORT_KIND = "long_short_window_aggregates"
SCHEMA_VERSION = 1
_SKELETON = "__skeleton__"
_MAX_BYTES = 1024**3
# Código que decide lo que vale cada agregado. Un cambio invalida los agregados guardados.
SCORING_SOURCES = (
    "evaluation/walk_forward_comparison.py",
    "evaluation/forecast_panel.py",
    "evaluation/forecast_scores.py",
    "evaluation/modality_strata.py",
    "evaluation/modality_ablation.py",
    "calibration/conformal_quantiles.py",
)
# Código de la cartera larga y corta, que lee los precios de la edición sin ajustar.
LONG_SHORT_SOURCES = (
    "evaluation/walk_forward_comparison.py",
    "evaluation/forecast_panel.py",
    "evaluation/long_short.py",
    "evaluation/long_short_comparison.py",
    "simulation/session_prices.py",
    "simulation/market_rules.py",
)
# Solo se reconstruyen clases de datos del propio proyecto.
_ALLOWED_MODULES = ("mars_titan.evaluation.", "mars_titan.calibration.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _lossless(dtype):
    """Los decimales de los agregados son float64. Ni objetos ni otra precisión."""
    return dtype.kind != "O" and (dtype.kind != "f" or dtype == np.float64)


def encode(value, arrays):
    """Esqueleto JSON de una estructura. Las matrices se añaden a `arrays`."""
    kind = type(value)
    if value is None or kind in (bool, str, int):
        return value
    if kind is float:
        return {"f": struct.pack("<d", value).hex()}
    if isinstance(value, np.generic):
        _require(_lossless(value.dtype), f"No se guardan escalares {value.dtype}")
        return {"s": value.dtype.str, "b": value.tobytes().hex()}
    if kind is np.ndarray:
        _require(_lossless(value.dtype), f"No se guardan matrices {value.dtype}")
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {"a": key}
    if kind is tuple:
        return {"t": [encode(item, arrays) for item in value]}
    if kind is list:
        return [encode(item, arrays) for item in value]
    if kind is dict:
        if all(type(key) is str for key in value):
            return {"d": {key: encode(item, arrays) for key, item in value.items()}}
        return {"m": [[encode(k, arrays), encode(v, arrays)] for k, v in value.items()]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = f"{kind.__module__}:{kind.__qualname__}"
        _require(kind.__module__.startswith(_ALLOWED_MODULES), f"No se guarda {name}")
        fields = {
            field.name: encode(getattr(value, field.name), arrays)
            for field in dataclasses.fields(value)
        }
        return {"c": name, "v": fields}
    raise ValueError(f"No se guarda un valor de tipo {kind.__name__}")


def _dataclass(name):
    module, _, qualname = name.partition(":")
    _require(module.startswith(_ALLOWED_MODULES), f"No se reconstruye {name}")
    cls = importlib.import_module(module)
    for part in qualname.split("."):
        cls = getattr(cls, part)
    _require(dataclasses.is_dataclass(cls) and isinstance(cls, type), f"{name} no es de datos")
    return cls


def decode(skeleton, arrays):
    """Estructura original desde su esqueleto y sus matrices."""
    if skeleton is None or type(skeleton) in (bool, str, int):
        return skeleton
    if type(skeleton) is list:
        return [decode(item, arrays) for item in skeleton]
    _require(type(skeleton) is dict and len(skeleton) in (1, 2), "Esqueleto no válido")
    if "f" in skeleton:
        return struct.unpack("<d", bytes.fromhex(skeleton["f"]))[0]
    if "s" in skeleton:
        return np.frombuffer(bytes.fromhex(skeleton["b"]), dtype=np.dtype(skeleton["s"]))[0]
    if "a" in skeleton:
        return arrays[skeleton["a"]]
    if "t" in skeleton:
        return tuple(decode(item, arrays) for item in skeleton["t"])
    if "d" in skeleton:
        return {key: decode(item, arrays) for key, item in skeleton["d"].items()}
    if "m" in skeleton:
        return {decode(k, arrays): decode(v, arrays) for k, v in skeleton["m"]}
    cls = _dataclass(skeleton["c"])
    names = [field.name for field in dataclasses.fields(cls)]
    _require(list(skeleton["v"]) == names, f"{skeleton['c']} cambió sus campos")
    # Se restauran los campos tal cual, sin repetir las validaciones de su construcción.
    value = object.__new__(cls)
    for name in names:
        object.__setattr__(value, name, decode(skeleton["v"][name], arrays))
    return value


def same(left, right):
    """Igualdad de tipos, formas y bits en toda la estructura."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray | np.generic):
        return (
            left.dtype == right.dtype
            and np.shape(left) == np.shape(right)
            and np.asarray(left).tobytes() == np.asarray(right).tobytes()
        )
    if type(left) is float:
        return struct.pack("<d", left) == struct.pack("<d", right)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(map(same, left, right))
    if type(left) is dict:
        return list(left) == list(right) and all(same(left[k], right[k]) for k in left)
    if dataclasses.is_dataclass(left):
        return all(
            same(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    return left == right


def identity(config, sources, window_id, ablation=None, *, code=SCORING_SOURCES):
    """Lo que determina los agregados de una ventana, sin leer predicciones."""
    files = {
        f"{name}/{seed}/{part}": record["sha256"]
        for (name, seed, window), parts in sorted(sources["files"].items(), key=str)
        if window == window_id
        for part, record in sorted(parts.items())
    }
    masked = None
    if ablation is not None:
        masked = {
            "/".join(map(str, key[:3])): record["sha256"]
            for key, record in sorted(ablation["files"].items(), key=str)
            if key[3] == window_id
        }
    root = Path(walk.__file__).parents[1]
    value = dict(
        comparison_sha256=config["sha256"],
        scope=sources["scope"],
        window=window_id,
        bounds=sources["windows"][window_id],
        view_sha256=sources["views"][window_id],
        predictions=files,
        ablation=masked,
        code={name: sha256(root / name) for name in code},
    )
    # La misma forma que tendrá al releerla del JSON, con listas en lugar de tuplas.
    return json.loads(json.dumps(value))


def path_for(folder, scope, window_id):
    return Path(folder) / scope / f"{window_id}.npz"


def long_short_path(folder, scope, window_id):
    return Path(folder) / scope / f"{window_id}.long_short.npz"


def _save(path, kind, expected, value):
    """Escribir el esqueleto y las matrices de `value` de forma atómica."""
    arrays = {}
    document = dict(
        schema_version=SCHEMA_VERSION, kind=kind, identity=expected, scores=encode(value, arrays)
    )
    # Sin ordenar claves: el orden de cada diccionario forma parte de lo que se restaura.
    text = json.dumps(document, ensure_ascii=True).encode()
    arrays[_SKELETON] = np.frombuffer(text, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(buffer.getvalue())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return dict(path=path, sha256=sha256(path), bytes=path.stat().st_size)


def _load(path, kind, expected, label):
    """Releer unos agregados y exigir que su identidad sea la de las fuentes actuales."""
    _require(path.is_file() and not path.is_symlink(), f"Faltan los agregados de {label}")
    _require(path.stat().st_size <= _MAX_BYTES, f"{path.name} supera su presupuesto")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    document = json.loads(arrays.pop(_SKELETON).tobytes())
    _require(
        document.get("kind") == kind and document.get("schema_version") == SCHEMA_VERSION,
        f"{path.name} no son agregados de una ventana",
    )
    stored_identity = document["identity"]
    changed = sorted(key for key in expected if stored_identity.get(key) != expected[key])
    _require(
        not changed,
        f"Los agregados de {label} proceden de otras fuentes o código: {', '.join(changed)}",
    )
    return decode(document["scores"], arrays)


def write(folder, config, sources, window_id, ablation=None):
    """Puntuar una ventana y guardar sus agregados. Devuelve su ruta y su huella."""
    scored = walk._score_window(sources, config, window_id, ablation)
    path = path_for(folder, sources["scope"], window_id)
    record = _save(path, KIND, identity(config, sources, window_id, ablation), scored)
    restored = read(folder, config, sources, window_id, ablation)
    _require(same(restored, scored), f"Los agregados de {window_id} no se releen igual")
    return record


def read(folder, config, sources, window_id, ablation=None):
    """Agregados de una ventana, comprobando que proceden de estas mismas fuentes y código."""
    path = path_for(folder, sources["scope"], window_id)
    expected = identity(config, sources, window_id, ablation)
    return _load(path, KIND, expected, f"{sources['scope']} {window_id}")


def _long_short_identity(config, sources, window_id, edition):
    """Identidad de la ventana con el código de la cartera y la edición de precios."""
    from mars_titan.simulation.session_prices import SessionPrices

    start, end = sources["windows"][window_id]["evaluation"]
    # Como en los libros, solo los mercados elegibles de la ventana tienen precios.
    prices = {
        market: SessionPrices(edition, market, start, end).identity()
        for market in sources["markets"]
        if window_id in sources["eligible"][market]
    }
    value = dict(identity(config, sources, window_id, code=LONG_SHORT_SOURCES), prices=prices)
    return json.loads(json.dumps(value))


def write_long_short(folder, config, sources, window_id, edition):
    """Calcular los libros de la cartera de una ventana y guardarlos sin pérdida."""
    from . import long_short_comparison as portfolio

    declared = config[portfolio.SECTION]
    books = portfolio._window(sources, config, window_id, edition, declared)
    path = long_short_path(folder, sources["scope"], window_id)
    expected = _long_short_identity(config, sources, window_id, edition)
    record = _save(path, LONG_SHORT_KIND, expected, books)
    restored = read_long_short(folder, config, sources, window_id, edition)
    _require(same(restored, books), f"La cartera de {window_id} no se relee igual")
    return record


def read_long_short(folder, config, sources, window_id, edition):
    """Libros de la cartera de una ventana, con la misma identidad de fuentes y precios."""
    path = long_short_path(folder, sources["scope"], window_id)
    expected = _long_short_identity(config, sources, window_id, edition)
    return _load(path, LONG_SHORT_KIND, expected, f"la cartera de {sources['scope']} {window_id}")
