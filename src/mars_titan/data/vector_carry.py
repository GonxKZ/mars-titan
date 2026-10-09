"""Reutilizar los vectores de una edición anterior cuando la entrada codificada es idéntica.

Un gráfico se codifica a partir de su PNG y un texto a partir de su contenido. Con el mismo
codificador, la misma entrada da el mismo vector. La edición v3 lo comprobó al recodificar
activos con la caché vacía y obtener muestras idénticas bit a bit. Por eso un gráfico de la v3.1
cuyo PNG tiene la huella de uno de la v3 recibe el vector ya calculado, y solo los gráficos
nuevos o cambiados necesitan la GPU. Los textos se buscan primero en la caché nueva y después,
en solo lectura, en las cachés de respaldo.

La regeneración tiene tres pasadas. La de recogida, en CPU, recorre cada activo con los vectores
existentes, confirma los activos completos y anota en `pending-vectors*.sqlite` el texto o el PNG
de cada vector que falta. `encode_pending` codifica en GPU solo esas entradas, una a una como en
la codificación en línea, y las guarda en `computed-vectors.sqlite`. La pasada final, otra vez en
CPU, completa los activos pendientes con esos vectores. La GPU no espera a ningún dibujo.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .cohort_files import read_manifest
from .cohort_samples import _digest
from .embeddings import EmbeddingCache
from .storage import sha256

PENDING_PATTERN = "pending-vectors*.sqlite"
COMPUTED = "computed-vectors.sqlite"
_WIDTHS = {"text": 384, "image": 512}


class MissingVector(ValueError):
    """La entrada no tiene vector previo y necesita el codificador en GPU."""


class ReuseOnlyEncoders:
    """Codificador con la identidad del anterior que nunca calcula un vector nuevo.

    Permite recorrer en CPU los activos cuyos vectores ya existen. El primer vector que falte
    detiene el activo con `MissingVector`, que la codificación registra como pendiente de GPU.
    """

    def __init__(self, spec):
        if not isinstance(spec, dict) or not spec:
            raise ValueError("El codificador reutilizado necesita la identidad de la edición")
        self.spec = spec

    def text(self, text):
        raise MissingVector("Falta el vector de un texto y hace falta el codificador en GPU")

    def images(self, pngs):
        raise MissingVector("Falta el vector de un gráfico y hace falta el codificador en GPU")


class CarriedVectors:
    """Caché que añade los gráficos de la edición anterior, activo por activo.

    `fallbacks` son cachés de solo lectura que se consultan en orden cuando falta un vector.
    """

    def __init__(self, cache, previous_root, encoder_hash, *, fallbacks=()):
        self.cache, self.fallbacks = cache, tuple(fallbacks)
        self.previous = Path(previous_root)
        self.encoder_hash = encoder_hash
        self.cache_charts = cache.cache_charts
        self.charts = {}
        self.carried = 0

    def select(self, market, symbol, encoders_spec):
        """Cargar los gráficos ya codificados del mismo activo en la edición anterior."""
        self.charts = {}
        folder = self.previous / "samples" / market / symbol
        if not (folder / "manifest.json").exists():
            return 0
        receipt, _ = read_manifest(folder / "manifest.json")
        if receipt.get("encoders") != encoders_spec:
            raise ValueError("La edición anterior usó otro codificador")
        if sha256(folder / "samples.parquet") != receipt.get("samples_sha256"):
            raise ValueError("Las muestras de la edición anterior cambiaron")
        with pq.ParquetFile(folder / "samples.parquet") as file:
            for batch in file.iter_batches(
                batch_size=4096, columns=["chart_hash", "charts"], use_threads=False
            ):
                width = batch.schema.field("charts").type.list_size
                vectors = batch.column(1).values.to_numpy().reshape(-1, width)
                for digest, vector in zip(batch.column(0).to_pylist(), vectors, strict=True):
                    self.charts.setdefault(digest, np.array(vector, dtype=np.float32))
        return len(self.charts)

    def get(self, identity, **kwargs):
        if (
            identity.get("kind") == "chart"
            and identity.get("encoder") == self.encoder_hash
            and identity.get("content") in self.charts
        ):
            self.carried += 1
            return self.charts[identity["content"]].copy()
        vector = self.cache.get(identity, **kwargs)
        for fallback in self.fallbacks:
            if vector is not None:
                break
            vector = fallback.get(identity, **kwargs)
        return vector

    def put(self, identity, vector):
        self.cache.put(identity, vector)

    def close(self):
        self.cache.close()
        for fallback in self.fallbacks:
            fallback.close()


def _key(identity):
    text = json.dumps(identity, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest(), text


class PendingVectors:
    """Registro de las entradas sin vector, con el texto o el PNG que hay que codificar."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS pending "
            "(key TEXT PRIMARY KEY, identity TEXT, kind TEXT, payload BLOB, checksum TEXT)"
        )

    def add(self, identity, kind, payload):
        if kind not in _WIDTHS or not isinstance(payload, bytes) or not payload:
            raise ValueError("La entrada pendiente no es un texto ni un PNG")
        key, text = _key(identity)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO pending VALUES (?,?,?,?,?)",
                (key, text, kind, payload, hashlib.sha256(payload).hexdigest()),
            )

    def close(self):
        self.db.close()


def pending_items(edition):
    """Entradas pendientes de todos los registros de una edición, sin repetir identidades."""
    seen = set()
    for path in sorted(Path(edition).glob(PENDING_PATTERN)):
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for key, text, kind, payload, checksum in db.execute(
                "SELECT key,identity,kind,payload,checksum FROM pending ORDER BY key"
            ):
                if (
                    _key(json.loads(text)) != (key, text)
                    or kind not in _WIDTHS
                    or hashlib.sha256(payload).hexdigest() != checksum
                ):
                    raise ValueError(f"El registro de pendientes está corrupto: {path}")
                if key not in seen:
                    seen.add(key)
                    yield json.loads(text), kind, bytes(payload)
        finally:
            db.close()


class CollectingEncoders:
    """Codificador de la pasada de recogida. Anota la entrada y devuelve ceros.

    Los ceros nunca se guardan: `CollectingVectors` desvía su escritura al registro de
    pendientes y la codificación descarta el activo antes de confirmarlo.
    """

    def __init__(self, spec):
        if not isinstance(spec, dict) or not spec:
            raise ValueError("La recogida necesita la identidad del codificador de la edición")
        self.spec = spec
        self.last = None

    def text(self, text):
        self.last = ("text", text.encode("utf-8"))
        return np.zeros(_WIDTHS["text"], dtype=np.float32)

    def images(self, pngs):
        if len(pngs) != 1:
            raise ValueError("La recogida anota los gráficos de uno en uno")
        self.last = ("image", bytes(pngs[0]))
        return np.zeros((1, _WIDTHS["image"]), dtype=np.float32)


class CollectingVectors:
    """Caché de la pasada de recogida: lee como la caché envuelta y anota cada vector ausente."""

    def __init__(self, cache, encoders, pending):
        self.cache, self.encoders, self.pending = cache, encoders, pending
        self.cache_charts = cache.cache_charts
        self.added = 0

    def select(self, *args):
        return self.cache.select(*args)

    def get(self, identity, **kwargs):
        return self.cache.get(identity, **kwargs)

    def put(self, identity, vector):
        if self.encoders.last is None:
            raise ValueError("No hay ninguna entrada que anotar como pendiente")
        kind, payload = self.encoders.last
        self.encoders.last = None
        self.pending.add(identity, kind, payload)
        self.added += 1

    def close(self):
        self.cache.close()
        self.pending.close()


def encode_pending(edition, encoders, *, max_items=None):
    """Codificar en GPU solo las entradas pendientes y guardarlas en `computed-vectors.sqlite`.

    Cada entrada se codifica sola, con la misma llamada que la codificación en línea, así que el
    vector es el que esta habría calculado. Se puede detener con `max_items` y reanudar, porque
    las entradas ya guardadas se saltan.
    """
    edition = Path(edition)
    configured, _ = read_manifest(edition / "configuration.json")
    if configured.get("encoders") != encoders.spec:
        raise ValueError("El codificador no es el de la edición")
    if max_items is not None and (type(max_items) is not int or max_items < 1):
        raise ValueError("El tramo de la GPU necesita un número positivo de entradas")
    encoder = _digest(encoders.spec)
    store = EmbeddingCache(edition / COMPUTED)
    counts = dict(encoded=0, already=0, remaining=0)
    try:
        for identity, kind, payload in pending_items(edition):
            if identity.get("encoder") != encoder:
                raise ValueError("Una entrada pendiente pertenece a otro codificador")
            if store.get(identity) is not None:
                counts["already"] += 1
                continue
            if max_items is not None and counts["encoded"] >= max_items:
                counts["remaining"] += 1
                continue
            if kind == "text":
                vector = encoders.text(payload.decode("utf-8"))
            else:
                vector = encoders.images([payload])[0]
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (_WIDTHS[kind],) or not np.isfinite(vector).all():
                raise ValueError("El codificador produjo dimensiones o valores no válidos")
            store.put(identity, vector)
            counts["encoded"] += 1
    finally:
        store.close()
    return counts
