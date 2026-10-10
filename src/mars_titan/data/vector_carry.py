"""Vectores de la edición v3.1: tres pasadas y herencia solo desde FP32 estricto.

Un gráfico se codifica a partir de su PNG y un texto a partir de su contenido. Con el mismo
codificador, la misma entrada da el mismo vector. Por eso una edición puede heredar los vectores
de otra cuya identidad de codificador sea idéntica, siempre que esa identidad registre FP32
estricto, sin TF32 en matmul ni en cuDNN. La v3 registra TF32 en cuDNN, así que la v3.1 no hereda
ninguno de sus vectores y los recalcula todos.

La regeneración tiene tres pasadas. La de recogida, en CPU, recorre cada activo con los vectores
existentes, confirma los activos completos y anota en `pending-vectors*.sqlite` el texto o el PNG
de cada vector que falta, junto con el activo que lo necesita. `encode_pending` codifica en GPU
solo esas entradas, una a una como en la codificación en línea, y las guarda en
`computed-vectors.sqlite`. La pasada final, otra vez en CPU, completa los activos pendientes con
esos vectores. La GPU no espera a ningún dibujo. Cuando todos los activos anotados están
confirmados, `release_vectors` borra los PNG pendientes y los vectores de gráficos ya escritos en
las muestras, de modo que el disco adicional se limita al tramo de activos en curso.
"""

import fcntl
import hashlib
import json
import sqlite3
import time
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
_BATCH = 512
# Un lote de gráficos de cada 64 se contrasta con la codificación de su último gráfico solo.
_CHECK_EVERY = 64


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


def strict_fp32_spec(spec):
    """La identidad del codificador registra FP32 sin TF32 en matmul ni en cuDNN."""
    precision = spec.get("runtime_precision") if isinstance(spec, dict) else None
    return (
        isinstance(precision, dict)
        and precision.get("dtype") == "float32"
        and precision.get("matmul_tf32") is False
        and precision.get("cudnn_tf32") is False
    )


class CarriedVectors:
    """Caché con respaldos de solo lectura y, si hay edición anterior, sus gráficos por activo.

    `fallbacks` son cachés de solo lectura que se consultan en orden cuando falta un vector.
    """

    def __init__(self, cache, previous_root, encoder_hash, *, fallbacks=()):
        self.cache, self.fallbacks = cache, tuple(fallbacks)
        self.previous = Path(previous_root) if previous_root is not None else None
        self.encoder_hash = encoder_hash
        self.cache_charts = cache.cache_charts
        self.charts = {}
        self.carried = 0

    def select(self, market, symbol, encoders_spec):
        """Cargar los gráficos ya codificados del mismo activo en la edición anterior."""
        self.charts = {}
        if self.previous is None:
            return 0
        folder = self.previous / "samples" / market / symbol
        if not (folder / "manifest.json").exists():
            return 0
        receipt, _ = read_manifest(folder / "manifest.json")
        if receipt.get("encoders") != encoders_spec or not strict_fp32_spec(encoders_spec):
            raise ValueError("La edición anterior usó otro codificador o no registra FP32 estricto")
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


class PendingVectors:
    """Registro de las entradas sin vector, con el texto o el PNG y los activos que las piden."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS pending "
            "(key TEXT PRIMARY KEY, identity TEXT, kind TEXT, payload BLOB, checksum TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS assets "
            "(market TEXT, symbol TEXT, PRIMARY KEY(market, symbol))"
        )

    def add(self, identity, kind, payload, asset):
        if kind not in _WIDTHS or not isinstance(payload, bytes) or not payload:
            raise ValueError("La entrada pendiente no es un texto ni un PNG")
        if not isinstance(asset, tuple) or len(asset) != 2 or not all(asset):
            raise ValueError("La entrada pendiente necesita el activo que la pide")
        key, text = EmbeddingCache.identity(identity)
        self.db.execute("INSERT OR IGNORE INTO assets VALUES (?,?)", asset)
        self.db.execute(
            "INSERT OR IGNORE INTO pending VALUES (?,?,?,?,?)",
            (key, text, kind, payload, hashlib.sha256(payload).hexdigest()),
        )

    def flush(self):
        """Confirmar las entradas anotadas. Se llama una vez por activo y no por entrada.

        Si el proceso se corta antes, el activo sigue sin confirmar y la recogida siguiente lo
        vuelve a anotar entero.
        """
        self.db.commit()

    def close(self):
        self.flush()
        self.db.close()


def _read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def pending_items(edition):
    """Entradas pendientes de todos los registros de una edición, sin repetir identidades.

    Se leen en el orden en que se anotaron. Recorrerlas por clave obligaba a saltar por todo el
    archivo y, con la caché del sistema fría, la lectura era unas ocho veces más lenta.
    """
    seen = set()
    for path in sorted(Path(edition).glob(PENDING_PATTERN)):
        db = _read_only(path)
        try:
            for key, text, kind, payload, checksum in db.execute(
                "SELECT key,identity,kind,payload,checksum FROM pending ORDER BY rowid"
            ):
                if (
                    EmbeddingCache.identity(json.loads(text)) != (key, text)
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
        self.asset = None
        self.added = 0

    def select(self, market, symbol, encoders_spec):
        self.pending.flush()
        self.asset = (market, symbol)
        return self.cache.select(market, symbol, encoders_spec)

    def get(self, identity, **kwargs):
        return self.cache.get(identity, **kwargs)

    def put(self, identity, vector):
        if self.encoders.last is None:
            raise ValueError("No hay ninguna entrada que anotar como pendiente")
        kind, payload = self.encoders.last
        self.encoders.last = None
        self.pending.add(identity, kind, payload, self.asset)
        self.added += 1

    def close(self):
        self.cache.close()
        self.pending.close()


def encode_pending(edition, encoders, *, max_items=None):
    """Codificar en GPU solo las entradas pendientes y guardarlas en `computed-vectors.sqlite`.

    Cada texto se codifica solo, con la misma llamada que la codificación en línea. Los gráficos
    se agrupan en lotes de `encoders.image_batch_size`. En la RTX 4070 de este equipo, un lote de
    8 dio vectores idénticos bit a bit a los de un gráfico por llamada, y lotes de 32 o más no.
    Como esa igualdad depende de los algoritmos que elija cuDNN, cada cierto número de lotes se
    vuelve a codificar solo el último gráfico del lote y cualquier diferencia detiene la pasada. Se puede
    detener con `max_items` y reanudar, porque las entradas ya guardadas se saltan.
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
    # Los vectores se guardan por tandas. Un corte pierde como mucho la tanda en curso, que se
    # vuelve a codificar al reanudar.
    batch = []
    # Tiempo de las llamadas al codificador por modalidad, para medir el caudal de la GPU, y
    # tiempo total del recorrido, que añade la lectura de pendientes y las escrituras.
    by_kind = {kind: dict(encoded=0, seconds=0.0) for kind in _WIDTHS}
    started_all = time.perf_counter()
    group, size, groups = [], getattr(encoders, "image_batch_size", 1), 0

    def keep(identity, kind, vector):
        nonlocal batch
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (_WIDTHS[kind],) or not np.isfinite(vector).all():
            raise ValueError("El codificador produjo dimensiones o valores no válidos")
        batch.append((identity, vector))
        if len(batch) >= _BATCH:
            store.put_many(batch)
            batch = []

    def encode_group():
        nonlocal group, groups
        if not group:
            return
        started = time.perf_counter()
        vectors = encoders.images([payload for _, payload in group])
        # El último lote, más corto, también se contrasta porque su forma no se ha medido.
        if (groups % _CHECK_EVERY == 0 or len(group) < size) and len(group) > 1:
            alone = np.asarray(encoders.images([group[-1][1]])[0], dtype=np.float32)
            if alone.tobytes() != np.asarray(vectors[-1], dtype=np.float32).tobytes():
                raise ValueError("El lote de gráficos no reproduce el vector de un gráfico solo")
        by_kind["image"]["seconds"] += time.perf_counter() - started
        by_kind["image"]["encoded"] += len(group)
        for (identity, _), vector in zip(group, vectors, strict=True):
            keep(identity, "image", vector)
        group, groups = [], groups + 1

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
            counts["encoded"] += 1
            if kind == "image":
                group.append((identity, payload))
                if len(group) == size:
                    encode_group()
                continue
            started = time.perf_counter()
            vector = encoders.text(payload.decode("utf-8"))
            by_kind["text"]["seconds"] += time.perf_counter() - started
            by_kind["text"]["encoded"] += 1
            keep(identity, "text", vector)
        encode_group()
        store.put_many(batch)
    finally:
        store.close()
    return {**counts, "by_kind": by_kind, "loop_seconds": time.perf_counter() - started_all}


def release_vectors(edition):
    """Borrar los PNG pendientes y los vectores de gráficos que ya están en las muestras.

    Solo procede si todos los activos que anotaron entradas están confirmados, con el candado
    exclusivo de la edición. Los textos calculados se conservan porque otros activos comparten
    noticias. Un gráfico repetido en un activo posterior se vuelve a anotar y a codificar.
    """
    edition = Path(edition)
    with (edition / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = sorted(edition.glob(PENDING_PATTERN))
        assets = set()
        for path in files:
            db = _read_only(path)
            try:
                assets.update(db.execute("SELECT market,symbol FROM assets").fetchall())
            finally:
                db.close()
        waiting = sorted(
            f"{market}/{symbol}"
            for market, symbol in assets
            if not (edition / "samples" / market / symbol / "manifest.json").exists()
        )
        if waiting:
            raise ValueError(f"Hay activos con vectores pendientes sin confirmar: {waiting[:5]}")
        released = 0
        if (edition / COMPUTED).exists():
            db = sqlite3.connect(edition / COMPUTED)
            try:
                with db:
                    released = db.execute(
                        "DELETE FROM embeddings WHERE json_extract(identity,'$.kind')='chart'"
                    ).rowcount
            finally:
                db.close()
        for path in files:
            for name in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
                name.unlink(missing_ok=True)
    return dict(assets=len(assets), pending_files=len(files), charts_released=released)
