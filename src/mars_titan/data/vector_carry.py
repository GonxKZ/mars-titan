"""Vectores de la edición v3.1: tres pasadas, herencia en FP32 estricto y textos contrastados.

Un gráfico se codifica a partir de su PNG y un texto a partir de su contenido. Con el mismo
codificador, la misma entrada da el mismo vector. Por eso una edición puede heredar los vectores
de otra cuya identidad de codificador sea idéntica, siempre que esa identidad registre FP32
estricto, sin TF32 en matmul ni en cuDNN. La v3 registra TF32 en cuDNN, así que la v3.1 recalcula
todos sus gráficos.

Los textos de la v3 son otro caso. PyTorch deja TF32 desactivado en matmul por defecto y el modelo
de texto no usa cuDNN, de modo que la marca de cuDNN no llegó a afectarlos. En la porción medida,
los 5051 textos comunes coinciden bit a bit. Aun así, la herencia de textos (`text_carry`) no se
apoya solo en ese argumento. Cada activo vuelve a codificar en GPU una muestra determinista de los
textos que heredaría y exige igualdad bit a bit. Si alguno difiere, se recodifican todos sus
textos y queda constancia en `text-carry/<mercado>/<símbolo>.json`.

La regeneración tiene tres pasadas. La de recogida, en CPU, recorre cada activo con los vectores
existentes, confirma los activos completos y anota en `pending-vectors*.sqlite` el texto o el PNG
de cada vector que falta, junto con el activo que lo necesita. `encode_pending` codifica en GPU
solo esas entradas, contrasta los textos heredables y lo guarda todo en `computed-vectors.sqlite`.
La pasada final, otra vez en CPU, completa los activos pendientes con esos vectores. La GPU no
espera a ningún dibujo. Cuando todos los activos anotados están confirmados, `release_vectors`
borra los PNG pendientes y los vectores de gráficos ya escritos en las muestras, de modo que el
disco adicional se limita al tramo de activos en curso.
"""

import fcntl
import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .cohort_files import read_manifest
from .cohort_samples import _digest
from .embeddings import EmbeddingCache
from .storage import atomic_json, sha256

PENDING_PATTERN = "pending-vectors*.sqlite"
COMPUTED = "computed-vectors.sqlite"
_WIDTHS = {"text": 384, "image": 512}
_BATCH = 512
# Un lote de gráficos de cada 64 se contrasta con la codificación de su último gráfico solo.
_CHECK_EVERY = 64
TEXT_CARRY = "text-carry"
REUSED, REENCODED = "reused_after_check", "reencoded_after_mismatch"
# Muestra de textos heredables que se vuelve a codificar en cada activo: el primero, el último,
# uno de cada 64 y, si el activo tiene pocos, al menos 8 repartidos entre todos ellos.
_TEXT_CHECK_EVERY, _TEXT_CHECK_MIN = 64, 8
# Campos de la identidad que no intervienen en el vector de un texto. La precisión se comprueba
# aparte, el código del módulo cambió por las comprobaciones de FP32 y el lote de gráficos solo
# afecta a las imágenes. El lote de fragmentos de texto sí debe coincidir.
_TEXT_NEUTRAL = {"code_sha256", "runtime_precision", "batch_sizes"}


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


def check_text_source(previous, current):
    """Comprobar que una edición anterior puede ceder sus textos a la actual.

    Se exige el mismo modelo, tokenizador, versiones, ubicación de la tabla de palabras y lote de
    fragmentos, y que la anterior no usara TF32 en matmul. TF32 en cuDNN se tolera porque el
    modelo de texto no pasa por cuDNN. La comprobación es necesaria pero no suficiente, y por eso
    cada activo contrasta además una muestra de sus textos en la GPU.
    """
    if not strict_fp32_spec(current):
        raise ValueError("La edición actual no registra FP32 estricto")
    precision = previous.get("runtime_precision") if isinstance(previous, dict) else None
    if (
        not isinstance(precision, dict)
        or precision.get("dtype") != "float32"
        or precision.get("matmul_tf32") is not False
    ):
        raise ValueError("La edición anterior no calculó sus textos en FP32 sin TF32 en matmul")
    fields = (set(previous) | set(current)) - _TEXT_NEUTRAL
    chunks = [(spec.get("batch_sizes") or {}).get("text_chunks") for spec in (previous, current)]
    if any(previous.get(k) != current.get(k) for k in fields) or chunks[0] != chunks[1]:
        raise ValueError("La edición anterior usó otro modelo o configuración de texto")


def text_carry_identity(source, current):
    """Identidad de la herencia de textos, tal como la registra la configuración de la edición."""
    source = Path(source).resolve()
    configured, configured_hash = read_manifest(source / "configuration.json")
    check_text_source(configured.get("encoders"), current)
    return dict(
        edition=str(source),
        configuration_sha256=configured_hash,
        encoder_sha256=_digest(configured["encoders"]),
        rule="same_text_model_without_matmul_tf32_and_bit_identical_sample_per_asset",
        check=dict(every=_TEXT_CHECK_EVERY, minimum=_TEXT_CHECK_MIN, first_and_last=True),
    )


def text_check_positions(count):
    """Posiciones contrastadas entre los `count` textos heredables de un activo."""
    if count <= _TEXT_CHECK_MIN:
        return set(range(count))
    # Con 8 textos o más, los 8 puntos repartidos son distintos e incluyen el primero y el último.
    spread = {j * (count - 1) // (_TEXT_CHECK_MIN - 1) for j in range(_TEXT_CHECK_MIN)}
    return spread | set(range(0, count, _TEXT_CHECK_EVERY))


def text_carry_record(edition, market, symbol):
    """Constancia del contraste de textos heredados de un activo, o None si no lo tiene."""
    path = Path(edition) / TEXT_CARRY / market / f"{symbol}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    owner = (record.get("market"), record.get("symbol"))
    if record.get("decision") not in {REUSED, REENCODED} or owner != (market, symbol):
        raise ValueError(f"La constancia de textos heredados no es válida: {path}")
    return record


class CarriedTexts:
    """Textos de una edición anterior, que solo se ceden a un activo con el contraste superado.

    La identidad de un texto incluye la huella del codificador, distinta en cada edición. Para
    buscarlo en la anterior se sustituye esa huella por la suya, sin tocar el contenido ni la
    política de disponibilidad.
    """

    def __init__(self, edition, carry, encoder_hash):
        self.edition, self.encoder = Path(edition), encoder_hash
        self.source = carry["encoder_sha256"]
        self.store = EmbeddingCache(Path(carry["edition"]) / "embeddings.sqlite", read_only=True)
        self.approved = False

    def select(self, market, symbol):
        record = text_carry_record(self.edition, market, symbol)
        self.approved = record is not None and record["decision"] == REUSED

    def previous(self, identity):
        if identity.get("kind") != "news" or identity.get("encoder") != self.encoder:
            return None
        return self.store.get({**identity, "encoder": self.source})

    def get(self, identity, **kwargs):
        return self.previous(identity) if self.approved else None

    def close(self):
        self.store.close()


class CarriedVectors:
    """Caché con respaldos de solo lectura y, si hay edición anterior, sus gráficos por activo.

    `fallbacks` son cachés de solo lectura que se consultan en orden cuando falta un vector.
    `texts` son los textos heredables de otra edición y se consultan los últimos, de modo que un
    vector ya calculado en FP32 estricto siempre tiene preferencia.
    """

    def __init__(self, cache, previous_root, encoder_hash, *, fallbacks=(), texts=None):
        self.cache, self.fallbacks, self.texts = cache, tuple(fallbacks), texts
        self.previous = Path(previous_root) if previous_root is not None else None
        self.encoder_hash = encoder_hash
        self.cache_charts = cache.cache_charts
        self.charts = {}
        self.carried = 0

    def select(self, market, symbol, encoders_spec):
        """Cargar los gráficos ya codificados del mismo activo en la edición anterior."""
        self.charts = {}
        if self.texts is not None:
            self.texts.select(market, symbol)
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
        if vector is None and self.texts is not None:
            vector = self.texts.get(identity)
        return vector

    def put(self, identity, vector):
        self.cache.put(identity, vector)

    def close(self):
        self.cache.close()
        for fallback in self.fallbacks:
            fallback.close()
        if self.texts is not None:
            self.texts.close()


def _check_asset(asset):
    if not isinstance(asset, tuple) or len(asset) != 2 or not all(asset):
        raise ValueError("La entrada pendiente necesita el activo que la pide")


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
        # Textos heredables de cada activo. Se guarda el contenido de todos porque, si el
        # contraste falla, el activo los recodifica todos en la misma pasada de GPU.
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS carried (market TEXT, symbol TEXT, key TEXT, "
            "identity TEXT, payload BLOB, checksum TEXT, sampled INTEGER, "
            "PRIMARY KEY(market, symbol, key))"
        )

    def add_carried(self, asset, entries):
        """Anotar los textos heredables de un activo y pedir a la GPU los de la muestra.

        `entries` son pares (identidad, texto en bytes) en el orden en que el activo los pidió,
        que es determinista. Si el activo se vuelve a recoger, una entrada ya anotada conserva su
        marca de muestra, de modo que el contraste nunca se reduce.
        """
        _check_asset(asset)
        if not entries:
            return
        checked = text_check_positions(len(entries))
        for position, (identity, payload) in enumerate(entries):
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("El texto heredable no tiene contenido")
            key, text = EmbeddingCache.identity(identity)
            sampled = position in checked
            self.db.execute(
                "INSERT INTO carried VALUES (?,?,?,?,?,?,?) ON CONFLICT(market,symbol,key) "
                "DO UPDATE SET sampled=max(sampled,excluded.sampled)",
                (*asset, key, text, payload, hashlib.sha256(payload).hexdigest(), int(sampled)),
            )
            if sampled:
                # El primer texto siempre está en la muestra, y así el activo queda anotado.
                self.add(identity, "text", payload, asset)

    def add(self, identity, kind, payload, asset):
        if kind not in _WIDTHS or not isinstance(payload, bytes) or not payload:
            raise ValueError("La entrada pendiente no es un texto ni un PNG")
        _check_asset(asset)
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


def carried_texts(edition):
    """Textos heredables de cada activo, reunidos de todos los registros de pendientes.

    Devuelve, activo a activo, las entradas (identidad, texto, en la muestra) en el orden en que
    se anotaron en cada registro. Un activo recogido en varios registros une sus entradas, y un
    texto está en la muestra si lo está en cualquiera de ellos.
    """
    dbs = [_read_only(path) for path in sorted(Path(edition).glob(PENDING_PATTERN))]
    try:
        assets = sorted(
            {tuple(row) for db in dbs for row in db.execute("SELECT market,symbol FROM carried")}
        )
        for asset in assets:
            entries = {}
            for db in dbs:
                for key, text, payload, checksum, sampled in db.execute(
                    "SELECT key,identity,payload,checksum,sampled FROM carried "
                    "WHERE market=? AND symbol=? ORDER BY rowid",
                    asset,
                ):
                    identity = json.loads(text)
                    if (
                        EmbeddingCache.identity(identity) != (key, text)
                        or identity.get("kind") != "news"
                        or hashlib.sha256(payload).hexdigest() != checksum
                    ):
                        raise ValueError("El registro de textos heredables está corrupto")
                    entry = entries.setdefault(key, [identity, bytes(payload), False])
                    entry[2] = entry[2] or bool(sampled)
            yield asset, [tuple(entry) for entry in entries.values()]
    finally:
        for db in dbs:
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
    """Caché de la pasada de recogida: lee como la caché envuelta y anota cada vector ausente.

    Con `texts`, un texto ausente que la edición anterior ya tiene no pasa directamente a la GPU.
    Se reúne con los demás textos heredables del activo y, al terminar el activo, solo los de la
    muestra de contraste quedan pendientes de codificar.
    """

    def __init__(self, cache, encoders, pending, *, texts=None):
        self.cache, self.encoders, self.pending, self.texts = cache, encoders, pending, texts
        self.cache_charts = cache.cache_charts
        self.asset = None
        self.added = 0
        self.candidates = {}

    def _finish(self):
        # Se anota todo el activo de una vez, en la misma transacción que sus pendientes.
        if self.candidates:
            self.pending.add_carried(self.asset, list(self.candidates.values()))
            self.candidates = {}
        self.pending.flush()

    def select(self, market, symbol, encoders_spec):
        self._finish()
        self.asset = (market, symbol)
        return self.cache.select(market, symbol, encoders_spec)

    def get(self, identity, **kwargs):
        return self.cache.get(identity, **kwargs)

    def put(self, identity, vector):
        if self.encoders.last is None:
            raise ValueError("No hay ninguna entrada que anotar como pendiente")
        kind, payload = self.encoders.last
        self.encoders.last = None
        key = EmbeddingCache.identity(identity)[0]
        # Un texto se repite en varias ventanas y solo se busca en la edición anterior una vez.
        if (
            kind == "text"
            and self.texts is not None
            and (key in self.candidates or self.texts.previous(identity) is not None)
        ):
            self.candidates.setdefault(key, (identity, payload))
        else:
            self.pending.add(identity, kind, payload, self.asset)
        # El activo recibió ceros en lugar del vector y no puede confirmarse en esta pasada.
        self.added += 1

    def close(self):
        try:
            self._finish()
        finally:
            self.cache.close()
            self.pending.close()


def encode_pending(edition, encoders, *, max_items=None):
    """Codificar en GPU solo las entradas pendientes y guardarlas en `computed-vectors.sqlite`.

    Cada texto se codifica solo, con la misma llamada que la codificación en línea. Los gráficos
    se agrupan en lotes de `encoders.image_batch_size`. En la RTX 4070 de este equipo, un lote de
    8 dio vectores idénticos bit a bit a los de un gráfico por llamada, y lotes de 32 o más no.
    Como esa igualdad depende de los algoritmos que elija cuDNN, cada cierto número de lotes se
    vuelve a codificar solo el último gráfico del lote y cualquier diferencia detiene la pasada.
    Se puede detener con `max_items` y reanudar, porque las entradas ya guardadas se saltan.

    Si la edición hereda textos, al terminar los pendientes se contrasta la muestra de cada activo
    con `_check_carried_texts`. Con `max_items` y entradas sin codificar, ese contraste espera a la
    pasada que las complete.
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
        batch.append((identity, _checked(vector, kind)))
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
        carry = configured.get("text_carry")
        # La muestra de textos heredados solo se contrasta con todos los pendientes ya guardados.
        if carry is not None and not counts["remaining"]:
            counts["text_carry"] = _check_carried_texts(edition, carry, store, encoders, by_kind)
    finally:
        store.close()
    return {**counts, "by_kind": by_kind, "loop_seconds": time.perf_counter() - started_all}


def _checked(vector, kind):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (_WIDTHS[kind],) or not np.isfinite(vector).all():
        raise ValueError("El codificador produjo dimensiones o valores no válidos")
    return vector


def _check_carried_texts(edition, carry, store, encoders, by_kind):
    """Contrastar la muestra de textos heredables de cada activo y dejar constancia.

    Los textos de la muestra ya están codificados en `store` en FP32 estricto. Si alguno difiere
    del vector de la edición anterior, aquí mismo se codifican todos los textos heredables del
    activo, de modo que ninguno de sus vectores procede de la anterior. La constancia se escribe
    después de guardar esos vectores y un activo que ya la tiene se salta al reanudar.

    Un texto fuera de la muestra que difiriera pasaría inadvertido. Ese es el límite de contrastar
    por muestreo, y la comparación de cada activo con la v3 antes de sustituirlo no lo cubre,
    porque el vector heredado es justamente el de la v3.
    """
    source = Path(carry["edition"])
    if read_manifest(source / "configuration.json")[1] != carry["configuration_sha256"]:
        raise ValueError("Ha cambiado la edición de la que se heredan los textos")
    previous = EmbeddingCache(source / "embeddings.sqlite", read_only=True)
    totals = dict(assets=0, reused=0, reencoded=0, compared=0, mismatched=0, reencoded_texts=0)
    try:
        for (market, symbol), entries in carried_texts(edition):
            if text_carry_record(edition, market, symbol) is not None:
                continue
            sampled = [identity for identity, _, flag in entries if flag]
            mismatched = []
            for identity in sampled:
                new = store.get(identity)
                old = previous.get({**identity, "encoder": carry["encoder_sha256"]})
                if new is None or old is None:
                    raise ValueError("Falta un vector de la muestra de textos heredados")
                if new.tobytes() != old.tobytes():
                    mismatched.append(identity["content"])
            fresh = []
            if mismatched:
                started = time.perf_counter()
                for identity, payload, _ in entries:
                    if store.get(identity) is None:
                        vector = encoders.text(payload.decode("utf-8"))
                        fresh.append((identity, _checked(vector, "text")))
                by_kind["text"]["seconds"] += time.perf_counter() - started
                by_kind["text"]["encoded"] += len(fresh)
                store.put_many(fresh)
            atomic_json(
                edition / TEXT_CARRY / market / f"{symbol}.json",
                dict(
                    market=market,
                    symbol=symbol,
                    source={
                        k: carry[k] for k in ("edition", "configuration_sha256", "encoder_sha256")
                    },
                    rule=carry["rule"],
                    check=carry["check"],
                    candidates=len(entries),
                    compared=len(sampled),
                    mismatched=len(mismatched),
                    mismatched_content=mismatched[:8],
                    decision=REENCODED if mismatched else REUSED,
                    reencoded=len(fresh),
                    checked_at=datetime.now(UTC).isoformat(),
                ),
            )
            totals["assets"] += 1
            totals["reencoded" if mismatched else "reused"] += 1
            totals["compared"] += len(sampled)
            totals["mismatched"] += len(mismatched)
            totals["reencoded_texts"] += len(fresh)
    finally:
        previous.close()
    return totals


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
