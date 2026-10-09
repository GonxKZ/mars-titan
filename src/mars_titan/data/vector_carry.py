"""Reutilizar los vectores de una edición anterior cuando la entrada codificada es idéntica.

Un gráfico se codifica a partir de su PNG y un texto a partir de su contenido. Con el mismo
codificador, la misma entrada da el mismo vector. La edición v3 lo comprobó al recodificar
activos con la caché vacía y obtener muestras idénticas bit a bit. Por eso un gráfico de la v3.1
cuyo PNG tiene la huella de uno de la v3 recibe el vector ya calculado, y solo los gráficos
nuevos o cambiados necesitan la GPU. Los textos se buscan primero en la caché nueva y después,
en solo lectura, en la de la edición anterior.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .cohort_files import read_manifest
from .storage import sha256


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
    """Caché que añade los gráficos de la edición anterior, activo por activo."""

    def __init__(self, cache, previous_root, encoder_hash, *, fallback=None):
        self.cache, self.fallback = cache, fallback
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
        if vector is None and self.fallback is not None:
            vector = self.fallback.get(identity, **kwargs)
        return vector

    def put(self, identity, vector):
        self.cache.put(identity, vector)

    def close(self):
        self.cache.close()
        if self.fallback is not None:
            self.fallback.close()
