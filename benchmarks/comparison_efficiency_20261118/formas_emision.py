"""Contar formas y simular una caché LRU sobre los índices de ajuste de la comparación."""

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256


def contar_formas(filas_por_instante, bloque, capacidades=(1, 4, 8, 16)):
    """Dividir cada instante en el orden de emisión y contar aciertos por tamaño exacto."""
    formas = []
    for filas in filas_por_instante:
        completos, resto = divmod(filas, bloque)
        formas.extend([bloque] * completos)
        if resto:
            formas.append(resto)
    histograma = Counter(formas)
    aciertos = {}
    for capacidad in capacidades:
        cache, hits = OrderedDict(), 0
        for n in formas:
            hits += n in cache
            cache[n] = None
            cache.move_to_end(n)
            if len(cache) > capacidad:
                cache.popitem(last=False)
        aciertos[capacidad] = hits
    return dict(
        emisiones=len(formas),
        formas_distintas=len(histograma),
        bloques_completos=histograma[bloque],
        histograma=dict(sorted(histograma.items())),
        aciertos_lru=aciertos,
    )


def leer_indice(ruta):
    meta, _ = read_manifest(ruta)
    if (
        meta.get("final_test_opened") is not False
        or meta["identity"]["phase"]["partition"] != "train"
    ):
        raise ValueError("Se necesita un índice de ajuste cerrado a la reserva final")
    eventos = ruta.parent / meta["events_path"]
    if (
        eventos.parent != ruta.parent
        or eventos.is_symlink()
        or sha256(eventos) != meta["events_sha256"]
    ):
        raise ValueError("La ruta o la huella de los eventos no corresponde al índice")
    sesiones = Counter()
    with pq.ParquetFile(eventos) as archivo:
        for lote in archivo.iter_batches(
            batch_size=65_536, columns=["event_at", "kind"], use_threads=False
        ):
            instantes = lote["event_at"].to_numpy()[lote["kind"].to_numpy() == 0]
            claves, filas = np.unique(instantes, return_counts=True)
            sesiones.update({int(k): int(n) for k, n in zip(claves, filas, strict=True)})
    orden = [sesiones[k] for k in sorted(sesiones)]
    return dict(
        indice_sha256=meta["events_sha256"],
        sesiones=len(orden),
        filas=sum(orden),
        bloques={b: contar_formas(orden, b) for b in (512, 1024)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estado", type=Path)
    parser.add_argument("salida", type=Path)
    args = parser.parse_args()
    por_origen = {}
    for ruta in sorted((args.estado / "cohorts").glob("*/cohort.json")):
        cohorte, _ = read_manifest(ruta)
        raiz = args.estado / "indices" / cohorte["manifest_sha256"][:16]
        indices = sorted(raiz.glob("train-*/manifest.json"))
        if indices:
            if len(indices) != 1:
                raise ValueError("Hay más de un índice de ajuste para la cohorte")
            por_origen[ruta.parent.name] = leer_indice(indices[0])
    if not por_origen:
        raise ValueError("El estado no tiene índices de ajuste disponibles")
    with args.salida.open("x", encoding="utf-8") as f:
        json.dump(por_origen, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
