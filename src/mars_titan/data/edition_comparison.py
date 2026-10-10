"""Comparar dos ediciones codificadas activo por activo y registrar cada diferencia.

Sirve para verificar la v3.1 frente a la v3. Para cada sesión común exige vectores idénticos
cuando el gráfico tiene la misma huella y el mismo codificador, y registra cada gráfico que
cambia. Si las ediciones declaran codificadores distintos, como la v3 con TF32 en cuDNN y la v3.1
en FP32 estricto, el vector de un mismo PNG puede cambiar y se mide su diferencia absoluta y
relativa. Las sesiones nuevas se cuentan, y las que desaparecen se registran porque la revisión
no debería perder ninguna. También mide cuánto cambian las ventanas de precios comunes.

Los textos heredados de la v3 deben dar la misma media de noticias. Solo un activo cuyo contraste
de textos falló y que los recodificó todos puede cambiarla, y entonces se mide como los gráficos.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .cohort_files import read_manifest
from .storage import atomic_json
from .vector_carry import REENCODED, text_carry_record

# El gráfico se compara por su huella y la posición del precio cambia si se admiten filas
# anteriores. Su efecto se mide sobre la ventana. El resto de columnas debe coincidir.
_COMPARED_APART = {"chart_hash", "charts", "price_end_index"}


def _table(root, market, symbol):
    folder = Path(root) / "samples" / market / symbol
    receipt, _ = read_manifest(folder / "manifest.json")
    table = pq.read_table(folder / "samples.parquet")
    return receipt, {row["session"]: row for row in table.to_pylist()}, table.column_names


def _vector_changes(old, new):
    """Diferencias entre los vectores FP32 de los mismos PNG con dos codificadores."""
    if not old:
        return dict(compared=0, identical=0, max_abs=0.0, max_rel=0.0, max_norm_rel=0.0)
    old, new = np.asarray(old, dtype=np.float32), np.asarray(new, dtype=np.float32)
    identical = (old.view(np.uint32) == new.view(np.uint32)).all(axis=1)
    old, new = old.astype(np.float64), new.astype(np.float64)
    gap = np.abs(old - new)
    scale = np.maximum(np.abs(old), np.abs(new))
    norms = np.linalg.norm(old, axis=1)
    return dict(
        compared=len(old),
        identical=int(identical.sum()),
        max_abs=float(gap.max()),
        max_rel=float((gap[scale > 0] / scale[scale > 0]).max(initial=0.0)),
        max_norm_rel=float(
            (np.linalg.norm(old - new, axis=1)[norms > 0] / norms[norms > 0]).max(initial=0.0)
        ),
    )


def _merge_vector_changes(records):
    if not records:
        return None
    return dict(
        compared=sum(r["compared"] for r in records),
        identical=sum(r["identical"] for r in records),
        **{key: max(r[key] for r in records) for key in ("max_abs", "max_rel", "max_norm_rel")},
    )


def compare_asset(previous_root, current_root, market, symbol):
    """Diferencias de un activo entre la edición anterior y la nueva."""
    _, before, old_columns = _table(previous_root, market, symbol)
    receipt, after, new_columns = _table(current_root, market, symbol)
    if old_columns != new_columns:
        raise ValueError("Las ediciones comparadas tienen columnas distintas")
    configurations = [
        json.loads((Path(root) / "configuration.json").read_text())
        for root in (previous_root, current_root)
    ]
    reencoded = configurations[0]["encoders"] != configurations[1]["encoders"]
    texts = text_carry_record(current_root, market, symbol)
    reencoded_texts = texts is not None and texts["decision"] == REENCODED
    apart = _COMPARED_APART | ({"news"} if reencoded_texts else set())
    same = [name for name in new_columns if name not in apart]
    common = sorted(before.keys() & after.keys())
    record = dict(
        market=market,
        symbol=symbol,
        previous_samples=len(before),
        samples=len(after),
        common=len(common),
        new_sessions=len(after.keys() - before.keys()),
        dropped_sessions=sorted(before.keys() - after.keys()),
        market_absent_windows=receipt.get("market_absent_windows", {}).get("windows", 0),
        changed_charts=[],
        other_differences=[],
    )
    same_png, news = ([], []), ([], [])
    for session in common:
        old, new = before[session], after[session]
        if old["chart_hash"] != new["chart_hash"]:
            record["changed_charts"].append([session, old["chart_hash"], new["chart_hash"]])
        elif reencoded:
            same_png[0].append(old["charts"])
            same_png[1].append(new["charts"])
        elif old["charts"] != new["charts"]:
            record["other_differences"].append([session, "charts_with_same_png"])
        if reencoded_texts:
            # Una ventana sin noticias tiene un vector de ceros, nunca un valor nulo.
            news[0].append(old["news"])
            news[1].append(new["news"])
        for name in same:
            if old[name] != new[name]:
                record["other_differences"].append([session, name])
    record["reencoded_charts"] = _vector_changes(*same_png) if reencoded else None
    record["reencoded_texts"] = _vector_changes(*news) if reencoded_texts else None
    record.update(_window_changes(configurations, market, symbol, before, after))
    return record


def _window_changes(configurations, market, symbol, before, after):
    """Cambio de las ventanas de precio de las sesiones comunes, canales OHLCV."""
    from mars_titan.training.corpus_inputs import _price_contexts

    prices, contexts = {}, set()
    for name, configuration in zip(("before", "after"), configurations, strict=True):
        contexts.add(configuration["context_sessions"])
        path = Path(configuration["prepared_root"]) / market / symbol / "prices.parquet"
        table = pq.read_table(path, columns=["open", "high", "low", "close", "volume"])
        prices[name] = np.column_stack([table[c].to_numpy() for c in table.column_names])
    # Una sesión común tiene ventana completa en ambas ediciones: la anterior no admitía huecos.
    full = sorted(before.keys() & after.keys())
    if len(contexts) != 1:
        raise ValueError("Las ediciones comparadas usan contextos distintos")
    (context,) = contexts
    changed, largest = 0, 0.0
    for start in range(0, len(full), 256):
        chunk = full[start : start + 256]
        old = _price_contexts(
            prices["before"], np.array([before[s]["price_end_index"] for s in chunk]), context
        )
        new = _price_contexts(
            prices["after"], np.array([after[s]["price_end_index"] for s in chunk]), context
        )
        differ = (old.view(np.uint32) != new.view(np.uint32)).any(axis=(1, 2))
        changed += int(differ.sum())
        largest = max(largest, float(np.abs(old - new).max(initial=0.0)))
    return dict(windows_compared=len(full), windows_changed=changed, window_max_abs_change=largest)


def compare_editions(previous_root, current_root, output, *, workers=4):
    """Recorrer todos los activos de la edición nueva con un registro reanudable por activo."""
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("El número de procesos debe estar entre 1 y 16")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "assets.jsonl"
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            done.add((row["market"], row["symbol"]))
    root = Path(current_root) / "samples"
    pending = [
        (folder.parent.name, folder.name)
        for folder in sorted(root.glob("*/*"))
        if (folder / "manifest.json").exists()
        and (folder.parent.name, folder.name) not in done
        and (Path(previous_root) / "samples" / folder.parent.name / folder.name).exists()
    ]
    # Procesos nuevos en lugar de fork, que no es seguro con hilos de Arrow o BLAS ya activos.
    spawn = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as pool,
        log.open("a") as stream,
    ):
        for record in pool.map(
            compare_asset,
            [previous_root] * len(pending),
            [current_root] * len(pending),
            *zip(*pending, strict=True) if pending else ([], []),
            chunksize=4,
        ):
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    summary = dict(
        assets=len(rows),
        previous_samples=sum(r["previous_samples"] for r in rows),
        samples=sum(r["samples"] for r in rows),
        common_sessions=sum(r["common"] for r in rows),
        new_sessions=sum(r["new_sessions"] for r in rows),
        dropped_sessions=sum(len(r["dropped_sessions"]) for r in rows),
        market_absent_windows=sum(r["market_absent_windows"] for r in rows),
        changed_charts=sum(len(r["changed_charts"]) for r in rows),
        other_differences=sum(len(r["other_differences"]) for r in rows),
        windows_compared=sum(r["windows_compared"] for r in rows),
        windows_changed=sum(r["windows_changed"] for r in rows),
        window_max_abs_change=max((r["window_max_abs_change"] for r in rows), default=0.0),
        reencoded_charts=_merge_vector_changes(
            [r["reencoded_charts"] for r in rows if r.get("reencoded_charts")]
        ),
        reencoded_texts=_merge_vector_changes(
            [r["reencoded_texts"] for r in rows if r.get("reencoded_texts")]
        ),
        log=str(log),
    )
    atomic_json(output / "summary.json", summary)
    return summary
