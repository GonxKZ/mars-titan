"""Mismas filas y objetivos en todos los modelos, y ninguna fila de 2024, por otro camino.

La comparación walk-forward ya exige que todos los modelos evalúen las mismas filas mediante
`ForecastPanel.cohort_sha256`. Esta comprobación lo repite sin ese código:
1. Ordena las filas de cada archivo de predicciones por (mercado, activo, instante).
2. Calcula una huella SHA-256 propia con los bytes de esas tres columnas y los bits exactos
   del objetivo en float64.
3. Exige la misma huella para todos los modelos y semillas en cada ventana y tramo.
4. Exige que cada fila caiga en el tramo declarado, sea de un mercado del ámbito, no esté
   duplicada, tenga un objetivo finito y no pertenezca a la reserva de 2024.

Un cambio de un solo bit en un objetivo, una fila de más o una de menos cambian la huella.

Con el diseño conjunto, la comparación quita de cada ventana las filas de los mercados aún
no elegibles y, en un modelo prestado del ámbito conjunto, las de los mercados ajenos al
ámbito. La comprobación vuelve a aplicar esa regla declarada antes de calcular la huella y
cuenta por mercado las filas quitadas, para cotejarlas con las que publica la comparación.
Las filas de cualquier otro mercado siguen contando como defecto.

Las predicciones se leen con `prediction_files.read`, que comprueba la huella del archivo
y devuelve los mismos bits si la retención v2 lo compactó. Una ventana cuyas filas se
liberaron debe regenerarse antes (`run_masked_campaign.py regenerate`).
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from mars_titan.data import prediction_files
from mars_titan.evaluation import walk_forward_comparison as comparison

KIND = "walk_forward_row_identity"
FINAL_TEST_START = "2024-01-01"
KEYS = ("market", "asset_id", "prediction_at")
COLUMNS = (*KEYS, "target")


def _microseconds(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _string_bytes(array):
    """Desplazamientos relativos y bytes de un arreglo de texto, sin relleno de búfer."""
    array = array.cast(pa.large_string())
    _, offsets, data = array.buffers()
    bounds = np.frombuffer(offsets, dtype="<i8", count=len(array) + 1, offset=8 * array.offset)
    payload = b"" if data is None else data.to_pybytes()[bounds[0] : bounds[-1]]
    return (bounds - bounds[0]).tobytes() + payload


def row_digest(table):
    """Huella de (mercado, activo, instante, objetivo) en orden canónico y número de filas."""
    order = pc.sort_indices(table, sort_keys=[(name, "ascending") for name in KEYS])
    ordered = table.select(list(COLUMNS)).take(order).combine_chunks()
    digest = hashlib.sha256(b"mars-titan-row-identity-v1\0")
    for name in ("market", "asset_id"):
        digest.update(_string_bytes(ordered.column(name).chunk(0)))
    moments = ordered.column("prediction_at").cast(pa.int64()).to_numpy()
    digest.update(np.ascontiguousarray(moments, dtype="<i8").tobytes())
    target = ordered.column("target").to_numpy()
    digest.update(np.ascontiguousarray(target, dtype="<f8").tobytes())
    return digest.hexdigest(), ordered


def _duplicates(ordered):
    """Filas con la misma clave que la anterior una vez ordenadas."""
    if ordered.num_rows < 2:
        return 0
    same = None
    for name in KEYS:
        column = ordered.column(name)
        if name == "prediction_at":
            column = column.cast(pa.int64())
        equal = pc.equal(column.slice(1), column.slice(0, ordered.num_rows - 1))
        same = equal if same is None else pc.and_(same, equal)
    return int(pc.sum(same).as_py() or 0)


def dropped_markets(sources, window_id, arm):
    """Mercados cuyas filas no cuenta la comparación en esa ventana, según el diseño declarado.

    Son los mercados del ámbito que aún no son elegibles en la ventana y, en un modelo prestado
    del ámbito conjunto, los mercados de ese ámbito que no son de este.
    """
    markets = sources["markets"]
    dropped = {market for market in markets if window_id not in sources["eligible"][market]}
    if arm in sources["borrowed"]:
        dropped |= set(comparison.SCOPES[sources["joint_scope"]]) - set(markets)
    return sorted(dropped)


def kept_rows(table, dropped):
    """Filas que cuenta la comparación y recuento por mercado de las que quita."""
    if not dropped:
        return table, {}
    market = table.column("market").to_numpy(zero_copy_only=False)
    removed = np.isin(market, dropped)
    counts = {name: int(np.count_nonzero(market == name)) for name in dropped}
    return table.filter(pa.array(~removed)), counts


def inspect_table(table, segment, markets, dropped=()):
    """Huella y defectos de una tabla de predicciones frente a su tramo declarado.

    `dropped` son los mercados que el diseño quita de esta ventana y este modelo. Sus filas
    no entran en la huella y se cuentan aparte.
    """
    nulls = sum(table.column(name).null_count for name in COLUMNS)
    if nulls:
        return dict(rows=table.num_rows, null_values=nulls, passed=False)
    table, excluded = kept_rows(table, dropped)
    digest, ordered = row_digest(table)
    moments = ordered.column("prediction_at").cast(pa.int64()).to_numpy()
    start, end = (_microseconds(day) for day in segment)
    market = ordered.column("market").to_numpy(zero_copy_only=False)
    defects = dict(
        rows_in_2024=int(np.count_nonzero(moments >= _microseconds(FINAL_TEST_START))),
        rows_outside_segment=int(np.count_nonzero((moments < start) | (moments >= end))),
        rows_outside_scope=int(np.count_nonzero(~np.isin(market, list(markets)))),
        duplicated_rows=_duplicates(ordered),
        non_finite_targets=int(np.count_nonzero(~np.isfinite(ordered.column("target").to_numpy()))),
    )
    return dict(
        rows=ordered.num_rows,
        digest=digest,
        excluded_rows=excluded,
        **defects,
        passed=not any(defects.values()),
    )


def check_sources(config_path, sources_path, scope):
    """Huellas por ventana y tramo y defectos de cada archivo de un manifiesto de fuentes."""
    config = comparison.resolve_config(config_path)
    return check_loaded(config, comparison.load_sources(sources_path, config, scope))


def check_loaded(config, sources):
    """Lo mismo que `check_sources` con la configuración y las fuentes ya validadas."""
    windows, markets, scope = sources["windows"], sources["markets"], sources["scope"]
    files, by_segment = [], {}
    for (arm, seed, window_id), records in sorted(sources["files"].items()):
        dropped = dropped_markets(sources, window_id, arm)
        for partition, record in sorted(records.items()):
            table = prediction_files.read(record["path"], record["sha256"], COLUMNS)
            finding = inspect_table(table, windows[window_id][partition], markets, dropped)
            files.append(dict(arm=arm, seed=seed, window=window_id, partition=partition, **finding))
            if "digest" in finding:
                by_segment.setdefault(f"{window_id}/{partition}", {})[f"{arm}/{seed}"] = finding[
                    "digest"
                ]
    shared = {}
    for segment, digests in sorted(by_segment.items()):
        distinct = sorted(set(digests.values()))
        shared[segment] = dict(
            files=len(digests),
            distinct_digests=len(distinct),
            groups=[sorted(k for k, v in digests.items() if v == value) for value in distinct],
            passed=len(distinct) == 1,
        )
    failures = sum(not entry["passed"] for entry in files) + sum(
        not entry["passed"] for entry in shared.values()
    )
    return dict(
        schema_version=1,
        kind=KIND,
        created_at_utc=datetime.now(UTC).isoformat(),
        scope=scope,
        configuration_sha256=config["sha256"],
        sources_sha256=sources["sha256"],
        final_test_start=FINAL_TEST_START,
        files=files,
        shared_rows=shared,
        failures=failures,
        passed=failures == 0,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("sources")
    parser.add_argument("scope")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = check_sources(args.config, args.sources, args.scope)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
