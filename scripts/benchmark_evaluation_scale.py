"""Medir tiempo y memoria de la comparación walk-forward y de la cartera a escala real.

Los datos son sintéticos con las formas de la campaña A: las filas por ventana de
calibración y evaluación del informe de vistas, los brazos y semillas de la comparación
declarada y el protocolo real del ámbito. Las predicciones no salen de ningún modelo. Cada
ventana escribe ``--variants`` tablas distintas que los brazos y semillas comparten en
ciclo, para no multiplicar el disco por el número de brazos. La lectura y la puntuación
se repiten igualmente para cada brazo y semilla.

``prepare`` escribe configuración, vistas, predicciones, fuentes y, con ``--edition``,
una edición sin ajustar sintética con los activos usados. ``measure`` ejecuta la
comparación o la cartera en un proceso aparte con ``/usr/bin/time`` y guarda el tiempo,
la CPU y la memoria residente máxima. Nada lee la edición real ni el año 2024.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.data.unadjusted_edition import CUTOFF, EVENT_SCHEMA, PRICE_SCHEMA
from mars_titan.evaluation.splits import build_folds
from mars_titan.models.quantile_head import QUANTILE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
VIEWS = ROOT / "reports/data/campaign-a-views-20261009.json"
OFFSETS = np.array([-1.96, -1.28, 0.0, 1.28, 1.96])
SECONDARY = ("modality_strata", "modality_ablation", "long_short")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _decisions(market, segment):
    """Cierres del segmento sin el último, que la purga retira porque t+1 queda fuera."""
    end = str(np.datetime64(segment[1]) - np.timedelta64(1, "D"))
    clock = MarketClock(market, segment[0], end)
    return [moment for moment in clock.decisions][:-1]


def _table(market, segment, rows, assets, variant, seed):
    moments = _decisions(market, segment)
    per = np.full(len(moments), rows // len(moments))
    per[: rows % len(moments)] += 1
    times = np.repeat(np.array([int(m.timestamp()) * 1_000_000 for m in moments]), per)
    index = np.concatenate([np.arange(k) for k in per])
    # El objetivo es común a todas las variantes. Solo cambia el ruido de la predicción.
    target = np.random.default_rng(seed).normal(0.0, 0.02, rows)
    noise = np.random.default_rng([seed, variant]).normal(0.0, 0.02, rows)
    prediction = (0.1 + 0.05 * variant) * target + noise
    columns = dict(
        asset_id=pa.array(np.asarray(assets)[index], pa.string()),
        market=pa.array([market] * rows, pa.string()),
        prediction_at=pa.array(times, pa.timestamp("us", tz="UTC")),
        target=target,
        prediction=prediction,
    )
    columns.update(zip(QUANTILE_COLUMNS, (prediction[:, None] + 0.02 * OFFSETS).T, strict=True))
    return pa.table(columns), int(per.max())


def _edition(root, market, assets, first):
    """Edición sintética con precios en céntimos, verificados y sin eventos."""
    clock = MarketClock(market, first, CUTOFF)
    days = np.array([day.isoformat() for day in clock.days])
    receipts = {}
    for k, asset in enumerate(assets):
        rng = np.random.default_rng(k)
        close = np.round(20 + np.cumsum(rng.normal(0, 0.1, len(days))).clip(-15, None), 2)
        opened = np.round(close * (1 + rng.normal(0, 0.005, len(days))), 2)
        prices = pa.table(
            [
                pa.array(days),
                opened,
                np.maximum(opened, close) + 0.05,
                np.minimum(opened, close) - 0.05,
                close,
                np.full(len(days), 1e6),
                np.ones(len(days)),
                np.zeros(len(days)),
                np.ones(len(days), dtype=bool),
                np.ones(len(days), dtype=bool),
            ],
            schema=PRICE_SCHEMA,
        )
        folder = root / "assets" / asset
        folder.mkdir(parents=True, exist_ok=True)
        pq.write_table(prices, folder / "prices.parquet")
        pq.write_table(EVENT_SCHEMA.empty_table(), folder / "events.parquet")
        receipts[asset] = {
            name: sha256(folder / name) for name in ("prices.parquet", "events.parquet")
        }
    identity = dict(schema_version=1, policy=dict(cutoff=CUTOFF), code={"benchmark": "1"})
    receipts = dict(sorted(receipts.items()))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for artifacts in receipts.values():
        digest.update(json.dumps(artifacts, sort_keys=True).encode())
    manifest = dict(
        kind="unadjusted_price_edition",
        edition_id=digest.hexdigest(),
        identity=identity,
        price_basis="unadjusted_reconstructed",
        corporate_actions_complete=False,
        receipts=receipts,
    )
    (root / "manifest.json").write_text(json.dumps(manifest))


def prepare(root, scope, windows, variants, edition):
    """Escribir configuración, vistas, predicciones y fuentes con las formas reales."""
    market = scope
    counts = json.loads(VIEWS.read_text())["preparation"]["scopes"][scope]["counts_by_window"]
    declared = json.loads(COMPARISON.read_text())
    protocols = {
        m: str((COMPARISON.parent / name).resolve())
        for m, name in declared["scopes"][scope]["protocols"].items()
    }
    protocol = json.loads(Path(protocols[market]).read_text())
    folds = build_folds(protocol)[-windows:]
    declared["scopes"] = {scope: dict(protocols=protocols, windows=[fold["id"] for fold in folds])}
    _write_json(root / "config" / "comparison.json", declared)
    # La comparación se mide sin secciones secundarias: los estratos necesitarían las
    # muestras reales de las vistas. La cartera usa la configuración completa.
    plain = {key: value for key, value in declared.items() if key not in SECONDARY}
    _write_json(root / "config" / "comparison-v1.json", dict(plain, schema_version=1))
    assets = [f"{market}/S{k:05d}" for k in range(6000)]
    used, sources_windows, files = 0, {}, {}
    shapes = []
    for fold, count in zip(folds, counts[-windows:], strict=True):
        view = dict(
            kind="corpus_supervision",
            cohort_complete=True,
            final_test_opened=False,
            temporal_view=dict(
                schema_version=2,
                **policy_identity(HISTORICAL_MASKED),
                protocol=protocol,
                fold=fold,
                selection_partition="validation",
                recover_annual_boundaries=True,
                parent_manifest="/unavailable/parent/manifest.json",
                parent_sha256="e" * 64,
            ),
            **policy_identity(HISTORICAL_MASKED),
        )
        path = root / "views" / f"{fold['id']}.json"
        sources_windows[fold["id"]] = dict(
            view=dict(path=str(path), sha256=_write_json(path, view))
        )
        for partition in ("calibration", "evaluation"):
            for variant in range(variants):
                seed = zlib.crc32(f"{fold['id']}/{partition}".encode())
                table, width = _table(
                    market, fold[partition], count[partition], assets, variant, seed
                )
                used = max(used, width)
                target = root / "predictions" / fold["id"] / f"{partition}-{variant}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, target)
                files[fold["id"], partition, variant] = dict(
                    path=str(target), sha256=sha256(target)
                )
        shapes.append(
            dict(
                window=fold["id"], calibration=count["calibration"], evaluation=count["evaluation"]
            )
        )
    arms, k = {}, 0
    for name, arm in declared["arms"].items():
        if arm["output"] == "zero_control":
            continue
        arms[name] = {}
        for seed in arm["seeds"]:
            entries = arms[name][str(seed)] = {}
            for fold in folds:
                entry = dict(
                    input_policy=HISTORICAL_MASKED,
                    view_sha256=sources_windows[fold["id"]]["view"]["sha256"],
                )
                parts = ["evaluation"] + (["calibration"] if arm["output"] != "point" else [])
                for part in parts:
                    entry[part] = files[fold["id"], part, k % variants]
                entries[fold["id"]] = entry
            k += 1
    _write_json(
        root / "sources" / f"{scope}.json",
        dict(
            schema_version=1,
            kind="walk_forward_prediction_sources",
            scope=scope,
            input_policy=HISTORICAL_MASKED,
            windows=sources_windows,
            arms=arms,
        ),
    )
    if edition:
        first = folds[0]["calibration"][0]
        _edition(root / "edition", market, assets[:used], first)
    shape = dict(
        scope=scope,
        windows=windows,
        variants=variants,
        arms=len(declared["arms"]),
        arm_seed_series=sum(len(seeds) for seeds in arms.values()),
        assets_per_session_max=used,
        rows=shapes,
        evaluation_rows=sum(item["evaluation"] for item in shapes),
        calibration_rows=sum(item["calibration"] for item in shapes),
        edition=edition,
    )
    _write_json(root / "shape.json", shape)
    return shape


def measure(root, what, output, scope):
    """Ejecutar una medida en un proceso aparte y registrar tiempo, CPU y memoria."""
    timing = output.with_suffix(".time")
    command = [sys.executable, "-m"]
    if what == "walk_forward":
        command += [
            "mars_titan.evaluation.walk_forward_comparison",
            "--config",
            str(root / "config" / "comparison-v1.json"),
        ]
    else:
        command += [
            "mars_titan.evaluation.long_short_comparison",
            "--config",
            str(root / "config" / "comparison.json"),
            "--edition",
            str(root / "edition"),
        ]
    command += ["--sources", str(root / "sources" / f"{scope}.json"), "--scope", scope]
    command += ["--output", str(output)]
    began = time.perf_counter()
    result = subprocess.run(
        ["/usr/bin/time", "-f", "%e %U %S %M", "-o", str(timing), *command],
        capture_output=True,
        text=True,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "-1"},
    )
    elapsed = time.perf_counter() - began
    if result.returncode != 0:
        raise SystemExit(result.stderr[-4000:])
    wall, user, system, rss = timing.read_text().split()[-4:]
    return dict(
        what=what,
        wall_seconds=float(wall),
        user_seconds=float(user),
        system_seconds=float(system),
        peak_rss_bytes=int(rss) * 1024,
        harness_seconds=elapsed,
        stdout=result.stdout.strip(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    run = commands.add_parser("measure")
    for command in (prepared, run):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--scope", choices=("US", "CN"), default="US")
    prepared.add_argument("--windows", type=int, default=19)
    prepared.add_argument("--variants", type=int, default=3)
    prepared.add_argument("--edition", action="store_true")
    run.add_argument("--what", choices=("walk_forward", "long_short"), required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.root.exists():
            parser.error("La raíz debe ser nueva")
        began = time.perf_counter()
        shape = prepare(args.root, args.scope, args.windows, args.variants, args.edition)
        print(json.dumps(dict(shape, prepare_seconds=time.perf_counter() - began), indent=1))
        return
    result = measure(args.root, args.what, args.output, args.scope)
    result.update(
        shape=json.loads((args.root / "shape.json").read_text()),
        host=dict(
            platform=platform.platform(),
            cpus=os.cpu_count(),
            load_average=os.getloadavg(),
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        ),
        measured_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(args.output.with_suffix(".json"), result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
