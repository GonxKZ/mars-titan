"""Comprobar con datos reales y medir el codificador de control sin preentrenamiento.

Usa noticias y precios de la edición preparada desde 2000. No carga ningún modelo de la
campaña, no usa la GPU ni ejecuta pasos de optimizador:

1. Codifica una muestra de noticias reales de US y CN dos veces con instancias distintas y
   exige vectores idénticos, finitos y de norma 1. Mide el tiempo por texto y por carácter.
2. Dibuja gráficos reales de 64 sesiones con `charts.chart_png`, los codifica dos veces y mide
   por separado el dibujo y el codificador.
3. Cuenta las noticias y las filas de precios de toda la edición para estimar el coste de
   codificar una edición de control completa.
4. Mide la sensibilidad (`evaluation.encoder_sensitivity`) con dos comparaciones sintéticas del
   tamaño de la de US: 19 ventanas, tres semillas y los dos brazos declarados.
5. Con `--weights-history`, descarga el `pytorch_model.bin` del commit de MiniLM del 23 de junio
   de 2021 y comprueba tensor a tensor que coincide con el `model.safetensors` fijado.

Los vectores no se guardan. El recibo solo contiene recuentos, tiempos y huellas.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data import pretraining_free_encoders as pf
from mars_titan.data.charts import chart_png
from mars_titan.data.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PUBLISHED = "e62509716f15c5fd03a6fd3156a4bc5e43f83f26"
PINNED = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


def _rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _assets(prepared, market, count, rng):
    folders = sorted(path for path in (prepared / market).iterdir() if path.is_dir())
    chosen = rng.choice(len(folders), size=min(count, len(folders)), replace=False)
    return [folders[i] for i in sorted(chosen.tolist())]


def _texts(folders, limit, rng):
    texts = []
    for folder in folders:
        path = folder / "news" / "news.parquet"
        if path.is_file():
            texts += [t for t in pq.read_table(path, columns=["text"])["text"].to_pylist() if t]
    order = rng.permutation(len(texts))[:limit]
    return [texts[i] for i in order]


def _charts(folders, limit, rng):
    """Gráficos de las últimas 64 filas hasta sesiones al azar de cada activo."""
    pngs, rejected = [], 0
    per_asset = max(1, limit // len(folders))
    started = time.perf_counter()
    for folder in folders:
        table = pq.read_table(folder / "prices.parquet", columns=["open", "high", "low", "close"])
        prices = np.column_stack([table[name].to_numpy() for name in table.column_names])
        if len(prices) < 64:
            continue
        for end in rng.integers(63, len(prices), size=per_asset).tolist():
            try:
                pngs.append(chart_png(prices, end_index=end))
            except ValueError:
                rejected += 1
    return pngs[:limit], rejected, time.perf_counter() - started


def _timed(function, items, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        outputs = [function(item) for item in items]
        samples.append(time.perf_counter() - started)
    return outputs, samples


def _check_texts(texts, repeats):
    first, second = pf.PretrainingFreeEncoders(), pf.PretrainingFreeEncoders()
    vectors, samples = _timed(first.text, texts, repeats)
    again = [second.text(text) for text in texts]
    identical = all(np.array_equal(a, b) for a, b in zip(vectors, again, strict=True))
    norms = np.array([np.linalg.norm(v.astype(np.float64)) for v in vectors])
    characters = sum(len(text) for text in texts)
    best = min(samples)
    return dict(
        texts=len(texts),
        characters=characters,
        identical_between_instances=identical,
        finite=bool(all(np.isfinite(v).all() for v in vectors)),
        max_norm_error=float(np.abs(norms - 1).max()),
        seconds=dict(p50=statistics.median(samples), min=best, repeats=repeats),
        microseconds_per_text=1e6 * best / len(texts),
        microseconds_per_thousand_characters=1e9 * best / characters,
    )


def _check_charts(pngs, repeats):
    first, second = pf.PretrainingFreeEncoders(), pf.PretrainingFreeEncoders()
    batches = [pngs[i : i + 64] for i in range(0, len(pngs), 64)]
    outputs, samples = _timed(first.images, batches, repeats)
    again = [second.images(batch) for batch in batches]
    vectors = np.concatenate(outputs)
    best = min(samples)
    return dict(
        charts=len(pngs),
        identical_between_instances=bool(np.array_equal(vectors, np.concatenate(again))),
        within_unit_interval=bool(vectors.min() >= 0 and vectors.max() <= 1),
        seconds=dict(p50=statistics.median(samples), min=best, repeats=repeats),
        microseconds_per_chart=1e6 * best / len(pngs),
    )


def _edition_counts(prepared):
    counts = {}
    for market in ("US", "CN"):
        news = prices = assets = 0
        for folder in (prepared / market).iterdir():
            if not folder.is_dir():
                continue
            assets += 1
            prices += pq.ParquetFile(folder / "prices.parquet").metadata.num_rows
            path = folder / "news" / "news.parquet"
            if path.is_file():
                news += pq.ParquetFile(path).metadata.num_rows
        counts[market] = dict(assets=assets, news_rows=news, price_rows=prices)
    return counts


def _sensitivity_cost(folder, repeats):
    """Evaluar la sensibilidad sobre dos comparaciones sintéticas con la forma de US."""
    import pyarrow as pa

    from mars_titan.data.storage import sha256
    from mars_titan.evaluation import encoder_sensitivity as es
    from mars_titan.evaluation.walk_forward_comparison import REPORT_KIND

    declaration = es.load_declaration(ROOT / "configs/encoders/pretraining-free-sensitivity.json")
    day, moments = date(2005, 1, 3), []
    while day < date(2024, 1, 1):
        if day.weekday() < 5:
            moments.append(datetime(day.year, day.month, day.day, 21, tzinfo=UTC))
        day += timedelta(days=1)
    rng = np.random.default_rng(7)
    for edition in ("frozen", "control"):
        rows = []
        for arm in declaration["arms"]:
            for seed in (42, 43, 44):
                mae = 1.0 + rng.uniform(0, 0.2, len(moments))
                rows.append(
                    dict(
                        arm=pa.array([arm] * len(moments)),
                        seed=pa.array([seed] * len(moments)),
                        window=pa.array([f"fold-{m.year - 2005:03d}" for m in moments]),
                        quantiles=pa.array(["raw"] * len(moments)),
                        market=pa.array(["US"] * len(moments)),
                        prediction_at=pa.array(moments, pa.timestamp("us", tz="UTC")),
                        samples=pa.array([500] * len(moments), pa.int64()),
                        mae=pa.array(mae),
                        positive_targets=pa.array([250] * len(moments), pa.int64()),
                        negative_targets=pa.array([250] * len(moments), pa.int64()),
                    )
                )
        table = pa.concat_tables([pa.table(row) for row in rows])
        target = folder / edition
        target.mkdir(parents=True)
        pq.write_table(table, target / "sessions.parquet")
        report = dict(
            kind=REPORT_KIND,
            status="completed",
            final_test_opened=False,
            scope="US",
            markets=["US"],
            configuration=dict(name="synthetic", sha256="c" * 64),
            edition={"US": edition},
            windows={
                f"fold-{y - 2005:03d}": dict(evaluation=[f"{y}-01-01", f"{y + 1}-01-01"])
                for y in range(2005, 2024)
            },
            artifacts={"sessions.parquet": sha256(target / "sessions.parquet")},
        )
        (target / "comparison.json").write_text(json.dumps(report))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        es.evaluate(declaration, folder / "frozen", folder / "control")
        samples.append(time.perf_counter() - started)
    return dict(
        sessions=len(moments),
        arms=len(declaration["arms"]),
        seeds=3,
        replicates=declaration["bootstrap"]["replicates"],
        seconds=dict(p50=statistics.median(samples), min=min(samples), repeats=repeats),
    )


def _weights_history(scratch):
    """Pesos del 23 de junio de 2021 frente a los fijados, tensor a tensor."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    pinned = hf_hub_download(
        TEXT_MODEL, "model.safetensors", revision=PINNED, local_files_only=True, token=False
    )
    started = time.perf_counter()
    old = hf_hub_download(
        TEXT_MODEL, "pytorch_model.bin", revision=PUBLISHED, cache_dir=scratch, token=False
    )
    download = time.perf_counter() - started
    digest = hashlib.sha256(Path(old).read_bytes()).hexdigest()
    published = torch.load(old, map_location="cpu", weights_only=True)
    current = load_file(pinned, device="cpu")
    # Algunos ficheros antiguos guardan buffers que safetensors omite, como position_ids.
    shared = sorted(set(published) & set(current))
    equal = all(torch.equal(published[k], current[k]) for k in shared)
    return dict(
        published_commit=PUBLISHED,
        pinned_revision=PINNED,
        published_bin_sha256=digest,
        pinned_safetensors_sha256=hashlib.sha256(Path(pinned).read_bytes()).hexdigest(),
        tensors_compared=len(shared),
        only_in_published=sorted(set(published) - set(current)),
        only_in_pinned=sorted(set(current) - set(published)),
        identical=equal,
        download_seconds=download,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=int, default=40, help="Activos por mercado")
    parser.add_argument("--texts", type=int, default=2_000, help="Noticias por mercado")
    parser.add_argument("--charts", type=int, default=1_024, help="Gráficos por mercado")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--weights-history", action="store_true")
    parser.add_argument("--scratch", type=Path, help="Carpeta temporal para la descarga")
    args = parser.parse_args(argv)
    prepared = args.prepared / "prepared"
    rng = np.random.default_rng(20261010)
    load = os.getloadavg()
    markets = {}
    for market in ("US", "CN"):
        folders = _assets(prepared, market, args.assets, rng)
        texts = _texts(folders, args.texts, rng)
        pngs, rejected, render = _charts(folders, args.charts, rng)
        markets[market] = dict(
            assets=len(folders),
            text=_check_texts(texts, args.repeats),
            chart=dict(
                _check_charts(pngs, args.repeats),
                rejected_windows=rejected,
                render_seconds=render,
                render_microseconds_per_chart=1e6 * render / max(1, len(pngs)),
            ),
        )
    counts = _edition_counts(prepared)
    estimate = {
        market: dict(
            text_hours=counts[market]["news_rows"]
            * markets[market]["text"]["microseconds_per_text"]
            / 3.6e9,
            chart_hours_upper_bound=counts[market]["price_rows"]
            * markets[market]["chart"]["microseconds_per_chart"]
            / 3.6e9,
        )
        for market in counts
    }
    scratch = Path(tempfile.mkdtemp(dir=args.scratch))
    try:
        sensitivity = _sensitivity_cost(scratch / "sensitivity", args.repeats)
        history = _weights_history(scratch / "hub") if args.weights_history else None
    finally:
        shutil.rmtree(scratch)
    receipt = dict(
        schema_version=1,
        kind="pretraining_free_encoder_real_data_check",
        issue=10,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        encoder=pf.PretrainingFreeEncoders().spec,
        sources_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "src/mars_titan/data/pretraining_free_encoders.py",
                "src/mars_titan/evaluation/encoder_sensitivity.py",
                "configs/encoders/pretraining-free-sensitivity.json",
                "benchmarks/pretraining_free_encoders_real.py",
            )
        },
        data=dict(
            prepared=str(args.prepared),
            prepared_manifest_sha256=hashlib.sha256(
                (args.prepared / "manifest.json").read_bytes()
            ).hexdigest(),
            note="noticias y precios reales preparados hasta 2023. Sin modelos de la campaña",
        ),
        markets=markets,
        edition_counts=counts,
        control_edition_estimate=estimate,
        sensitivity=sensitivity,
        weights_history=history,
        environment=dict(
            cpu=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            python=platform.python_version(),
            omp_threads=os.environ.get("OMP_NUM_THREADS"),
            load_average_at_start=load,
            process_peak_rss_mib=_rss_mib(),
        ),
        not_measured=[
            "GPU, porque el control trabaja en CPU",
            "energía, sin instrumento",
            "la codificación completa de la edición de control, solo estimada con la muestra",
            "el ajuste de los brazos sobre la edición de control, bloqueado",
        ],
    )
    atomic_json(args.output, receipt)
    print(json.dumps(dict(markets=markets, estimate=estimate, history=history), indent=1))


if __name__ == "__main__":
    main()
