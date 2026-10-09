"""Medir el recorrido tabular de la campaña sin ajustar ningún modelo.

Órdenes:

- `read`: caudal del lector y de la matriz float32 de una vista.
- `gram`: Gram Ridge por bloque en FP64, FP32 y TF32 con datos aleatorios.
- `ridge`: estadísticas Ridge actuales frente a una revisión de referencia, con filas reales.
- `matrix`: construcción de la matriz cuantizada externa con datos aleatorios.
- `validation`: evaluación de validación de una ronda con un bosque escrito a mano.

Ninguna orden llama a un optimizador ni a `xgb.train` ni resuelve un sistema Ridge. Las
rutas CUDA usan `cuda:0` y fallan si no está disponible. Cada orden imprime un JSON.
"""

import argparse
import hashlib
import importlib
import json
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = "historical_masked_2000_v1"


def environment():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible. No se cambia a CPU")
    versions = {}
    for name in ("xgboost", "cupy"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except ImportError:
            versions[name] = None
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        commit=commit or None,
        **versions,
    )


def peak_rss_mib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class Peaks:
    """Muestrear VRAM ocupada de cuda:0, RSS del proceso y bytes de un directorio."""

    def __init__(self, directory=None, interval=0.05):
        import cupy as cp

        self.cp, self.directory, self.interval = cp, directory, interval
        self.device = self.rss = self.disk = 0
        self.stop = threading.Event()
        free, total = cp.cuda.runtime.memGetInfo()
        self.before = total - free
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        page = resource.getpagesize()
        while not self.stop.is_set():
            free, total = self.cp.cuda.runtime.memGetInfo()
            self.device = max(self.device, total - free)
            with open("/proc/self/statm") as stream:
                self.rss = max(self.rss, int(stream.read().split()[1]) * page)
            if self.directory is not None and Path(self.directory).exists():
                self.disk = max(self.disk, directory_bytes(self.directory))
            time.sleep(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def summary(self):
        return dict(
            device_used_before_mib=round(self.before / 2**20),
            device_used_peak_mib=round(max(self.device, self.before) / 2**20),
            rss_peak_mib=round(self.rss / 2**20),
            disk_peak_bytes=self.disk,
        )


def directory_bytes(directory):
    total = 0
    for path in Path(directory).rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def reader_blocks(manifest, partition, rows, batch_size):
    """Bloques float32 del lector real, hasta `rows` filas."""
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.tabular_corpus import _matrix

    dataset = CorpusDataset(manifest, input_policy=POLICY)
    seen = 0
    for batch in dataset.batches(partition=partition, batch_size=batch_size, epoch=0, seed=0):
        yield _matrix(batch, np.float32, presence=True), batch
        seen += len(batch["target"])
        if seen >= rows:
            return


def run_read(args):
    started = time.perf_counter()
    rows, features = 0, None
    for matrix, batch in reader_blocks(args.manifest, args.partition, args.rows, args.batch_size):
        rows += len(batch["target"])
        features = matrix.shape[1]
    elapsed = time.perf_counter() - started
    return dict(
        manifest=str(args.manifest),
        partition=args.partition,
        batch_size=args.batch_size,
        features=features,
        rows=rows,
        seconds=round(elapsed, 2),
        rows_per_second=round(rows / elapsed),
        peak_rss_mib=peak_rss_mib(),
    )


def run_gram(args):
    """Z'Z por bloque. FP64 es la referencia. FP32 y TF32 solo se miden y se comparan."""
    import torch

    from mars_titan.models.baselines import ridge

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(1)
    host = [rng.standard_normal((args.rows, args.features), dtype=np.float32) for _ in range(4)]
    mean = rng.standard_normal(args.features)
    scale = rng.random(args.features) + 0.5
    target = rng.standard_normal(args.rows)
    results = {}

    def timed(name, step):
        for index in range(3):
            step(index)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for index in range(args.blocks):
            step(index)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / args.blocks
        results[name] = dict(
            ms_per_block=round(seconds * 1e3, 3), rows_per_second=round(args.rows / seconds)
        )

    center = torch.as_tensor(mean, device=device)
    spread = torch.as_tensor(scale, device=device)
    gram = torch.zeros((args.features, args.features), dtype=torch.float64, device=device)

    def previous(index):
        # Ruta anterior: float64 en CPU, estandarización numpy y copia paginable.
        x = host[index % 4].astype(np.float64)
        features = torch.as_tensor((x - mean) / scale, dtype=torch.float64, device=device)
        gram.addmm_(features.T, features)

    timed("previous_cpu_standardize_pageable_fp64", previous)
    blocks = [(host[i % 4], target) for i in range(args.blocks)]
    ridge.centered_normal_equations(
        iter(blocks[:3]), mean, scale, 0.0, 3 * args.rows, device=device
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    # Ruta actual del proyecto: float32 fijado por turnos y estandarización en la GPU.
    ridge.centered_normal_equations(
        iter(blocks), mean, scale, 0.0, args.rows * args.blocks, device=device
    )
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / args.blocks
    results["current_pinned_fp32_gpu_standardize_fp64"] = dict(
        ms_per_block=round(seconds * 1e3, 3), rows_per_second=round(args.rows / seconds)
    )
    z64 = [(torch.from_numpy(h).to(device).to(torch.float64) - center) / spread for h in host]
    z32 = [z.to(torch.float32) for z in z64]
    timed("gemm_fp64", lambda i: gram.addmm_(z64[i % 4].T, z64[i % 4]))
    timed("gemm_fp32_accumulate_fp64", lambda i: gram.add_(z32[i % 4].T @ z32[i % 4]))
    torch.backends.cuda.matmul.allow_tf32 = True
    timed("gemm_tf32_accumulate_fp64", lambda i: gram.add_(z32[i % 4].T @ z32[i % 4]))
    reference = z64[0].T @ z64[0]
    norm = reference.abs().max().item()
    tf32 = (z32[0].T @ z32[0]).to(torch.float64)
    results["tf32_max_abs_error_over_max"] = (tf32 - reference).abs().max().item() / norm
    torch.backends.cuda.matmul.allow_tf32 = False
    fp32 = (z32[0].T @ z32[0]).to(torch.float64)
    results["fp32_max_abs_error_over_max"] = (fp32 - reference).abs().max().item() / norm
    flops = 2 * args.rows * args.features**2
    for name in ("gemm_fp64", "gemm_fp32_accumulate_fp64", "gemm_tf32_accumulate_fp64"):
        results[f"{name}_tflops"] = round(flops / (results[name]["ms_per_block"] / 1e3) / 1e12, 3)
    results.update(
        rows_per_block=args.rows,
        features=args.features,
        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
    )
    return results


def reference_ridge(revision):
    """Cargar `ridge.py` e `inputs.py` de una revisión como paquete aislado."""
    root = Path(tempfile.mkdtemp(prefix="ridge-reference-"))
    try:
        package = root / "reference_baselines"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in ("ridge", "inputs"):
            source = subprocess.check_output(
                ["git", "show", f"{revision}:src/mars_titan/models/baselines/{name}.py"],
                cwd=ROOT,
                text=True,
            )
            (package / f"{name}.py").write_text(source)
        sys.path.insert(0, str(root))
        try:
            return importlib.import_module("reference_baselines.ridge")
        finally:
            sys.path.remove(str(root))
    finally:
        shutil.rmtree(root)


def reference_statistics(reference, factory):
    """Primera pasada de `fit_ridge_blocks` en la referencia, sin resolver el sistema."""
    module = sys.modules[reference.__package__ + ".inputs"]
    count, mean, m2, target_mean = 0, None, None, 0.0
    for x, y in module.validated_blocks(factory):
        n = len(x)
        block_mean = x.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            block_m2 = np.square(x - block_mean).sum(axis=0)
        if mean is None:
            mean, m2 = block_mean, block_m2
        else:
            delta = block_mean - mean
            m2 += block_m2 + np.square(delta) * count * n / (count + n)
            mean += delta * n / (count + n)
        target_mean += (float(y.mean()) - target_mean) * n / (count + n)
        count += n
    variance = m2 / count
    epsilon = np.finfo(np.float64).eps
    constant = variance <= count * epsilon * variance + np.square(count * mean * epsilon)
    scale = np.sqrt(variance)
    scale[constant] = 1.0
    return count, mean, scale, target_mean


def run_ridge(args):
    import torch

    from mars_titan.models.baselines import ridge

    reference = reference_ridge(args.reference)
    module = sys.modules[reference.__package__ + ".inputs"]
    blocks, targets = [], []
    for matrix, batch in reader_blocks(args.manifest, "train", args.rows, args.batch_size):
        blocks.append(matrix)
        targets.append(batch["target"])

    def factory():
        # La ruta anterior recibía float64 del lector: la conversión es exacta.
        for x, y in zip(blocks, targets, strict=True):
            yield x.astype(np.float64), y

    def current():
        yield from zip(blocks, targets, strict=True)

    def timed(function):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, round(time.perf_counter() - start, 3)

    count, mean, scale, target_mean = reference_statistics(reference, factory)
    (old_gram, old_rhs, old_sum, _), old_seconds = timed(
        lambda: reference.centered_normal_equations(
            module.validated_blocks(factory), mean, scale, target_mean, count, device="cuda:0"
        )
    )
    statistics_, both_seconds = timed(lambda: ridge.ridge_statistics(current))
    rng = np.random.default_rng(1)
    unfitted = dict(
        mean=mean, scale=scale, coefficient=rng.normal(size=len(mean)) * 1e-3, intercept=0.3
    )
    old_model, new_model = reference.RidgeModel(**unfitted), ridge.RidgeModel(**unfitted)
    old_predict = new_predict = 0.0
    predictions_equal = True
    for x in blocks[: args.predict_blocks]:
        a, seconds_a = timed(lambda x=x: old_model.predict(x.astype(np.float64)))
        b, seconds_b = timed(lambda x=x: new_model.predict(x))
        old_predict += seconds_a
        new_predict += seconds_b
        predictions_equal &= bool(np.array_equal(a, b))

    def same(left, right):
        left = left.cpu().numpy() if hasattr(left, "cpu") else np.asarray(left)
        right = right.cpu().numpy() if hasattr(right, "cpu") else np.asarray(right)
        return bool(np.array_equal(left, right))

    rows = sum(len(block) for block in blocks)
    return dict(
        manifest=str(args.manifest),
        reference=args.reference,
        rows=rows,
        features=int(blocks[0].shape[1]),
        reference_gram_pass_seconds=old_seconds,
        current_both_passes_seconds=both_seconds,
        gram_equal=same(statistics_.gram, old_gram),
        rhs_equal=same(statistics_.rhs, old_rhs),
        sum_x_equal=same(statistics_.sum_x, old_sum),
        mean_scale_equal=same(statistics_.mean, mean) and same(statistics_.scale, scale),
        target_mean_equal=statistics_.target_mean == target_mean,
        count_equal=statistics_.count == count,
        constant_columns=int((scale == 1.0).sum()),
        predictions_equal=predictions_equal,
        predict_blocks=min(args.predict_blocks, len(blocks)),
        reference_predict_seconds=round(old_predict, 3),
        current_predict_seconds=round(new_predict, 3),
        peak_vram_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        peak_rss_mib=peak_rss_mib(),
    )


def synthetic_blocks(rows, features, block_rows, *, distinct=8, seed=20261009):
    """Pocos bloques distintos reciclados: se mide la biblioteca, no el generador."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(distinct):
        x = rng.standard_normal((block_rows, features), dtype=np.float32)
        x[rng.random(x.shape) < 0.3] = 0.0
        x[:, -5:] = rng.integers(0, 2, size=(block_rows, 5))
        pool.append(x)
    target = rng.standard_normal(block_rows)

    def factory():
        for index, start in enumerate(range(0, rows, block_rows)):
            size = min(block_rows, rows - start)
            yield pool[index % distinct][:size], target[:size]

    return factory


def digest_directory(directory):
    digest, size = hashlib.sha256(), 0
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            size += path.stat().st_size
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 24), b""):
                    digest.update(chunk)
    return digest.hexdigest(), size


def run_matrix(args):
    """Construir la matriz cuantizada con filas reales o aleatorias, sin ningún árbol."""
    from mars_titan.models.baselines.external_boosting import build_external_matrix

    if args.manifest is not None:
        # Filas reales en RAM: cada construcción recorre dos veces los mismos bloques.
        pairs = [
            (matrix, batch["target"])
            for matrix, batch in reader_blocks(args.manifest, "train", args.rows, 1024)
        ]
        rows, features = sum(len(y) for _, y in pairs), pairs[0][0].shape[1]

        def factory():
            yield from pairs
    else:
        rows, features = args.rows, args.features
        factory = synthetic_blocks(rows, features, args.block_rows)
    builds = []
    with tempfile.TemporaryDirectory(prefix="matrix-", dir=args.cache_root) as root:
        for max_bin in args.max_bin:
            for repeat in range(args.repeat):
                directory = Path(root) / f"pages-{max_bin}-{repeat}"
                with Peaks(directory) as peak:
                    started = time.perf_counter()
                    matrix = build_external_matrix(
                        factory,
                        directory,
                        expected_rows=rows,
                        max_bin=max_bin,
                        max_batch_bytes=args.max_batch_bytes,
                        on_host=False,
                        max_disk_cache_bytes=args.max_disk_cache_bytes,
                    )
                    elapsed = time.perf_counter() - started
                try:
                    indptr, values = matrix.data.get_quantile_cut()
                    pages, disk = digest_directory(directory)
                    builds.append(
                        dict(
                            max_bin=max_bin,
                            construction_seconds=round(elapsed, 2),
                            rows_per_second=round(rows / elapsed),
                            passes=list(matrix.iterator.completed),
                            disk_bytes=disk,
                            bits_per_value=round(8 * disk / (rows * features), 4),
                            cuts_sha256=hashlib.sha256(
                                np.asarray(indptr).tobytes() + np.asarray(values).tobytes()
                            ).hexdigest(),
                            pages_sha256=pages,
                            **peak.summary(),
                        )
                    )
                finally:
                    matrix.close()
                builds[-1]["left_after_close"] = directory.exists()
        left = sum(1 for _ in Path(root).rglob("*"))
    identical = {
        max_bin: len(
            {(b["cuts_sha256"], b["pages_sha256"]) for b in builds if b["max_bin"] == max_bin}
        )
        == 1
        for max_bin in args.max_bin
    }
    return dict(
        manifest=None if args.manifest is None else str(args.manifest),
        rows=rows,
        features=features,
        max_batch_bytes=args.max_batch_bytes,
        builds=builds,
        identical_repeats=identical,
        entries_left_in_cache_root=left,
        peak_rss_mib=peak_rss_mib(),
    )


def quantile_cuts(x, max_bin):
    """Cortes por cuantiles en el formato `(indptr, values)` de XGBoost, solo para umbrales."""
    indptr, values = [0], []
    quantiles = np.linspace(0, 1, max_bin + 1)[1:-1]
    for column in x.T:
        cuts = np.unique(np.quantile(column, quantiles).astype(np.float32))
        values.extend(cuts)
        values.append(np.float32(column.max() + 1))
        indptr.append(len(values))
    return np.asarray(indptr), np.asarray(values, dtype=np.float32)


def run_validation(args):
    """Evaluación de validación por ronda: npz en disco (anterior) frente a RAM y GPU.

    La ruta anterior y la completa predicen con los `trees` árboles. La incremental
    recorre las rondas 1..`trees` con un mismo booster que gana un árbol por ronda, como
    en `xgb.train`, y se compara con la completa en varias rondas.
    """
    import cupy as cp
    import xgboost as xgb

    from mars_titan.evaluation.session_metrics import SessionErrors
    from mars_titan.models.baselines.boosting_selection import (
        ResidentValidation,
        session_validation,
    )
    from mars_titan.models.baselines.external_boosting import ExternalBoostingModel

    sys.path.insert(0, str(ROOT))
    from tests.models.hand_forest import hand_forest

    rng = np.random.default_rng(7)
    factory = synthetic_blocks(args.rows, args.features, args.batch_size, distinct=16)
    markets = np.where(rng.random(args.batch_size) < 0.3, "CN", "US")

    def blocks():
        for index, (x, y) in enumerate(factory()):
            moments = (
                np.full(len(y), 1_300_000_000 + index // 8, dtype=np.int64) * 1_000_000
            ).astype("datetime64[us]")
            yield x, y, markets[: len(y)], moments

    sample = next(factory())[0]
    forest = hand_forest(
        quantile_cuts(sample, 64), args.features, trees=args.trees, depth=args.depth, seed=5
    )

    def wrap(booster):
        rounds = booster.num_boosted_rounds()
        return ExternalBoostingModel(booster, args.rows, args.features, dict(rounds=rounds))

    model = wrap(forest)

    def timed(function):
        cp.cuda.Device(0).synchronize()
        start = time.perf_counter()
        value = function()
        cp.cuda.Device(0).synchronize()
        return value, time.perf_counter() - start

    def summary(times):
        return dict(
            median_seconds=round(statistics.median(times), 4),
            max_seconds=round(max(times), 4),
            rows_per_second=round(args.rows / statistics.median(times)),
        )

    def repeated(function):
        values, times = zip(*(timed(function) for _ in range(args.repeat)), strict=True)
        return values[0], summary(times)

    with tempfile.TemporaryDirectory(prefix="validation-", dir=args.cache_root) as root:
        # Ruta anterior: un npz por bloque, leído de disco (o de la caché de páginas) cada ronda.
        paths = []
        for index, (x, y, market, moment) in enumerate(blocks()):
            path = Path(root) / f"{index:06}.npz"
            np.savez(path, values=x, target=y, markets=market, moments=moment)
            paths.append(path)
        disk = sum(path.stat().st_size for path in paths)

        def from_disk():
            sessions = SessionErrors()
            for path in paths:
                with np.load(path, allow_pickle=False) as block:
                    x, y, market, moment = (
                        block[key] for key in ("values", "target", "markets", "moments")
                    )
                sessions.update(market, moment, model.predict(x) - y)
            return sessions.summary()["session_mae"]

        reference, previous = repeated(from_disk)
    resident = ResidentValidation(blocks, expected_rows=args.rows, max_bytes=32 * 1024**3)
    results = dict(previous_npz=previous)
    for name, budget in (("ram", 0), ("device", args.device_bytes)):
        placed = resident.place(budget)

        def full():
            # Sin estado incremental, cada ronda predice con todos los árboles.
            saved, resident.incremental = resident.incremental, False
            try:
                return resident.evaluate(model)
            finally:
                resident.incremental = saved

        value, complete = repeated(full)
        growing, times, checked, equal = xgb.Booster(), [], 0, value == reference
        for count in range(1, args.trees + 1):
            growing.load_model(forest[:count].save_raw("json"))
            growing.set_param(dict(device="cuda:0", nthread=4))
            score, seconds = timed(lambda booster=growing: resident.evaluate(wrap(booster)))
            times.append(seconds)
            if count % max(1, args.trees // 8) == 0 or count == args.trees:
                checked += 1
                plain = lambda: iter(resident.blocks)  # noqa: E731  Ruta por bloques en RAM.
                equal &= score == session_validation(wrap(growing), plain, expected_rows=args.rows)
        tail = times[-max(1, args.trees // 10) :]
        results[name] = dict(
            device_bytes=placed,
            full_round=complete,
            incremental_restart_seconds=round(times[0], 4),
            incremental_last_rounds=summary(tail),
            incremental_all_rounds_seconds=round(sum(times), 2),
            leaf_sums=resident.incremental,
            checked_rounds=checked,
            identical=bool(equal),
        )
    resident.release()
    return dict(
        rows=args.rows,
        features=args.features,
        trees=args.trees,
        depth=args.depth,
        batch_size=args.batch_size,
        repeat=args.repeat,
        npz_disk_bytes=disk,
        resident_ram_bytes=resident.bytes,
        **results,
        peak_rss_mib=peak_rss_mib(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("read")
    read.add_argument("--manifest", type=Path, required=True)
    read.add_argument("--partition", default="train")
    read.add_argument("--rows", type=int, default=100_000)
    read.add_argument("--batch-size", type=int, default=1024)
    gram = commands.add_parser("gram")
    gram.add_argument("--rows", type=int, default=1024)
    gram.add_argument("--features", type=int, default=1719)
    gram.add_argument("--blocks", type=int, default=64)
    ridge = commands.add_parser("ridge")
    ridge.add_argument("--manifest", type=Path, required=True)
    ridge.add_argument("--reference", required=True)
    ridge.add_argument("--rows", type=int, default=100_000)
    ridge.add_argument("--batch-size", type=int, default=1024)
    ridge.add_argument("--predict-blocks", type=int, default=40)
    matrix = commands.add_parser("matrix")
    matrix.add_argument("--rows", type=int, default=200_000)
    matrix.add_argument("--features", type=int, default=1719)
    matrix.add_argument("--block-rows", type=int, default=1024)
    matrix.add_argument("--manifest", type=Path)
    matrix.add_argument("--max-bin", type=int, nargs="+", default=[64, 128, 256])
    matrix.add_argument("--max-batch-bytes", type=int, default=64 * 1024**2)
    matrix.add_argument("--max-disk-cache-bytes", type=int, default=32 * 1024**3)
    matrix.add_argument("--repeat", type=int, default=1)
    matrix.add_argument("--cache-root", type=Path, required=True)
    validation = commands.add_parser("validation")
    validation.add_argument("--rows", type=int, default=100_000)
    validation.add_argument("--features", type=int, default=1719)
    validation.add_argument("--batch-size", type=int, default=1024)
    validation.add_argument("--trees", type=int, default=200)
    validation.add_argument("--depth", type=int, default=6)
    validation.add_argument("--repeat", type=int, default=5)
    validation.add_argument("--device-bytes", type=int, default=2 * 1024**3)
    validation.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "cache_root", None) is not None:
        args.cache_root.mkdir(parents=True, exist_ok=True)
    run = dict(
        read=run_read, gram=run_gram, ridge=run_ridge, matrix=run_matrix, validation=run_validation
    )[args.command]
    result = dict(command=args.command, environment=environment(), result=run(args))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
