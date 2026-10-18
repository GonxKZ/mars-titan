"""Coste por etapas de derivar velas sobre un corpus técnico acotado, sin aprendizaje."""

import argparse
import platform
import resource
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.data.candle_geometry_edition import derive_geometry_edition
from mars_titan.data.cohort_preparation import prepare_cohort_asset
from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.pretraining_free_encoders import PretrainingFreeEncoders
from mars_titan.data.price_windows import calendar_digest, price_window_contract
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock
from mars_titan.hardware.platform_identity import cpu_name


def _fixture(root, samples):
    clock = MarketClock("US", "2000-01-01", "2004-12-31")
    raw = root / "raw"
    raw.mkdir()
    days = clock.days[: samples + 63]
    prices = raw / "prices.csv"
    prices.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        + "".join(
            f"{day},{100 + i},{103 + i},{98 + i},{101 + i},1000\n"
            for i, day in enumerate(days)
            if i != 10
        )
    )
    asset = dict(
        market="US",
        symbol="FIXTURE",
        hashes={"prices.csv": sha256(prices)},
        paths=dict(prices=["prices.csv"], news=[], fundamentals=[], charts=[]),
    )
    prepare_cohort_asset(
        raw,
        root / "prepared",
        asset,
        clock,
        cohort="original_audited",
        reviews={},
        input_policy=HISTORICAL_MASKED,
    )
    preparation = root / "preparation.json"
    atomic_json(
        preparation,
        dict(
            **policy_identity(HISTORICAL_MASKED),
            schema_version=2,
            kind="prepared_cohort",
            status="completed",
            cohort_id="original_audited",
            candidate_count=1,
            failed_assets=0,
            prepared_root=str(root / "prepared"),
            assets=[
                dict(
                    market="US",
                    symbol="FIXTURE",
                    state="prepared",
                    manifest_sha256=sha256(root / "prepared/US/FIXTURE/manifest.json"),
                )
            ],
        ),
    )
    contract = price_window_contract(
        {
            "US": dict(
                start=clock.days[0].isoformat(),
                end=clock.days[-1].isoformat(),
                decisions_sha256=calendar_digest(clock),
            )
        },
        {"US": [clock.days[10].isoformat()]},
    )
    source = root / "source"
    encode_corpus(
        preparation,
        source,
        macros={},
        macro_indicators=[f"fixture_{i}" for i in range(140)],
        encoders=PretrainingFreeEncoders(),
        clocks={"US": clock},
        company_factors=False,
        fundamental_concepts=[f"fixture:concept_{i}:USD" for i in range(26)],
        input_policy=HISTORICAL_MASKED,
        price_window=contract,
    )
    return source, clock


def measure_derivation(directory, *, samples=128, repeats=5, batch_rows=128):
    """Una pasada de calentamiento y hasta nueve derivaciones en destinos nuevos temporales."""
    if any(
        type(v) is not int or not 1 <= v <= limit
        for v, limit in ((samples, 512), (repeats, 9), (batch_rows, 512))
    ):
        raise ValueError("La medición supera su presupuesto de muestras, repeticiones o lote")
    began = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geometry-derivation-", dir=directory) as temporary:
        root = Path(temporary)
        source, clock = _fixture(root, samples)
        fixture_seconds = time.perf_counter() - began
        source_hash = sha256(source / "samples/US/FIXTURE/samples.parquet")
        runs = []
        for index in range(repeats + 1):
            started = time.perf_counter()
            result = derive_geometry_edition(
                source,
                root / f"derived-{index}",
                batch_rows=batch_rows,
                clocks={"US": clock},
                reserve_bytes=64 * 1024**2,
            )
            elapsed = time.perf_counter() - started
            report = result["chart_derivation"]
            if index:
                runs.append(
                    dict(
                        samples=result["samples"],
                        total_seconds=elapsed,
                        timing_seconds=report["timing_seconds"],
                        unique_input_file_bytes=report["unique_input_file_bytes"],
                        samples_output_bytes=report["samples_output_bytes"],
                    )
                )
        if sha256(source / "samples/US/FIXTURE/samples.parquet") != source_hash:
            raise ValueError("El benchmark ha modificado el origen")
    durations = [run["total_seconds"] for run in runs]
    return dict(
        measured_at=datetime.now(UTC).isoformat(),
        device="cpu",
        learned_models_loaded=0,
        benchmark_code_sha256=sha256(Path(__file__)),
        derivation_code=report["source"]["code"],
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        versions=dict(
            python=platform.python_version(), numpy=np.__version__, pyarrow=pa.__version__
        ),
        fixture=dict(
            kind="synthetic_technical_fixture",
            samples=samples,
            missing_calendar_slot=10,
            macro_indicators=140,
            fundamental_concepts=26,
            optional_modalities="absent",
            source_samples_sha256=source_hash,
            preparation_seconds=fixture_seconds,
        ),
        warmup=1,
        batch_rows=batch_rows,
        runs=runs,
        total_seconds=dict(
            mean=float(np.mean(durations)),
            p50=float(np.percentile(durations, 50)),
            p95=float(np.percentile(durations, 95)),
            p99=float(np.percentile(durations, 99)),
        ),
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        process_seconds=time.perf_counter() - began,
        limits=[
            "No usa corpus real ni mide inferencia o calidad predictiva.",
            "Las modalidades opcionales ausentes comprimen más que vectores reales.",
            "La RAM incluye la creación del fixture y el proceso completo.",
            "No mide energía, VRAM, coste monetario ni una aceleración equivalente.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-rows", type=int, default=128)
    args = parser.parse_args(argv)
    report = measure_derivation(
        args.output.parent, samples=args.samples, repeats=args.repeats, batch_rows=args.batch_rows
    )
    atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
