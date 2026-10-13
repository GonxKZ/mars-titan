"""Codificación episódica con la proyección de einsum y con el núcleo nativo del mismo orden.

Mide `FrozenEpisodeCodec.encode` y, por separado, la proyección de sus seis bloques, con las
anchuras de la vista de la campaña (precios 5 × 64, noticias 384, gráficos 512, fundamentales
78 y macro 420) y entradas sintéticas normales con todas las modalidades presentes. Exige los
mismos bits en las seis salidas, la misma huella del resultado y la misma memoria declarada.
El enlace se carga desde `MARS_TITAN_EPISODIC_NATIVE` y debe incluir `project_codec_block`.
Sin banco, sin modelo y sin GPU.

Uso: `MARS_TITAN_EPISODIC_NATIVE=... PYTHONPATH=src python benchmarks/episodic_codec_projection.py
salida.json`
"""

import json
import os
import platform
import statistics
import sys
import time

import numpy as np
from numpy._core import _multiarray_umath

from mars_titan.data.input_policy import STRICT_INPUTS
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory import episodic_codec
from mars_titan.memory.episodic_codec import FrozenEpisodeCodec, projection_backend
from mars_titan.memory.native_backend import load_native
from mars_titan.models.titans.financial_inputs import FinancialInputSpec, validated_cpu_batch

DIMENSIONS = dict(prices=5, news=384, charts=512, fundamentals=78, macro=420)
CONTEXT = 64
ROWS = (1, 16, 128, 512)
REPETITIONS = 7


def specification():
    return FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        input_policy=STRICT_INPUTS,
        dimensions=DIMENSIONS,
        representation=dict(
            fundamental_concepts=["assets"],
            macro_indicators=["rate"],
            encoders={"benchmark": "synthetic"},
            representation_code={"benchmark.py": "c" * 64},
            text_aggregation="mean",
            context_sessions=CONTEXT,
            news_lookback_sessions=5,
        ),
    )


def view(spec, rows, seed=520):
    random = np.random.default_rng(seed)
    inputs = {
        name: random.normal(
            size=(rows, CONTEXT, width) if name == "prices" else (rows, width)
        ).astype(np.float32)
        for name, width in DIMENSIONS.items()
    }
    at = 1_609_459_200_000_000
    return validated_cpu_batch(
        dict(
            inputs=inputs,
            sample_ids=[f"US/A{index:05d}/{at}" for index in range(rows)],
            prediction_at=np.full(rows, at, dtype="datetime64[us]"),
            input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
        ),
        spec,
    )


def same_bits(left, right):
    arrays = (
        "key_inputs",
        "values",
        "input_l2",
        "normalized_l2",
        "key_contribution_l2",
        "value_contribution_l2",
    )
    return (
        all(getattr(left, name).tobytes() == getattr(right, name).tobytes() for name in arrays)
        and left._digest == right._digest
        and left.estimated_peak_bytes == right.estimated_peak_bytes
    )


def alternated(variants, repetitions):
    seconds = {name: [] for name in variants}
    for repetition in range(repetitions):
        # Se alterna el orden para repartir entre las variantes la deriva de la máquina.
        order = list(variants) if repetition % 2 == 0 else list(reversed(variants))
        for name in order:
            began = time.perf_counter()
            variants[name]()
            seconds[name].append(time.perf_counter() - began)
    return seconds


def measure(codec, spec, native, rows):
    batch = view(spec, rows)
    reference = codec.encode(batch)
    accelerated = codec.encode(batch, native=native)
    encode = alternated(
        dict(
            numpy_einsum=lambda: codec.encode(batch),
            native=lambda: codec.encode(batch, native=native),
        ),
        REPETITIONS,
    )
    # La proyección sola, con bloques de las mismas formas. Su tiempo no depende de los valores.
    random = np.random.default_rng(rows)
    parts, start = [], 0
    for width in (*codec._input_widths, 5):
        end = start + width
        block = random.normal(size=(rows, width))
        parts.append((block, codec._projection[:, start:end], codec._transposed[start:end]))
        start = end
    projection = alternated(
        dict(
            numpy_einsum=lambda: [
                np.einsum("bi,oi->bo", block, weights, optimize=False)
                for block, weights, _ in parts
            ],
            native=lambda: [
                native.project_codec_block(block, transposed) for block, _, transposed in parts
            ],
        ),
        REPETITIONS,
    )
    medians = {
        part: {name: statistics.median(values) for name, values in seconds.items()}
        for part, seconds in (("encode", encode), ("projection", projection))
    }
    return dict(
        rows=rows,
        same_bits_digest_and_memory=same_bits(reference, accelerated),
        estimated_peak_bytes=accelerated.estimated_peak_bytes,
        seconds=dict(encode=encode, projection=projection),
        median_seconds=medians,
        speedup={
            part: values["numpy_einsum"] / values["native"] for part, values in medians.items()
        },
    )


def main(output):
    native = load_native()
    if projection_backend(native) != "native":
        raise SystemExit("El enlace no incluye la proyección o einsum suma en otro orden")
    spec = specification()
    codec = FrozenEpisodeCodec(spec)
    features = _multiarray_umath.__cpu_features__
    report = dict(
        widths=dict(zip(episodic_codec.BLOCKS, (*codec._input_widths, 5), strict=True)),
        repetitions=REPETITIONS,
        warmup_calls_per_variant=1,
        measurements=[measure(codec, spec, native, rows) for rows in ROWS],
        native_sha256=native.binary_sha256,
        native_isa=native.codec_projection_isa,
        cpu=dict(
            model=cpu_name(),
            machine=platform.machine(),
            avx2=features.get("AVX2"),
            avx512f=features.get("AVX512F"),
            numpy_baseline=list(_multiarray_umath.__cpu_baseline__),
        ),
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        load_average=list(os.getloadavg()),
        limits=[
            "Entradas sintéticas con las anchuras de la vista. Los valores no cambian el tiempo.",
            "La medida del recorrido completo de los lectores está en el informe.",
            "Otras aplicaciones siguieron activas y no se mide energía.",
        ],
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not all(row["same_bits_digest_and_memory"] for row in report["measurements"]):
        raise SystemExit("El núcleo nativo cambió la codificación")


if __name__ == "__main__":
    main(sys.argv[1])
