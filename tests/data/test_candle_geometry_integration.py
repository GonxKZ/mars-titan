"""Una edición adicional conserva filas y entradas al cambiar solo las velas."""

import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import cohort_samples, corpus_encoding
from mars_titan.data.candle_geometry import MODEL, GeometryEncoders
from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.pretraining_free_encoders import PretrainingFreeEncoders
from mars_titan.data.storage import sha256
from mars_titan.data.vector_carry import CollectingEncoders
from tests.data.test_cohort_samples import Encoders
from tests.data.test_historical_input_masks import (
    CONCEPTS,
    HISTORICAL,
    prepared_with_gap,
    window_contract,
)


def prepared(tmp_path):
    source, clock = prepared_with_gap(tmp_path)
    report = json.loads((source / "manifest.json").read_text())
    path = tmp_path / "prepared-cohort.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version=2,
                kind="prepared_cohort",
                status="completed",
                cohort_id="original_audited",
                input_policy=HISTORICAL,
                mask_contract=report["mask_contract"],
                candidate_count=1,
                failed_assets=0,
                prepared_root=str(source.parents[1]),
                assets=[
                    dict(
                        market="US",
                        symbol="A",
                        state="prepared",
                        manifest_sha256=sha256(source / "manifest.json"),
                    )
                ],
            )
        )
    )
    options = dict(
        macros={},
        macro_indicators=["a", "b"],
        encoders=Encoders(),
        clocks={"US": clock},
        company_factors=False,
        fundamental_concepts=CONCEPTS,
        input_policy=HISTORICAL,
        price_window=window_contract(clock, [clock.days[10].isoformat()]),
    )
    return path, options


def test_materialization_keeps_rows_masks_and_other_inputs_without_rasterizing(
    tmp_path, monkeypatch
):
    source, clock = prepared_with_gap(tmp_path)
    contract = window_contract(clock, [clock.days[10].isoformat()])
    macro = MacroVectors(None, indicators=["a", "b"], input_policy=HISTORICAL)
    cache = EmbeddingCache(tmp_path / "vectors.sqlite")
    base = Encoders()
    options = dict(
        cohort="original_audited",
        company_factors=False,
        fundamental_concepts=CONCEPTS,
        input_policy=HISTORICAL,
        price_window=contract,
    )
    old, new = tmp_path / "old", tmp_path / "new"
    try:
        legacy = cohort_samples.materialize_cohort_asset(
            source, old, clock, macro, base, cache, **options
        )
        before = sha256(old / "samples.parquet")

        def no_rasterization(*args, **kwargs):
            raise AssertionError("La geometría no debe dibujar un PNG")

        monkeypatch.setattr(cohort_samples, "chart_png", no_rasterization)
        encoder = GeometryEncoders(base)
        result = cohort_samples.materialize_cohort_asset(
            source, new, clock, macro, encoder, cache, **options
        )
        assert result["samples"] == legacy["samples"] == 5
        assert result["excluded_reasons"] == legacy["excluded_reasons"]
        rows, baseline = [pq.read_table(p / "samples.parquet").to_pylist() for p in (new, old)]
        for row, original in zip(rows, baseline, strict=True):
            assert {k: v for k, v in row.items() if k not in {"charts", "chart_hash"}} == {
                k: v for k, v in original.items() if k not in {"charts", "chart_hash"}
            }
            vector = np.asarray(row["charts"]).reshape(64, 8)
            assert vector[:, 7].sum() == 63 and (vector[vector[:, 7] == 0] == 0).all()
            assert row["chart_hash"] != original["chart_hash"]
        assert sha256(old / "samples.parquet") == before
        assert result["encoders"]["image_model"] == MODEL
        repeated = cohort_samples.materialize_cohort_asset(
            source, new, clock, macro, encoder, cache, **options
        )
        assert repeated["reused"] is True
    finally:
        cache.close()


def test_corpus_option_is_explicit_and_cannot_overwrite_another_representation(tmp_path):
    source, options = prepared(tmp_path)
    old, new = tmp_path / "legacy", tmp_path / "geometry"
    default = corpus_encoding.encode_corpus(source, old, **options)
    old_hash = sha256(old / "samples/US/A/samples.parquet")
    result = corpus_encoding.encode_corpus(source, new, chart_representation=MODEL, **options)
    assert result["cohort_complete"] is True and result["samples"] == default["samples"] == 5
    assert result["training_ready"] is False and result["final_test_opened"] is False
    configuration = json.loads((new / "configuration.json").read_text())
    assert configuration["encoders"]["image_model"] == MODEL
    assert default["configuration"]["encoders"] == options["encoders"].spec
    from benchmarks.candle_geometry import main as compare

    assert (
        compare(
            [
                "compare",
                "--previous",
                str(old),
                "--current",
                str(new),
                "--output",
                str(tmp_path / "comparison"),
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="configuración|edición"):
        corpus_encoding.encode_corpus(source, old, chart_representation=MODEL, **options)
    assert sha256(old / "samples/US/A/samples.parquet") == old_hash
    with pytest.raises(ValueError, match="representación"):
        corpus_encoding.encode_corpus(
            source, tmp_path / "unknown", chart_representation="other", **options
        )
    with pytest.raises(ValueError, match="64"):
        corpus_encoding.encode_corpus(
            source, tmp_path / "short", chart_representation=MODEL, context=2, **options
        )
    assert not (tmp_path / "short").exists()


def test_raw_geometry_rejects_a_collector_that_only_retains_png(tmp_path):
    source, options = prepared(tmp_path)
    options["encoders"] = CollectingEncoders(PretrainingFreeEncoders().spec)
    with pytest.raises(ValueError, match="pasada|geométrica"):
        corpus_encoding.encode_corpus(
            source, tmp_path / "bad", chart_representation=MODEL, **options
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "change", ["macro", "price_end_index", "remove_row", "order", "preparation"]
)
def test_comparison_rejects_changes_outside_the_chart_representation(tmp_path, change):
    from benchmarks.candle_geometry import compare_representations

    source, options = prepared(tmp_path)
    old, new = tmp_path / "legacy", tmp_path / "geometry"
    corpus_encoding.encode_corpus(source, old, **options)
    corpus_encoding.encode_corpus(source, new, chart_representation=MODEL, **options)
    if change == "preparation":
        path = new / "configuration.json"
        document = json.loads(path.read_text())
        document["preparation_sha256"] = "a" * 64
        path.write_text(json.dumps(document))
    else:
        path = new / "samples/US/A/samples.parquet"
        table = pq.read_table(path)
        rows = table.to_pylist()
        if change == "remove_row":
            rows.pop()
        elif change == "order":
            rows.reverse()
        elif change == "macro":
            rows[0]["macro"][0] = 1.0
        else:
            rows[0]["price_end_index"] += 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        receipt = new / "samples/US/A/manifest.json"
        document = json.loads(receipt.read_text())
        document["samples_sha256"] = sha256(path)
        receipt.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="filas|preparación"):
        compare_representations(old, new, tmp_path / "comparison")
    assert not (tmp_path / "comparison/summary.json").exists()


def test_cli_selects_geometry_with_the_existing_text_encoder(monkeypatch):
    import torch

    captured = []
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(
        corpus_encoding,
        "encode_corpus",
        lambda *args, **kwargs: (
            captured.append(kwargs)
            or dict(
                cohort_id="fixture",
                candidate_count=1,
                samples=1,
                failed_assets=0,
                cohort_complete=True,
                reused_assets=0,
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encode",
            "--prepared",
            "p",
            "--output",
            "o",
            "--encoders",
            "pretraining_free",
            "--chart-representation",
            MODEL,
        ],
    )
    assert corpus_encoding.main() == 0
    assert captured[0]["chart_representation"] == MODEL
    assert isinstance(captured[0]["encoders"], PretrainingFreeEncoders)


@pytest.mark.parametrize(
    "mode",
    [
        ["--collect"],
        ["--reuse-only"],
        ["--encode-pending"],
        ["--release-vectors"],
        ["--text-carry", "previous"],
        ["--substitute-previous", "previous", "--substitution-records", "records"],
    ],
)
def test_geometry_cli_rejects_unsupported_passes_before_loading_models(monkeypatch, mode):
    monkeypatch.setattr(
        sys,
        "argv",
        ["encode", "--prepared", "p", "--output", "o", "--chart-representation", MODEL, *mode],
    )
    with pytest.raises(SystemExit) as failure:
        corpus_encoding.main()
    assert failure.value.code == 2
