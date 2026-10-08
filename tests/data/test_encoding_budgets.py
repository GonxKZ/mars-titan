"""Límites de preparación sin recortar población ni ocultar valores corruptos."""

import json
import shutil
import sqlite3
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mars_titan.data import corpus_encoding, embeddings
from mars_titan.data.storage import sha256
from tests.data.test_cohort_samples import Encoders
from tests.data.test_corpus_encoding import prepared_edition


def test_optional_chart_cache_keeps_news_and_validates_uncached_values(tmp_path):
    path = tmp_path / "cache.sqlite"
    cache = embeddings.EmbeddingCache(path, cache_charts=False)
    chart = dict(kind="chart", encoder="e", content="c")
    news = dict(kind="news", encoder="e", content="n")
    try:
        cache.put(chart, np.ones(512))
        cache.put(news, np.ones(384))
        assert cache.get(chart) is None
        np.testing.assert_array_equal(cache.get(news), np.ones(384))
        with pytest.raises(ValueError):
            cache.put(chart, np.array([np.nan]))
        with pytest.raises(ValueError):
            cache.get(chart, max_bytes=0)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    finally:
        cache.close()
    with pytest.raises(ValueError):
        embeddings.EmbeddingCache(tmp_path / "invalid.sqlite", cache_charts=1)
    assert not (tmp_path / "invalid.sqlite").exists()


def _two_assets(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    data = json.loads(manifest.read_text())
    source = Path(data["prepared_root"]) / "US" / "A"
    destination = source.with_name("C")
    shutil.copytree(source, destination)
    receipt = destination / "manifest.json"
    metadata = json.loads(receipt.read_text())
    metadata["symbol"] = "C"
    receipt.write_text(json.dumps(metadata))
    data["assets"].append(
        dict(market="US", symbol="C", state="prepared", manifest_sha256=sha256(receipt))
    )
    data["candidate_count"] += 1
    manifest.write_text(json.dumps(data))
    return manifest, clock, macro


def test_pause_resumes_new_assets_and_never_publishes_partial_census(tmp_path):
    manifest, clock, macro = _two_assets(tmp_path)
    encoders = Encoders()
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, encoders=encoders, context=2)
    output = tmp_path / "encoded"
    first = corpus_encoding.encode_corpus(
        manifest, output, **kwargs, max_new_assets=1, cache_charts=False
    )
    assert first["cohort_complete"] is False
    assert first["stop_reason"] == "new_asset_limit"
    assert not (output / "manifest.json").exists()
    assert first["samples"] == 5
    second = corpus_encoding.encode_corpus(
        manifest, output, **kwargs, max_new_assets=1, cache_charts=False
    )
    assert second["cohort_complete"] is True
    assert second["reused_assets"] == 1
    assert second["samples"] == 10
    assert len(second["coverage"]) == 3
    before = encoders.calls
    digest = sha256(output / "manifest.json")
    third = corpus_encoding.encode_corpus(manifest, output, **kwargs, cache_charts=False)
    assert third["reused_assets"] == 2
    assert encoders.calls == before
    assert sha256(output / "manifest.json") == digest


def test_low_disk_is_rejected_before_encoder_loading(tmp_path, monkeypatch):
    manifest, clock, macro = prepared_edition(tmp_path)
    monkeypatch.setattr(
        corpus_encoding, "FrozenEncoders", lambda: pytest.fail("Se cargó un modelo")
    )
    with pytest.raises(OSError, match="disco"):
        corpus_encoding.encode_corpus(
            manifest,
            tmp_path / "encoded",
            macros={"US": macro},
            clocks={"US": clock},
            context=2,
            min_free_disk_bytes=2**63,
        )


def test_encoder_options_reach_factory_and_cannot_override_supplied_encoder(tmp_path, monkeypatch):
    manifest, clock, macro = prepared_edition(tmp_path)
    options = dict(cuda_memory_bytes=1024**3, text_batch_size=2, image_batch_size=2)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return Encoders()

    monkeypatch.setattr(corpus_encoding, "FrozenEncoders", factory)
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2)
    result = corpus_encoding.encode_corpus(
        manifest, tmp_path / "encoded", **kwargs, encoder_options=options
    )
    assert result["cohort_complete"] is True
    assert calls == [options]
    with pytest.raises(ValueError):
        corpus_encoding.encode_corpus(
            manifest,
            tmp_path / "invalid",
            **kwargs,
            encoders=Encoders(),
            encoder_options=options,
        )
    assert not (tmp_path / "invalid").exists()


def test_disk_pause_recovers_and_chart_cache_does_not_change_parquet(tmp_path, monkeypatch):
    manifest, clock, macro = _two_assets(tmp_path)
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2)
    space = iter([1024, 1024, 0])
    monkeypatch.setattr(corpus_encoding, "_free_disk_bytes", lambda _: next(space))
    paused = corpus_encoding.encode_corpus(
        manifest,
        tmp_path / "bounded",
        **kwargs,
        encoders=Encoders(),
        cache_charts=False,
        min_free_disk_bytes=512,
    )
    assert paused["stop_reason"] == "disk_reserve"
    assert not (tmp_path / "bounded/manifest.json").exists()
    resumed = corpus_encoding.encode_corpus(
        manifest,
        tmp_path / "bounded",
        **kwargs,
        encoders=Encoders(),
        cache_charts=False,
    )
    original = corpus_encoding.encode_corpus(
        manifest,
        tmp_path / "default",
        **kwargs,
        encoders=Encoders(),
    )
    assert resumed["samples"] == original["samples"] == 10
    for name in ("A", "C"):
        relative = f"samples/US/{name}/samples.parquet"
        assert sha256(tmp_path / "bounded" / relative) == sha256(tmp_path / "default" / relative)


@pytest.mark.parametrize("filename", ["configuration.json", "manifest.json"])
def test_recovery_rejects_integer_replacing_boolean_cache_policy(tmp_path, filename):
    manifest, clock, macro = prepared_edition(tmp_path)
    output = tmp_path / "encoded"
    options = dict(
        macros={"US": macro},
        clocks={"US": clock},
        context=2,
        encoders=Encoders(),
        cache_charts=False,
    )
    corpus_encoding.encode_corpus(manifest, output, **options)
    path = output / filename
    value = json.loads(path.read_text())
    configuration = value if filename == "configuration.json" else value["configuration"]
    assert configuration["cache_charts"] is False
    configuration["cache_charts"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="configuración|manifiesto"):
        corpus_encoding.encode_corpus(manifest, output, **options)


def test_image_subbatches_keep_order_tail_and_no_gradient():
    encoder = embeddings.FrozenEncoders.__new__(embeddings.FrozenEncoders)
    encoder.device = torch.device("cpu")
    encoder.image_batch_size = 2
    encoder.mean = torch.zeros(1, 3, 1, 1)
    encoder.std = torch.ones(1, 3, 1, 1)
    batches = []

    def model(value):
        assert not torch.is_grad_enabled()
        batches.append(len(value))
        return value[:, 0, 0, 0][:, None].expand(-1, 512)

    encoder.image_model = model
    pngs = []
    for value in (0, 25, 50, 100, 255):
        buffer = BytesIO()
        Image.new("RGB", (224, 224), (value, value, value)).save(buffer, format="PNG")
        pngs.append(buffer.getvalue())
    actual = encoder.images(pngs)
    assert actual.shape == (5, 512)
    assert batches == [2, 2, 1]
    np.testing.assert_allclose(actual[:, 0], np.array([0, 25, 50, 100, 255]) / 255, atol=1e-7)


def test_text_subbatches_keep_all_tokens_and_weighted_mean():
    encoder = embeddings.FrozenEncoders.__new__(embeddings.FrozenEncoders)
    encoder.device = torch.device("cpu")
    tokens = list(range(1, 884))
    batches = []

    class Tokenizer:
        cls_token_id, sep_token_id = 1000, 1001

        def __call__(self, text, **kwargs):
            return {"input_ids": tokens}

        def pad(self, examples, **kwargs):
            width = max(len(row["input_ids"]) for row in examples)
            ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in examples]
            array = torch.tensor(ids)
            return {"input_ids": array, "attention_mask": array != 0}

    def model(input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        batches.append(len(input_ids))
        return SimpleNamespace(last_hidden_state=input_ids.float()[..., None].expand(-1, -1, 384))

    encoder.tokenizer, encoder.text_model = Tokenizer(), model
    encoder.text_batch_size = 32
    reference = encoder.text("texto técnico de prueba")
    batches.clear()
    encoder.text_batch_size = 3
    actual = encoder.text("texto técnico de prueba")
    assert batches == [3, 3, 2]
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    "options",
    [
        dict(max_new_assets=0),
        dict(max_new_assets=True),
        dict(min_free_disk_bytes=-1),
        dict(min_free_disk_bytes=True),
        dict(cache_charts=0),
    ],
)
def test_invalid_execution_budget_is_rejected_before_output(tmp_path, options):
    manifest, clock, macro = prepared_edition(tmp_path)
    with pytest.raises(ValueError):
        corpus_encoding.encode_corpus(
            manifest,
            tmp_path / "encoded",
            macros={"US": macro},
            clocks={"US": clock},
            context=2,
            encoders=Encoders(),
            **options,
        )
    assert not (tmp_path / "encoded").exists()


def test_cuda_budget_checks_free_memory_and_never_uses_cpu(monkeypatch):
    selected, fractions = [], []
    monkeypatch.setattr(embeddings.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=8 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (2 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda *v: fractions.append(v)
    )
    assert embeddings.require_cuda(
        max_bytes=1024**3, min_free_bytes=1536 * 1024**2
    ) == torch.device("cuda:0")
    assert fractions == [(0.125, 0)]
    assert selected == ["cuda:0"]
    with pytest.raises(RuntimeError, match="libre"):
        embeddings.require_cuda(max_bytes=1024**3, min_free_bytes=3 * 1024**3)
    assert len(fractions) == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        embeddings.require_cuda(max_bytes=1024**3)


@pytest.mark.parametrize(
    "options", [dict(max_bytes=0), dict(max_bytes=True), dict(min_free_bytes=-1)]
)
def test_invalid_cuda_budget_fails_before_device_access(monkeypatch, options):
    monkeypatch.setattr(
        embeddings.subprocess, "run", lambda *a, **k: pytest.fail("Se consultó CUDA")
    )
    with pytest.raises(ValueError):
        embeddings.require_cuda(**options)


@pytest.mark.parametrize(
    "options",
    [
        dict(text_batch_size=0),
        dict(text_batch_size=True),
        dict(image_batch_size=65),
        dict(cuda_memory_bytes=0),
        dict(min_free_cuda_bytes=-1),
    ],
)
def test_encoder_budgets_fail_before_loading_weights(monkeypatch, options):
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: pytest.fail("Se abrió CUDA"))
    with pytest.raises(ValueError):
        embeddings.FrozenEncoders(**options)
