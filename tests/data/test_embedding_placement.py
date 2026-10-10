"""Ubicación explícita de la tabla congelada, sin CUDA ni pesos externos."""

import importlib
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mars_titan.data import corpus_encoding, embeddings
from tests.data.test_cohort_samples import Encoders
from tests.data.test_corpus_encoding import prepared_edition


def placement():
    return importlib.import_module("mars_titan.data.embedding_placement")


def table():
    values = torch.arange(17 * 7, dtype=torch.float32, device="cpu").reshape(17, 7) / 8
    return torch.nn.Embedding.from_pretrained(values, freeze=True, padding_idx=0).eval()


def test_cpu_lookup_keeps_exact_rows_repeats_and_zero_padding_without_rng_or_graph():
    original = table()
    lookup = partial(placement().cpu_word_values, original, destination=torch.device("cpu"))
    ids = torch.tensor([[0, 16, 3, 3], [2, 0, 1, 8]], dtype=torch.int64, device="cpu")
    before = original.weight.clone()
    rng = torch.get_rng_state()
    with torch.inference_mode():
        expected = original(ids)
        actual = lookup(ids)
        actual[0, 0] = -3
        again = lookup(ids)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    assert torch.equal(original.weight, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert original.weight.device.type == "cpu"
    assert again.dtype == torch.float32 and again.device.type == "cpu"
    assert not again.requires_grad and again.grad_fn is None


@pytest.mark.parametrize("change", ["trainable", "training", "fp64", "meta", "max_norm"])
def test_cpu_lookup_rejects_incompatible_table_before_transfer(change):
    original = table()
    if change == "trainable":
        original.requires_grad_(True)
    elif change == "training":
        original.train()
    elif change == "fp64":
        original.double()
    elif change == "meta":
        original.to("meta")
    else:
        original.max_norm = 1
    with pytest.raises(ValueError, match="tabla"):
        placement().check_cpu_word_table(original)


def test_cpu_lookup_rejects_gradient_and_later_movement_or_training():
    original = table()
    lookup = partial(placement().cpu_word_values, original, destination=torch.device("cpu"))
    ids = torch.tensor([[1]], dtype=torch.int64, device="cpu")
    with pytest.raises(ValueError, match="inferencia"):
        lookup(ids)
    with torch.inference_mode():
        original.train()
        with pytest.raises(ValueError, match="tabla"):
            lookup(ids)
        original.eval().double()
        with pytest.raises(ValueError, match="tabla"):
            lookup(ids)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((33, 2), torch.int64),
        ((2, 129), torch.int64),
        ((0, 3), torch.int64),
        ((1, 0), torch.int64),
        ((2,), torch.int64),
        ((2, 3), torch.float32),
    ],
)
def test_cpu_lookup_bounds_queries_before_calling_embedding(shape, dtype):
    original = table()
    lookup = partial(placement().cpu_word_values, original, destination=torch.device("cpu"))
    ids = torch.zeros(shape, dtype=dtype, device="cpu")
    with torch.inference_mode(), pytest.raises(ValueError, match="consulta"):
        lookup(ids)


def test_cpu_lookup_accepts_maximum_query_and_rejects_bad_indices():
    original = table()
    lookup = partial(placement().cpu_word_values, original, destination=torch.device("cpu"))
    with torch.inference_mode():
        actual = lookup(torch.zeros((32, 128), dtype=torch.int64, device="cpu"))
        assert actual.shape == (32, 128, 7)
        for index in (-1, 17):
            with pytest.raises(IndexError):
                lookup(torch.tensor([[index]], device="cpu"))


def test_cpu_lookup_rejects_ids_outside_cpu_before_indexing():
    indices = torch.empty((1, 2), dtype=torch.int64, device="meta")
    with torch.inference_mode(), pytest.raises(ValueError, match="consulta"):
        placement().cpu_word_values(table(), indices, torch.device("cpu"))


def test_placement_excludes_table_before_moving_model_and_restores_on_failure():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = table()
            self.fail = False

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, value):
            self.embedding = value

        def to(self, device):
            assert isinstance(self.embedding, torch.nn.Identity)
            if self.fail:
                raise RuntimeError("Transferencia fallida")
            return super().to(device)

    model = Model().eval()
    original = model.embedding
    model.fail = True
    with pytest.raises(RuntimeError, match="Transferencia"):
        placement().place_cpu_word_embeddings(model, torch.device("cpu"))
    assert model.embedding is original
    model.fail = False
    result = placement().place_cpu_word_embeddings(model, torch.device("cpu"))
    assert result is model
    assert model.embedding is original
    assert not model.embedding.training


@pytest.mark.parametrize("option", [None, True, "auto", "cpu:0", "cuda:1"])
def test_placement_option_is_checked_before_cuda_or_loading(monkeypatch, option):
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: pytest.fail("Se abrió CUDA"))
    with pytest.raises(ValueError, match="ubicación"):
        embeddings.FrozenEncoders(word_embedding_placement=option)


def test_global_float64_is_rejected_before_loading_or_cuda(monkeypatch):
    previous = torch.get_default_dtype()
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: pytest.fail("Se abrió CUDA"))
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match="FP32"):
            embeddings.FrozenEncoders(word_embedding_placement="cpu")
    finally:
        torch.set_default_dtype(previous)


@pytest.fixture
def strict_flags():
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.get_default_dtype(),
    )
    embeddings.strict_fp32()
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous[0]
    torch.backends.cudnn.allow_tf32 = previous[1]
    torch.set_float32_matmul_precision(previous[2])
    torch.set_default_dtype(previous[3])


def enable_tf32(field):
    if field == "matmul_tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    elif field == "cudnn_tf32":
        torch.backends.cudnn.allow_tf32 = True
    elif field == "matmul_precision":
        torch.set_float32_matmul_precision("high")


def test_strict_fp32_records_every_flag(strict_flags):
    assert embeddings._strict_precision() == dict(
        dtype="float32", matmul_tf32=False, cudnn_tf32=False, float32_matmul_precision="highest"
    )


@pytest.mark.parametrize("field", ["matmul_tf32", "cudnn_tf32", "matmul_precision"])
def test_tf32_stops_the_encoder_at_startup_before_cuda_or_loading(monkeypatch, strict_flags, field):
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: pytest.fail("Se abrió CUDA"))
    monkeypatch.setattr(embeddings, "_tokenizer", lambda: pytest.fail("Se cargó el tokenizador"))
    enable_tf32(field)
    with pytest.raises(ValueError, match="FP32 estricto"):
        embeddings.FrozenEncoders(word_embedding_placement="cpu")
    with pytest.raises(ValueError, match="FP32 estricto"):
        embeddings.encoder_spec(word_embedding_placement="cpu")


@pytest.mark.parametrize("mode,argument", [("text", "contenido"), ("images", [b"fixture"])])
@pytest.mark.parametrize(
    "field", ["matmul_tf32", "cudnn_tf32", "matmul_precision", "dtype", "autocast"]
)
def test_precision_change_is_rejected_before_each_modality(strict_flags, mode, argument, field):
    encoder = embeddings.FrozenEncoders.__new__(embeddings.FrozenEncoders)
    encoder.spec = {"runtime_precision": embeddings._strict_precision()}
    enable_tf32(field)
    if field == "dtype":
        torch.set_default_dtype(torch.float64)
    with torch.autocast("cpu", enabled=field == "autocast"):
        with pytest.raises(ValueError, match="precisión|FP32"):
            getattr(encoder, mode)(argument)


@pytest.mark.parametrize("mode,placement", [("text", "cuda"), ("text", "cpu"), ("images", "cuda")])
@pytest.mark.parametrize("change", ["dropout", "float64", "trainable"])
def test_late_model_change_is_rejected_before_inputs(mode, placement, change):
    encoder = embeddings.FrozenEncoders.__new__(embeddings.FrozenEncoders)
    encoder.spec = {"runtime_precision": embeddings._runtime_precision()}
    encoder.word_embedding_placement = placement
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.5))
    model.eval().requires_grad_(False)
    if change == "dropout":
        model[1].train()
        assert not model.training
    elif change == "float64":
        model.double()
    else:
        model[0].requires_grad_(True)
    encoder.text_model = encoder.image_model = model
    encoder.tokenizer = lambda *a, **k: pytest.fail("Se preparó texto con un modelo alterado")
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="evaluación|FP32|congelad"):
        getattr(encoder, mode)("contenido" if mode == "text" else [b"fixture"])
    assert torch.equal(torch.get_rng_state(), before)


def test_cpu_text_path_supplies_embeddings_without_transferring_ids(monkeypatch):
    encoder = embeddings.FrozenEncoders.__new__(embeddings.FrozenEncoders)
    encoder.spec = {"runtime_precision": embeddings._runtime_precision()}
    encoder.device = torch.device("cpu")
    encoder.text_batch_size = 3
    encoder.word_embedding_placement = "cuda"

    class Tokenizer:
        cls_token_id, sep_token_id = 1000, 1001

        def __call__(self, *args, **kwargs):
            return {"input_ids": list(range(1, 884))}

        def pad(self, examples, **kwargs):
            width = max(len(row["input_ids"]) for row in examples)
            ids = torch.tensor(
                [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in examples],
                dtype=torch.int64,
                device="cpu",
            )
            return dict(input_ids=ids, attention_mask=ids != 0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.arange(1024 * 384, dtype=torch.float32, device="cpu").reshape(1024, 384)
                / 10000,
                freeze=True,
            )

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, attention_mask, input_ids=None, inputs_embeds=None):
            if encoder.word_embedding_placement == "cpu":
                assert input_ids is None and inputs_embeds is not None
            else:
                assert inputs_embeds is None
                inputs_embeds = self.embedding(input_ids)
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    encoder.tokenizer, encoder.text_model = Tokenizer(), Model().eval()
    expected = encoder.text("Texto por fragmentos")
    encoder.word_embedding_placement = "cpu"
    original_to = torch.Tensor.to

    def checked_to(tensor, *args, **kwargs):
        assert tensor.dtype != torch.int64, "Los IDs no deben cruzar el dispositivo"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", checked_to)
    actual = encoder.text("Texto por fragmentos")
    np.testing.assert_array_equal(actual, expected)


def test_constructor_records_placement_and_implementation_without_changing_default(
    monkeypatch, tmp_path, strict_flags
):
    import huggingface_hub
    import torchvision.models
    import transformers

    class TextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = table()

        def get_input_embeddings(self):
            return self.embedding

        def set_input_embeddings(self, value):
            self.embedding = value

    class Tokenizer:
        cls_token_id, sep_token_id = 0, 2
        backend_tokenizer = SimpleNamespace(to_str=lambda: "fixture")

        def __call__(self, *args, **kwargs):
            return {"input_ids": [0, 2]}

    weights = tmp_path / "checkpoints" / "weights"
    weights.parent.mkdir()
    weights.write_bytes(b"fixture")
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        (weights.parent / name).write_text("{}")
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: torch.device("cpu"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: TextModel())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(weights))
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(torchvision.models, "resnet18", lambda **k: torch.nn.Module())
    monkeypatch.setattr(
        torchvision.models,
        "ResNet18_Weights",
        SimpleNamespace(
            IMAGENET1K_V1=SimpleNamespace(
                url="https://example.invalid/weights", get_state_dict=lambda **k: {}
            )
        ),
    )
    original = embeddings.FrozenEncoders()
    explicit = embeddings.FrozenEncoders(word_embedding_placement="cuda")
    host = embeddings.FrozenEncoders(word_embedding_placement="cpu")
    assert original.spec == explicit.spec
    assert original.spec["runtime_precision"] == embeddings._runtime_precision()
    assert original.spec["runtime_precision"]["cudnn_tf32"] is False
    assert host.mean.dtype == host.std.dtype == torch.float32
    assert original.spec["word_embedding_placement"] == "cuda"
    assert "word_embedding_lookup" not in original.spec
    assert type(original.text_model.get_input_embeddings()) is torch.nn.Embedding
    assert host.spec["word_embedding_placement"] == "cpu"
    assert host.spec["word_embedding_lookup"]["policy"] == "cpu_word_inputs_embeds_fp32_v1"
    from pathlib import Path

    assert host.spec["word_embedding_lookup"]["code_sha256"] == embeddings.sha256(
        Path(placement().__file__)
    )
    assert type(host.text_model.get_input_embeddings()) is torch.nn.Embedding
    ids = torch.tensor([[1, 0, 3, 1]], dtype=torch.int64, device="cpu")
    with torch.inference_mode():
        assert torch.equal(
            original.text_model.get_input_embeddings()(ids),
            host.text_model.get_input_embeddings()(ids),
        )
    assert host.execution_budget == original.execution_budget
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda *a, **k: TextModel().double()
    )
    with pytest.raises(ValueError, match="FP32"):
        embeddings.FrozenEncoders()
    # La identidad calculada en CPU, sin CUDA ni modelos, es la del codificador cargado.
    monkeypatch.setattr(embeddings, "require_cuda", lambda **k: pytest.fail("Se abrió CUDA"))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: pytest.fail())
    assert embeddings.encoder_spec() == original.spec
    assert embeddings.encoder_spec(word_embedding_placement="cpu") == host.spec


@pytest.mark.parametrize("field", ["placement", "matmul_tf32", "cudnn_tf32", "dtype"])
def test_different_runtime_cannot_recover_old_corpus_or_cache(tmp_path, field):
    manifest, clock, macro = prepared_edition(tmp_path)
    previous, following = Encoders(), Encoders()
    previous.spec = {
        **previous.spec,
        "word_embedding_placement": "cuda",
        "runtime_precision": embeddings._runtime_precision(),
    }
    following.spec = deepcopy(previous.spec)
    if field == "placement":
        following.spec["word_embedding_placement"] = "cpu"
    elif field == "dtype":
        following.spec["runtime_precision"][field] = "float64"
    else:
        following.spec["runtime_precision"][field] = not previous.spec["runtime_precision"][field]
    output = tmp_path / "encoded"
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2)
    corpus_encoding.encode_corpus(manifest, output, encoders=previous, **kwargs)
    with pytest.raises(ValueError, match="configuración"):
        corpus_encoding.encode_corpus(manifest, output, encoders=following, **kwargs)
    assert following.calls == 0
    cache = embeddings.EmbeddingCache(tmp_path / "cache.sqlite")
    try:
        cache.put({"encoder": previous.spec}, np.ones(384))
        assert cache.get({"encoder": following.spec}) is None
    finally:
        cache.close()


@pytest.mark.parametrize(
    "arguments,expected", [([], "cuda"), (["--word-embedding-placement", "cpu"], "cpu")]
)
def test_cli_forwards_explicit_placement_without_loading_models(monkeypatch, arguments, expected):
    import sys

    captured = []
    monkeypatch.setattr(sys, "argv", ["encode", "--prepared", "p", "--output", "o", *arguments])
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)

    def encode(*args, **kwargs):
        captured.append(kwargs)
        return dict(
            cohort_id="fixture",
            candidate_count=1,
            samples=1,
            failed_assets=0,
            cohort_complete=True,
            reused_assets=0,
            stop_reason=None,
        )

    monkeypatch.setattr(corpus_encoding, "encode_corpus", encode)
    corpus_encoding.main()
    assert captured[0]["encoder_options"]["word_embedding_placement"] == expected
