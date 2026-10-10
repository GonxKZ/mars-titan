"""Textos heredados de la v3 con un contraste bit a bit por activo, sin modelos."""

import json
import sqlite3

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data import vector_carry
from mars_titan.data.cohort_samples import _digest
from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.edition_comparison import compare_asset, compare_editions
from mars_titan.data.edition_substitution import substitute_asset
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.vector_carry import (
    REENCODED,
    REUSED,
    CarriedTexts,
    CarriedVectors,
    CollectingEncoders,
    PendingVectors,
    ReuseOnlyEncoders,
    carried_texts,
    check_text_source,
    encode_pending,
    pending_items,
    release_vectors,
    text_carry_record,
    text_check_positions,
)
from tests.data.test_vector_carry import STRICT, Encoders, TF32Encoders, first_edition


class Counting(Encoders):
    """Codificador estricto que anota los textos que recibe."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def text(self, text):
        self.texts.append(text)
        return super().text(text)


class DriftedTexts(TF32Encoders):
    """Como la v3, pero cada texto difiere en el último bit de todas sus componentes."""

    def text(self, text):
        vector = super().text(text)
        return np.nextafter(vector, np.float32(np.inf))


# Doce noticias en días distintos de julio de 2023, para que el activo herede varios textos.
ARTICLES = [
    dict(
        Date=f"2023-07-{3 + i // 2:02d}",
        Stock_symbol="A",
        Article=f"La empresa A publica la nota {i} con {'detalle ' * i}.",
        Article_title=f"Nota {i}",
        Url=f"https://example.org/a/{i}",
    )
    for i in range(12)
]


def v3_edition(tmp_path, encoders=TF32Encoders):
    return first_edition(tmp_path, encoders, ARTICLES)


def collect(manifest, output, previous, kwargs):
    result = encode_corpus(
        manifest,
        output,
        encoders=CollectingEncoders(Encoders.spec),
        text_carry=previous,
        **kwargs,
    )
    # Los textos recibieron ceros en la recogida, así que el activo nunca se confirma en ella.
    assert result["failed_assets"] == 1 and "GPU" in result["coverage"][0]["detail"]
    ((asset, entries),) = list(carried_texts(output))
    assert asset == ("US", "A")
    return entries


def finish(manifest, output, previous, kwargs):
    return encode_corpus(
        manifest, output, encoders=ReuseOnlyEncoders(Encoders.spec), text_carry=previous, **kwargs
    )


def test_the_sample_takes_the_first_the_last_one_in_64_and_at_least_eight():
    assert text_check_positions(0) == set()
    assert text_check_positions(5) == set(range(5))
    assert text_check_positions(8) == set(range(8))
    nine = text_check_positions(9)
    assert len(nine) == 8 and {0, 8} <= nine
    for count in (9, 63, 100, 449, 1000, 5000):
        positions = text_check_positions(count)
        assert {0, count - 1} | set(range(0, count, 64)) <= positions
        assert len(positions) >= 8 and max(positions) < count
        # Entre dos textos contrastados nunca quedan más de 63 sin contrastar.
        assert np.diff(sorted(positions)).max() <= 64
    assert len(text_check_positions(5000)) == 79 + 8 - 1


def test_v3_texts_are_reused_after_a_bit_identical_sample(tmp_path, monkeypatch):
    # Con un mínimo de 2, la muestra es el primer texto y el último, y el resto se hereda.
    monkeypatch.setattr(vector_carry, "_TEXT_CHECK_MIN", 2)
    manifest, previous, kwargs = v3_edition(tmp_path)
    output = tmp_path / "v31"
    entries = collect(manifest, output, previous, kwargs)
    assert len(entries) >= 3
    assert [sampled for *_, sampled in entries] == [True] + [False] * (len(entries) - 2) + [True]
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["text_carry"]["edition"] == str(previous.resolve())
    assert configuration["text_carry"]["encoder_sha256"] == _digest(TF32Encoders.spec)
    # Con entradas sin codificar, el contraste espera a la pasada que las complete.
    partial = encode_pending(output, Counting(), max_items=1)
    assert partial["remaining"] > 0 and "text_carry" not in partial
    assert text_carry_record(output, "US", "A") is None
    encoders = Counting()
    counts = encode_pending(output, encoders)
    # La GPU solo codifica los textos de la muestra, además de los gráficos.
    assert len(encoders.texts) == counts["by_kind"]["text"]["encoded"] == 2
    assert {text.encode() for text in encoders.texts} == {entries[0][1], entries[-1][1]}
    assert counts["text_carry"] == dict(
        assets=1, reused=1, reencoded=0, compared=2, mismatched=0, reencoded_texts=0
    )
    record = text_carry_record(output, "US", "A")
    assert record["decision"] == REUSED and record["reencoded"] == 0
    assert (record["candidates"], record["compared"], record["mismatched"]) == (len(entries), 2, 0)
    assert record["source"] == {
        k: configuration["text_carry"][k]
        for k in ("edition", "configuration_sha256", "encoder_sha256")
    }
    final = finish(manifest, output, previous, kwargs)
    assert final["cohort_complete"] is True and final["failed_assets"] == 0
    reference = pq.read_table(previous / "samples/US/A/samples.parquet")
    assert pq.read_table(output / "samples/US/A/samples.parquet").equals(reference)
    comparison = compare_asset(previous, output, "US", "A")
    assert comparison["other_differences"] == [] and comparison["reencoded_texts"] is None
    # Una segunda pasada de GPU no vuelve a contrastar un activo con constancia.
    assert encode_pending(output, Counting())["text_carry"]["assets"] == 0
    # Un vector ya calculado en FP32 estricto tiene preferencia sobre el heredado.
    inherited = entries[1][0]
    computed = EmbeddingCache(output / "computed-vectors.sqlite")
    computed.put(inherited, np.full(384, 7.0, dtype=np.float32))
    computed.close()
    vectors = CarriedVectors(
        EmbeddingCache(tmp_path / "own.sqlite"),
        None,
        _digest(Encoders.spec),
        fallbacks=[EmbeddingCache(output / "computed-vectors.sqlite", read_only=True)],
        texts=CarriedTexts(output, configuration["text_carry"], _digest(Encoders.spec)),
    )
    try:
        vectors.select("US", "A", Encoders.spec)
        assert (vectors.get(inherited) == 7.0).all()
        assert vectors.get(entries[2][0]).tobytes() == (
            TF32Encoders().text(entries[2][1].decode()).tobytes()
        )
    finally:
        vectors.close()
    assert release_vectors(output)["assets"] == 1
    # Tras liberar los pendientes, el activo confirmado se reutiliza tal cual.
    again = finish(manifest, output, previous, kwargs)
    assert again["cohort_complete"] is True and again["reused_assets"] == 1


def test_a_text_that_differs_reencodes_every_text_of_the_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_carry, "_TEXT_CHECK_MIN", 2)
    manifest, previous, kwargs = v3_edition(tmp_path, DriftedTexts)
    strict = tmp_path / "strict"
    encode_corpus(manifest, strict, encoders=Encoders(), **kwargs)
    output = tmp_path / "v31"
    entries = collect(manifest, output, previous, kwargs)
    encoders = Counting()
    counts = encode_pending(output, encoders)
    assert counts["text_carry"] == dict(
        assets=1, reused=0, reencoded=1, compared=2, mismatched=2, reencoded_texts=len(entries) - 2
    )
    # Todos los textos del activo salen de la GPU y ninguno de la v3.
    assert sorted(encoders.texts) == sorted(payload.decode() for _, payload, _ in entries)
    record = text_carry_record(output, "US", "A")
    assert record["decision"] == REENCODED and record["reencoded"] == len(entries) - 2
    # Un activo recodificado nunca recibe un texto de la v3.
    configuration = json.loads((output / "configuration.json").read_text())
    texts = CarriedTexts(output, configuration["text_carry"], _digest(Encoders.spec))
    try:
        texts.select("US", "A")
        assert texts.previous(entries[1][0]) is not None and texts.get(entries[1][0]) is None
    finally:
        texts.close()
    assert record["mismatched_content"] == [entries[0][0]["content"], entries[-1][0]["content"]]
    assert finish(manifest, output, previous, kwargs)["cohort_complete"] is True
    current = pq.read_table(output / "samples/US/A/samples.parquet")
    assert current.equals(pq.read_table(strict / "samples/US/A/samples.parquet"))
    assert not current.equals(pq.read_table(previous / "samples/US/A/samples.parquet"))
    # La comparación mide la media de noticias recodificada en lugar de tratarla como un error.
    comparison = compare_asset(previous, output, "US", "A")
    assert comparison["other_differences"] == []
    changes = comparison["reencoded_texts"]
    assert changes["compared"] == comparison["common"] > changes["identical"]
    assert 0 < changes["max_rel"] < 1e-6
    summary = compare_editions(previous, output, tmp_path / "comparison", workers=1)
    assert summary["reencoded_texts"]["compared"] == changes["compared"]
    # Sin la constancia del activo, el mismo cambio sería una diferencia no declarada.
    path = output / "text-carry/US/A.json"
    saved = path.read_bytes()
    path.unlink()
    assert {
        name for _, name in compare_asset(previous, output, "US", "A")["other_differences"]
    } == {"news"}
    path.write_bytes(saved)
    substituted = substitute_asset(previous, output, tmp_path / "records", "US", "A")
    assert substituted["comparison"]["reencoded_texts"] == changes
    assert not (previous / "samples/US/A/samples.parquet").exists()


def test_only_a_compatible_text_model_lends_its_vectors(tmp_path):
    check_text_source(TF32Encoders.spec, Encoders.spec)
    # Como entre la v3 y la v3.1, el código del módulo y el lote de gráficos pueden cambiar.
    shared = dict(word_embedding_placement="cpu", text_model="m")
    v3 = {
        **TF32Encoders.spec,
        **shared,
        "code_sha256": "a",
        "batch_sizes": dict(text_chunks=8, images=2),
    }
    v31 = {
        **Encoders.spec,
        **shared,
        "code_sha256": "b",
        "batch_sizes": dict(text_chunks=8, images=8),
    }
    check_text_source(v3, v31)
    with pytest.raises(ValueError, match="actual no registra FP32 estricto"):
        check_text_source(v3, {**v31, "runtime_precision": {**STRICT, "cudnn_tf32": True}})
    for precision in ({**STRICT, "matmul_tf32": True}, {**STRICT, "dtype": "float16"}, None):
        with pytest.raises(ValueError, match="sin TF32 en matmul"):
            check_text_source({**v3, "runtime_precision": precision}, v31)
    for change in (
        {"text_model": "other"},
        {"word_embedding_placement": "cuda"},
        {"batch_sizes": dict(text_chunks=4, images=2)},
        {"tokenizers_version": "0"},
    ):
        with pytest.raises(ValueError, match="otro modelo"):
            check_text_source({**v3, **change}, v31)
    manifest, previous, kwargs = v3_edition(tmp_path)
    with pytest.raises(ValueError, match="FP32 estricto"):
        encode_corpus(
            manifest,
            tmp_path / "tf32",
            encoders=CollectingEncoders(TF32Encoders.spec),
            text_carry=previous,
            **kwargs,
        )
    output = tmp_path / "v31"
    collect(manifest, output, previous, kwargs)
    # Todas las pasadas declaran la misma herencia de textos.
    with pytest.raises(ValueError, match="otra edición"):
        encode_corpus(manifest, output, encoders=ReuseOnlyEncoders(Encoders.spec), **kwargs)
    # Si la edición de origen cambia, la GPU no contrasta contra ella.
    configuration = previous / "configuration.json"
    configuration.write_text(json.dumps(json.loads(configuration.read_text())))
    with pytest.raises(ValueError, match="Ha cambiado la edición"):
        encode_pending(output, Encoders())


def test_a_collected_again_asset_never_shrinks_its_sample(tmp_path):
    identities = [
        dict(encoder="e" * 64, kind="news", content=f"{i:064x}", policy="p") for i in range(20)
    ]
    entries = [(identity, f"text {i}".encode()) for i, identity in enumerate(identities)]
    store = PendingVectors(tmp_path / "pending-vectors.sqlite")
    with pytest.raises(ValueError, match="activo"):
        store.add_carried(("US",), entries)
    store.add_carried(("US", "A"), entries)
    store.add_carried(("US", "A"), entries[:10])
    store.add_carried(("US", "B"), [])
    store.close()
    # Otro registro, como el de un fragmento, recoge una parte del mismo activo.
    shard = PendingVectors(tmp_path / "pending-vectors-1-of-2.sqlite")
    shard.add_carried(("US", "A"), entries[10:])
    shard.close()
    ((asset, carried),) = list(carried_texts(tmp_path))
    expected = (
        text_check_positions(20)
        | text_check_positions(10)
        | {10 + position for position in text_check_positions(10)}
    )
    assert asset == ("US", "A") and len(carried) == 20
    # Los registros se leen por orden de nombre, así que se compara por contenido.
    assert {identity["content"] for identity, _, sampled in carried if sampled} == {
        identities[i]["content"] for i in expected
    }
    assert sorted(identity["content"] for identity, _, _ in carried) == sorted(
        identity["content"] for identity in identities
    )
    # Solo la muestra queda pendiente de la GPU.
    assert len(list(pending_items(tmp_path))) == len(expected)
    with sqlite3.connect(tmp_path / "pending-vectors.sqlite") as db:
        db.execute("UPDATE carried SET payload = ? WHERE rowid = 3", (b"other",))
    with pytest.raises(ValueError, match="corrupto"):
        list(carried_texts(tmp_path))


def test_an_invalid_record_of_the_check_is_rejected(tmp_path):
    path = tmp_path / "text-carry/US/A.json"
    path.parent.mkdir(parents=True)
    for record in (
        dict(market="US", symbol="A", decision="trusted"),
        dict(market="US", symbol="B", decision=REUSED),
    ):
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match="no es válida"):
            text_carry_record(tmp_path, "US", "A")
    assert text_carry_record(tmp_path, "US", "C") is None
