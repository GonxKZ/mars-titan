"""Reutilización de vectores entre ediciones y recorrido por fragmentos, sin modelos."""

import json

import pyarrow.parquet as pq
import pytest

from mars_titan.data.corpus_encoding import encode_corpus
from mars_titan.data.storage import sha256
from mars_titan.data.vector_carry import MissingVector, ReuseOnlyEncoders
from tests.data.test_cohort_samples import Encoders
from tests.data.test_corpus_encoding import prepared_edition


def first_edition(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    # Como en la v3, los gráficos no se guardan en la caché y solo viven en las muestras.
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2, cache_charts=False)
    previous = tmp_path / "previous"
    result = encode_corpus(manifest, previous, encoders=Encoders(), **kwargs)
    assert result["cohort_complete"] is True
    return manifest, previous, kwargs


def test_identical_inputs_reuse_every_vector_without_an_encoder(tmp_path):
    manifest, previous, kwargs = first_edition(tmp_path)
    output = tmp_path / "carried"
    result = encode_corpus(
        manifest,
        output,
        encoders=ReuseOnlyEncoders(Encoders.spec),
        vector_carry=previous,
        **kwargs,
    )
    assert result["cohort_complete"] is True and result["failed_assets"] == 0
    configuration = json.loads((output / "configuration.json").read_text())
    assert configuration["vector_carry"]["edition"] == str(previous.resolve())
    old = pq.read_table(previous / "samples/US/A/samples.parquet")
    new = pq.read_table(output / "samples/US/A/samples.parquet")
    assert new.equals(old)
    receipt = json.loads((output / "samples/US/A/manifest.json").read_text())
    assert receipt["cache_misses"] == {}
    assert receipt["cache_hits"]["chart"] == new.num_rows
    # La edición anterior no cambia por leerse.
    assert (
        sha256(previous / "samples/US/A/samples.parquet")
        == json.loads((previous / "samples/US/A/manifest.json").read_text())["samples_sha256"]
    )


def test_a_missing_vector_leaves_the_asset_pending_for_the_gpu(tmp_path):
    manifest, previous, kwargs = first_edition(tmp_path)
    (previous / "samples/US/A/manifest.json").unlink()
    output = tmp_path / "carried"
    pending = encode_corpus(
        manifest, output, encoders=ReuseOnlyEncoders(Encoders.spec), vector_carry=previous, **kwargs
    )
    assert pending["failed_assets"] == 1 and not (output / "manifest.json").exists()
    assert "GPU" in pending["coverage"][0]["detail"]
    assert not (output / "samples/US/A/samples.parquet").exists()
    encoders = Encoders()
    done = encode_corpus(manifest, output, encoders=encoders, vector_carry=previous, **kwargs)
    assert done["cohort_complete"] is True and encoders.calls > 0
    with pytest.raises(MissingVector):
        ReuseOnlyEncoders(Encoders.spec).images([b"png"])


def test_another_encoder_cannot_reuse_the_previous_vectors(tmp_path):
    manifest, previous, kwargs = first_edition(tmp_path)
    other = ReuseOnlyEncoders({**Encoders.spec, "version": 2})
    with pytest.raises(ValueError, match="otro codificador"):
        encode_corpus(manifest, tmp_path / "other", encoders=other, vector_carry=previous, **kwargs)


def test_shards_share_the_edition_and_leave_publication_to_the_full_pass(tmp_path):
    manifest, clock, macro = prepared_edition(tmp_path)
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2, encoders=Encoders())
    output = tmp_path / "sharded"
    with pytest.raises(ValueError, match="configuración ya publicada"):
        encode_corpus(manifest, output, shard=(0, 2), **kwargs)
    with pytest.raises(ValueError, match="fragmento"):
        encode_corpus(manifest, output, shard=(2, 2), **kwargs)
    full = encode_corpus(manifest, output, **kwargs)
    identity = sha256(output / "manifest.json")
    first = encode_corpus(manifest, output, shard=(0, 2), **kwargs)
    second = encode_corpus(manifest, output, shard=(1, 2), **kwargs)
    assert [c["symbol"] for c in first["coverage"]] == ["A"]
    assert [c["symbol"] for c in second["coverage"]] == ["B"]
    assert first["reused_assets"] == 1
    assert (output / "progress-0-of-2.json").exists() and (output / "progress-1-of-2.json").exists()
    assert sha256(output / "manifest.json") == identity
    assert full["cohort_complete"] is True


def test_edition_comparison_logs_every_difference_and_resumes(tmp_path):
    from mars_titan.data.edition_comparison import compare_editions

    manifest, previous, kwargs = first_edition(tmp_path)
    output = tmp_path / "carried"
    encode_corpus(
        manifest, output, encoders=ReuseOnlyEncoders(Encoders.spec), vector_carry=previous, **kwargs
    )
    summary = compare_editions(previous, output, tmp_path / "comparison", workers=1)
    assert summary["assets"] == 1 and summary["common_sessions"] == summary["samples"] > 0
    assert summary["changed_charts"] == summary["other_differences"] == 0
    assert summary["dropped_sessions"] == summary["windows_changed"] == 0
    # Una edición distinta queda registrada sesión a sesión.
    table = pq.read_table(output / "samples/US/A/samples.parquet")
    rows = table.to_pylist()
    rows[0]["chart_hash"] = "0" * 64
    rows[1]["macro"] = [v + 1 for v in rows[1]["macro"]]
    changed = tmp_path / "changed"
    (changed / "samples/US/A").mkdir(parents=True)
    for name in ("configuration.json",):
        (changed / name).write_bytes((output / name).read_bytes())
    pq.write_table(
        type(table).from_pylist(rows, schema=table.schema), changed / "samples/US/A/samples.parquet"
    )
    (changed / "samples/US/A/manifest.json").write_bytes(
        (output / "samples/US/A/manifest.json").read_bytes()
    )
    record = compare_editions(previous, changed, tmp_path / "difference", workers=1)
    assert record["changed_charts"] == 1 and record["other_differences"] == 1
    again = compare_editions(previous, changed, tmp_path / "difference", workers=1)
    assert again == record


def test_carried_vectors_check_the_previous_edition_and_the_encoder(tmp_path):
    from mars_titan.data.cohort_samples import _digest
    from mars_titan.data.embeddings import EmbeddingCache
    from mars_titan.data.vector_carry import CarriedVectors

    _, previous, _ = first_edition(tmp_path)
    carried = CarriedVectors(
        EmbeddingCache(tmp_path / "cache.sqlite"), previous, _digest(Encoders.spec)
    )
    try:
        assert carried.select("US", "A", Encoders.spec) > 0
        identity = dict(
            encoder=_digest(Encoders.spec), kind="chart", content=next(iter(carried.charts))
        )
        assert carried.get(identity) is not None
        # Otro codificador no hereda el vector aunque el PNG coincida.
        assert carried.get({**identity, "encoder": "0" * 64}) is None
        with pytest.raises(ValueError, match="otro codificador"):
            carried.select("US", "A", {**Encoders.spec, "version": 2})
        samples = previous / "samples/US/A/samples.parquet"
        samples.write_bytes(samples.read_bytes() + b"x")
        with pytest.raises(ValueError, match="cambiaron"):
            carried.select("US", "A", Encoders.spec)
    finally:
        carried.close()


def test_the_window_contract_must_declare_the_calendars_of_the_edition(tmp_path):
    from mars_titan.data.price_windows import calendar_digest, price_window_contract

    manifest, clock, macro = prepared_edition(tmp_path)
    kwargs = dict(macros={"US": macro}, clocks={"US": clock}, context=2, encoders=Encoders())
    declared = dict(
        start=clock.days[1].isoformat(),
        end=clock.days[-1].isoformat(),
        decisions_sha256=calendar_digest(clock),
    )
    # La huella coincide, pero el inicio declarado no reconstruye el mismo calendario.
    for calendars in ({"US": declared}, {"US": declared, "CN": declared}):
        contract = price_window_contract(calendars, dict.fromkeys(calendars, []))
        with pytest.raises(ValueError, match="no declara los calendarios"):
            encode_corpus(manifest, tmp_path / "edition", price_window=contract, **kwargs)


def pending_count(edition):
    from mars_titan.data.vector_carry import pending_items

    return sum(1 for _ in pending_items(edition))


def test_collection_confirms_complete_assets_and_leaves_only_missing_inputs_to_the_gpu(tmp_path):
    from mars_titan.data.embeddings import EmbeddingCache
    from mars_titan.data.vector_carry import CollectingEncoders, encode_pending

    manifest, previous, kwargs = first_edition(tmp_path)
    reference = pq.read_table(previous / "samples/US/A/samples.parquet")
    collector = CollectingEncoders(Encoders.spec)
    # Con todos los vectores disponibles, la recogida confirma el activo en una sola pasada.
    whole = encode_corpus(
        manifest, tmp_path / "whole", encoders=collector, vector_carry=previous, **kwargs
    )
    assert whole["cohort_complete"] is True and pending_count(tmp_path / "whole") == 0
    assert pq.read_table(tmp_path / "whole/samples/US/A/samples.parquet").equals(reference)
    # Sin los gráficos ni los textos de la edición anterior, todo queda pendiente de GPU.
    (previous / "samples/US/A/manifest.json").unlink()
    (previous / "embeddings.sqlite").unlink()
    EmbeddingCache(previous / "embeddings.sqlite").close()
    output = tmp_path / "carried"
    first = encode_corpus(manifest, output, encoders=collector, vector_carry=previous, **kwargs)
    assert first["failed_assets"] == 1 and "GPU" in first["coverage"][0]["detail"]
    assert not (output / "samples/US/A/manifest.json").exists()
    assert not (output / "collect/US/A").exists()
    # Un resto de una recogida interrumpida nunca confirmado se descarta al repetirla.
    (output / "collect/US/A").mkdir(parents=True)
    (output / "collect/US/A/configuration.json").write_text("{}")
    again = encode_corpus(manifest, output, encoders=collector, vector_carry=previous, **kwargs)
    assert again["coverage"] == first["coverage"]
    pending = pending_count(output)
    charts = len(set(reference.column("chart_hash").to_pylist()))
    assert pending >= charts > 0
    encoders = Encoders()
    assert encode_pending(output, encoders, max_items=1) == dict(
        encoded=1, already=0, remaining=pending - 1
    )
    assert encode_pending(output, encoders) == dict(encoded=pending - 1, already=1, remaining=0)
    # Cada gráfico distinto se codifica una sola vez.
    assert encoders.calls == charts
    other = type("Other", (), {"spec": {"version": 2}})()
    for edition in (output, tmp_path / "whole"):
        # Otro codificador se rechaza aunque no quede nada pendiente.
        with pytest.raises(ValueError, match="codificador"):
            encode_pending(edition, other)
    final = encode_corpus(
        manifest,
        output,
        encoders=ReuseOnlyEncoders(Encoders.spec),
        vector_carry=previous,
        **kwargs,
    )
    assert final["cohort_complete"] is True
    # Los vectores calculados aparte coinciden con los de la codificación en línea.
    assert pq.read_table(output / "samples/US/A/samples.parquet").equals(reference)


def test_pending_inputs_are_read_once_and_a_corrupted_record_is_rejected(tmp_path):
    import sqlite3

    from mars_titan.data.vector_carry import PendingVectors, pending_items

    identity = dict(encoder="e" * 64, kind="chart", content="c" * 64)
    for name in ("pending-vectors-0-of-2.sqlite", "pending-vectors-1-of-2.sqlite"):
        store = PendingVectors(tmp_path / name)
        store.add(identity, "image", b"png")
        store.add(identity, "image", b"png")
        store.close()
    assert list(pending_items(tmp_path)) == [(identity, "image", b"png")]
    with pytest.raises(ValueError, match="texto ni un PNG"):
        PendingVectors(tmp_path / "pending-vectors.sqlite").add(identity, "audio", b"x")
    with sqlite3.connect(tmp_path / "pending-vectors-1-of-2.sqlite") as db:
        db.execute("UPDATE pending SET payload = ?", (b"other",))
    with pytest.raises(ValueError, match="corrupto"):
        list(pending_items(tmp_path))
