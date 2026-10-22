"""Archivos regionales comprimidos reales y rechazo de contenido dañado sin publicar."""

import json
import zlib
from copy import deepcopy

import pytest
from test_regional_work_reservation import (
    IDENTITY,
    ROWS,
    archive_event,
    archive_proof,
    confirmed_history,
    index_bytes,
    latest,
    ledger,
    prefix,
    prune_archived,
    workspace,
)

from mars_titan.memory.mechanism_history import (
    CHUNK_KIND,
    MechanismHistory,
    MechanismHistoryConfig,
)

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


def configure_history(env, compression):
    archive = MechanismHistory(
        env.artifacts,
        identity=IDENTITY["contract_id"],
        model_id=IDENTITY["model_id"],
        config=MechanismHistoryConfig(
            target_chunk_bytes=128 * 1024,
            max_event_bytes=192 * 1024,
            compression=compression,
        ),
    )
    env.link["history"] = env.history = archive


def native_roots(env):
    head = json.loads((env.directory / "latest.json").read_bytes())
    generation = head["generation"]
    names = (
        "latest.json",
        f"checkpoint-{generation}.json",
        f"checkpoint-{generation - 1}.json",
        f"record-{generation}.json",
    )
    return {name: (env.directory / name).read_bytes() for name in names}


@pytest.mark.parametrize("compression", ["none", "zlib1"])
def test_real_history_keeps_complete_receipts_after_compaction_pruning_and_resume(
    tmp_path, compression
):
    root = tmp_path / "run"
    receipts, sealed = [], []
    with workspace(root, financial=True, history=True) as env:
        configure_history(env, compression)
        book = ledger(env)
        book.initialize(prefix(env))
        for cutoff in (10, 20, 30):
            receipt = archive_event(env, book, cutoff=cutoff)
            receipts.append(receipt)
            sealed.append(
                env.artifacts.read(
                    receipt["reference"],
                    identity=IDENTITY["contract_id"],
                    kind="regional_work_reservation",
                )
            )
        proof = archive_proof(env)
        block = env.artifacts.read(
            proof["reference"], identity=IDENTITY["contract_id"], kind=CHUNK_KIND
        )
        assert block["compression"] == compression and len(block["events"]) == 3
        roots = native_roots(env)
        head = latest(env)
        book.compact(head, archive=proof)
        assert native_roots(env) == roots and latest(env) == head
        assert book.snapshot()["archived"]["cohorts"] == 1
        assert len(book.snapshot()["cohorts"]) == 2
        prune_archived(env, book)
        assert not (env.artifacts.directory / receipts[0]["reference"]["name"]).exists()
        compacted = index_bytes(env)

    with workspace(root, resume=True, financial=True, history=True) as env:
        configure_history(env, compression)
        book = ledger(env)
        book.resume(latest(env))
        assert index_bytes(env) == compacted
        assert native_roots(env) == roots and latest(env) == head
        manifest = confirmed_history(env)
        assert env.history.verify(manifest, strong=True)["events"] == 3
        events = list(env.history.iter_events(manifest))
        for generation, (event, receipt, stored) in enumerate(
            zip(events, receipts, sealed, strict=True), 1
        ):
            work = event["payload"]["regional_work"]
            assert work["reservation_id"] == receipt["reservation_id"]
            assert work["reference"] == receipt["reference"]
            assert work["receipts"] == [stored]
            assert work["cohort"]["charged_units"] == receipt["units"]
            assert work["cohort"]["attempts"][0]["status"] == "confirmed"
            assert (
                book.verify_archive_payload(
                    work,
                    generation=generation,
                    cutoff=10 * generation,
                    rows_digest=ROWS,
                    event_kind="decision",
                )
                == work
            )
        assert (
            book.snapshot()["cohorts"][-1]["attempts"][-1]["reservation_id"]
            == receipts[-1]["reservation_id"]
        )
        assert env.executor.cursor == 3


def damaged_event(encoded, *, compression, damage, maximum):
    if damage == "corrupt":
        return b"invalid-zlib-stream"
    if damage == "truncated":
        return encoded[:-1]
    if damage == "trailing":
        return encoded + b"trailing"
    if damage == "oversized":
        return zlib.compress(b" " * (maximum + 1), level=1)
    if damage == "json":
        plain = b"{"
    elif damage == "utf8":
        plain = b"\xff"
    else:
        plain = zlib.decompress(encoded) if compression == "zlib1" else encoded
        plain = b'{"generation":3,' + plain[1:]
    return zlib.compress(plain, level=1) if compression == "zlib1" else plain


@pytest.mark.parametrize(
    ("compression", "damage", "message"),
    [
        ("zlib1", "corrupt", "no conserva su compresión"),
        ("zlib1", "truncated", "incompleto o excede su límite"),
        ("zlib1", "trailing", "incompleto o excede su límite"),
        ("zlib1", "oversized", "incompleto o excede su límite"),
        ("zlib1", "json", "no conserva su JSON"),
        ("zlib1", "utf8", "no conserva su JSON"),
        ("zlib1", "duplicate", "no conserva su JSON"),
        ("none", "json", "no conserva su JSON"),
        ("none", "utf8", "no conserva su JSON"),
        ("none", "duplicate", "no conserva su JSON"),
    ],
)
def test_authentic_chunk_with_damaged_event_cannot_change_index_or_confirmed_roots(
    tmp_path, compression, damage, message
):
    with workspace(tmp_path / "run", financial=True, history=True) as env:
        configure_history(env, compression)
        book = ledger(env)
        book.initialize(prefix(env))
        receipts = [archive_event(env, book, cutoff=cutoff) for cutoff in (10, 20, 30)]
        proof = archive_proof(env)
        original = env.artifacts.read(
            proof["reference"], identity=IDENTITY["contract_id"], kind=CHUNK_KIND
        )
        damaged = deepcopy(original)
        damaged["events"][-1] = damaged_event(
            original["events"][-1],
            compression=compression,
            damage=damage,
            maximum=env.artifacts.max_bytes,
        )
        reference = env.artifacts.stage(damaged, identity=IDENTITY["contract_id"], kind=CHUNK_KIND)
        assert reference != proof["reference"]
        assert (
            env.artifacts.read(reference, identity=IDENTITY["contract_id"], kind=CHUNK_KIND)
            == damaged
        )
        before, roots, head = index_bytes(env), native_roots(env), latest(env)
        references = book.references()
        with pytest.raises(ValueError, match=message):
            book.compact(head, archive=dict(proof, reference=reference))
        assert index_bytes(env) == before
        assert native_roots(env) == roots and latest(env) == head
        assert book.references() == references
        assert archive_proof(env) == proof
        assert env.executor.cursor == 3
        book.compact(head, archive=proof)
        assert book.snapshot()["archived"]["cohorts"] == 1
        assert len(book.snapshot()["cohorts"]) == 2
        work = env.history.read_event(confirmed_history(env), 3)["payload"]["regional_work"]
        assert work["reservation_id"] == receipts[-1]["reservation_id"]
        assert work["reference"] == receipts[-1]["reference"]
        assert native_roots(env) == roots and latest(env) == head
