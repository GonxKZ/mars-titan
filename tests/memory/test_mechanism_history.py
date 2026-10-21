"""Historial sellado de propuestas técnicas, sin otro punto de publicación."""

import copy
import importlib
import os

import pytest
from test_native_episode_backend import native as native

from mars_titan.memory.session_artifacts import SessionArtifacts

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")
MODEL = "a" * 64
CONTRACT = "b" * 64
START = 1_609_459_200_000_000


def history(native, folder, **options):
    try:
        api = importlib.import_module("mars_titan.memory.mechanism_history")
    except ModuleNotFoundError:
        pytest.fail("Falta el historial sellado de mecanismos")
    return api.MechanismHistory(
        SessionArtifacts(native, folder),
        identity=CONTRACT,
        model_id=MODEL,
        config=api.MechanismHistoryConfig(**options),
    )


def event(sequence, **changes):
    return (
        dict(
            schema_version=1,
            model_id=MODEL,
            generation=sequence,
            cutoff=START + sequence,
            kind="decision",
            native_record_sha256="c" * 64,
            payload=dict(quantiles=[-2.0, -1.0, 0.0, 1.0, 2.0], action="off/k1/none"),
        )
        | changes
    )


def test_history_keeps_old_outputs_after_later_chunks_and_artifact_pruning(native, tmp_path):
    archive = history(native, tmp_path, max_events_per_chunk=2)
    manifest = archive.empty()
    expected = []
    for generation in range(1, 8):
        current = event(generation)
        current["payload"]["quantiles"][0] -= generation
        expected.append(copy.deepcopy(current))
        manifest = archive.append(manifest, current)
        archive.artifacts.prune_unreferenced(archive.live_references(manifest))
    assert manifest["events"] == 7 and len(manifest["chunks"]) == 4
    assert list(archive.iter_events(manifest)) == expected
    assert archive.read_event(manifest, 1) == expected[0]
    restored = history(native, tmp_path, max_events_per_chunk=2)
    assert restored.verify(manifest, strong=True)["events"] == 7
    assert restored.read_event(manifest, 1) == expected[0]


def test_new_proposal_does_not_change_the_previous_history_or_alias_caller_payloads(
    native, tmp_path
):
    archive = history(native, tmp_path)
    before = archive.empty()
    current = event(1)
    proposed = archive.append(before, current)
    current["payload"]["quantiles"][0] = -999
    assert before == archive.empty()
    with pytest.raises(ValueError):
        archive.read_event(before, 1)
    assert archive.read_event(proposed, 1)["payload"]["quantiles"][0] == -2.0


@pytest.mark.parametrize("change", ["bytes", "first", "events", "cutoff", "float_cutoff", "model"])
def test_forged_manifest_is_rejected_even_after_a_cache_hit(native, tmp_path, change):
    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    archive.verify(manifest)
    forged = copy.deepcopy(manifest)
    if change == "bytes":
        forged["bytes"] += 1
    elif change == "first":
        forged["chunks"][0]["first"] = 2
    elif change == "events":
        forged["events"] = True
    elif change == "cutoff":
        forged["cutoff"] += 1
    elif change == "float_cutoff":
        forged["cutoff"] = float(forged["cutoff"])
    else:
        forged["model_id"] = "d" * 64
    with pytest.raises(ValueError):
        archive.verify(forged)


def test_changed_chunk_bytes_are_rejected_when_size_and_mtime_are_restored(native, tmp_path):
    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    archive.verify(manifest)
    reference = manifest["chunks"][0]["reference"]
    path = tmp_path / reference["name"]
    stamp = path.stat()
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError):
        archive.verify(manifest)


def test_exhaustion_rejects_before_staging_without_discarding_old_evidence(native, tmp_path):
    archive = history(native, tmp_path, max_chunks=1, max_events_per_chunk=1)
    manifest = archive.append(archive.empty(), event(1))
    count = len(archive.artifacts._inventory())
    with pytest.raises(ValueError):
        archive.append(manifest, event(2))
    assert len(archive.artifacts._inventory()) == count
    assert archive.read_event(manifest, 1) == event(1)


@pytest.mark.parametrize(
    "changes",
    [
        dict(model_id="d" * 64),
        dict(generation=True),
        dict(generation=3),
        dict(cutoff=1),
        dict(kind="unknown"),
        dict(native_record_sha256="unknown"),
        dict(payload={"value": float("nan")}),
        dict(payload={"value": float("inf")}),
    ],
)
def test_invalid_event_envelopes_are_rejected_before_staging(native, tmp_path, changes):
    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    with pytest.raises(ValueError):
        archive.append(manifest, event(2, **changes))
    assert archive.read_event(manifest, 1) == event(1)


def test_chunk_byte_target_packs_small_events_and_isolates_larger_ones(native, tmp_path):
    archive = history(native, tmp_path, target_chunk_bytes=2048, max_event_bytes=4096)
    manifest = archive.empty()
    for index in range(1, 5):
        manifest = archive.append(manifest, event(index))
    assert len(manifest["chunks"]) == 1
    manifest = archive.append(manifest, event(5, payload={"text": "x" * 2500}))
    assert len(manifest["chunks"]) == 2
    before = len(archive.artifacts._inventory())
    with pytest.raises(ValueError):
        archive.append(manifest, event(6, payload={"text": "x" * 5000}))
    assert len(archive.artifacts._inventory()) == before


@pytest.mark.parametrize("location", ["event_kind", "event_model", "manifest_model"])
def test_foreign_objects_are_rejected_before_comparing_them(native, tmp_path, location):
    class Foreign:
        def __eq__(self, other):
            raise AssertionError("No se deben comparar objetos ajenos al contrato")

    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    with pytest.raises(ValueError):
        if location == "manifest_model":
            archive.verify(dict(manifest, model_id=Foreign()))
        else:
            archive.append(
                manifest,
                event(2, **{"kind" if location == "event_kind" else "model_id": Foreign()}),
            )


def test_budget_exhaustion_has_a_distinct_type_and_does_not_convert_contract_errors(
    native, tmp_path
):
    archive = history(native, tmp_path, target_chunk_bytes=2048, max_event_bytes=4096)
    api = importlib.import_module("mars_titan.memory.mechanism_history")
    with pytest.raises(api.HistoryCapacityError):
        archive.append(archive.empty(), event(1, payload={"text": "x" * 5000}))
    with pytest.raises(ValueError) as error:
        archive.append(archive.empty(), event(1, payload={"value": object()}))
    assert not isinstance(error.value, api.HistoryCapacityError)


def test_verification_cache_is_bounded_and_does_not_keep_payloads_of_old_proposals(
    native, tmp_path
):
    archive = history(native, tmp_path, max_chunks=2, max_events_per_chunk=32)
    manifest = archive.empty()
    for index in range(1, 21):
        manifest = archive.append(manifest, event(index))
        archive.artifacts.prune_unreferenced(archive.live_references(manifest))
        assert len(archive._verified) <= 4
        assert all(len(summary) == 4 for _, summary in archive._verified.values())
    assert len(archive.artifacts._inventory()) == 1
    assert archive.read_event(manifest, 1) == event(1)


def test_unchanged_history_uses_cached_validation_and_strong_recovery_reads_every_chunk(
    native, tmp_path, monkeypatch
):
    archive = history(native, tmp_path, max_events_per_chunk=1)
    manifest = archive.empty()
    for index in range(1, 5):
        manifest = archive.append(manifest, event(index))
    calls = []
    original = archive.artifacts.read

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(archive.artifacts, "read", counted)
    archive.verify(manifest)
    archive.verify(manifest)
    assert not calls
    archive.verify(manifest, strong=True)
    assert len(calls) == 4
    archive.read_event(manifest, 2)
    assert len(calls) == 5


@pytest.mark.parametrize(
    "field,value",
    [("bytes", 1), ("identity", "d" * 64), ("name", "../other.bin"), ("kind", "bank")],
)
def test_reference_metadata_cannot_bypass_validation_on_a_cached_file(
    native, tmp_path, field, value
):
    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    archive.verify(manifest)
    changed = copy.deepcopy(manifest)
    changed["chunks"][0]["reference"][field] = value
    with pytest.raises(ValueError):
        archive.verify(changed)


def test_history_configuration_cannot_change_when_reopening_existing_chunks(native, tmp_path):
    archive = history(native, tmp_path, max_events_per_chunk=2)
    manifest = archive.append(archive.empty(), event(1))
    other = history(native, tmp_path, max_events_per_chunk=3)
    with pytest.raises(ValueError):
        other.verify(manifest)
    assert archive.read_event(manifest, 1) == event(1)


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_history_requires_the_exact_schema_type(native, tmp_path, version):
    archive = history(native, tmp_path)
    manifest = archive.append(archive.empty(), event(1))
    changed = dict(manifest, schema_version=version)
    with pytest.raises(ValueError):
        archive.verify(changed)


def test_removing_an_earlier_chunk_cannot_create_a_valid_history_with_a_gap(native, tmp_path):
    archive = history(native, tmp_path, max_events_per_chunk=1)
    manifest = archive.append(archive.empty(), event(1))
    manifest = archive.append(manifest, event(2))
    forged = copy.deepcopy(manifest)
    forged["chunks"] = forged["chunks"][1:]
    forged["bytes"] = sum(chunk["reference"]["bytes"] for chunk in forged["chunks"])
    with pytest.raises(ValueError):
        archive.verify(forged)


def test_json_capacity_counts_the_entire_event_before_writing(native, tmp_path):
    archive = history(native, tmp_path, target_chunk_bytes=2048, max_event_bytes=4096)
    api = importlib.import_module("mars_titan.memory.mechanism_history")
    with pytest.raises(api.HistoryCapacityError):
        archive.append(archive.empty(), event(1, payload={"values": [0.0] * 4096}))
    assert archive.artifacts._inventory() == []
