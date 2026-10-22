"""Codificación de trazas sin pérdida y decodificación con un límite explícito."""

import copy
import zlib

import numpy as np
import pytest
from test_mechanism_history import CONTRACT, event, history
from test_native_episode_backend import native as native

pytestmark = pytest.mark.native("MARS_TITAN_EPISODIC_NATIVE")


@pytest.mark.parametrize("compression", ["none", "zlib1"])
def test_event_bytes_preserve_float_bits_int64_and_unicode(native, tmp_path, compression):
    archive = history(native, tmp_path, compression=compression)
    values = np.array([0.0, -0.0, np.nextafter(0.0, 1.0), np.finfo(float).max, -1e-300])
    current = event(1, payload=dict(values=values.tolist(), identifier=2**63 - 1, text="á\n中"))
    manifest = archive.append(archive.empty(), current)
    reference = manifest["chunks"][0]["reference"]
    stored = archive.artifacts.read(reference, identity=CONTRACT, kind="mechanism_trace_chunk")
    assert stored["encoding"] == "canonical_json_utf8_v1"
    assert stored["compression"] == compression
    assert type(stored["events"][0]) is bytes
    restored = archive.read_event(manifest, 1)
    assert np.asarray(restored["payload"]["values"], dtype=np.float64).tobytes() == values.tobytes()
    assert restored["payload"]["identifier"] == 2**63 - 1
    assert restored["payload"]["text"] == "á\n中"


def altered_manifest(archive, manifest, change):
    changed = copy.deepcopy(manifest)
    reference = changed["chunks"][0]["reference"]
    stored = archive.artifacts.read(reference, identity=CONTRACT, kind="mechanism_trace_chunk")
    stored["events"][0] = change(stored["events"][0])
    reference = archive.artifacts.stage(stored, identity=CONTRACT, kind="mechanism_trace_chunk")
    changed["chunks"][0]["reference"] = reference
    changed["bytes"] = reference["bytes"]
    return changed


@pytest.mark.parametrize("case", ["expansion", "incomplete", "trailing", "invalid"])
def test_compressed_input_cannot_exceed_the_decode_limit_or_hide_other_bytes(
    native, tmp_path, case
):
    archive = history(
        native, tmp_path, compression="zlib1", target_chunk_bytes=2048, max_event_bytes=4096
    )
    manifest = archive.append(archive.empty(), event(1))
    changes = dict(
        expansion=lambda _: zlib.compress(b" " * 5000, level=1),
        incomplete=lambda data: data[:-2],
        trailing=lambda data: data + b"other",
        invalid=lambda _: b"invalid",
    )
    changed = altered_manifest(archive, manifest, changes[case])
    with pytest.raises(ValueError):
        archive.verify(changed, strong=True)


@pytest.mark.parametrize("case", ["duplicate", "whitespace", "not_json"])
def test_raw_events_require_the_unique_canonical_json_representation(native, tmp_path, case):
    archive = history(native, tmp_path, compression="none")
    manifest = archive.append(archive.empty(), event(1))
    changes = dict(
        duplicate=lambda data: data.replace(b'"generation":1', b'"generation":1,"generation":1'),
        whitespace=lambda data: b" " + data,
        not_json=lambda _: b"not-json",
    )
    changed = altered_manifest(archive, manifest, changes[case])
    with pytest.raises(ValueError):
        archive.verify(changed, strong=True)


def test_appending_a_verified_chunk_does_not_decode_all_earlier_payloads(
    native, tmp_path, monkeypatch
):
    archive = history(native, tmp_path)
    manifest = archive.empty()
    for index in range(1, 5):
        manifest = archive.append(manifest, event(index))
    original = archive._decode
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(archive, "_decode", counted)
    manifest = archive.append(manifest, event(5))
    archive.verify(manifest)
    assert calls == []
    assert archive.read_event(manifest, 1) == event(1)
    assert calls == [1]
    archive.verify(manifest, strong=True)
    assert len(calls) == 6


def test_compression_identity_is_checked_before_reopening_an_existing_history(native, tmp_path):
    first = history(native, tmp_path, compression="none")
    manifest = first.append(first.empty(), event(1))
    second = history(native, tmp_path, compression="zlib1")
    with pytest.raises(ValueError):
        second.verify(manifest)


def test_decode_supplies_a_bound_to_the_real_zlib_decompressor(native, tmp_path, monkeypatch):
    archive = history(
        native, tmp_path, compression="zlib1", target_chunk_bytes=2048, max_event_bytes=4096
    )
    manifest = archive.append(archive.empty(), event(1))
    forged = altered_manifest(archive, manifest, lambda _: zlib.compress(b" " * 5000, level=1))
    create = zlib.decompressobj
    observed = []

    class ObservedDecoder:
        def __init__(self):
            self.delegate = create()

        def decompress(self, data, max_length=0):
            observed.append(max_length)
            return self.delegate.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    monkeypatch.setattr(zlib, "decompressobj", ObservedDecoder)
    with pytest.raises(ValueError):
        archive.verify(forged, strong=True)
    assert observed == [4097]
