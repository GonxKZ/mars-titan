"""Reproducciones independientes sobre fixtures, sin datos de la campaña."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import joint_corpus as api
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.data.test_joint_corpus import pair


def test_recovery_binds_configuration_to_validated_bytes(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    output = tmp_path / "joint"
    original_project = api._project

    def interrupt(*_args, **_kwargs):
        raise InterruptedError("Corte antes del primer recibo de activo")

    monkeypatch.setattr(api, "_project", interrupt)
    with pytest.raises(InterruptedError):
        api.prepare_joint_corpus(sources(parents), output)
    monkeypatch.setattr(api, "_project", original_project)
    original_read = api.read_manifest

    def replace_after_read(path, *args, **kwargs):
        result = original_read(path, *args, **kwargs)
        if Path(path) == output / "configuration.json":
            altered = json.loads(Path(path).read_text())
            altered["policy"] = "configuration_replaced_after_read"
            atomic_json(path, altered)
        return result

    monkeypatch.setattr(api, "read_manifest", replace_after_read)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources(parents), output)
    assert not (output / "report.json").exists()


def change_artifact(parents, kind, transform):
    manifest, paths, _ = parents["US"]
    path = paths[kind]
    pq.write_table(transform(pq.read_table(path)), path)
    meta = json.loads(manifest.read_text())
    meta["assets"][0][kind + "_sha256"] = sha256(path)
    atomic_json(manifest, meta)


def sources(parents):
    return {market: item[0] for market, item in parents.items()}


def replacement(table, name, value):
    return table.set_column(table.schema.get_field_index(name), name, value)


@pytest.mark.parametrize("damage", ["price_index", "null_availability", "label_partition"])
def test_no_confirmation_of_inputs_rejected_by_reader(tmp_path, damage):
    parents = pair(tmp_path)
    if damage == "price_index":
        change_artifact(
            parents, "samples", lambda t: replacement(t, "price_end_index", pa.array([0, 63]))
        )
    elif damage == "null_availability":

        def corrupt(table):
            values = table["input_availability"].to_pylist()
            values[0]["fundamentals"] = None
            return replacement(
                table, "input_availability", pa.array(values, type=table["input_availability"].type)
            )

        change_artifact(parents, "samples", corrupt)
    else:
        change_artifact(
            parents,
            "labels",
            lambda t: replacement(t, "partition", pa.array(["validation", "train"])),
        )
    output = tmp_path / "joint"
    try:
        report = api.prepare_joint_corpus(sources(parents), output)
    except ValueError:
        assert not (output / "report.json").exists()
        return
    assert report["status"] == "completed"
    with pytest.raises(ValueError):
        list(
            CorpusDataset(output / "supervised/manifest.json").batches(
                partition="train",
                batch_size=2,
                epoch=0,
                seed=42,
            )
        )
    pytest.fail(f"La unión confirmó {damage}, pero CorpusDataset.batches lo rechazó")


def test_configuration_changed_during_projection_is_not_confirmed(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    output = tmp_path / "joint"
    original = api._project
    changed = False

    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            config = output / "configuration.json"
            payload = json.loads(config.read_text())
            payload["policy"] = "changed_during_projection"
            atomic_json(config, payload)
            changed = True
        return result

    monkeypatch.setattr(api, "_project", mutate)
    try:
        report = api.prepare_joint_corpus(sources(parents), output)
    except ValueError:
        assert not (output / "report.json").exists()
        return
    assert report["configuration_sha256"] != sha256(output / "configuration.json")
    pytest.fail("Se publicó completed con una huella que no corresponde a configuration.json")


def test_completed_reuse_rechecks_sources_at_return(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    output = tmp_path / "joint"
    api.prepare_joint_corpus(sources(parents), output)
    original = api._checked
    changed = False

    def mutate(path, expected=None):
        nonlocal changed
        result = original(path, expected)
        if not changed and Path(path) == output / "encoded/manifest.json":
            source = parents["US"][0]
            payload = json.loads(source.read_text())
            payload["final_test_opened"] = True
            atomic_json(source, payload)
            changed = True
        return result

    monkeypatch.setattr(api, "_checked", mutate)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources(parents), output)


def test_partial_receipt_preserves_integer_types(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    output = tmp_path / "joint"
    original = api._project

    def interrupt(path, *args, **kwargs):
        if path == parents["CN"][1]["samples"]:
            raise InterruptedError("Corte")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(api, "_project", interrupt)
    with pytest.raises(InterruptedError):
        api.prepare_joint_corpus(sources(parents), output)
    receipt = output / "encoded/samples/US/SAME/projection.json"
    payload = json.loads(receipt.read_text())
    payload["asset"]["samples"] = 2.0
    atomic_json(receipt, payload)
    monkeypatch.setattr(api, "_project", original)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources(parents), output)


def test_invalid_vector_width_is_rejected_before_payload_read(tmp_path, monkeypatch):
    parents = pair(tmp_path)
    change_artifact(
        parents,
        "samples",
        lambda table: replacement(
            table, "news", pa.array([[1.0] * 385] * 2, type=pa.list_(pa.float32(), 385))
        ),
    )
    original = pq.ParquetFile

    class NoPayload:
        def __init__(self, path, *args, **kwargs):
            self.file = original(path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.file.close()

        def iter_batches(self, *args, **kwargs):
            pytest.fail(
                "Se inició la descompresión de una anchura inválida conocida por el esquema"
            )

    monkeypatch.setattr(api.pq, "ParquetFile", NoPayload)
    with pytest.raises(ValueError):
        api.prepare_joint_corpus(sources(parents), tmp_path / "joint")
