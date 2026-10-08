"""Vistas temporales históricas comprobadas con datos técnicos y sin evaluación."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation.splits import build_folds
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.joint_temporal_corpus import prepare_joint_temporal_corpus
from mars_titan.training.temporal_contract import temporal_contracts
from mars_titan.training.temporal_corpus import prepare_temporal_corpus
from tests.training.historical_temporal_fixture import historical_temporal_fixture


def prepare(fixture, output, **kwargs):
    return prepare_temporal_corpus(
        fixture.parent,
        fixture.protocols["US"],
        None,
        None,
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
        **kwargs,
    )


def reader(path):
    return CorpusDataset(path, input_policy=HISTORICAL_MASKED)


def rows(dataset, partition):
    result = {}
    for batch in dataset.batches(partition=partition, batch_size=3, epoch=0, seed=42):
        for index, key in enumerate(batch["sample_ids"]):
            result[key] = (
                batch["target"][index],
                batch["presence"][index].copy(),
                {name: values[index].copy() for name, values in batch["inputs"].items()},
            )
    return result


def sample_day(key):
    return str(np.datetime64(int(key.rsplit("/", 1)[1]), "us"))[:10]


def test_masked_temporal_view_keeps_early_history_and_absent_macro(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    parent = reader(fixture.parent)
    original = rows(parent, "train") | rows(parent, "validation")
    output = tmp_path / "views"
    report = prepare(fixture, output)
    assert len(report["folds"]) == 10
    path = output / "fold-000/manifest.json"
    with pytest.raises(ValueError):
        CorpusDataset(path)
    viewed = reader(path)
    assert viewed.partitions == ("train", "validation", "calibration", "evaluation")
    assert viewed.manifest["temporal_view"]["selection_partition"] == "validation"
    assert viewed.manifest["temporal_view"]["recover_annual_boundaries"] is True
    assert viewed.manifest["temporal_view"]["protocol"]["train_start"] == "2000-01-01"
    actual = rows(viewed, "train")
    assert any(sample_day(key).startswith("2000-") for key in actual)
    assert actual and all("presence" not in values[2] for values in actual.values())
    assert any(not values[1][-1] for values in actual.values())
    for partition in viewed.partitions:
        for key, value in rows(viewed, partition).items():
            if key in original:
                assert value[0] == original[key][0]
                np.testing.assert_array_equal(value[1], original[key][1])
                for name in value[2]:
                    np.testing.assert_array_equal(value[2][name], original[key][2][name])
    table = pq.read_table(viewed.roots["labels"] / "US/A0000/labels.parquet")
    assert table.num_rows == fixture.metadata["samples"]
    reasons = table["reason"].to_pylist()
    assert reasons[0] == "insufficient_history"
    assert reasons[-1] == "target_after_cutoff"
    assert viewed.manifest["training_ready"] is False


def test_historical_policy_requires_explicit_annual_recovery(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    with pytest.raises(ValueError, match="anual|recuper"):
        prepare_temporal_corpus(
            fixture.parent,
            fixture.protocols["US"],
            None,
            None,
            tmp_path / "views",
            input_policy=HISTORICAL_MASKED,
        )
    assert not (tmp_path / "views").exists()


def test_annual_target_is_reused_in_a_later_train_and_purged_at_a_real_boundary(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    output = tmp_path / "views"
    prepare(fixture, output)
    viewed = reader(output / "fold-002/manifest.json")
    table = pq.read_table(viewed.roots["labels"] / "US/A0000/labels.parquet").to_pylist()
    annual = next(row for row in table if row["prediction_at"].date().isoformat() == "2022-12-30")
    assert annual["reason"] == "accepted" and annual["partition"] == "train"
    assert annual["source_reason"] == "target_crosses_partition_boundary"
    source = pq.read_table(
        Path(fixture.metadata["roots"]["labels"]) / "US/A0000/labels.parquet"
    ).to_pylist()
    original = next(row for row in source if row["prediction_at"] == annual["prediction_at"])
    assert annual["target"] == original["target"]
    assert annual["target_available_at"] == original["target_available_at"]
    config = json.loads(fixture.protocols["US"].read_text())
    config.update(first_validation_start="2023-01-01", gap_sessions=0)
    atomic_json(fixture.protocols["US"], config)
    prepare(fixture, tmp_path / "purged")
    rejected = pq.read_table(
        tmp_path / "purged/fold-000/labels/US/A0000/labels.parquet"
    ).to_pylist()
    annual = next(
        row for row in rejected if row["prediction_at"].date().isoformat() == "2022-12-30"
    )
    assert annual["reason"] == "label_crosses_boundary" and annual["partition"] is None
    assert annual["target"] is None


def test_calibration_cannot_be_declared_as_the_selection_partition(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    prepare(fixture, tmp_path / "views")
    path = tmp_path / "views/fold-000/manifest.json"
    meta = json.loads(path.read_text())
    meta["temporal_view"]["selection_partition"] = "calibration"
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="selección|contrato"):
        reader(path)


def test_joint_historical_views_keep_the_disjoint_union_and_local_holidays(tmp_path):
    fixture = historical_temporal_fixture(tmp_path, markets=("US", "CN"))
    output = tmp_path / "joint"
    report = prepare_joint_temporal_corpus(
        fixture.parent,
        {market: dict(protocol=path) for market, path in fixture.protocols.items()},
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    assert len(report["folds"]) == 10
    combined = reader(output / "fold-008/manifest.json")
    assert set(temporal_contracts(combined.manifest, input_policy=HISTORICAL_MASKED)) == {
        "US",
        "CN",
    }
    for partition in combined.partitions:
        expected = {}
        for market in ("US", "CN"):
            expected.update(
                rows(reader(output / f"markets/{market}/fold-008/manifest.json"), partition)
            )
        actual = rows(combined, partition)
        assert actual.keys() == expected.keys()
        for key in actual:
            assert actual[key][0] == expected[key][0]
            np.testing.assert_array_equal(actual[key][1], expected[key][1])
    calibration = rows(combined, "calibration")
    assert any(key.startswith("US/") and sample_day(key) == "2023-10-02" for key in calibration)
    assert not any(key.startswith("CN/") and sample_day(key) == "2023-10-02" for key in calibration)
    assert combined.manifest["candidate_count"] == 4


def test_historical_protocols_reuse_the_existing_matrix_except_train_start():
    for market, reference in [
        ("US", "real-expanded-walk-forward.json"),
        ("CN", "chinese-real-walk-forward.json"),
    ]:
        original = json.loads((Path("configs/evaluation") / reference).read_text())
        historical = json.loads(
            (
                Path("configs/evaluation") / f"historical-masked-{market.lower()}-walk-forward.json"
            ).read_text()
        )
        assert historical == dict(original, train_start="2000-01-01")
        assert len(build_folds(historical)) == len(build_folds(original)) == 10


def test_default_policy_rejects_a_historical_parent_without_optional_paths(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    with pytest.raises(ValueError):
        prepare_temporal_corpus(
            fixture.parent,
            fixture.protocols["US"],
            None,
            None,
            tmp_path / "views",
            recover_annual_boundaries=True,
        )
    assert not (tmp_path / "views").exists()


def test_historical_view_rejects_an_external_macro_or_admission(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    with pytest.raises(ValueError, match="macro|admisión"):
        prepare_temporal_corpus(
            fixture.parent,
            fixture.protocols["US"],
            tmp_path / "not-read",
            None,
            tmp_path / "views",
            input_policy=HISTORICAL_MASKED,
            recover_annual_boundaries=True,
        )
    assert not (tmp_path / "views").exists()


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", True), ("train_start", "2009-01-01"), ("final_test_start", "2025-01-01")],
)
def test_historical_protocol_cannot_change_its_version_or_reserved_cut(tmp_path, field, value):
    fixture = historical_temporal_fixture(tmp_path)
    protocol = json.loads(fixture.protocols["US"].read_text())
    protocol[field] = value
    atomic_json(fixture.protocols["US"], protocol)
    with pytest.raises(ValueError):
        prepare(fixture, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_changed_prices_before_confirmation_prevent_publication(tmp_path, monkeypatch):
    from mars_titan.training import temporal_corpus as module

    fixture = historical_temporal_fixture(tmp_path)
    path = Path(fixture.metadata["roots"]["prepared"]) / "US/A0000/prices.parquet"
    original = module.atomic_parquet

    def changed(destination, table):
        original(destination, table)
        path.write_bytes(path.read_bytes() + b"changed")

    monkeypatch.setattr(module, "atomic_parquet", changed)
    with pytest.raises(ValueError, match="artefacto|confirmación|huella"):
        prepare(fixture, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_historical_cursor_and_cache_preserve_masks_and_ids(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    output = tmp_path / "views"
    prepare(fixture, output)
    dataset = CorpusDataset(
        output / "fold-009/manifest.json",
        input_policy=HISTORICAL_MASKED,
        cache_bytes=1024**2,
        cache_sample_tables=True,
    )
    options = dict(partition="train", batch_size=2, epoch=3, seed=43)
    complete = list(dataset.batches(**options))
    assert len(complete) > 1
    resumed = list(dataset.batches(**options, cursor=complete[0]["confirmed_cursor"]))
    assert [key for batch in resumed for key in batch["sample_ids"]] == [
        key for batch in complete[1:] for key in batch["sample_ids"]
    ]
    for left, right in zip(resumed, complete[1:], strict=True):
        np.testing.assert_array_equal(left["presence"], right["presence"])
        np.testing.assert_array_equal(left["inputs"]["macro"], right["inputs"]["macro"])
    assert dataset.cached_bytes <= 1024**2


def test_published_historical_view_cannot_be_replaced_and_parent_change_is_detected(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    output = tmp_path / "views"
    prepare(fixture, output)
    path = output / "fold-000/manifest.json"
    digest = sha256(path), path.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        prepare(fixture, output)
    assert digest == (sha256(path), path.stat().st_mtime_ns)
    dataset = reader(path)
    fixture.parent.write_bytes(fixture.parent.read_bytes() + b" ")
    with pytest.raises(ValueError, match="fuente"):
        list(dataset.batches(partition="train", batch_size=3, epoch=0, seed=42))


def test_real_target_maturity_controls_the_purge(tmp_path):
    fixture = historical_temporal_fixture(tmp_path)
    path = Path(fixture.metadata["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(path)
    records = table.to_pylist()
    item = next(row for row in records if row["prediction_at"].date().isoformat() == "2022-11-15")
    item["target_available_at"] = fixture.clocks["US"].decision("2022-12-02")
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    meta = json.loads(fixture.parent.read_text())
    meta["assets"][0]["labels_sha256"] = sha256(path)
    atomic_json(fixture.parent, meta)
    prepare(fixture, tmp_path / "views")
    records = pq.read_table(tmp_path / "views/fold-000/labels/US/A0000/labels.parquet").to_pylist()
    item = next(row for row in records if row["prediction_at"].date().isoformat() == "2022-11-15")
    assert item["reason"] == "label_crosses_boundary"
    assert item["partition"] is None and item["target"] is None


def test_historical_views_do_not_read_complete_macro_admission(tmp_path, monkeypatch):
    from mars_titan.training import temporal_corpus as module

    fixture = historical_temporal_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("La edición histórica no requiere admisión macro completa")

    monkeypatch.setattr(module, "_complete_dates", forbidden)
    monkeypatch.setattr(module, "MacroVectors", forbidden)
    prepare(fixture, tmp_path / "views")
    assert rows(reader(tmp_path / "views/fold-000/manifest.json"), "train")


def test_temporal_view_cannot_relabel_the_parent_catalog_with_the_same_width(tmp_path):
    from mars_titan.training.cohort_contract import representation_hash

    fixture = historical_temporal_fixture(tmp_path)
    prepare(fixture, tmp_path / "views")
    path = tmp_path / "views/fold-000/manifest.json"
    meta = json.loads(path.read_text())
    meta["representation"]["macro_indicators"] = ["other_a", "other_b"]
    signature = representation_hash(meta["representation"], input_policy=HISTORICAL_MASKED)
    for asset in meta["assets"]:
        asset["representation_sha256"] = signature
    atomic_json(path, meta)
    with pytest.raises(ValueError, match="representación|catálogo"):
        reader(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("selection_partition", "calibration"),
        ("recover_annual_boundaries", False),
        ("schema_version", True),
    ],
)
def test_joint_historical_contracts_cannot_diverge(tmp_path, field, value):
    fixture = historical_temporal_fixture(tmp_path, markets=("US", "CN"))
    output = tmp_path / "views"
    prepare_joint_temporal_corpus(
        fixture.parent,
        {market: dict(protocol=path) for market, path in fixture.protocols.items()},
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    path = output / "fold-000/manifest.json"
    meta = json.loads(path.read_text())
    meta["temporal_views"]["CN"][field] = value
    atomic_json(path, meta)
    with pytest.raises(ValueError):
        reader(path)


def test_historical_row_budget_is_checked_before_reading_groups(tmp_path, monkeypatch):
    from mars_titan.training.temporal_corpus import _masked_sample_state

    fixture = historical_temporal_fixture(tmp_path)
    parent = reader(fixture.parent)
    original = pq.ParquetFile

    class Metadata:
        num_rows = 1_000_001

        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class Guard:
        def __init__(self, path, *args, **kwargs):
            self.file = original(path, *args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def __getattr__(self, name):
            return Metadata(self.file.metadata) if name == "metadata" else getattr(self.file, name)

        def read_row_group(self, *args, **kwargs):
            raise AssertionError("El presupuesto de filas debe comprobarse antes de leer grupos")

    def bounded(path, *args, **kwargs):
        return (
            Guard(path, *args, **kwargs)
            if Path(path).name == "samples.parquet"
            else original(path, *args, **kwargs)
        )

    monkeypatch.setattr(pq, "ParquetFile", bounded)
    with pytest.raises(ValueError, match="presupuesto"):
        _masked_sample_state(parent, parent.assets[0])


def test_strict_contract_keeps_its_original_field_order(tmp_path):
    from tests.training.test_temporal_corpus import inputs

    paths = inputs.__wrapped__(tmp_path)
    prepare_temporal_corpus(*paths, tmp_path / "views")
    meta = json.loads((tmp_path / "views/fold-000/manifest.json").read_text())
    assert list(meta["temporal_view"]) == [
        "schema_version",
        "protocol",
        "fold",
        "macro_path",
        "macro_sha256",
        "admission_path",
        "admission_sha256",
        "parent_manifest",
        "parent_sha256",
    ]


def test_joint_projection_keeps_a_market_without_valid_objectives(tmp_path):
    import pyarrow as pa

    fixture = historical_temporal_fixture(tmp_path, markets=("US", "CN"))
    meta = json.loads(fixture.parent.read_text())
    asset = next(item for item in meta["assets"] if item["market"] == "US")
    path = Path(meta["roots"]["labels"]) / "US/A0000/labels.parquet"
    table = pq.read_table(path)
    records = table.to_pylist()
    for row in records:
        row.update(
            target=None, target_available_at=None, partition=None, reason="insufficient_history"
        )
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    asset.update(labels_sha256=sha256(path), counts={"train": 0, "validation": 0})
    meta["counts"] = {
        part: sum(item["counts"][part] for item in meta["assets"])
        for part in ("train", "validation")
    }
    atomic_json(fixture.parent, meta)
    output = tmp_path / "views"
    result = prepare_joint_temporal_corpus(
        fixture.parent,
        {market: dict(protocol=path) for market, path in fixture.protocols.items()},
        output,
        input_policy=HISTORICAL_MASKED,
        recover_annual_boundaries=True,
    )
    combined = reader(output / "fold-000/manifest.json")
    assert {asset["market"] for asset in combined.assets} == {"US", "CN"}
    assert result["folds"][0]["market_counts"]["US"] == dict.fromkeys(combined.partitions, 0)
    assert not result["folds"][0]["has_all_partitions"]
    assert all(key.startswith("CN/") for key in rows(combined, "train"))
    assert combined.manifest["samples"] == fixture.metadata["samples"]


def test_joint_historical_publication_failure_leaves_no_partial_view(tmp_path, monkeypatch):
    from mars_titan.training import joint_temporal_corpus as module

    fixture = historical_temporal_fixture(tmp_path, markets=("US", "CN"))
    sources = {market: dict(protocol=path) for market, path in fixture.protocols.items()}
    output = tmp_path / "views"
    signature = sha256(fixture.parent)
    original = module._publish_directory

    def fail(*args):
        raise OSError("Corte técnico antes de confirmar la unión")

    monkeypatch.setattr(module, "_publish_directory", fail)
    with pytest.raises(OSError, match="Corte técnico"):
        prepare_joint_temporal_corpus(
            fixture.parent,
            sources,
            output,
            input_policy=HISTORICAL_MASKED,
            recover_annual_boundaries=True,
        )
    assert not output.exists() and sha256(fixture.parent) == signature
    monkeypatch.setattr(module, "_publish_directory", original)
    assert (
        prepare_joint_temporal_corpus(
            fixture.parent,
            sources,
            output,
            input_policy=HISTORICAL_MASKED,
            recover_annual_boundaries=True,
        )["status"]
        == "temporal_views_prepared"
    )


def test_historical_source_availability_is_checked_even_for_an_excluded_target(tmp_path):
    from datetime import timedelta

    import pyarrow as pa

    fixture = historical_temporal_fixture(tmp_path)
    meta = json.loads(fixture.parent.read_text())
    path = Path(meta["roots"]["samples"]) / "US/A0000/samples.parquet"
    table = pq.read_table(path)
    records = table.to_pylist()
    records[0]["input_availability"]["prices"] += timedelta(seconds=1)
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), path)
    meta["assets"][0]["samples_sha256"] = sha256(path)
    atomic_json(fixture.parent, meta)
    with pytest.raises(ValueError, match="disponibilidad"):
        prepare(fixture, tmp_path / "views")
    assert not (tmp_path / "views").exists()


def test_historical_cli_does_not_require_complete_macro_files(tmp_path, capsys):
    from mars_titan.training.temporal_corpus import main

    fixture = historical_temporal_fixture(tmp_path)
    assert (
        main(
            [
                "--parent",
                str(fixture.parent),
                "--protocol",
                str(fixture.protocols["US"]),
                "--output",
                str(tmp_path / "views"),
                "--input-policy",
                HISTORICAL_MASKED,
                "--recover-annual-boundaries",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["input_policy"] == HISTORICAL_MASKED
    assert report["scientific_training_started"] is False


def test_strict_cli_still_requires_macro_and_admission(tmp_path):
    from mars_titan.training.temporal_corpus import main

    with pytest.raises(SystemExit) as error:
        main(["--parent", "missing", "--protocol", "missing", "--output", str(tmp_path / "views")])
    assert error.value.code == 2
    assert not (tmp_path / "views").exists()


def test_joint_historical_cli_uses_only_each_market_protocol(tmp_path, capsys):
    from mars_titan.training.joint_temporal_corpus import main

    fixture = historical_temporal_fixture(tmp_path, markets=("US", "CN"))
    args = [
        "--parent",
        str(fixture.parent),
        "--output",
        str(tmp_path / "views"),
        "--us-protocol",
        str(fixture.protocols["US"]),
        "--cn-protocol",
        str(fixture.protocols["CN"]),
        "--input-policy",
        HISTORICAL_MASKED,
        "--recover-annual-boundaries",
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["input_policy"] == HISTORICAL_MASKED
    assert set(report["sources"]) == {"US", "CN"}
    assert reader(tmp_path / "views/fold-000/manifest.json").manifest["candidate_count"] == 4


def test_joint_historical_cli_rejects_unused_macro_arguments(tmp_path):
    from mars_titan.training.joint_temporal_corpus import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--parent",
                "missing",
                "--output",
                str(tmp_path / "views"),
                "--us-protocol",
                "missing",
                "--cn-protocol",
                "missing",
                "--us-macro",
                "unused",
                "--input-policy",
                HISTORICAL_MASKED,
                "--recover-annual-boundaries",
            ]
        )
    assert error.value.code == 2
    assert not (tmp_path / "views").exists()


def test_temporal_producer_validates_float32_before_mask_conversion(tmp_path):
    import pyarrow as pa

    fixture = historical_temporal_fixture(tmp_path)
    meta = json.loads(fixture.parent.read_text())
    path = Path(meta["roots"]["samples"]) / "US/A0000/samples.parquet"
    table = pq.read_table(path)
    records = table.to_pylist()
    for row in records:
        row["presence"][3] = True
        row["input_availability"]["fundamentals"] = row["prediction_at"]
    table = pa.Table.from_pylist(records, schema=table.schema)
    table = table.set_column(
        table.schema.get_field_index("fundamentals"),
        "fundamentals",
        pa.array([[0.0, 1.00000001, 0.0]] * len(table), type=pa.list_(pa.float64())),
    )
    pq.write_table(table, path)
    meta["assets"][0]["samples_sha256"] = sha256(path)
    atomic_json(fixture.parent, meta)
    with pytest.raises(ValueError, match="float32"):
        prepare(fixture, tmp_path / "views")
    assert not (tmp_path / "views").exists()
