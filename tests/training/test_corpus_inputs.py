import importlib
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.storage import sha256


def module():
    try:
        return importlib.import_module("mars_titan.training.corpus_inputs")
    except ModuleNotFoundError:
        pytest.fail("Falta el lector supervisado del corpus")


def corpus(tmp_path, *, assets=2, rows=5, markets=("US",), group_size=3):
    roots = {name: tmp_path / name for name in ("prepared", "samples", "labels")}
    entries = []
    for market in markets:
        for number in range(assets):
            symbol = f"A{number:04}"
            folders = {name: root / market / symbol for name, root in roots.items()}
            for folder in folders.values():
                folder.mkdir(parents=True)
            prices = pa.table(
                {
                    "open": [10.0, 11.0, 12.0],
                    "high": [12.0, 13.0, 14.0],
                    "low": [9.0, 10.0, 11.0],
                    "close": [11.0, 12.0, 13.0],
                    "volume": [100.0, 120.0, 80.0],
                    "available_at": [datetime(2017, 1, day, tzinfo=UTC) for day in (1, 2, 3)],
                }
            )
            times = [datetime(2018, 1, 1, tzinfo=UTC) + timedelta(minutes=i) for i in range(rows)]
            samples = pa.table(
                {
                    "prediction_at": times,
                    "price_end_index": [2] * rows,
                    "news": [[float(i), 2.0] for i in range(rows)],
                    "charts": [[3.0]] * rows,
                    "fundamentals": [[4.0]] * rows,
                    "macro": [[5.0]] * rows,
                }
            )
            labels = pa.table(
                {
                    "sample_row": list(range(rows)),
                    "prediction_at": times,
                    "target_available_at": [t + timedelta(seconds=1) for t in times],
                    "target": [float(i) / 100 for i in range(rows)],
                    "partition": ["train"] * rows,
                }
            )
            paths = {
                name: folders[name] / file
                for name, file in (
                    ("prepared", "prices.parquet"),
                    ("samples", "samples.parquet"),
                    ("labels", "labels.parquet"),
                )
            }
            for name, table in (("prepared", prices), ("samples", samples), ("labels", labels)):
                pq.write_table(table, paths[name], row_group_size=group_size)
            entries.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "prices_sha256": sha256(paths["prepared"]),
                    "samples_sha256": sha256(paths["samples"]),
                    "labels_sha256": sha256(paths["labels"]),
                    "counts": {"train": rows, "validation": 0},
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "corpus_supervision",
                "context_sessions": 2,
                "roots": {k: str(v) for k, v in roots.items()},
                "assets": entries,
                "scope": "development_snapshot",
                "cohort_complete": False,
                "counts": {"train": len(entries) * rows, "validation": 0},
            }
        )
    )
    return manifest


def batches(manifest, **kwargs):
    return module().supervised_batches(
        manifest, partition="train", batch_size=4, epoch=0, seed=42, **kwargs
    )


def test_all_assets_and_last_partial_batch_are_visited_once(tmp_path):
    manifest = corpus(tmp_path, assets=130)
    observed = list(batches(manifest))
    ids = [key for batch in observed for key in batch["sample_ids"]]
    assert len(ids) == len(set(ids)) == 650
    assert len(observed[-1]["target"]) == 2
    assert set(observed[0]["inputs"]) == {"prices", "news", "charts", "fundamentals", "macro"}
    for batch in observed:
        np.testing.assert_allclose(batch["target"], batch["inputs"]["news"][:, 0] / 100)
        assert batch["inputs"]["prices"].shape[1:] == (2, 5)


def test_batches_preserve_actual_maturity_and_do_not_invent_missing_availability(tmp_path):
    manifest = corpus(tmp_path, assets=2, rows=5)
    for batch in batches(manifest):
        np.testing.assert_array_equal(
            batch["target_available_at"],
            batch["prediction_at"] + np.timedelta64(1, "s"),
        )
        assert np.isnat(batch["input_available_at"]).all()


def test_modality_availability_is_checked_and_retained_in_batches(tmp_path):
    manifest = corpus(tmp_path, assets=1, rows=5)
    metadata = json.loads(manifest.read_text())
    path = tmp_path / "samples/US/A0000/samples.parquet"
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["input_availability"] = {
            name: row["prediction_at"]
            for name in (
                "prices",
                "news",
                "charts",
                "fundamentals",
                "macro",
            )
        }
    pq.write_table(pa.Table.from_pylist(rows), path)
    metadata["assets"][0]["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(metadata))
    for batch in batches(manifest):
        np.testing.assert_array_equal(batch["input_available_at"], batch["prediction_at"])
    rows[0]["input_availability"]["news"] += timedelta(days=1)
    pq.write_table(pa.Table.from_pylist(rows), path)
    metadata["assets"][0]["samples_sha256"] = sha256(path)
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="disponibilidad|futuro"):
        list(batches(manifest))


def test_epoch_order_is_deterministic_and_changes_between_epochs(tmp_path):
    manifest = corpus(tmp_path, assets=3, rows=13)

    def order(epoch):
        return [
            s
            for b in module().supervised_batches(
                manifest, partition="train", batch_size=7, epoch=epoch, seed=42
            )
            for s in b["sample_ids"]
        ]

    assert order(0) == order(0)
    assert order(0) != order(1)
    assert sorted(order(0)) == sorted(order(1))


def test_confirmed_cursor_resumes_exact_remaining_order_despite_prefetch(tmp_path):
    manifest = corpus(tmp_path, assets=3, rows=9)
    uninterrupted = list(batches(manifest))
    iterator = iter(batches(manifest))
    first = next(iterator)
    next(iterator)
    remaining = list(batches(manifest, cursor=first["confirmed_cursor"]))
    assert [s for b in remaining for s in b["sample_ids"]] == [
        s for b in uninterrupted[1:] for s in b["sample_ids"]
    ]
    assert list(batches(manifest, cursor=uninterrupted[-1]["confirmed_cursor"])) == []


def test_market_and_symbol_form_the_identity(tmp_path):
    manifest = corpus(tmp_path, assets=1, markets=("US", "CN"))
    ids = [s for b in batches(manifest) for s in b["sample_ids"]]
    assert len(set(ids)) == 10
    assert {s.split("/")[0] for s in ids} == {"US", "CN"}


def test_dataset_identity_binds_the_same_manifest_bytes_it_parses(tmp_path, monkeypatch):
    manifest = corpus(tmp_path, assets=1)
    engine, original = module(), module().sha256
    changed = False

    def replace_after_hash(path):
        nonlocal changed
        digest = original(path)
        if path == manifest and not changed:
            source = json.loads(path.read_text())
            source["context_sessions"] = 3
            path.write_text(json.dumps(source))
            changed = True
        return digest

    monkeypatch.setattr(engine, "sha256", replace_after_hash)
    dataset = engine.CorpusDataset(manifest)
    assert dataset.identity == original(manifest)


def test_sparse_accepted_rows_do_not_retain_entire_parquet_groups(tmp_path):
    manifest = corpus(tmp_path, assets=1, rows=300, group_size=100)
    label_path = tmp_path / "labels/US/A0000/labels.parquet"
    rows = pq.read_table(label_path).to_pylist()
    for i, row in enumerate(rows):
        row["reason"] = "accepted" if i % 100 == 0 else "insufficient_history"
        if row["reason"] != "accepted":
            row.update(partition=None, target=None)
    pq.write_table(pa.Table.from_pylist(rows), label_path)
    metadata = json.loads(manifest.read_text())
    metadata["counts"]["train"] = 3
    metadata["assets"][0]["counts"]["train"] = 3
    metadata["assets"][0]["labels_sha256"] = sha256(label_path)
    manifest.write_text(json.dumps(metadata))
    dataset = module().CorpusDataset(manifest)
    observed = list(dataset.batches(partition="train", batch_size=2, epoch=0, seed=42))
    assert sum(len(batch["target"]) for batch in observed) == 3
    for batch in observed:
        for name in ("news", "charts", "fundamentals", "macro"):
            root = batch["inputs"][name]
            while isinstance(root.base, np.ndarray):
                root = root.base
            assert root.nbytes == batch["inputs"][name].nbytes


@pytest.mark.parametrize("batch_size", [1, 7, 129, 512])
def test_batches_own_contiguous_buffers_after_advancing_and_mutating_other_batches(
    tmp_path, batch_size
):
    manifest = corpus(tmp_path, assets=3, rows=257, group_size=100)
    observed = (
        module()
        .CorpusDataset(manifest)
        .batches(partition="train", batch_size=batch_size, epoch=2, seed=43)
    )
    first = next(observed)
    saved = {name: value.copy() for name, value in first["inputs"].items()}
    second = next(observed)
    for name, values in second["inputs"].items():
        assert values.flags.c_contiguous and values.flags.owndata
        assert not np.shares_memory(values, first["inputs"][name])
        values.fill(-999)
    last = second
    for batch in observed:
        last = batch
    for name, values in first["inputs"].items():
        np.testing.assert_array_equal(values, saved[name])
        assert last["inputs"][name].flags.owndata


def test_excluded_modalities_are_not_validated_as_admitted_samples(tmp_path):
    manifest = corpus(tmp_path, assets=1, rows=9, group_size=9)
    metadata = json.loads(manifest.read_text())
    sample_path = tmp_path / "samples/US/A0000/samples.parquet"
    samples = pq.read_table(sample_path).to_pylist()
    samples[1]["news"] = [np.nan, np.inf]
    pq.write_table(pa.Table.from_pylist(samples), sample_path, row_group_size=9)

    def exclude(rows):
        rows[1].update(reason="insufficient_history", partition=None, target=None)
        for index, row in enumerate(rows):
            if index != 1:
                row["reason"] = "accepted"
        return rows

    change_labels(manifest, exclude)
    metadata = json.loads(manifest.read_text())
    metadata["counts"]["train"] = metadata["assets"][0]["counts"]["train"] = 8
    metadata["assets"][0]["samples_sha256"] = sha256(sample_path)
    manifest.write_text(json.dumps(metadata))
    actual = list(batches(manifest))
    assert sum(len(batch["target"]) for batch in actual) == 8
    assert all(np.isfinite(batch["inputs"]["news"]).all() for batch in actual)


def test_invalid_later_batch_does_not_reject_or_mutate_the_preceding_batch(tmp_path):
    manifest = corpus(tmp_path, assets=1, rows=9, group_size=9)
    metadata = json.loads(manifest.read_text())
    for kind, folder in (("samples", "samples"), ("labels", "labels")):
        path = tmp_path / folder / "US/A0000" / f"{kind}.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row["prediction_at"] = row["prediction_at"].replace(year=2023)
            if kind == "labels":
                row["partition"] = "validation"
                row["target_available_at"] = row["target_available_at"].replace(year=2023)
        if kind == "samples":
            rows[5]["news"] = [np.nan, 2.0]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=9)
        metadata["assets"][0][kind + "_sha256"] = sha256(path)
    metadata["counts"] = metadata["assets"][0]["counts"] = dict(train=0, validation=9)
    manifest.write_text(json.dumps(metadata))
    iterator = (
        module()
        .CorpusDataset(manifest)
        .batches(partition="validation", batch_size=4, epoch=0, seed=42)
    )
    first = next(iterator)
    np.testing.assert_array_equal(first["target"], [0, 0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="finitos"):
        next(iterator)
    np.testing.assert_array_equal(first["inputs"]["news"][:, 0], [0, 1, 2, 3])


def test_modified_manifest_or_cursor_cannot_silently_restart(tmp_path):
    manifest = corpus(tmp_path)
    cursor = next(iter(batches(manifest)))["confirmed_cursor"]
    with pytest.raises(ValueError, match="cursor"):
        list(batches(manifest, cursor={**cursor, "seed": 9}))
    content = json.loads(manifest.read_text())
    content["scope"] = "changed"
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        list(batches(manifest, cursor=cursor))


def change_labels(manifest, transform):
    content = json.loads(manifest.read_text())
    entry = content["assets"][0]
    from pathlib import Path

    path = Path(content["roots"]["labels"]) / entry["market"] / entry["symbol"] / "labels.parquet"
    table = pq.read_table(path)
    pq.write_table(pa.Table.from_pylist(transform(table.to_pylist())), path)
    entry["labels_sha256"] = sha256(path)
    manifest.write_text(json.dumps(content))


def test_duplicate_label_keys_fail_instead_of_multiplying_observations(tmp_path):
    manifest = corpus(tmp_path)
    change_labels(manifest, lambda rows: [rows[0], rows[0], *rows[2:]])
    with pytest.raises(ValueError, match="duplicad"):
        list(batches(manifest))


def test_labels_crossing_the_training_boundary_are_rejected(tmp_path):
    manifest = corpus(tmp_path)

    def cross(rows):
        rows[0]["target_available_at"] = datetime(2023, 1, 1, tzinfo=UTC)
        return rows

    change_labels(manifest, cross)
    with pytest.raises(ValueError, match="partición"):
        list(batches(manifest))


def test_changed_artifact_is_detected_by_a_reused_dataset(tmp_path):
    manifest = corpus(tmp_path, assets=1)
    dataset = module().CorpusDataset(manifest)
    list(dataset.batches(partition="train", batch_size=4, epoch=0, seed=42))
    content = json.loads(manifest.read_text())
    from pathlib import Path

    sample = Path(content["roots"]["samples"]) / "US/A0000/samples.parquet"
    data = pq.read_table(sample).to_pylist()
    data[0]["news"] = [99.0, 2.0]
    pq.write_table(pa.Table.from_pylist(data), sample)
    with pytest.raises(ValueError, match="cambiado"):
        list(dataset.batches(partition="train", batch_size=4, epoch=1, seed=42))


def test_total_corpus_has_no_hundred_thousand_row_limit(tmp_path):
    manifest = corpus(tmp_path, assets=130, rows=800, group_size=256)
    count = sum(
        len(batch["target"])
        for batch in module().supervised_batches(
            manifest, partition="train", batch_size=31, epoch=0, seed=42
        )
    )
    assert count == 104_000


def test_missing_label_cannot_be_disguised_by_lowering_the_count(tmp_path):
    manifest = corpus(tmp_path, assets=1)
    change_labels(manifest, lambda rows: rows[1:])
    content = json.loads(manifest.read_text())
    content["assets"][0]["counts"]["train"] = content["counts"]["train"] = 4
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="etiqueta"):
        list(batches(manifest))


def test_explicit_exclusions_preserve_the_original_denominator(tmp_path):
    manifest = corpus(tmp_path, assets=1)

    def annotate(rows):
        for row in rows:
            row["reason"] = "accepted"
        rows[0].update(
            target=None, target_available_at=None, partition=None, reason="insufficient_history"
        )
        return rows

    change_labels(manifest, annotate)
    content = json.loads(manifest.read_text())
    content["assets"][0]["counts"]["train"] = content["counts"]["train"] = 4
    manifest.write_text(json.dumps(content))
    actual = list(batches(manifest))
    assert sum(len(b["target"]) for b in actual) == 4
    assert all(
        s.rsplit("/", 1)[-1] != str(int(datetime(2018, 1, 1, tzinfo=UTC).timestamp() * 1_000_000))
        for b in actual
        for s in b["sample_ids"]
    )


def test_price_window_matches_hand_calculated_values(tmp_path):
    manifest = corpus(tmp_path, assets=1)
    actual = next(iter(batches(manifest)))["inputs"]["prices"][0]
    expected = np.array(
        [
            [-0.087011377, 0.080042708, -0.182321557, 0.0, 0.788457360],
            [0.0, 0.154150680, -0.087011377, 0.080042708, 0.587786665],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7)
