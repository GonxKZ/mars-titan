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
    point = dict(asset=0, group=0, offset=0, consumed=0)
    observed = list(dataset._rows("train", 0, 42, point))
    assert len(observed) == 3
    for inputs, *_ in observed:
        for name in ("news", "charts", "fundamentals", "macro"):
            root = inputs[name]
            while isinstance(root.base, np.ndarray):
                root = root.base
            assert root.nbytes == inputs[name].nbytes


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
