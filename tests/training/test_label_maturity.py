"""Maduración de las etiquetas que una vista expone al predictor, sobre una vista mínima.

La vista tiene dos activos con etiquetas de todos los tramos y un tercero sin etiquetas de
ajuste. Las comprobaciones calculan el máximo por su cuenta y cada defensa del lector (huella,
enlaces, salida de la raíz, recuento y tramos admitidos) se rompe una vez.
"""

import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.training.label_maturity import (
    CALIBRATION_PARTITIONS,
    FIT_PARTITIONS,
    label_maturity,
    window_labels_until,
)

UTC = pa.timestamp("us", tz="UTC")


def micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


# La evaluación madura más tarde que todo lo demás y la fila purgada (sin tramo) aún más,
# así que un lector que no filtre por tramo devuelve otro máximo.
ROWS = {
    "AAA": [
        ("train", "2020-01-03"),
        ("train", "2020-02-03"),
        ("validation", "2020-03-02"),
        ("calibration", "2020-04-01"),
        ("evaluation", "2020-06-01"),
        (None, "2020-09-01"),
    ],
    "BBB": [
        ("train", "2020-01-10"),
        ("validation", "2020-03-16"),
        ("calibration", "2020-04-20"),
        ("evaluation", "2020-07-01"),
    ],
    "CCC": [("evaluation", "2020-08-03"), (None, "2020-10-01")],
}


def write_labels(path, rows, timestamp=UTC):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "partition": pa.array([name for name, _ in rows], pa.string()),
            "target_available_at": pa.array([micros(day) for _, day in rows], pa.int64()).cast(
                timestamp
            ),
        }
    )
    pq.write_table(table, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_view(root, rows=ROWS):
    labels = root / "labels"
    assets = []
    for symbol, values in rows.items():
        digest = write_labels(labels / "US" / symbol / "labels.parquet", values)
        counts = {name: 0 for name in (*FIT_PARTITIONS, "evaluation")}
        for name, _ in values:
            if name is not None:
                counts[name] += 1
        assets.append(dict(market="US", symbol=symbol, counts=counts, labels_sha256=digest))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(roots=dict(labels=str(labels)), assets=assets)))
    return manifest


def edit_manifest(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


@pytest.fixture
def view(tmp_path):
    return write_view(tmp_path / "view")


def test_the_maturity_is_the_latest_label_of_the_requested_partitions(view):
    assert label_maturity(view, FIT_PARTITIONS) == (micros("2020-04-20"), 7)
    assert label_maturity(view, ("train",)) == (micros("2020-02-03"), 3)
    assert label_maturity(view, ("train", "validation")) == (micros("2020-03-16"), 5)
    assert label_maturity(view, CALIBRATION_PARTITIONS) == (micros("2020-04-20"), 2)


def test_moving_one_label_one_microsecond_moves_the_limit(tmp_path):
    rows = {key: list(value) for key, value in ROWS.items()}
    view = write_view(tmp_path / "base", rows)
    base, _ = label_maturity(view, FIT_PARTITIONS)
    later = dict(rows, BBB=[*rows["BBB"][:2], ("calibration", "2020-04-20T00:00:00.000001")])
    moved = write_view(tmp_path / "moved", later)
    assert label_maturity(moved, FIT_PARTITIONS)[0] == base + 1
    # Una etiqueta de ajuste que madura en la evaluación pasa a ser el límite.
    leaked = dict(rows, AAA=[("train", "2020-06-02"), *rows["AAA"][1:]])
    assert label_maturity(write_view(tmp_path / "leak", leaked), FIT_PARTITIONS)[0] == micros(
        "2020-06-02"
    )


def test_an_asset_without_requested_labels_is_not_read(view):
    (view.parent / "labels/US/CCC/labels.parquet").unlink()
    assert label_maturity(view, FIT_PARTITIONS) == (micros("2020-04-20"), 7)
    # Si el manifiesto le atribuye una etiqueta de ajuste, el archivo ausente se rechaza.
    edit_manifest(view, lambda value: value["assets"][2]["counts"].update(train=1))
    with pytest.raises(ValueError, match="CCC no son las de la vista"):
        label_maturity(view, FIT_PARTITIONS)


def test_a_changed_labels_file_is_rejected(view):
    path = view.parent / "labels/US/AAA/labels.parquet"
    write_labels(path, [("train", "2020-01-02"), *ROWS["AAA"][1:]])
    with pytest.raises(ValueError, match="AAA no son las de la vista"):
        label_maturity(view, FIT_PARTITIONS)


def test_a_linked_labels_file_is_rejected_even_with_its_fingerprint(view):
    # El destino está dentro de la raíz y tiene la huella declarada.
    path = view.parent / "labels/US/AAA/labels.parquet"
    os.replace(path, path.with_name("copy.parquet"))
    path.symlink_to("copy.parquet")
    with pytest.raises(ValueError, match="AAA no son las de la vista"):
        label_maturity(view, FIT_PARTITIONS)


def test_a_folder_that_leaves_the_labels_root_is_rejected(view, tmp_path):
    folder = view.parent / "labels/US/BBB"
    outside = tmp_path / "elsewhere"
    os.replace(folder, outside)
    folder.symlink_to(outside, target_is_directory=True)
    assert (folder / "labels.parquet").is_file()
    with pytest.raises(ValueError, match="BBB no son las de la vista"):
        label_maturity(view, FIT_PARTITIONS)


@pytest.mark.parametrize("partition", ["train", "validation", "calibration"])
def test_counts_that_do_not_reconcile_with_the_view_are_rejected(view, partition):
    edit_manifest(view, lambda value: value["assets"][1]["counts"].update({partition: 2}))
    with pytest.raises(ValueError, match="BBB no concilian con la vista"):
        label_maturity(view, FIT_PARTITIONS)


@pytest.mark.parametrize("partitions", [(), ("evaluation",), ("train", "evaluation"), ("test",)])
def test_only_fit_selection_or_calibration_labels_can_be_read(view, partitions):
    with pytest.raises(ValueError, match="Solo se leen etiquetas"):
        label_maturity(view, partitions)


def test_a_view_without_labels_in_the_partitions_is_rejected(tmp_path):
    view = write_view(tmp_path / "view", {"CCC": ROWS["CCC"]})
    with pytest.raises(ValueError, match="no tiene etiquetas en los tramos"):
        label_maturity(view, FIT_PARTITIONS)


def test_maturities_without_a_time_zone_are_rejected(tmp_path):
    view = tmp_path / "view"
    digest = write_labels(
        view / "labels/US/AAA/labels.parquet", ROWS["AAA"][:2], timestamp=pa.timestamp("us")
    )
    counts = dict(train=2, validation=0, calibration=0, evaluation=0)
    asset = dict(market="US", symbol="AAA", counts=counts, labels_sha256=digest)
    (view / "manifest.json").write_text(
        json.dumps(dict(roots=dict(labels=str(view / "labels")), assets=[asset]))
    )
    with pytest.raises(ValueError, match="necesitan una zona"):
        label_maturity(view / "manifest.json", FIT_PARTITIONS)


def test_a_fitted_window_reads_only_its_own_fit_partitions(view):
    calls = []

    def recorded(path, partitions):
        calls.append((path, partitions))
        return label_maturity(path, partitions)

    assert window_labels_until(view, view, recorded) == micros("2020-04-20")
    assert calls == [(view, FIT_PARTITIONS)]


def test_a_carried_window_adds_its_own_calibration_to_the_anchor(tmp_path):
    anchor = write_view(tmp_path / "anchor")
    later = {
        "AAA": [("calibration", "2020-05-11"), ("evaluation", "2020-07-01")],
        "BBB": [("train", "2020-06-01"), ("validation", "2020-06-15")],
    }
    view = write_view(tmp_path / "carried", later)
    calls = []

    def recorded(path, partitions):
        calls.append((path, partitions))
        return label_maturity(path, partitions)

    # El ajuste de la vista trasladada no cuenta: sus parámetros se fijaron en el ancla.
    assert window_labels_until(anchor, view, recorded) == micros("2020-05-11")
    assert calls == [(anchor, FIT_PARTITIONS), (view, CALIBRATION_PARTITIONS)]
