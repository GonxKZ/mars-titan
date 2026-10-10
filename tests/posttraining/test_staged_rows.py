"""Filas nuevas del walk-forward por etapas frente a las del padre, sin ajustar nada.

Las vistas son las dos ventanas US de la campaña reducida (`campaign_fixture`), con dos
activos. Las comprobaciones leen las etiquetas de las vistas por su cuenta y comparan
recuentos, huellas, madurez e intersecciones con las de `staged_rows`. Cada frontera se
muta una vez y la prueba exige que la disjunción falle.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.environments.actions import ActionGrid
from mars_titan.posttraining import matrix_runs, staged_rows
from mars_titan.training import masked_campaign as engine
from mars_titan.training.campaign_plan import load_campaign
from mars_titan.training.corpus_inputs import CorpusDataset
from tests.posttraining.campaign_fixture import write_configs
from tests.training.historical_temporal_fixture import historical_temporal_fixture
from tests.training.test_walk_forward_v2_views import sessions

PARTITIONS = ("train", "validation", "calibration")


def micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


@pytest.fixture(scope="module")
def views(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged-rows")
    campaign, _ = write_configs(root / "config", "A")
    data = historical_temporal_fixture(
        root / "data", markets=("US",), days={"US": sessions("US")}, assets=2
    )
    prepared = engine.prepare_views(campaign, data.parent, root / "views")
    windows = prepared["US"]["windows"]
    folds = load_campaign(campaign)["comparison_config"]["resolved_scopes"]["US"]["windows"]
    datasets = {
        name: CorpusDataset(windows[name]["path"], input_policy=HISTORICAL_MASKED)
        for name in ("fold-000", "fold-001")
    }
    return SimpleNamespace(root=root, windows=windows, folds=folds, datasets=datasets)


def labels(dataset):
    """Etiquetas aceptadas de la vista leídas directamente de sus archivos."""
    root = dataset.manifest["roots"]["labels"]
    result = []
    for asset in dataset.assets:
        path = f"{root}/{asset['market']}/{asset['symbol']}/labels.parquet"
        table = pq.read_table(path).filter(pa.compute.field("partition").is_valid())
        result.append(
            dict(
                key=(asset["market"], asset["symbol"]),
                rows=table["sample_row"].to_numpy(),
                decision=table["prediction_at"].cast(pa.int64()).to_numpy(),
                maturity=table["target_available_at"].cast(pa.int64()).to_numpy(),
                partition=np.asarray(table["partition"].to_pylist(), dtype=object),
                target=table["target"].to_numpy(),
            )
        )
    return result


def proof(views):
    return staged_rows.fit_rows_proof(
        views.datasets["fold-000"],
        views.datasets["fold-001"],
        parent_fold=views.folds["fold-000"],
        fold=views.folds["fold-001"],
    )


def test_new_rows_start_at_the_end_of_the_parent_calibration(views):
    parent, fold = views.folds["fold-000"], views.folds["fold-001"]
    assert staged_rows.posttraining_rows(parent, fold) == ("2022-01-01", "2022-04-01")
    assert parent["calibration"][1] == "2022-01-01" and fold["train"][1] == "2022-04-01"
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        staged_rows.posttraining_rows(fold, parent)
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        staged_rows.posttraining_rows(fold, fold)
    late = dict(parent, calibration=[parent["calibration"][0], "2022-05-01"])
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        staged_rows.posttraining_rows(late, fold)


def test_the_proof_matches_an_independent_reading_of_both_views(views):
    result = proof(views)
    since, until = micros("2022-01-01"), micros("2022-04-01")
    parent, current = labels(views.datasets["fold-000"]), labels(views.datasets["fold-001"])
    used = {
        name: sum(int((item["partition"] == name).sum()) for item in parent) for name in PARTITIONS
    }
    assert result["parent_rows"] == used
    assert used == {name: views.windows["fold-000"]["counts"][name] for name in PARTITIONS}
    fresh = {}
    for item in current:
        mask = (item["partition"] == "train") & (item["decision"] >= since)
        mask &= item["decision"] < until
        if mask.any():
            fresh[item["key"]] = (item["rows"][mask], item["decision"][mask])
    assert len(fresh) == 2
    rows = sum(len(value[0]) for value in fresh.values())
    assert result["rows"] == rows > 0
    assert result["intersection"] == dict(train=0, validation=0, calibration=0)
    decisions = np.concatenate([value[1] for value in fresh.values()])
    assert (result["first_decision"], result["last_decision"]) == (
        int(decisions.min()),
        int(decisions.max()),
    )
    # La huella del plan de la campaña: activo, número de filas y huella de sus filas.
    digest = hashlib.sha256(b"mars-titan-chain-rows-v1")
    for (market, symbol), (positions, _) in sorted(fresh.items()):
        inner = hashlib.sha256(np.sort(positions).astype("<i8").tobytes()).hexdigest()
        digest.update(f"{market}/{symbol}:{len(positions)}:{inner}\n".encode())
    assert result["sha256"] == digest.hexdigest()
    used_maturity = max(
        int(item["maturity"][np.isin(item["partition"], PARTITIONS)].max()) for item in parent
    )
    window_maturity = max(
        int(item["maturity"][np.isin(item["partition"], PARTITIONS)].max()) for item in current
    )
    assert result["parent_labels_mature_until"] == used_maturity < since
    assert result["labels_used_until"] == max(used_maturity, window_maturity)
    assert result["labels_used_until"] < micros(views.folds["fold-001"]["evaluation"][0])
    assert staged_rows.labels_used_until(views.datasets["fold-001"]) == window_maturity
    assert staged_rows.labels_used_until(views.datasets["fold-000"]) == used_maturity


@pytest.mark.parametrize("start", ["2021-10-01", "2021-04-01", "2021-01-01"])
def test_moving_the_start_into_the_parent_rows_is_rejected(views, monkeypatch, start):
    original = staged_rows.posttraining_rows

    def earlier(parent_fold, fold):
        _, end = original(parent_fold, fold)
        return start, end

    monkeypatch.setattr(staged_rows, "posttraining_rows", earlier)
    with pytest.raises(ValueError, match="ya la usó el padre"):
        proof(views)
    # Sin la comprobación de la intersección, el recuento delata la fila compartida.
    monkeypatch.setattr(np, "isin", lambda *args, **kw: np.zeros(len(args[0]), dtype=bool))
    with pytest.raises(ValueError, match="madura después"):
        proof(views)


def test_swapped_views_or_changed_samples_are_rejected(views, monkeypatch):
    with pytest.raises(ValueError, match="no tiene filas nuevas"):
        staged_rows.fit_rows_proof(
            views.datasets["fold-001"],
            views.datasets["fold-000"],
            parent_fold=views.folds["fold-001"],
            fold=views.folds["fold-000"],
        )
    original = staged_rows.partition_rows

    def other_samples(dataset, partitions):
        rows, samples = original(dataset, partitions)
        if dataset is views.datasets["fold-000"]:
            samples = {key: "0" * 64 for key in samples}
        return rows, samples

    monkeypatch.setattr(staged_rows, "partition_rows", other_samples)
    with pytest.raises(ValueError, match="archivo de muestras"):
        proof(views)


def test_row_fingerprint_ignores_order_and_rejects_repeated_rows():
    markets = np.array(["US", "US", "CN", "US"])
    symbols = np.array(["B", "A", "X", "A"])
    rows = np.array([5, 9, 1, 3])
    order = np.array([3, 0, 2, 1])
    expected = staged_rows.row_fingerprint(markets, symbols, rows)
    assert staged_rows.row_fingerprint(markets[order], symbols[order], rows[order]) == expected
    assert expected[0] == 4
    parts = {
        ("CN", "X"): staged_rows.asset_digest([1]),
        ("US", "A"): staged_rows.asset_digest([3, 9]),
        ("US", "B"): staged_rows.asset_digest([5]),
    }
    assert staged_rows.combine(parts) == expected
    assert staged_rows.row_fingerprint(markets, symbols, np.array([5, 9, 1, 4])) != expected
    with pytest.raises(ValueError, match="sin repetir"):
        staged_rows.row_fingerprint(markets, np.array(["A", "A", "X", "A"]), np.array([3, 9, 1, 3]))


def test_the_window_fits_its_grid_only_with_the_new_rows(views, tmp_path):
    since = micros("2022-01-01")
    window = matrix_runs.MatrixWindow(
        views.windows["fold-001"]["path"],
        tmp_path / "window",
        encoding=views.windows["fold-001"]["sha256"],
        input_policy=HISTORICAL_MASKED,
        batch_size=8,
        stop=SimpleNamespace(requested=False),
        max_block_bytes=64 * 1024**2,
        since=since,
    )
    with window:
        assert all(at >= since for at, _ in window.train.index)
        targets = np.concatenate(
            [
                item["target"][(item["partition"] == "train") & (item["decision"] >= since)]
                for item in labels(views.datasets["fold-001"])
            ]
        )
        expected = ActionGrid.fit(
            targets, source_sha256=window.source_sha256, partition="train"
        ).to_dict()
        assert window.grid.to_dict() == expected
        assert window.grid.training_samples == proof(views)["rows"]
        # La población del índice sigue siendo la de la vista entera.
        manifest = json.loads(window.manifest.read_text())
        assert manifest["counts"] == views.windows["fold-001"]["counts"]
    with pytest.raises(ValueError, match="por bloques"):
        matrix_runs.MatrixWindow(
            views.windows["fold-001"]["path"],
            tmp_path / "ordered",
            encoding=views.windows["fold-001"]["sha256"],
            input_policy=HISTORICAL_MASKED,
            batch_size=8,
            stop=SimpleNamespace(requested=False),
            since=since,
        )
