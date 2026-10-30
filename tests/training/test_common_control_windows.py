"""Ventanas de la regla y de LightGBM de la criba (#746) sobre la vista corta del fixture.

La regla no ajusta nada. LightGBM se sustituye por un ajuste doble cuya predicción es una
función fija de las características y de las rondas, así que ninguna prueba estima parámetros.
"""

import json
import shutil
import types

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.environments.cohorts import FINAL_TEST_START_US
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import ForecastPanel
from mars_titan.models.baselines import momentum_reversal
from mars_titan.models.baselines import same_inputs_lightgbm as lightgbm_control
from mars_titan.training import common_control_windows as windows
from mars_titan.training import selection
from mars_titan.training.candidate_walk_forward import view_rows
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.walk_forward_phases import PREDICTED, window_phases
from mars_titan.training.window_indices import phase_sources
from tests.models.test_momentum_reversal import CONFIG as momentum_rule_config
from tests.models.test_same_inputs_lightgbm import declared as lightgbm_config
from tests.training.test_titans_walk_forward import views

RULE = momentum_rule_config
TABULAR = lightgbm_config()
COLUMNS = ["sample_id", "asset_id", "market", "prediction_at", "target", "prediction", "zero"]


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    # Las vistas de la criba son de la v3.1, con ventanas del calendario y su canal de
    # presencia, que son las que leen la regla y LightGBM.
    manifest, protocol = views(tmp_path_factory.mktemp("common-controls"), price_window=True)
    return manifest, json.loads(protocol.read_text())


def dataset(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def reference_inputs(data, partition):
    """Entradas de cada fila leídas con otro tamaño de tanda, por activo e instante."""
    found = {}
    for batch in data.batches(partition=partition, batch_size=5, epoch=0, seed=0):
        moments = batch["prediction_at"].astype("datetime64[us]").astype(np.int64)
        for row, sample in enumerate(batch["sample_ids"]):
            inputs = {name: values[row : row + 1] for name, values in batch["inputs"].items()}
            key = f"{sample.rsplit('/', 1)[0]}/{moments[row]}"
            found[key] = (
                batch["inputs"]["prices"][row],
                lightgbm_control.features(inputs, batch["presence"][row : row + 1])[0],
            )
    return found


def test_the_rows_of_each_partition_are_those_of_the_view_in_canonical_order(view):
    manifest, _ = view
    data = dataset(manifest)
    for name in PREDICTED:
        rows = windows.partition_rows(data, name, with_features=True)
        expected = view_rows(data, name)
        assert rows.panel.rows == len(expected["target"]) == data.manifest["counts"][name]
        keys = zip(expected["asset_id"], expected["prediction_at"].tolist(), strict=True)
        assert set(zip(rows.asset_id, rows.panel.prediction_at.tolist(), strict=True)) == set(keys)
        assert np.all(np.diff(rows.panel.prediction_at) >= 0)
        assert np.all(rows.panel.prediction == 0.0)
        found = reference_inputs(data, name)
        for row, sample in enumerate(rows.panel.row_id.to_pylist()):
            assert sample == f"{rows.asset_id[row]}/{rows.panel.prediction_at[row]}"
            np.testing.assert_array_equal(rows.prices[row], found[sample][0])
            np.testing.assert_array_equal(rows.features[row], found[sample][1])
        assert windows.partition_rows(data, name, with_features=False).features is None


def test_the_fit_rows_are_the_train_decisions_of_the_fit_months(view, tmp_path):
    manifest, protocol = view
    data = dataset(manifest)
    fold = windows._fold(data, protocol, "fold-000")
    phase = window_phases(fold, 0, 1)["train"]
    rows = windows.partition_rows(data, "train", with_features=False, since=phase.decision_start)
    decisions = phase_sources(data, {"train": phase}, tmp_path / "indices")[
        "train"
    ].label_decisions()
    expected = {(asset, int(at)) for asset, moments in decisions.items() for at in moments}
    assert set(zip(rows.asset_id, rows.panel.prediction_at.tolist(), strict=True)) == expected
    assert (
        rows.panel.rows == len(expected) and rows.panel.prediction_at.min() >= phase.decision_start
    )
    every = windows.partition_rows(data, "train", with_features=False)
    assert every.panel.rows == data.manifest["counts"]["train"] > rows.panel.rows
    # Las filas decididas justo en el instante inicial también entran.
    earliest = int(every.panel.prediction_at.min())
    since_earliest = windows.partition_rows(data, "train", with_features=False, since=earliest)
    assert since_earliest.panel.rows == every.panel.rows
    with pytest.raises(ValueError, match="no tiene filas"):
        windows.partition_rows(data, "train", with_features=False, since=2**62)


def test_a_view_outside_its_window_is_refused(view):
    manifest, protocol = view
    data = dataset(manifest)
    with pytest.raises(ValueError, match="no pertenece al protocolo"):
        windows._fold(data, protocol, "fold-009")
    # Otras semillas no cambian las ventanas, pero sí el protocolo que conserva la vista.
    with pytest.raises(ValueError, match="no conserva el protocolo"):
        windows._fold(data, dict(protocol, seeds=[42, 43]), "fold-000")
    opened = types.SimpleNamespace(manifest=dict(data.manifest, final_test_opened=True))
    with pytest.raises(ValueError, match="abre el test final"):
        windows._fold(opened, protocol, "fold-000")


def test_a_window_never_writes_the_final_test_or_a_non_finite_prediction(tmp_path):
    times = np.array([FINAL_TEST_START_US - 1, FINAL_TEST_START_US], dtype=np.int64)
    panel = ForecastPanel.from_columns(
        pa.array([f"US/A/{at}" for at in times]),
        np.array(["US", "US"]),
        times,
        np.zeros(2),
        np.zeros(2),
    )
    rows = windows.Rows(panel=panel, asset_id=np.array(["US/A"] * 2), prices=None, features=None)
    with pytest.raises(ValueError, match="no puede predecir el test final"):
        windows._write(tmp_path, "evaluation", rows, np.zeros(2), None)
    with pytest.raises(ValueError, match="valor finito por fila"):
        windows._write(tmp_path, "evaluation", rows, np.array([0.0, np.nan]), None)
    with pytest.raises(ValueError, match="valor finito por fila"):
        windows._write(tmp_path, "evaluation", rows, np.zeros(3), None)
    assert not any(tmp_path.iterdir())


def predictions(output, report):
    return {
        name: pq.read_table(output / record["path"])
        for name, record in report["predictions"].items()
    }


def test_the_rule_window_writes_the_rule_on_the_rows_of_the_view(view, tmp_path, learning_doubles):
    manifest, protocol = view
    output = tmp_path / "rule"
    report = windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)
    assert report["status"] == "completed" and report["kind"] == windows.RULE_KIND
    assert report["request"]["config"] == RULE and report["final_test_opened"] is False
    data = dataset(manifest)
    for name, table in predictions(output, report).items():
        record = report["predictions"][name]
        assert (
            sha256(output / record["path"]) == record["sha256"] and record["rows"] == table.num_rows
        )
        assert table.column_names == COLUMNS
        rows = windows.partition_rows(data, name, with_features=False)
        expected = momentum_reversal.predict(rows.prices, rows.panel.session, RULE)
        np.testing.assert_array_equal(table["prediction"].to_numpy(), expected)
        assert table["sample_id"].to_pylist() == rows.panel.row_id.to_pylist()
        assert ForecastPanel.from_arrow(table).cohort_sha256 == rows.panel.cohort_sha256
        assert np.all(table["zero"].to_numpy() == 0.0)
    assert json.loads((output / "run.json").read_text()) == report


def test_a_completed_rule_window_is_verified_and_not_repeated(
    view, tmp_path, learning_doubles, monkeypatch
):
    manifest, protocol = view
    output = tmp_path / "rule"
    report = windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)

    def unread(*_, **__):
        raise AssertionError("Una ventana terminada no vuelve a leer la vista")

    monkeypatch.setattr(windows, "CorpusDataset", unread)
    assert (
        windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)
        == report
    )
    changed = dict(RULE, reversal_sessions=3)
    with pytest.raises(ValueError, match="La petición de la ventana ha cambiado"):
        windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=changed)
    path = output / report["predictions"]["evaluation"]["path"]
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="Ha cambiado evaluation-predictions.parquet"):
        windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)


def test_a_paused_rule_window_resumes_and_foreign_outputs_are_refused(
    view, tmp_path, learning_doubles
):
    manifest, protocol = view
    output = tmp_path / "rule"
    stop = types.SimpleNamespace(requested=True)
    paused = windows.run_rule_window(
        manifest, protocol, "fold-000", output=output, config=RULE, stop=stop
    )
    assert paused["status"] == "paused" and "predictions" not in paused
    resumed = windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)
    assert (
        resumed["status"] == "completed" and resumed["started_at_utc"] == paused["started_at_utc"]
    )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="datos de otra ejecución"):
        windows.run_rule_window(manifest, protocol, "fold-000", output=foreign, config=RULE)


class Booster:
    """Predicción fija que gira con las rondas: cambia de dirección y con ella el IC."""

    def __init__(self, seed, width):
        generator = np.random.default_rng(seed)
        self.first, self.second = generator.normal(size=(2, width))

    def predict(self, values, num_iteration=None):
        angle = num_iteration / 1000 * np.pi
        return values @ (np.cos(angle) * self.first + np.sin(angle) * self.second)

    def model_to_string(self, num_iteration=None):
        assert num_iteration == -1
        return json.dumps(dict(first=self.first.tolist(), second=self.second.tolist()))

    @classmethod
    def from_text(cls, text):
        booster, values = cls.__new__(cls), json.loads(text)
        booster.first, booster.second = np.asarray(values["first"]), np.asarray(values["second"])
        return booster


@pytest.fixture
def doubled_fit(monkeypatch, learning_doubles):
    """Ajuste doble que puntúa la validación como el real y registra lo que recibe.

    Puntúa con la predicción original aunque una prueba sustituya después la de la emisión.
    """
    calls, score = [], lightgbm_control.predict

    def fit(train, validation, config, seed, *, min_assets, check_stop=None):
        calls.append(dict(train=train, validation=validation, config=config, seed=seed))
        booster = Booster(seed, train.features.shape[1])
        series = [
            sd.session_correlations(
                validation.panel,
                score(booster, validation.features, rounds),
                min_assets=min_assets,
            )
            for rounds in range(10, 1001, 10)
        ]
        return booster, series

    monkeypatch.setattr(lightgbm_control, "fit", fit)
    monkeypatch.setattr(lightgbm_control, "load", Booster.from_text)
    return calls


def tabular(manifest, protocol, output, **changes):
    options = dict(seeds=[42, 43, 44], fit_months=1, config=TABULAR, radius=1, min_assets=3)
    return windows.run_tabular_window(
        manifest, protocol, "fold-000", output=output, **(options | changes)
    )


@pytest.mark.parametrize("runner", ["rule", "tabular"])
@pytest.mark.parametrize("source", ["view", "prices"])
def test_source_outputs_are_refused_before_creating_a_window_record(
    view, tmp_path, doubled_fit, runner, source
):
    manifest, protocol = view
    root = manifest.parent if source == "view" else next(iter(dataset(manifest).roots.values()))
    output = root / f"rejected-{tmp_path.name}-{runner}"
    assert not output.exists()
    try:
        with pytest.raises(ValueError, match="fuera del directorio de origen"):
            if runner == "rule":
                windows.run_rule_window(manifest, protocol, "fold-000", output=output, config=RULE)
            else:
                tabular(manifest, protocol, output)
        assert not output.exists()
        assert doubled_fit == []
    finally:
        if output.exists():
            shutil.rmtree(output)


def test_the_tabular_window_emits_every_seed_with_their_common_rounds(view, tmp_path, doubled_fit):
    manifest, protocol = view
    output = tmp_path / "tabular"
    report = tabular(manifest, protocol, output)
    assert report["status"] == "completed" and report["kind"] == windows.TABULAR_KIND
    assert [call["seed"] for call in doubled_fit] == [42, 43, 44]
    data = dataset(manifest)
    fold = windows._fold(data, protocol, "fold-000")
    since = window_phases(fold, 0, 1)["train"].decision_start
    train = doubled_fit[0]["train"]
    assert train.panel.prediction_at.min() >= since and report["train_rows"] == train.panel.rows
    labels, kept = lightgbm_control.session_labels(train.panel)
    assert report["fit_rows"] == int(kept.sum())
    # Las rondas son la época común de las tres series de validación por diez.
    series = []
    for call in doubled_fit:
        booster = Booster(call["seed"], call["train"].features.shape[1])
        series.append(
            [
                sd.session_correlations(
                    call["validation"].panel,
                    lightgbm_control.predict(booster, call["validation"].features, rounds),
                    min_assets=3,
                )
                for rounds in range(10, 1001, 10)
            ]
        )
    scores = selection.seed_epoch_scores(series)
    chosen = selection.common_epoch(scores, radius=1)
    # La puntuación cambia con las rondas, así que la elección no es trivial.
    assert len({round(score, 12) for score in scores}) > 1
    assert report["selection"]["epoch"] == chosen["epoch"]
    assert report["selection"]["rounds"] == 10 * chosen["epoch"]
    for seed in (42, 43, 44):
        member = report["seeds"][str(seed)]
        folder = output / f"seed-{seed}"
        assert sha256(output / member["path"]) == member["sha256"]
        record = json.loads((folder / "run.json").read_text())
        assert record["status"] == "completed" and record["rounds"] == report["selection"]["rounds"]
        for name, table in predictions(folder, record).items():
            rows = windows.partition_rows(data, name, with_features=True)
            expected = Booster(seed, rows.features.shape[1]).predict(
                rows.features, report["selection"]["rounds"]
            )
            np.testing.assert_array_equal(table["prediction"].to_numpy(), expected)
            assert table.column_names == COLUMNS


def test_a_completed_tabular_window_checks_every_seed_file(view, tmp_path, doubled_fit):
    manifest, protocol = view
    output = tmp_path / "tabular"
    report = tabular(manifest, protocol, output)
    assert "fits" not in report and not (output / "fits").exists()
    doubled_fit.clear()
    assert tabular(manifest, protocol, output) == report and doubled_fit == []
    # Un corte justo después de confirmar dejaría los ajustes guardados, que ya sobran.
    (output / "fits").mkdir()
    (output / "fits" / "seed-42.json").write_text("{}")
    assert tabular(manifest, protocol, output) == report and not (output / "fits").exists()
    path = output / "seed-43" / "calibration-predictions.parquet"
    original = path.read_bytes()
    path.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="Ha cambiado calibration-predictions.parquet"):
        tabular(manifest, protocol, output)
    path.write_bytes(original)
    record = output / "seed-44" / "run.json"
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="Ha cambiado run.json"):
        tabular(manifest, protocol, output)
    with pytest.raises(ValueError, match="La petición de la ventana ha cambiado"):
        tabular(manifest, protocol, output, radius=2)


class StopAfter:
    """Pide parar cuando el ajuste doble lleva `fits` semillas."""

    def __init__(self, calls, fits):
        self.calls, self.fits = calls, fits

    @property
    def requested(self):
        return len(self.calls) == self.fits


def test_a_paused_tabular_window_keeps_its_fitted_seeds_and_fits_only_the_rest(
    view, tmp_path, doubled_fit
):
    manifest, protocol = view
    output = tmp_path / "tabular"
    paused = tabular(manifest, protocol, output, stop=StopAfter(doubled_fit, 2))
    assert paused["status"] == "paused" and [call["seed"] for call in doubled_fit] == [42, 43]
    assert sorted(paused["fits"]) == ["42", "43"] and not (output / "seed-42").exists()
    for record in paused["fits"].values():
        assert sha256(output / record["path"]) == record["sha256"]
    assert json.loads((output / "run.json").read_text()) == paused
    resumed = tabular(manifest, protocol, output)
    # Solo se ajusta la semilla que faltaba y las guardadas emiten lo mismo que sin pausa.
    assert [call["seed"] for call in doubled_fit] == [42, 43, 44]
    assert resumed["status"] == "completed" and sorted(resumed["seeds"]) == ["42", "43", "44"]
    assert "fits" not in resumed and not (output / "fits").exists()
    whole = tabular(manifest, protocol, tmp_path / "whole")
    assert whole["selection"] == resumed["selection"]
    assert {seed: member["sha256"] for seed, member in whole["seeds"].items()} == {
        seed: member["sha256"] for seed, member in resumed["seeds"].items()
    }


def test_a_stop_inside_a_seed_keeps_only_the_previously_confirmed_seeds(
    view, tmp_path, doubled_fit, monkeypatch
):
    manifest, protocol = view
    output = tmp_path / "tabular"
    original = lightgbm_control.fit
    stop = types.SimpleNamespace(requested=False)

    def interrupted(train, validation, config, seed, *, min_assets, check_stop=None):
        if seed == 43:
            stop.requested = True
            if check_stop is not None:
                check_stop()
        return original(train, validation, config, seed, min_assets=min_assets)

    monkeypatch.setattr(lightgbm_control, "fit", interrupted)
    paused = tabular(manifest, protocol, output, stop=stop)
    assert paused["status"] == "paused" and sorted(paused["fits"]) == ["42"]
    assert not (output / "fits" / "seed-43.json").exists()
    monkeypatch.setattr(lightgbm_control, "fit", original)
    resumed = tabular(manifest, protocol, output)
    assert resumed["status"] == "completed"
    assert [call["seed"] for call in doubled_fit] == [42, 43, 44]


def test_a_changed_saved_fit_is_refused_when_the_window_resumes(view, tmp_path, doubled_fit):
    manifest, protocol = view
    output = tmp_path / "tabular"
    paused = tabular(manifest, protocol, output, stop=StopAfter(doubled_fit, 1))
    path = output / paused["fits"]["42"]["path"]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="ajuste guardado de la semilla 42"):
        tabular(manifest, protocol, output)


@pytest.mark.parametrize("seeds", [[], [42, 42], [42.0], [2**31], "42"])
def test_the_tabular_window_needs_valid_distinct_seeds(view, tmp_path, seeds, doubled_fit):
    manifest, protocol = view
    with pytest.raises(ValueError, match="semillas de la criba"):
        tabular(manifest, protocol, tmp_path / "tabular", seeds=seeds)
    assert not (tmp_path / "tabular").exists()


def test_the_emitted_validation_must_be_the_scored_one(view, tmp_path, doubled_fit, monkeypatch):
    manifest, protocol = view
    original = lightgbm_control.predict
    # La emisión usa otra predicción que la puntuada al elegir las rondas.
    monkeypatch.setattr(
        lightgbm_control,
        "predict",
        lambda booster, values, rounds: original(booster, values, rounds)[::-1].copy(),
    )
    with pytest.raises(ValueError, match="no reproduce la puntuada"):
        tabular(manifest, protocol, tmp_path / "tabular")
