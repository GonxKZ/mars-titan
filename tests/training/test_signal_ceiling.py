"""Ventana del techo de señal (#747) sobre la vista técnica, con modelos dobles que no ajustan.

La vista corta de las pruebas recibe un factor de mercado y presentaciones técnicas. Las filas,
el retorno bruto, las características y su causalidad se comprueban con la lectura real. Ridge y
LightGBM se sustituyen por dobles de predicción fija, así que ninguna prueba estima parámetros.
"""

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mars_titan.data import anomaly_features as af
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.evaluation import scale_diagnostics as sd
from mars_titan.evaluation.forecast_panel import DAY_MICROSECONDS, ForecastPanel
from mars_titan.models.baselines import qlib_lightgbm, session_ridge
from mars_titan.training import selection
from mars_titan.training import signal_ceiling as sc
from mars_titan.training.candidate_walk_forward import view_rows
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.learning_hold import LearningHoldError
from mars_titan.training.temporal_contract import temporal_fold
from mars_titan.training.walk_forward_phases import PREDICTED, window_phases
from tests.training.chronological_fixture import clock, decision, micros
from tests.training.test_titans_walk_forward import views

ROOT = Path(__file__).resolve().parents[2]
DECLARATION = json.loads((ROOT / "configs/experiments/signal-ceiling-v1.json").read_text())
SCREENING = json.loads((ROOT / "configs/experiments/variant-screening-v1.json").read_text())
ORIGIN = {"year": 2023, "fold": "fold-000"}
# El corte cae en la calibración, así que separa filas de los tres tramos medidos.
CUT = decision("2023-10-16")
STEP = DECLARATION["models"]["lightgbm"]["checkpoint_rounds"]
ROUNDS = DECLARATION["models"]["lightgbm"]["max_rounds"]


def add_sources(manifest, *, perturb_after=None):
    """Factor de mercado y presentaciones técnicas, alterados después de `perturb_after`."""
    document = json.loads(manifest.read_text())
    decisions = np.asarray([micros(moment) for moment in clock().decisions], dtype=np.int64)
    generator = np.random.default_rng(77)
    close = 400 * np.exp(np.cumsum(generator.normal(0, 0.01, len(decisions))))
    if perturb_after is not None:
        close[decisions > perturb_after] *= 1.5
    factor = manifest.parents[2] / "factor" / "prices.parquet"
    factor.parent.mkdir(parents=True, exist_ok=True)
    stamps = pa.array(decisions, type=pa.timestamp("us", tz="UTC"))
    pq.write_table(pa.table(dict(close=close, available_at=stamps)), factor)
    document["market_factors"] = {
        "US": dict(market="US", symbol="SPY", prices_path=str(factor), prices_sha256=sha256(factor))
    }
    prepared = Path(document["roots"]["prepared"])
    for k, asset in enumerate(document["assets"]):
        positions = np.arange(30 + 7 * k, len(decisions) - 3, 63)
        if asset["symbol"] == "A0002":
            # Un fondo sin presentaciones.
            positions = positions[:0]
        names = [f"{asset['symbol']}-{p}" for p in positions]
        # Cada presentación tiene un hecho posterior, que no cambia cuándo se conoce.
        accessions = names + names
        moments = np.r_[decisions[positions], decisions[positions + 3]]
        if perturb_after is not None and len(positions):
            accessions.append(f"{asset['symbol']}-later")
            moments = np.r_[moments, decisions[np.searchsorted(decisions, perturb_after) + 2]]
        folder = prepared / "US" / asset["symbol"]
        table = pa.table(
            dict(
                accession=pa.array(accessions, type=pa.string()),
                available_at=pa.array(moments, type=pa.timestamp("us", tz="UTC")),
            )
        )
        pq.write_table(table, folder / "fundamentals.parquet")
        artifacts = {
            "prices.parquet": asset["prices_sha256"],
            "fundamentals.parquet": sha256(folder / "fundamentals.parquet"),
        }
        (folder / "manifest.json").write_text(json.dumps(dict(artifacts=artifacts)))
    manifest.write_text(json.dumps(document))
    return manifest


@pytest.fixture(scope="module")
def view(tmp_path_factory):
    manifest, _ = views(tmp_path_factory.mktemp("signal-ceiling"), price_window=True)
    return add_sources(manifest)


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    manifest, _ = views(
        tmp_path_factory.mktemp("signal-ceiling-later"), price_window=True, perturb_after=CUT
    )
    return add_sources(manifest, perturb_after=CUT)


def dataset(manifest):
    return CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)


def declaration(**screening):
    document = copy.deepcopy(DECLARATION)
    document["screening"].update(origins=[ORIGIN], fit_months=1, **screening)
    return document


@pytest.fixture
def technical_origin(monkeypatch):
    # La vista técnica tiene una única ventana, la de 2023, que no es un origen de la criba.
    monkeypatch.setattr(sc, "ORIGIN_FOLDS", {2023: "fold-000"})


def test_the_declaration_shares_the_screening_rows_seeds_selection_and_inference():
    shared = DECLARATION["screening"]
    assert shared["origins"] == SCREENING["origins"] and shared["seeds"] == SCREENING["seeds"]
    assert shared["fit_months"] == SCREENING["fit"]["fit_months"]
    assert shared["smoothing_radius"] == SCREENING["selection"]["smoothing_radius"]
    assert shared["minimum_assets"] == SCREENING["contrast"]["primary"]["minimum_assets"]
    assert shared["inference"] == {key: SCREENING["inference"][key] for key in shared["inference"]}
    assert {"block_days", "sensitivity_block_days", "replicates", "seed"} <= set(
        shared["inference"]
    )
    # LightGBM es el de los controles comunes con otras características y su etiqueta.
    common = SCREENING["common_controls"]["models"]["lightgbm"]["config"]
    own = {"identity", "features", "label"}
    assert {k: v for k, v in DECLARATION["models"]["lightgbm"].items() if k not in own} == {
        k: v for k, v in common.items() if k not in own
    }
    assert sc.checked_declaration(copy.deepcopy(DECLARATION)) == DECLARATION
    assert DECLARATION["features"]["families"] == {k: list(v) for k, v in af.FAMILIES.items()}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("kind",), "otro"),
        (("final_test_access",), True),
        (("features", "identity"), "otras"),
        (("features", "families", "momentum"), []),
        (("features", "minimum_share"), 0.5),
        (("targets",), {"residual": "Residuo"}),
        (("screening", "origins"), [{"year": 2019, "fold": "fold-015"}]),
        (("screening", "seeds"), [42, 42]),
        (("screening", "seeds"), [42.0]),
        (("screening", "fit_months"), 0),
        (("screening", "smoothing_radius"), -1),
        (("screening", "minimum_assets"), 2),
        (("screening", "inference", "block_days"), 0),
        (("screening", "inference", "sensitivity_block_days"), [8.0]),
        (("screening", "inference", "confidence"), 1.0),
        (("screening", "inference", "replicates"), "2000"),
        (("models", "ridge", "alphas"), [10.0, 1.0]),
        (("models", "lightgbm", "identity"), "same_inputs_lightgbm_v1"),
        (("importance", "metric"), "session_rank_ic"),
        (("importance", "seed"), -1),
    ],
)
def test_a_changed_declaration_is_refused(path, value):
    document = copy.deepcopy(DECLARATION)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="declara|comparte"):
        sc.checked_declaration(document)


def test_the_rows_are_those_of_the_view_with_their_next_session_return(view):
    data = dataset(view)
    cohort = sc.load_cohort(data, "US")
    decisions, absent = data.market_calendar("US")
    prices = {f"US/{asset['symbol']}": data.asset_prices(asset) for asset in data.assets}
    for name in PREDICTED:
        rows = sc.partition_rows(data, cohort, name)
        expected = view_rows(data, name)
        assert rows.panel.rows == data.manifest["counts"][name] == len(expected["target"])
        keys = set(zip(expected["asset_id"], expected["prediction_at"].tolist(), strict=True))
        names = np.asarray(list(cohort.index))[rows.asset]
        assert set(zip(names, rows.panel.prediction_at.tolist(), strict=True)) == keys
        assert np.all(rows.panel.prediction == 0.0)
        for row in range(rows.panel.rows):
            values, available = prices[names[row]]
            position = int(np.searchsorted(decisions, rows.panel.prediction_at[row]))
            # El retorno bruto es el apertura-cierre del activo en la sesión siguiente.
            following = int(np.searchsorted(available, decisions[position + 1]))
            assert available[following] == decisions[position + 1]
            assert rows.raw[row] == values[following, 3] / values[following, 0] - 1
            session = position - int(np.searchsorted(absent, position))
            assert rows.session[row] == session
            asset = rows.asset[row]
            np.testing.assert_array_equal(
                rows.features[row],
                np.r_[cohort.ranks[asset, session], cohort.missing[asset, session]],
            )
        assert rows.features.shape == (rows.panel.rows, len(sc.COLUMNS))
        raw = rows.target("raw_open_close")
        np.testing.assert_array_equal(raw.panel.target, rows.raw)
        np.testing.assert_array_equal(raw.panel.prediction_at, rows.panel.prediction_at)
        assert rows.target("residual") is rows
    with pytest.raises(ValueError, match="Objetivo desconocido"):
        rows.target("raw")


def test_the_fit_rows_are_the_train_decisions_of_the_fit_months(view):
    data = dataset(view)
    cohort = sc.load_cohort(data, "US")
    fold = temporal_fold(data.manifest, input_policy=HISTORICAL_MASKED)
    since = window_phases(fold, 0, 1)["train"].decision_start
    rows = sc.partition_rows(data, cohort, "train", since=since)
    every = view_rows(data, "train")
    assert rows.panel.rows == int(np.count_nonzero(every["prediction_at"] >= since)) > 0
    assert rows.panel.prediction_at.min() >= since
    assert rows.panel.rows < len(every["target"])
    with pytest.raises(ValueError, match="no tiene filas"):
        sc.partition_rows(data, cohort, "train", since=2**62)


def test_the_cohort_reads_the_declared_factor_and_filings_and_ranks_by_session(view):
    data = dataset(view)
    cohort = sc.load_cohort(data, "US")
    decisions, absent = data.market_calendar("US")
    assert len(absent) == 0
    assert cohort.ranks.shape == (3, len(decisions), len(af.FEATURES))
    assert list(cohort.index) == ["US/A0000", "US/A0001", "US/A0002"]
    assert cohort.sources["market_factor"]["symbol"] == "SPY"
    # El fondo no tiene presentaciones y sus características de presentación faltan siempre.
    filings = [af.FEATURES.index(name) for name in af.FAMILIES["filings"]]
    assert cohort.sources["assets"]["US/A0002"]["filings"] == 0
    assert cohort.missing[cohort.index["US/A0002"]][:, filings].all()
    # Cada presentación cuenta una vez, aunque tenga varios hechos.
    count = len(np.arange(30, len(decisions) - 3, 63))
    assert cohort.sources["assets"]["US/A0000"]["filings"] == count
    first = next(asset for asset in data.assets if asset["symbol"] == "A0000")
    _, available = data.asset_prices(first)
    last = int(np.searchsorted(decisions, available[-1]))
    missing = cohort.missing[cohort.index["US/A0000"]]
    assert not missing[last, filings].any() and missing[0, filings].all()
    # Sin precio, la sesión no tiene ninguna característica.
    assert last + 1 < len(decisions) and missing[last + 1 :].all()
    # Los rangos de cada sesión están en [−1, 1] y valen cero donde falta el valor.
    assert np.abs(cohort.ranks).max() <= 1.0 and np.all(cohort.ranks[cohort.missing] == 0.0)


def test_no_feature_of_a_row_reads_prices_factor_or_filings_after_its_decision(view, perturbed):
    first, second = dataset(view), dataset(perturbed)
    cohorts = sc.load_cohort(first, "US"), sc.load_cohort(second, "US")
    kept = changed = 0
    for name in PREDICTED:
        rows = [
            sc.partition_rows(data, cohort, name)
            for data, cohort in zip((first, second), cohorts, strict=True)
        ]
        np.testing.assert_array_equal(rows[0].panel.prediction_at, rows[1].panel.prediction_at)
        before = rows[0].panel.prediction_at <= CUT
        different = np.any(rows[0].features != rows[1].features, axis=1)
        assert not np.any(different & before)
        kept += int(np.count_nonzero(before))
        changed += int(np.count_nonzero(different))
    # Hay filas a los dos lados del corte y la alteración cambia las posteriores, así que la
    # prueba mira algo.
    assert kept > 0 and changed > 0


def test_a_view_row_must_belong_to_the_cohort_and_calendar_and_have_its_next_return(view):
    data = dataset(view)
    cohort = sc.load_cohort(data, "US")
    other = dataclasses.replace(cohort, index={"US/A0000": 0, "US/A0001": 1})
    with pytest.raises(ValueError, match="no pertenece a la cohorte"):
        sc.partition_rows(data, other, "evaluation")
    shifted = dataclasses.replace(cohort, decisions=cohort.decisions + 1)
    with pytest.raises(ValueError, match="no pertenece al calendario"):
        sc.partition_rows(data, shifted, "evaluation")
    open_close = cohort.open_close.copy()
    open_close[:, -40:] = np.nan
    with pytest.raises(ValueError, match="retorno de la sesión siguiente"):
        sc.partition_rows(data, dataclasses.replace(cohort, open_close=open_close), "evaluation")


def test_a_changed_factor_or_filings_file_is_refused(view):
    data = dataset(view)
    factor = Path(data.manifest["market_factors"]["US"]["prices_path"])
    prepared = data.roots["prepared"] / "US" / "A0001"
    manifest = json.loads((prepared / "manifest.json").read_text())
    manifest["artifacts"]["prices.parquet"] = "0" * 64
    for path, changed, message in (
        (factor, None, "factor de mercado cambió"),
        (prepared / "fundamentals.parquet", None, "fundamentales del activo cambiaron"),
        (prepared / "manifest.json", json.dumps(manifest).encode(), "con los precios de la vista"),
    ):
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"x" if changed is None else changed)
            with pytest.raises(ValueError, match=message):
                sc.load_cohort(data, "US")
        finally:
            path.write_bytes(original)
    without = dataset(view)
    without.manifest = dict(without.manifest, market_factors={})
    with pytest.raises(ValueError, match="factor de mercado de su residuo"):
        sc.load_cohort(without, "US")
    # Un factor declarado con su huella que no sigue el calendario no sirve para el residuo.
    table = pq.read_table(factor)
    later = table["available_at"].cast(pa.int64()).to_numpy() + 1
    shifted = factor.with_name("shifted.parquet")
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("available_at"),
            "available_at",
            pa.array(later, type=pa.timestamp("us", tz="UTC")),
        ),
        shifted,
    )
    entry = dict(
        data.manifest["market_factors"]["US"],
        prices_path=str(shifted),
        prices_sha256=sha256(shifted),
    )
    elsewhere = dataset(view)
    elsewhere.manifest = dict(elsewhere.manifest, market_factors={"US": entry})
    with pytest.raises(ValueError, match="no sigue el calendario declarado"):
        sc.load_cohort(elsewhere, "US")


def test_the_session_of_a_row_skips_the_sessions_absent_in_the_whole_market(view):
    data = dataset(view)
    cohort = sc.load_cohort(data, "US")
    rows = sc.partition_rows(data, cohort, "evaluation")
    # Con la primera posición ausente en todo el mercado, cada fila pasa a la sesión anterior
    # de la lista de sesiones con mercado y lee las mismas características.
    absent = dataclasses.replace(
        cohort, absent=np.array([0]), ranks=cohort.ranks[:, 1:], missing=cohort.missing[:, 1:]
    )
    shifted = sc.partition_rows(data, absent, "evaluation")
    np.testing.assert_array_equal(shifted.session, rows.session - 1)
    np.testing.assert_array_equal(shifted.features, rows.features)


def test_the_permutation_moves_only_the_family_columns_inside_each_session(view):
    data = dataset(view)
    rows = sc.partition_rows(data, sc.load_cohort(data, "US"), "evaluation")
    columns = sc._family_columns()["reversal"]
    assert columns == [0, 1, 10, 11]
    permuted = sc.permuted_features(rows, columns, np.random.default_rng(5))
    np.testing.assert_array_equal(
        permuted, sc.permuted_features(rows, columns, np.random.default_rng(5))
    )
    others = [k for k in range(len(sc.COLUMNS)) if k not in columns]
    np.testing.assert_array_equal(permuted[:, others], rows.features[:, others])
    moved = 0
    for start, count in zip(rows.panel.session_starts, rows.panel.session_samples, strict=True):
        block = slice(start, start + count)
        # Las columnas de la familia se mueven juntas, así que cada fila sigue entera.
        original = sorted(map(tuple, rows.features[block][:, columns]))
        assert sorted(map(tuple, permuted[block][:, columns])) == original
        moved += not np.array_equal(permuted[block], rows.features[block])
    assert moved > 0


def test_a_constant_prediction_does_not_order_and_counts_as_a_zero_ic(view):
    data = dataset(view)
    panel = sc.partition_rows(data, sc.load_cohort(data, "US"), "evaluation").panel
    eligible = sc.eligible_sessions(panel, 3)
    assert eligible.tolist() == [
        bool(count >= 3 and np.ptp(panel.target[start : start + count]) > 0)
        for start, count in zip(panel.session_starts, panel.session_samples, strict=True)
    ]
    assert eligible.any() and not eligible.all()
    # Una sesión con el objetivo constante tampoco puede medir un orden.
    times = np.repeat(1_700_000_000_000_000 + DAY_MICROSECONDS * np.arange(3), 3)
    target = np.array([0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.0, 0.0, 0.05])
    ids = [f"US/A{k % 3}/{time}" for k, time in enumerate(times)]
    constant = ForecastPanel.from_columns(ids, np.full(9, "US"), times, target, np.zeros(9))
    assert sc.eligible_sessions(constant, 3).tolist() == [True, False, True]
    assert not sc.eligible_sessions(constant, 4).any()
    values, defined = sc.ordering(panel, np.zeros(panel.rows), min_assets=3)
    assert np.all(values == 0.0) and np.array_equal(defined, eligible)
    prediction = np.random.default_rng(3).normal(size=panel.rows)
    for rank in (False, True):
        values, defined = sc.ordering(panel, prediction, min_assets=3, rank=rank)
        reference, known = sd.session_correlations(panel, prediction, rank=rank, min_assets=3)
        np.testing.assert_array_equal(values, reference)
        assert np.array_equal(known, defined)


class Linear:
    """Modelo doble de predicción fija, con un segundo peso que crece con las rondas."""

    def __init__(self, weights, growing):
        self.weights, self.growing = weights, growing
        self.inputs, self.rounds = [], []

    def predict(self, values, num_iteration=None):
        self.inputs.append(np.array(values))
        self.rounds.append(num_iteration)
        if num_iteration == STEP:
            # El primer punto de control no ordena, como un LightGBM que aún no ha partido.
            return np.zeros(len(values))
        share = 0.0 if num_iteration is None else num_iteration / ROUNDS
        return np.asarray(values) @ (self.weights + share * self.growing)


def weights(seed):
    values = np.random.default_rng(seed).normal(size=len(sc.COLUMNS))
    # El momento no pesa, así que permutarlo no cambia la predicción.
    values[sc._family_columns()["momentum"]] = 0.0
    return values


@pytest.fixture
def doubled_models(monkeypatch, learning_doubles):
    calls = dict(ridge=[], lightgbm=[])

    def ridge(train, validation, config, *, min_assets):
        model = Linear(weights(0), np.zeros(len(sc.COLUMNS)))
        calls["ridge"].append(dict(train=train, model=model))
        return model, dict(alpha=config["alphas"][0], validation_ic=0.0)

    def lightgbm(train, validation, config, seed, *, min_assets):
        booster = Linear(weights(seed), weights(seed + 1))
        series = [
            sd.session_correlations(
                validation.panel, booster.predict(validation.features, rounds), min_assets=3
            )
            for rounds in range(STEP, ROUNDS + 1, STEP)
        ]
        booster.inputs.clear()
        booster.rounds.clear()
        calls["lightgbm"].append(dict(train=train, seed=seed, booster=booster, series=series))
        return booster, series

    monkeypatch.setattr(session_ridge, "fit", ridge)
    monkeypatch.setattr(qlib_lightgbm, "fit", lightgbm)
    return calls


def test_an_origin_measures_both_models_with_both_targets_and_reuses_its_report(
    view, tmp_path, technical_origin, doubled_models
):
    document = declaration(seeds=[42, 43])
    output = tmp_path / "ceiling"
    report = sc.run_origin(view, ORIGIN, document, output)
    assert report["status"] == "completed" and report["final_test_opened"] is False
    assert json.loads((output / "origin-2023.json").read_text()) == report
    data = dataset(view)
    assert {name: report["rows"][name] for name in PREDICTED} == {
        name: data.manifest["counts"][name] for name in PREDICTED
    }
    # Cada objetivo ajusta Ridge una vez y LightGBM con cada semilla, con las filas de ajuste.
    assert len(doubled_models["ridge"]) == 2
    assert [call["seed"] for call in doubled_models["lightgbm"]] == [42, 43, 42, 43]
    residual, raw = (call["train"] for call in doubled_models["ridge"])
    np.testing.assert_array_equal(raw.panel.target, residual.raw)
    assert residual.panel.rows == report["rows"]["train"]
    fold = temporal_fold(data.manifest, input_policy=HISTORICAL_MASKED)
    assert report["fit_start_us"] == window_phases(fold, 0, 1)["train"].decision_start
    assert residual.panel.prediction_at.min() >= report["fit_start_us"]
    sessions = report["sessions"]
    assert np.all(np.diff(sessions["time"]) > 0)
    assert sessions["day"] == [time // DAY_MICROSECONDS for time in sessions["time"]]
    cohort = sc.load_cohort(data, "US")
    parts = {name: sc.partition_rows(data, cohort, name) for name in PREDICTED}
    evaluation = parts["evaluation"]
    families = len(af.FAMILIES)
    shared = doubled_models["ridge"][0]["model"].inputs[len(PREDICTED) :]
    # La permutación de cada familia sale de su flujo de la semilla declarada.
    streams = np.random.SeedSequence(DECLARATION["importance"]["seed"]).spawn(families)
    columns = sc._family_columns().values()
    for permuted, family, stream in zip(shared, columns, streams, strict=True):
        expected = sc.permuted_features(evaluation, family, np.random.default_rng(stream))
        np.testing.assert_array_equal(permuted, expected)
    for k, target in enumerate(sc.TARGETS):
        rows = evaluation.target(target)
        result = report["targets"][target]
        assert result["zero_mae"] == sd.session_mae(rows.panel, np.zeros(rows.panel.rows)).tolist()
        prediction = rows.features @ weights(0)
        values, defined = sc.ordering(rows.panel, prediction, min_assets=3)
        assert result["ridge"]["session_ic"] == dict(
            values=np.where(defined, values, 0.0).tolist(), defined=defined.tolist()
        )
        assert result["ridge"]["session_mae_raw"] == sd.session_mae(rows.panel, prediction).tolist()
        # El λ de cada mercado sale de la calibración y se aplica a la evaluación.
        calibration = parts["calibration"].target(target)
        factors = sd.market_shrinkage(calibration.panel, calibration.features @ weights(0))
        shrunk = sd.apply_shrinkage(rows.panel, prediction, factors)
        assert (
            result["ridge"]["session_mae_calibrated"] == sd.session_mae(rows.panel, shrunk).tolist()
        )
        assert set(result["ridge"]["importance"]) == set(af.FAMILIES)
        # La importancia es la caída del IC al permutar, en las sesiones que ambos definen.
        changed, kept = sc.ordering(rows.panel, shared[0] @ weights(0), min_assets=3)
        both = defined & kept
        assert result["ridge"]["importance"]["reversal"] == dict(
            values=np.where(both, values - changed, 0.0).tolist(), defined=both.tolist()
        )
        # Permutar una familia sin peso no cambia el IC de ninguna sesión.
        assert not any(result["ridge"]["importance"]["momentum"]["values"])
        assert any(result["ridge"]["importance"]["reversal"]["values"])
        # Cada tramo y cada familia permutada se predicen una vez, con la misma permutación.
        model = doubled_models["ridge"][k]["model"]
        assert len(model.inputs) == len(PREDICTED) + families
        for permuted, reference in zip(model.inputs[len(PREDICTED) :], shared, strict=True):
            np.testing.assert_array_equal(permuted, reference)
        # Las rondas salen de la época común con el IC nulo de las predicciones constantes.
        chosen = result["lightgbm"]["selection"]
        fits = doubled_models["lightgbm"][2 * k : 2 * k + 2]
        eligible = sc.eligible_sessions(parts["validation"].target(target).panel, 3)
        series = [[(values, eligible) for values, _ in fit["series"]] for fit in fits]
        assert chosen["scores"] == selection.seed_epoch_scores(series)
        assert chosen == dict(
            selection.common_epoch(chosen["scores"], radius=1), rounds=STEP * chosen["epoch"]
        )
        for fit in fits:
            booster = fit["booster"]
            assert booster.rounds == [chosen["rounds"]] * (len(PREDICTED) + families)
            for permuted, reference in zip(booster.inputs[len(PREDICTED) :], shared, strict=True):
                np.testing.assert_array_equal(permuted, reference)
        assert len(result["lightgbm"]["session_mae_calibrated"]) == len(sessions["time"])
    assert set(report["coverage"]) == {"train", *PREDICTED}
    assert set(report["versions"]) == set(sc.VERSIONED)
    # Un informe terminado con la misma petición se devuelve sin ajustar nada.
    assert sc.run_origin(view, ORIGIN, document, output) == report
    assert len(doubled_models["ridge"]) == 2 and len(doubled_models["lightgbm"]) == 4
    with pytest.raises(ValueError, match="petición del origen ha cambiado"):
        sc.run_origin(view, ORIGIN, declaration(seeds=[42]), output)


def test_an_origin_refuses_another_window_an_undeclared_origin_or_an_output_inside_the_view(
    view, tmp_path, technical_origin, doubled_models, monkeypatch
):
    with pytest.raises(ValueError, match="no está declarado"):
        sc.run_origin(view, {"year": 2022, "fold": "fold-017"}, declaration(), tmp_path / "a")
    monkeypatch.setattr(sc, "ORIGIN_FOLDS", {2023: "fold-001"})
    other = declaration()
    other["screening"]["origins"] = [{"year": 2023, "fold": "fold-001"}]
    with pytest.raises(ValueError, match="no es la ventana del origen"):
        sc.run_origin(view, {"year": 2023, "fold": "fold-001"}, other, tmp_path / "b")
    monkeypatch.setattr(sc, "ORIGIN_FOLDS", {2023: "fold-000"})
    with pytest.raises(ValueError, match="fuera del directorio de origen"):
        sc.run_origin(view, ORIGIN, declaration(), view.parent / "inside")
    assert doubled_models == dict(ridge=[], lightgbm=[])
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_hold_stops_an_origin_before_reading_the_view(tmp_path):
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        sc.run_origin(tmp_path / "absent.json", ORIGIN, {}, tmp_path / "out")
    assert not (tmp_path / "out").exists()
