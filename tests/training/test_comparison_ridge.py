"""Ventana de Ridge de la campaña de comparación (#836) con el ajuste sustituido por un doble.

El doble escribe el informe y las tablas que dejaría `run_tabular_reference`, con el MAE de
validación que la prueba asigna a cada alpha, así que ninguna prueba estima coeficientes.
"""

import json

import pytest

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import sha256
from mars_titan.training import comparison_ridge as ridge
from mars_titan.training.reference_run import HELDOUT_FULL_TRAIN_SESSIONS

CONFIG = dict(identity=ridge.IDENTITY, alphas=[0.1, 1.0, 10.0], batch_size=1024)
PARTITIONS = ("validation", "calibration", "evaluation")


class Fits:
    """Doble del ajuste: registra cada llamada y escribe lo que dejaría un ajuste real."""

    def __init__(self, scores, *, cut=(), status="completed"):
        self.scores, self.cut, self.status, self.calls = scores, set(cut), status, []

    def __call__(self, view, output, **options):
        self.calls.append((view, output, options))
        output.mkdir(parents=True, exist_ok=False)
        alpha = options["alpha"]
        if alpha in self.cut:
            self.cut.discard(alpha)
            raise RuntimeError("corte")
        predictions = {}
        for partition in PARTITIONS:
            path = output / f"{partition}-predictions.parquet"
            path.write_text(f"{partition} {alpha}")
            # Solo la validación ordena los alphas. Las otras particiones llevan otro valor.
            score = self.scores[alpha] + (0.0 if partition == "validation" else 1.0)
            predictions[partition] = dict(
                path=path.name, sha256=sha256(path), metrics=dict(session_mae=score)
            )
        report = dict(status=self.status, alpha=alpha, predictions=predictions)
        (output / "run.json").write_text(json.dumps(report))
        return report


class StopAfter:
    """Parada que se pide cuando el doble ya ha ajustado `count` alphas."""

    def __init__(self, fits, count):
        self.fits, self.count = fits, count

    @property
    def requested(self):
        return len(self.fits.calls) >= self.count


@pytest.fixture
def view(tmp_path):
    path = tmp_path / "view.json"
    path.write_text(json.dumps(dict(kind="corpus_supervision")))
    return path


@pytest.fixture
def fitted(monkeypatch, learning_doubles):
    def install(scores, **options):
        double = Fits(scores, **options)
        monkeypatch.setattr(ridge, "run_tabular_reference", double)
        return double

    return install


def test_ridge_fits_each_alpha_once_and_emits_the_best_validation(view, tmp_path, fitted):
    fits = fitted({0.1: 0.03, 1.0: 0.01, 10.0: 0.02})
    output = tmp_path / "out"

    report = ridge.run_ridge_window(view, output=output, config=CONFIG)

    assert [(call[0], call[1].relative_to(output).as_posix()) for call in fits.calls] == [
        (view, "alpha-0.1/attempt-0001"),
        (view, "alpha-1.0/attempt-0001"),
        (view, "alpha-10.0/attempt-0001"),
    ]
    assert [call[2] for call in fits.calls] == [
        dict(
            kind="ridge",
            alpha=alpha,
            batch_size=1024,
            input_policy=HISTORICAL_MASKED,
            prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
        )
        for alpha in (0.1, 1.0, 10.0)
    ]
    assert report["status"] == "completed" and report["kind"] == ridge.KIND
    assert report["final_test_opened"] is False
    assert report["request"] == dict(view_sha256=sha256(view), config=CONFIG)
    assert report["selection"] == dict(alpha=1.0, rule=ridge.RULE)
    assert {name: fit["validation_session_mae"] for name, fit in report["fits"].items()} == {
        "alpha-0.1": 0.03,
        "alpha-1.0": 0.01,
        "alpha-10.0": 0.02,
    }
    chosen = output / "alpha-1.0/attempt-0001"
    assert report["fits"]["alpha-1.0"] == dict(
        alpha=1.0,
        path="alpha-1.0/attempt-0001",
        run_sha256=sha256(chosen / "run.json"),
        validation_session_mae=0.01,
    )
    assert set(report["predictions"]) == set(PARTITIONS)
    for partition, record in report["predictions"].items():
        path = output / record["path"]
        assert path == chosen / f"{partition}-predictions.parquet"
        assert record["sha256"] == sha256(path)
        expected = 0.01 if partition == "validation" else 1.01
        assert record["metrics"] == dict(session_mae=expected)
    assert json.loads((output / "run.json").read_text()) == report
    # De los alphas no elegidos queda su informe, y sus tablas se retiran.
    assert report["released_tables"] == [
        f"alpha-{alpha}/attempt-0001/{partition}-predictions.parquet"
        for alpha in ("0.1", "10.0")
        for partition in sorted(PARTITIONS)
    ]
    for alpha in ("0.1", "10.0"):
        folder = output / f"alpha-{alpha}/attempt-0001"
        assert [path.name for path in folder.iterdir()] == ["run.json"]
    assert sorted(path.name for path in chosen.iterdir()) == [
        "calibration-predictions.parquet",
        "evaluation-predictions.parquet",
        "run.json",
        "validation-predictions.parquet",
    ]


def test_a_tie_in_validation_goes_to_the_larger_alpha(view, tmp_path, fitted):
    fitted({0.1: 0.01, 1.0: 0.02, 10.0: 0.01})

    report = ridge.run_ridge_window(view, output=tmp_path / "out", config=CONFIG)

    assert report["selection"]["alpha"] == 10.0
    assert report["predictions"]["validation"]["path"].startswith("alpha-10.0/")


def test_an_integer_alpha_is_fitted_and_named_as_its_float(view, tmp_path, fitted):
    fits = fitted({1.0: 0.01})
    config = dict(CONFIG, alphas=[1])

    report = ridge.run_ridge_window(view, output=tmp_path / "out", config=config)

    assert type(fits.calls[0][2]["alpha"]) is float
    assert list(report["fits"]) == ["alpha-1.0"] and report["selection"]["alpha"] == 1.0


def test_a_completed_window_is_returned_without_fitting_again(view, tmp_path, fitted):
    fits = fitted({0.1: 0.03, 1.0: 0.01, 10.0: 0.02})
    first = ridge.run_ridge_window(view, output=tmp_path / "out", config=CONFIG)

    again = ridge.run_ridge_window(view, output=tmp_path / "out", config=CONFIG)

    assert again == first and len(fits.calls) == 3


def test_a_stop_pauses_before_the_next_alpha_and_the_same_call_continues(view, tmp_path, fitted):
    fits = fitted({0.1: 0.03, 1.0: 0.01, 10.0: 0.02})
    output = tmp_path / "out"

    paused = ridge.run_ridge_window(view, output=output, config=CONFIG, stop=StopAfter(fits, 1))

    assert paused["status"] == "paused" and list(paused["fits"]) == ["alpha-0.1"]
    assert "predictions" not in paused and "selection" not in paused
    # Una ventana pausada conserva las tablas de lo ya ajustado: aún no hay alpha elegido.
    assert (output / "alpha-0.1/attempt-0001/validation-predictions.parquet").is_file()
    assert json.loads((output / "run.json").read_text()) == paused

    report = ridge.run_ridge_window(view, output=output, config=CONFIG)

    assert report["status"] == "completed" and report["selection"]["alpha"] == 1.0
    assert [call[2]["alpha"] for call in fits.calls] == [0.1, 1.0, 10.0]
    assert report["started_at_utc"] == paused["started_at_utc"]


def test_a_stop_requested_from_the_start_validates_and_fits_nothing(view, tmp_path, fitted):
    fits = fitted({})
    output = tmp_path / "out"

    report = ridge.run_ridge_window(view, output=output, config=CONFIG, stop=StopAfter(fits, 0))

    assert report["status"] == "paused" and report["fits"] == {} and fits.calls == []
    assert [path.name for path in output.iterdir()] == ["run.json"]


def test_a_cut_fit_keeps_its_folder_and_the_retry_uses_a_new_attempt(view, tmp_path, fitted):
    fits = fitted({0.1: 0.03, 1.0: 0.01, 10.0: 0.02}, cut=[1.0])
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="corte"):
        ridge.run_ridge_window(view, output=output, config=CONFIG)

    saved = json.loads((output / "run.json").read_text())
    assert saved["status"] == "running" and list(saved["fits"]) == ["alpha-0.1"]

    report = ridge.run_ridge_window(view, output=output, config=CONFIG)

    assert [call[1].relative_to(output).as_posix() for call in fits.calls] == [
        "alpha-0.1/attempt-0001",
        "alpha-1.0/attempt-0001",
        "alpha-1.0/attempt-0002",
        "alpha-10.0/attempt-0001",
    ]
    assert (output / "alpha-1.0/attempt-0001").is_dir()
    assert report["fits"]["alpha-1.0"]["path"] == "alpha-1.0/attempt-0002"
    assert report["predictions"]["evaluation"]["path"].startswith("alpha-1.0/attempt-0002/")


def test_a_fit_that_does_not_finish_is_an_error(view, tmp_path, fitted):
    fitted({0.1: 0.03}, status="not_executed_budget")

    with pytest.raises(ValueError, match="no ha terminado el ajuste de alpha-0.1"):
        ridge.run_ridge_window(view, output=tmp_path / "out", config=CONFIG)


def test_a_changed_request_is_rejected(view, tmp_path, fitted):
    fits = fitted({0.1: 0.03, 1.0: 0.01, 10.0: 0.02, 100.0: 0.001})
    output = tmp_path / "out"
    ridge.run_ridge_window(view, output=output, config=CONFIG)

    with pytest.raises(ValueError, match="La petición de la ventana ha cambiado"):
        ridge.run_ridge_window(view, output=output, config=dict(CONFIG, alphas=[100.0]))
    view.write_text(json.dumps(dict(kind="otra")))
    with pytest.raises(ValueError, match="La petición de la ventana ha cambiado"):
        ridge.run_ridge_window(view, output=output, config=CONFIG)

    assert len(fits.calls) == 3


def test_a_changed_report_of_the_chosen_alpha_is_rejected(view, tmp_path, fitted):
    fits = fitted({0.1: 0.01, 1.0: 0.02, 10.0: 0.03})
    output = tmp_path / "out"
    ridge.run_ridge_window(view, output=output, config=CONFIG, stop=StopAfter(fits, 2))
    chosen = output / "alpha-0.1/attempt-0001/run.json"
    chosen.write_text(chosen.read_text() + " ")

    with pytest.raises(ValueError, match="Ha cambiado el informe del alpha elegido"):
        ridge.run_ridge_window(view, output=output, config=CONFIG)


@pytest.mark.parametrize(
    "config",
    [
        None,
        {key: value for key, value in CONFIG.items() if key != "batch_size"},
        dict(CONFIG, extra=1),
        dict(CONFIG, identity="ridge"),
        dict(CONFIG, alphas=(0.1, 1.0)),
        dict(CONFIG, alphas=[]),
        dict(CONFIG, alphas=[0.1] * 2),
        dict(CONFIG, alphas=[float(value) for value in range(1, 10)]),
        dict(CONFIG, alphas=[0.0]),
        dict(CONFIG, alphas=[-1.0]),
        dict(CONFIG, alphas=[float("inf")]),
        dict(CONFIG, alphas=[True]),
        dict(CONFIG, alphas=["1.0"]),
        dict(CONFIG, batch_size=0),
        dict(CONFIG, batch_size=4097),
        dict(CONFIG, batch_size=1024.0),
    ],
)
def test_invalid_configurations_are_rejected_before_any_fit(config, view, tmp_path, fitted):
    fits = fitted({})

    with pytest.raises(ValueError, match="Ridge declara su identidad"):
        ridge.run_ridge_window(view, output=tmp_path / "out", config=config)

    assert fits.calls == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "config",
    [
        dict(CONFIG, alphas=[float(value) for value in range(1, 9)]),
        dict(CONFIG, alphas=[1e-9]),
        dict(CONFIG, batch_size=1),
        dict(CONFIG, batch_size=4096),
    ],
)
def test_configurations_at_their_limits_are_accepted(config):
    assert ridge.checked_config(config) is config
