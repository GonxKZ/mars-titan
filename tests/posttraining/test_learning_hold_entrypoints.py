"""Los puntos de entrada del postentrenamiento respetan la protección del aprendizaje."""

import json
from argparse import Namespace

import pytest

from mars_titan.posttraining import completion
from mars_titan.posttraining.queue import run_queue
from mars_titan.posttraining.run import run_case
from mars_titan.training.learning_hold import HOLD_ENV, LearningHoldError


class Untouchable:
    """Cualquier acceso a la fuente indica que se abrió antes de comprobar el bloqueo."""

    def __getattr__(self, name):
        raise AssertionError(f"Se ha leído {name} antes de comprobar el bloqueo")


def hold(tmp_path, monkeypatch, value):
    path = tmp_path / "hold.json"
    path.write_text(json.dumps({"training_allowed": value}))
    monkeypatch.setenv(HOLD_ENV, str(path))


def entry_points(tmp_path):
    output = tmp_path / "output"
    args = Namespace(
        reference=tmp_path / "reference",
        encoded=tmp_path / "encoded/manifest.json",
        tabular_config=tmp_path / "tabular.json",
        post_config=tmp_path / "post.json",
        output=output,
        after=None,
        fold="fold-000",
    )
    return output, {
        "run_case": lambda: run_case(Untouchable(), output, {}, Untouchable(), {}),
        "run_queue": lambda: run_queue(
            tmp_path / "missing.json", tmp_path / "r", tmp_path / "t", tmp_path / "e", output
        ),
        "run_completion": lambda: completion.run_completion(args, None),
        "tabular_stage": lambda: completion._stage(Namespace(**vars(args), stage="tabular")),
        "posttraining_stage": lambda: completion._stage(
            Namespace(**vars(args), stage="posttraining")
        ),
    }


@pytest.mark.parametrize(
    "name", ["run_case", "run_queue", "run_completion", "tabular_stage", "posttraining_stage"]
)
def test_entry_points_refuse_before_opening_sources_or_outputs(tmp_path, monkeypatch, name):
    hold(tmp_path, monkeypatch, False)
    output, calls = entry_points(tmp_path)
    with pytest.raises(LearningHoldError, match="Bloqueo de aprendizaje vigente"):
        calls[name]()
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["hold.json"]


@pytest.mark.parametrize("name", ["run_case", "run_queue", "run_completion"])
def test_permitted_learning_reaches_the_ordinary_checks(tmp_path, monkeypatch, name):
    hold(tmp_path, monkeypatch, True)
    _, calls = entry_points(tmp_path)
    with pytest.raises(Exception) as error:
        calls[name]()
    assert not isinstance(error.value, LearningHoldError)


def test_absent_hold_admits_and_ambiguous_hold_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(HOLD_ENV, str(tmp_path / "absent.json"))
    _, calls = entry_points(tmp_path)
    # Sin protección se llega a las comprobaciones ordinarias del dispositivo.
    with pytest.raises(ValueError, match="cuda:0"):
        calls["run_case"]()
    hold(tmp_path, monkeypatch, "no")
    with pytest.raises(ValueError, match="training_allowed"):
        calls["run_case"]()


def test_frozen_evaluation_stage_is_not_a_fit(tmp_path, monkeypatch):
    hold(tmp_path, monkeypatch, False)
    _, calls = entry_points(tmp_path)
    args = Namespace(
        reference=tmp_path / "reference",
        encoded=tmp_path / "encoded.json",
        tabular_config=tmp_path / "t.json",
        post_config=tmp_path / "p.json",
        output=tmp_path / "output",
        after=None,
        fold="fold-000",
        stage="evaluation",
    )
    # La evaluación congelada no ajusta parámetros. Falla después, al leer su referencia.
    with pytest.raises(Exception) as error:
        completion._stage(args)
    assert not isinstance(error.value, LearningHoldError)
    assert "Bloqueo" not in str(error.value)
