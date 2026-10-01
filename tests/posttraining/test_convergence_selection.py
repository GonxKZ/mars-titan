"""Continuación real con mínimo y recibos de parada recuperables en CPU."""

import pytest
import torch
from test_continuation_selection import (
    assert_same_state,
    configuration,
    controlled_scores,
    selected,
)
from test_inputs import dataset
from test_run import state

from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.posttraining.run import run_case, validate_case
from mars_titan.posttraining.selection import selection_policy


def convergence(data, *, epochs=8):
    kwargs = configuration(data, epochs=epochs, patience=2)
    kwargs["case"]["selection"].update(version=3, minimum_epochs=3)
    return kwargs


@pytest.mark.parametrize(
    "mode",
    [
        "reinforce",
        "expected",
        "mae",
        "klpo_full",
        "klpo_mc",
        "klpo_exact",
        "neural_mae",
        "neural_mse",
    ],
)
def test_new_policy_waits_past_minimum_and_recovers_exactly(tmp_path, monkeypatch, mode):
    data = dataset(tmp_path)
    kwargs = convergence(data)
    kwargs["case"]["mode"] = mode
    if mode.startswith("neural_"):
        torch.manual_seed(42)
        model = MultimodalReference(
            "gru",
            {key: shape[-1] for key, shape in data.shapes.items()},
            context=4,
            hidden_size=32,
            layers=1,
            dropout=0.1,
        )
        parent = FrozenParent(
            model, dict(model="gru", checkpoint_sha256="a" * 64), data.shapes, "cpu"
        )
        data.parent.predictor = parent.predict
        kwargs["parent"] = parent
    scores = (0.1,) + (0.2,) * 5
    controlled_scores(monkeypatch, scores)
    whole = run_case(output=tmp_path / "whole", **kwargs)
    controlled_scores(monkeypatch, scores)
    paused = run_case(output=tmp_path / "split", max_updates=9, **kwargs)
    assert paused["status"] == "paused"
    assert "stop_reason" not in paused
    resumed = run_case(output=tmp_path / "split", resume=True, **kwargs)
    assert resumed["stop_reason"] == "validation_plateau"
    assert resumed["last_epoch_improved"] is False
    assert resumed["best_epoch"] == 0
    assert len(resumed["epochs"]) == 5
    assert resumed["stopped_early"] is True
    assert selected(tmp_path / "split")["global_step"] == 0
    assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
    assert whole["predictions"] == resumed["predictions"]
    data.parent.close()


def test_budget_exhaustion_does_not_claim_plateau_while_improving(tmp_path, monkeypatch):
    data = dataset(tmp_path)
    controlled_scores(monkeypatch, (0.9, 0.8, 0.7, 0.6, 0.5))
    result = run_case(output=tmp_path / "run", **convergence(data, epochs=4))
    assert result["stop_reason"] == "budget_exhausted"
    assert result["last_epoch_improved"] is True
    assert result["best_epoch"] == 4
    assert result["stopped_early"] is False
    data.parent.close()


@pytest.mark.parametrize(
    "change",
    [
        {"minimum_epochs": True},
        {"minimum_epochs": -1},
        {"minimum_epochs": 8},
        {"minimum_epochs": float("inf")},
        {"patience": None},
        {"version": 2},
    ],
)
def test_new_policy_rejects_invalid_or_legacy_mixed_contract(change):
    case = dict(
        condition="real",
        epochs=8,
        selection=dict(
            version=3,
            metric="session_mae",
            minimum_epochs=3,
            patience=2,
            min_delta=0.0,
        ),
    )
    case["selection"].update(change)
    with pytest.raises(ValueError):
        selection_policy(case)


@pytest.mark.parametrize("condition", ["real_resampled", "real_synthetic"])
def test_convergence_cannot_silently_break_paired_budgets(condition):
    with pytest.raises(ValueError):
        selection_policy(
            dict(
                condition=condition,
                epochs=50,
                selection=dict(
                    version=3,
                    metric="session_mae",
                    minimum_epochs=5,
                    patience=8,
                    min_delta=1e-5,
                ),
            )
        )


def test_version_three_is_the_only_case_that_expands_the_epoch_ceiling(tmp_path):
    data = dataset(tmp_path)
    case = convergence(data, epochs=50)["case"]
    validate_case(case)
    case["epochs"] = 51
    with pytest.raises(ValueError):
        validate_case(case)
    case["selection"].pop("minimum_epochs")
    case["selection"]["version"] = 2
    case["epochs"] = 31
    with pytest.raises(ValueError):
        validate_case(case)
    data.parent.close()
