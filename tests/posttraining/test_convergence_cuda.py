"""Selección v3 y recuperación CUDA con validación real sobre una vista pequeña."""

import os

import numpy as np
import pytest
import torch
from test_continuation_selection import assert_same_state
from test_run import options, state
from test_temporal_inputs import ordered_view

from mars_titan.data.storage import sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.environments.corpus_source import ParquetCohortSource
from mars_titan.episodes.parents import ParentCache
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.posttraining.inputs import PairedInputs, fit_normalization
from mars_titan.posttraining.parents import FrozenParent
from mars_titan.posttraining.run import run_case
from mars_titan.training.checkpoints import StopRequest, load_training_state
from mars_titan.training.experiment_resources import GpuLease
from tests.training.test_temporal_corpus import inputs as inputs


@pytest.mark.skipif(
    os.environ.get("MARS_TITAN_CUDA_INTEGRATION") != "1",
    reason="Requiere una ventana CUDA exclusiva y activación explícita",
)
@pytest.mark.parametrize("mode", ["klpo_mc", "neural_mae"])
def test_version_three_recovers_cuda_state_and_selects_real_validation(
    inputs, tmp_path, monkeypatch, mode
):
    _, ordered, _ = ordered_view(inputs, tmp_path)
    with (
        GpuLease() as lease,
        ParquetCohortSource(ordered, partition="train") as train,
        ParquetCohortSource(ordered, partition="validation") as validation,
    ):
        torch.manual_seed(42)
        model = MultimodalReference(
            "gru",
            {name: shape[-1] for name, shape in train.shapes.items()},
            context=train.shapes["prices"][0],
            hidden_size=32,
            layers=1,
            dropout=0.1,
        )
        checkpoint = tmp_path / "initial-parent.pt"
        torch.save(model.state_dict(), checkpoint)
        parent = FrozenParent(
            model,
            dict(
                model="gru",
                checkpoint_sha256=sha256(checkpoint),
                source_sha256=train.source_sha256,
                counts=train.population_counts,
            ),
            train.shapes,
            "cuda:0",
        )
        with ParentCache(
            tmp_path / "parent.sqlite",
            sha256(checkpoint),
            {"fixture": "cuda_convergence"},
            parent.predict,
        ) as cache:
            data = PairedInputs(train, validation, cache)
            targets = np.concatenate([train(index)["target"] for index in range(len(train))])
            grid = ActionGrid.fit(targets, source_sha256=train.source_sha256, partition="train")
            case = dict(
                options(mode),
                condition="real",
                epochs=4,
                selection=dict(
                    version=3,
                    metric="session_mae",
                    minimum_epochs=1,
                    patience=2,
                    min_delta=0.0,
                ),
            )
            kwargs = dict(
                dataset=data,
                case=case,
                grid=grid,
                normalization=fit_normalization(data),
                parent=parent,
                batch_size=1,
                device="cuda:0",
                lease=lease,
            )
            original = {name: value.clone() for name, value in parent.model.state_dict().items()}
            whole = run_case(output=tmp_path / "whole", **kwargs)
            stop = StopRequest()
            step = torch.optim.AdamW.step

            def pause_after_update(optimizer, *args, **kwargs):
                result = step(optimizer, *args, **kwargs)
                stop.request_stop()
                return result

            with monkeypatch.context() as patch:
                patch.setattr(torch.optim.AdamW, "step", pause_after_update)
                paused = run_case(output=tmp_path / "split", stop=stop, **kwargs)
            assert paused["status"] == "paused" and paused["global_step"] == 1
            assert "stop_reason" not in paused
            resumed = run_case(output=tmp_path / "split", resume=True, **kwargs)
            assert_same_state(state(tmp_path / "whole"), state(tmp_path / "split"))
            assert_same_state(original, parent.model.state_dict())
            assert whole["baseline"] == resumed["baseline"]
            assert whole["predictions"] == resumed["predictions"]
            assert resumed["status"] == "completed"
            scores = [resumed["baseline"]["session_mae"]] + [
                epoch["validation"]["session_mae"] for epoch in resumed["epochs"]
            ]
            assert resumed["best_score"] == min(scores)
            assert resumed["best_epoch"] == scores.index(min(scores))
            assert resumed["predictions"]["validation"]["metrics"]["session_mae"] == min(scores)
            best = load_training_state(
                tmp_path / "split/checkpoints",
                expected_identity=resumed["identity"],
                selection="best",
            )
            assert best["epoch"] == resumed["best_epoch"]
            assert resumed["stop_reason"] == (
                "validation_plateau" if resumed["selection"]["should_stop"] else "budget_exhausted"
            )
            assert resumed["last_epoch_improved"] == resumed["selection"]["last_improved"]
            lease.check()
