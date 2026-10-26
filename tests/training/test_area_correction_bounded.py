"""Correcciones por áreas y núcleo sobre un padre con un mes de ajuste, sin modificar pesos."""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from mars_titan.training.mars_titan_correction import run_correction_window
from tests.training.test_titans_walk_forward import recipe, views, window


@pytest.fixture(autouse=True)
def permitted_doubles(learning_doubles):
    """Solo se usa el registrador de gradientes del padre y selección con datos de prueba."""


@pytest.fixture(autouse=True)
def backend():
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    yield
    torch.backends.mha.set_fastpath_enabled(previous)


@pytest.mark.parametrize("arm,case", [("area_memory", "eta25e-2"), ("nadaraya_watson", "tau1e-1")])
def test_correction_inherits_bounded_parent_and_keeps_every_predicted_row(tmp_path, arm, case):
    view, protocol = views(tmp_path / "data")
    parent = tmp_path / "parent"
    plan = recipe(tmp_path, walk_forward=dict(warmup_months=1, fit_months=1))
    expected, _ = window(view, protocol, plan, parent)
    config = json.loads(Path("configs/experiments/redundant-area-memory-v1.json").read_text())[
        "arms"
    ][arm]
    result = run_correction_window(
        view,
        parent,
        config["recipe"],
        components=config["components"],
        seed=42,
        output=tmp_path / "correction",
        search_case=case,
        device="cpu",
        indices=tmp_path / "indices",
    )
    assert result["status"] == "completed"
    assert result["identity"]["memory_policy"]["fit_months"] == 1
    for partition in ("validation", "calibration", "evaluation"):
        assert result["identity"]["phases"][partition] == expected["identity"]["phases"][partition]
        original = pq.read_table(parent / f"{partition}-predictions.parquet").sort_by("sample_id")
        actual = pq.read_table(
            tmp_path / "correction" / f"{partition}-predictions.parquet"
        ).sort_by("sample_id")
        for name in ("sample_id", "prediction_at", "target"):
            assert original[name].equals(actual[name])
