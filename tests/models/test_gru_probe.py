"""Rechazos tempranos de la sonda, sin iniciar una carga CUDA."""

import pytest


@pytest.mark.parametrize("epochs", [0, 1, 11])
def test_invalid_epoch_budget_fails_before_reading_data(tmp_path, epochs):
    from mars_titan.gru_probe import run_gru_probe

    with pytest.raises(ValueError, match="épocas"):
        run_gru_probe(tmp_path, tmp_path, tmp_path / "run", tmp_path / "report", epochs=epochs)
    assert not (tmp_path / "run").exists()


def test_missing_deterministic_configuration_fails_before_training(tmp_path, monkeypatch):
    from mars_titan.gru_probe import run_gru_probe

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        run_gru_probe(tmp_path, tmp_path, tmp_path / "run", tmp_path / "report")
    assert not (tmp_path / "run").exists()
