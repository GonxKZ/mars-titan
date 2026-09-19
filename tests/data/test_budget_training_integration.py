"""Contrato completo de la sonda supervisada con datos sintéticos y CUDA real."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mars_titan.data.preparation import prepare_asset
from mars_titan.data.samples import sample_table
from mars_titan.data.storage import sha256
from mars_titan.data.temporal import MarketClock


@pytest.fixture
def cuda_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("nvidia-smi unavailable, no CPU fallback")
    if result.returncode:
        pytest.skip(f"nvidia-smi failed, no CPU fallback: {result.stderr.strip()}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable, no CPU fallback")
    torch.cuda.set_device("cuda:0")
    yield torch
    torch.cuda.empty_cache()


@pytest.fixture
def synthetic_budget_inputs(tmp_path, monkeypatch, cuda_runtime):
    monkeypatch.chdir(tmp_path)
    clock = MarketClock("US", "2021-07-01", "2024-01-03")
    sessions = [day.isoformat() for day in clock.days]
    market_returns = np.resize(np.array([-0.02, 0.01, 0.03, -0.01]), len(sessions))
    source = Path("dataset")
    market_path = source / "time_series/S&P500_time_series/spy.csv"
    market_path.parent.mkdir(parents=True)

    def write_prices(path, returns):
        close = 100 * (1 + returns)
        pd.DataFrame(
            {
                "Date": sessions,
                "Open": 100.0,
                "High": np.maximum(100.0, close) + 1,
                "Low": np.minimum(100.0, close) - 1,
                "Close": close,
                "Volume": 1000.0,
            }
        ).to_csv(path, index=False)

    write_prices(market_path, market_returns)
    train_days = [day for day in sessions if day.startswith("2022-")][:17]
    validation_days = [day for day in sessions if day.startswith("2023-")][:5]
    sample_days = [sessions[63], *train_days, "2022-12-30", *validation_days, "2023-12-29"]
    sample_days.append("2024-01-02")
    prepared = tmp_path / "prepared"
    inputs = [market_path]
    sample_paths = []
    # El contrato admite paneles de cuatro activos, pero exige 22 activos preparados.
    for number in range(22):
        symbol = f"S{number:02}"
        returns = 0.003 + 1.5 * market_returns
        returns[sessions.index("2023-01-04")] += 0.02
        raw_prices = source / f"{symbol}.csv"
        raw_news = source / f"{symbol}.jsonl"
        raw_facts = source / f"{symbol}.json"
        write_prices(raw_prices, returns)
        raw_news.write_text(
            json.dumps(
                {"Date": sessions[62], "Article": "Synthetic fixture news", "Stock_symbol": symbol}
            )
            + "\n"
        )
        fact = {"end": "2021-03-31", "filed": "2021-06-30", "val": 100, "accn": "fixture"}
        raw_facts.write_text(
            json.dumps(
                {"filings": [{"facts": {"us-gaap": {"Assets": {"units": {"USD": [fact]}}}}}]}
            )
        )
        asset = {
            "symbol": symbol,
            "paths": {
                "prices": [raw_prices.name],
                "news": [raw_news.name],
                "fundamentals": [raw_facts.name],
            },
        }
        manifest = prepare_asset(source, prepared, asset, clock)
        sample_path = prepared / "samples/US" / symbol / "samples.parquet"
        sample_path.parent.mkdir(parents=True)
        rows = [
            {
                "session": day,
                "prediction_at": clock.decision(day),
                "price_end_index": sessions.index(day),
                "news": [0.1 + number / 100] * 384,
                "charts": [0.2 + number / 100] * 512,
                "fundamentals": [1.0] * 8 + [1.0] * 8 + [0.5] * 8,
                "macro": [1.0, 2.0, 1.0, 1.0, 0.5, 0.5],
            }
            for day in sample_days
        ]
        # La fila de 2024 debe quedar fuera antes de validar o consumir sus entradas.
        rows[-1]["news"][0] = float("nan")
        rows[-1]["price_end_index"] = -1
        pq.write_table(sample_table(rows), sample_path)
        sample_manifest = sample_path.parent / "manifest.json"
        sample_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "synthetic_test_fixture",
                    "market": "US",
                    "symbol": symbol,
                    "prepared_fingerprint": manifest["fingerprint"],
                    "samples_sha256": sha256(sample_path),
                }
            )
        )
        inputs.extend(
            [
                raw_prices,
                raw_news,
                raw_facts,
                prepared / "US" / symbol / "prices.parquet",
                prepared / "US" / symbol / "manifest.json",
                sample_path,
                sample_manifest,
            ]
        )
        sample_paths.append(sample_path)
    return {
        "prepared": prepared,
        "sample_paths": sample_paths,
        "input_hashes": {path: sha256(path) for path in inputs},
        "train_days": train_days,
        "validation_days": validation_days,
    }


def test_budget_grid_generates_labels_trains_and_resumes_without_opening_2024(
    tmp_path, synthetic_budget_inputs, cuda_runtime
):
    from torch.utils.data import DataLoader

    from mars_titan import budget_training
    from mars_titan.profiling import CostProbe

    torch = cuda_runtime
    fixture = synthetic_budget_inputs
    prepared = fixture["prepared"]
    report_path = tmp_path / "report.json"
    output = tmp_path / "run"
    report = budget_training.train_budget_grid(
        prepared, report_path, output, epochs=2, panel_sizes=(4,), kinds=("mlp", "gru")
    )
    assert json.loads(report_path.read_text()) == report
    assert report["device"] == "cuda:0"
    assert report["final_test_opened"] is False
    assert [case["kind"] for case in report["cases"]] == ["mlp", "gru"]
    assert len(report["target_audit"]["assets"]) == 22
    package = Path(budget_training.__file__).parent
    for relative in (
        "budget_training.py",
        "data/budget_targets.py",
        "data/streaming.py",
        "profiling.py",
    ):
        actual = package / relative
        assert (
            report["input_hashes"][f"src/mars_titan/{relative}"]
            == hashlib.sha256(actual.read_bytes()).hexdigest()
        )
        assert str(actual) not in report["input_hashes"]
    expected_exclusions = {
        "insufficient_history": 1,
        "target_crosses_partition_boundary": 1,
        "target_after_cutoff": 1,
    }
    targets = {}
    for symbol in ["S00", "S01", "S02", "S03"]:
        audit = report["target_audit"]["assets"][symbol]
        assert {key: audit[key] for key in ("train", "validation", "excluded")} == {
            "train": 17,
            "validation": 5,
            "excluded": 3,
        }
        assert audit["excluded_reasons"] == expected_exclusions
        label_path = output / "targets" / f"{symbol}-targets.parquet"
        labels = pq.read_table(label_path).to_pandas()
        assert labels.prediction_at.dt.year.max() == 2023
        known = labels.set_index(labels.prediction_at.dt.strftime("%Y-%m-%d"))
        shock = known.loc["2023-01-03"]
        assert shock.alpha == pytest.approx(0.003)
        assert shock.beta == pytest.approx(1.5)
        assert shock.target == pytest.approx(0.02)
        assert shock.history_pairs == 252
        assert shock.target_available_at.date().isoformat() == "2023-01-04"
        targets[f"US/{symbol}"] = {
            known.loc[day].prediction_at.isoformat(): (float(known.loc[day].target), partition)
            for partition in ("train", "validation")
            for day in fixture[f"{partition}_days"]
        }
    for case in report["cases"]:
        assert case["symbols"] == ["S00", "S01", "S02", "S03"]
        assert case["effective_training_assets"] == case["effective_validation_assets"] == 4
        assert case["dimensions"] == {
            "prices": 5,
            "news": 384,
            "charts": 512,
            "fundamentals": 24,
            "macro": 6,
        }
        assert case["resume_check"]["exact_weights"] is True
        assert case["resume_check"]["from_epoch"] == 1
        assert len(case["epochs"]) == 2
        for epoch in case["epochs"]:
            assert epoch["train"]["samples"] == 68 and epoch["train"]["steps"] == 2
            assert epoch["validation"]["samples"] == 20 and epoch["validation"]["steps"] == 1
            assert np.isfinite(epoch["train"]["diagnostic_mse"])
            assert np.isfinite(epoch["validation"]["diagnostic_mse"])
            assert sha256(Path(epoch["checkpoint"])) == epoch["checkpoint_sha256"]
        first = torch.load(case["epochs"][0]["checkpoint"], map_location="cpu", weights_only=True)
        second = torch.load(case["epochs"][1]["checkpoint"], map_location="cpu", weights_only=True)
        assert first["next_epoch"] == 1 and second["next_epoch"] == 2
        for prefix in [
            "price_encoder",
            "encoders.news",
            "encoders.charts",
            "encoders.fundamentals",
            "encoders.macro",
        ]:
            assert any(
                not torch.equal(value, second["model"][name])
                for name, value in first["model"].items()
                if name.startswith(prefix + ".")
            )
        model = CostProbe(case["kind"], case["dimensions"]).to("cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        assert (
            budget_training.load_checkpoint(
                case["epochs"][0]["checkpoint"],
                model,
                optimizer,
                config=first["config"],
                hashes=report["input_hashes"],
            )
            == 1
        )
        for partition in ("train", "validation"):
            loader = DataLoader(
                budget_training.SupervisedWorkload(
                    fixture["sample_paths"][:4], prepared, targets, partition
                ),
                batch_size=64,
                num_workers=0,
                pin_memory=True,
            )
            budget_training.run_epoch(
                model, optimizer if partition == "train" else None, loader, torch.device("cuda:0")
            )
        assert all(
            torch.equal(value.cpu(), second["model"][name])
            for name, value in model.state_dict().items()
        )
        assert torch.equal(torch.get_rng_state(), second["rng_torch"])
    assert all(sha256(path) == digest for path, digest in fixture["input_hashes"].items())
    confirmed_report = report_path.read_bytes()
    with pytest.raises(ValueError, match="fresh output"):
        budget_training.train_budget_grid(prepared, report_path, output, epochs=2, panel_sizes=(4,))
    assert report_path.read_bytes() == confirmed_report
