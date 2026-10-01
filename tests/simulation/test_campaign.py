"""Comparadores y referencias con validación separada y presupuesto fijado."""

import json
from pathlib import Path

import pytest

from mars_titan.episodes.worlds import WorldConfig, generate_world
from mars_titan.simulation.campaign import run_campaign
from mars_titan.simulation.market import MarketTape


def settings():
    config = json.loads(Path("configs/simulation/comparators.json").read_text())
    config["training"].update(
        total_steps=4, batch_size=2, rollout_steps=2, replay_capacity=8, warmup_steps=2
    )
    return config


def source(partition, seed):
    world = generate_world(
        WorldConfig(assets=2, sessions=10, context=6, partition=partition, seed=seed)
    )
    return MarketTape.from_world(world, lambda x: 0.002 * x["news"][:, 0])


def test_small_campaign_and_resume_are_technical(tmp_path):
    train, validation = source("train", 42), source("validation", 1042)
    result = run_campaign(train, [validation], tmp_path / "run", settings(), diagnostic=True)
    assert result["status"] == "completed"
    assert result["domain"] == "technical"
    assert result["training_runs"] == 6
    assert result["evaluations"] == 27
    assert result["final_test_opened"] is False
    recovered = run_campaign(
        train, [validation], tmp_path / "run", settings(), diagnostic=True, resume=True
    )
    assert result["identity"] == recovered["identity"]
    changed = settings()
    changed["environment"]["capital"] *= 2
    with pytest.raises(ValueError, match="identidad"):
        run_campaign(train, [validation], tmp_path / "run", changed, diagnostic=True, resume=True)


def test_validation_must_not_reuse_training(tmp_path):
    train = source("train", 42)
    with pytest.raises(ValueError):
        run_campaign(train, [train], tmp_path / "run", settings(), diagnostic=True)


def test_validation_rejects_repeated_generator_seed(tmp_path):
    with pytest.raises(ValueError, match="semillas"):
        run_campaign(
            source("train", 42),
            [source("validation", 42)],
            tmp_path / "run",
            settings(),
            diagnostic=True,
        )
