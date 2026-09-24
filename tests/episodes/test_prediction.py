"""El contrato de episodios conserva acciones, maduración y recuperación predictivas."""

import numpy as np

from mars_titan.environments.actions import ActionGrid
from mars_titan.episodes.prediction import EpisodePredictionEnv
from mars_titan.episodes.windows import EpisodeView, resample_windows
from mars_titan.episodes.worlds import WorldConfig, generate_world


def test_prediction_episode_resumes_same_matured_reward_and_resets():
    world = generate_world(WorldConfig(assets=3, sessions=40, context=8))
    window = resample_windows(world, seed=42, episodes=1, decisions=5, warmup=4)[0]
    view = EpisodeView(world, window)
    grid = ActionGrid.fit(
        np.concatenate([world(i)["target"] for i in range(len(world))]),
        source_sha256="a" * 64,
        partition="train",
    )
    first, second = EpisodePredictionEnv(view, grid), EpisodePredictionEnv(view, grid)
    observation, _ = first.reset(seed=42)
    assert "target" not in observation
    action = np.full(3, 10, dtype=np.int64)
    first.step(action)
    state = first.snapshot()
    second.restore(state)
    a, b = first.step(action), second.step(action)
    assert a[1:] == b[1:]
    first.reset(seed=42)
    assert first.snapshot()["environment"]["pending"] == []
