"""Los episodios mantienen cohortes y reinicios, sin concatenar historias."""

import numpy as np
import pytest

from mars_titan.episodes.windows import EpisodeView, resample_windows
from mars_titan.episodes.worlds import WorldConfig, generate_world


def test_resampling_retains_joint_cohorts_and_training_only():
    world = generate_world(WorldConfig(assets=6, sessions=60, context=8))
    plan = resample_windows(world, seed=42, episodes=3, decisions=10, warmup=8)
    assert plan == resample_windows(world, seed=42, episodes=3, decisions=10, warmup=8)
    for spec in plan:
        episode = EpisodeView(world, spec)
        assert len(episode) == 10
        assert episode(10) is None
        assert len(list(episode.warmup())) == 8
        assert episode(0)["asset_ids"] == world(spec.decision_start)["asset_ids"]
        np.testing.assert_array_equal(episode(0)["target"], world(spec.decision_start)["target"])
    validation = generate_world(WorldConfig(partition="validation"))
    with pytest.raises(ValueError, match="entrenamiento"):
        resample_windows(validation, seed=42, episodes=3, decisions=10, warmup=8)


def test_episode_rejects_changed_source_and_label_crossing_boundary():
    world = generate_world(WorldConfig(assets=3, sessions=60, context=8))
    spec = resample_windows(world, seed=42, episodes=1, decisions=10, warmup=8)[0]
    other = generate_world(WorldConfig(assets=3, sessions=60, context=8, seed=91))
    with pytest.raises(ValueError, match="fuente"):
        EpisodeView(other, spec)
    episode = EpisodeView(world, spec)
    world.times[spec.stop + world.config.context - 1] += 100 * 86400 * 1000000
    with pytest.raises(ValueError, match="límite"):
        episode(len(episode) - 1)
