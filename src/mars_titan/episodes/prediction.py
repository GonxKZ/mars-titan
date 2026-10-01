"""Reutilizar las 21 acciones y su maduración en episodios con identidad propia."""

from dataclasses import asdict

import gymnasium as gym

from mars_titan.environments.prediction import CausalPredictionEnv


class EpisodePredictionEnv(gym.Wrapper):
    def __init__(self, episode, grid):
        # La rejilla conserva su fuente de ajuste, distinta de las entradas del episodio.
        super().__init__(
            CausalPredictionEnv(
                episode,
                source_sha256=grid.source_sha256,
                grid=grid,
                shapes=episode.shapes,
                max_assets=episode.max_assets,
                partition=episode.partition,
            )
        )
        self.episode = episode
        self.episode_identity = asdict(episode.window)

    def snapshot(self):
        return dict(
            schema_version=1, episode=self.episode_identity, environment=self.env.snapshot()
        )

    def restore(self, state):
        if state.get("schema_version") != 1 or state.get("episode") != self.episode_identity:
            raise ValueError("El estado pertenece a otro episodio")
        self.env.restore(state["environment"])
