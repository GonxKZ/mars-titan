"""Ventanas por cohortes completas, con calentamiento y reinicio explícitos."""

from dataclasses import dataclass

import numpy as np

from mars_titan.environments.cohorts import FINAL_TEST_START_US, VALIDATION_START_US, read_cohort


@dataclass(frozen=True)
class EpisodeWindow:
    source_sha256: str
    start: int
    decision_start: int
    stop: int
    label_cutoff: int
    partition: str
    origin: str = "resampled"


class EpisodeView:
    """Presentar una ventana de una fuente compartida sin duplicar sus modalidades."""

    def __init__(self, source, window):
        if source.manifest_sha256 != window.source_sha256:
            raise ValueError("El episodio pertenece a otra fuente")
        if (
            window.partition != source.partition
            or window.origin not in {"real", "resampled", "synthetic"}
            or not 0 <= window.start < window.decision_start < window.stop <= len(source)
            or any(
                type(getattr(window, key)) is not int
                for key in ("start", "decision_start", "stop", "label_cutoff")
            )
        ):
            raise ValueError("Los límites del episodio no corresponden a la fuente")
        self.source, self.window = source, window
        limit = VALIDATION_START_US if window.partition == "train" else FINAL_TEST_START_US
        if not 0 < window.label_cutoff < limit:
            raise ValueError("El límite del episodio cruza su partición")
        self.shapes, self.max_assets, self.partition = (
            source.shapes,
            source.max_assets,
            source.partition,
        )

    def __len__(self):
        return self.window.stop - self.window.decision_start

    def warmup(self):
        for i in range(self.window.start, self.window.decision_start):
            raw = self.source(i)
            yield {
                key: raw[key] for key in ("prediction_at", "asset_ids", "inputs", "available_at")
            }

    def __call__(self, position):
        if type(position) is not int or not 0 <= position <= len(self):
            raise ValueError("El cursor cruza el límite del episodio")
        if position == len(self):
            return None
        raw = self.source(self.window.decision_start + position)
        checked = read_cohort(raw, self.shapes, self.max_assets, 64 * 1024**2)
        if (checked["target_available_at"] > self.window.label_cutoff).any():
            raise ValueError("La etiqueta cruza el límite del episodio")
        return {key: value for key, value in checked.items() if key != "sha256"}


def resample_windows(source, *, seed, episodes, decisions, warmup):
    if source.partition != "train":
        raise ValueError("Solo se remuestrean bloques de entrenamiento")
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or any(type(v) is not int or not 1 <= v <= 100_000 for v in (episodes, decisions, warmup))
        or decisions + warmup > len(source)
    ):
        raise ValueError("El remuestreo necesita un presupuesto compatible con la fuente")
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(episodes):
        start = int(rng.integers(0, len(source) - decisions - warmup + 1))
        stop = start + decisions + warmup
        cutoff = source.index[stop][0] if stop < len(source) else VALIDATION_START_US - 1
        result.append(
            EpisodeWindow(
                source.manifest_sha256, start, start + warmup, stop, cutoff, source.partition
            )
        )
    return result
