"""Presupuestos de aumento por cohortes y orden compartido de cada época."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .windows import EpisodeWindow, resample_windows
from .worlds import WorldConfig, generate_world


@dataclass(frozen=True)
class Visit:
    arm: str
    episode: int
    cohort: int
    reset: bool


def augmentation_windows(source, *, seed, decisions=16, warmup=16, fraction=0.25):
    """Redondear el 25 % hacia la siguiente cohorte completa, nunca partir empresas."""
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("La proporción adicional necesita un valor entre cero y uno")
    required = math.ceil(sum(row[1] for row in source.index) * fraction)
    selected, count = [], 0
    rng = np.random.default_rng(seed)
    while count < required:
        if len(selected) >= 100_000:
            raise ValueError("El aumento supera el presupuesto de episodios")
        spec = resample_windows(
            source, seed=int(rng.integers(0, 2**32)), episodes=1, decisions=decisions, warmup=warmup
        )[0]
        stop = spec.decision_start
        while stop < spec.stop and count < required:
            count += source.index[stop][1]
            stop += 1
        cutoff = source.index[stop][0] if stop < len(source) else spec.label_cutoff
        selected.append(replace(spec, stop=stop, label_cutoff=cutoff))
    return selected


def training_visits(source, windows, *, epoch, seed):
    """Recorrer todas las cohortes reales y después bloques adicionales con reinicios."""
    if source.partition != "train" or any(
        type(n) is not int or not 0 <= n < 2**32 for n in (epoch, seed)
    ):
        raise ValueError("La época solo admite entrenamiento y semillas válidas")
    if any(window.source_sha256 != source.manifest_sha256 for window in windows):
        raise ValueError("El aumento pertenece a otra fuente")
    rng = np.random.default_rng([seed, epoch])
    visits = [Visit("real", -1, int(i), True) for i in rng.permutation(len(source))]
    for episode in rng.permutation(len(windows)):
        window = windows[int(episode)]
        visits.extend(
            Visit("extra", int(episode), i, i == window.decision_start)
            for i in range(window.decision_start, window.stop)
        )
    return visits


def paired_world(source, window, *, seed, volatility=0.006, signal=0.002):
    """Generar el mismo número de filas adicionales y un calentamiento separado."""
    if source.partition != "train" or source.manifest_sha256 != window.source_sha256:
        raise ValueError("El mundo emparejado requiere un bloque de entrenamiento identificado")
    counts = [source.index[i][1] for i in range(window.start, window.stop)]
    context = source.shapes["prices"][0]
    active = (counts[0],) * (context - 1) + tuple(counts) + (counts[-1],)
    config = WorldConfig(
        assets=max(counts),
        sessions=len(active),
        context=context,
        seed=seed,
        volatility=volatility,
        signal=signal,
        active_assets=active,
    )
    world = generate_world(config)
    counterpart = EpisodeWindow(
        world.manifest_sha256,
        0,
        window.decision_start - window.start,
        len(world),
        int(world.times[-1]),
        "train",
        "synthetic",
    )
    return world, counterpart


def fit_volatility(source):
    """Estimar una escala ilustrativa solo con etiquetas maduras de entrenamiento."""
    if source.partition != "train":
        raise ValueError("La calibración del generador solo usa entrenamiento")
    count, mean, m2 = 0, 0.0, 0.0
    for i in range(len(source)):
        values = source(i)["target"]
        if not np.isfinite(values).all():
            raise ValueError("La calibración contiene etiquetas no finitas")
        n, block_mean = len(values), float(np.mean(values))
        delta = block_mean - mean
        m2 += float(np.sum((values - block_mean) ** 2)) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if count < 2:
        raise ValueError("La calibración necesita al menos dos etiquetas")
    observed = math.sqrt(m2 / (count - 1))
    return dict(
        fit_partition="train",
        source_sha256=source.manifest_sha256,
        samples=count,
        residual_std=observed,
        volatility=float(np.clip(observed, 0.0001, 0.03)),
        method="training_residual_standard_deviation",
        mean=mean,
    )
