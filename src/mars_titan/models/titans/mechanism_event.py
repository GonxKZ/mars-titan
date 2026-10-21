"""Estados y propuestas provisionales del consumidor de mecanismos."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from mars_titan.calibration.cohort_feedback import CohortFeedback

from .mechanism_catalog import ResolvedMechanisms


@dataclass(frozen=True)
class MechanismState:
    model_id: str
    cutoff: int | None
    context: object
    calibration: CohortFeedback | None
    terminal: object = None


@dataclass(frozen=True)
class MechanismEvent:
    model_id: str
    context_id: str
    cutoff: int
    flow_ids: tuple[str, ...]
    matures_at: tuple[int, ...]
    plans: tuple[tuple[str, ResolvedMechanisms], ...]
    context: object
    calibration: CohortFeedback | None
    snapshot: object
    warmup: bool
    rows: tuple = ()


@dataclass(frozen=True)
class MechanismBlock:
    next_state: object
    event: MechanismEvent


@dataclass(frozen=True)
class MechanismEmission:
    state: MechanismState
    flow_ids: tuple[str, ...]
    point_predictions: np.ndarray | None
    quantiles: np.ndarray | None
    raw_quantiles: np.ndarray | None
    rows: tuple
    work: object
    publication_status: str = "proposed_not_durable"


def frozen_array(values):
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def frozen_row(**values):
    return MappingProxyType(values)
