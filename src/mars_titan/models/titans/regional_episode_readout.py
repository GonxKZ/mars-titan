"""Resultados regionales de lectura, separados de los pesos del lector anterior."""

from dataclasses import dataclass

import torch

from .evidence_bridge import BridgeRead


@dataclass(frozen=True)
class RegionalEpisodeRead:
    values: torch.Tensor
    presence: torch.Tensor
    bridge: BridgeRead | None


@dataclass(frozen=True)
class BridgedReadoutResult:
    state: torch.Tensor
    reads: tuple[RegionalEpisodeRead, ...]


def regional_read(model, state, snapshot, context_id, cutoff, positions):
    if snapshot is None:
        return (
            RegionalEpisodeRead(
                torch.zeros_like(state),
                torch.zeros((len(state), 1), dtype=torch.bool, device=state.device),
                None,
            ),
            None,
            None,
        )
    bridge = model.evidence_bridge(
        state, snapshot, context_id=context_id, cutoff=cutoff, positions=positions
    )
    return (
        RegionalEpisodeRead(
            bridge.values,
            torch.full(
                (len(state), 1),
                snapshot.count > 0 and any(model.evidence_bridge.config.active_views()),
                dtype=torch.bool,
                device=state.device,
            ),
            bridge,
        ),
        bridge.selected_positions,
        None,
    )
