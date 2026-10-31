"""Supresión E2 del soporte común en todas las vistas de un lector regional K1."""

import torch
from torch import nn

from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.titans.bridge_hierarchy import hierarchy_mix
from mars_titan.models.titans.bridge_recurrence import adaptive_mix, nonlinear_mix
from mars_titan.models.titans.episodic_readout import (
    EpisodicReadout,
    _normalize,
    apply_episodic_readout,
)
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.evidence_bridge import SharedEvidenceBridge
from mars_titan.models.titans.financial import PreparedDecisions
from mars_titan.models.titans.regional_episode_readout import (
    BridgedReadoutResult,
    RegionalEpisodeRead,
)
from mars_titan.models.titans.state import check_finite, deferred_checks

from .spaced_utility import _require
from .spaced_utility_readout import SpacedUtilityDiagnostics, _finish_prediction, _metadata


def _validate(prepared, head, extension, snapshot, cutoff, differentiable, mixed):
    _require(
        isinstance(prepared, PreparedDecisions)
        and type(extension) is EpisodicReadout
        and type(snapshot) is EpisodeSnapshot
        and type(extension.evidence_bridge) is SharedEvidenceBridge,
        "E2 regional necesita preparación, instantánea y lector regional identificados",
    )
    _require(
        extension.config.mode == "bank"
        and extension.config.refinements == 1
        and not extension.training
        and not extension.evidence_bridge.training
        and differentiable is False,
        "El productor E2 regional necesita bank K1 en eval y sin diferenciación",
    )
    _require(
        type(head) in (nn.Linear, QuantileHead)
        and not head.training
        and isinstance(head, QuantileHead) == (prepared.quantiles is not None)
        and head.in_features == extension.config.hidden_size
        and head.out_features == (5 if isinstance(head, QuantileHead) else 1),
        "La cabeza E2 regional debe conservar su forma escalar o cuantílica y eval",
    )
    base = prepared.working_state
    extension._state(base)
    _require(
        all(p.dtype == base.dtype and p.device == base.device for p in head.parameters())
        and not torch.is_autocast_enabled(base.device.type),
        "E2 regional requiere precisión y dispositivo comunes, sin autocast",
    )
    _metadata(
        prepared.next_state.flow_ids,
        prepared.next_state.last_prediction_at,
        cutoff,
        len(base),
        allow_mixed_markets=mixed,
    )


def _pool(bridge, base, snapshot, positions):
    """Recalcular una consulta y un pool para todos los diagnósticos, sin otra búsqueda."""
    batch, count = positions.shape
    keys, values, labels = snapshot._values[:3]
    flat = positions.flatten()
    selected_keys = keys.index_select(0, flat).reshape(batch, count, 64).to(base.dtype)
    selected = torch.cat(
        (values.index_select(0, flat), labels.index_select(0, flat).to(base.dtype).unsqueeze(-1)),
        -1,
    )
    pool = bridge.value(selected).reshape(batch, count, bridge.config.hidden_size)
    raw = bridge.query(base)
    queries = _normalize(raw[:, None] + torch.einsum("bh,rhd->brd", base, bridge.region_query))
    logits = torch.bmm(queries, selected_keys.transpose(1, 2)) / bridge.config.temperature
    for tensor, name in (
        (pool, "El pool E2 regional"),
        (raw, "La consulta E2 regional"),
        (logits, "Los logits E2 regionales"),
    ):
        check_finite(tensor, name)
    return pool, logits


def _deleted_values(bridge, pool, logits, read, available, column):
    masked = logits.clone()
    masked[:, :, column] = -torch.inf
    weights = masked.softmax(-1) * available[:, :, None]
    mixed = torch.bmm(weights, pool)
    transformed = torch.einsum("brh,rhd->brd", mixed, bridge.region_value)
    check_finite(transformed, "La transformación regional E2")
    local = transformed.tanh()
    config, graph = bridge.config, read.bridge_weights
    if config.hierarchy is not None:
        local = hierarchy_mix(
            local,
            read.hierarchy_weights,
            bridge.region_value,
            available,
            local_steps=config.hierarchy.local_steps,
            gain=config.bridge_gain,
        )[:, :4]
    elif config.recurrence is None:
        for _ in range(config.cycles):
            local = (1 - config.bridge_gain) * local + config.bridge_gain * torch.bmm(graph, local)
    elif config.recurrence.stopping is None:
        local = nonlinear_mix(
            local,
            graph,
            bridge.region_value,
            available,
            steps=config.recurrence.steps,
            gain=config.bridge_gain,
        )
    else:
        local = adaptive_mix(
            local,
            graph,
            bridge.region_value,
            available,
            steps=config.recurrence.steps,
            gain=config.bridge_gain,
            stopping=config.recurrence.stopping,
        ).values
    values = bridge.output(local.reshape(len(pool), -1))
    check_finite(values, "La lectura regional E2 suprimida")
    return values


def apply_regional_spaced_utility_readout(
    prepared,
    head,
    extension=None,
    snapshot=None,
    *,
    enabled=False,
    context_id=None,
    cutoff=None,
    differentiable=False,
    allow_mixed_markets=False,
):
    """Conservar la emisión ordinaria y retirar cada ID de todas sus vistas, sin rellenar.

    La ruta activa mantiene el lector original intacto. Reproyecta su pool una vez para
    los diagnósticos, compartiéndolo entre supresiones. No modifica parámetros ni estados.
    """
    _require(type(enabled) is bool, "La activación regional E2 debe ser booleana")
    _require(type(allow_mixed_markets) is bool, "La agrupación regional E2 debe ser booleana")
    if not enabled:
        return apply_episodic_readout(
            prepared,
            head,
            extension,
            snapshot,
            context_id=context_id,
            cutoff=cutoff,
            differentiable=differentiable,
        ), None
    _validate(prepared, head, extension, snapshot, cutoff, differentiable, allow_mixed_markets)
    count = min(snapshot.count, extension.config.neighbors)
    base, bridge = prepared.working_state.detach(), extension.evidence_bridge
    with torch.no_grad(), deferred_checks():
        extension._validate_read(base, snapshot, context_id, cutoff, refinements=count + 1)
        prediction = apply_episodic_readout(
            prepared, head, extension, snapshot, context_id=context_id, cutoff=cutoff
        )
        ordinary = prediction.readout.reads[0]
        shared = ordinary.bridge
        available = torch.tensor(bridge.config.active_views(), device=base.device)[None].expand(
            len(base), -1
        )
        pool, logits = (
            _pool(bridge, base, snapshot, shared.selected_positions) if count > 1 else (None, None)
        )
        points = [prediction.point_predictions]
        for column in range(count):
            values = (
                _deleted_values(bridge, pool, logits, shared, available, column)
                if count > 1
                else torch.zeros_like(base)
            )
            presence = torch.full_like(
                ordinary.presence, count > 1 and any(bridge.config.active_views())
            )
            deleted = RegionalEpisodeRead(values, presence, None)
            state = extension._step(base, base, deleted)
            alternate = _finish_prediction(head, BridgedReadoutResult(state, (deleted,)))
            points.append(alternate.point_predictions)
        diagnostics = SpacedUtilityDiagnostics(
            prepared.next_state.flow_ids,
            cutoff,
            snapshot._digest,
            shared.evidence_ids,
            torch.stack(points, -1),
            allow_mixed_markets,
        )
    return prediction, diagnostics
