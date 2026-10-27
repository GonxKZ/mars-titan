"""Pronósticos E2 al suprimir episodios del soporte K1, sin rellenarlo ni repetir MAC."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from mars_titan.models.quantile_head import LEVELS, QuantileHead, median
from mars_titan.models.titans.config import MAX_BLOCK_ROWS
from mars_titan.models.titans.episodic_readout import (
    EpisodeRead,
    EpisodicPrediction,
    EpisodicReadout,
    ReadoutResult,
    _normalize,
    apply_episodic_readout,
)
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.financial import PreparedDecisions
from mars_titan.models.titans.state import check_finite, deferred_checks

from .spaced_utility import _flow, _integer, _require


def _metadata(flows, moments, cutoff, size):
    _require(
        isinstance(flows, tuple)
        and 1 <= len(flows) == size <= MAX_BLOCK_ROWS
        and _integer(cutoff)
        and isinstance(moments, tuple)
        and len(moments) == size
        and all(type(at) is int and at == cutoff for at in moments),
        "Los diagnósticos E2 necesitan flujos y decisiones alineados en un único corte",
    )
    markets = {_flow(flow) for flow in flows}
    _require(
        len(markets) == 1 and len(set(flows)) == size,
        "Los diagnósticos E2 necesitan un mercado sin activos repetidos",
    )


@dataclass(frozen=True)
class SpacedUtilityDiagnostics:
    """Puntos emitidos [filas, original más supresiones] e IDs [filas, supresiones]."""

    flow_ids: tuple[str, ...]
    cutoff: int
    snapshot_sha256: str
    episode_ids: torch.Tensor
    points: torch.Tensor

    def ledger_rows(self, label_not_before):
        """Convertir el bloque a CPU. Las fechas previstas no contienen resultados futuros."""
        _require(
            isinstance(self.episode_ids, torch.Tensor)
            and self.episode_ids.ndim == 2
            and self.episode_ids.dtype == torch.int64
            and self.episode_ids.shape[1] <= 8
            and isinstance(self.points, torch.Tensor)
            and self.points.shape == (len(self.flow_ids), self.episode_ids.shape[1] + 1)
            and len(self.episode_ids) == len(self.flow_ids)
            and self.points.dtype in (torch.float32, torch.float64)
            and self.points.device == self.episode_ids.device
            and self.points.grad_fn is None
            and not self.points.requires_grad,
            "El diagnóstico E2 no conserva formas, precisión o aislamiento del grafo",
        )
        _metadata(self.flow_ids, (self.cutoff,) * len(self.flow_ids), self.cutoff, len(self.points))
        _require(
            type(label_not_before) is dict
            and len(label_not_before) == len(self.flow_ids)
            and set(label_not_before) == set(self.flow_ids),
            "Las maduraciones previstas E2 deben alinearse por identidad de flujo",
        )
        _require(
            all(
                at is None or (_integer(at) and at > self.cutoff)
                for at in label_not_before.values()
            ),
            "La maduración prevista E2 debe ser posterior a la decisión",
        )
        points = self.points.detach().cpu().tolist()
        identifiers = self.episode_ids.detach().cpu().tolist()
        _validate_host_rows(points, identifiers)
        return [
            dict(
                flow_id=flow,
                point=point[0],
                deleted_points=list(zip(ids, point[1:], strict=True)),
                label_not_before=label_not_before[flow],
            )
            for flow, point, ids in zip(self.flow_ids, points, identifiers, strict=True)
        ]


def _validate_host_rows(points, identifiers):
    # Los tipos y formas tensoriales ya están comprobados. No hace falta crear otra
    # copia normalizada de todas las filas que el consumidor descartaría al terminar.
    _require(
        all(math.isfinite(point) for row in points for point in row),
        "Los puntos emitidos E2 deben ser finitos",
    )
    _require(
        all(all(i > 0 for i in ids) and len(set(ids)) == len(ids) for ids in identifiers),
        "Los episodios suprimidos E2 deben ser positivos y únicos por fila",
    )


def _validate(prepared, head, extension, snapshot, cutoff, differentiable):
    _require(
        isinstance(prepared, PreparedDecisions)
        and type(extension) is EpisodicReadout
        and type(snapshot) is EpisodeSnapshot,
        "E2 necesita preparación, lector e instantánea identificados",
    )
    _require(
        extension.config.refinements == 1
        and extension.config.mode == "bank"
        and extension.config.evidence_bridge is None
        and not extension.training
        and differentiable is False,
        "Este productor E2 necesita lector estándar bank en eval, K1 y sin diferenciación",
    )
    _require(
        type(head) in (nn.Linear, QuantileHead)
        and not head.training
        and isinstance(head, QuantileHead) == (prepared.quantiles is not None)
        and head.in_features == extension.config.hidden_size
        and head.out_features == (5 if isinstance(head, QuantileHead) else 1),
        "E2 necesita la cabeza escalar o cuantílica correspondiente, en eval",
    )
    extension._state(prepared.working_state)
    base = prepared.working_state
    _require(
        all(p.dtype == base.dtype and p.device == base.device for p in head.parameters()),
        "La cabeza E2 debe conservar precisión y dispositivo del lector",
    )
    _require(
        not torch.is_autocast_enabled(base.device.type),
        "Este productor E2 no admite autocast sin un contraste de precisión independiente",
    )
    _metadata(
        prepared.next_state.flow_ids,
        prepared.next_state.last_prediction_at,
        cutoff,
        len(prepared.working_state),
    )


def _finish_prediction(head, result):
    """Aplicar la cabeza existente sin modificar la huella del lector histórico."""
    output = head(result.state)
    quantiles = None
    if isinstance(head, QuantileHead):
        _require(
            output.shape == (len(result.state), len(LEVELS)),
            "La cabeza de cuantiles E2 debe devolver cinco niveles por flujo",
        )
        check_finite(output, "Los cuantiles refinados E2")
        quantiles, predictions = output, median(output)
    else:
        predictions = output.squeeze(-1)
    _require(
        predictions.shape == (len(result.state),),
        "La cabeza E2 debe devolver una salida escalar por flujo",
    )
    check_finite(predictions, "La predicción refinada E2")
    return EpisodicPrediction(predictions, result, quantiles)


def apply_spaced_utility_readout(
    prepared,
    head,
    extension=None,
    snapshot=None,
    *,
    enabled=False,
    context_id=None,
    cutoff=None,
    differentiable=False,
):
    """Devolver el pronóstico ordinario y diagnósticos opcionales, sin actualizar al padre.

    La ruta apagada conserva las opciones de la interfaz anterior. La activa solo admite
    el lector estándar K1. El integrador verifica los bytes de la instantánea en su frontera
    de publicación y recuperación, como en el lector original.
    """
    _require(type(enabled) is bool, "La activación E2 debe ser booleana")
    if not enabled:
        return (
            apply_episodic_readout(
                prepared,
                head,
                extension,
                snapshot,
                context_id=context_id,
                cutoff=cutoff,
                differentiable=differentiable,
            ),
            None,
        )
    _validate(prepared, head, extension, snapshot, cutoff, differentiable)
    count = min(snapshot.count, extension.config.neighbors)
    base = prepared.working_state.detach()
    with torch.no_grad(), deferred_checks():
        # La cota incluye el refinamiento original y todas las supresiones, antes de leer.
        extension._validate_read(base, snapshot, context_id, cutoff, refinements=count + 1)
        read, _, selected = extension._read_parts(base, snapshot, context_id, cutoff, repeated=True)
        check_finite(extension.step_logit, "La puerta de refinamiento")
        result = ReadoutResult(extension._step(base, base, read), (read,))
        prediction = _finish_prediction(head, result)
        points = [prediction.point_predictions]
        if count:
            # Recalcular solo esta proyección evita añadir una caché de logits al baseline.
            raw = extension.query_projection(base)
            check_finite(raw, "La consulta episódica E2")
            query = _normalize(raw)
            logits = (selected.keys * query.unsqueeze(1)).sum(-1) / extension.config.temperature
            for column in range(count):
                if count == 1:
                    weights = torch.zeros_like(read.weights)
                else:
                    masked = logits.clone()
                    masked[:, column] = -torch.inf
                    weights = masked.softmax(-1)
                values = (selected.projected * weights.unsqueeze(-1)).sum(1)
                check_finite(values, "La lectura suprimida E2")
                presence = torch.full_like(read.presence, count > 1)
                deleted = EpisodeRead(values, weights, read.ids, presence)
                state = extension._step(base, base, deleted)
                alternate = _finish_prediction(head, ReadoutResult(state, (deleted,)))
                points.append(alternate.point_predictions)
        diagnostics = SpacedUtilityDiagnostics(
            prepared.next_state.flow_ids,
            cutoff,
            snapshot._digest,
            read.ids,
            torch.stack(points, dim=-1),
        )
    return prediction, diagnostics
