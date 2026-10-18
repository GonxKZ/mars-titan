"""Elegir una ruta después de K1 con ganancias de MAE y costes explícitos."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..quantile_head import QuantileHead, median
from .config import require_identity
from .episodic_readout import EpisodeRead, _SelectedEpisodes
from .refinement_routes import _ACTIONS, ROUTES, RouteResult, _checked_base, _Decision, _Node
from .state import check_finite, deferred_checks, require_true

FEATURES = (
    "base_prediction",
    "first_prediction",
    "first_prediction_change",
    "first_interval_width",
    "attention_entropy",
    "attention_maximum",
    "read_rms",
    "state_change_rms",
    "memory_present",
)


class RoutePolicy(nn.Module):
    """Regresión congelada de ganancias firmadas, con K1 fuera del soporte observado.

    Los costes ya están convertidos a unidades de MAE por el protocolo. No son
    milisegundos ni probabilidades. El orden de ROUTES resuelve los empates.
    """

    def __init__(self, *, mean, scale, coefficients, support_min, support_max, costs, min_gain):
        super().__init__()
        if type(min_gain) not in (int, float) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("La ganancia mínima debe ser finita y no negativa")
        self.min_gain = float(min_gain)
        for name, value in (
            ("mean", mean),
            ("scale", scale),
            ("coefficients", coefficients),
            ("support_min", support_min),
            ("support_max", support_max),
            ("costs", costs),
        ):
            if not isinstance(value, torch.Tensor):
                raise ValueError("La política necesita tensores explícitos")
            self.register_buffer(name, value.detach().clone())
        self.validate()
        self.eval()

    def validate(self):
        widths = dict.fromkeys(("mean", "scale", "support_min", "support_max"), (len(FEATURES),))
        widths.update(coefficients=(len(FEATURES) + 1, 5), costs=(5,))
        for name, shape in widths.items():
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.dtype not in (torch.float32, torch.float64)
                or value.dtype != self.mean.dtype
                or value.device != self.mean.device
                or value.layout != torch.strided
            ):
                raise ValueError("La política no conserva formas, precisión o dispositivo")
            check_finite(value, "Los parámetros de la política")
        require_true((self.scale > 0).all(), "La escala debe ser positiva")
        require_true((self.costs >= 0).all(), "Los costes deben ser no negativos")
        require_true((self.support_min <= self.support_max).all(), "El soporte está invertido")

    def get_extra_state(self):
        return dict(
            schema_version=1,
            policy="signed_mae_gain_one_choice_after_k1_v1",
            features=list(FEATURES),
            routes=list(ROUTES),
            min_gain=self.min_gain,
            fallback="k1_outside_observed_feature_ranges",
            costs="incremental_mae_units_relative_to_k1",
            tie="first_route_in_declared_order",
        )

    def set_extra_state(self, value):
        require_identity(value, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        supplied = {
            name[len(prefix) :]: value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        if set(supplied) != set(self.state_dict()):
            raise ValueError("El checkpoint no conserva todos los campos de la política")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        for name, value in self.named_buffers():
            other = state_dict.get(prefix + name)
            if (
                not isinstance(other, torch.Tensor)
                or other.shape != value.shape
                or other.dtype != value.dtype
            ):
                raise ValueError("El checkpoint no conserva los tensores de la política")
            check_finite(other, "Los tensores de la política")
        RoutePolicy(
            # La carga no pertenece al camino crítico. Validar en CPU impide que un
            # contexto CUDA diferido aplace el rechazo hasta después de copiar buffers.
            **{name: supplied[name].detach().cpu() for name, _ in self.named_buffers()},
            min_gain=self.min_gain,
        )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.validate()

    def forward(self, features):
        if (
            not isinstance(features, torch.Tensor)
            or features.ndim != 2
            or features.shape[1] != len(FEATURES)
            or not 1 <= len(features) <= 256
            or features.device != self.mean.device
            or features.dtype not in (torch.float32, torch.float64)
        ):
            raise ValueError("Los rasgos de la política no corresponden al contrato")
        check_finite(features, "Los rasgos de la política")
        features = features.to(self.mean.dtype)
        supported = ((features >= self.support_min) & (features <= self.support_max)).all(-1)
        # Las filas fuera del soporte terminan en K1. No se extrapola su regresión,
        # que podría desbordarse aunque los rasgos originales fuesen finitos.
        allowed = torch.where(supported[:, None], features, self.mean)
        standardized = (allowed - self.mean) / self.scale
        gain = standardized @ self.coefficients[:-1] + self.coefficients[-1]
        score = gain - self.costs - self.min_gain
        check_finite(score, "Las utilidades previstas")
        scores = torch.cat((torch.zeros_like(score[:, :1]), score), dim=-1)
        choices = scores.argmax(dim=-1)
        return torch.where(supported, choices, torch.zeros_like(choices))


@dataclass(frozen=True)
class RoutedPrediction:
    state: torch.Tensor
    point_predictions: torch.Tensor
    quantiles: torch.Tensor | None
    route_ids: torch.Tensor
    refinements: torch.Tensor
    searches: torch.Tensor
    features: torch.Tensor
    route_codes: tuple[int, ...]


def _output(head, state):
    output = head(state)
    quantiles = output if isinstance(head, QuantileHead) else None
    predictions = median(output) if quantiles is not None else output.squeeze(-1)
    if predictions.shape != (len(state),):
        raise ValueError("La cabeza debe devolver una predicción por fila")
    check_finite(output, "La salida de la cabeza")
    return predictions, quantiles


def route_features(base, first, head):
    """Rasgos disponibles tras K1, sin etiquetas ni rutas alternativas."""
    baseline, _ = _output(head, base)
    predicted, quantiles = _output(head, first.state)
    read = first.reads[-1]
    if read.weights.shape[1]:
        entropy = -(read.weights * read.weights.clamp_min(1e-30).log()).sum(-1)
        maximum = read.weights.amax(-1)
    else:
        entropy = maximum = torch.zeros_like(predicted)
    width = (
        quantiles[:, -1] - quantiles[:, 0] if quantiles is not None else torch.zeros_like(predicted)
    )
    features = torch.stack(
        (
            baseline,
            predicted,
            predicted - baseline,
            width,
            entropy,
            maximum,
            read.values.square().mean(-1).sqrt(),
            (first.state - base).square().mean(-1).sqrt(),
            read.presence[:, 0].to(base.dtype),
        ),
        dim=-1,
    )
    check_finite(features, "Los rasgos de K1")
    return features, predicted, quantiles


def _subset(node, rows):
    def selected(value):
        return value.index_select(0, rows)

    read = node.result.reads[0]
    chosen = EpisodeRead(
        *(selected(getattr(read, name)) for name in ("values", "weights", "ids", "presence"))
    )
    cache = node.cache
    cache = (
        None
        if cache is None
        else _SelectedEpisodes(
            *(selected(getattr(cache, name)) for name in ("keys", "projected", "ids"))
        )
    )
    return _Node(
        RouteResult(
            selected(node.result.state),
            (chosen,),
            1,
            node.result.searches,
            node.result.value_projections,
        ),
        cache,
    )


def apply_routes(
    model, base, head, snapshot=None, *, policy=None, choices=None, context_id=None, cutoff=None
):
    """Elegir una ruta por fila y ejecutar únicamente sus continuaciones.

    La copia de los códigos a CPU es una sincronización explícita de planificación.
    Todas las filas hacen K1. Las que paran no entran en los bloques posteriores.
    """
    if (policy is None) == (choices is None):
        raise ValueError("Se necesita una política o rutas explícitas, pero no ambas")
    if policy is not None and type(policy) is not RoutePolicy:
        raise ValueError("La política no tiene una identidad admitida")
    with deferred_checks(), torch.no_grad():
        base = _checked_base(base)
        codes = None if choices is None else _route_codes(choices, base)
        maximum = 1 if codes is None else max(int(ROUTES[code][1]) for code in codes)
        decision = _Decision(model, base.detach(), snapshot, context_id, cutoff, maximum)
        first = decision.extend()
        features, predictions, quantiles = route_features(base, first.result, head)
        chosen = policy(features) if policy is not None else choices
        if codes is None:
            codes = _route_codes(chosen, base)
            maximum = max(int(ROUTES[code][1]) for code in codes)
            model._validate_read(base, snapshot, context_id, cutoff, refinements=maximum)
        state = first.result.state.clone()
        refinements = torch.ones_like(chosen)
        searches = torch.full_like(chosen, first.result.searches)
        for code in range(1, len(ROUTES)):
            positions = [index for index, value in enumerate(codes) if value == code]
            if not positions:
                continue
            rows = torch.tensor(positions, dtype=torch.int64, device=base.device)
            local = base.index_select(0, rows)
            node = _subset(first, rows)
            for index, refresh in enumerate(_ACTIONS[ROUTES[code]]):
                node = decision.extend(
                    node, refresh=refresh, steps=1 if index == 0 else 2, base=local
                )
            point, intervals = _output(head, node.result.state)
            state.index_copy_(0, rows, node.result.state)
            predictions.index_copy_(0, rows, point)
            if quantiles is not None:
                quantiles.index_copy_(0, rows, intervals)
            refinements.index_fill_(0, rows, node.result.refinements)
            searches.index_fill_(0, rows, node.result.searches)
        return RoutedPrediction(
            state,
            predictions,
            quantiles,
            chosen.clone(),
            refinements,
            searches,
            features,
            tuple(codes),
        )


def _route_codes(chosen, base):
    if (
        not isinstance(chosen, torch.Tensor)
        or chosen.shape != (len(base),)
        or chosen.dtype != torch.int64
        or chosen.device != base.device
        or not 1 <= len(base) <= 256
    ):
        raise ValueError("Las rutas deben ser códigos int64 alineados en el mismo dispositivo")
    codes = chosen.cpu().tolist()
    if any(code < 0 or code >= len(ROUTES) for code in codes):
        raise ValueError("El código de ruta no está admitido")
    return codes
