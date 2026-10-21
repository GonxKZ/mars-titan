"""Utilidades puntuales y de intervalos sobre el catálogo cerrado, sin ajuste en inferencia.

La política recibe una observación anterior a MAC y admisibilidad explícita. La
extracción temporal y la publicación de su decisión corresponden al enlace financiero.
Los coeficientes dados no acreditan entrenamiento ni utilidad predictiva.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn

from .config import require_identity
from .episodic_snapshot import _digest_id
from .mechanism_catalog import MechanismCatalog
from .state import check_finite, deferred_checks, require_true

FEATURES = (
    "log1p_cohort_rows",
    "prices_present_fraction",
    "news_present_fraction",
    "charts_present_fraction",
    "fundamentals_present_fraction",
    "macro_present_fraction",
    "prices_mean_log1p_rms",
    "news_mean_log1p_rms",
    "charts_mean_log1p_rms",
    "fundamentals_mean_log1p_rms",
    "macro_mean_log1p_rms",
    "log1p_max_input_age_seconds",
    "log1p_eligible_bank_entries",
    "previous_context_present",
    "log1p_previous_context_age_days",
    "calibration_present",
    "log1p_pending_calibration_rows",
    "static_correction_mean",
    "static_correction_max_abs",
    "online_correction_mean",
    "online_correction_max_abs",
)
ACTIONS = tuple(action.key for action in MechanismCatalog().actions())
POINT_ACTIONS = tuple(dict.fromkeys(key.rsplit("/", 1)[0] for key in ACTIONS))


def _utility_scalar(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("El peso de intervalos y la ganancia mínima deben ser finitos")
    return float(value)


@dataclass(frozen=True)
class MechanismChoice:
    action_ids: torch.Tensor
    scores: torch.Tensor
    point_gains: torch.Tensor
    interval_gains: torch.Tensor
    supported: torch.Tensor
    available: torch.Tensor
    standardized: torch.Tensor


class MechanismPolicy(nn.Module):
    """Escoger una acción por cohorte, con ganancias previstas y costes declarados.

    Las tres calibraciones comparten la ganancia puntual de su contexto y ruta.
    El soporte comprueba rangos marginales, no soporte conjunto ni un OOD calibrado.
    Los costes ya están convertidos a las unidades de la utilidad por el protocolo.
    """

    def __init__(
        self,
        *,
        mean,
        scale,
        point_coefficients,
        interval_coefficients,
        support_min,
        support_max,
        costs,
        interval_weight,
        min_gain,
        action_mask=None,
        fit_receipt_sha256=None,
    ):
        super().__init__()
        self.interval_weight, self.min_gain = (
            _utility_scalar(interval_weight),
            _utility_scalar(min_gain),
        )
        if fit_receipt_sha256 is not None:
            _digest_id(fit_receipt_sha256, "El recibo declarado del ajuste")
        self.fit_receipt_sha256 = fit_receipt_sha256
        for name, value in (
            ("mean", mean),
            ("scale", scale),
            ("point_coefficients", point_coefficients),
            ("interval_coefficients", interval_coefficients),
            ("support_min", support_min),
            ("support_max", support_max),
            ("costs", costs),
        ):
            if not isinstance(value, torch.Tensor):
                raise ValueError("La política necesita tensores explícitos")
            self.register_buffer(name, value.detach().clone())
        if action_mask is None:
            self.action_mask = None
        else:
            if (
                not isinstance(action_mask, torch.Tensor)
                or action_mask.layout != torch.strided
                or action_mask.shape != (len(ACTIONS),)
                or action_mask.dtype != torch.bool
                or action_mask.device != self.mean.device
            ):
                raise ValueError("La máscara de acciones debe ser un vector booleano del catálogo")
            self.register_buffer("action_mask", action_mask.detach().clone())
        self.validate()
        self.eval()

    def validate(self):
        _utility_scalar(self.interval_weight)
        _utility_scalar(self.min_gain)
        if self.fit_receipt_sha256 is not None:
            _digest_id(self.fit_receipt_sha256, "El recibo declarado del ajuste")
        shapes = dict.fromkeys(("mean", "scale", "support_min", "support_max"), (len(FEATURES),))
        shapes.update(
            point_coefficients=(len(FEATURES) + 1, len(POINT_ACTIONS)),
            interval_coefficients=(len(FEATURES) + 1, len(ACTIONS)),
            costs=(len(ACTIONS),),
        )
        for name, shape in shapes.items():
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.layout != torch.strided
                or value.dtype not in (torch.float32, torch.float64)
                or value.dtype != self.mean.dtype
                or value.device != self.mean.device
            ):
                raise ValueError("La política no conserva formas, precisión o dispositivo")
            check_finite(value, "Los tensores de la política")
        require_true((self.scale > 0).all(), "Las escalas deben ser positivas")
        require_true((self.costs >= 0).all(), "Los costes deben ser no negativos")
        require_true((self.support_min <= self.mean).all(), "La media queda fuera del soporte")
        require_true((self.mean <= self.support_max).all(), "La media queda fuera del soporte")
        require_true(
            (self.point_coefficients[:, 0] == 0).all()
            & (self.interval_coefficients[:, 0] == 0).all()
            & (self.costs[0] == 0),
            "La referencia debe conservar ganancia y coste incrementales cero",
        )
        if self.action_mask is not None:
            if (
                self.action_mask.shape != (len(ACTIONS),)
                or self.action_mask.dtype != torch.bool
                or self.action_mask.device != self.mean.device
                or self.action_mask.layout != torch.strided
            ):
                raise ValueError("La máscara de acciones cambió de forma, precisión o dispositivo")
            require_true(self.action_mask[0], "La máscara debe conservar la acción de referencia")

    def get_extra_state(self):
        result = dict(
            schema_version=1,
            kind="pre_mac_cohort_mechanism_policy_v1",
            features=list(FEATURES),
            actions=list(ACTIONS),
            point_actions=list(POINT_ACTIONS),
            selection_unit="market_cohort_before_mac",
            max_cohorts=2,
            interval_weight=self.interval_weight,
            min_gain=self.min_gain,
            score="point_gain_plus_weighted_interval_gain_minus_cost_and_min_gain",
            baseline="off/k1/none_with_zero_score",
            tie="first_admissible_action_in_declared_catalog_order",
            support="marginal_feature_ranges_not_joint_or_calibrated_ood",
            costs="incremental_declared_utility_units_not_raw_milliseconds",
        )
        if self.action_mask is not None:
            result["action_support"] = "declared_boolean_mask_not_fit_attestation"
        if self.fit_receipt_sha256 is not None:
            result["declared_fit_receipt_sha256"] = self.fit_receipt_sha256
        return result

    def set_extra_state(self, state):
        require_identity(state, self.get_extra_state())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        supplied = {
            name[len(prefix) :]: value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        expected = self.state_dict()
        if set(supplied) != set(expected):
            raise ValueError("El checkpoint no conserva todos los campos de la política")
        require_identity(supplied["_extra_state"], self.get_extra_state())
        for name, value in self.named_buffers():
            other = supplied[name]
            if (
                not isinstance(other, torch.Tensor)
                or other.shape != value.shape
                or other.dtype != value.dtype
            ):
                raise ValueError("El checkpoint no conserva los tensores de la política")
        # Validar una copia CPU hace que los rechazos precedan a cualquier copia de pesos,
        # también si el llamador está dentro de verificaciones CUDA diferidas.
        MechanismPolicy(
            **{name: supplied[name].detach().cpu() for name, _ in self.named_buffers()},
            interval_weight=self.interval_weight,
            min_gain=self.min_gain,
            fit_receipt_sha256=self.fit_receipt_sha256,
        )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def forward(self, features, available):
        if (
            not isinstance(features, torch.Tensor)
            or features.layout != torch.strided
            or features.ndim != 2
            or features.shape[1] != len(FEATURES)
            or not 1 <= len(features) <= 2
            or features.dtype != self.mean.dtype
            or features.device != self.mean.device
        ):
            raise ValueError("La observación no corresponde al contrato de cohortes")
        if (
            not isinstance(available, torch.Tensor)
            or available.layout != torch.strided
            or available.shape != (len(features), len(ACTIONS))
            or available.dtype != torch.bool
            or available.device != features.device
        ):
            raise ValueError("La admisibilidad no corresponde al catálogo y sus cohortes")
        if torch.is_autocast_enabled(features.device.type):
            raise ValueError("La política requiere FP32 o FP64 explícitos, sin autocast")
        if self.action_mask is not None:
            available = available & self.action_mask[None, :]
        with deferred_checks():
            check_finite(features, "Los rasgos de la decisión")
            require_true(available[:, 0].all(), "K1 de referencia debe ser admisible")
            supported = ((features >= self.support_min) & (features <= self.support_max)).all(-1)
            safe = torch.where(supported[:, None], features, self.mean)
            standardized = (safe - self.mean) / self.scale
            point = standardized @ self.point_coefficients[:-1] + self.point_coefficients[-1]
            point = point.repeat_interleave(3, dim=-1)
            interval = (
                standardized @ self.interval_coefficients[:-1] + self.interval_coefficients[-1]
            )
            utilities = point + self.interval_weight * interval - self.costs
            scores = torch.cat(
                (torch.zeros_like(utilities[:, :1]), utilities[:, 1:] - self.min_gain), dim=-1
            )
            check_finite(point, "Las ganancias puntuales previstas")
            check_finite(interval, "Las ganancias de intervalos previstas")
            check_finite(scores, "Las utilidades previstas")
            candidates = torch.where(available, scores, torch.full_like(scores, -torch.inf))
            choices = candidates.argmax(-1)
            choices = torch.where(supported, choices, torch.zeros_like(choices))
        return MechanismChoice(
            choices, scores, point, interval, supported, available.clone(), standardized
        )
