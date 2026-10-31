"""N1: dos señales externas para las puertas existentes de Titans.

Los signos financieros son una hipótesis del proyecto. Esta regla no reproduce
la dinámica de acetilcolina o noradrenalina ni aprende sus coeficientes.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .state import require_true

DUAL_NEUROMODULATION = "dual_neuromodulation_v1"


@dataclass(frozen=True)
class DualNeuromodulationConfig:
    write_expected: float = 1.0
    write_unexpected: float = 1.0
    forget_unexpected: float = 1.0
    bound: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 4
                or (name == "bound" and value == 0)
            ):
                raise ValueError("N1 requiere coeficientes en [0, 4] y una cota en (0, 4]")
            object.__setattr__(self, name, float(value))

    def identity(self):
        return dict(
            schema_version=1,
            kind=DUAL_NEUROMODULATION,
            coefficients=asdict(self),
            signal_columns=["expected", "unexpected"],
            signal_range=[0.0, 1.0],
            write="bound_tanh_write_unexpected_u_minus_write_expected_e",
            forget="bound_tanh_forget_unexpected_u",
            insertion="theta_and_alpha_logits_before_the_existing_gate_box",
            momentum="eta_unchanged",
            parameters="fixed_configuration_no_learned_coefficients",
            maturity="external_source_must_validate_availability",
        )


class DualNeuromodulation(nn.Module):
    """Convertir [filas, esperado/inesperado] en dos desplazamientos de logit."""

    def __init__(self, config):
        super().__init__()
        if type(config) is not DualNeuromodulationConfig:
            raise ValueError("N1 necesita su configuración identificada")
        self.config = config

    def validate(self, signal, rows, *, reference):
        if (
            type(self.config) is not DualNeuromodulationConfig
            or type(rows) is not int
            or not 1 <= rows <= 8192
            or not isinstance(signal, torch.Tensor)
            or not isinstance(reference, torch.Tensor)
            or signal.shape != (rows, 2)
            or signal.dtype not in (torch.float32, torch.float64)
            or signal.dtype != reference.dtype
            or signal.device != reference.device
            or signal.device.type not in ("cpu", "cuda")
            or signal.requires_grad
        ):
            raise ValueError("La señal N1 no conserva filas, precisión, dispositivo o gradiente")
        if torch.is_autocast_enabled(signal.device.type):
            raise ValueError("N1 exige precisión explícita, sin autocast")
        require_true(
            (torch.isfinite(signal) & (signal >= 0) & (signal <= 1)).all(),
            "Las señales N1 deben ser finitas y pertenecer a [0, 1]",
        )

    def forward(self, signal):
        if not isinstance(signal, torch.Tensor) or signal.ndim != 2:
            raise ValueError("N1 necesita un tensor de señales con dos ejes")
        self.validate(signal, signal.shape[0], reference=signal)
        expected, unexpected = signal[:, :1], signal[:, 1:2]
        config = self.config
        return dict(
            write=config.bound
            * torch.tanh(config.write_unexpected * unexpected - config.write_expected * expected),
            forget=config.bound * torch.tanh(config.forget_unexpected * unexpected),
        )
