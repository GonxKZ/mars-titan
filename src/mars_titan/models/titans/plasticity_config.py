"""Configuración de la modulación neuronal, independiente de los controles C y M."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MemoryPlasticity:
    """Escritura por coherencia y captura por elegibilidad con controles constantes.

    La coherencia describe los gradientes asociativos observados. No es una
    probabilidad posterior ni una medida identificada de ruido financiero.
    La ausencia de esta configuración conserva la memoria histórica.
    """

    write: str = "coherence"
    capture: str = "none"
    beta: float = 0.9
    trace_decay: float = 0.95
    epsilon: float = 1e-8
    write_floor: float = 0.1
    capture_floor: float = 0.1
    write_constant: float = 0.5
    capture_constant: float = 0.5
    gradient_contract: str = "row_broadcast_v1"

    def __post_init__(self) -> None:
        for name, choices in (
            ("write", ("none", "coherence", "constant")),
            ("capture", ("none", "eligibility", "constant")),
            ("gradient_contract", ("row_broadcast_v1", "scalar_controls_v2")),
        ):
            value = getattr(self, name)
            if type(value) is not str or value not in choices:
                raise ValueError(f"{name} no identifica una regla de plasticidad admitida")
        for name in (
            "beta",
            "trace_decay",
            "epsilon",
            "write_floor",
            "capture_floor",
            "write_constant",
            "capture_constant",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} debe ser un número finito")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} debe ser un número finito")
            if name == "epsilon":
                valid = value > 0
            elif name in ("beta", "trace_decay"):
                valid = 0 <= value < 1
            else:
                valid = 0 <= value <= 1
            if not valid:
                raise ValueError(f"{name} está fuera del dominio de la regla de plasticidad")
            object.__setattr__(self, name, value)

    def configuration(self) -> dict:
        """Conservar los nueve campos históricos al serializar el contrato original."""
        result = asdict(self)
        if self.gradient_contract == "row_broadcast_v1":
            result.pop("gradient_contract")
        return result

    def identity(self) -> dict:
        version = 1 if self.gradient_contract == "row_broadcast_v1" else 2
        return dict(
            **self.configuration(),
            schema_version=version,
            source=f"neural_plasticity_v{version}",
            statistics="per_flow_layer_row_after_associative_gradient_clipping",
            history="gradient_mean_energy_eligibility",
            coherence="bounded_bias_corrected_squared_mean_over_energy",
            capture_signal="local_eligibility_times_mean_row_coherence",
            coupling="first_cascade_interface_both_directions",
            financial_labels_used=False,
            differentiation="full_when_differentiable_detached_otherwise",
        )
