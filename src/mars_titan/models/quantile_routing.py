"""Contrato de separación de las señales de mediana y dispersión."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QuantileRouting:
    """Controles de aprendizaje, sin parámetros ni estado tensorial adicionales."""

    median_from_tails: bool = True
    encoder_from_spreads: bool = True

    def __post_init__(self):
        if type(self.median_from_tails) is not bool or type(self.encoder_from_spreads) is not bool:
            raise ValueError("Las dos rutas de cuantiles necesitan booleanos explícitos")

    def identity(self):
        return dict(
            **asdict(self),
            schema_version=1,
            kind="quantile_feedback_routing_v1",
            surrogate_gradient=not (self.median_from_tails and self.encoder_from_spreads),
            inference="original_ordered_quantiles",
            persistent_tensor_state=0,
        )
