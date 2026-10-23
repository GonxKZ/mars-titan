"""Presupuesto explícito del estimador financiero experimental de Dust."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DustConfig:
    """Draws por sitio, agrupación física y límites previos al cálculo."""

    population: int = 16
    chunk_size: int = 2
    sigma: float = 0.05
    seed: int = 42
    max_forward_calls: int = 512
    max_workspace_bytes: int = 128 * 1024**2

    def __post_init__(self):
        integers = (
            (self.population, 2, 4096),
            (self.chunk_size, 1, self.population),
            (self.seed, 0, 2**63 - 1),
            (self.max_forward_calls, 1, 1_000_000),
            (self.max_workspace_bytes, 1, 2 * 1024**3),
        )
        if any(type(value) is not int or not low <= value <= high for value, low, high in integers):
            raise ValueError("La población, agrupación y límites de Dust no son válidos")
        if (
            type(self.sigma) not in (int, float)
            or not math.isfinite(self.sigma)
            or not 0 < self.sigma <= 10
        ):
            raise ValueError("El ruido de Dust necesita una escala positiva y finita")

    def identity(self):
        """Declarar también las diferencias frente a la receta original de tokens."""
        return dict(
            schema_version=1,
            kind="dust_financial_activation_estimator_v1",
            experimental=True,
            **asdict(self),
            centering=dict(
                axis="complete_population_per_financial_row",
                correction="population_over_population_minus_one",
            ),
            attention_credit=dict(gamma=1.0, support="original_causal_attention_window"),
            head="cached_scalar_output_with_the_same_residual_loss",
            precision="predictor_dtype_without_autocast",
            optimizer="external_not_part_of_estimate",
            global_weight_es=False,
            workspace_scope="logical_tensor_bound_excluding_allocator_and_backend_internals",
        )
