"""Configuración independiente de las transformaciones de claves y consultas."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KeyCompetition:
    """Centrado y selección opcionales, sin parámetros ni estado persistente nuevos."""

    center: bool = False
    retained: int | None = None
    selection: str = "magnitude"

    def __post_init__(self):
        if type(self.center) is not bool:
            raise ValueError("center debe ser booleano")
        if self.retained is not None and (
            type(self.retained) is not int or not 1 <= self.retained <= 512
        ):
            raise ValueError("retained debe ser None o un entero entre 1 y 512")
        if type(self.selection) is not str or self.selection not in ("magnitude", "fixed"):
            raise ValueError("selection debe ser magnitude o fixed")
        if self.selection == "fixed" and self.retained is None:
            raise ValueError("La selección fija necesita declarar retained")

    def identity(self):
        return dict(
            **asdict(self),
            schema_version=1,
            source="key_competition_v1",
            normalization="l2_after_centering_and_selection_eps_1e-12",
            ties="stable_absolute_magnitude_then_coordinate",
            fixed_coordinates="floor_j_times_dim_over_retained",
            differentiation="fixed_selection_mask_piecewise",
        )
