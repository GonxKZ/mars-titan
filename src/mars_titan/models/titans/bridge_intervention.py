"""Máscaras y cortes fijos del puente, separados de la evidencia canónica."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BridgeInterventionConfig:
    active_regions: tuple[bool, ...]
    removed_edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if (
            type(self.active_regions) is not tuple
            or not 1 <= len(self.active_regions) <= 8
            or any(type(value) is not bool for value in self.active_regions)
        ):
            raise ValueError("La intervención necesita entre una y ocho regiones booleanas")
        if type(self.removed_edges) is not tuple:
            raise ValueError("Los cortes necesitan una tupla de pares receptor y emisor")
        for edge in self.removed_edges:
            if (
                type(edge) is not tuple
                or len(edge) != 2
                or any(
                    type(index) is not int or not 0 <= index < len(self.active_regions)
                    for index in edge
                )
            ):
                raise ValueError("Un corte no identifica un par regional válido")
        if tuple(sorted(set(self.removed_edges))) != self.removed_edges:
            raise ValueError("Los cortes deben ser únicos y estar en orden canónico")

    def edges(self, regions, topology):
        """Derivar la adyacencia fija sin confundir un corte con retirar un lector."""
        if regions != len(self.active_regions):
            raise ValueError("La máscara fija no corresponde a las regiones del puente")
        base = tuple(
            tuple(
                target == sender
                or topology == "complete"
                or (topology == "ring" and sender == (target + 1) % regions)
                for sender in range(regions)
            )
            for target in range(regions)
        )
        if any(not base[target][sender] for target, sender in self.removed_edges):
            raise ValueError("El corte no pertenece a la topología declarada")
        cuts = set(self.removed_edges)
        return tuple(
            tuple(
                base[target][sender]
                and self.active_regions[target]
                and self.active_regions[sender]
                and (target, sender) not in cuts
                for sender in range(regions)
            )
            for target in range(regions)
        )


def intervention_options(value):
    """Leer la forma JSON sin aceptar booleanos implícitos o campos desconocidos."""
    if type(value) is not dict or set(value) != {"active_regions", "removed_edges"}:
        raise ValueError("La intervención declara regiones activas y cortes explícitos")
    active, cuts = value["active_regions"], value["removed_edges"]
    if type(active) not in (list, tuple) or type(cuts) not in (list, tuple):
        raise ValueError("La intervención necesita secuencias de regiones y cortes")
    if any(type(edge) not in (list, tuple) for edge in cuts):
        raise ValueError("Cada corte necesita un par receptor y emisor")
    return BridgeInterventionConfig(tuple(active), tuple(tuple(edge) for edge in cuts))
