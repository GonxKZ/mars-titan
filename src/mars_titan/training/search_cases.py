"""Casos de búsqueda del optimizador que declaran las recetas cronológicas.

Titans-MAC, el lector de MARS-TITAN y CM-v1, y la GRU episódica comparten la misma regla.
Admite de uno a tres casos distintos y con nombre válido, que sustituyen los mismos
hiperparámetros del optimizador. Esos hiperparámetros no aparecen en la receta base, así que
ningún valor base queda sin usar o sin elegir. La campaña ajusta tantos casos como índices
del diseño ajusta cada referencia neuronal.
"""

import json
import re

# Un caso solo puede variar estos hiperparámetros. La arquitectura queda fija.
SEARCHED = ("learning_rate", "max_grad_norm")
CASE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")


def checked_search_cases(cases, base, build):
    """Validar los casos frente a la receta base y construir cada receta con `build`."""
    valid = (
        isinstance(cases, dict)
        and isinstance(base, dict)
        and all(isinstance(case, dict) and case for case in cases.values())
    )
    keys = {frozenset(case) for case in cases.values()} if valid else set()
    if not (
        valid
        and 1 <= len(cases) <= 3
        and all(isinstance(name, str) and CASE_NAME.fullmatch(name) for name in cases)
        and len(keys) == 1
        and next(iter(keys)) <= set(SEARCHED)
        and not next(iter(keys)) & set(base)
        and len({json.dumps(case, sort_keys=True) for case in cases.values()}) == len(cases)
    ):
        raise ValueError(
            "Los casos de búsqueda deben ser de uno a tres, distintos, con nombre válido y "
            "sustituir los mismos hiperparámetros del optimizador, ausentes de la receta base"
        )
    for case in cases.values():
        build(**(base | case))
    return cases


def case_options(base, cases, search_case):
    """Devuelve las opciones de la receta del caso elegido, o las de la base si no hay casos."""
    if cases is None:
        if search_case is not None:
            raise ValueError("La receta no declara casos de búsqueda")
        return dict(base)
    if not isinstance(search_case, str) or search_case not in cases:
        raise ValueError("Elige uno de los casos de búsqueda que declara la receta")
    return base | cases[search_case]
