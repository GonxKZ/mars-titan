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


# Cambios del predictor que un modelo de Titans-MAC declara sobre la receta común, como la
# memoria acotada PT1 (#453). Son extensiones desactivables del núcleo: sin ellos el modelo es
# el de la receta. Su contenido lo valida `FinancialConfig` al construir el predictor.
PREDICTOR_CHANGES = (
    "gate_bias",
    "memory_stability",
    "memory_multiscale",
    "memory_plasticity",
    "memory_key_competition",
    "quantile_routing",
    "presence_modulation",
    "attention_layout",
    "regional_fusion",
    "regional_execution",
    "memory_bank_readout",
    "temporal_encoder",
    "regional_variant",
    "holographic_network",
    "axial_time_region",
    "fractal_shared_operator",
    "session_rank_inputs",
    "session_ic_objective",
    "cross_asset_attention",
    "asset_groups",
    # El control de fuga de H4 lee el sector actual de las fichas, con {"source": carpeta}.
    "sector_map",
    "laplace_scale_head",
    "auxiliary_horizons",
    "opponent_heads",
    "mature_reliability",
    "memory_huber",
    "memory_time_scales",
    "differential_attention",
    # Modalidades que el predictor no recibe nunca: {"names": [...]}.
    "removed_modalities",
    # #836: escalas de aprendizaje del tramo de ajuste y codificador de una referencia.
    "learning_scale",
    "reference_encoder",
    # Rama de los rasgos técnicos del bloque del gráfico: columns, bins, embedding y hidden.
    "technical_branch",
    # Opciones de memoria neuronal y consulta de salida de MAC (WP4).
    "memory_value_normalization",
    "output_query",
    # Compuerta softmax enmascarada por presencia en la fusión común: presence_softmax_v1.
    "modality_gating",
    # Memoria transversal de rango y residuo maduro (CRRM).
    "cross_sectional_rank_memory",
    # --- ic_titans_mlp ---
    "ic_titans_mlp",
    # --- fin ic_titans_mlp ---
    # --- lead_lag_memory ---
    "lead_lag_memory",
    # --- fin lead_lag_memory ---
    # Interferómetro holográfico de sesión en fase dual (WP11).
    "session_hologram",
    # --- memory_cross_sectional_surprise ---
    # Ruta de gradiente interno de la memoria y sorpresa relativa transversal (CSOS).
    "memory_inner_gradient",
    "memory_cross_sectional_surprise",
    # --- fin memory_cross_sectional_surprise ---
    # --- factor_memory ---
    # Memoria rápida factorizada con blanqueo espectral de sesión (LFM-SW, Idea 4).
    "factor_memory",
    # --- fin factor_memory ---
    # --- contrastive_memory ---
    # Idea 5 (InfoNCE-Titans): memoria asociativa contrastiva transversal compartida por mercado.
    "contrastive_memory",
    # --- fin contrastive_memory ---
    # --- session_resonator ---
    # Resonador holográfico de descomposición latente transversal (CSRF, WP13-2).
    "session_resonator",
    # --- fin session_resonator ---
)
# Cambios que se declaran con un nombre en lugar de un objeto.
NAMED_CHANGES = (
    "presence_modulation",
    "modality_gating",
    "attention_layout",
    "temporal_encoder",
    "session_rank_inputs",
    "differential_attention",
    "reference_encoder",
    "memory_value_normalization",
    "output_query",
    # --- memory_cross_sectional_surprise ---
    "memory_inner_gradient",
    # --- fin memory_cross_sectional_surprise ---
)


def checked_predictor_changes(changes):
    """Validar los cambios del predictor de un modelo, o devolver None si no declara ninguno."""
    if changes is None:
        return None
    if not (
        isinstance(changes, dict)
        and changes
        and set(changes) <= set(PREDICTOR_CHANGES)
        and all(
            (type(value) is str and bool(value))
            if name in NAMED_CHANGES
            else (isinstance(value, dict) and bool(value))
            for name, value in changes.items()
        )
    ):
        raise ValueError(
            "Los cambios del predictor admiten gate_bias, memory_stability y memory_multiscale "
            "junto a memory_plasticity, memory_key_competition, quantile_routing, regional_fusion, "
            "regional_execution, memory_bank_readout, regional_variant, holographic_network, "
            "axial_time_region, fractal_shared_operator, session_ic_objective, "
            "cross_asset_attention, asset_groups, sector_map, laplace_scale_head, "
            "auxiliary_horizons, opponent_heads, "
            "mature_reliability, memory_huber y memory_time_scales como objetos no vacíos, y "
            "presence_modulation, modality_gating, attention_layout, temporal_encoder, "
            "session_rank_inputs, differential_attention, reference_encoder, "
            "memory_value_normalization y output_query como nombres no vacíos. También admiten "
            "learning_scale, removed_modalities, technical_branch y session_hologram como objetos "
            "no vacíos"
            # --- memory_cross_sectional_surprise ---
            ", memory_inner_gradient como nombre no vacío y memory_cross_sectional_surprise como "
            "objeto no vacío"
            # --- fin memory_cross_sectional_surprise ---
        )
    return changes


def case_options(base, cases, search_case):
    """Devuelve las opciones de la receta del caso elegido, o las de la base si no hay casos."""
    if cases is None:
        if search_case is not None:
            raise ValueError("La receta no declara casos de búsqueda")
        return dict(base)
    if not isinstance(search_case, str) or search_case not in cases:
        raise ValueError("Elige uno de los casos de búsqueda que declara la receta")
    return base | cases[search_case]
