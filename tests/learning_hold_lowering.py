"""Pruebas nuevas que solo pueden comprobar su recorrido rebajando la protección.

Mientras rige el bloqueo de aprendizaje ninguna prueba nueva puede rebajarlo (#546). Algunas
comprobaciones necesitan que un punto de entrada protegido admita al menos un trabajo, aunque
los ejecutores estén sustituidos y no ajusten nada. Otras, de la publicación, necesitan
completar la etapa de adaptadores con el optimizador que solo registra gradientes, o
regenerar predicciones en el recorrido de la retención v2, y esos puntos de entrada se
detienen con la protección. Hasta que exista una puerta de solo evaluación separada de la de
ajuste, esas pruebas se marcan con `lowers_learning_hold` y se omiten mientras la protección
real bloquee.

`INVENTORY` es su entrada en el inventario de #546: cada prueba o módulo con el motivo por el
que rebaja la protección. La marca solo admite claves del inventario.
"""

import pytest

from mars_titan.training.learning_hold import learning_blocked

INVENTORY = {
    "tests/posttraining/test_publication_from_aggregates.py": (
        "completa la etapa de adaptadores con el optimizador que solo registra gradientes y "
        "recorre la retención v2, cuya regeneración de predicciones se detiene con la protección"
    ),
    "tests/training/test_rolling_retention.py::"
    "test_the_campaign_publication_reads_the_aggregates_with_portfolio_and_ablation": (
        "parte del recorrido de la retención v2 del fixture `walked`, que regenera predicciones "
        "y ejecuta la etapa de ablación con la protección rebajada"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_integrity_recounts_the_rows_the_joint_design_drops": (
        "publica la comparación conjunta desde la campaña A v2 reducida del fixture "
        "`joint_campaign`, que ejecuta los dobles de ajuste con la protección rebajada"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_integrity_catches_a_comparison_that_drops_other_rows": (
        "publica la comparación conjunta desde la campaña A v2 reducida del fixture "
        "`joint_campaign`, que ejecuta los dobles de ajuste con la protección rebajada"
    ),
    "tests/integrity/test_fit_budget.py::"
    "test_a_chronological_fit_with_unchanged_weights_keeps_its_initial_state": (
        "lee los informes reales de las cuatro variantes de Titans-MAC del fixture `base`, "
        "que las ejecuta con la protección rebajada y un optimizador que no cambia pesos"
    ),
    "tests/integrity/test_fit_budget.py::test_a_neural_reference_from_scratch_counts_as_fitted": (
        "ejecuta la referencia neuronal enmascarada en CPU con la protección rebajada por "
        "`cpu_runner` para leer su informe real"
    ),
    "tests/training/test_quantile_head_control.py::"
    "test_hold_reinstated_after_a_job_stops_before_the_next_one": (
        "el primer trabajo del control de la cabeza tiene que pasar la protección para "
        "comprobar que vuelve a leerse antes del segundo, aunque el ejecutor sustituido no ajusta"
    ),
    "tests/training/test_modality_ablation_carry.py::"
    "test_b6_ablation_equals_the_correction_with_a_real_absence": (
        "prepara dos ventanas B6 con run_correction_window y traslada una con carry_correction "
        "sobre ventanas técnicas de Titans-MAC. B6 no ajusta parámetros, pero las dos entradas "
        "comprueban la protección porque eligen η con la validación y predicen ventanas de la "
        "campaña"
    ),
    "tests/training/test_mars_titan_campaign.py::"
    "test_correction_frozen_parent_reproduces_the_carry_and_predicts_the_next_validation": (
        "traslada como padre congelado una ventana B6 de la campaña B reducida, que el fixture "
        "campaign_run ejecuta con la protección rebajada de permitted. carry_correction no ajusta, "
        "pero comprueba la protección porque predice ventanas de la campaña"
    ),
    "tests/test_suite_support.py::test_unguarded_fits_run_when_the_hold_allows_learning": (
        "la sesión de pytest hija lee una protección temporal que permite aprender para "
        "comprobar que la marca unguarded_fit deja de omitir, aunque la prueba sondeada no ajusta"
    ),
}


def lowers_learning_hold(key):
    """Marca que omite la prueba o el módulo `key` del inventario mientras rige el bloqueo.

    La protección se lee al importar el módulo de pruebas, antes de que ningún fixture la
    sustituya, así que decide la protección real de la sesión.
    """
    if key not in INVENTORY:
        raise KeyError(f"{key} no figura en el inventario de pruebas que rebajan la protección")
    return pytest.mark.skipif(
        learning_blocked(),
        reason=f"Bloqueo de aprendizaje vigente: {INVENTORY[key]} (inventario de #546)",
    )
