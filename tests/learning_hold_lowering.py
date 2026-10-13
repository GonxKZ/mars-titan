"""Pruebas nuevas que solo pueden comprobar su recorrido rebajando la protección.

Mientras rige el bloqueo de aprendizaje ninguna prueba nueva puede rebajarlo (#546). Algunas
comprobaciones necesitan que un punto de entrada protegido admita al menos un trabajo, aunque
los ejecutores estén sustituidos y no ajusten nada. Hasta que exista una puerta de solo
evaluación separada de la de ajuste, esas pruebas se marcan con `lowers_learning_hold` y se
omiten mientras la protección real bloquee.

`INVENTORY` es su entrada en el inventario de #546: cada prueba o módulo con el motivo por el
que rebaja la protección. La marca solo admite claves del inventario.
"""

import pytest

from mars_titan.training.learning_hold import learning_blocked

INVENTORY = {
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
