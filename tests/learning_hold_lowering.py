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

# Motivos compartidos por varias pruebas de las opciones de la campaña A v2 (#363).
_LAUNCH = (
    "la ventana reducida se lanza con el motor y los ejecutores reales hasta la entrada de "
    "cada entrenador, donde los dobles la detienen sin ajustar, y el motor exige la protección "
    "antes de leer la campaña"
)
_ADAPTER = (
    "`run_case` exige la protección antes de leer el caso, y el optimizador sustituido de "
    "`recorder` registra los gradientes sin cambiar pesos"
)

INVENTORY = {
    "tests/training/test_quantile_head_control.py::"
    "test_hold_reinstated_after_a_job_stops_before_the_next_one": (
        "el primer trabajo del control de la cabeza tiene que pasar la protección para "
        "comprobar que vuelve a leerse antes del segundo, aunque el ejecutor sustituido no ajusta"
    ),
    "tests/training/test_campaign_launch_options.py::"
    "test_every_trainer_of_the_window_receives_the_declared_options_in_its_slot": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_a_cut_campaign_resumes_every_job_with_the_same_options": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_changed_options_reach_the_trainers_that_use_them": _LAUNCH,
    "tests/training/test_campaign_launch_options.py::"
    "test_an_undeclared_value_is_rejected_before_opening_views_or_output": (
        "comprueba que el motor rechaza el valor antes de abrir vistas o crear la salida, y el "
        "motor exige la protección antes de leer la campaña. Los mismos rechazos se comprueban "
        "sin la guarda con `check_campaign`"
    ),
    "tests/training/test_campaign_a_joint.py::"
    "test_neural_receipts_record_the_declared_kernels_in_their_identity": (
        "los recibos salen de la campaña reducida que el motor ejecuta con ejecutores "
        "sustituidos que no ajustan"
    ),
    "tests/training/test_reference_run_precision.py::"
    "test_resume_keeps_the_batch_precision_and_graph_step_it_started_with": (
        "la reanudación compara la identidad dentro de `run_reference_case`, después de su "
        "guarda, y el recorrido se detiene antes de leer lotes de ajuste"
    ),
    "tests/posttraining/test_masked_run.py::"
    "test_parent_and_adapter_run_with_the_strict_precision_of_the_parent_case": _ADAPTER,
    "tests/posttraining/test_masked_run.py::"
    "test_an_adapter_whose_process_allows_tf32_mid_run_fails_at_the_next_save": _ADAPTER,
    "tests/posttraining/test_masked_run.py::"
    "test_a_final_evaluation_under_tf32_is_not_confirmed": _ADAPTER,
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
