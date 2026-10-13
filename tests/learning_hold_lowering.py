"""Pruebas que rebajan la protección de aprendizaje, cada una con su motivo (#546).

Llevan la marca `lowers_learning_hold` y se omiten antes de preparar sus fixtures mientras la
protección bloquee. Al levantar el bloqueo se ejecutarán, o antes se migrarán a un camino de
solo evaluación que no necesite rebajarla. Las pruebas anteriores que la rebajan sin esta
marca quedan para el inventario completo de #546.
"""

LOWERING = {
    "tests/training/test_modality_ablation_carry.py::"
    "test_b6_ablation_equals_the_correction_with_a_real_absence": (
        "Prepara dos ventanas B6 con run_correction_window y traslada una con carry_correction "
        "sobre ventanas técnicas de Titans-MAC. B6 no ajusta parámetros, pero las dos entradas "
        "comprueban la protección porque eligen η con la validación y predicen ventanas de la "
        "campaña"
    ),
}
