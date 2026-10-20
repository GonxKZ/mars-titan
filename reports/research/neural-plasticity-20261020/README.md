# Verificación de plasticidad neuronal

La escritura por coherencia, la captura por elegibilidad y su combinación están integradas en el núcleo Titans-MAC y en la frontera financiera. El [catálogo](../../../configs/experiments/neural-plasticity-v1.json) incluye referencias sin modulación, PT5, controles constantes y un control de ganancias unitarias. No se ha medido todavía su efecto predictivo.

La [revisión de antecedentes](../../../docs/research/neural-plasticity.md) separa motivación biológica y propuesta computacional. Las [ecuaciones y contratos](../../../docs/engineering/neural-plasticity.md) describen los estados adicionales, el alcance matemático y su persistencia.

La regresión registrada ejecutó 577 casos correctos, sin fallos ni omisiones. Después se añadieron comprobaciones de las ecuaciones realmente aplicadas, presupuesto previo a asignación, fronteras inválidas y catálogo. Sus selecciones dieron 51 y 42 casos correctos. Esos conjuntos se solapan y no se suman como pruebas independientes. La selección final de los tres archivos nuevos dio 67 casos correctos. Ocho casos específicos comprobaron CUDA en FP32 y FP64, recuperación, gradientes y paridad. El [recibo](verification.json) conserva huellas y alcance.

La cobertura acumulada de los cinco módulos examinados alcanza 641 de 656 sentencias y 210 de 224 ramas. La medición usa coverage.py 7.16.1 con ramas y Radon 6.0.1. CRAP se calcula como `C²·(1−cobertura de sentencias)³+C`. El máximo es 32,111 en `NeuralMemory.validate_state`, con complejidad 32 y 40 de 42 sentencias cubiertas. El [informe por función](quality.json) conserva los denominadores. Esa complejidad sigue siendo una limitación de mantenimiento.

Los [nueve mutantes dirigidos](mutations.json) fueron detectados. Alteraban coherencia, traza, captura local, ganancia aplicada al momentum, intercambio en ambos sentidos, serialización, edad del momento, bytes previos a asignación y dimensión del Jacobiano. La restauración pasó. El resultado no representa el 100 % de las mutaciones posibles del proyecto.

Los cinco lemas de [NeuralPlasticity.lean](../lean/NeuralPlasticity.lean) compilaron con Lean 4.35.0 y el Mathlib fijado por el proyecto. El control que ampliaba indebidamente el intervalo de intercambio fue rechazado. El [recibo formal](../lean/neural-plasticity-verification-20261020.json) declara los axiomas estándar y las comprobaciones no ejecutadas. No certifica la implementación numérica ni la red completa.

Las comprobaciones anteriores son técnicas. No se ha abierto el test final ni cambiado la protección de la campaña. Las medidas de coste y cualquier estudio de desarrollo necesitan su propio manifiesto y no se deducen de estos resultados.
