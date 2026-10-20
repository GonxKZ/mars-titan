# Verificación de plasticidad neuronal

La escritura por coherencia, la captura por elegibilidad y su combinación están integradas en el núcleo Titans-MAC y en la frontera financiera. El [catálogo](../../../configs/experiments/neural-plasticity-v1.json) incluye referencias sin modulación, PT5, controles constantes y un control de ganancias unitarias. No se ha medido todavía su efecto predictivo.

La [revisión de antecedentes](../../../docs/research/neural-plasticity.md) separa motivación biológica y propuesta computacional. Las [ecuaciones y contratos](../../../docs/engineering/neural-plasticity.md) describen los estados adicionales, el alcance matemático y su persistencia.

La [segunda verificación](verification-v2.json) registra 621 casos de regresión correctos, sin fallos ni omisiones. Incluye proyecciones del artículo en CUDA, gradientes, recuperación, ventanas convolucionales en el Jacobiano, rotaciones y el límite de escala impuesto por epsilon. Una comprobación específica impide retener el grafo de autograd en la memoria de inferencia. El preparador rechaza mercados distintos y activos repetidos aunque coincidan el tamaño y las fechas de la vista.

Se detectaron los [20 mutantes de la segunda revisión](mutations-v2.json), incluida la retención del grafo y las alteraciones del mercado y la composición de la cohorte. La restauración pasó. La cobertura acumulada de 77 funciones es de 915/957 sentencias y 277/312 ramas. Incluye los programas de medida y preparación, además de los cinco módulos del núcleo. El [CRAP máximo](quality-v2.json) sigue siendo 32,111 en `NeuralMemory.validate_state`. Los tiempos recogidos con instrumentación de cobertura se excluyen del análisis de rendimiento.

El [informe de coste](../../engineering/neural-plasticity-20261020/README.md) separa el coste añadido de las variantes de la optimización que reutiliza su frontera validada. El [protocolo del piloto](../../../docs/research/neural-plasticity-protocol.md) fija once configuraciones, una semilla y un presupuesto local. Su preparación se ha ejecutado y verificado, pero todavía no se ha iniciado ningún ajuste científico nuevo.

El [primer intento](pilot-attempts.json) se detuvo antes del ajuste, con cero pasos de optimizador. El lanzador pasaba `{}` para la referencia sin cambios y el consumidor exige `None`. La corrección conserva datos, semillas, épocas y parámetros. Sus [15 comprobaciones y dos mutaciones](baseline-contract.json) utilizan ahora el validador real del consumidor en esa frontera. Los 4,488 segundos del intento se conservan en el presupuesto acumulado. No son resultados de entrenamiento.

## Primera verificación

La regresión registrada ejecutó 577 casos correctos, sin fallos ni omisiones. Después se añadieron comprobaciones de las ecuaciones realmente aplicadas, presupuesto previo a asignación, fronteras inválidas y catálogo. Sus selecciones dieron 51 y 42 casos correctos. Esos conjuntos se solapan y no se suman como pruebas independientes. La selección final de los tres archivos nuevos dio 67 casos correctos. Ocho casos específicos comprobaron CUDA en FP32 y FP64, recuperación, gradientes y paridad. El [recibo](verification.json) conserva huellas y alcance.

La cobertura acumulada de los cinco módulos examinados alcanza 641 de 656 sentencias y 210 de 224 ramas. La medición usa coverage.py 7.16.1 con ramas y Radon 6.0.1. CRAP se calcula como `C²·(1−cobertura de sentencias)³+C`. El máximo es 32,111 en `NeuralMemory.validate_state`, con complejidad 32 y 40 de 42 sentencias cubiertas. El [informe por función](quality.json) conserva los denominadores. Esa complejidad sigue siendo una limitación de mantenimiento.

Los [nueve mutantes dirigidos](mutations.json) fueron detectados. Alteraban coherencia, traza, captura local, ganancia aplicada al momentum, intercambio en ambos sentidos, serialización, edad del momento, bytes previos a asignación y dimensión del Jacobiano. La restauración pasó. El resultado no representa el 100 % de las mutaciones posibles del proyecto.

Los cinco lemas de [NeuralPlasticity.lean](../lean/NeuralPlasticity.lean) compilaron con Lean 4.35.0 y el Mathlib fijado por el proyecto. El control que ampliaba indebidamente el intervalo de intercambio fue rechazado. El [recibo formal](../lean/neural-plasticity-verification-20261020.json) declara los axiomas estándar y las comprobaciones no ejecutadas. No certifica la implementación numérica ni la red completa.

Las comprobaciones anteriores son técnicas. No se ha abierto el test final ni cambiado la protección de la campaña. El coste y cualquier estudio de desarrollo tienen manifiestos separados y no se deducen de la cobertura.
