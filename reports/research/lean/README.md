# Comprobaciones formales de memoria y selección de rutas

`MemoryFeedback.lean` prueba sobre los reales que una escritura escalar con momentum, olvido y recorte puede tener dos puntos fijos distintos cuando su valor depende del estado. Los estados son ±(1/9, 1/18), las puertas están dentro de la caja PT1 y la distancia L1 entre los puntos es 1/3. No existe un factor menor que uno que contraiga todas las parejas. Las ramas alrededor de ambos puntos son afines y la suma de cuadrados de sus coeficientes es 27/100. La correspondencia con la recurrencia está en la [sección 7.2 del análisis](../../../docs/research/titans-memory-certificate.md#72-la-realimentación-de-mac-cambia-la-condición).

El [recibo del 17 de octubre](verification-20261017.json) fija Lean 4.35.0, Mathlib, las dependencias transitivas, las huellas del código y del export. `lake --wfail build` terminó sin avisos. Los ocho teoremas dependen únicamente de `propext`, `Quot.sound` y `Classical.choice`, axiomas estándar. No se usa evaluación nativa para admitir una conclusión. Un control que sustituye 1/9 por 1/8 fue rechazado.

El export fue aceptado por Lean default, Lean paranoid, lean4lean, nanoda, con-leche y con-ron. Esa comprobación valida el export concreto. El intento de comprobar todo el proyecto dentro del aislamiento de Lake falló al resolver el enlace simbólico local a Mathlib. No se desactivó el aislamiento. Una repetición adicional de todos los imports mediante `leanchecker --fresh MemoryFeedback` agotó su límite de 300 segundos, por lo que no cuenta como comprobación completada. Tampoco se ejecutó una comparación contra un enunciado escrito en un proyecto independiente. Los binarios de Lean y los artefactos de Mathlib proceden de sus distribuciones publicadas, sin reconstruir la cadena completa de compiladores.

La prueba no formaliza el certificado matricial de PT1, las formalizaciones de Crouzeix o k-median, la implementación PyTorch ni sus errores de redondeo. La ausencia de contracción se prueba en L1. La explicación del Jacobiano se deriva de las ramas afines, sin una formalización adicional de cálculo diferencial. No se demuestra novedad bibliográfica ni utilidad predictiva.

`RoutePolicyRegret.lean` comprueba que maximizar una utilidad estimada con error uniforme como mucho δ pierde como mucho 2δ frente a cualquier alternativa. El [alcance de la política](../../../docs/engineering/refinement-policy-mathematics.md) distingue esta hipótesis de una garantía del ridge o de la regla de rangos marginales. Su [recibo separado](route-policy-verification-20261018.json) no hereda los comprobadores adicionales del contraejemplo de memoria.

`NeuralPlasticity.lean` comprueba intervalos de ganancias y elegibilidad, además de la identidad de energía y su no incremento para un intercambio pasivo entre capacidades 1 y 2. El [recibo del 20 de octubre](neural-plasticity-verification-20261020.json) registra compilación y rechazo de un intervalo alterado. Se reutilizaron imports precompilados y no se ejecutó otro comprobador. El [alcance del mecanismo](../../../docs/engineering/neural-plasticity.md) separa esos lemas de la recurrencia completa y de la aritmética FP32/FP64.

Con la versión de `lean-toolchain` disponible, desde esta carpeta:

```bash
export LEAN_NUM_THREADS=2
lake --wfail build
artifact_dir=$(mktemp -d)
lake env leanexport MemoryFeedback -- \
  MarsTitan.MemoryFeedback.positive_fixed MarsTitan.MemoryFeedback.negative_fixed \
  MarsTitan.MemoryFeedback.fixed_points_distance MarsTitan.MemoryFeedback.gates_inside_pt1 \
  MarsTitan.MemoryFeedback.positive_branch MarsTitan.MemoryFeedback.negative_branch \
  MarsTitan.MemoryFeedback.affine_matrix_frobenius_squared \
  MarsTitan.MemoryFeedback.not_globally_contractive > "$artifact_dir/memory-feedback.ndjson"
lake check --paranoid --from-export="$artifact_dir/memory-feedback.ndjson"
```

La [guía de validación de Lean](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) explica la diferencia entre comprobar el término, inspeccionar sus axiomas y contrastar su significado. La aceptación de estos verificadores no elimina las hipótesis de su cadena de confianza.
