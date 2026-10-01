# Acumulación de errores por sesión

La comparación del 24 de septiembre de 2026 reduce el trabajo de `SessionErrors.update` en la evaluación existente. Los lotes de hasta 128 filas conservan la agrupación secuencial. Los mayores usan `np.unique` para identificar los instantes y `np.bincount` para contar y sumar los errores por mercado e instante. La combinación con el estado anterior recorre una sola vez los grupos del lote.

Las fechas mantienen su tipo durante la agrupación. El código combina el índice de cada instante y el mercado, sin convertir valores `uint64` a `int64`. `np.minimum.at` identifica la primera aparición de cada clave y permite conservar el orden del diccionario. Las sumas de cada lote se completan antes de añadir el estado anterior. El estado público solo cambia después de comprobar todo el lote.

La ruta secuencial tiene coste O(B), con B filas por lote. La ruta vectorizada incluye ordenaciones O(B log B). Sus buffers son O(B) y los índices de grupos tienen como máximo 8192 posiciones, porque el lote admite hasta 4096 filas y dos mercados. El límite por defecto sigue siendo de 50.000 sesiones. Las ventajas indicadas proceden de las medidas con estos límites.

## Evaluación completa

Se comparó la referencia de `889f649` con la implementación seleccionada. Cada caso evalúa 16.384 filas sintéticas mediante un modelo lineal de 16 entradas en CPU. Hay cinco repeticiones dentro de un proceso, después de un calentamiento, con un hilo numérico. Los mercados están mezclados. En las cohortes se repite el instante cada 128 filas. En el otro caso, cada fila tiene una fecha distinta.

| Sesiones | Filas por lote | Sin destino, antes → después (ms) | Con Parquet, antes → después (ms) |
| --- | ---: | ---: | ---: |
| Cohortes | 32 | 21,096 → 20,172 | 156,945 → 158,365 |
| Cohortes | 512 | 8,228 → 2,707 | 26,825 → 20,775 |
| Cohortes | 4096 | 7,706 → 1,078 | 17,354 → 11,671 |
| Fechas únicas | 32 | 44,521 → 36,641 | 185,011 → 176,018 |
| Fechas únicas | 512 | 30,392 → 22,961 | 52,439 → 41,650 |
| Fechas únicas | 4096 | 30,493 → 20,963 | 45,316 → 31,463 |

La tabla recoge medianas del recorrido completo de `_evaluate`, incluidas inferencia, acumulación, resumen y escritura cuando hay destino. Con lote 512, el tiempo sin destino baja un 67,1 % en cohortes y un 24,5 % en fechas únicas. En cohortes con lote 32 y Parquet, la diferencia es de 1,42 ms frente a desviaciones de 3,28 y 2,70 ms entre repeticiones. Los demás casos medidos reducen la mediana.

El pico de RAM se lee mediante `VmHWM` después de `exec` y antes del perfil. En las cohortes sin destino de lote 4096 pasa de 604,41 a 604,91 MB decimales. En fechas únicas del mismo tamaño pasa de 612,37 a 611,00 MB. El informe conserva los bytes sin redondear, el caudal y los tiempos de cada repetición. El tiempo de proceso incluye las importaciones, el calentamiento, las repeticiones y el perfil posterior.

El perfil posterior registra 288 reanudaciones de `_group_errors` en cohortes con lote 512 y 16.416 en fechas únicas. La instrumentación de esas llamadas eleva el coste observado por `cProfile`. Sus tiempos se conservan como diagnóstico y la comparación de velocidad utiliza las ejecuciones sin perfilador.

## Selección del recorrido y comprobaciones

La primera variante agrupaba con NumPy y construía otro diccionario antes de combinar los valores. En la prueba aislada empeoró los lotes con fechas únicas. La combinación de grupos y estado en una sola pasada mejoró ese caso, pero vectorizar todos los tamaños empeoró la evaluación completa con lote 32. La ruta vectorizada también añadió un 4,2 % en cohortes y un 5,7 % en fechas únicas con 65 filas por lote.

Se seleccionó la reducción secuencial hasta 128 filas. La comprobación en 128 y 129 no observó una regresión al cambiar de ruta. En cohortes, los tiempos fueron 10,52 → 10,33 ms y 10,96 → 8,14 ms, respectivamente. En fechas únicas fueron 33,46 → 27,33 ms y 32,84 → 28,23 ms. Este corte se respalda en los tamaños registrados, sin atribuirle optimalidad para toda carga posible.

Pasaron 89 pruebas y se detectaron seis mutaciones dirigidas. Las pruebas comparan el estado con un oráculo secuencial, incluido el orden de claves, en lotes de 31, 32, 33, 64, 65, 128, 129, 512 y 4096 filas. Cubren `int64`, `uint64` y fechas en microsegundos, desbordamientos, presupuestos y rechazos sin cambios parciales. Un caso añade errores unitarios a un estado previo de magnitud 2⁵³. Otro cambia el orden de filas para hacer observable el redondeo y exigir el mismo resultado que la referencia.

El estado, las métricas y el SHA-256 de las predicciones guardadas coinciden exactamente. No se han ampliado tolerancias numéricas. La cobertura de líneas ejecutables es del 100 % en `_group_errors` y `update`. El [informe de verificación](../../reports/resources/session-metrics-performance-quality.json) conserva la complejidad, la convención CRAP y las mutaciones ejecutadas. El [informe de rendimiento](../../reports/resources/session-metrics-performance.json) incluye referencias, versiones, perfiles, memoria, tamaños cercanos al corte y alternativas descartadas.

```bash
CUDA_VISIBLE_DEVICES='' uv run --extra cuda --extra reinforcement python benchmarks/session_metrics.py \
  --work artifacts/benchmarks/session-metrics \
  --output artifacts/benchmarks/session-metrics.json

CUDA_VISIBLE_DEVICES='' uv run --extra cuda --extra reinforcement python benchmarks/session_metrics.py \
  --worker --layout unique --batch-size 129 \
  --work artifacts/benchmarks/session-metrics
```

Para repetir la referencia se utiliza el mismo benchmark con `evaluation/session_metrics.py` de `889f649` en una copia separada. Las medidas son diagnósticos CPU con datos residentes y cachés calientes. No miden entrenamiento ni aceleración GPU. La suite completa CUDA no se ejecutó mientras la GPU tenía otra carga activa. Energía, coste monetario, VRAM y transferencias CPU/GPU no se midieron.
