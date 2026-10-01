# Entornos por lotes, ventajas y recuperación

Medición local del 28 de septiembre de 2026 sobre un AMD Ryzen 9 8945HS, con 8 núcleos y 16 hilos. El motor es C++20, compilado con Clang 21.1.8 en Release, `-O3`, sin `fast-math` ni contracción implícita. La campaña CUDA existente siguió activa. No se modificó el modo de energía ni se detuvieron otras aplicaciones. No se midieron energía ni coste monetario.

## Recorrido contable

El [ejecutable](../../native/benchmarks/batch_rollout.cpp) genera una cinta analítica y aplica la misma secuencia de acciones por entorno a sesiones secuenciales o al lote. El tiempo total incluye construcción, calentamiento, reset, pasos, observaciones, comprobación de salidas y una recuperación en memoria. El rollout y la latencia por paso se registran por separado. No se mide carga Parquet, inferencia neural ni persistencia de checkpoints en disco.

Cada forma tiene tres procesos por modo, con orden rotatorio. El calentamiento ejecuta un paso y vuelve al inicio. Todos los modos producen el mismo checksum de recompensas y observaciones. Las pruebas de integración comparan también valores y estados completos, con acciones distintas por entorno y repartos no divisibles entre trabajadores. Compartir cinta no convierte las carteras en escenarios de mercado independientes.

| Entornos × activos × sesiones | Referencia secuencial, mediana total | Lote, mediana total | Trabajadores |
| --- | ---: | ---: | ---: |
| 16 × 8 × 64 | 0,001339 s | 0,001060 s | 1 |
| 256 × 64 × 128 | 0,162939 s | 0,054777 s | 8 |
| 2048 × 128 × 128 | 2,912347 s | 0,922871 s | 8 |
| 4096 × 64 × 128 | 2,984027 s | 0,982598 s | 8 |

En el lote de 16 entornos, ocho trabajadores tardan 0,002823 s. Se conserva un trabajador por defecto y la elección queda explícita. Para las tres formas mayores, la mediana con ocho trabajadores es aproximadamente tres veces menor que la referencia secuencial actual. Tres repeticiones con carga concurrente no permiten prometer ese cociente en otros equipos o cargas. Los tiempos de proceso, dispersión, p50/p95/p99 y RSS por ejecución están en el [registro JSON](batched-rl-benchmark.json).

La optimización se hizo en dos pasos medidos. Primero se evitó validar repetidamente una cinta compartida e inmutable al crear o recuperar cada sesión. Con 4096 entornos y ocho trabajadores, el tiempo total pasó de 1,8403 a 1,1136 s. Después se especializó el workspace de la contabilidad para una cuenta, conservando la misma función matemática y la ruta de hasta 32 cuentas. Ese caso evita inicializar parciales de 31 cuentas que no intervienen. La mediana pasó de 1,1136 a 0,9826 s. Los agregados siguen usando las mismas sumas compensadas y se comprueba paridad con la ruta de varias cuentas.

Callgrind registró inicialmente 7.249.925.516 instrucciones simuladas en el recorrido de 256 entornos, 64 activos, 128 sesiones y un trabajador. La validación de cintas concentró 869.557.248 y `memset` 4.347.933.565. Son contadores del instrumento, no contadores de hardware ni porcentajes de tiempo. `perf` no permitió acceder a los eventos con `perf_event_paranoid=4`. Se conservó esa configuración del sistema.

Al repetir el perfil completo, el total bajó a 2.101.072.550 instrucciones simuladas y `memset` a 90.674.585. La construcción de observaciones pasó a ser el componente con más instrucciones propias. Los informes de Clang indican que varios bucles contables no se vectorizan bajo la semántica numérica vigente. No se relajó esa semántica ni se dedujo de estas medidas que el sistema alcanzase el límite del hardware.

Para repetir la comparación desde la raíz, con la compilación Release preparada:

```bash
uv run python scripts/benchmark_native_batches.py \
  --binary build/native/native-release/mars-titan-batch-benchmark \
  --output artifacts/native/batched-rl-comparison.json \
  --concurrent-workload 'Campaña de referencias CUDA activa'
```

El registro identifica el ejecutable, la biblioteca contable en Linux y el código del instrumento mediante SHA-256. Las rondas previas se conservan separadas. La primera ronda solo registró las huellas del motor emitidas por el programa, por lo que no se le atribuye una huella binaria que no se guardó.

## Ventajas y atomicidad

La referencia de GAE es la función de `e977545`, ejecutada una vez por entorno. Con 128 pasos, semilla 20260928, tres calentamientos y 21 repeticiones alternadas, la versión por lotes produce resultados exactamente iguales en `float64`. Las medianas para 1, 32, 128 y 512 entornos pasan de 0,0591, 2,0025, 8,4259 y 33,6871 ms a 0,0413, 0,1579, 0,2256 y 0,4856 ms. Es una medición de la primitiva CPU, no de PPO completo. El temporal de continuidad ocupa 512 KiB para 128 × 512. Una primera versión que empeoraba el caso 1D fue descartada. Las [medidas y comprobaciones](batched-gae-quality.json) incluyen dispersión, paridad y ocho mutaciones detectadas.

La revisión de recuperación encontró dos defectos distintos. Una restauración fallida podía dejar la cartera adelantada respecto al modelo y al cursor del entrenador. Una transición fallida podía conservar una orden recién emitida. Ahora se conserva el estado confirmado y se revierte el intento. Los identificadores corporativos duplicados se rechazan antes de operar.

La protección de `FinancialEnv.step` añade entre un 1,1 % y un 6,5 % de tiempo mediano en los casos de 4, 128 y 512 activos medidos con siete repeticiones, sin un cambio apreciable de RSS. Se conserva ese coste por corrección. No se presenta como aceleración. Las [medidas](recovery-step-measurements.json) y la [cobertura y mutación](recovery-quality.json) registran su alcance. La copia completa de recuperación del entrenador se crea en CPU únicamente al restaurar, no en cada paso.

## Límites científicos

El filtro HMM se contrasta con enumeración exacta, permutación de estados, fechas, máscaras y recuperación. La revisión adversarial detectó pérdida de diferencias gaussianas al restar densidades absolutas muy grandes. La versión corregida usa diferencias factorizadas y suma compensada. Rechaza atómicamente las cancelaciones que superan la tolerancia declarada. Las mediciones del filtro anteriores a esta corrección no describen su versión final y no se publican como tales.

Las pruebas del motor y los benchmarks no acreditan una política experta, una mejora del MAE real o rentabilidad histórica. La integración del lote con inferencia y entrenamiento neural sigue pendiente de una ventana CUDA exclusiva y de un protocolo registrado. Las [hipótesis de especialistas](../../docs/research/contextual-experts.md) requieren datos comparables, controles de capacidad y evaluación temporal.

La [verificación de esta entrega](rl-batch-verification.json) recoge 2071 pruebas Python superadas, 37 omitidas y 120 apartadas por necesitar CUDA. La GPU seguía reservada a la campaña. Se corrigieron dos fixtures anteriores que apuntaban a una dependencia retirada. Permanecen dos avisos de Gymnasium sobre límites infinitos del espacio predictivo existente.

Los seis tests nativos pasan en Release con Clang y GCC, análisis estático, ASan/UBSan, TSan y MSan. El perfil de fuzzing añade dos pruebas de 1000 entradas sobre el núcleo y la sesión. Se detectaron cuatro mutaciones del lote y nueve del filtro entre las comprobaciones iniciales y las regresiones numéricas. Las coberturas finales de líneas son 264/278 para el lote y 288/292 para el filtro. El [informe LLVM y Lizard](batched-native-quality.json) conserva complejidad y CRAP. El exportador LLVM avisa de cuatro mapas con hash cero sin perfil para dos métodos inline compartidos de `CashMovements`. El aviso se mantiene y esas cifras se usan como diagnóstico, no como garantía de corrección.
