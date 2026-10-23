# Matriz técnica de programas regionales y modos de memoria

La matriz completó 208 condiciones en CUDA, con 26 configuraciones admitidas, FP32 y FP64, lotes de 1 y 256 flujos de prueba y dos órdenes globales inversos. Las 22 configuraciones incompatibles del producto original de 48 se comprobaron mediante el constructor real antes de inicializar CUDA. Sus rechazos quedan registrados con su motivo. No se han entrenado estas variantes ni medido cambios de MAE.

El [guion](../../../benchmarks/regional_program_matrix_cuda.py) y la [especificación](../../../docs/experiments/regional-program-v1.md) describen el recorrido. [matrix-summary.json](matrix-summary.json) conserva las 104 combinaciones de configuración, precisión y lote con sus dos órdenes. Los cuatro archivos `cost-float32-1.json`, `cost-float32-256.json`, `cost-float64-1.json` y `cost-float64-256.json` contienen todos los registros y muestras. [source-manifest.json](source-manifest.json) conserva las huellas comunes de fuentes, fixtures y configuración.

El guion y sus tests se incorporaron en `830e6802b` después de medir. Se comprobó que sus fuentes confirmadas coinciden con las huellas de las ejecuciones. El modelo conserva las fuentes de `6e12bfff4`. Ruff, el formato y la comprobación del repositorio pasaron con el entorno del proyecto. [repository-checks.json](repository-checks.json) conserva los resultados y los intentos previos con un entorno incompleto.

## Configuraciones y comprobaciones

Se utilizan los modos existentes `transformer_direct`, `mac_disabled`, `mac_frozen` y `mac_online`, sobre el mismo codificador de fixture. No son cuatro codificadores distintos. El primero no construye MAC. El segundo conserva el módulo con lectura y actualización de memoria desactivadas. El tercero permite lectura sin actualización asociativa y el cuarto permite actualizar los pesos rápidos. PT4 solo está admitido con `mac_online`. El selector por norma exige ese contexto.

El perfil usa H = 32, semilla 42 y la cabeza de cuantiles existente. Los cinco grupos de modo y contexto conservan 25.093 parámetros comunes. Sin MAC hay 25.093 parámetros almacenados. Los modos con MAC y sin contexto tienen 35.943. `mac_online` con `market_memory_v1` tiene 43.209, con 18.116 adicionales sobre la parte común. Los calendarios de un mismo grupo conservan exactamente sus parámetros. Estos recuentos no acreditan la misma capacidad efectiva ni el mismo presupuesto entre modos.

Cada condición reconstruye dos observaciones anteriores y repite la tercera desde su propio estado rápido y contexto confirmado. Se realizaron 416 preparaciones de prefijo, 416 calentamientos y 1.040 mediciones, que suman 1.872 preparaciones. Son repeticiones de tres eventos de fixture, no sesiones financieras independientes. Los 256 flujos tampoco representan 256 acciones históricas evaluadas.

Las entradas, los parámetros con nombres comunes en cada precisión, los estados iniciales reconstruidos y los resultados repetidos superaron sus comparaciones exactas. Las 40 parejas sin resumen conservaron todos los tensores del resultado, excluida la traza regional, dentro del mismo modo, contexto, precisión, tamaño y orden. Se comprobaron RNG, entradas inalteradas, ausencia de gradientes compartidos y los pasos asociativos permitidos por cada modo. En `mac_online` se actualizan los pesos rápidos, sin pasos de optimizador sobre los parámetros compartidos.

Los 13 casos del nuevo protocolo pasaron sin errores ni omisiones. Se detectaron los cuatro mutantes dirigidos a recuento, orden global, admisión de contexto e importación de Torch. Los controles de origen y restaurados suman 52 ejecuciones de casos cada uno. Las versiones mutadas provocaron 22 fallos en fase `call`, sin contar errores de colección como detecciones. Son los mismos 13 casos repetidos y cuatro objetivos concretos, no una puntuación de mutación de todo el repositorio. [mutations.json](mutations.json) y [verification.json](verification.json) conservan los recibos.

coverage.py 7.16.1 registró 297 de 342 sentencias y 84 de 120 ramas del nuevo benchmark, un 86,84 % y un 70 %. La medida combina los tests del protocolo con otro estrato completo instrumentado de 52 condiciones. Sus tiempos quedan fuera de la comparación de coste. Radon 6.0.1 analizó 15 funciones. Con `CRAP = C² × (1 − c)³ + C`, la función `condition` tiene complejidad 52 y CRAP 54,09 y `run` tiene complejidad 38 y CRAP 45,44. Los recorridos de error y parte del CLI siguen sin cobertura registrada. [coverage.json](coverage.json) y [quality.json](quality.json) mantienen el detalle sin ajustar la puntuación.

El observador de llamadas aparece con cobertura de líneas cero, aunque sus contadores se comprobaron durante los prefijos. Python suspende el trazado mientras ejecuta la función de `sys.setprofile`, como recoge la [documentación de Python](https://docs.python.org/3.12/library/sys.html#sys.call_tracing). Por eso esa cifra no permite concluir que el observador no se haya ejecutado.

## Coste observado

Se utilizó una NVIDIA GeForce RTX 4070 Laptop GPU, con driver 595.91.07, PyTorch 2.14.0+cu130, CUDA 13.0 y Python 3.12.14. Cada proceso usó dos hilos CPU y uno de interoperabilidad, con TF32, autocast y fastpath de MHA desactivados y algoritmos deterministas activados. Solo residía una instancia GPU de predictor por condición.

El intervalo rodea `FinancialPredictor.prepare` completo y la sincronización que confirma su finalización. Incluye consulta de contexto cuando corresponde, codificación, fusión, MAC, cabeza de cuantiles y propuesta de estado. La construcción, los dos prefijos, las comprobaciones y la escritura de informes quedan fuera y sus costes se registran aparte. Cada registro contiene dos calentamientos y cinco muestras. No se estiman percentiles de cola ni intervalos de confianza.

Esta tabla reúne las medianas en milisegundos del grupo `mac_online/market_memory_v1`, en formato orden original / inverso. Los otros grupos y todas las muestras permanecen en los JSON.

| Programa | FP32, B = 1 | FP32, B = 256 | FP64, B = 1 | FP64, B = 256 |
| --- | ---: | ---: | ---: | ---: |
| PT4 posterior a la fusión | 65,905 / 103,097 | 115,648 / 71,722 | 38,580 / 53,731 | 74,737 / 100,035 |
| Resumen final | 89,138 / 82,101 | 137,270 / 71,883 | 33,578 / 65,271 | 120,562 / 95,062 |
| Resumen intercalado | 87,580 / 134,370 | 80,396 / 88,917 | 63,650 / 54,011 | 44,353 / 32,131 |
| Regla por norma contextual | 98,158 / 78,308 | 140,443 / 90,416 | 82,177 / 56,310 | 63,710 / 54,848 |
| Final sin resumen | 68,582 / 74,329 | 93,543 / 112,949 | 59,748 / 30,599 | 73,876 / 50,949 |
| Intercalado sin resumen | 61,668 / 93,557 | 77,446 / 109,200 | 56,143 / 80,833 | 124,148 / 58,080 |

La regla contextual eligió el calendario intercalado en todas las filas de esta fixture y en ambos órdenes. Esta matriz no comprueba por sí sola todos los bordes de su umbral. Los calendarios fijos conservan la posición de consulta anterior a MAC. Compararlos con el control de PT4 posterior a la fusión cambia también la posición de consulta.

La variación entre muestras y órdenes impide afirmar una aceleración estable. El equipo tenía carga concurrente. Las instantáneas registran `llama-server` con 5.438 MiB de VRAM y el equipo declaraba 16 CPU lógicas con carga media del sistema elevada. La ocupación de VRAM no mide la utilización de cálculo de ese otro proceso. No se aislaron sus recursos ni se atribuye toda la variación a una sola causa. Estos tiempos tampoco son directamente comparables con el perfil anterior de cabeza escalar.

Los máximos registrados en las muestras fueron 104.398.336 bytes asignados y 123.731.968 reservados por Torch. El RSS muestreado alcanzó 1.582.116.864 bytes. El máximo de vida del proceso comunicado por `getrusage` fue 1.580.752.896 bytes. Se mantienen separados porque proceden de lecturas distintas y no son picos por condición. El mayor conjunto de tensores CPU retenidos para comparación se registra por proceso y se mantuvo por debajo de 128 MiB. El asignador tenía una cota de 512 MiB y se exigieron al menos 256 MiB globales libres. Estas cifras no son la VRAM total del proceso.

Los cuatro procesos completos sumaron 541,425 segundos y sus ejecuciones supervisadas 563,349 segundos. Esa suma no incluye desarrollo, intervalos entre procesos, instrumentación ni intentos incompletos. No se midieron energía ni transferencias físicas. Las rutas personales se normalizan en los artefactos públicos y los procesos GPU se identifican por su ejecutable, con las huellas de los originales conservadas.

## Intentos conservados y decisiones

La primera matriz agotó 180 segundos exteriores y guardó 62 registros completos de los 208 previstos. Quedó incompleta, sin afirmar su cierre interno. [incomplete-attempt.json](incomplete-attempt.json) e [incomplete-receipt.json](incomplete-receipt.json) conservan ese intento. La repetición se dividió en cuatro estratos con límites internos de 240 segundos, exteriores de 300 y 1.500 segundos técnicos para la ronda. Se conservaron combinaciones, muestras y controles, sin escoger los tiempos más favorables.

La primera ronda de mutación agotó sus 240 segundos después de completar dos objetivos y ejecutar origen y mutante del tercero. La restauración del tercero no llegó a verificarse y el cuarto no empezó. Los dos pendientes se repitieron con controles completos en una fase de verificación separada, con un límite de 600 segundos que también cubrió la instrumentación de cobertura. La evidencia incompleta no se suma como detección aprobada.

El [perfil de operadores](profiling.json) contiene dos preparaciones instrumentadas por programa y no sustituye la medición de latencia. Una [sonda adicional de finitud](finite-duplicate-probe.json), repetida tres veces en FP32 y B = 256 con programa intercalado, observó 87 peticiones diferidas, 22 aciertos de la caché existente y 68 indicadores CUDA por preparación. No encontró candidatos vivos adicionales que una caché débil pudiera evitar entre tramos. Esa caché no se incorporó. La sonda no mide aceleración ni demuestra que ese resultado se extienda a todas las cargas.

Las fuentes del modelo, sus configuraciones científicas y sus checkpoints permanecen en la versión del incremento anterior. La guarda de aprendizaje conservó su huella y el test final de 2024 siguió cerrado. No se ejecutaron entrenamientos, pasos de optimizador ni evaluaciones científicas. Siguen pendientes la jerarquía anterior a MAC y el selector aprendido de [#709](https://github.com/GonxKZ/mars-titan/issues/709), junto al ciclo exterior de propuestas de [#711](https://github.com/GonxKZ/mars-titan/issues/711). Esta matriz no aporta una mejora financiera ni evidencia de superioridad frente al estado del arte.
