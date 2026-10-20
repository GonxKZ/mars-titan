# Feedback parcial de cohortes PT2

Está implementada la recepción recuperable de resultados parciales y la cancelación explícita de cohortes incompletas. Es una pieza de la integración de [#701](https://github.com/GonxKZ/mars-titan/issues/701), que continúa pendiente. No se ha conectado aún al consumidor común ni a la publicación atómica de `FinancialSession`.

La selección final pasó 78 casos CPU, 49 nuevos y 29 anteriores de PT2, sin fallos ni omisiones. No utilizó datos reales ni pasos de optimizador. Los 13 mutantes dirigidos quedaron detectados con las referencias inicial y restaurada aprobadas. La campaña anterior a la optimización detectó sus 12 mutantes. No se suman las repeticiones como casos o mutantes nuevos.

El [recibo](verification.json), las [mutaciones finales](mutations.json), las [anteriores](initial-mutations.json) y el [análisis de calidad](quality.json) identifican fuentes y límites. Coverage.py registra 447/449 sentencias y 78/78 ramas en tres módulos. Las dos sentencias no ejecutadas pertenecen al rechazo de valores JSON no serializables. El mayor CRAP es 34 en la restauración histórica de `OnlineConformal`. En el código nuevo es 18 en `unpack_calibrator`, 17 en `validate_cohorts` y 17 en la resolución de pendientes. La cobertura de ramas no enumera todas las combinaciones de operandos booleanos y no demuestra ausencia de errores.

Las pruebas adversariales detectaron dos defectos de la primera implementación. La restauración intentaba serializar cuantiles antes de comprobar su límite de filas y algunos estados dañados propagaban un `TypeError` del calibrador. Se añadieron los casos que fallaban y se corrigieron ambos comportamientos. Los recibos intermedios se conservan fuera del repositorio.

## Medida de coste

La medida usa CPU AMD Ryzen 9 8945HS, Python 3.12.14 y NumPy 2.5.3. Recorre 12 condiciones: 64 o 256 filas, una o cuatro cohortes pendientes y feedback completo, dividido o con exclusión. Cada condición tiene tres calentamientos y 30 repeticiones, con orden rotado. Hay 396 recorridos por ronda. Los valores son fixtures y no representan resultados predictivos.

Cada recorrido propone una nueva emisión, recibe los resultados de la cohorte más antigua y exporta el estado. El feedback dividido requiere dos propuestas de recepción. La referencia de PT2 y la restauración de auditoría se comprueban fuera de cada tramo cronometrado y están dentro del tiempo de proceso. La emisión no sustituye una confirmación en disco.

El [perfil inicial](profile-before.txt) atribuyó la mayor parte del tiempo a `copy.deepcopy`. Los metadatos de las filas ya validadas contienen escalares inmutables. Se sustituyó la copia recursiva de esos metadatos por la copia de sus contenedores. Se conservaron las comprobaciones del calibrador, los límites y el aislamiento. El mutante adicional que comparte esos contenedores quedó detectado.

Estos son los p50 del recorrido completo con 256 filas y cuatro cohortes pendientes:

| Feedback | Antes, ms | Primera ronda candidata, ms | Ronda final, ms |
| --- | ---: | ---: | ---: |
| Completo | 30,36 | 11,36 | 14,50 |
| Dividido | 43,52 | 15,69 | 20,04 |
| Exclusión de una fila | 28,80 | 10,52 | 13,57 |

Los p50 bajaron en las doce condiciones de las tres rondas candidatas sin otras comprobaciones del proyecto. Las rondas no se intercalaron por versión y existe variación temporal de CPU. Los p99 de 30 recorridos son descripciones de esa muestra, no garantías de latencia. No se afirma que mejoren todas las métricas de cola.

El tiempo externo del comando pasó de 10,98 s a 7,60 s en la ronda final. El interno pasó de 9,27 s a 5,98 s y está incluido en el externo. La primera ronda candidata dio 6,41 s externos y 4,85 s internos. La segunda ronda candidata dio 7,45 s externos y 5,96 s internos. El pico observado por `/usr/bin/time` fue aproximadamente 678 MB en las rondas. No se observa ahorro de RAM. El contador interno de Python registró unos 556 MB y tiene otro ámbito. No se suman ambos contadores. No se midieron energía, VRAM ni transferencias GPU.

La [comparación legible por máquina](comparison.json) conserva percentiles, dispersión, caudal derivado y bytes por condición. Los estados serializados tienen el mismo tamaño antes y después. Se comprobó paridad con el calibrador original y recuperación en los 396 recorridos de cada ronda, además de la inmutabilidad de los snapshots iniciales.

Los recibos [anterior](cost-before.json), [primera candidata](cost-after-first.json), [segunda candidata](cost-after-second.json) y [final](cost-after.json) permanecen íntegros. Las primeras candidatas se ejecutaron sobre cambios locales basados en `633a3298d`. Sus hashes numéricos coinciden con `cdf733b2e`, donde se ejecutó la ronda final. El [programa](benchmark.py) final conserva los mismos árboles de sintaxis de las funciones de medida que el anterior, según el [recibo de paridad](benchmark-parity.json). Solo cambiaron fuera de ellas el formato y los metadatos del informe. Los [tiempos de proceso](process-cost.json) y el [inventario externo](external-artifacts.json) permiten localizar los recibos originales.

No hay estudio científico nuevo ni comprobación CUDA del controlador completo. La política de exclusiones y el presupuesto de JSON están descritos en el [contrato del componente](../../../docs/engineering/mechanism-controller.md). La integración restante de #701 y la evaluación predictiva continúan pendientes.

La [ronda con comprobaciones concurrentes](cost-concurrent-checks.json) coincidió con Ruff, formato y el verificador del repositorio. Se conserva como diagnóstico de concurrencia y no se usa en el contraste principal. La ronda final empezó tras confirmar cerrados esos procesos. El cambio previo del programa solo corrigió el texto que describía los percentiles cuando se modifica el número de repeticiones, sin alterar las funciones cronometradas.
