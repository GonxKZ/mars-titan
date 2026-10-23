# Jerarquía regional antes de MAC: comprobaciones y coste CUDA

La familia opcional de ocho estados está implementada y comprobada en la rama de [#723](https://github.com/GonxKZ/mars-titan/pull/723). Reutiliza los parámetros regionales, añade dos grupos y una raíz como estado temporal y conserva la ruta anterior cuando está desactivada. La integración en `develop` sigue pendiente. La [especificación](../../../docs/experiments/regional-hierarchy-v1.md) describe las ecuaciones, controles y límites.

Los cuatro estratos completaron 128 registros distintos y 1152 preparaciones: 256 de prefijo, 256 calentamientos y 640 muestras cronometradas. Se compararon siete configuraciones nuevas y la referencia `legacy_self_uniform_interleaved`, con y sin PT4, FP32/FP64, lotes de 1/256 y dos órdenes inversos. El fixture usa `mac_online`, H = 32, semilla 42 y la cabeza de cuantiles existente. Las 256 filas son flujos del fixture, no una evaluación sobre 256 acciones reales.

No hay una aceleración consistente del árbol ni una mejora predictiva medida. La matriz no utiliza etiquetas financieras. Los parámetros compartidos permanecen congelados y la memoria rápida de MAC realiza su actualización asociativa identificada. No se han ejecutado entrenamientos, postentrenamientos, RL o evaluación predictiva. La campaña completa y el test final siguen cerrados.

## Latencia, memoria y trabajo

Este ejemplo conserva las dos pasadas completas de PT4, FP32 y lote de 256. Cada celda es la mediana de cinco muestras de `FinancialPredictor.prepare` y la sincronización CUDA. El prefijo, la construcción del modelo y las comprobaciones por hashes quedan fuera de ese intervalo y dentro del tiempo total de proceso.

| Configuración | Orden original, ms | Orden inverso, ms |
| --- | ---: | ---: |
| Referencia anterior | 45,092 | 42,688 |
| Árbol | 36,728 | 61,048 |
| Estrella | 45,397 | 53,040 |
| Árbol con partición permutada | 48,884 | 48,944 |
| Árbol sin realimentación auxiliar | 50,965 | 45,441 |
| Control plano | 47,122 | 64,541 |
| Árbol con salto común | 50,716 | 59,981 |
| Control plano con salto común | 43,136 | 46,396 |

El árbol cambia de signo respecto a la referencia entre esas pasadas. No se agrupan los tiempos para escoger el resultado favorable. Los [128 registros en CSV](aggregate/conditions.csv) y el [resumen JSON](aggregate/summary.json) mantienen separados contexto, precisión, lote y orden, con cada denominador tomado de su misma condición. De las 56 parejas de configuraciones nuevas entre ambos órdenes, dos son más rápidas en ambos, 31 más lentas en ambos y 23 cambian de signo. Son descripciones de esta carga, sin intervalos de confianza ni selección de un ganador.

Las siete configuraciones nuevas ejecutan 20 evaluaciones locales y 10 auxiliares por fila. La referencia anterior ejecuta 20 y dos, con otro calendario, proyecciones y comprobaciones numéricas. Su comparación no aísla únicamente la jerarquía. El control plano nuevo comparte las 30 evaluaciones y el calendario del árbol. Una cuenta de receptores no es una cuenta de FLOPs.

Todas las configuraciones de un mismo contexto conservan nombres, formas y valores de los parámetros. Hay 35 943 parámetros sin PT4 y 43 209 con PT4. La familia nueva añade geometría y estado temporal, sin parámetros aprendibles adicionales. Los hashes completos incluyen buffers geométricos y pueden diferir entre topologías. La igualdad de los pesos entre ellas procede de las comparaciones exactas del productor fijado, no de reconstruir pesos que no se archivaron.

El máximo comunicado por Torch fue de 104 398 848 bytes asignados (99,5625 MiB) y 130 023 424 reservados (124 MiB), frente al límite de asignador de 512 MiB. El RSS muestreado máximo fue de 1 588 637 696 bytes (1515,043 MiB). El máximo de vida del proceso comunicado por otra interfaz se conserva por separado. No son picos de memoria atribuibles a cada operador ni una medida completa del consumo físico de GPU. La estimación lógica de admisión de 128 MiB tampoco certifica esos picos.

| Estrato | Proceso completo, s | Con supervisión, s |
| --- | ---: | ---: |
| FP32, lote 1 | 48,539 | 52,419 |
| FP32, lote 256 | 61,216 | 64,589 |
| FP64, lote 1 | 47,497 | 51,498 |
| FP64, lote 256 | 79,410 | 83,489 |
| Total | 236,661 | 251,995 |

Se utilizó una RTX 4070 Laptop de 8188 MiB, CUDA 13.0 y PyTorch 2.14.0+cu130, con TF32 y autocast desactivados. El equipo tiene un Ryzen 9 8945HS, 16 CPU lógicas y 32 313 376 768 bytes de RAM. Cada proceso fijó dos hilos CPU y uno de interoperabilidad. Había otras aplicaciones residentes, incluido un servidor que comunicaba 5438 MiB de GPU. Se conservan sus consultas de hardware y las cargas medias. Cinco muestras no permiten estimar colas de latencia o demostrar una aceleración estable. No se midieron energía, contadores de kernels o volúmenes físicos de transferencias.

## Pruebas y revisión adversarial

La [conciliación de pruebas](verification.json) recoge 484 casos únicos del proyecto. Una ronda instrumentada pasó 369 casos de modelos, archivo, integración y regresión. La corrida nativa de 47 casos pasó 46 y detectó una suposición errónea de orden en la prueba restante. Ese caso corregido está incluido en los 369. Otra ronda pasó 88 casos: 58 nuevos del payload, 19 de sesión repetidos y 11 del protocolo. Esos casos repetidos no aumentan el total único. El agregador tiene otras 20 pruebas con fixtures identificados, que no se confunden con las 640 muestras CUDA.

Las comprobaciones cubren las 32 máscaras del componente. La entrada financiera exige precios y gráficos, por lo que admite ocho y rechaza 24 antes de preparar el núcleo. Incluyen desenrollado independiente, gradientes, diferencias finitas, aislamiento de estados y RNG, checkpoints, publicación nativa, reserva previa a MAC y recuperación sin devolver cuotas. Las trazas exportadas se contrastan con la presencia real de entrada y con la alternativa del mismo flujo.

La revisión encontró dos defectos numéricos de la familia nueva: Q/K podían descartarse antes de comprobar su finitud y una suma residual ausente podía desbordar antes de la máscara. Las 26 regresiones reprodujeron ambos. Las guardas se añadieron antes de las medidas. Los [mutantes dirigidos](directed-mutations.json) detectan los dos defectos, la infrarreserva, la alteración de disponibilidad, errores de protocolo, inversión del cociente y mezcla de interfaces de memoria. Se detectaron 12 objetivos del proyecto y dos del agregador, con sus bases y restauraciones correctas. No es una puntuación de mutación de todo el repositorio.

La [cobertura](coverage.json) y el [CRAP](crap.json) registran 175 funciones de diez fuentes seleccionadas, 1974/2437 sentencias y 653/1034 ramas. El componente nuevo tiene 283/297 sentencias y 96/108 ramas. `RegionalHierarchy.export_trace` pasó de 11/14 sentencias y CRAP 37,275 a 14/14 sentencias, 6/6 ramas y CRAP 29 tras las pruebas adicionales. Se usaron coverage.py 7.16.1 y Radon 6.0.1, con las regiones de coverage.py y la fórmula `C² × (1 − cobertura de sentencias)³ + C`. Las ramas se publican aparte.

La instrumentación nominal del benchmark repitió 32 registros y 288 preparaciones en una ejecución separada, de 80,226 segundos supervisados. Esos tiempos no entran en la tabla de coste. El agregador conserva [cobertura propia](coverage-aggregation.json) y [CRAP propio](crap-aggregation.json), incluidos sus casos negativos. Ninguna de estas medidas demuestra ausencia de errores o superioridad predictiva.

Se mantienen los intentos que fallaron o quedaron incompletos en el registro de verificaciones. El filtrado de cobertura por nombres de módulos falló durante la importación de Torch. El filtrado por archivos pasó una comprobación acotada y las rondas posteriores. [Coverage.py documenta las posibles importaciones adicionales al usar nombres de módulos](https://coverage.readthedocs.io/en/7.16.1/source.html). La ejecución amplia con instrumentación agotó 240 segundos y no se declara aprobada. Los errores de fixtures de dtype, contexto pendiente, directorio aún inexistente y orden canónico se corrigieron sin relajar el producto ni retirar casos.

## Artefactos y reproducción

Cada estrato conserva `measurement.json` y `source-manifest.json` en su carpeta. Los cuatro manifiestos coinciden y las fuentes y la guarda de aprendizaje permanecieron intactas. La [procedencia de publicación](publication-provenance.json) conserva las huellas originales y las publicadas. Se han sustituido rutas locales de ejecutables por sus nombres y la ubicación de la fuente del agregador en el informe de cobertura. Al regenerar el resumen se actualizaron los tamaños y hashes de las cuatro mediciones publicadas. Sus demás campos y los 128 registros numéricos coinciden exactamente con los originales. No se publican pesos, datasets o checkpoints históricos.

Los binarios nativos se reutilizaron sin recompilarlos. Sus nombres y huellas figuran en el [contexto de ejecución](runtime-context.json). Las pruebas nativas requieren el backend y las variables descritas en [native/README.md](../../../native/README.md). Las comprobaciones se ejecutaron localmente y no se lanzó la suite científica completa.

Con el entorno del proyecto preparado, el protocolo puede consultarse sin importar Torch ni crear archivos:

```bash
uv run --no-sync python benchmarks/regional_hierarchy_matrix_cuda.py
```

Para repetir la matriz, las salidas deben ser nuevas y externas al repositorio. Los procesos se ejecutan en secuencia:

```bash
hierarchy_runs=$(mktemp -d)
for dtype in float32 float64
do
  for rows in 1 256
  do
    uv run --no-sync python benchmarks/regional_hierarchy_matrix_cuda.py \
      --run --dtype "$dtype" --batch "$rows" \
      --output "$hierarchy_runs/$dtype-$rows"
  done
done
uv run --no-sync python \
  reports/engineering/regional-hierarchy-matrix-20261022/aggregate_hierarchy_measurements.py \
  --input-dir "$hierarchy_runs" --output "$hierarchy_runs/summary"
```

Para comprobar solo la agregación publicada, se puede usar esa carpeta como `--input-dir` y otra salida nueva. Sus 20 pruebas se ejecutan con:

```bash
uv run --no-sync python \
  reports/engineering/regional-hierarchy-matrix-20261022/test_aggregate_hierarchy_measurements.py
```
