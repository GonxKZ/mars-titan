# Ridge y XGBoost con la edición de máscaras

Ridge y XGBoost pueden leer la edición `historical_masked_2000_v1` cuando se declara `input_policy`. Sin ese argumento mantienen el contrato estricto y rechazan la edición histórica antes de crear la salida. Este documento describe la adaptación técnica de sus entradas, recibos y presupuestos. No se ha ajustado ningún modelo con ella. El bloqueo de aprendizaje sigue vigente y estas comprobaciones no aportan resultados predictivos.

## Entradas y bits de presencia

Cada fila concatena las modalidades en el orden canónico de `CorpusDataset`. Con la representación histórica con contexto contable son 320 valores de la ventana de 64 sesiones, 384 de noticias, 512 de gráficos, 78 de fundamentales (los 26 conceptos de `JOINT_CONCEPTS` con valor, observación y antigüedad) y 420 de macro (140 indicadores con el mismo triplete). Suman 1.714 columnas. La anchura se obtiene siempre del primer lote leído, no de esta cuenta.

La política con máscaras añade al final los cinco bits de presencia del lector, en el orden precios, noticias, gráficos, fundamentales y macro. Ridge los recibe como 0 o 1 en `float64` y XGBoost en `float32`. Las referencias Titans y la GRU candidata ya reciben esos bits. Sin ellos, un bloque ausente relleno con ceros y un bloque observado no se distinguen en la parte de noticias y gráficos, y Ridge no podría separar el nivel de cada patrón de ausencia. La matriz pasa a 1.719 columnas.

Los bits de precios y gráficos son siempre verdaderos por contrato. Se conservan para mantener el orden de las cinco modalidades común a los modelos neuronales. Tras centrar y escalar, Ridge los convierte en columnas nulas de la Gram y su coeficiente queda en cero. XGBoost no puede dividir sobre una columna constante.

Los recibos tabulares añaden `input_policy`, `mask_contract`, `feature_order` con `presence` al final y la huella de `data/input_policy.py`. En XGBoost la política forma parte de `identity` y de sus opciones, por lo que una continuación no puede mezclar ediciones. La ruta estricta conserva sus claves, su orden y la misma matriz byte a byte.

La población es la del lector neuronal con la misma política. Las pruebas comparan las filas, los objetivos y los identificadores de muestra de las predicciones con los lotes de `CorpusDataset`. También comprueban que un bloque con bit falso solo contiene ceros y que cambiar todos los objetivos deja la matriz de entrada idéntica. Las características proceden solo de las entradas y sus máscaras.

## Ridge por bloques

El ajuste recorre la población dos veces. La primera combina medias y varianzas por bloque. La segunda, en `centered_normal_equations`, acumula en `float64` los productos Z'Z y Z'y de los datos tipificados con `addmm_` y `addmv_`. El centrado final usa las sumas de esa pasada y el recuento de la primera. Una huella SHA-256 exige que ambas pasadas lean los mismos valores.

La memoria crece con el cuadrado del número de columnas y con un bloque, no con las filas. Para 1.719 columnas la Gram ocupa 23,6 MB y un bloque de 256 filas en `float64` 3,5 MB. La matriz completa de unos 15,4 millones de filas de ajuste ocuparía unos 212 GB en `float64` y nunca se forma.

La comprobación usa matrices aleatorias pequeñas con objetivos independientes. No hay una señal que aprender. En CPU se compara la acumulación con una referencia densa `float64`, con bloques de 1, 7 y 256 filas y medias desplazadas que hacen imprescindible el centrado. Con la anchura real de 1.719 columnas y 600 filas, Gram y lado derecho coinciden con tolerancia relativa de 1e-10 y la solución del sistema normal coincide con NumPy. Otra prueba sigue los bloques con referencias débiles y confirma que solo el anterior permanece vivo.

Se probaron cinco mutaciones de la acumulación: sin centrado de la Gram, sin centrado del lado derecho, rasgos `float32`, entradas redondeadas a `float32` y materialización de todos los bloques. Todas hacen fallar alguna prueba. La extracción de la función conserva las mismas operaciones en el mismo orden dentro de `fit_ridge_blocks`. Su paridad en `cuda:0` queda pendiente porque la GPU está ocupada.

## XGBoost con memoria externa

`ExtMemQuantileDMatrix` guarda las páginas cuantizadas en RAM o en disco. La tabla resume las estimaciones para 1.719 columnas. La cota superior de filas es el total de 17.076.024 ventanas del [censo histórico](../data/historical-materialization.md), que incluye todas las particiones. La cifra de unos 15,4 millones procede de la estimación previa a la codificación de la población de ajuste. No es todavía un recuento de objetivos válidos.

| Filas | Caché host `N × (4F + 16)` | Disco denso, 64/128/256 bins | Disco con índices globales, 64/128/256 bins |
|---|---|---|---|
| 17.076.024 | 117,7 GB | 25,7 / 29,4 / 33,0 GB | 62,4 / 66,0 / 69,7 GB |
| 15.400.000 | 106,1 GB | 23,2 / 26,5 / 29,8 GB | 56,3 / 59,6 / 62,9 GB |

Con las ventanas anuales expansivas previstas en [#363](https://github.com/GonxKZ/mars-titan/issues/363), la ventana más tardía es la que fija estos tamaños. La caché en RAM no es viable. Supera el tope de 24 GiB de la ruta y los 32 GB del equipo. La caché en disco depende del formato de las páginas. La estimación densa supone que XGBoost 3.3 guarda para datos sin NaN un índice de bin local por característica, con `ceil(log2(max_bin + 1))` bits por valor. Es una hipótesis que la documentación consultada no confirma. La cota global supone el formato con índices de todo el histograma, `ceil(log2(F × max_bin + 1))` bits. La [documentación de memoria externa](https://xgboost.readthedocs.io/en/release_3.3.0/tutorials/external_memory.html) solo indica que la caché comprimida suele ocupar menos que los datos `float32` y que la caché en disco con GPU es experimental y probablemente más lenta que la CPU.

### Comprobación previa y guardia

`run_external_reference` lee un lote para conocer la anchura y llama a `external_cache_plan` antes de crear la salida y de cargar CUDA. La edición con máscaras exige `max_disk_cache_bytes` cuando la caché va a disco. El plan falla si la estimación densa supera ese presupuesto o si las páginas y la caché de validación no caben en el disco libre. En modo host falla si la cota `float32` supera el presupuesto o `MemAvailable`. El plan queda en el recibo de cada intento con ambas estimaciones, la RAM disponible y el disco libre medidos.

Como la estimación densa no está verificada, `fit_external_boosting` también mide los bytes escritos en el directorio de páginas antes de cada bloque y al terminar la construcción. Si superan el presupuesto declarado falla antes de la primera ronda, sin llenar el disco compartido. Los bytes observados se guardan en la auditoría del modelo. La ruta estricta mantiene el presupuesto de disco opcional y no cambia su identidad cuando no se declara.

La caché de validación de la selección se escribe en el mismo disco. Ocupa `V × (4F + 24)` bytes más 4 KiB por bloque, con su propio límite. Con unos 1,25 millones de filas de 2023, que es una estimación del orden de magnitud, serían unos 8,6 GB. En cada evaluación se vuelve a leer completa.

La [configuración histórica](../../configs/baselines/tabular-historical-masked.json) mantiene la rejilla, la selección y las semillas de la configuración de convergencia estadounidense. Añade la política y un presupuesto de páginas de 32 GiB. Ese valor cubre la estimación densa con 256 bins incluso con la cota superior de filas, pero no la cota global. Si la hipótesis densa fuera falsa, la guardia detendría la construcción y habría que revisar el presupuesto con la medida real.

El 9 de octubre `df` mostraba unos 100 GB libres en el disco del proyecto, a la espera de que terminara la codificación. Esa edición terminó el mismo día y sus muestras ocupan 57,5 GB ([recibo](../../reports/data/historical-edition-v3-targets-20261009.json)). Con 32 GiB de páginas, unos 8,6 GB de validación, tres modelos de hasta 128 MiB y las predicciones en Parquet, el margen sería pequeño. La comprobación previa vuelve a medir el disco libre justo antes de cada intento, y el presupuesto de disco de la campaña A se está preparando en [#363](https://github.com/GonxKZ/mars-titan/issues/363).

### Memoria en modo disco y coste

En modo disco las páginas no ocupan RAM del proceso, pero no hay una cota cerrada del resto. Las partes controladas son la caché del lector (1 GiB por defecto), los bloques de entrada limitados por `max_batch_bytes` y las etiquetas de ajuste, unos 62 MB en `float32` para 15,4 millones de filas. Los búferes internos de XGBoost y CUDA no se han medido. El recibo conserva el RSS máximo del proceso y debe revisarse en la primera ejecución autorizada.

Cada ronda recorre todas las páginas. Con unos 26 a 30 GB por ronda, el tiempo quedará limitado por la lectura del disco y PCIe, además de la validación. No se ha medido el ancho de banda real ni la duración de una ronda. La búsqueda tiene 12 configuraciones y dos repeticiones de semilla, con hasta 2.000 rondas cada una, por lo que el coste total debe estimarse con un piloto antes de comprometer la campaña.

## Alternativas que cambian el algoritmo

Ninguna de estas opciones está activada. Cada una exigiría otra identidad y su propio contraste.

- Reducir `max_bin` solo afecta a la estimación densa en un bit por valor. Cambia la discretización y forma parte de la rejilla, no de un ajuste de memoria.
- `sampling_method="gradient_based"` o `subsample` menor que uno hacen que cada árbol vea una muestra de filas. Reducen memoria de dispositivo, pero el estimador deja de ser determinista con la población completa y no sería comparable con la versión actual.
- Codificar las ausencias como NaN cambia el tratamiento de los bloques ausentes, que pasarían a seguir direcciones por defecto. Además rompe el formato denso y podría aumentar las páginas.
- Comprimir las incrustaciones de noticias y gráficos con PCA o hashing cambia la identidad de las entradas frente a las redes, que reciben los vectores completos.
- XGBoost en CPU con caché en disco mantiene el algoritmo hist, pero con otra implementación y otro coste. Necesitaría un contraste numérico con la ruta CUDA.
- Reducir la población de ajuste queda descartado porque rompe la comparación con filas comunes.

## Comprobaciones y pendientes

Las pruebas nuevas cubren los bits, la población común, la ausencia de fuga de objetivos, la paridad estricta, la propagación de la política en la búsqueda tabular y el plan de memoria. Se ejecutaron en CPU con `CUDA_VISIBLE_DEVICES=-1`, sin ajustes ni pasos de optimizador. Los estimadores se sustituyen por capturas que recorren la factoría y se detienen. Catorce mutaciones dirigidas de la lógica nueva hacen fallar alguna prueba.

Comprobación en `cuda:0`, sin ejecutar rondas:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync pytest -q tests/models/test_external_cache_cuda.py
```

Las dos pruebas pasan desde el 9 de octubre ([resumen](../../reports/engineering/cuda-checks-20261009/README.md)). Usan `learning_doubles`, porque la protección del aprendizaje detiene `fit_external_boosting` en su entrada y antes se omitían sin construir la caché. La primera prueba construye páginas en disco con 20.000 filas aleatorias de 1.719 columnas y contrasta los bytes reales con la estimación densa. La segunda comprueba que la guardia detiene la construcción. En ambas `xgb.train` se sustituye por una excepción. La paridad de `fit_ridge_blocks` y las pruebas CUDA anteriores de Ridge y XGBoost ajustan modelos sobre fixtures y esperan al levantamiento del bloqueo.

Los padres tabulares de postentrenamiento reciben ya los cinco bits después de las modalidades. El [postentrenamiento con la edición histórica](masked-posttraining.md) describe la carga y sus comprobaciones.
