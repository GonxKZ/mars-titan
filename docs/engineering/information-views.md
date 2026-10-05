# Vistas de información sobre la cohorte admitida

`InformationView` aplica una retirada de fuentes o variables a los tensores de una cohorte ya admitida. Conserva filas, etiquetas, orden y dimensiones. Los valores retirados se sustituyen por cero, junto con sus máscaras y edades. Este contrato permite comprobar intervenciones y preparar comparaciones posteriores. No entrena un modelo reducido ni acredita su calidad predictiva.

`ViewDataset` envuelve `CorpusDataset` y transforma sus lotes finales. En ese punto ya se ha resuelto `TemporalInputs.lookup`, incluida la sustitución de macro. Retirar macro antes de esa sustitución permitiría restaurarla. El envoltorio no copia el corpus a otro directorio ni cambia sus reglas de admisión. Una fila incompleta de la referencia completa sigue excluida aunque la nueva vista retire esa modalidad.

## Variables y dependencias

El manifiesto identifica la cohorte de origen, representación, dimensiones, variables, fuentes permitidas y dependencias. Cada variable enumera las posiciones de nivel, presencia y antigüedad que le pertenecen, además de su transformación e historia. Las posiciones deben cubrir exactamente los tensores y no pueden solaparse. Se rechazan dependencias desconocidas, ciclos y variables admitidas que conserven una ruta hacia información retirada.

`corpus_view(dataset, macro_catalog=...)` deriva ese contrato de la representación existente. Reutiliza `FACTOR_DEFINITIONS` para ratios contables y el catálogo macro con sus fórmulas y retardos. Incluye las dependencias que no tienen una columna propia en el tensor. Por ejemplo, el denominador de un ratio sigue siendo información utilizada aunque no aparezca como característica separada.

El catálogo se contrasta con `catalog_sha256` de la admisión del panel temporal efectivo. También se verifica la huella de ese recibo y su vínculo con el panel. En un corpus sin vista temporal, el productor debe haber registrado `macro_catalog_sha256` en el manifiesto al preparar sus datos. Si falta esa procedencia, la fábrica rechaza la vista. Añadir después una declaración a datos antiguos no acredita su origen. Los controles sintéticos registran expresamente su catálogo y no se presentan como materializaciones históricas.

Cuando existen ratios empresariales, `representation_code["company_factors.py"]` debe coincidir con la implementación de la que se obtienen sus dependencias. Así, un catálogo actualizado o una fórmula contable distinta no cambian silenciosamente qué entradas se retiran de vectores ya calculados.

Los precios `open`, `high` y `low` normalizados dependen del primer `close` de la ventana. Los gráficos dependen del prefijo de precios. Retirar `prices/close` también retira esas rutas. Retirar `macro/us_cpi` elimina las variables derivadas que dependen de ella, incluidas sus máscaras y edades. Los nombres y las posiciones proceden de los manifiestos y del catálogo. No se deducen por la dimensión de un vector.

`view.without(sources=..., variables=...)` calcula la retirada transitiva y crea otra identidad. La vista completa devuelve valores exactamente iguales a los del lector original, en buffers separados. La vista reducida conserva los cinco bloques que requieren las referencias existentes. Esta implementación no reduce sus dimensiones ni promete ahorro de lectura, codificación o inferencia. Retirar físicamente columnas sería otra transformación que necesitaría su propia comprobación.

## Padre, memoria, HMM y calibración

`ViewConsumer` exige un contrato de la misma vista de ajuste y uso, cohorte y representación. Los roles admitidos son `parent`, `memory`, `hmm`, `calibration`, `router` y `normalization`. El contrato incluye la huella del artefacto y las de sus dependencias. Todas deben estar presentes y declarar la misma vista. Un normalizador completo no puede reutilizarse silenciosamente dentro de un padre reducido.

El productor guarda la declaración obtenida con `view.artifact(...)` junto al estado que realmente ha producido. Para artefactos persistidos, `artifact_path` contrasta sus bytes y detecta cambios posteriores antes de entregar entradas. El límite es 512 MiB por archivo. Un contrato escrito después no demuestra cómo se ajustó un modelo antiguo. Los productores siguen siendo responsables de registrar su procedencia real. Los padres y estados anteriores sin esta declaración no se aceptan por tener dimensiones iguales.

El consumidor recibe exclusivamente el diccionario de tensores. No recibe objetivos, fechas de maduración ni la disponibilidad agregada usada para auditar la admisión. Comprueba el marcador de vista y vuelve a aplicar sus máscaras antes de llamar al operador. Así, una modificación accidental de un tensor previamente reducido no restaura una señal excluida.

`attach_parent(batch, parent)` calcula la predicción después de aplicar la vista y construye el vector residual en el orden de `MODALITIES`, seguido de la predicción del padre. Rechaza resultados no finitos o desalineados. Una predicción o un vector de características añadido previamente al lote se rechaza como ruta indirecta sin contrato.

Estos adaptadores no implementan nuevos algoritmos de memoria o HMM ni cambian los de la campaña. Sus productores pueden conservar algoritmos distintos, pero necesitan acreditar la misma vista de información antes de componerlos.

## Recuperación y límites

El cursor de `ViewDataset` contiene la identidad de vista y el cursor confirmado del lector original. Cambiar la vista, la cohorte o la edición invalida la recuperación. Un cursor interior ausente se rechaza, por lo que no puede provocar un reinicio silencioso. La alineación de identificadores, etiquetas y fechas se comprueba antes de entregar un lote.

Cada lote admite hasta 4.096 filas y 64 MiB de tensores de entrada. La descripción permite hasta 16.384 componentes por muestra y 4.096 variables. El manifiesto leído desde disco se limita a 2 MiB. La operación conserva dimensiones y crea una copia de las entradas, por lo que su pico de memoria incluye también el lote original y los temporales del consumidor. No se presenta el límite de entrada como límite del RSS total.

## Ejemplo ejecutable

```bash
PYTHONPATH=src uv run --no-sync python scripts/example_information_views.py \
  --output /tmp/information-control --assets 8 --rows 64 --repeats 5
```

El ejemplo crea un corpus sintético en ese directorio y lo lee mediante `CorpusDataset`. Escribe `full-view.json`, `reduced-view.json`, `consumer-contracts.json` y `comparison.json`. Retira noticias, precios y `macro/us_cpi`, comprueba sus rutas derivadas y conserva la población completa.

Los seis consumidores ejecutan una suma de control de las señales retiradas, que debe ser cero. No son seis modelos entrenados ni resultados de un HMM real. El recorrido incluye el vector residual del padre y el rechazo de un contrato de la vista completa. Las fechas, variables y objetivos sintéticos no se presentan como evidencia histórica. Los manifiestos de corpus reales se utilizan con las mismas APIs `corpus_view`, `ViewDataset`, `ViewConsumer` y `attach_parent`.

## Pruebas y medidas

El 5 de octubre de 2026 pasaron 117 pruebas CPU relacionadas con vistas, documentos, cachés, materialización y lectores de corpus. La integración comprueba la sustitución macro efectiva en entrenamiento, validación, calibración y evaluación. Las pruebas cubren conservación de filas y etiquetas, recuperación, perturbaciones de entradas retiradas, dependencias indirectas y artefactos corruptos o incompatibles, incluida una corrupción antes de publicar el índice.

La revisión posterior reprodujo y corrigió la aceptación de un catálogo distinto con los mismos identificadores. Las regresiones mantienen iguales los datos mientras cambian las dependencias, comprueban la vinculación con la admisión temporal y rechazan procedencia ausente o una implementación contable incompatible. No se modificaron los datos ni los manifiestos de las campañas existentes.

Después de esa corrección pasaron 88 pruebas relacionadas y dos mutaciones adicionales detectaron la retirada de las comprobaciones de procedencia macro y contable. La [evidencia](../../reports/resources/information-view-provenance-20261005.json) registra 77/83 sentencias y 30/36 ramas de `information_inputs.py`, con CRAP máximo 36,03. El ejemplo completo de 512 filas se volvió a ejecutar con la nueva declaración de procedencia.

Se detectaron seis mutaciones dirigidas: omitir la huella del vector documental, ampliar la ventana hasta el futuro, conservar máscaras y edades de una variable retirada, ignorar la vista de ajuste del padre, omitir la transformación posterior a macro y admitir un cursor interior nulo.

Coverage.py 7.16.2, con ramas activadas, dio un 88 % para `document_index.py`, un 85 % para `information_views.py` y un 90 % para `information_inputs.py`, combinando líneas y ramas. Los subprocesos de los ejemplos se verificaron funcionalmente, sin incluirlos en esa cobertura. La cobertura del archivo `embeddings.py` incluye el codificador CUDA que no se ejecutó en esta comprobación CPU, por lo que su 50 % no describe únicamente los cambios de caché.

Se calculó complejidad con Radon 6.0.1 y CRAP como `CC² × (1 − cobertura_de_sentencias)³ + CC`, usando las líneas de cada función o método y excluyendo agregados de clase. El mayor valor de estos módulos fue 39,01 en `DocumentIndex.batches`, seguido de 29,93 en `InformationView.apply`. Son diagnósticos de complejidad y partes no ejercitadas, no garantías de ausencia de defectos.

Las siguientes medidas usan un AMD Ryzen 9 8945HS, Python 3.12.14, NumPy 2.5.3 y PyArrow 25.0.1 en Linux x86_64. Cada iteración recorre el corpus, aplica la vista, comprueba los seis consumidores y construye el vector residual. Se utilizaron lotes de 32 filas, un calentamiento y cinco repeticiones. Había otras tareas locales en ejecución.

| Activos × filas | Filas totales | Bytes de entrada por pasada | Pasada p50, ms | Pasada p95, ms | Filas/s | Proceso completo, s | Pico del proceso, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 × 8 | 16 | 58.752 | 4,958 | 5,613 | 3.118,06 | 0,52 | 126,71 |
| 8 × 64 | 512 | 1.880.064 | 47,357 | 49,772 | 10.702,89 | 0,85 | 137,52 |
| 64 × 128 | 8.192 | 30.081.024 | 677,067 | 683,563 | 12.090,08 | 5,16 | 141,01 |

El proceso completo y su pico de RSS se midieron con `/usr/bin/time` e incluyen preparación e intérprete. `comparison.json` conserva tiempos individuales, formas implícitas en los manifiestos y versiones. No se utilizó GPU ni se midieron energía o coste económico. Estas cifras describen el control técnico y no una ventaja sobre modelos reajustados para entradas reducidas.
