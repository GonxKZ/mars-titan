# Referencias neuronales: núcleos, precisión y caudal

Medidas del 9 y 10 de octubre de 2026 sobre la rama `perf/campaign-kernels`. Recibo completo en `references.json`. No se ha entrenado ningún modelo ni se ha llamado al optimizador de ninguno: los pasos miden forward, pérdida pinball y backward, y los gradientes se liberan sin tocar pesos.

## Alcance y método

- Casos: los candidatos `-00` y `-10` de la campaña A para RNN, LSTM, GRU, DLinear y Transformer compacto, y `transformer-11` (dos capas, anchura 128) del diseño de búsqueda para cubrir el camino de dos capas.
- Datos: 32 lotes reales de 512 de la partición de ajuste de la vista `US+CN/fold-012` de la edición v3 (semilla 42), con las formas de la campaña: precios 64 × 5, noticias 384, gráficos 512, fundamentales 78, macro 420 y presencia 5. Se materializaron una vez sin GPU y se dividen en mitades para medir el lote 256. La campaña A declara 256. El lote 512 se mide solo como información para el rediseño conjunto.
- Precisión: FP32 estricto, como `kernel_policy.apply_kernel_policy` (sin TF32 en cuBLAS ni en cuDNN y `float32_matmul_precision` en `highest`).
- Versiones: antes es `develop` en 14c6b1dd. Después es `perf/campaign-kernels` en 5b93f28e, antes de rebasar sobre el develop actual, con el camino de último token (76a4f47f en la rama rebasada) y la reutilización del tensor de niveles de pinball (ae823814, de otro trabajo de la rama).
- Procedimiento: procesos alternados antes y después sobre el mismo archivo de lotes, cada uno en una plaza corta de GPU. Hay 4 repeticiones por lote en orden ABBA, cada una con 3 lotes de calentamiento y 3 pasadas. Se da la mediana. La carga media de la CPU estuvo entre 13 y 20 por otros trabajos, y los modelos pequeños dependen del lanzamiento de núcleos desde Python, así que su dispersión entre procesos llega al 20 %.
- Equipo: RTX 4070 Laptop de 8 GB en modo power-saver, PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24, Python 3.12.

## Cambios

1. `CompactPriceTransformer` calcula completas todas las capas salvo la última. En la última solo proyecta claves y valores de toda la ventana, y calcula la consulta, la atención, la FFN y las normalizaciones para el último token. Con la máscara causal el último token ve toda la ventana, así que coincide con `encode_sequence(...)[:, -1]` en aritmética exacta. El `state_dict` y la configuración por defecto no cambian.
2. Las capas se evalúan con sus propios pesos y `scaled_dot_product_attention`, sin llamar a `TransformerEncoderLayer.forward`. En evaluación sin gradiente esa llamada toma en CUDA la ruta nativa de PyTorch, que aproxima GELU con tanh: con los mismos pesos, develop daba en evaluación salidas que diferían de las de entrenamiento hasta en 7,5e-6 en FP32 y en torno a 2e-4 en FP64. Ahora entrenamiento y evaluación coinciden bit a bit en CUDA y el resultado no depende del indicador global de fastpath. Por eso `test_financial_control` espera ahora cero llamadas a la ruta nativa. La máscara causal se conserva como buffer porque la huella de parámetros de Titans-MAC recorre también los buffers no persistentes.
3. Las comprobaciones de finitud de la referencia Transformer se agrupan en una sola copia al host con los mismos mensajes y el mismo orden. Las sincronizaciones por forward bajan de 8 a 1, y las del paso del bucle de `reference_run` hasta el optimizador bajan de 20 a 12 (una de ellas por ae823814).
4. `max_batch` opcional en el codificador y en `MultimodalReference` (de 1 a 8192). El presupuesto de atención conserva los elementos por ventana del valor por defecto. `transformer_batch_options` solo añade la opción por encima de 256 ventanas, de modo que los checkpoints con lotes de hasta 256 conservan su contrato. `reference_run` y `carried_predictions` la aplican con el lote de la identidad.
5. Precisión declarada por caso en `reference_run`: el campo opcional `case["precision"]` aplica la política antes de construir el modelo y de registrar los valores numéricos, añade `kernel_policy` a la identidad, se vuelve a exigir en cada guardado y debe coincidir en las continuaciones. Sin el campo la identidad no cambia. `carried_predictions` aplica la política del ancla antes de comparar su entorno.

## Validación numérica

| Comprobación | Resultado | Tolerancia |
|---|---|---|
| Último token frente a la secuencia completa, FP64, 1 y 2 capas, 1, 4 y 8 cabezas, entrenamiento y evaluación | Pasa en salidas y gradientes | rtol 1e-12 y atol 1e-13 en salidas, rtol 1e-10 y atol 1e-12 en gradientes |
| Capas propias frente a `nn.TransformerEncoderLayer` en su ruta estándar, FP64 | Pasa | La misma |
| Error FP32 frente a FP64 del último token y del camino completo, CPU y CUDA | El del último token no supera 4 veces el del camino completo más 2e-7 | Declarada en la prueba |
| Antes y después con los mismos pesos en CUDA: RNN, LSTM, GRU y DLinear | Idénticas bit a bit en entrenamiento y evaluación | Exacta |
| Antes y después: Transformer en entrenamiento | Diferencia máxima de 2,4e-7 | Redondeo FP32 |
| Después: Transformer en evaluación frente a entrenamiento | Idénticas | Exacta |

Las mutaciones dirigidas sobre una copia privada del árbol se detectaron todas: 18 de 18 en el codificador (normas, residuos, consulta, claves, causalidad, orden QKV, cabezas, comprobaciones y presupuesto), 7 de 7 en la precisión de `reference_run` y 3 de 3 en `carried_predictions`.

## Caudal antes y después

Filas por segundo en miles. «Cómputo» usa lotes residentes en la GPU. «Con copias» copia cada lote desde la memoria del proceso como el ajuste. «Serie» combina en serie el lector medido (21,8 mil filas por segundo) con «con copias». Esa estimación representa al bucle actual, que espera a la GPU en cada paso. El pico es la memoria asignada del proceso de medida e incluye 107,5 MiB de lotes residentes.

Lote 256, el declarado:

| Caso | Cómputo antes | Cómputo después | Evaluación antes | Evaluación después | Con copias antes | Con copias después | Serie antes | Serie después | Pico MiB antes | Pico MiB después |
|---|---|---|---|---|---|---|---|---|---|---|
| rnn-00 | 89,7 | 101,0 | 471,5 | 535,5 | 70,0 | 90,8 | 16,6 | 17,5 | 215 | 215 |
| rnn-10 | 84,4 | 99,7 | 499,2 | 475,6 | 71,0 | 73,1 | 16,7 | 16,8 | 251 | 251 |
| lstm-00 | 92,7 | 85,0 | 491,7 | 480,4 | 78,7 | 73,2 | 17,0 | 16,8 | 239 | 239 |
| lstm-10 | 72,4 | 87,2 | 339,5 | 338,9 | 58,8 | 57,1 | 15,9 | 15,8 | 298 | 298 |
| gru-00 | 89,9 | 89,6 | 442,9 | 393,5 | 73,9 | 68,7 | 16,8 | 16,5 | 235 | 235 |
| gru-10 | 71,6 | 83,6 | 393,6 | 367,4 | 60,9 | 55,6 | 16,0 | 15,6 | 291 | 291 |
| dlinear-00 | 110,0 | 106,6 | 504,3 | 487,7 | 94,1 | 84,6 | 17,7 | 17,3 | 173 | 173 |
| dlinear-10 | 96,8 | 101,5 | 571,5 | 509,8 | 95,3 | 87,1 | 17,7 | 17,4 | 173 | 173 |
| transformer-00 | 53,8 | 59,4 | 125,9 | 210,8 | 46,7 | 51,4 | 14,8 | 15,3 | 212 | 200 |
| transformer-10 | 44,0 | 60,8 | 105,9 | 217,3 | 39,9 | 50,2 | 14,1 | 15,2 | 241 | 213 |
| transformer-11 | 16,6 | 34,1 | 47,7 | 125,6 | 16,0 | 30,2 | 9,2 | 12,6 | 424 | 340 |

Lote 512, solo como información:

| Caso | Cómputo antes | Cómputo después | Evaluación antes | Evaluación después | Con copias antes | Con copias después | Pico MiB antes | Pico MiB después |
|---|---|---|---|---|---|---|---|---|
| transformer-00 | 72,3 | 107,6 | 160,4 | 357,5 | 64,5 | 86,0 | 253 | 229 |
| transformer-10 | 49,9 | 101,8 | 125,2 | 405,2 | 46,0 | 81,1 | 307 | 254 |
| transformer-11 | 15,9 | 33,6 | 47,7 | 115,9 | 15,5 | 32,0 | 669 | 498 |

Con 512 las demás familias quedan entre 90 y 190 mil filas por segundo de cómputo y sin cambio de memoria. Sus valores están en el recibo.

Lectura de las tablas:

- El Transformer mejora de forma consistente: entre 1,1 y 2,1 veces en ajuste y entre 1,7 y 2,6 veces en evaluación con lote 256, y hasta 2,1 veces y 3,2 veces con 512. Su memoria baja entre un 5 % y un 26 %. Los rangos de antes y después no se solapan.
- RNN, LSTM, GRU y DLinear no cambian de código en este trabajo. Sus diferencias en ambos sentidos están dentro de la dispersión, salvo los casos `-10` de LSTM y GRU, que ganan entre un 17 % y un 27 % en cómputo con rangos separados. La explicación más probable es la copia al host que se retiró del pinball en ae823814, que dejaba de solapar pasos consecutivos, pero no se ha aislado con una medida propia.
- El lector es el cuello de botella. Un proceso con 2 hilos entrega 21,8 mil filas por segundo y, tras los cambios, ninguna familia baja de 30 mil con copias. Mientras el bucle espere a la GPU en cada paso, el ajuste queda entre 12,6 y 17,5 mil filas por segundo con lote 256, unos 14 a 19 minutos por época de los 14,6 millones de muestras de ajuste de esta ventana. Con lectura solapada el límite sería el propio lector.
- La construcción de `CorpusDataset` calcula el SHA-256 de cada artefacto de la vista y ocupó casi todos los 9 min 45 s de la fase de lectura con el disco compartido. Es un coste fijo por proceso y no entra en el caudal del lector.

## Precisión frente a FP64

Errores de 8 lotes de 256 frente a una copia FP64 con los mismos pesos y sin dropout: salida máxima absoluta, gradiente relativo máximo por lote y gradiente acumulado de los 8 lotes, sin optimizador.

| Caso | FP32 estricto | FP32 con TF32 en cuDNN | TF32 | BF16 autocast |
|---|---|---|---|---|
| rnn-10 | 3,6e-7 / 2,1e-7 / 1,5e-7 | Igual que estricto | 1,7e-4 / 1,9e-4 / 1,7e-4 | 1,8e-3 / 8,4e-3 / 3,3e-3 |
| lstm-10 | 2,5e-7 / 1,3e-7 / 1,0e-7 | Igual que estricto | 2,0e-4 / 2,3e-4 / 2,0e-4 | 2,7e-3 / 1,8e-2 / 1,4e-2 |
| gru-10 | 2,4e-7 / 1,5e-7 / 1,3e-7 | Igual que estricto | 2,4e-4 / 2,3e-4 / 2,1e-4 | 3,9e-3 / 3,5e-3 / 2,8e-3 |
| dlinear-10 | 2,1e-7 / 3,4e-7 / 1,4e-7 | Igual que estricto | 1,5e-4 / 2,8e-4 / 2,4e-4 | 1,7e-3 / 2,6e-2 / 5,6e-3 |
| transformer-10 | 2,3e-7 / 2,3e-7 / 2,0e-7 | Igual que estricto | 1,9e-4 / 2,1e-4 / 1,7e-4 | 2,5e-3 / 1,2e-2 / 5,0e-3 |
| transformer-11 | 2,0e-7 / 2,6e-7 / 2,0e-7 | Igual que estricto | 1,1e-4 / 2,6e-4 / 2,4e-4 | 1,1e-3 / 3,9e-3 / 3,3e-3 |

Los casos `-00` siguen el mismo patrón y están en el recibo. TF32 multiplica el error por unas mil y BF16 por diez mil o más, sin una ganancia de tiempo que se sostenga entre repeticiones. Ambos quedan medidos y descartados, de acuerdo con la política FP32 estricta. Activar `cudnn.allow_tf32`, el valor por defecto de PyTorch, no cambia ningún error en estas formas, así que la política estricta no tiene un coste medible en las recurrentes.

## Núcleos

- cuDNN: RNN, LSTM y GRU pasan por `aten::_cudnn_rnn` y `aten::_cudnn_rnn_backward` con los núcleos persistentes `RNN_blockPersist_*` en FP32. Los pesos forman un único bloque contiguo y no aparece el aviso de compactación en el modelo medido. Sí aparece en copias hechas con `copy.deepcopy`, que solo usa el estudio numérico.
- SDPA: en FP32 el Transformer usa la atención eficiente en memoria (`fmha_cutlassF_f32` y `fmha_cutlassB_f32`). Flash no está disponible en FP32 y la ruta `math` es más lenta (de 5,3 a 7,8 ms en transformer-00 y de 7,6 a 9,8 ms en transformer-11). Se mantiene la selección automática.
- AdamW `fused` frente a `foreach`: descartado. Cambia la aritmética del optimizador y la campaña solo admite aceleraciones que la conserven. Además, medirlo exigiría pasos del optimizador, que el bloqueo de aprendizaje no permite ni sobre tensores sintéticos.
- CUDA Graphs, forward y backward sin optimizador con lote 256, en dos procesos medidos en momentos distintos: GRU pasa de 2,3 a 3,3 ms en eager a entre 1,5 y 1,7 ms, y DLinear de 2,0 a 3,1 ms a 0,27 ms, con salidas idénticas bit a bit (los gradientes no se compararon). En el Transformer la captura falla porque la comprobación de finitud copia al host dentro del grafo.
- `torch.compile` con lote fijo de 256, en los mismos dos procesos: el Transformer pasa de 4,2 a 4,6 ms en eager a entre 2,1 y 3,1 ms en modo `default` y entre 1,8 y 3,2 ms con `max-autotune-no-cudagraphs`, tras 6 a 12 s de compilación. DLinear baja a entre 1,4 y 1,6 ms y GRU no mejora de forma estable (de 2,2 a 2,8 ms). En el Transformer y en DLinear la fusión cambia el redondeo (2,4e-7 de diferencia en la salida), así que se descarta. El último lote parcial de cada época obligaría además a recompilar o a una ruta aparte.

## Memoria y concurrencia

Entradas para el empaquetador de la GPU: un proceso por trabajo con lote 256 en FP32 estricto, sin MPS, sobre la vista US+CN fold-012. `context_mib` es la memoria del proceso según nvidia-smi menos el pico reservado por PyTorch. La columna «sin lotes» descuenta los 107,5 MiB de lotes residentes de la medida, porque un ajuste real solo mantiene el lote en curso.

| Familia | Filas por segundo | Pico asignado MiB | Sin lotes MiB | Pico reservado MiB | Proceso MiB | Contexto MiB | 2 trabajos | 3 trabajos |
|---|---|---|---|---|---|---|---|---|
| rnn-10 | 99,6 mil | 251 | 143 | 288 | 504 | 216 | 163,2 mil | 162,5 mil |
| lstm-10 | 78,2 mil | 298 | 191 | 342 | 558 | 216 | 97,6 mil | 97,4 mil |
| gru-10 | 76,7 mil | 291 | 184 | 322 | 538 | 216 | 99,6 mil | 99,4 mil |
| dlinear-10 | 112,1 mil | 173 | 66 | 176 | 388 | 212 | 203,0 mil | 246,3 mil |
| transformer-10 | 53,2 mil | 213 | 105 | 234 | 446 | 212 | 99,9 mil | 123,8 mil |

Las cifras son agregados de cómputo con lotes residentes, sin lector. Una primera repetición con más carga de CPU dio el mismo patrón y está en el recibo. GRU y LSTM llenan la GPU con dos trabajos. RNN deja de ganar con el tercero. DLinear y el Transformer, limitados por lanzamientos, siguen ganando con tres. Ningún trabajo supera 560 MiB de proceso, así que la memoria no limita el empaquetado en 8 GB.

## Decisión por familia

| Familia | Se adopta | Medido y descartado o pendiente |
|---|---|---|
| RNN, LSTM, GRU | FP32 estricto con cuDNN persistente, ya en uso. La política no tiene coste medible. CUDA Graphs con `cuda_graphs: true` | TF32 y BF16 descartados. GRU y LSTM no ganan con más de dos trabajos por GPU |
| DLinear | FP32 estricto y CUDA Graphs con `cuda_graphs: true` | TF32 y BF16 descartados. Admite tres trabajos por GPU |
| Transformer compacto | Último token con capas explícitas, GELU exacta en todos los modos, finitud en una sincronización y `max_batch` explícito | TF32 y BF16 descartados. Flash no existe en FP32. `torch.compile` cambia el redondeo y se descarta. CUDA Graphs con `cuda_graphs: true`, con la finitud leída fuera del grafo |
| Común | Precisión declarada por caso y anclas trasladadas con su política y su lote | AdamW `fused` descartado. El lector es el límite actual |

## Pruebas ejecutadas

- CPU con `CUDA_VISIBLE_DEVICES=-1` y `memslot`: `tests/models`, `tests/training/test_reference_run.py`, `test_reference_run_precision.py` y `test_kernel_policy.py`, con 1058 pruebas superadas y 285 omitidas. Fallan 19 pruebas, todas con «CUDA no está disponible» porque exigen la GPU. Después, `test_carried_predictions.py` y `test_carried_predictions_precision.py` dan 15 superadas y `test_reference_run_precision.py` da 10.
- GPU con `memslot gpu`: `test_compact_transformer_last_token.py`, `test_compact_transformer.py`, `test_multimodal_reference.py` y `titans/test_financial_control.py`, con 174 superadas.
- Comandos de medida: `benchmarks/reference_kernels.py materialize|measure`, `reference_numerics.py`, `reference_concurrency.py` y `reference_graphs.py`, con los argumentos del recibo.

## Pendiente

- Lector: es el límite del ajuste de las referencias. La lectura solapada o con varios procesos corresponde al trabajo de la tubería de datos.
- CUDA Graphs: implementado después de estas medidas como opción `cuda_graphs` del caso (`training/reference_step_graph.py`). El último lote parcial usa la ruta eager, la finitud se lee fuera del grafo y el generador se restaura tras el calentamiento. Las cinco familias dan predicciones y gradientes idénticos bit a bit a la ruta eager, con un paso entre 1,4 y 3,0 veces más rápido. Ver [README.md](README.md) y [reference-step-graph.json](reference-step-graph.json).
- La sección neuronal de `campaign_plan` no admite todavía `precision` ni `cuda_graphs` y limita el lote a 256. `campaign_throughput._reference` y `reference_search` ya construyen el Transformer con el mismo contrato de lote que `reference_run` (21560868), y `posttraining/parents.py` reconstruye el padre con el lote de su identidad (83a9877f y c40e5b95).
- Las medidas usan la edición v3. La v3.1 añade un bit de presencia por sesión a los precios. El codificador acepta cualquier anchura de entrada, pero conviene repetir la medida, que cuesta pocos minutos con el archivo de lotes.
- No se ha medido la energía.
