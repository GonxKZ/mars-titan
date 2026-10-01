# Revisión del cálculo y de las mejoras medidas

La revisión del 24 de septiembre de 2026 conserva cambios en construcción de lotes, lectura Parquet, evaluación y simulación C++20. Cada comparación identifica sus datos, versión y referencia. Los resultados CPU no se presentan como aceleración del entrenamiento CUDA.

| Recorrido existente | Operaciones y primitivas | Evidencia y decisión |
| --- | --- | --- |
| Corpus de entrenamiento | Ventanas vectorizadas de precios, validación temporal, selección de filas y construcción de arrays NumPy. | [Lotes directos](corpus-batches.md): se eliminan copias por muestra y `np.stack`. El lector denso de 16.384 filas y lote 512 pasa de 473,55 a 287,14 ms, con paridad exacta. |
| Episodios y preparación | Factores, regímenes, RNG por semilla y transformaciones NumPy. Persistencia y decodificación mediante Arrow y Parquet C++. | [Lectura de bloques](../../docs/engineering/evaluation-io-performance.md): un lector activo y dos grupos acotados. En 128 activos, el recorrido con simulación pasa de 778,87 a 720,61 ms. Se mantienen firmas, máscaras y cierre explícito. |
| Evaluación y métricas | Inferencia por lotes, reducciones de error y agregación por mercado y sesión. | La misma [comparación de evaluación](../../docs/engineering/evaluation-io-performance.md) elimina tablas Arrow descartadas. En el diagnóstico CPU de 65.536 filas, 60,52 a 32,48 ms. Las predicciones persistidas conservan su SHA-256. |
| Acumulación por sesión | Agrupación NumPy para lotes mayores y reducción secuencial hasta 128 filas. Combinación atómica con el estado anterior. | La [iteración posterior](../../docs/engineering/session-metrics-performance.md) mide 16.384 filas con lote 512: 8,23 a 2,71 ms en cohortes y 30,39 a 22,96 ms con fechas únicas. Conserva exactamente sumas, orden y predicciones, incluso al repetir claves entre lotes. |
| Cartera persistente | Contabilidad FP64, sumas por parciales, ordenación de candidatos y ejecución de escenarios independientes. | [Ejecutable autónomo](native-go-no-go.md) con una cinta compartida, buffers propios y hasta ocho trabajadores medidos. La [serialización aplazada](native-checkpoints.md) reduce el caso de 512 activos de 539,29 a 430,45 ms con un trabajador. |
| Referencias neuronales | `nn.Linear`, RNN, LSTM, GRU, pooling de DLinear, activaciones, pérdidas de PyTorch, backward y AdamW. | Se conservan las primitivas compiladas del framework y su aritmética. El [perfil previo del corpus](full-us-training-profile.md) y la [medida posterior de ventanas](batched-price-contexts.json) motivaron reducir preparación. No se ha medido aquí una nueva configuración de precisión, fusión u optimizador CUDA. |
| Ridge y boosting | Acumulación de Gram con `addmm_` y `addmv_`, `torch.linalg.solve` y backend nativo de XGBoost. | Se reutilizan esas implementaciones. No se ha demostrado una ventaja de sustituirlas por álgebra lineal o árboles propios. |
| Ajuste predictivo | Rejilla de 21 acciones, REINFORCE, pérdida esperada, MAE y variantes KLPO con operaciones de PyTorch. | Se mantienen referencia y presupuesto. Las [continuaciones de edición 2](../../docs/engineering/continuation-selection.md) seleccionan desde la época 0. La parada independiente se limita al control real para conservar el presupuesto entre los aumentos. |
| PPO y Double DQN | Redes de dos capas de 64 unidades, replay contiguo, GAE de horizonte acotado y estados recuperables. | Existen comprobaciones CPU de lógica y recuperación. No se ha medido una nueva ejecución CUDA de rollouts e inferencia. No se modifica su recompensa para favorecer una medición de rendimiento. |
| Representaciones y gráficos | Encoders congelados, transformaciones de imágenes, tokenización y fórmulas de fundamentales y macro. | Sus implementaciones de biblioteca y fechas de disponibilidad se conservan. No se sustituyen por representaciones sintéticas ni por kernels sin medidas del recorrido real. |
| Etiquetas residuales | Regresiones temporales con arrays NumPy, cortes y máscaras causales. | La [optimización anterior](residual-preparation.md) ya comparó la referencia pandas y la variante NumPy. No se cambia el algoritmo OLS por una recurrencia con otro error de acumulación sin una evaluación específica. |

Los perfiles posteriores a estas mejoras siguen distinguiendo trabajo de decodificación, validación, acumulación y persistencia. La inspección del ensamblado nativo confirma el uso de operaciones vectoriales del compilador y de las bibliotecas en algunos caminos, sin relajar las sumas contables ni afirmar que todos los bucles estén vectorizados. Los contadores de hardware de `perf` no son accesibles con `perf_event_paranoid=4`. Callgrind y los perfiles CPU aportan evidencia separada, sin inventar GFLOP/s, tráfico de memoria ni límites Roofline.

El tiempo de `read_row_group` incluye lectura, decodificación y construcción de arrays. No demuestra por sí solo que la descompresión domine el recorrido. No se duplicó el corpus en una caché Arrow IPC sin compresión a partir de esa atribución incompleta. Tampoco se sustituyeron las reducciones precisas por un orden aritmético distinto para obtener vectorización.

## Contraste de LTO

Se construyó otra versión con `MARS_TITAN_ENABLE_IPO=ON`, que Clang 21.1.8 aplica mediante ThinLTO. Pasa cuatro CTests y conserva métricas, snapshots y eventos en las 60 ejecuciones del [benchmark](native-lto-benchmark.json). La compilación Release habitual quedó intacta.

En 128 activos, LTO pasa de 310,28 a 320,58 ms con un trabajador y de 103,92 a 109,66 ms con ocho. En 512 activos pasa de 450,31 a 433,76 ms y de 159,16 a 158,71 ms. El resultado no justifica activar LTO por defecto en la carga comparada. La opción explícita sigue disponible. PGO y las instrucciones específicas de otra arquitectura no se presentan como mejoras demostradas.

Desde `native/`, la reproducción utiliza el preset Release existente en otro directorio:

```bash
cmake --preset native-release -B ../build/native/native-release-lto \
  -DMARS_TITAN_ENABLE_IPO=ON
cmake --build ../build/native/native-release-lto --parallel 2
```

Desde la raíz del repositorio se comparan ambos binarios mediante [native_checkpoints.py](../../benchmarks/native_checkpoints.py), pasando las rutas en `--reference` y `--candidate` y una carpeta nueva en `--work`.

## Compatibilidad y límites de ejecución

Las fuentes y el checkout de la campaña activa permanecen intactos. Cambiar la construcción del lector cambia su huella de código. La carga de padres conserva su comprobación de identidad y no acepta silenciosamente una implementación distinta. La revisión `4a5bc69` contiene la selección de continuaciones desde época 0 con el contrato del lector anterior, útil para los padres de esa edición. Los padres y los checkpoints deben utilizar su revisión compatible. Una comprobación técnica de paridad no reetiqueta artefactos científicos anteriores.

Se comprobó CUDA disponible con PyTorch `2.14.0+cu130`. La [comprobación explícita de admisión](cuda-admission-20260924.json) rechazó iniciar otra carga al detectar cómputo CUDA activo. No se inició otra carga científica simultánea. Por ello permanecen sin medir una nueva precisión mixta, Tensor Cores, fusión de optimizador, CUDA Graphs, transferencias y solapamiento de rollouts. No se ofrece una alternativa CPU silenciosa al entrenamiento. Tampoco se detuvieron otras aplicaciones ni se modificaron permisos, ASLR o drivers.

El despliegue utiliza una GPU CUDA. No se añadió infraestructura distribuida ni colectivas propias. No hay una comparación ejecutada de energía o coste monetario. Los candidatos basados en FFT, matrices dispersas, RLHF, RLAIF o MoE no se incorporan a recorridos que no los necesitan. MARS-TITAN continúa como diseño futuro y el test final permanece sellado.

Las comprobaciones conjuntas de las entregas integradas pasaron 482 pruebas CPU de los módulos afectados y sus integraciones. Incluyen el contrato de publicación, episodios, corpus, métricas, selección y simulación. Los informes específicos conservan sanitizadores, cobertura, complejidad, mutaciones y límites. Esta batería no sustituye las pruebas de entrenamiento CUDA ni demuestra que todo el sistema haya alcanzado el límite del hardware.
