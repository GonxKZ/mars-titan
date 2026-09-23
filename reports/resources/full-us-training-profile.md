# Perfilado del corpus estadounidense completo

Los cuatro modelos caben con lotes de 512 en la GPU disponible. En la sonda,
la lectura y preparación en CPU representan entre el 57 % y el 85 % del tiempo
por lote. Esa parte del recorrido merece una medición detallada antes de añadir
kernels propios. No se ha medido una aceleración frente a otra implementación.

## Población confirmada

La cohorte `original_audited` recorre los 4.784 candidatos estadounidenses.
Se prepararon 2.639 instrumentos y 2.145 carecen de alguna fuente original.
La intersección temporal deja 1.816.369 muestras de 2.226 activos. La auditoría
original y la verificación editorial externa son cohortes distintas, no dos
nombres para el mismo nivel de evidencia.

La supervisión admite 1.548.307 filas de entrenamiento de 2.055 activos y
261.069 filas de validación de 2.168 activos. Hay 1.999 activos comunes y
2.224 en la unión de ambas particiones. NEWTI y NRUC tienen muestras, pero
ninguna etiqueta admitida. No se fuerza su inclusión inventando historial.

Las 6.993 exclusiones se reparten en 5.777 casos sin historial suficiente,
811 etiquetas que cruzan una frontera temporal y 405 sin siguiente sesión.
Los recuentos se reconcilian con los recibos por activo. La preparación de
etiquetas duró 405,853 segundos, con un pico RSS de proceso de 289.173.504 bytes.
La reserva desde 2024 permanece cerrada. China y el brazo conjunto siguen
pendientes de procedencia contable admisible.

## Carga medida

Cada modelo usa 128 unidades, dos capas y dropout 0,2 en la fusión. Se conservan
las cuatro modalidades y macro, con precios de 64 × 5, texto de 384 componentes,
gráficos de 512, fundamentales de 45 y macro de 420. El paso utiliza AdamW,
tasa 0,001 y MSE, con la política numérica de las referencias.

Se ejecutaron tres repeticiones de 24 lotes por combinación, después de cinco
actualizaciones de calentamiento. La tabla presenta la mediana del caudal y
de la fracción de lectura, además del mayor pico asignado por PyTorch.

| Modelo | Lote | Muestras/s | Lectura y preparación | Pico asignado, MiB |
| --- | ---: | ---: | ---: | ---: |
| RNN | 256 | 14.354 | 66,9 % | 232,8 |
| RNN | 512 | 15.155 | 72,9 % | 396,6 |
| LSTM | 256 | 10.959 | 54,5 % | 356,9 |
| LSTM | 512 | 11.950 | 57,2 % | 641,4 |
| GRU | 256 | 12.025 | 58,5 % | 354,3 |
| GRU | 512 | 12.751 | 61,6 % | 638,1 |
| DLinear | 256 | 16.082 | 76,6 % | 72,0 |
| DLinear | 512 | 17.754 | 84,9 % | 76,8 |

Los lotes de 256 recorren 6.144 muestras de siete activos por repetición.
Los de 512 recorren 12.288 muestras de catorce activos. El origen es el corpus
completo, pero esta medición no recorre toda una época ni demuestra precisión.
La selección de tamaño de lote se basa en recursos, no en resultados de validación.

La comprobación del origen y la inicialización duraron 15,815 segundos.
El bloque de 24 casos completo tardó 37,261 segundos. Otra aplicación ocupaba
5.620 MiB de VRAM. El pico asignado no incluye toda la memoria del contexto CUDA,
el controlador ni esa aplicación. Los tiempos tampoco representan una GPU
exclusiva. La caché del sistema operativo no se ha vaciado ni controlado.

Extrapolar únicamente los pasos medidos con lote 512 da entre 87 y 130 segundos
de actualizaciones por época. No es una estimación completa del entrenamiento.
Faltan validación, checkpoints, comprobaciones iniciales y diferencias de coste
entre activos. La duración observada de las primeras épocas completas permitirá
sustituir esa proyección parcial.

## Reproducción y comprobación

Las mediciones por repetición, latencias, bytes lógicos de transferencia, versiones
y huellas están en [el recibo del perfilado](full-us-training-profile.json).
Los bytes lógicos no son contadores físicos PCIe. No se estima energía a partir
de una lectura aislada de potencia.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python \
  reports/analysis/benchmark_full_corpus.py \
  --manifest data/processed/original-audited-us-supervision-20260923/manifest.json \
  --output data/interim/full-us-profile-repeat
```

La salida debe ser nueva. Una sonda interrumpida conserva sus casos parciales,
pero se repite desde el principio para no mezclar condiciones de medición.
Los recibos final y de progreso confirman el mismo estado al terminar.

Pasan 44 pruebas del paso de medición, lector y entrenador de referencia.
Incluyen cuatro nuevas para importación sin ejecución accidental, paso CUDA,
rechazo de un origen incompleto y cierre coherente de los recibos. La revisión
independiente detectó el estado de progreso que quedaba abierto. Se reprodujo
y corrigió antes de ejecutar la sonda real. El núcleo científico no cambia en
esta entrega.
