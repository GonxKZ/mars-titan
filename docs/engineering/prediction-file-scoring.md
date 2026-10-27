# Puntuación de archivos por sesiones

[`score_prediction_file`](../../src/mars_titan/evaluation/forecast_files.py) ordena un archivo Parquet de desarrollo y calcula sus métricas con `ForecastPanel` y `SessionScores`. Reúne cada sesión completa, definida por mercado e instante, aunque cruce lotes Arrow o grupos Parquet. Primero puntúa sus activos y después permite promediar las sesiones con la ponderación declarada. No sustituye ese recorrido por un promedio global de filas.

El llamador declara el número de filas, las columnas de cada pronóstico y el mínimo de activos para Rank IC. Una columna nula en la declaración genera el predictor residual cero. Las igualdades entre columnas permiten comprobar que un control intacto conserva la emisión original. Los cuantiles solo se puntúan si la salida ya los contiene y se declaran sus columnas y niveles. El predictor cero no recibe cuantiles inventados.

El lector comprueba el pie Parquet, los tipos y el conteo de filas. Rechaza identificadores repetidos, ausencias, valores no finitos y decisiones desde el 1 de enero de 2024. Las operaciones estadísticas que desbordan FP64 producen un error. No se sustituyen valores ni se recortan predicciones. Una anchura de intervalo puede desbordar aunque sus extremos sean finitos, y ese caso tiene una prueba específica.

DuckDB usa dos hilos, 256 MiB de memoria y hasta 1 GiB de archivos temporales en el directorio indicado. La lectura admite lotes de hasta 8.192 filas, sesiones de hasta 8.192 activos, 50.000 sesiones y 32 condiciones. Conserva los estadísticos por sesión y agrupa su concatenación en bloques fijos de 128 sesiones. Estas cotas no son un certificado del pico de RAM de todo el proceso. El consumidor puede imponer límites menores de filas y condiciones.

La huella de población incluye identificadores, mercados, instantes y objetivos, sin las predicciones. Se conserva al cambiar el tamaño del lote. Dos archivos con igual número de filas y distintos objetivos producen huellas diferentes. Comprobar esta igualdad no demuestra la disponibilidad histórica de las entradas con las que se generaron los pronósticos.

El [análisis B4](../experiments/lesion-protocol-v1.md) y el [comparador E2](../experiments/spaced-utility-comparison-v1.md) reutilizan el lector y conservan su guarda de aprendizaje antes de leer fuentes o crear salidas. La API pura de puntuación no concede permiso para ejecutar una evaluación científica.

## Evidencia técnica del 27 de octubre de 2026

La [verificación](../../reports/engineering/forecast-files-20261027/verification.json) registra 97 casos CPU aprobados, incluidos veinte nuevos, 24 regresiones de B4, 36 de contrastes y diecisiete guardas. No hay omisiones en esta selección. La suite completa no se ha ejecutado. Las [seis mutaciones dirigidas](../../reports/engineering/forecast-files-20261027/mutations.json) se detectan con controles verdes antes y después, sin intentos de pasos de optimizador. Cubren acceso a 2024, presupuesto de sesiones, sustitución del pronóstico por cero, igualdad de la emisión, orden de cuantiles y conexión de las columnas de lesiones.

La [cobertura y el CRAP](../../reports/engineering/forecast-files-20261027/quality.json) usan coverage.py 7.16.1 y Radon 6.0.1. El lector recorre 112/112 sentencias y 24/26 ramas. Los dos módulos modificados reúnen 206/208 sentencias y 30/32 ramas, con CRAP máximo 23. CRAP usa cobertura de sentencias. La cifra no acredita todas las combinaciones de condiciones ni la ausencia de errores.

Las dos medidas comparan el lector anterior de `917c6fdad` con el común, en CPU FP64, con 32 sesiones y cinco condiciones. Cada tamaño e implementación tiene un calentamiento y tres muestras medidas. Los doce pares de cada pasada conservan exactamente las tablas por sesión, sus huellas y sus resúmenes. El tiempo incluye leer, ordenar y puntuar, sin crear los fixtures ni comparar las salidas. Los totales del proceso sí incluyen esas dos operaciones.

| Activos por sesión | Anterior, pasada 1 (ms) | Común, pasada 1 (ms) | Anterior, pasada 2 (ms) | Común, pasada 2 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 64 | 144,985 | 135,389 | 113,695 | 122,235 |
| 256 | 196,292 | 186,118 | 163,211 | 199,497 |
| 1.024 | 348,501 | 367,701 | 354,415 | 361,321 |

Las diferencias cambian entre pasadas. No sostienen una aceleración ni una regresión uniforme y no miden la latencia del predictor. Se conservan las muestras de la [primera pasada](../../reports/engineering/forecast-files-20261027/cost-first.json) y de la [segunda](../../reports/engineering/forecast-files-20261027/cost-confirmation.json), incluidos tiempos y picos RSS de proceso. No se midieron energía, VRAM ni transferencias. Los datos son fixtures manuales y no hay una mejora predictiva medida.

El [script reproducible](../../reports/engineering/forecast-files-20261027/cost_probe.py) necesita un destino nuevo. Sus literales largos se dividieron después de las medidas. Los parches de [la primera fuente](../../reports/engineering/forecast-files-20261027/cost-probe-first.patch) y [la segunda](../../reports/engineering/forecast-files-20261027/cost-probe-confirmation.patch), aplicados sobre una copia del script actual, recuperan exactamente los bytes ejecutados. Se han comprobado sus hashes y la igualdad del AST sin posiciones. No cambian los cálculos ni el protocolo.

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/forecast-files-20261027/cost_probe.py "$HOME/.cache/mars-titan/forecast-files-cost"
```

Estas comprobaciones no utilizan un modelo entrenado ni abren una evaluación real. Las rutas CUDA de los modelos requieren su propia evidencia y continúan pendientes.
