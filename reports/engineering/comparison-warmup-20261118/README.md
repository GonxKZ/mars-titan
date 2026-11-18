# Coste del calentamiento de validación en la RTX 4070

La comparación usa el checkpoint seleccionado de un lector terminado, su padre congelado y las filas reales de validación. Cada pasada comprueba la igualdad de la tabla completa con el Parquet emitido por la versión original. También compara todas las métricas y contadores con la primera pasada. El ensayo utiliza FP32 y no ejecuta pasos de optimizador.

En el origen 2022, semilla 43, el historial anterior a la primera decisión comprende 252 instantes. M1 reutilizó 63,95 MiB de tensores. La primera pasada de referencia tardó 53,69 s. La primera con captura tardó 47,90 s. Las dos siguientes repeticiones de referencia tardaron 47,32 y 46,51 s, frente a 28,67 y 28,19 s al restaurar el estado. Esto supone 18,48 s menos por validación (39,4 %) y un caudal medio de unas 4.506 filas puntuadas por segundo frente a 2.731. Las seis pasadas coincidieron en las 128.118 filas, incluidos los cuantiles. [Medidas de M1](m1.json)

M2 se midió con la revisión científica `e23a44034c8584682397d3d5f7ac76dbc7bd8e12` en la misma ventana y semilla. Las dos referencias en caliente tardaron 50,96 y 49,86 s, y las dos restauraciones 31,70 s cada una. Son 18,71 s menos por validación (37,1 %). Las seis pasadas reprodujeron las 128.118 filas propias de M2 y todos los contadores. La preparación tardó 32,47 s y el ensayo entero 302,90 s. La prueba anterior de M1 se hizo antes de incorporar la recuperación de las copias sin memoria. Las pruebas CUDA de M0, M1, M2 y M3 pasaron en la revisión publicada. [Medidas de M2](m2.json)

El tiempo de cada pasada incluye lectura, validación de los archivos, inferencia y recogida de predicciones hasta la sincronización CUDA. La comparación con el Parquet se realiza después de cronometrar cada pasada. Los dos lectores viven en el mismo proceso. A partir de la primera captura el consumo del proceso incluye esa copia incluso al medir la referencia. Los máximos CUDA del JSON son del asignador PyTorch y el RSS es el máximo acumulado del proceso. La energía no se midió.

Las pruebas cubren M0 a M3, un cambio posterior de los pesos del lector, una parada, un padre modificado y fallos de memoria durante la captura y la restauración. Se detectaron las diez mutaciones dirigidas. Las pruebas CPU específicas cubrieron 78 de las 82 sentencias y 21 de las 26 ramas del módulo de calentamiento. El recibo indica la cobertura y la complejidad por función de la caché, la herencia y el análisis de los ZIP. [Cobertura, CRAP y mutaciones](quality.json)

En la primera migración de PC1, M3 rechazó el enlace intermedio al directorio de su padre. Los tres intentos fallaron antes del primer paso de ajuste y quedaron registrados. Se copiaron los cinco padres heredados a directorios regulares y se cotejaron todos sus archivos con el origen. La herramienta de administración incluye esta copia al preparar un estado para la revisión fijada. Una migración técnica independiente volvió a abrir el padre real de 2023 con el verificador ordinario del lector y comprobó la huella del checkpoint seleccionado. [Comprobación del acceso al padre](parent-access.json)

Con cinco reutilizaciones como las de M1, la diferencia estimada es de 92 s por ajuste, alrededor del 3 % de los 51,09 minutos que tardó este lector en la campaña original. La estimación supone que la fase de ajuste mantiene su duración. El ensayo mide la validación, con una sola GPU y una semilla, y no mide el ahorro de toda la campaña. El ajuste del padre Titans-MAC cambia sus pesos entre épocas y requiere recalcular su memoria.

La vista de 2023 contiene 3.339.253 observaciones de ajuste. Por fila hay 64 × 6 valores de precios, 512 de gráficos, 78 de fundamentales, 420 de macroeconomía y 384 de noticias. Materializar solo esos valores en FP32 ocuparía 22,12 GiB, antes de reservar espacio para los estados, los gradientes y PyTorch. El recorrido por bloques permite procesarlos en la RTX 4070 de 8 GB.

El ASUS Ascent GX10 de 64 GB dispone de memoria unificada para la CPU Arm y la GPU. La memoria disponible para un proceso depende de ambas. El entorno de esta campaña fija Linux x86-64 y el enlace episódico compilado también es x86-64. Para medir el GX10 habría que resolver las dependencias para Arm, compilar el enlace y repetir la prueba de paridad. [Especificaciones del GX10 de 64 GB](https://press.asus.com/news/press-releases/asus-ascent-gx10-64gb-ai-supercomputer/)

La [prueba reproducible](../../../benchmarks/validation_warmup.py) necesita un estado de campaña que conserve el lector y su padre. Estas órdenes miden M1 en el entorno del equipo.

```bash
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MARS_TITAN_DECODE_WORKERS=2 MARS_TITAN_PREFETCH_BATCHES=2 MARS_TITAN_GROUP_CACHE_MIB=1536
export MARS_TITAN_EPISODIC_NATIVE="$(uv run --no-sync python -c 'from mars_titan.training.comparison_campaign import Layout; print(Layout().native)')"
uv run --no-sync python benchmarks/validation_warmup.py \
  --state "$HOME/.local/state/mars-titan/comparison-campaign-gpu-v1" \
  --model mars_titan_m1 --year 2022 --seed 43 --output "$HOME/medida-calentamiento-m1.json"
```
