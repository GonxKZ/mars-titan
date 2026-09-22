# Campañas recuperables de referencias

El ejecutor utiliza las etiquetas ya preparadas por activo. Cada época recorre
todo el tramo de ajuste de la edición y conserva el último lote parcial. La
validación usa únicamente 2023 y no modifica pesos ni consulta el test final.

La rejilla mantiene RNN, LSTM, GRU y la adaptación multimodal de DLinear. Cada
familia tiene tres semillas, tres pérdidas y dos tasas de aprendizaje. Se ejecutan
30 épocas base y cinco de continuación. Las continuaciones MAE y MSE parten
del control MSE con tasa 0,001 de la misma semilla y población. Crean un AdamW
nuevo y no eligen retrospectivamente el mejor padre de validación.

`full-corpus.json` exige una edición declarada completa. Programa 96 casos por
brazo para Estados Unidos, China y su unión. Añade otros 96 para contrastar la
unión con igual peso por mercado. La ponderación natural conserva peso uno por
fila. La equilibrada utiliza $w_m=N/(M N_m)$, donde $N_m$ es el número de muestras
de ajuste del mercado, $N$ es el total y $M$ el número de mercados. No retira filas
del mercado mayor ni calcula pesos con validación.

El brazo conjunto conserva exactamente las claves de ambos mercados. No significa
que se haya identificado una influencia causal entre bolsas. El contexto cruzado
es una ampliación separada que requiere disponibilidad temporal comprobada.

## Ejecución de la edición disponible

La configuración de desarrollo permite trabajar mientras continúa la curación.
No cambia el alcance de la campaña completa ni autoriza marcarla como terminada.
La orden siguiente corresponde a los artefactos locales del 22 de septiembre.
La salida debe ser nueva la primera vez.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
uv run --no-sync python -m mars_titan.training.reference_campaign \
  --config configs/baselines/verified-development.json \
  --manifest data/processed/training-supervision-20260922/manifest.json \
  --output data/interim/verified-streaming-campaign-20260922
```

Para continuar se repite la orden con `--resume`. Los casos terminados se
verifican mediante sus recibos y huellas, sin entrenarlos otra vez. Un caso
interrumpido restaura modelo, AdamW, generadores, estadísticas de la época y cursor
confirmado. La lectura adelantada no modifica ese cursor.

SIGINT y SIGTERM solicitan una parada entre operaciones. El estado se guarda tras
una actualización completa y al cerrar cada época. También se solicita guardar
cada 900 segundos. Un proceso terminado de forma abrupta vuelve al último estado
confirmado. Los cambios de datos, configuración o código impiden mezclar estados.

## Artefactos y límites

`summary.json` conserva el estado de cada configuración. Cada ejecución guarda
`run.json`, checkpoints y predicciones de ajuste y validación en Parquet. Las
predicciones se escriben por bloques e incluyen clave de muestra, mercado, fecha,
objetivo, estimación y referencia cero. Las métricas por época son MAE y MSE, con
recuentos y tiempo de recorrido. No constituyen por sí solas la evaluación
estadística o financiera completa.

Los modelos son compactos, con 32 unidades en los codificadores y las cuatro
modalidades más contexto macro. FP32, AdamW y estado recurrente reiniciado por
ventana. No se utilizan pesos entrenados de MARS-TITAN. Los grupos Parquet, lotes
y puntos de control tienen presupuestos explícitos. Los originales no se
modifican.

El lote inicial sigue siendo 16. No se aumenta por disponer de VRAM libre sin
medir su efecto sobre tiempo y trayectoria de optimización. Las referencias
tabulares conservan su implementación propia y requieren la misma población.
La implementación del motor no demuestra que todas las fuentes estén verificadas
ni que se haya ejecutado la campaña completa.
