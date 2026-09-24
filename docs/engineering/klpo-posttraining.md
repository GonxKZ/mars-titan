# Ajuste predictivo con KLPO

La ampliación usa el entrenador recuperable de los controles predictivos.
No modifica la arquitectura MARS-TITAN ni los entrenamientos padre.
La [derivación y las fuentes](../references/klpo-review.md) delimitan qué parte
del artículo se adapta y qué garantías no se trasladan al problema financiero.

## Diseño reproducible

[klpo-adaptation.json](../../configs/baselines/klpo-adaptation.json) define
36 ajustes por padre. Las semillas son 42, 43 y 44. Cada ajuste recorre cinco
épocas completas de la población de entrenamiento admitida.

| Objetivo | Variantes por semilla | Muestreo |
| --- | --- | --- |
| REINFORCE | Una | Una acción de la política actual |
| Pérdida esperada | Una | Enumeración de las 21 acciones |
| MAE del centro | Una | Sin acciones discretas |
| KLPO Full-KL | β de 0,03, 0,1 y 0,3 | Una acción del padre, corrección exacta |
| KLPO MC | Los mismos tres valores de β | Una acción y 128 auxiliares independientes del padre |
| Varianza exacta de KLPO | Los mismos tres valores de β | Enumeración, sin Monte Carlo |

Todos comparten el adaptador residual lineal, las cuatro modalidades, macro y
la predicción del padre. El optimizador es AdamW con tasa $10^{-4}$, decaimiento
0,01 y norma máxima de gradiente 1. El lote tiene 256 filas. El lector utiliza
Parquet por bloques y no carga el corpus completo en RAM o VRAM.

La cola selecciona el ganador de la búsqueda de cada familia mediante MAE por
sesión de validación, con desempate por identificador. Usa RNN, LSTM, GRU,
DLinear, Ridge y XGBoost. No selecciona retrospectivamente el mejor padre de
una continuación. Los seis padres dan lugar a 216 ajustes. Las tres semillas
del adaptador no sustituyen las réplicas de entrenamiento del padre.

## Admisión y recuperación

`mars_titan.training.klpo_queue` exige que las campañas neuronal y tabular hayan
terminado. Comprueba sus recuentos, familias, predicciones, checkpoints y huellas.
La vista del corpus debe ser idéntica. Una campaña incompleta o una población
diferente se rechaza antes de preparar los ajustes.

El origen ordenado se prepara una sola vez. Cada padre añade una caché escalar
alineada por identificador, no otra copia de las modalidades. La normalización
se ajusta solo con entrenamiento y se reutiliza en todos sus controles.

El punto de recuperación conserva pesos, optimizador, cursor confirmado,
estadísticas de época, selección y estados aleatorios. KLPO guarda por separado
el generador de acciones y el de auxiliares. La distribución del padre queda
identificada por la rejilla, la mezcla, la caché y sus huellas. Las probabilidades
se reconstruyen a partir de esos datos fijos, no a partir de una política nueva.
Se guarda al menos al terminar una época y según el intervalo de 60 segundos.

Las salidas completadas se comprueban antes de reutilizarlas. No se reinician
con otra semilla ni se sobrescriben para acomodar cambios de código o datos.
Cada campaña nueva necesita otro directorio. La pausa de una ejecución se
solicita con SIGINT y se retoma con la misma configuración.

La cola, el origen ordenado, la caché padre, el estudio y cada ajuste registran
su identidad antes de generar artefactos. Un fallo en la primera escritura
puede retomarse si la carpeta solo contiene sus archivos de inicialización.
Una carpeta con contenido ajeno o con artefactos cuyo recibo se ha perdido se
rechaza. La identidad se comprueba también al regresar del último ajuste.

Para un padre y un corpus ordenado ya preparados:

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 uv run --no-sync --with gymnasium==1.3.0 \
  python -m mars_titan.training.predictive_study \
  --config configs/baselines/klpo-adaptation.json \
  --ordered data/interim/causal-source-verified-check-final-20260923/manifest.json \
  --parent data/interim/tabular-post-scan-20260923/ridge-1/run.json \
  --output data/interim/klpo-verified-check-20260924
```

Esas rutas corresponden a una comprobación histórica de 470 filas, no al corpus
completo. La cola de población completa utiliza las vistas y los padres de las
campañas completas:

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 uv run --no-sync --with gymnasium==1.3.0 \
  python -m mars_titan.training.klpo_queue \
  --config configs/baselines/klpo-adaptation.json \
  --reference data/interim/original-audited-us-scientific-search-20260923/summary.json \
  --tabular data/interim/original-audited-us-tabular-search-20260923/summary.json \
  --output data/interim/original-audited-us-klpo-20260924
```

La cola recupera una salida existente únicamente si su identidad coincide.
El comando de un único estudio requiere `--resume` para recuperar. No se debe
ejecutar otra campaña CUDA simultánea si no hay margen de memoria. El perfil
`:16:8` reduce el espacio de trabajo determinista de cuBLAS. No reduce la
precisión. Su coste debe medirse separado de `:4096:8`, sin presentar una
ejecución con cargas concurrentes como un benchmark aislado.

## Evidencia y datos sintéticos

Cada ejecución conserva predicciones Parquet de entrenamiento y validación,
MAE y MSE por filas y sesiones, comparación con el padre y retorno cero,
entropía, masa extrema y saturación. KLPO añade $D_{KL}(q\|p)$ y la varianza
exacta $F$. Los informes incluyen tiempos y picos de RAM y VRAM. La entropía de
esta política no es por sí misma una estimación calibrada de incertidumbre.

Las pruebas sintéticas enumeran un problema de tres acciones, contrastan
gradientes con diferencias finitas, verifican el óptimo de Gibbs sin restricciones
y ajustan una señal lineal conocida. También fuerzan soporte extremo y una pausa
de entrenamiento. Son datos de prueba identificados como tales. No simulan la
procedencia de una noticia real ni completan fundamentales ausentes.

El corpus estadounidense conserva la política `source_audited_not_external`.
No equivale a una verificación externa de cada noticia. La cola no amplía por
sí misma la cobertura china ni resuelve fechas de publicación pendientes.
Los resultados del conjunto de prueba final no se consultan en este flujo.

La [comprobación ejecutada sobre 470 muestras](../../reports/baselines/klpo-check-20260924.md)
conserva sus 36 ajustes y sus límites. Ninguno superó el retorno cero en esa
edición. Es una comprobación previa, no el resultado de la cola completa.
