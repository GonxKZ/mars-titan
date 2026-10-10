# Escritura episódica M3

M3 es la selección completa de [#19](https://github.com/GonxKZ/mars-titan/issues/19): el índice selectivo del banco episódico ordena los episodios maduros por una combinación de error maduro, anomalía de entradas y relevancia de información publicada. Comparte con [M2](mature-error-write-policy.md) la capacidad, los cupos 50/25/25, la semilla del reservorio y las ofertas a los tres índices. Solo cambia la puntuación del índice selectivo.

Está implementada como `episodic_m3_three_index_v1` en `memory/write_scores.py` y `memory/write_policy.py`, conectada en `FinancialSession`, en el recorrido cronológico del lector, en la ventana walk-forward, en el traslado de la variante B y en el plan de la campaña con el brazo `mars_titan_m3`. Se ha comprobado en CPU con pruebas técnicas, sin pasos de optimizador que modifiquen pesos. No se ha ejecutado con datos, no se ha usado la GPU y no hay ningún resultado sobre su utilidad. La puntuación es una hipótesis que la comparación A11 frente a M2 puede descartar.

## Componentes

Los tres componentes se calculan y se guardan por separado. Ninguno usa la sorpresa asociativa de Titans, que pertenece a la memoria neuronal y no ve etiquetas.

| Componente | Qué mide | Entrada | Disponible en | Origen |
| --- | --- | --- | --- | --- |
| e | Error absoluto de la predicción emitida, `abs(etiqueta − predicción emitida)` | Etiqueta madura y valor emitido del registro | Maduración | Fórmula de la [arquitectura candidata](../research/candidate-architecture.md#escritura-y-retención), igual que M2 |
| a | Choque del último rendimiento frente al pasado reciente del activo | Ventana de precios de 64 sesiones de la decisión | Corte de la decisión | Derivación propia a partir de la propuesta de la arquitectura candidata |
| r | Frescura de las publicaciones del activo | Bit de noticias y bloque de fundamentales con máscaras y edades | Corte de la decisión | Derivación propia sobre los campos de la edición v3 |

### Anomalía

La ventana de precios de la edición con máscaras contiene `log(OHLC / primer cierre)` en las 64 sesiones que terminan en la sesión de la decisión. Con el canal de cierre se obtienen 63 log-rendimientos. La anomalía compara el último con los 62 anteriores:

```text
a_raw = |ρ_t| / max(MAD(ρ_{t−62}, …, ρ_{t−1}), 1e−12)
```

donde MAD es la mediana de las desviaciones absolutas respecto a la mediana. El último rendimiento no entra en su propia escala, así que un choque aislado no se diluye. La constante 1,4826 que convierte la MAD en una desviación típica no se aplica porque se cancela al normalizar con la mediana de entrenamiento. Si más de la mitad de los 62 rendimientos coinciden, por ejemplo con muchas sesiones sin cambio de precio, la MAD vale cero. Entonces un rendimiento nulo da anomalía cero y uno distinto de cero da un valor finito muy alto, que la normalización lleva cerca de 1.

La [arquitectura candidata](../research/candidate-architecture.md#escritura-y-retención) proponía partir de un cambio observado de retorno o volatilidad respecto a una escala histórica robusta. Se eligió el rendimiento del propio activo porque los precios son la única modalidad siempre presente en la edición con máscaras y porque la ventana de la decisión ya acredita su disponibilidad. Se descartó medir la distancia de la clave del codec a la distribución de entrenamiento: la clave normaliza cada modalidad por su norma, pierde la magnitud del movimiento y mezcla los bits de presencia, que pertenecen a r.

### Relevancia

El lote de decisión de la edición v3 conserva el bit de presencia de noticias (al menos un artículo admitido en las cinco sesiones que terminan en el corte de la decisión, `news_lookback_sessions=5`) y, para los fundamentales, una máscara y `log1p(edad en días)` por concepto. Con esos campos:

- Frescura contable: `r_f = m_f / (m_f + edad)`, con la edad del concepto observado más reciente. Vale 1 con una publicación del mismo día y 0,5 con la edad mediana de entrenamiento.
- Noticias: `r_n = 1 − p_news / 2` cuando hay noticias, donde `p_news` es la proporción de decisiones de entrenamiento con noticias. Es el rango medio de la presencia, así que una noticia es más informativa cuanto menos frecuente fue en entrenamiento.
- `r = max(componentes conocidos)`. Si no se conoce ninguno, `r = 0,5`.

Una modalidad ausente no es relevancia nula. El lote no distingue una fuente sin cobertura de una ausencia real de publicaciones, porque `missing_reasons` y `news_count` se quedan en el Parquet de muestras. Por eso una ausencia se trata como desconocida y recibe el valor neutro 0,5, la mediana de entrenamiento. Se usa el máximo y no la media para que una publicación adicional nunca reduzca la relevancia. Con la media, una noticia frecuente podría bajar la relevancia de una presentación contable fresca.

Los indicadores macro no forman parte de r. Son comunes a todos los activos de un mercado en cada sesión, las series diarias dominan su edad mínima y no existe un catálogo acreditado de importancia por indicador. Por tanto r mide la relevancia de las publicaciones del propio activo, no de toda la información publicada. Esta limitación figura en la declaración como `macro_relevance: not_included`. No se construye una sorpresa frente a expectativas porque no hay consenso histórico verificable.

## Escalas congeladas

`fit_write_scalers` recorre una vez el índice de observaciones del tramo `train` de la ventana:

| Escala | Definición |
| --- | --- |
| `error_median` | Mediana de `abs(etiqueta)` de las etiquetas maduras del tramo. Es el error mediano del pronóstico nulo y no depende de ningún modelo |
| `anomaly_median` | Mediana de `a_raw` de las decisiones del intervalo de decisión del tramo |
| `filing_age_median` | Mediana de la edad contable de las decisiones con fundamentales, o `None` si no hay ninguna |
| `news_share` | Proporción de decisiones con noticias |

Cada mediana se calcula sobre una muestra uniforme de reservorio de hasta 65.536 valores, con PCG64, semilla 19 y un flujo por estadístico. Por debajo de esa cifra la mediana es exacta. Una mediana nula o no finita hace fallar el ajuste, porque no permite normalizar.

`WriteScalers` guarda las medianas, los recuentos, el índice de entrenamiento y su intervalo de decisión, con una huella SHA-256. Forma parte de `CompositeScoreConfig`, así que entra en la identidad del banco, en la identidad del ajuste del lector, en `fit/run.json` y en cada checkpoint. `ReadoutTrainer` rechaza unas escalas que no procedan de su propio tramo de entrenamiento. Una ventana reanudada reutiliza las escalas de su `fit/run.json` sin volver a recorrer el tramo. El traslado de la variante B aplica las escalas del ancla y registra su huella en `memory_policy.write_scalers_sha256`.

Dentro del tramo de entrenamiento, las escalas usan todo el tramo, como cualquier normalización ajustada con entrenamiento y como el propio padre Titans-MAC. Esa información posterior solo actúa dentro del ajuste. Validación, calibración y evaluación nunca intervienen en las escalas.

## Normalización y puntuación

Cada componente se transforma con `u(x) = x / (x + m)`, donde `m` es su mediana de entrenamiento. La transformación es estrictamente creciente, está acotada en [0, 1] y lleva la mediana a 0,5. Así un error o un choque extremos no monopolizan el índice, como pedía la arquitectura candidata. Se descartó la función de distribución empírica porque crea empates entre puntos de referencia y rompe la equivalencia exacta con M2 cuando solo pesa el error.

```text
s = w_e·e_norm + w_a·a_norm + w_r·r_norm,   w = (1/3, 1/3, 1/3)
```

El índice selectivo conserva las mayores puntuaciones con desempate por menor ID, como M2. Con `w = (1, 0, 0)` selecciona exactamente los mismos episodios que M2, y las pruebas lo comprueban.

Los pesos están fijados antes de ejecutar. Cada brazo de MARS-TITAN busca dos casos, las tasas 1e-4 y 1e-3, y M3 conserva esos dos casos. Elegir pesos en validación habría exigido más casos para M3 que para M2 o renunciar a la búsqueda de la tasa. Los tres componentes comparten escala y orientación, y no hay evidencia de desarrollo que favorezca a uno. En ese caso los pesos unitarios suelen ser robustos frente a pesos estimados (Dawes, 1979), aunque esa evidencia procede de otros dominios. Aquí es una decisión de diseño, no un resultado. Los pesos forman parte de la configuración, así que las ablaciones de un solo componente se pueden declarar con la misma implementación como identidades nuevas.

M3 no tiene umbrales. El cupo selectivo, un cuarto de B_mem, limita las escrituras igual que en M2. Un umbral de admisión cambiaría el número de escrituras respecto a M1 y M2. La diversidad respecto a los episodios existentes queda fuera de M3. Sería un cuarto criterio con otra identidad y con el coste de comparar cada candidato con el índice selectivo.

## Banco, registro y recuperación

`MatureErrorBank` recibe `CompositeScoreConfig` y compone los mismos tres índices nativos que M2. Todos los candidatos maduros se ofrecen a los tres índices y cada uno cuenta tres ofertas. El reservorio y los recientes coinciden con los de M2 con las mismas ofertas, y la capacidad física total sigue siendo B_mem.

El banco guarda los componentes sin normalizar de cada episodio retenido en U, S o R (error absoluto, anomalía, edad contable o `None` y presencia de noticias). Al restaurar comprueba que S contenga exactamente las mayores puntuaciones de los retenidos. El recibo de cada evento añade los componentes normalizados, la puntuación y la máscara de relevancia conocida de cada candidato ofrecido, junto con los admitidos, rechazados y expulsados del selectivo. La verificación concilia esos campos con los índices y con los componentes guardados. La sesión contrasta además el error de cada episodio retenido con su predicción emitida, como hace con M2.

El recorrido del lector acumula por pasada `selective_admitted`, `selective_rejected`, `selective_evicted` y `relevance_unknown`, que llegan al historial de `run.json`. Los rasgos de cada predicción pendiente y de cada etiqueta en espera se guardan en el checkpoint de recuperación con el banco empaquetado, sus componentes, sus contadores y las escalas de la identidad.

## Orden y causalidad

La anomalía y la relevancia se calculan con las entradas de la decisión. En el recorrido del lector se obtienen al emitir y se conservan con la predicción pendiente. En la sesión se recalculan al madurar con los inputs retenidos de esa decisión, que la sesión ya conserva para cada pendiente. Las dos rutas aplican la misma función con aritmética elemental por fila, así que no dependen del lote. El error solo existe al madurar la etiqueta. La selección ocurre después de emitir todas las predicciones del evento y su efecto empieza en el evento siguiente.

## Coste

Por candidato, M3 añade la mediana de 62 rendimientos dos veces, el mínimo de las edades observadas y unas pocas operaciones escalares. La selección ordena los mismos residentes y candidatos que M2. La sesión lee una vez más los artefactos de inputs retenidos en los eventos con maduraciones, y cada ajuste recorre una vez su tramo de entrenamiento para estimar las escalas. La orden de caudal de la campaña ya mide el lector M3 como los demás lectores, hasta el paso y sin cambiar pesos, y estima antes sus escalas con `window_scalers`, la misma regla que la campaña, sobre el tramo de entrenamiento de la ventana medida. Guarda su huella y la duración de ese recorrido en `write_scalers`, que no se suma a las horas, igual que los demás normalizadores. La orden no se ha ejecutado, así que estos costes siguen sin medir sobre el corpus.

## Revisión adversarial

- Fuga temporal. Ninguna etiqueta inmadura entra en la admisión y ningún dato de validación o evaluación entra en las escalas. Las pruebas alteran etiquetas que maduran después de un corte y entradas posteriores, y las admisiones anteriores no cambian. Alterar entradas, precios y etiquetas posteriores al tramo de entrenamiento no cambia ninguna escala.
- Sesgo hacia la volatilidad. La anomalía favorece periodos agitados, que también suelen tener errores grandes. Puede conservar ruido en lugar de señal. Por eso se contrasta con M2, que ya selecciona por error.
- Sesgo de cobertura. La relevancia favorece activos y épocas con noticias o fundamentales. El valor neutro evita castigar la ausencia, pero un activo con cobertura completa puede tener más oportunidades de puntuar alto.
- Activos poco negociados. Con más de la mitad de rendimientos nulos en la ventana, cualquier movimiento recibe una anomalía normalizada cercana a 1. M3 puede favorecer así a activos ilíquidos frente a choques reales de activos líquidos. No se corrige sin evidencia, pero conviene revisar la composición del índice selectivo por liquidez en la comparación.
- Escala del error. La mediana de `abs(etiqueta)` supera previsiblemente la del error de un buen predictor. Entonces e se concentra por debajo de 0,5 y pesa algo menos que a y r en la práctica.
- Alternativas más sencillas. M2 selecciona solo por error y M1 usa un reservorio uniforme con la misma capacidad. Si M3 no mejora el MAE residual por sesión frente a M2 y M1, con emparejamiento temporal e incertidumbre por bloques, se descarta. Una mejora en la recuperación de episodios no basta.

## Comprobaciones

| Archivo | Qué comprueba |
| --- | --- |
| `tests/memory/test_write_scores.py` | Anomalía frente a un oráculo con `statistics`, independencia del lote, edad contable, ausencias, normalización, escalas solo de entrenamiento, reservorio uniforme y medianas frente a un recorrido directo |
| `tests/memory/test_write_policy_m3.py` | Mismos cupos, ofertas, reservorio y recientes que M2, equivalencia con M2 cuando solo pesa el error, desempates, valor neutro, recuperación exacta y snapshots alterados |
| `tests/memory/test_mars_titan_session_parity.py` | La sesión y el recorrido del lector emiten las mismas 20 predicciones bit a bit con M3, K = 1 y K = 2 con episodios fijos, con los tres componentes variando entre flujos |
| `tests/memory/test_financial_session_m3.py` | Etiquetas inmaduras y entradas posteriores, igualdad de candidatos y capacidad frente a M1 y M2, corte antes de publicar y contraste del error con la emisión |
| `tests/training/test_mars_titan_run.py` | Escalas del propio tramo, bucle hasta el paso con el registrador, contadores y reanudación con escalas, rasgos y contadores |
| `tests/training/test_mars_titan_walk_forward.py` | Ventana M3 completa, escalas guardadas, reutilización al reanudar y escalas del ancla |
| `tests/training/test_mars_titan_campaign.py` | `mars_titan_m3` sin pendientes, la comparación de 27 brazos con productor, salvo el control en línea, cuando se declaran las cuatro secciones y una campaña B reducida con M3 ajustado y trasladado |
| `tests/training/test_campaign_throughput.py` | Medida del lector M3 hasta el paso sin cambiar pesos, con sus contadores del selectivo y escalas iguales a las de la regla de la campaña |
| `tests/training/test_campaign_extensions.py` | Declaración preparada con M3 entre sus brazos: 5.985 ajustes en A, 2.261 ajustes y 2.380 traslados en B y 29 predictores en la etapa de políticas |

Las pruebas que recorren el ajuste usan el registrador de gradientes, que no modifica pesos, con la protección de aprendizaje activa. Las etiquetas son manuales o proceden del corpus técnico sintético de las pruebas, y no se genera ningún objetivo real.

Se aplicaron de una en una 15 mutaciones dirigidas en una copia aislada, comprobando antes que las pruebas importaban la copia. Catorce hicieron fallar las pruebas: incluir el último rendimiento en su propia MAD, tratar la relevancia desconocida como cero, usar el mínimo en lugar del máximo, puntuar la noticia con `1 − p_news`, estimar la escala del error con el signo, aceptar otra partición o decisiones anteriores al intervalo de entrenamiento, retirar la anomalía de la puntuación, desempatar por el mayor ID, entregar a la sesión los rasgos en otro orden, reanudar sin la presencia de noticias, aceptar en el lector escalas de otro tramo, dejar de contrastar en la sesión el error M3 con la emisión y volver a estimar las escalas al reanudar una ventana. La restante retiraba la comparación de IDs del selectivo al restaurar el banco. Era equivalente, porque la comparación de puntuaciones ya la cubría, así que se retiró del código. En la medida de caudal se aplicaron cinco mutaciones más (M3 sin escalas, escalas del tramo de validación, sin contadores del selectivo, recorrido creado sin esos contadores y escalas sin registrar) y las cinco hicieron fallar las pruebas.

## Pendiente

- Ejecutar la comparación A11 en la campaña A. La edición histórica desde 2000 y sus objetivos ya están verificados, pero el bloqueo de aprendizaje sigue activo.
- Ejecutar en `cuda:0` la orden de caudal con `--extensions`, que ya incluye `mars_titan_m3` y necesita las vistas reales.
- Declarar, si se justifica, las ablaciones de un solo componente, la diversidad y la relevancia macro como identidades nuevas.

## Comprobaciones CUDA

Dos comprobaciones tienen casos M3. Las dos se ejecutaron en `cuda:0` el 9 de octubre, sin pasos de optimizador y con el enlace `native-candidate-cuda` de `develop` ([resumen](../../reports/engineering/cuda-checks-20261009/README.md)):

- `tests/memory/cuda_mature_error_check.py::test_cuda_m3_scores_parity_and_recovery` recorre la sesión M3 con B_mem = 4 y K = 1 en CPU y en el dispositivo, en FP32 y FP64. Exige los mismos IDs, ofertas y rasgos de entrada, el error, la puntuación y las predicciones dentro de las tolerancias del caso M2, la recuperación exacta en cada dispositivo y un margen mínimo de 1e-4 entre la puntuación elegida y la mejor descartada. En `cuda:0` pasó con un margen de 6,8e-3 y un error máximo de predicción de 1,9e-9 en FP32 y 2,6e-18 en FP64 ([recibo](../../reports/engineering/cuda-checks-20261009/m3-mature-error-cuda.json)). `MARS_TITAN_M3_CHECK_DEVICE=cpu` permite ensayarla sin GPU.
- `tests/training/cuda_mars_titan_run_check.py` compara el ajuste sin pasos y su validación con M3 y K = 1, y con M3, K = 4 y episodios fijos, además de los casos M1. Para M3 exige IDs, índices, anomalía, relevancia y máscara iguales, error y puntuación dentro de la tolerancia y un margen mínimo de 1e-4. En `cuda:0` pasaron sus diez casos, con un margen mínimo de 6,8e-4 en los casos M3 ([recibo](../../reports/engineering/cuda-checks-20261009/mars-titan-readout-cuda.json)). `MARS_TITAN_MARS_RUN_CHECK_DEVICE=cpu` permite ensayarla sin GPU.

En la repetición del 9 de octubre el caso M2 de la primera comprobación exigía el enlace nativo con huella `e7559ed4…`, así que la orden de M3 lo excluyó. [#418](https://github.com/GonxKZ/mars-titan/pull/418) retiró esa huella fija y, con el enlace de `develop`, la misma orden sin `-k` da M2 y M3 superadas en `cuda:0`, como recoge la [guía de M2](mature-error-write-policy.md). Los casos M3 registran la huella del enlace usado. Las órdenes, desde la raíz del repositorio:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo compilado> \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/memory/cuda_mature_error_check.py -k m3
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  MARS_TITAN_EPISODIC_NATIVE=<enlace episódico nativo compilado> \
  MARS_TITAN_MARS_RUN_CHECK_REPORT=$PWD/mars-titan-readout-cuda.json \
  UV_PROJECT_ENVIRONMENT=<entorno uv con PyTorch CUDA> \
  uv run --no-sync python -m pytest -q tests/training/cuda_mars_titan_run_check.py
```

## Referencias

Dawes, R. M. (1979). The robust beauty of improper linear models in decision making. *American Psychologist, 34*(7), 571–582. https://doi.org/10.1037/0003-066X.34.7.571
