# Post-entrenamientos para los ajustes finales: diseño (sin ejecutar)

Punto C del encargo del coordinador (22 de noviembre de 2026, 20:50 UTC). Hay tres propuestas.
Cada una cambia una sola variable frente a su control, con un criterio de abandono fijado antes
de medir. Nada de esto se ha ejecutado: los costes son estimaciones a partir de tiempos medidos.

## Tiempos medidos

| Trabajo | Medido |
|---|---|
| Ajuste de `titans_mac_nogain` en la DGX (2013, semilla 42; mejor época 11) | 6.248 s, 1,74 h |
| Regeneración de las emisiones de un ajuste (validación, calibración y evaluación) en la GPU local | 5.413 s (paridad de los gemelos de la v10, con congelaciones) |
| `lightgbm_technical` en CPU (3 hilos, 2013) | 511 s |

## Reglas comunes

- Cada variante se mide frente a su control con la misma semilla y en los mismos tramos. El efecto
  se informa con las tres semillas en 2013 (desarrollo) y después en los nueve orígenes limpios.
- Las diferencias van pareadas por sesión con t HAC, y entre orígenes con 8 grados de libertad.
- El criterio de abandono se decide con validación y calibración, nunca con evaluación. 2024 sigue
  sellado.
- **Purga y embargo** (ml4t-purging-embargo). La etiqueta es el residuo de la sesión siguiente:
  horizonte de 1 sesión, más 1 de embargo.
  - Ninguna fila de ajuste puede tener su madurez dentro de la sesión de purga ni de la de embargo.
  - Se usa la madurez exacta de la vista, no un retardo supuesto.

## Mercados, rasgos y factor de mercado (D-7, H10)

Añadido el 23 de noviembre, con el hallazgo D-7 del cuaderno (8a8c5e9e de dgx-trainning) y la
hipótesis H10 (protocolo 47ede1c7).

**El hallazgo.**
- La vista declara para China el factor CSI 300 (`…/csi300-inherited-gaps-recovery/price-factor-2006-2023-v16/prices.parquet`,
  huella b96cd972…).
- Ese fichero no está en la DGX. Comprobado el 23-11: tampoco está en este equipo, mientras que el
  de EE. UU. (SPY, 3540094000…) sí está.
- `corpus_inputs._market_factor` apaga entonces sin aviso los rasgos del modelo de mercado:
  alpha_percent, alpha_t, beta, log_residual_volatility, has_market_model, realized_label_1/5/21,
  alpha_21_percent y alpha_63_percent valen cero en todos los activos chinos.
- La etiqueta china sí es un residuo del CSI 300: y = r(j+1) − α(j) − β(j)·m(j+1).

Por eso todo ajuste con China hasta hoy ha entrenado sin el modelo de mercado.

**Factor por mercado en los ajustes finales.**
- EE. UU.: SPY de apertura a cierre, como ahora.
- China, por orden de preferencia:
  1. **El CSI 300 declarado, restaurado con su huella.** Es el de la etiqueta, así que no cambia
     ninguna otra variable.
  2. **El factor despejado de las propias etiquetas** (`x7_factor_reconstruido.py`; residuo
     mediano de 1e-9 a 1e-6 desde 2008).
     - Ganancia medida con árboles en 2013: +0,0079.
     - Riesgo de fuga: m(t) solo puede salir de etiquetas ya maduras en t, que son las de las
       decisiones de t − 1. Hay que reconstruirlo en un recorrido cronológico y con la madurez
       exacta de la vista, nunca de una vez con toda la serie.
  3. **El segundo modelo de mercado de H10**, contra la media simple del rendimiento intradía del
     propio universo en cada sesión, en los dos mercados.
     - Ganancia medida con árboles en 2013: de 0,117 a 0,130 (+0,0098 a +0,0130 en tres semillas,
       t 3,6 a 4,4).
     - Entra como un segundo juego de rasgos. No da el alfa de la etiqueta, así que no sirve de
       desplazamiento conocido para lo operable.

  La elección entre los tres es de H10 y del coordinador, no de estos post-entrenamientos. Cada
  post-entrenamiento se mide con el factor ya fijado, nunca a la vez que se cambia.
- **Regla común: fallar en vez de apagar.**
  - Si un mercado con filas declara un factor cuyo fichero no está o no conserva su huella, el
    ajuste no arranca.
  - El recibo de cada ajuste guarda, por mercado y tramo, la fracción de filas con
    has_market_model = 1. En China, desde 2008, debe ser la misma que en la vista.

**Rasgos.**
- `long_horizon_v3` son los 46 de la v2 (prefijo bit a bit) más 16 de precio y volumen validados
  en China.
- La rama técnica lee el ancho de la versión declarada.
- Toda columna que se cuenta desde el final del bloque del gráfico se deriva del nombre del rasgo
  en la versión declarada, nunca de un número fijo:
  - `offset_column`, el alfa: −46 con v2 y −62 con v3;
  - también `session_ic_term.liquidity_column`. Hoy la regla de cac92bf8 solo comprueba
    `technical_branch.columns` y `offset_column`, y con v3 un `liquidity_column` escrito para la
    v2 leería otra columna sin aviso.

**Paquete de Colab.** Para que China no entrene a ciegas también allí, el paquete lleva:
1. Los ficheros de factor de cada mercado con sus huellas, en las rutas que declara su manifiesto:
   - SPY;
   - para China, el CSI 300 restaurado, o el factor elegido en H10 con su receta y su huella.
2. Una comprobación previa que corre antes de cargar la GPU y para con un mensaje. Exige, por
   mercado:
   - que el factor declarado existe y conserva su huella;
   - que la fracción de has_market_model = 1 en el tramo de ajuste supera la de la vista menos un
     margen fijo;
   - que el ancho de rasgos y las columnas contadas desde el final cuadran con la versión
     declarada.
3. La versión de rasgos con su lista de nombres, para derivar por nombre las columnas que se
   cuentan desde el final.
4. El recibo de la comprobación dentro del run.json del ajuste.

**Lectura por mercado.**
- Toda medida de estos post-entrenamientos sobre estados conjuntos (H9:
  `dgx-titans-us-cn-v31/v11-inputs-v2|v3`) se lee por mercado (`dgx_seed_contrasts.py --market`).
- El IC operable y los costes solo existen en EE. UU. mientras China no tenga alfa conocido.

## C1. Destilación del conjunto hacia el núcleo

**Variable:** la pérdida del alumno incluye o no la distancia al profesor, con un peso w = 0,3
fijado de antemano. El control es el mismo ajuste sin destilación.

**Profesor:** las predicciones de los árboles fuera de ajuste y estrictamente cronológicas.
- El ajuste se divide en bloques anuales. Los árboles de cada bloque se ajustan solo con los
  bloques anteriores (ventana creciente, ml4t-walk-forward-cv). Se quitan las filas cuya madurez
  cae en el bloque o en su embargo.
- El primer bloque no tiene profesor y lleva peso 0.
- El conjunto con el propio núcleo pediría ajustes del núcleo por bloque (unas 10 veces el coste
  en GPU). Como el alumno ya aprende su visión de las etiquetas, lo que le falta es la de los
  árboles, y esos son baratos.

**Qué tiene el entrenador** (`training/distillation.py` y `financial_run.py`):
- La pérdida (1 − w)·tarea + w·L1 hasta las salidas del profesor.
- El profesor es un único ajuste del mismo tramo.
- Sus salidas en el tramo de ajuste se declaran `in_sample`: sus pesos han visto todo el tramo.

**Qué falta:**
1. **Otro tipo de objetivos del profesor, fuera de ajuste.** Con su corte de ajuste por fila y la
   comprobación, fila a fila, de que corte + purga < decisión. `TeacherTargets` exige hoy
   `in_sample` y `out_of_sample_predictions = False`.
2. **Filas sin objetivo con peso 0.** Hacen falta para el primer bloque y para las purgas. Hoy
   `values()` falla si falta la salida de una fila.
3. **Una distancia en escala de rangos.** El profesor es una puntuación de rangos: la distancia
   sería 1 − correlación por sesión entre la mediana del alumno y el profesor. Hoy la L1 exige
   salidas de la misma forma (los cinco cuantiles o el punto) en las mismas unidades.
4. **Una identidad del profesor por bloques.** Cada ajuste de árbol con su ventana y su receta,
   comprobada contra el corte. Hoy `teacher_identity` exige el mismo tramo de ajuste que el alumno.

**Por dónde puede colarse el futuro:**
- Los objetivos dentro de la muestra, el diseño de hoy: el profesor memoriza ruido posterior a
  cada fila.
- Etiquetas que solapan el borde de cada bloque, sin purga ni embargo.
- Estadísticos de normalización calculados con todo el tramo: hay que usar
  `integrity/past_only_preprocessing`.
- La receta de los árboles se fijó con el desarrollo de 2013: en los bloques anteriores a 2013 es
  una elección con información posterior, leve pero declarada.
- w elegido fuera de validación. Aquí se fija de antemano.
- China (D-7): los árboles que hacen de profesor también vieron ceros en el modelo de mercado. El
  profesor y el alumno se ajustan con el mismo factor ya fijado. Si es el reconstruido, su m(t)
  sale de etiquetas maduras en t, en el mismo recorrido cronológico que el profesor.

**Abandono:** se abandona si cualquiera de las dos condiciones falla en validación y calibración
de 2013, con tres semillas:
- la media de rangos del alumno destilado y los árboles no supera a S1 en ≥ 0,002 con t ≥ 2;
- el alumno solo no supera a su control en ≥ 0,002.

La destilación solo compensa si añade algo que la mezcla ya no tenga.

**Coste:** un ajuste del alumno por origen y semilla, ≈ 1,7 h de GPU. En 2013 con tres semillas,
≈ 5,2 h. En los nueve orígenes limpios, ≈ 46 h. Los árboles por bloque son ≈ 1 h de CPU por origen.

## C2. Ajuste fino de recencia (63 y 252 sesiones)

**Variable:** la longitud W de la ventana de ajuste fino, 63 o 252 sesiones. El control es el
punto elegido tal cual.

**Protocolo** (ml-training-recipes):
- Se parte del estado elegido del núcleo.
- Se ajusta sobre las últimas W sesiones maduras antes de la calibración: la cola del ajuste más
  la validación, menos la purga.
- La tasa de aprendizaje es 0,1 de la base, sin calentamiento y con un número fijo de pasos
  iguales en los dos brazos (≈ 3 épocas de 252 sesiones).
- No hay parada temprana: la validación está dentro de la ventana.
- Normalizadores congelados del ajuste original.
- La memoria en línea se calienta con un recorrido cronológico hasta el inicio de la ventana.
- Calibración y evaluación quedan intactas, así que lo operable y los intervalos se calibran igual
  que en el control.

**Por dónde puede colarse el futuro:**
- La ventana no puede llegar a la calibración: hay que purgar la última sesión de validación.
- Usar la validación dos veces (en la parada original y en el ajuste fino) la deja inservible
  como tramo de decisión de estos brazos. La decisión queda con la calibración, 61 sesiones, de
  poca potencia.
- Recalcular la normalización con datos recientes cambiaría dos variables a la vez.
- Un cambio de factor de mercado no se puede hacer como ajuste fino. En China los normalizadores
  y los pesos se ajustaron con ceros en los rasgos del modelo de mercado, así que pasar a valores
  reales pide un ajuste completo. C2 se mide después de fijar el factor y con el mismo factor en
  los dos brazos.

**Abandono:** se abandona un brazo si en calibración pierde frente al control en dos de tres
semillas, o si su diferencia media en los orígenes limpios no supera +0,001.

**Coste** (estimado): ≈ 0,2 s de GPU por sesión y época, más una pasada de inferencia; ≈ 0,3–0,5 h
por brazo, origen y semilla. Los dos brazos en 10 orígenes con 3 semillas son ≈ 20–30 h.

## C3. Cabeza de calibración posterior frente a CQR

**Variable:** el método de calibración posterior de los intervalos. Brazos:
- la cabeza, una regresión cuantílica lineal ajustada en validación sobre la anchura del núcleo y
  la media de rangos de S1;
- la cabeza conformalizada;
- CQR sobre los cuantiles del núcleo.

Control: los cuantiles del núcleo tal cual.

**Protocolo:** es la medida A2 de `evaluation/agent_core.py`, que ya se calcula por origen en CPU
y no necesita GPU. Hacerla dentro de la red, con una cabeza entrenada de extremo a extremo, sería
otra variable y no se propone.

**Medidas en evaluación:**
- el error de cobertura |cobertura − nominal| al 80 % y al 95 %;
- la anchura media a igual cobertura;
- la mayor desviación entre tercios de liquidez y deciles de volatilidad;
- la cobertura por trimestre, porque la intercambiabilidad no se cumple en el tiempo
  (ml4t-non-stationarity);
- en estados conjuntos, todo lo anterior por mercado. La corrección conforme se calibra por
  mercado: CQR con un cuantil común mezclaría dos escalas de error.

**Por dónde puede colarse el futuro:**
- La cabeza se ajusta en validación, que ya decidió la parada del núcleo: es levemente optimista.
- CQR usa calibración, que está limpia.
- Ninguna de las dos toca evaluación.

**Abandono:** la cabeza se abandona si, en 2013 y en los tres primeros orígenes limpios, su error
de cobertura no baja al menos un punto del de CQR o su anchura no es al menos un 5 % menor con la
misma cobertura.

**Coste:** 0 h de GPU y minutos de CPU por origen, dentro de la orden del núcleo agéntico.
