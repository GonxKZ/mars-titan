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
  (ml4t-non-stationarity).

**Por dónde puede colarse el futuro:**
- La cabeza se ajusta en validación, que ya decidió la parada del núcleo: es levemente optimista.
- CQR usa calibración, que está limpia.
- Ninguna de las dos toca evaluación.

**Abandono:** la cabeza se abandona si, en 2013 y en los tres primeros orígenes limpios, su error
de cobertura no baja al menos un punto del de CQR o su anchura no es al menos un 5 % menor con la
misma cobertura.

**Coste:** 0 h de GPU y minutos de CPU por origen, dentro de la orden del núcleo agéntico.
