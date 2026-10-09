# Banco de integridad de la evaluación

Un resultado de la campaña solo se publica si supera las comprobaciones de este banco. Cada una responde a una pregunta concreta sobre si una cifra puede creerse: si las métricas están bien calculadas, si todos los modelos se evalúan con las mismas filas, si alguna información futura se ha colado o si una política explota un fallo del entorno. El banco desarrolla [#28](https://github.com/GonxKZ/mars-titan/issues/28) y se sigue en [#442](https://github.com/GonxKZ/mars-titan/issues/442).

El código vive en `src/mars_titan/integrity/`. No sustituye a las comprobaciones que ya hacen la comparación walk-forward, el contrato de recibos ni la auditoría de los entornos. Las repite por otro camino o las reúne en un informe común.

## Segunda implementación de las métricas

`integrity/independent_scores.py` recalcula las métricas por sesión de la comparación a partir de su especificación en [métricas predictivas](metrics.md), sin importar `evaluation/forecast_scores.py` ni `evaluation/forecast_panel.py`.
- **Agrupación.** Usa pandas sobre el par (mercado, instante) en lugar de los índices de sesión y `bincount` del código principal.
- **Probabilidad implícita de subida.** Recorre cada fila buscando el último cuantil que no supera el cero, en lugar de la fórmula vectorial.

Si las dos implementaciones coinciden, un error de agregación o de convención tendría que estar repetido en las dos para pasar desapercibido.

Cubre:
- MAE y MSE por sesión;
- dirección con la convención de abstención;
- precisión y exhaustividad de cada signo;
- Rank IC con sus motivos de indefinición;
- pinball y frecuencia por nivel;
- cobertura, anchura y puntuación de los intervalos centrales del 80 % y del 95 %;
- Brier del signo y recuentos por intervalo de probabilidad para el ECE.

`integrity/score_recheck.py` aplica esa implementación a una comparación ya publicada:
1. Comprueba las huellas de `comparison.json`, `sessions.parquet`, la configuración y el manifiesto de fuentes.
2. Vuelve a leer las predicciones crudas de cada brazo, semilla y ventana, también con sus huellas.
3. Coteja la tabla de sesiones columna a columna y vuelve a agregar MAE, MSE y dirección en cada vista con la ponderación declarada.

El resultado cuenta las sesiones que faltan o sobran y las discrepancias por columna, y sale con código distinto de cero si alguna comprobación falla.

```bash
uv run python -m mars_titan.integrity.score_recheck \
  configs/evaluation/historical-masked-2000-comparison.json SOURCES.json US+CN COMPARISON_DIR \
  --output RECHECK.json
```

**Tolerancias.** Los recuentos y los motivos deben coincidir exactamente. Los valores reales usan una tolerancia relativa de 1e-10 y absoluta de 1e-13, que cubre la diferencia de orden de suma entre las dos implementaciones. Un valor indefinido solo coincide con otro indefinido.

**Límites declarados.**
- Las métricas con los cuantiles calibrados no se recalculan todavía, porque el calibrador común no tiene segunda implementación.
- El control cero no tiene archivo de predicciones.

El informe de cada recálculo indica las dos exclusiones.

**Pruebas.** `tests/integrity/` incluye:
- los ejemplos de la especificación;
- la regla de la probabilidad de subida con empates en cero;
- la coincidencia con el código principal en paneles con objetivos y predicciones nulos, sesiones de un activo y dos mercados;
- seis corrupciones de valores publicados que deben detectarse.

También hay dos mutaciones del código principal, una en la media por sesión y otra en la puntuación de intervalo. El recálculo las señala columna a columna sobre un estudio de juguete completo.

## Mismas filas en todos los brazos

La comparación ya exige que todos los brazos evalúen las mismas filas, con `ForecastPanel.cohort_sha256`. `integrity/row_identity.py` lo comprueba por otro camino:
1. Ordena las filas de cada archivo de predicciones por mercado, activo e instante con Arrow.
2. Calcula una huella SHA-256 propia con los bytes de esas tres claves y los bits exactos del objetivo en float64.
3. Agrupa las huellas por ventana y tramo (calibración y evaluación) y exige que todos los pares brazo-semilla compartan una sola.

Un bit distinto en un objetivo, una fila de más o una de menos separan al brazo afectado en su propio grupo, y el informe lo nombra.

Además, cuenta en cada archivo:
- las filas de la reserva de 2024;
- las filas fuera del tramo declarado de su ventana;
- las de mercados fuera del ámbito;
- las claves duplicadas;
- los objetivos no finitos.

Antes de leer cada archivo comprueba su huella frente al manifiesto de fuentes.

```bash
uv run python -m mars_titan.integrity.row_identity \
  configs/evaluation/historical-masked-2000-comparison.json SOURCES.json US+CN --output ROWS.json
```

**Coste medido.** Entre 0,6 y 0,9 s por archivo de 1,25 millones de filas en CPU, con dos hilos. Es el orden de una ventana de evaluación US+CN.

**Alcance.** Cubre los archivos de calibración y evaluación que entran en la comparación. Las predicciones de validación con las que cada runner elige su punto de parada no pasan por este manifiesto. Su comprobación contra la reserva de 2024 queda en la lista siguiente.

**Pruebas.** Sobre el estudio de juguete de la comparación:
- un estudio fiel pasa en US y en US+CN;
- se detectan un solo bit cambiado en un objetivo, una fila que falta en un brazo y una fila movida a 2024;
- se cuentan las claves duplicadas y los objetivos no finitos;
- se rechaza un archivo cambiado tras publicar el manifiesto;
- la huella no depende del orden de las filas ni de su partición en bloques.

## Registro previo y desviaciones

`integrity/preregistration.py` fija qué se va a comparar antes de producir resultados. El registro es un archivo JSON Lines que solo crece, con dos tipos de entrada:
- **Declaración.** Ruta y huella SHA-256 de un documento, por ejemplo la configuración de la comparación walk-forward, con sus brazos, métricas, familias de contrastes y parámetros del bootstrap. Va acompañada de una nota.
- **Desviación.** Cambio posterior sobre una declaración, con el motivo y la huella del documento que la sustituye. La declaración original no se borra.

Cada entrada guarda la huella canónica de la anterior. Si se edita, se borra o se reordena una línea, la cadena se rompe, aunque se recalcule la huella de la línea editada, porque la entrada siguiente apunta a la huella antigua. Las entradas inválidas se rechazan antes de escribir: desviaciones sin declaración, un documento declarado dos veces o fechas que retroceden.

`check_report` comprueba un informe de comparación:
1. Busca la declaración cuya huella coincide con `configuration.sha256` del informe.
2. Exige que la fecha de la declaración sea anterior a la del informe.
3. Lista las desviaciones registradas sobre esa declaración.
4. Si recibe el repositorio, busca el primer commit que añade la entrada, exige que sea anterior al informe e indica si ese commit ya está en un remoto.

```bash
uv run python -m mars_titan.integrity.preregistration declare \
  configs/evaluation/preregistration.jsonl configs/evaluation/historical-masked-2000-comparison.json \
  --note "Comparación principal de la campaña A"
uv run python -m mars_titan.integrity.preregistration check \
  configs/evaluation/preregistration.jsonl COMPARISON_DIR/comparison.json --repository . --output PREREG.json
```

**Límite declarado.** Las fechas locales y las de commit las fija el propio autor, así que por sí solas no prueban nada frente a terceros. La prueba externa es publicar el commit de la declaración en GitHub antes de lanzar la evaluación: el informe lo indica en `published_before_report`. La cadena tampoco impide declarar muchas variantes y publicar solo la favorable. Por eso todas las declaraciones y desviaciones del registro se publican con los resultados.

**Uso previsto.** La configuración definitiva de la campaña se declara cuando esté fusionada en `develop` y antes de generar ninguna predicción de evaluación. Hasta entonces el registro del repositorio no existe, para no acumular desviaciones sobre configuraciones que siguen cambiando.

**Pruebas.**
- Una declaración y su desviación forman una cadena válida.
- Se detectan una entrada editada (también con la huella recalculada), una eliminada y dos reordenadas.
- Las entradas inválidas no llegan al archivo.
- Un informe anterior a su declaración, o sin declarar, falla.
- Con un repositorio git temporal y un remoto local, se distinguen el commit local, el commit publicado y el commit posterior al informe.

## Comprobaciones pendientes

Siguen en #442, en este orden:
1. Sondas sin ajuste:
   - canarios de información futura en entradas de prueba que los detectores deben señalar;
   - ninguna fila de 2024 en las predicciones de validación y en los demás artefactos de selección;
   - invariantes de caja, posiciones, costes y acciones imposibles en las cintas reales con políticas fijas;
   - cota con oráculo de información futura, marcada como diagnóstico y excluida de toda comparación.
2. Falsaciones que ajustan modelos, solo declaradas mientras dure el bloqueo de aprendizaje: placebo con etiquetas permutadas dentro de cada sesión, desplazamiento temporal de las entradas, predicción constante y aleatoria y estabilidad entre semillas. Cada una con su coste y su criterio de éxito o fracaso escritos de antemano.
3. Informe de integridad generado por la orden única de la campaña, que bloquea la publicación si alguna comprobación falla. Invocará también el verificador de disjunción del diseño por etapas ([#437](https://github.com/GonxKZ/mars-titan/issues/437)) en lugar de duplicarlo.
