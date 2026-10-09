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

## Comprobaciones pendientes

Siguen en #442, en este orden:
1. Sondas sin ajuste:
   - canarios de información futura en entradas de prueba que los detectores deben señalar;
   - ninguna fila de 2024 en artefactos de selección;
   - invariantes de caja, posiciones, costes y acciones imposibles en las cintas reales con políticas fijas;
   - cota con oráculo de información futura, marcada como diagnóstico y excluida de toda comparación.
2. Falsaciones que ajustan modelos, solo declaradas mientras dure el bloqueo de aprendizaje: placebo con etiquetas permutadas dentro de cada sesión, desplazamiento temporal de las entradas, predicción constante y aleatoria y estabilidad entre semillas. Cada una con su coste y su criterio de éxito o fracaso escritos de antemano.
3. Registro previo de comparaciones, métricas y criterios con huella y fecha, y registro de desviaciones.
4. Informe de integridad generado por la orden única de la campaña, que bloquea la publicación si alguna comprobación falla. Invocará también el verificador de disjunción del diseño por etapas ([#437](https://github.com/GonxKZ/mars-titan/issues/437)) en lugar de duplicarlo.
