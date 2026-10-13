# Informe de integridad de la publicación (10 de octubre de 2026)

Este informe acompaña a [#442](https://github.com/GonxKZ/mars-titan/issues/442). Recoge cómo el banco de integridad pasa a ser un paso obligatorio de la publicación de la campaña, cómo se ha comprobado que detecta los fallos que persigue y cuánto cuesta en la forma real de A v2. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto. No se ha leído ningún objetivo, vista real ni modelo.

## Qué se ha implementado

| Pieza | Módulo | Pruebas |
| --- | --- | --- |
| Exclusión declarada del diseño conjunto (mercados aún no elegibles y mercados ajenos a los modelos prestados) en la identidad de filas y en el recálculo | `integrity/row_identity.py`, `integrity/score_recheck.py` | `tests/integrity/test_row_identity.py`, `tests/integrity/test_score_recheck.py` |
| Recálculo del control cero desde los objetivos del panel de referencia | `integrity/score_recheck.py` | `tests/integrity/test_score_recheck.py` |
| Registro de integridad de cada ventana antes de liberar sus filas, y comprobación de una comparación publicada desde filas o desde agregados | `integrity/comparison_integrity.py` | `tests/integrity/test_comparison_integrity.py` |
| Todos los trabajos declarados, con sus intentos, sin recibos de más, y las semillas y ventanas de cada comparación publicada | `integrity/declared_runs.py` | `tests/integrity/test_declared_runs.py`, `tests/posttraining/test_publication_from_aggregates.py` |
| Elección con la validación anterior a cada ventana, recibos de ventana, orden de los tramos y disjunción de la cadena por etapas | `integrity/window_selection.py` | `tests/integrity/test_window_selection.py`, `tests/posttraining/test_publication_from_aggregates.py` |
| Informe que bloquea la publicación | `integrity/publication_gate.py`, `training/campaign_publication.py` | `tests/integrity/test_publication_gate.py`, `tests/posttraining/test_publication_from_aggregates.py`, `tests/training/test_campaign_a_joint.py` |
| Registro de integridad en la fase de agregados del recorrido, de la base y de cada padre | `training/rolling_retention.py`, `posttraining/stage_comparison.py` | `tests/integrity/test_publication_gate.py`, `tests/training/test_rolling_retention.py`, `tests/posttraining/test_publication_from_aggregates.py` |

La descripción de cada comprobación está en el [banco de integridad](../../../docs/research/integrity-bench.md#informe-de-integridad-de-la-publicación).

## Mutación dirigida

Cada mutante sustituye una línea de un módulo del banco o de su conexión con la campaña, ejecuta con `pytest -x` las pruebas que lo persiguen y restaura el archivo. El [registro](mutation.json) guarda el texto original, el mutado, las pruebas y el resultado de cada uno.

- Primera ronda: 36 mutantes con las pruebas de los commits de funcionalidad. Mueren 30.
- Sobreviven 6, que señalan huecos reales de las pruebas: la huella de filas y el recálculo sin la exclusión del diseño conjunto (las pruebas solo cubrían el estudio US), los agregados sin el control cero, las filas quitadas que publica el informe sin cotejo con las recontadas, la publicación que no exige las métricas recalculadas y las selecciones de la cadena que se dan por confirmadas sin leerlas.
- Segunda ronda: con las pruebas añadidas en 964cdf0e mueren los 6.

Resultado: 36 de 36 mutantes detectados. Una puntuación de mutación no garantiza que no haya otros fallos. Indica que cada decisión mutada tiene al menos una prueba que la exige.

### Con la protección de aprendizaje vigente

Las pruebas de extremo a extremo (`tests/posttraining/test_publication_from_aggregates.py`, la publicación con cartera de `tests/training/test_rolling_retention.py` y las dos de integridad de `tests/training/test_campaign_a_joint.py`) recorren la campaña reducida, y la etapa de adaptadores y la regeneración solo se recorren rebajando la protección. Mientras rija el bloqueo se omiten y figuran en el inventario de [#546](https://github.com/GonxKZ/mars-titan/issues/546) (`tests/learning_hold_lowering.py`). En las dos primeras rondas, 16 mutantes de ejecuciones declaradas, selección y conexión con la publicación solo morían con ellas.

Para que el informe siga comprobado durante el bloqueo, `tests/integrity/test_declared_runs.py`, las pruebas nuevas de `tests/integrity/test_window_selection.py` y `tests/integrity/test_publication_gate.py` repiten esas comprobaciones sin ejecutar ninguna campaña. Usan recibos e informes escritos en la prueba, recibos de ventana del protocolo US v2 versionado y la etapa de `Stage`, con predicciones de valores conocidos. En la publicación, el recorrido y la etapa sustituyen solo las piezas costosas por resultados conocidos y mantienen real el orden entre agregados, registro de integridad y destino.

- Tercera ronda: los 36 mutantes con la protección real y solo con pruebas que no la rebajan. Mueren los 36. Sin mutantes, las 102 pruebas de `tests/integrity` pasan (`with_hold` en `mutation.json`).

## Coste medido en la forma de A v2

`measure_stage_aggregates.py --integrity` escribe los agregados de la comparación postentrenada con predicciones sintéticas de la forma real de `fold-018` de US+CN (la ventana con más filas: 1.027.173 de evaluación en US y 194.649 en China) y mide después `comparison_integrity.write_window`, que vuelve a leer todas las filas y recalcula sus puntuaciones por otro camino. Se han medido los tres primeros padres, sin estratos, con dos hilos y la CPU compartida con otras cargas (carga media de 9,5 a 14,5). Ninguna predicción sale de un modelo ([medida](integrity-fold-018.json)).

| Padre | Series | Agregados (s) | Integridad (s) | Integridad por serie (s) | Registro (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| rnn | 36 | 159,1 | 168,2 | 4,67 | 150.833 |
| lstm | 36 | 95,9 | 134,2 | 3,73 | 151.085 |
| gru | 36 | 104,8 | 133,7 | 3,71 | 150.833 |

La integridad cuesta algo más que los propios agregados, porque además de puntuar recorre las filas para la huella de identidad. Con 3,7 a 4,7 segundos por serie, las 9.828 series de la comparación postentrenada de A v2 (546 por ventana en 18 ventanas) suman de 10 a 13 horas de CPU. La comparación walk-forward de la campaña tiene 1.862 series (66 por ventana en US+CN durante 19 ventanas y 19 en US y en China, con sus modelos prestados, durante 19 y 13), que añaden de 1,9 a 2,4 horas. En total, de 12 a 15 horas de CPU. Es una cota superior por dos razones: `fold-018` es la ventana con más filas y antes de `fold-006` China no entra en las métricas. Los registros ocupan unos 151 KB por padre y ventana, unos 54 MB en todo el recorrido de A v2, despreciables frente a los agregados.

El trabajo es de CPU y ocurre en la fase de agregados, antes de liberar cada ventana. Si la retención v2 la recorre en serie con el ajuste, alarga el calendario en esas horas. No se ha perfilado qué parte corresponde a la lectura de filas y qué parte al recálculo. Antes de optimizarlo habría que medir ese reparto con la forma real.

## Límites y trabajo pendiente

- Siguen pendientes en #442, con su propia prueba de mutación y dentro del mismo informe: el presupuesto declarado frente al consumido por cada modelo, con las diferencias declaradas y la igualdad de actualizaciones de los controles emparejados, la abstención con su cobertura (curva riesgo-cobertura y valor a una cobertura fija declarada) y el ajuste del preprocesado solo con el pasado de cada corte.
- El escáner de fugas y la cota con información futura todavía no se aplican a las matrices y cintas reales de cada ventana.
- Las falsaciones que ajustan modelos siguen solo declaradas mientras dure el bloqueo de aprendizaje.
- La campaña no se ha ejecutado. No hay ninguna cifra de resultados.

## Reproducir

```bash
export PYTHONPATH=src
R=reports/engineering/campaign-publication-20261010
uv run --no-sync python $R/measure_stage_aggregates.py --root <carpeta nueva> \
  --window fold-018 --parents 3 --integrity \
  --output reports/engineering/publication-integrity-20261010/integrity-fold-018.json
```

La medida necesita unos 2 GB de memoria y 1 GB de disco temporal, que el script borra al terminar.
