# Abstención con su cobertura en la publicación (13 de octubre de 2026)

Este informe acompaña al tercer bloque de [#442](https://github.com/GonxKZ/mars-titan/issues/442). Recoge qué reglas de abstención publican la comparación walk-forward y la matriz, cómo las repite el informe de integridad que bloquea la publicación, cómo se ha comprobado que detecta los cambios que persigue y cuánto cuesta en la forma real de A v2. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto.

## Por qué hace falta

Una predicción puntual cero es una abstención. La exactitud de dirección ya la cuenta como fallo, pero las precisiones de subida y de bajada solo juzgan las llamadas. Un modelo que llama una vez por sesión, en la fila más clara, puede tener precisión perfecta. Desde la versión 4 de la comparación esas precisiones son series contrastables, así que ese modelo podía ganar un contraste publicado sin predecir casi nada.

## Reglas declaradas

Están en `evaluation/abstention_coverage.py`, fechadas el 13 de octubre de 2026, antes de evaluar ninguna campaña. Cada comparación walk-forward y cada matriz las publican en `abstention`. El detalle está en [métricas](../../../docs/research/metrics.md#abstención-con-su-cobertura).

| Regla | Valor |
| --- | --- |
| Curva riesgo-cobertura de dirección | En cada sesión, exactitud entre las ceil(c · n) filas con objetivo no nulo y mayor \|predicción\|, con c = 0,1, 0,2, …, 1. Empates por la identidad de fila `mercado/activo/instante` y la predicción cero cuenta como fallo |
| Cobertura fija | 0,5 (`direction_accuracy@0.5`) |
| Cobertura de llamadas | Llamadas sobre filas con objetivo no nulo, sumadas sobre todas las sesiones de la vista y las semillas del modelo |
| Cobertura mínima | 0,5. Un contraste de `up_precision` o `down_precision` con algún modelo por debajo se publica con su cobertura y marcado no elegible |

Con c = 1 la curva es la exactitud de dirección. Como abstenerse nunca acierta, un modelo que se abstiene no mejora su exactitud a ninguna cobertura. El MAE y el MSE puntúan todas las filas, y abstenerse en todas equivale al control cero, que ya se compara de forma explícita.

La cobertura mínima de 0,5 es una decisión de diseño que conviene que Gonzalo revise antes de lanzar la campaña. Un valor más bajo deja competir a modelos que llaman en menos filas, y uno más alto puede dejar fuera de las precisiones a modelos que predicen cero con frecuencia por diseño.

## Qué comprueba el informe de integridad

`integrity/abstention_marks.py` lee la tabla de sesiones publicada y no usa el código de la comparación:

1. recalcula la cobertura de llamadas de cada modelo y vista y exige que coincida con la publicada;
2. exige las reglas declaradas en la comparación y en la matriz;
3. repite la marca y la cobertura de cada contraste condicionado a las llamadas, en la comparación y en cada vista de la matriz;
4. comprueba que cada punto de la curva de cada semilla y vista publique las filas retenidas y los aciertos que suman sus sesiones.

La segunda implementación de las métricas (`integrity/independent_scores.py`) recalcula los aciertos de cada cobertura con su propia ordenación por identidad de fila, y el recálculo los coteja sesión a sesión con los publicados. La puerta de la publicación añade la matriz que lee cada comparación walk-forward.

## Prueba con un modelo que casi siempre se abstiene

En el estudio de juguete de la comparación walk-forward, Ridge predice cero salvo en la fila de mayor objetivo positivo de cada sesión, donde acierta. Su precisión al alza es 1,0. La prueba comprueba que:
- su cobertura de llamadas queda por debajo de 0,5 en las tres vistas, y la del control cero es 0;
- todos sus contrastes de precisión salen no elegibles en la comparación y en la matriz, y el contraste entre los dos modelos que llaman sigue elegible;
- su exactitud de dirección y su exactitud a la cobertura fija, con las mismas filas retenidas, quedan por debajo de las de la GRU en las dos semillas;
- el control cero no acierta a ninguna cobertura.

## Mutación dirigida

Se aplicaron 33 mutantes dirigidos, uno a uno, y cada uno ejecuta las pruebas que deberían detectarlo con `pytest -x` ([registro](mutation.json)). Mueren los 33. Por módulo:

| Módulo | Mutantes | Qué rompen |
| --- | --- | --- |
| `evaluation/forecast_scores.py` | 6 | Orden por predicción con signo, filas sin juzgar retenidas, una sesión sin filas juzgables que retiene una, aciertos o filas sin recortar a la cobertura y tabla sin aciertos selectivos |
| `evaluation/abstention_coverage.py` | 4 | Cobertura mínima estricta, contraste siempre elegible, marca en todas las métricas y llamadas por encima de las filas admitidas |
| `evaluation/walk_forward_comparison.py`, `session_table_contrasts.py` y `comparison_matrix.py` | 5 | Cobertura de una sola semilla, comparación o matriz sin marcas, informe sin cobertura de llamadas y matriz sin reglas |
| `integrity/abstention_marks.py` | 7 | Marca, cobertura del contraste, cobertura publicada, reglas, filas y aciertos de la curva sin cotejar, y cobertura recalculada fija |
| `integrity/independent_scores.py`, `score_recheck.py` y `comparison_integrity.py` | 9 | En la curva independiente, empates por orden de lectura o sin la \|predicción\|, aciertos que no siguen a su fila al desempatar, filas retenidas por defecto, filas sin juzgar y sesiones sin filas juzgables con aciertos. Además, cotejo ciego de los aciertos selectivos, recálculo sin identidad de fila e integridad sin la abstención |
| `integrity/publication_gate.py` y `training/campaign_publication.py` | 2 | Puerta y publicación sin la matriz de cada comparación |

Las pruebas que los detectan (`tests/evaluation/test_abstention_coverage.py`, `tests/integrity/test_abstention_marks.py`, `tests/integrity/test_independent_scores.py` y `tests/integrity/test_publication_gate.py`) corren con la protección de aprendizaje vigente.

## Coste medido en la forma de A v2

`measure_abstention_cost.py` construye con `measure_stage_aggregates.table` las predicciones de evaluación de una serie con la forma real de `fold-018` de US+CN, la ventana con más filas (1.027.173 en US y 194.649 en China, 490 sesiones y hasta 4.126 activos por sesión). Son sintéticas y no salen de ningún modelo. Antes de medir, el recálculo del informe coteja las dos implementaciones sesión a sesión en las 35 columnas, también los aciertos de cada cobertura, y coinciden. Después alterna cinco veces cada variante con dos hilos y la CPU compartida con otras cargas (carga media de 14 a 19), así que las medianas tienen ruido de algunas décimas.

| Camino | Sin aciertos selectivos | Con aciertos selectivos | Añadido |
| --- | --- | --- | --- |
| `score_sessions` y tabla (agregados y comparación) | 0,780 s | 1,039 s | 0,259 s (33 %) |
| Recálculo independiente, primera versión | 3,218 s | 5,607 s | 2,389 s (74 %) |
| Recálculo independiente, versión final | 3,798 s | 4,023 s | 0,225 s (6 %) |

En la primera versión el recálculo independiente construía la identidad de fila `mercado/activo/instante` de todas las filas y ordenaba por ella con pandas. El perfil de una llamada mostró 1,43 s en los aciertos selectivos, de los que 1,30 s eran esa ordenación (0,98 s en factorizar 1,2 millones de textos). La versión final ordena por mercado, instante y |predicción| con claves numéricas y solo construye la identidad para los tramos empatados en esas claves con aciertos distintos, que son los únicos cuyo desempate cambia el recuento. Las predicciones cero empatadas nunca aciertan y no la necesitan. Una prueba con activos cuyo orden por la identidad completa no es el del activo solo («A-B», «A/0», «A» y «AB») comprueba que coincide con el código de la comparación. El pico de memoria de la medida bajó de 2,11 a 1,89 GiB ([primera medida](cost-fold-018-identity-sort.json) y [final](cost-fold-018.json)).

A v2 puntúa 11.690 series en todo el recorrido, contando cada modelo, semilla y ventana (9.828 de la comparación postentrenada y 1.862 de la comparación walk-forward, según el [informe de integridad de la publicación](../publication-integrity-20261010/README.md)). Con el coste de `fold-018`, que es una cota superior, los aciertos selectivos añaden unos 50 minutos de CPU a la fase de agregados y unos 44 minutos a los registros de integridad de las ventanas, frente a 7,8 horas con la primera versión. En el recálculo independiente lo más costoso sigue siendo la probabilidad implícita de subida, que recorre las filas una a una por diseño (2,34 s del perfil). Es anterior a este bloque y no se ha cambiado. En el camino de la comparación no se ha perfilado el reparto de los 0,26 s añadidos entre la ordenación y los recuentos por cobertura.

## Límites

- La curva ordena por \|predicción\|, la única confianza que expresan todos los modelos, también los puntuales. `forecast_scores.selective_risk` calcula otra curva por la anchura del intervalo de los modelos con cuantiles, pero la comparación todavía no la publica.
- La campaña no se ha ejecutado. No hay ninguna cifra de resultados.
