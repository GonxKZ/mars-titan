# Repaso de filas anteriores en la continuación por etapas

Autor: Gonzalo García Lama. Implementación del 13 de octubre de 2026 para MT-053 (#55).

Este documento describe los tres controles de repaso de la matriz de adaptadores, dónde se insertan en el walk-forward por etapas, qué filas pueden repasar, cómo se igualan sus actualizaciones y qué control permite descartarlos. La decisión de no activarlos todavía en la campaña A está en [la decisión de consolidación](../../reports/memory/consolidation-decision.md). Nada de lo que sigue se ha ejecutado con aprendizaje: el bloqueo sigue vigente y no hay ningún resultado predictivo.

## Pregunta

En la ventana k la continuación completa ajusta todos los pesos del padre de k-1 solo con las filas nuevas, unos meses de datos después de veinte años de ajuste. Si esas actualizaciones borran parte de lo aprendido, volver a ver filas antiguas durante la continuación debería ayudar. Es la forma más directa de consolidación sin otra regularización (Rolnick et al., 2019, Chaudhry et al., 2019). La continuación anclada de #444 aborda el mismo riesgo desde los pesos. El repaso lo aborda desde los datos.

La pregunta es si el repaso mejora la validación de la ventana frente a la misma continuación con el mismo número de actualizaciones sobre las filas nuevas. Un repaso que solo gane a la continuación completa no responde: hace un 50 % más de actualizaciones.

## Dónde se inserta

La issue preveía un módulo `training/replay.py` y un comentario de #366 señalaba el entrenador cronológico de Titans como punto de inserción. Se ha implementado en la continuación de las referencias neuronales de la matriz, por dos motivos:

- Repasar una sesión antigua en un entrenador cronológico exige reconstruir el estado de sus flujos con su calentamiento propio. Sin esa reconstrucción, la memoria neuronal y el banco episódico verían la sesión con el estado de otra fecha. Con ella, cada sesión repasada cuesta su calentamiento completo. Las familias cronológicas quedan fuera y el motivo queda en el código.
- En las referencias cada sesión es una cohorte independiente: el modelo no lleva estado entre sesiones, así que repasar una sesión es exactamente volver a ver sus filas.

El `ReplaySchedule` en C++20 de #366 (calendarios `uniform`, `recent` y `spaced` con el mismo multiconjunto) sigue sin enlace con un entrenador. Aquí no hace falta: el plan de cada caso cabe en una lista de sesiones y lotes, y el orden lo fija NumPy con su propio generador.

## Qué filas se repasan

Se repasan entradas reales y no representaciones guardadas. El modelo vuelve a codificar cada sesión con sus pesos del momento, así que no hay claves ni valores obsoletos que reconstruir, el problema que plantea el repaso latente (Pellegrini et al., 2020). Una sesión anterior entra en el repaso si cumple las dos condiciones:

1. Todas sus filas del tramo de ajuste de la vista k son filas que el padre ajustó: mismo activo y misma fila de su archivo de muestras en el tramo `train` de la vista k-1. Así el repaso nunca añade filas de la validación o la calibración del padre, que el ajuste por etapas deja fuera de la continuación. Con una frontera solo temporal, la continuación con repaso habría visto nueve meses de filas recientes que la continuación completa no ve y la comparación ya no sería sobre el repaso.
2. Todas sus etiquetas maduran antes de la primera decisión nueva (`since`, el final de la calibración del padre). La lectura de cada cohorte lo vuelve a comprobar con sus propias etiquetas.

`replay_continuation.earlier_sessions` cuenta aparte las sesiones excluidas por cada condición, que quedan en la identidad del caso.

## Los tres controles

La sección opcional `replay_continuation` de la matriz v3 declara la proporción `r` de actualizaciones de repaso por actualización de filas nuevas y los ámbitos donde se propone. Con `U` actualizaciones de filas nuevas por época, cada control añade `E = ⌈r · U⌉` actualizaciones por época. Los tres tienen el mismo objetivo, presupuesto, λ, semilla y selección que la continuación completa, y entre ellos las mismas actualizaciones:

| Control | Fuente de las visitas adicionales | Selección |
| --- | --- | --- |
| `replay_uniform` | Sesiones anteriores admitidas | Permutación con el generador `[semilla, 5501]` |
| `replay_regime_balanced` | Sesiones anteriores admitidas | Una sesión de cada régimen por turno, cada régimen permutado con el mismo generador |
| `repeat_new_rows` | Sesiones de las filas nuevas | Permutación con el mismo generador |

El régimen de una sesión es la ruta de `observable_volatility_trend_v1` de #523 aplicada a la serie del factor de mercado que declara la vista (SPY en EE. UU.), con las últimas 64 sesiones cuyo cierre se conocía en la decisión. La regla original resume un mercado con la mediana de sus activos. Aquí el mercado es su factor, así que se usa con un activo como mínimo y el resto de la regla no cambia. Una sesión con filas de dos mercados o de un mercado sin factor queda sin clasificar y forma su propio grupo. El reparto por régimen da más peso a los regímenes poco frecuentes, que son los que más se olvidarían si solo se ajusta con los meses recientes.

El plan de cada caso se fija antes del primer ajuste: el orden de las candidatas y, sobre él, las sesiones y lotes hasta sumar exactamente `E` (la última sesión puede quedar recortada a sus primeros lotes, en el orden de sus activos). En cada época las visitas nuevas conservan el orden de la continuación completa y las visitas de repaso ocupan posiciones al azar con el generador `[semilla, época, 5502]`. El repaso de sesiones anteriores abre una fuente de la vista solo con las sesiones planificadas. Si caben en el presupuesto de bloque, se leen una vez y sirven para todas las épocas.

La normalización y la rejilla siguen siendo las de las filas nuevas, que son las filas reales del ajuste, y la prueba de filas nuevas de la etapa no cambia. El recibo de cada caso declara aparte el repaso con su plan, sus filas y sus actualizaciones.

## Control que descarta el repaso

`repeat_new_rows` hace las mismas actualizaciones adicionales sobre las filas nuevas. Si iguala a los dos repasos en validación, la ganancia viene de hacer más actualizaciones y no de volver a ver el pasado, y el repaso se descarta. La comparación de la etapa lo declara como familia `versus_repeat_new_rows`, con la repetición como base y los dos repasos como variantes, con la misma corrección por máximo estudentizado que el resto de familias. El reparto por régimen se descarta frente al uniforme con la misma lógica: si no lo mejora, la diversidad no aporta sobre el muestreo proporcional al tiempo.

En la cadena de la etapa los tres controles compiten como candidatos de tipo continuación y solo se eligen con la validación de la ventana, como los demás. Su presupuesto mayor no se iguala allí, porque la cadena no es un contraste emparejado. Se informa de todos los controles declarados, también de los que empeoren.

## Implementación

| Pieza | Función |
| --- | --- |
| `posttraining/replay_continuation.py` | Sección, controles, casos, proporción, sesiones admitidas, régimen por factor, orden, plan y posiciones |
| `posttraining/replay_inputs.py` | `ReplayInputs`, que añade las visitas al recorrido de `PairedInputs` y cuenta filas y actualizaciones |
| `posttraining/inputs.py` | Lectura de cada visita desde su fuente, con límite de filas y comprobación adicional por visita |
| `environments/view_cohorts.py` | `ViewCohortSource(sessions=...)`, fuente limitada a las sesiones enumeradas |
| `posttraining/matrix_runs.py` | `MatrixWindow.earlier` y `open_sessions`, y `MatrixParent.inputs`, que construye las cohortes de cada caso |
| `posttraining/adapter_matrix.py` | Casos de los tres controles y su presupuesto propio en el plan |
| `posttraining/run.py` | Validación del caso, huellas del código y comprobación de que las cohortes llevan el repaso del caso |
| `posttraining/campaign_stage.py` | Activación por `additional_controls`, solo en la variante A, y repaso en el recibo |
| `posttraining/stage_comparison.py` | Un papel propio para cada control |
| `training/campaign_throughput.py` | Filas de repaso en la estimación de horas |

Una etapa solo ejecuta los controles si nombra la sección en `additional_controls`. La matriz v3 los propone para las referencias, pero las etapas A y A v2 no los nombran, así que su plan no cambia.

## Comprobaciones

`tests/posttraining/test_replay_continuation.py` reúne las pruebas. Las puras fijan la sección y sus rechazos, los casos de cada familia, el presupuesto propio de los tres controles en el plan, la validación del caso, el orden por semilla con los generadores declarados, el recuento de actualizaciones por sesión, el reparto por régimen, el plan con las actualizaciones exactas, las posiciones y que el régimen de una decisión no cambie al modificar cierres posteriores. Las de datos usan las dos ventanas US de la campaña reducida y comparan con una lectura propia de las etiquetas las sesiones admitidas (las de la validación y la calibración del padre quedan fuera, y también una sesión a la que se quita una fila ajustada por el padre), los lotes de cada época (los de las filas nuevas idénticos a los de la continuación completa y los de repaso con las sesiones del plan, y la última sesión recortada a sus primeras filas cuando el plan no la cubre entera), las identidades, la reanudación desde un cursor dentro de la época, el rechazo de una sesión que madura tarde y el de una fuente con otras sesiones. El recorrido de la etapa A con los tres controles comprueba con el optimizador que no cambia pesos que cada caso da exactamente sus actualizaciones, que el recibo declara el repaso aparte y que la cadena los trata como continuaciones. Esas ventanas no tienen factor de mercado, todas sus sesiones quedan sin clasificar y el plan por régimen coincide con el uniforme, así que el reparto con varias rutas solo lo comprueban las pruebas puras. `tests/training/test_campaign_throughput.py` comprueba las filas de repaso en la estimación de horas.

La mutación dirigida sobre `replay_continuation.py`, `replay_inputs.py`, `inputs.py`, `matrix_runs.py`, `view_cohorts.py`, `adapter_matrix.py`, `run.py`, `campaign_stage.py`, `stage_comparison.py` y `campaign_throughput.py` ([informe](../../reports/engineering/replay-continuation-mutations-20261013.json)) introdujo 39 defectos y las pruebas detectan los 39. En una primera tanda sobrevivieron seis: el orden con el generador de las posiciones, la admisión de una sesión con alguna fila ajustada en lugar de todas, las actualizaciones nuevas contadas por filas y no por sesiones, la visita recortada leída entera (en el plan y en la lectura) y las rutas en la identidad de un control sin reparto por régimen. Las pruebas que los detectan son las de los generadores declarados, la sesión con una fila que el padre no ajustó, el recuento por sesión, la sesión recortada con lotes de una fila y la identidad del repaso uniforme.

## Coste

Medido el 13 de octubre de 2026 en CPU con `benchmarks/replay_continuation_cost.py` ([recibo](../../reports/engineering/replay-continuation-cost-20261013.json)), sobre las ventanas `fold-011` (padre) y `fold-012` de las vistas de 64 activos de EE. UU. que prepara la medida de #459, con r = 0,5, semilla 42 y lotes de 256 filas. La medida prepara y lee una época de cada control con un padre nulo, sin modelos, optimizadores, pérdidas ni errores. Mide la preparación y la lectura de datos, no el coste de las actualizaciones adicionales.

- La ventana tiene 3.840 filas nuevas en 60 lotes, uno por sesión. Cada control añade 30 actualizaciones y 1.920 filas por época, un 50 % más que la continuación completa.
- De las sesiones anteriores, 3.708 cumplen las dos condiciones (237.312 filas) y 192 quedan fuera porque el padre no ajustó todas sus filas. Ninguna queda fuera por madurar tarde. Por régimen hay 1.521, 538, 834 y 815 sesiones en las rutas 1 a 4. Calcular las sesiones admitidas y sus rutas tarda 1,1 s, y abrir la ventana, 12,9 s.
- El plan de cada caso tarda unos 0,04 s con repaso y menos de 1 ms con repetición.
- Una época de la continuación completa tarda 0,35 s. Con repaso, la primera época tarda 5,7 s (uniforme) y 7,3 s (por régimen), porque abre la fuente de las sesiones planificadas y las lee. Las siguientes tardan 0,50 y 0,61 s, con las sesiones ya residentes (7,5 MB). La repetición tarda 0,64 y 0,65 s.

El coste que importa son las actualizaciones adicionales, un 50 % más de pasos por época en cada control. `training/campaign_throughput.py` suma las filas de repaso a la estimación de horas. Con los recuentos verificados de A v2, 10 épocas efectivas y 16.000 muestras-época por segundo, nombrar la sección en la etapa A v2 añadía 810 ajustes (de 7.074 a 7.884), subía los adaptadores de 118 a 156 horas de GPU y llevaba el factor de caudal necesario para 580 h útiles con 250 h fijas de 7,90 a 8,01. Con los objetivos de #128 ya declarados en A v2, los mismos 810 ajustes irían de 8.694 a 9.504, los adaptadores de 180 a 218 horas y el factor de 8,09 a 8,20. Para la etapa A no hay informe de recuentos y no se ha estimado.

## Revisión adversarial

- **Más actualizaciones.** Es la explicación alternativa principal y la cubre `repeat_new_rows` con las mismas actualizaciones.
- **Más datos recientes.** Una frontera solo temporal habría metido en el repaso la validación y la calibración del padre. La admisión por identidad de fila lo impide y la prueba lo comprueba con las vistas reales de la campaña reducida.
- **Etiquetas del futuro.** Las sesiones repasadas maduran antes de la primera fila nueva, que a su vez madura antes de la validación de la ventana. La selección sigue siendo solo con esa validación.
- **Patrones obsoletos.** El repaso puede conservar relaciones que ya no se cumplen. Por eso no se activa sin evidencia de olvido (MT-052) y por eso el reparto por régimen se contrasta con el uniforme y no se presupone mejor.
- **Representaciones obsoletas.** No existen: se repasan entradas reales y el padre recodifica cada sesión.
- **Estado entre épocas.** Las referencias no llevan estado entre cohortes y cada visita reinicia su lectura, así que el final de una época no contamina el inicio de la siguiente.
- **Calendario del repaso.** Los tres controles fijan las mismas posiciones por época. El efecto del orden temporal del repaso, con el mismo multiconjunto y calendarios distintos, no se estudia aquí.
- **Memoria y repaso a la vez.** La combinación con la continuación anclada o con la memoria de Titans no se declara. Primero tiene que haber evidencia de que alguna de las dos sola aporta.

## Referencias

- Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). *On tiny episodic memories in continual learning* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1902.10486
- Pellegrini, L., Graffieti, G., Lomonaco, V., & Maltoni, D. (2020). *Latent replay for real-time continual learning* [Preprint]. arXiv. https://arxiv.org/abs/1912.01100v2
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. En *Advances in Neural Information Processing Systems* (Vol. 32). https://arxiv.org/abs/1811.11682v2
