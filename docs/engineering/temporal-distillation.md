# Destilación temporalmente válida

Autor: Gonzalo García Lama. Implementación del 13 de octubre de 2026 para MT-055 (#57).

Este documento describe el contraste de destilación de la ampliación: qué profesor y qué alumno se comparan, qué salidas del profesor puede usar el alumno y cuándo, cómo entran en su pérdida, qué se comprueba y qué se ha medido. La decisión de activarlo o descartarlo está en [el informe de destilación](../../reports/models/distillation.md). Nada se ha ajustado: el bloqueo de aprendizaje sigue vigente y no hay ningún resultado predictivo.

## Pregunta

La destilación desplaza cómputo de la inferencia al ajuste (Hinton et al., 2015). Un profesor caro deja sus salidas en las filas de ajuste y un alumno barato aprende a la vez de las etiquetas y de esas salidas. Solo tiene sentido si se cumplen dos cosas: el profesor predice mejor que el alumno entrenado directamente y su inferencia cuesta claramente más. Si falta la primera no hay nada que transferir. Si falta la segunda basta con usar el profesor.

## Profesor y alumno

La issue preveía un profesor con K = 4 y un alumno con K = 1. La medida del 10 de octubre en `cuda:0` (tabla de [coste de los modelos nuevos](mars-titan-extensions.md#coste-medido-de-los-modelos-nuevos)) ya mostraba que ese par apenas cambia el coste de inferencia: el lector M1 con K = 4 predijo 1.449 filas/s y con K = 1, 1.465, un 1 % de diferencia, porque los pasos de refinamiento son baratos frente a la preparación del núcleo y la búsqueda en el banco. Un alumno K = 1 destilado de K = 4 no ahorraría casi nada al predecir y costaría el ajuste del profesor y la generación de sus salidas. La medida del 13 de octubre sobre otra vista lo confirma: en las tres rondas el lector con K = 4 predijo 417, 421 y 278 filas/s y el de K = 1, 511, 298 y 364, así que K = 4 costó entre 0,71 y 1,31 veces lo que K = 1, dentro de la dispersión entre rondas. Ese par no se declara.

La [arquitectura de Titans-MAC](../research/titans-mac-architecture.md) (I15) fija otro par: profesor MARS-TITAN completo y alumno Titans-MAC. El profesor es `mars_titan_m1`, el lector episódico M1 con K = 1 sobre su padre `titans_mac_online`, y el alumno es el propio `titans_mac_online`. Los dos reciben las mismas entradas, así que el alumno no tiene menos información que el profesor, solo le falta el lector. El alumno directo es el mismo modelo sin destilación, con las mismas filas, ventanas, semillas, casos de búsqueda, actualizaciones y calibración.

## Qué salidas usa el alumno

Las salidas del profesor en su propio tramo de ajuste son objetivos derivados dentro de la muestra. Sus pesos ya han visto todas las etiquetas del tramo, así que no son predicciones históricas y nunca se presentan como tales. Se generan con una sola pasada cronológica del estado elegido del profesor sobre ese tramo, con banco vacío y memoria rápida inicial al empezar, como cualquier validación. Su memoria en cada fila solo contiene etiquetas maduras antes de la decisión, pero eso no las convierte en predicciones fuera de muestra.

`training/distillation.py` guarda esas salidas sin las etiquetas, con un manifiesto que declara `in_sample`, su único uso (`student_distillation_loss_only`), la memoria de la pasada, el profesor (modelo, ejecución, informe, estado elegido y época), el tramo (corpus, índice, representación, política de entrada y fase), la cabeza y el corte. Al cargarlas, `TeacherTargets` rechaza un archivo de otra vista, representación, fase, cabeza, memoria o profesor aunque coincidan las dimensiones, y uno cuyo contenido no conserve su huella. El profesor tiene que ser una ejecución completada, sin la reserva final abierta y ajustada con el mismo tramo que el alumno, y solo emite las salidas un entrenador con su identidad después de cargar su estado elegido.

Las salidas solo entran en la pérdida del alumno. No sustituyen a ningún error emitido, no escriben en un banco ni enseñan una puerta, y la validación, las predicciones emitidas y las métricas del alumno son las mismas con profesor o sin él. Si algún día se quisieran usar para aprender una escritura o una puerta, harían falta salidas generadas con cortes históricos válidos, y este archivo no lo es.

## Objetivo del alumno

En cada tramo con etiquetas maduras, el alumno minimiza

$$(1 - w)\,\ell_{\text{tarea}} + w\,\frac{1}{n\,d}\sum_{i,j} \lvert \hat y_{ij} - t_{ij} \rvert,$$

con $\ell_{\text{tarea}}$ la pérdida de su receta (pinball con la cabeza de cuantiles, MAE con la escalar), $\hat y$ sus salidas, $t$ las del profesor en las mismas filas y $d$ el número de salidas por fila (cinco cuantiles o un punto). La distancia L1 sobre cada salida numérica tiene la escala del MAE y no trata los cuantiles como una densidad. La receta declara `distillation_weight` y su identidad añade el peso y la distancia. Sin peso, la identidad y el cálculo no cambian. En el recorrido por bloques de flujos (`accumulation_rows`), cada bloque pondera su objetivo por su fracción de etiquetas, como la tarea, así que el gradiente es el del tramo completo. La pérdida que informa el ajuste sigue siendo la de la tarea.

El peso se elige por ventana entre 0,25 y 0,5 con la validación de la ventana y la métrica de la receta, como los casos de búsqueda. El alumno conserva su propia calibración con su tramo de calibración. La selección del profesor también usó esa validación, así que la comparación entre alumnos no se hace sobre ella sino sobre la evaluación de las ventanas walk-forward.

## Declaración y regla

[`configs/evaluation/temporal-distillation.json`](../../configs/evaluation/temporal-distillation.json) fija antes de ejecutar nada profesor, alumno, pesos, uso de las salidas, distancia, igualdades, métrica y regla. La métrica primaria es la diferencia relativa del MAE por sesión entre el alumno destilado y el directo, emparejada, con bootstrap circular por bloques de 16 sesiones. Se informa también qué parte de la ganancia del profesor conserva el alumno. La extensión solo se activa si el profesor mejora al alumno directo en la validación de la campaña A con el límite superior del intervalo del 95 % por debajo de cero y si su inferencia cuesta al menos 1,5 veces la del alumno con el código vigente. El alumno destilado se conserva si mejora al directo con el límite superior por debajo de cero y el ahorro de inferencia compensa el coste total. Se informan los dos pesos, el alumno directo y el profesor, también si el alumno directo gana en error o en latencia. La declaración guarda también la condición de coste con su recibo y qué partidas cuenta el presupuesto, que se fija en horas antes de generar ningún objetivo. `python -m mars_titan.training.distillation check` comprueba la declaración sin generar ningún objetivo.

## Comprobaciones

`tests/training/test_distillation.py` usa el corpus técnico de los entrenadores cronológicos. El profesor es otro Titans-MAC, con otra semilla, recorrido hasta el paso con el optimizador que solo registra gradientes, así que su informe es el de una ejecución completada con sus pesos iniciales. Las pruebas comprueban que las salidas guardadas son exactamente las que emite el profesor al recorrer de nuevo el tramo, sin etiquetas y dentro de sus decisiones. Rechazan diez manipulaciones del manifiesto o del archivo, otra cabeza, la fase de validación y otro corpus con las mismas dimensiones, filas cruzadas, tardías, repetidas o sin cuantiles, y profesores sin completar, con la reserva abierta, de otra fase o emitidos por otro entrenador. En el ajuste comprueban que peso y profesor van juntos y ligados al tramo, que la identidad los registra, que cada paso compara sus propias filas maduras con las del profesor y con lo que emitió el alumno, que el objetivo es afín en el peso con una distancia fija (con dos pesos, la distancia que se deduce del gradiente es la misma), que la acumulación por bloques da el mismo gradiente, que la falta de una fila del profesor detiene el paso y que el profesor no cambia predicciones emitidas, etiquetas, tramos, validación ni métricas. La declaración y sus nueve rechazos también tienen prueba, y otra liga la condición de coste a los caudales del recibo.

La mutación dirigida sobre `training/distillation.py` y la parte de destilación de `training/financial_run.py` ([informe](../../reports/engineering/temporal-distillation-mutations-20261013.json)) introdujo 30 defectos: pesos de tarea y distancia intercambiados, distancia con signo, comprobaciones del manifiesto omitidas, salidas emitidas por otra ejecución o desde otra carpeta, salidas sin cargar el estado elegido, etiquetas guardadas como salidas, identidad sin la destilación y la destilación omitida en el tramo completo, en la acumulación o con la penalización C, entre otros. Las pruebas detectan los 30.

## Coste

Medido el 13 de octubre de 2026 en `cuda:0` con `benchmarks/temporal_distillation_cost.py` ([recibo](../../reports/engineering/temporal-distillation-cost-20261013.json)). Cada ronda recorre, sobre la ventana `fold-012` de las vistas de 64 activos de EE. UU. que prepara la medida de #459, el alumno `titans_mac_online` y los lectores `mars_titan_m1` y `mars_titan_m1_k4` sobre ese mismo padre con pesos iniciales, con las funciones de la orden de caudal: 160 tramos de ajuste hasta el paso tras 10 de calentamiento y 512 eventos de inferencia tras 32, con 64 entradas como mucho por evento. Ningún peso cambia, porque el optimizador de la medida no actualiza y el gancho global prohíbe pasos. FP32 estricto sin TF32. Durante las tres rondas había pruebas de CPU de otros trabajos en el equipo, con una carga media de unos 25 observada al final de la medida.

| Ronda | Alumno (filas/s) | Profesor M1 (filas/s) | M1 con K = 4 (filas/s) | Profesor frente al alumno | K = 4 frente al alumno |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.474 | 511 | 417 | 4,84 | 5,93 |
| 2 | 2.014 | 298 | 421 | 6,75 | 4,78 |
| 3 | 1.678 | 364 | 278 | 4,61 | 6,03 |

Con la ronda menos favorable, el profesor tarda 4,6 veces lo que el alumno en predecir, por encima del umbral de 1,5 de la declaración. La medida del 10 de octubre sobre la vista conjunta US+CN, con hasta 4.089 entradas por evento y el código anterior a #446 y #482, dio 1,78 para el mismo par (2.607 frente a 1.465 filas/s). Una explicación probable es que el trabajo por evento del lector, la consulta y la escritura del banco, pesa menos frente al núcleo cuando la cohorte es grande, pero no se ha medido por separado. La carga de CPU también puede afectar más al lector que al núcleo. Por eso la declaración exige repetir la medida en la vista de la campaña A con el código vigente antes de activar la extensión, donde el margen sobre 1,5 puede ser pequeño.

Generar las salidas es una pasada de inferencia del profesor por el tramo de ajuste. Con la mediana medida, 364 filas/s, cada millón de filas cuesta unas 0,76 h de GPU en esta vista. A eso se suman dos ajustes del alumno por ventana y semilla, uno por peso, con el coste de un ajuste del alumno directo, y su calibración. La distancia añade al paso del alumno una resta y un valor absoluto sobre sus salidas, sin medir aparte. El ajuste del profesor ya forma parte de la campaña A.

## Revisión adversarial

- **Fuga del profesor.** El profesor solo se ajusta con el tramo de ajuste de la ventana y su estado se elige con la validación de esa ventana. Ninguna salida es de validación, calibración, evaluación ni de 2024. Su elección con la validación sí llega al alumno destilado a través de las salidas, por eso los alumnos se comparan sobre la evaluación de las ventanas y no sobre la validación.
- **Predicciones reconstruidas.** Las salidas se calculan después de elegir el estado del profesor. Por eso se declaran dentro de la muestra y solo entran en la pérdida, y el manifiesto lo deja escrito.
- **Más información.** El alumno no recibe ninguna entrada que no tenga el directo. Lo único nuevo son las salidas del profesor en las filas de ajuste.
- **Errores y exceso de confianza del profesor.** El alumno puede copiarlos. Por eso conserva la tarea con peso 1 − w, mantiene su propia calibración y se compara con el directo, que no los hereda.
- **Ganancia de cómputo.** Si el profesor apenas cuesta más que el alumno al predecir, la destilación solo añade coste. Es lo que ocurre con K = 4 frente a K = 1 y por eso ese par no se declara.
- **Alternativa más sencilla.** Usar directamente el profesor. La destilación solo se defiende si el ahorro de inferencia compensa el ajuste del profesor, la generación de sus salidas y los ajustes adicionales del alumno.

## Pendiente

- Ajustar el profesor y los alumnos en la campaña A cuando se levante el bloqueo y el resultado del profesor frente al alumno directo cumpla la condición de activación.
- Conectar la generación de salidas por ventana y el ajuste del alumno destilado en la ventana walk-forward de Titans-MAC (`titans_walk_forward`). Solo se hará si la extensión se activa.

## Referencias

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the knowledge in a neural network* [Preprint]. arXiv. https://arxiv.org/abs/1503.02531
