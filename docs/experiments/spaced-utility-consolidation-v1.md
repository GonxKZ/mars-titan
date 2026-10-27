# Evidencia de utilidad espaciada para la consolidación E2

El registro `SpacedUtilityLedger` y el productor de supresiones del lector estándar K1 están implementados y comprobados con fixtures CPU. Conservan pronósticos emitidos y calculan evidencia por episodio cuando madura el resultado. Todavía no deciden qué episodios conserva el banco ni están conectados a la sesión financiera. E2 sigue parcialmente implementada y no cambia las predicciones de las referencias por defecto.

E2 estudia si conviene conservar episodios que han ayudado en decisiones separadas. No es una técnica de regularización de los parámetros del predictor y no demuestra una reducción del sobreajuste. La comparación predictiva permanece sin ejecutar.

## Contraste y agregación

Para una predicción original emitida \(p_i\), una predicción diagnóstica \(p_{i,-e}\) emitida sin el episodio \(e\) y su resultado maduro \(y_i\), se define:

\[
g_{i,e}=|y_i-p_{i,-e}|-|y_i-p_i|.
\]

Un valor positivo indica que el pronóstico original tuvo menor error. El contraste para K = 1 elimina el episodio del soporte ya seleccionado, renormaliza sus pesos y no introduce otro vecino. No equivale a quitarlo de todo el banco y repetir la búsqueda. El productor utiliza una sola preparación del padre y una sola selección de vecinos. El registro recibe sus números, sin repetir la inferencia cuando llega la etiqueta. La conexión futura a la sesión debe conciliar esos números con su emisión efectiva, porque el registro por sí solo no acredita esa procedencia.

Dentro de cada mercado e instante se promedian las filas que utilizaron ese episodio. Después se actualiza la media de sus sesiones con evidencia. Varias acciones de una sesión no crean varias confirmaciones. Esta media condicional al soporte utilizado no es el MAE global del proyecto. Los episodios no utilizados carecen de evidencia y no reciben un cero inventado.

La calificación exige una media superior a `positive_gain_margin` y ganancias positivas en dos días UTC separados al menos por `min_gap_days`. Se guardan los extremos de esos días, de modo que un resultado antiguo que llegue tarde conserve su fecha de decisión. Dos mercados del mismo día no acreditan dos días distintos. No se afirma independencia estadística. La implementación admite exactamente este criterio de dos días extremos, no una regla de tres o más confirmaciones.

Una nueva ganancia negativa puede retirar la calificación. La media no tiene decaimiento y puede conservar evidencia obsoleta cuando cambia el mercado. El margen cero y un día de separación son elecciones iniciales del núcleo técnico, no valores validados ni constantes de la neurociencia.

La utilidad por supresión no es aditiva. Si dos episodios equivalentes permiten acertar indistintamente, quitar cualquiera de ellos puede dar ganancia cero y quitar ambos puede empeorar. Tampoco mide una contribución de Shapley ni una causa económica. La futura selección debe contrastarse con retención original, uniforme, reciente y utilidad aleatoria, a igual capacidad y calendario. No se ha implementado todavía esa comparación.

## Orden temporal y estado

`begin(cutoff)` abre un evento posterior al último publicado. `issue(snapshot, rows)` registra una cohorte completa de un mercado antes de recibir resultados. Todas las cohortes del evento comparten el mismo `EpisodeSnapshot`, con codec, corte, huella y episodios maduros comprobados. `maturity_at` es la maduración financiera del episodio. `available_at` describe la disponibilidad de sus entradas y no la sustituye.

Cada fila contiene flujo, pronóstico original, pares de episodio y pronóstico diagnóstico y `label_not_before`. No admite un objetivo financiero en la emisión. `receive` exige `decision_at < label_not_before <= available_at <= cutoff`. Si el calendario todavía no acredita una maduración, `label_not_before=None` solo permite censurar la fila. Una censura no aporta una etiqueta.

`publish()` espera a resolver o censurar todas las filas de cada cohorte. Prepara las actualizaciones antes de confirmarlas y publica primero por fecha de decisión y después por mercado. Las filas y referencias se procesan con orden canónico. La cola pendiente queda separada de las estadísticas recuperables. La llegada de resultados de un episodio descartado genera su recibo, pero no lo vuelve a insertar en el registro.

`scores()` devuelve copias de las estadísticas. `export()` solo guarda eventos publicados. `restore()` comprueba versión, identidad, límites, maduraciones, contadores y huella. Rechaza estructuras profundas en campos escalares antes de serializarlas. La huella detecta modificaciones del contenido, pero no autentica a su productor ni demuestra que los pronósticos sean reales.

Los límites configurables son 1.024 episodios, 40.960 filas pendientes, 512 cohortes y 262.144 referencias como máximos. Una cohorte admite hasta 8.192 filas y cada fila hasta 64 supresiones. La capacidad pendiente se comprueba antes de copiar otro snapshot. No se guarda el grafo de autograd ni se consume RNG. El registro usa escalares CPU y no necesita un backend CUDA.

Las validaciones de emisión y publicación recorren las referencias pendientes y las referencias de las cohortes que cierran, respectivamente. Su coste está acotado por los límites declarados, aunque no es constante. La verificación del snapshot incluye sus tensores y la serialización ocupa espacio transitorio adicional. No se mide solo el tamaño de las estadísticas finales.

## Productor del lector estándar K1

`apply_spaced_utility_readout` consume `PreparedDecisions`, la cabeza del padre, el lector y su snapshot. Devuelve la predicción ordinaria y un diagnóstico opcional. Con `enabled=False` delega la ruta anterior, incluidas las opciones diferenciables y K = 2 o K = 4. La activación exige el lector estándar `bank`, K = 1, módulos en `eval` y ninguna diferenciación. Los lectores regionales y los refinamientos superiores necesitan productores propios y se rechazan explícitamente. No se modifica la identidad ni el checkpoint del modelo.

El productor reutiliza claves, valores proyectados e IDs del soporte elegido. Hace una sola selección y una sola proyección de valores. Repite una proyección pequeña de la consulta para reconstruir los logits del soporte, evitando añadir una caché de logits a las rutas históricas. Para cada episodio enmascara su logit y calcula el softmax sobre los restantes. No utiliza la resta de contribuciones dividida por `1 - peso`, que puede fallar por cancelación o cuando el peso redondea a uno. Las pruebas incluyen ese caso saturado en FP32 y FP64.

Cada lectura suprimida pasa por el refinador completo y la misma cabeza. Si solo había un vecino, la lectura queda ausente y a cero, pero se aplica también el refinador sin banco, con su sesgo y su entrada de presencia. No se sustituye arbitrariamente por la predicción del padre. Con memoria vacía no se generan supresiones. La cabeza cuantílica existente conserva sus cinco salidas y su mediana. El diagnóstico solo guarda puntos para el criterio de error, sin añadir cabezas.

Antes de leer se comprueba el presupuesto conjunto mediante la cota del lector para `1 + min(episodios, vecinos)` refinamientos. Es conservadora porque algunas estructuras se reutilizan. El cálculo mantiene el orden de selección y acota las supresiones a los ocho vecinos admitidos por el lector. Se rechazan cabezas con otra precisión o dispositivo y autocast sin un contraste independiente. No se cambia la política numérica global para obtener cifras más rápidas.

`SpacedUtilityDiagnostics` conserva flujos, corte, huella del snapshot, IDs y puntos. `ledger_rows` exige maduraciones previstas alineadas por flujo antes de transferir los tensores. Convierte IDs y puntos por bloques a CPU y comprueba finitud e identidades sin construir una segunda copia normalizada de todas las filas. El registro mantiene su validación independiente al recibirlas. No introduce resultados financieros. Su huella identifica el snapshot declarado, sin autenticar al productor. La verificación fuerte de bytes corresponde a la frontera de publicación y recuperación de la instantánea, como en el lector anterior. Las comprobaciones rápidas por consulta no prometen detectar una modificación externa mediante `.data` después de esa frontera.

## Fuentes y alcance

Mattar y Daw, *Prioritized memory access explains planning and hippocampal replay*, Nature Neuroscience 21, 1609–1617 (2018), proponen ordenar el acceso a recuerdos por su utilidad para decisiones en una tarea de navegación. El resumen describe el equilibrio entre necesidad y ganancia. E2 toma la pregunta sobre utilidad, pero su diferencia de errores y su criterio temporal son una propuesta propia. No implementa el algoritmo de planificación de ese artículo. [Artículo y DOI 10.1038/s41593-018-0232-z](https://www.nature.com/articles/s41593-018-0232-z).

Glas, Hübener, Bonhoeffer y Goltstein, *Spaced training enhances memory and prefrontal ensemble stability in mice*, Current Biology 31(18), 4052–4061.e6 (2021), observaron efectos del espaciado sobre memoria y similitud de actividad prefrontal en una tarea con ratones. El intervalo de 60 minutos descrito no se traslada a sesiones bursátiles. [Metadatos y resumen, DOI 10.1016/j.cub.2021.06.085](https://pubmed.ncbi.nlm.nih.gov/34324833/).

La consulta del 27 de octubre de 2026 contrastó metadatos y resúmenes de ambos artículos. No se declara lectura íntegra, reproducción de sus experimentos, comprobación Lean ni garantía formal de este diseño. Las pruebas sintéticas validan el contrato del registro, sin resultados sobre FinMultiTime.

## Comprobaciones reproducibles

Desde la raíz de esta rama, con su entorno uv preparado:

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_utility.py tests/memory/test_lesion_protocol.py tests/memory/test_lesion_ledger.py -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-utility-20261027/cost_probe.py /tmp/mars-titan-e2-cost-20261027.json
```

El productor estándar y sus regresiones se comprueban por separado:

```bash
PYTHONPATH=src:. uv run --no-sync pytest tests/memory/test_spaced_utility_readout.py tests/memory/test_spaced_utility.py tests/models/titans/test_episodic_readout.py tests/models/titans/test_episodic_readout_fixed_episodes.py -m 'not cuda' -q
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-readout-20261027/cost_probe.py /tmp/mars-titan-e2-readout-cost.json
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run --no-sync python reports/engineering/spaced-readout-20261027/compare_rows.py /tmp/mars-titan-e2-rows-comparison.json
```

La batería del productor aprueba 207 casos CPU, incluidos 60 del módulo nuevo. Excluye expresamente cuatro casos CUDA, dos del lector anterior y dos del productor. Los del productor aún no se han ejecutado en GPU. Las ocho mutaciones dirigidas de esta versión se detectan con referencias verdes antes y después. El módulo nuevo registra 88/88 sentencias y 10/10 ramas, con CRAP máximo 17. No se ha ejecutado la suite completa y estas cifras se solapan con las del registro, por lo que no se suman.

Los 34 archivos que componen la huella canónica de `FrozenFinancialPredictor` para su configuración base conservan los mismos bytes que en `404b91649`. El archivo del lector histórico permanece intacto. La adaptación de la salida está aislada en el módulo experimental y reutiliza la misma instancia de cabeza y la mediana existente. Catorce casos contrastan salidas y gradientes de las referencias frente a ese commit. No se presenta esta comprobación como recuperación de todas las campañas ni como validación CUDA.

La [evidencia del productor](../../reports/engineering/spaced-readout-20261027/verification.json), la [comparación de conversión](../../reports/engineering/spaced-readout-20261027/rows-comparison.json) y el [coste del lector con diagnósticos](../../reports/engineering/spaced-readout-20261027/cost.json) identifican fuentes y medidas. La comparación reconstruye solo el conversor anterior en una carpeta temporal, conserva el resto del productor y alterna ambos modos sobre las mismas filas. Las aplicaciones ajenas y el perfil de energía no se modificaron. El criterio técnico exige al menos un 5 % menos de mediana con 1.024 filas en ambas pasadas y ninguna regresión mayor del 5 % con 64 o 256. No es un criterio de mejora financiera.

La comparación final reduce la mediana un 12,36 % y un 10,33 % con 1.024 filas en sus dos pasadas y también reduce las medianas de 64 y 256 filas. Conserva exactamente filas y tamaño del JSON. El pico Python medido por separado pasa de 2.085.085 y 2.094.685 bytes a 1.807.336 bytes en ambas pasadas. El comparador conserva las muestras y el patch permite reconstruir la referencia. No se interpreta esta reducción como ahorro de toda la RAM ni como aceleración del predictor completo.

La medida final de lectura, supresiones, filas y JSON utiliza FP32, dos hilos, dos calentamientos por modo y diez muestras por tamaño. El padre MAC se prepara una vez por tamaño, fuera del tiempo por llamada. Los episodios tienen 64 coordenadas y el lector ocho vecinos y dimensión 32. Son fixtures compactos con anchuras 5/2/2/3/3 y entradas repetidas por flujo con IDs distintos. No reproducen las formas ni la distribución del corpus.

| Filas por llamada | Lectura ordinaria y JSON, mediana en ms | Lectura, ocho supresiones y JSON, mediana en ms |
| --- | --- | --- |
| 64 | 0,626 | 1,942 |
| 256 | 1,865 | 5,633 |
| 1.024 | 15,874 | 30,865 |

El proceso completo tarda 3,10 segundos y su pico RSS absoluto externo es 751.668 KiB. El mayor JSON diagnóstico ocupa 334.763 bytes. La medida excluye ledger, consolidación, persistencia durable, GPU y transferencias. No mide energía. Los costes absolutos proceden de una pasada independiente de la comparación intercalada y no se usan para calcular su reducción porcentual.

El archivo del benchmark se escribe en el destino indicado por el último argumento. El [recibo técnico](../../reports/engineering/spaced-utility-20261027/verification.json) recoge fuentes, pruebas, mutaciones, cobertura, CRAP y límites. El [informe de coste](../../reports/engineering/spaced-utility-20261027/cost.json) conserva las muestras individuales de dos sesiones, con JSON, recuperación y publicación. Excluye predictor, generación de supresiones, disco y GPU. No acredita latencia del modelo ni mejora de generalización.

La batería del registro y las regresiones seleccionadas aprobaron 102 casos sin omisiones, incluidos 51 del registro. Otra batería aprobó 155 casos de los puntos de entrada protegidos, con ajustes sustituidos por dobles de prueba. Se detectaron ocho mutaciones concretas con referencias verdes antes y después. Coverage.py 7.16.1 y Radon 6.0.1 registran 320/320 sentencias y 60/60 ramas en los dos módulos nuevos, con CRAP máximo de 19. Esta evidencia no se extrapola a toda la canalización ni a todos los fallos posibles.

La confirmación de coste usa un Ryzen 9 8945HS, dos hilos de cómputo y diez muestras tras dos calentamientos. Para dos sesiones de 64, 256 y 1.024 filas, las medianas son 8,043, 29,923 y 122,094 ms. Los JSON pendientes máximos ocupan 20.211, 79.923 y 320.584 bytes. El mayor pico de asignación Python observado con tracemalloc es 4.964.924 bytes en una pasada separada. Excluye las asignaciones nativas. El tiempo total externo es 5,81 segundos y su pico RSS absoluto, 715.296 KiB, incluye el proceso completo. No se midieron energía ni incremento RAM atribuible al módulo. La primera medida solapó mutaciones y se conserva aparte. La confirmación no solapó otras pruebas propias, pero mantuvo activas las aplicaciones ajenas y el perfil de energía existente.

Quedan pendientes el productor de supresiones regional, su persistencia en el bundle, el selector con sus controles, la conexión con sesión y ventana y su evaluación autorizada. El núcleo y el productor estándar no habilitan entrenamientos, no cambian el baseline y no abren el test final.
