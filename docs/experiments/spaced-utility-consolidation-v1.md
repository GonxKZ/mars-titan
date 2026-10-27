# Evidencia de utilidad espaciada para la consolidación E2

El registro `SpacedUtilityLedger` está implementado y comprobado con fixtures CPU. Conserva pronósticos emitidos y calcula evidencia por episodio cuando madura el resultado. Todavía no produce las supresiones diagnósticas, no decide qué episodios conserva el banco y no está conectado a la sesión financiera. Por tanto, esta unidad no completa la variante `spaced_utility_consolidation_v1` ni cambia las predicciones de ninguna referencia.

E2 estudia si conviene conservar episodios que han ayudado en decisiones separadas. No es una técnica de regularización de los parámetros del predictor y no demuestra una reducción del sobreajuste. La comparación predictiva permanece sin ejecutar.

## Contraste y agregación

Para una predicción original emitida \(p_i\), una predicción diagnóstica \(p_{i,-e}\) emitida sin el episodio \(e\) y su resultado maduro \(y_i\), se define:

\[
g_{i,e}=|y_i-p_{i,-e}|-|y_i-p_i|.
\]

Un valor positivo indica que el pronóstico original tuvo menor error. El contraste propuesto para K = 1 elimina el episodio del soporte ya seleccionado, renormaliza sus pesos y no introduce otro vecino. No equivale a quitarlo de todo el banco y repetir la búsqueda. El registro recibe ambos números. La integración futura debe comprobar que proceden de la misma emisión y de una única preparación del padre, anterior a la etiqueta. El registro por sí solo no acredita esa procedencia.

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

El archivo del benchmark se escribe en el destino indicado por el último argumento. El [recibo técnico](../../reports/engineering/spaced-utility-20261027/verification.json) recoge fuentes, pruebas, mutaciones, cobertura, CRAP y límites. El [informe de coste](../../reports/engineering/spaced-utility-20261027/cost.json) conserva las muestras individuales de dos sesiones, con JSON, recuperación y publicación. Excluye predictor, generación de supresiones, disco y GPU. No acredita latencia del modelo ni mejora de generalización.

La batería del registro y las regresiones seleccionadas aprobaron 102 casos sin omisiones, incluidos 51 del registro. Otra batería aprobó 155 casos de los puntos de entrada protegidos, con ajustes sustituidos por dobles de prueba. Se detectaron ocho mutaciones concretas con referencias verdes antes y después. Coverage.py 7.16.1 y Radon 6.0.1 registran 320/320 sentencias y 60/60 ramas en los dos módulos nuevos, con CRAP máximo de 19. Esta evidencia no se extrapola a toda la canalización ni a todos los fallos posibles.

La confirmación de coste usa un Ryzen 9 8945HS, dos hilos de cómputo y diez muestras tras dos calentamientos. Para dos sesiones de 64, 256 y 1.024 filas, las medianas son 8,043, 29,923 y 122,094 ms. Los JSON pendientes máximos ocupan 20.211, 79.923 y 320.584 bytes. El mayor pico de asignación Python observado con tracemalloc es 4.964.924 bytes en una pasada separada. Excluye las asignaciones nativas. El tiempo total externo es 5,81 segundos y su pico RSS absoluto, 715.296 KiB, incluye el proceso completo. No se midieron energía ni incremento RAM atribuible al módulo. La primera medida solapó mutaciones y se conserva aparte. La confirmación no solapó otras pruebas propias, pero mantuvo activas las aplicaciones ajenas y el perfil de energía existente.

Quedan pendientes el productor de supresiones sin relleno, su persistencia en el bundle, el selector con sus controles, la conexión con sesión y ventana y su evaluación autorizada. El núcleo no habilita entrenamientos, no cambia el baseline y no abre el test final.
