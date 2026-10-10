# Control en línea del Transformer compacto

MARS-TITAN sigue adaptándose durante calibración y evaluación. La memoria de Titans-MAC se actualiza con las entradas publicadas y el banco episódico admite cada etiqueta en el instante en que madura. Si MARS-TITAN mejora a `transformer_compact`, parte de la diferencia podría deberse solo a seguir aprendiendo con esas etiquetas y no a la memoria. El control `transformer_compact_online` existe para poder descartar esa explicación. Lo decidió el usuario el 9 de octubre de 2026 ([#443](https://github.com/GonxKZ/mars-titan/issues/443)). La regla con sus valores se fijó el 10 de octubre, antes de cualquier resultado. Está implementado y comprobado sin pasos de optimizador. No se ha ejecutado.

## Qué hace

El control parte del estado elegido de `transformer_compact` en la misma ventana y semilla. Recorre validación, calibración y evaluación por separado, y cada tramo empieza otra vez desde ese estado, igual que el banco de MARS-TITAN empieza vacío en cada tramo. La validación solo sirve para elegir la tasa de aprendizaje. En cada instante:

1. Recibe las etiquetas que maduran en ese instante. Cada una debe corresponder a una predicción ya emitida en un instante anterior.
2. Emite las predicciones de las decisiones de ese instante con los pesos vigentes.
3. Si toca actualizar, da un solo paso con todas las etiquetas maduras desde la actualización anterior, hasta el tope.

Es el mismo orden del lector de MARS-TITAN, que predice con el banco anterior a las etiquetas del instante y las admite después. Una predicción nunca usa su propia etiqueta, porque esta solo madura en un instante posterior a su emisión.

## Mismas etiquetas e instantes que el banco

El control no repite la lógica de maduración. Reconstruye el índice de observaciones de MARS-TITAN (`memory/financial_observations.py`) con las fases que registra el informe de `mars_titan_m1` y exige que la identidad de cada índice coincida con la de ese informe. Así recibe exactamente las etiquetas maduras y los instantes de maduración del banco. Después exige además que el número de etiquetas de cada tramo coincida con el que registra M1. Los grupos del calentamiento se omiten, porque solo contienen entradas y el Transformer no conserva estado entre instantes.

`FinancialObservationSource.label_decisions` lee del propio índice qué decisiones tienen etiqueta en el tramo. El control solo guarda las entradas de esas decisiones mientras esperan su etiqueta, y las libera al usarlas. La validación usa también el índice y el tope de la validación de M1, que su informe registra porque M1 predice validación, calibración y evaluación.

## Regla declarada antes de ejecutar

La decisión del 10 de octubre fija que el control recibe exactamente las mismas etiquetas maduras, con la misma madurez, la misma cadencia de actualización y el mismo presupuesto de actualizaciones que el banco M1 de MARS-TITAN. La tasa de aprendizaje se elige en validación con una rejilla declarada antes de ejecutar, como en las demás familias. La campaña A v2 la declara en `online_controls`:

| Campo | Valor | Por qué |
| --- | --- | --- |
| `optimizer` | `sgd`, sin momento ni decaimiento de pesos | Cada paso depende solo de los pesos vigentes y de las etiquetas del instante. No hay estado del optimizador que arrastre información entre instantes, igual que el banco solo guarda lo que escribe |
| `update_every` | 1 | El banco escribe en cada instante de maduración todas las etiquetas que maduran en él. El control actualiza con la misma cadencia |
| `update_cap` | `episodic_bank_writes` | Mismo presupuesto de información: en cada tramo no usa más etiquetas que escrituras hizo el banco |
| `accumulation_rows` | 256 | Solo acota la memoria. Es el lote con el que se ajusta `transformer_compact` en la campaña (`neural.batch_size`), así que cada bloque ocupa lo mismo que un paso de su ajuste |
| `max_grad_norm` | 1,0 | El recorte de las recetas de Titans-MAC y MARS-TITAN. Acota cada actualización a una norma de como mucho la tasa, lo que da escala a la rejilla |
| `search_cases` | 0, 1e-3, 1e-2, 1e-1 y 1 | Rejilla de tasas, elegida con el MAE por sesión de la validación |

La pérdida es la del caso de `transformer_compact`: pinball con la cabeza de cuantiles o la pérdida escalar del caso. Cada actualización da un solo paso con la media de la pérdida de todas las etiquetas que maduraron desde la anterior. Se acumula por bloques de `accumulation_rows` filas: cada bloque aporta la suma de sus pérdidas dividida por el total, así que el gradiente es el de la media sobre todas las filas y solo cambia el orden de las sumas en FP32. La finitud de la pérdida se comprueba una vez por paso. Las actualizaciones se hacen con el modelo en modo de evaluación, sin dropout, para que el recorrido sea determinista.

**Tope y cadencia.** En cada tramo el control no usa más etiquetas que escrituras hizo el banco de `mars_titan_m1` en el mismo ámbito, ventana y semilla (`admitted` en las métricas de su informe). M1 admite cada etiqueta madura de una predicción emitida, de modo que el tope coincide con las etiquetas del tramo y el control ve las mismas. Con un paso por instante, el número de actualizaciones es el de instantes de maduración con etiquetas, el mismo que el de escrituras del banco por instante. La versión anterior daba un paso por bloque de etiquetas. Con más etiquetas por instante que filas por bloque, el control habría actualizado varias veces en un instante en que el banco escribe una sola vez. Las etiquetas que quedan después de la última actualización se cuentan aparte, porque ya no afectan a ninguna predicción.

**Rejilla de tasas.** No se ha podido medir la escala del gradiente, porque no hay estados ajustados de `transformer_compact` ni vistas preparadas mientras siga el bloqueo. La rejilla se razonó así:

- Con el recorte, un paso mueve los pesos como mucho la tasa en norma.
- En la cabeza de cuantiles, el gradiente de la pérdida pinball respecto al sesgo de un nivel es la media de `1[y < q] − τ`, dividida por los cinco niveles. Un error de cobertura de cinco puntos da del orden de 0,01.
- Los cuantiles del residuo diario están en torno a 0,01.

Así, una tasa de 1e-3 desplaza un cuantil del orden de una centésima de su escala por instante, y una de 1 del orden de toda su escala. Las cuatro décadas cubren desde un control casi congelado hasta uno muy agresivo. La tasa cero reproduce el Transformer congelado. Se incluye para que el control nunca quede por debajo de él en validación: si seguir aprendiendo empeora la validación, el control más fuerte con la misma información es no usarla. La comparación con MARS-TITAN es así más exigente para MARS-TITAN. Una tasa en el borde de la rejilla se informará como tal y no se ampliará la rejilla a la vista de la evaluación.

**Búsqueda y finalistas.** Como en las demás familias, hay un caso de búsqueda por tasa con la semilla de búsqueda (42), que parte del estado elegido de `transformer_compact` con esa semilla y usa el banco de M1 con la misma. La puntuación es el MAE por sesión de su validación en línea. Las semillas 43 y 44 repiten la tasa de la búsqueda con menor MAE. Un empate se resuelve por el identificador, y `search-lr0` va antes que las demás tasas, así que un empate con la tasa cero se queda con el control congelado. Cada finalista parte del estado elegido de `transformer_compact` y del banco de M1 de su propia semilla.

## Precisión

El control exige FP32 estricto antes de leer ninguna fuente: `float32_matmul_precision = highest`, sin TF32 en cuBLAS ni en cuDNN y sin autocast. PyTorch 2.14 tiene además la interfaz `fp32_precision`. Si pide TF32 en matmul, convolución o RNN, el control se detiene. Si las dos interfaces se mezclan, la lectura de los indicadores antiguos falla y también se detiene. Una petición genérica de TF32 (`torch.backends.fp32_precision`) se rechaza aunque cada submódulo la anule con `ieee`. En ese caso los indicadores antiguos se leen estrictos y solo la interfaz nueva muestra la petición. El control prefiere detenerse a depender de esa herencia.

## Campaña

El motor registra el ejecutor como `("neural", "online")`, con el informe `online.json` (versión 2, con la validación) y sin reanudación de intentos. `JobRun.bank` lleva el intento elegido de `mars_titan_m1` y el ancla es el estado elegido de `transformer_compact` en la misma ventana y semilla. El recibo comprueba como en un traslado que el informe parte de ese estado y guarda como puntuación el MAE por sesión de la validación. Los trabajos los declara el plan de la campaña A por etapas: `<ámbito>/<ventana>/transformer_compact_online/search-<tasa>` con la semilla 42 y `.../finalist-s<semilla>` con 43 y 44. `resolve_online` da a la búsqueda su tasa y al finalista la de la búsqueda ganadora (`case_source` en las fuentes de su identidad), con el ancla y el banco de su semilla. `selected` elige el control de cada semilla igual que en las demás familias. En el calendario ventana a ventana, búsquedas y finalistas van en la fase `online`, después de elegir el padre y el banco con todas sus semillas.

El control no forma parte de la cadena de predictores que alimenta la RL. Para los brazos en línea, el valor `labels_used_until` del recibo de ventana solo acota el estado de partida. La causalidad dentro de calibración y evaluación la comprueban las pruebas del ejecutor.

## Comparación

La comparación de la evaluación (`configs/evaluation/historical-masked-2000-comparison.json`) declara el brazo `transformer_compact_online` en la familia `online_control`, con la misma cabeza de cuantiles y las mismas semillas que `transformer_compact` y `mars_titan_m1`. Dos familias de contrastes lo emparejan sobre las mismas filas y sesiones:

| Familia | Contraste | Qué responde |
| --- | --- | --- |
| `online_learning` | `transformer_compact_online` menos `transformer_compact` | Cuánto gana el Transformer solo por seguir aprendiendo con las etiquetas del banco |
| `memory_vs_online_learning` | `mars_titan_m1` menos `transformer_compact_online` | Si MARS-TITAN mejora a un modelo que recibe la misma información en línea |

El brazo entra también en la familia de niveles. Si el control en línea alcanza a MARS-TITAN, su mejora frente al Transformer congelado no puede atribuirse solo a la memoria. Si MARS-TITAN lo supera con un intervalo simultáneo que excluye el cero, el aprendizaje en línea con las mismas etiquetas no basta para explicar la diferencia. El contraste no separa la memoria de Titans-MAC del banco episódico, porque M1 tiene las dos. Esa separación la dan `episodic_reader` y `episodic_write_policies`.

Las campañas A y B de la edición desde 2000 no declaran los trabajos del control, así que el plan lo lista entre las familias pendientes con la issue #443. La campaña A por etapas los declara en su sección `online_controls`, así que el plan ya no la cuenta entre las pendientes. Solo crea trabajos en los ámbitos cuya comparación evalúa el brazo, que en A v2 es el conjunto: 133 trabajos, 19 ventanas por cinco búsquedas y dos finalistas.

Con los recuentos de la campaña, cada trabajo predice 26.447.932 filas de validación, calibración y evaluación en las 19 ventanas, y como mucho ajusta una vez cada una. Con la cota de `budget` (una predicción y un paso por fila con el caudal más lento del Transformer), el control suma 4,3 h a 16.000 filas/s, 2,1 h a 33.300 y 1,6 h a 44.100, frente a 1,3, 0,6 y 0,5 h de la versión anterior. La validación añade un 38 % de filas y la búsqueda multiplica los trabajos por 2,3. No se ha medido el coste real del ejecutor.

## Comprobaciones

`tests/training/test_online_reference.py` usa un ancla Transformer con los pesos iniciales de la semilla y una ventana M1 ajustada con el registrador de gradientes sobre un padre Titans-MAC. El optimizador del control solo registra pasos, normas y gradientes.

- Con tope cero no hay pasos y las predicciones coinciden con las del Transformer congelado en validación, calibración y evaluación.
- Cada actualización es un solo paso con todas las etiquetas maduras desde la anterior, ya maduras y de predicciones emitidas antes, con `update_every` 1 y 2. Hay pasos con más etiquetas que filas por bloque y cada etiqueta se usa una vez.
- En los tres tramos, bloques de una fila, de dos y uno que cubre el instante dan las mismas métricas, el mismo recorrido y gradientes iguales con tolerancia relativa de 1e-5.
- El tope limita las etiquetas usadas: los pasos son los primeros del recorrido sin tope y el último se corta.
- Un instante con una sola etiqueta también actualiza, y una etiqueta no finita en cualquier bloque detiene el paso antes del optimizador.
- La puntuación de la búsqueda es el MAE de validación de un ajuste o de un control en línea, y se rechaza si no es un número finito no negativo.
- Una etiqueta adelantada al instante de su decisión o a un instante anterior detiene el control.
- Una etiqueta de una decisión que nunca se predijo detiene el control. Una decisión sin etiqueta en el tramo se predice y no se guarda.
- Las filas escritas son las de M1, con las mismas etiquetas y el mismo tope.
- Se rechazan TF32 por cualquiera de las dos interfaces, incluida la petición genérica, la precisión `high`, el autocast, una regla con valores pendientes, tasas fuera de [0, 1] o enteras, bloques fuera de 1 a 4.096, un banco de otra semilla, otra vista o otra escritura, otros índices, otro número de etiquetas, un ancla que no es Transformer y una salida existente. La tasa cero es válida.
- El motor resuelve el ancla y el banco de la misma ventana y semilla, da al finalista la tasa de la búsqueda con menor MAE (con el desempate por identificador), elige el control de cada semilla y se detiene entre instantes al pedir la parada.

`tests/training/test_campaign_chain.py` comprueba el plan: 133 trabajos, cinco búsquedas con la regla completa y su tasa, finalistas que dependen de todas las búsquedas y del padre y el banco de su semilla, la fase `online` en el calendario y que no hay bloqueos de lanzamiento. Rechaza 22 declaraciones de la regla, entre ellas otra cadencia, una rejilla sin el cero, tasas repetidas, enteras o mayores que uno y la falta de validación en los tramos, y además un recorte infinito leído de JSON.

Las pruebas eliminan los 21 mutantes de la primera mutación dirigida sobre el ejecutor y el motor. Al fijar la regla se aplicaron otros 30 defectos, de uno en uno y sobre una copia del árbol:

- en el paso: media por bloque, gradiente solo del último bloque, un paso por bloque, instantes de una etiqueta sin paso, finitud sin comprobar o solo del último bloque, tasa cero rechazada, sin validación y bloque sin límite;
- en la declaración: rejilla sin cero, otra cadencia, cadencia booleana, tasas repetidas o enteras, una sola tasa, recorte entero o infinito, caso sin tasa, finalista sin búsquedas y búsqueda con la semilla del finalista;
- en el motor: ganador por máximo, desempate inverso, finalista sin la tasa ganadora o sin la fuente de su caso, control sin puntuación, control resuelto como una búsqueda común y etapa desconocida admitida;
- en el calendario y las horas: búsqueda en la fase común, elección de la tasa entre las de la base y horas sin validación.

En la primera pasada sobrevivieron dos. Saltar los instantes con una sola etiqueta no se notaba porque todos los instantes de la prueba tenían varias. La tasa entera tampoco, porque los únicos enteros de [0, 1] repetían tasas de la rejilla. Se añadió la prueba del instante con una etiqueta y la tasa entera sustituye ahora a la de 1,0. Los 30 fallan.

No se ha medido el coste real. Cada tramo recorre una vez sus instantes con un forward por bloque de decisiones y un forward y backward por bloque de acumulación.
