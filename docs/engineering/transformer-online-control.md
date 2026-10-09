# Control en línea del Transformer compacto

MARS-TITAN sigue adaptándose durante calibración y evaluación. La memoria de Titans-MAC se actualiza con las entradas publicadas y el banco episódico admite cada etiqueta en el instante en que madura. Si MARS-TITAN mejora a `transformer_compact`, parte de la diferencia podría deberse solo a seguir aprendiendo con esas etiquetas y no a la memoria. El control `transformer_compact_online` existe para poder descartar esa explicación. Lo decidió el usuario el 9 de octubre de 2026 ([#443](https://github.com/GonxKZ/mars-titan/issues/443)). Está implementado y comprobado sin pasos de optimizador. No se ha ejecutado.

## Qué hace

El control parte del estado elegido de `transformer_compact` en la misma ventana y semilla. Recorre calibración y evaluación por separado, y cada tramo empieza otra vez desde ese estado, igual que el banco de MARS-TITAN empieza vacío en cada tramo. En cada instante:

1. Recibe las etiquetas que maduran en ese instante. Cada una debe corresponder a una predicción ya emitida en un instante anterior.
2. Emite las predicciones de las decisiones de ese instante con los pesos vigentes.
3. Si toca actualizar, usa las etiquetas maduras acumuladas, por bloques, hasta el tope.

Es el mismo orden del lector de MARS-TITAN, que predice con el banco anterior a las etiquetas del instante y las admite después. Una predicción nunca usa su propia etiqueta, porque esta solo madura en un instante posterior a su emisión.

## Mismas etiquetas e instantes que el banco

El control no repite la lógica de maduración. Reconstruye el índice de observaciones de MARS-TITAN (`memory/financial_observations.py`) con las fases que registra el informe de `mars_titan_m1` y exige que la identidad de cada índice coincida con la de ese informe. Así recibe exactamente las etiquetas maduras y los instantes de maduración del banco. Después exige además que el número de etiquetas de cada tramo coincida con el que registra M1. Los grupos del calentamiento se omiten, porque solo contienen entradas y el Transformer no conserva estado entre instantes.

`FinancialObservationSource.label_decisions` lee del propio índice qué decisiones tienen etiqueta en el tramo. El control solo guarda las entradas de esas decisiones mientras esperan su etiqueta, y las libera al usarlas.

## Regla declarada antes de ejecutar

| Campo | Significado |
| --- | --- |
| `optimizer` | `sgd`, sin momento ni decaimiento de pesos |
| `learning_rate` | Tasa del paso |
| `block_rows` | Etiquetas por paso como máximo |
| `update_every` | Instantes de maduración entre actualizaciones. Con 1 se actualiza en cada instante con etiquetas |
| `max_grad_norm` | Recorte de la norma del gradiente |
| `update_cap` | `episodic_bank_writes` |

La pérdida es la del caso de `transformer_compact`: pinball con la cabeza de cuantiles o la pérdida escalar del caso, promediada en el bloque. Las actualizaciones se hacen con el modelo en modo de evaluación, sin dropout, para que el recorrido sea determinista.

**Tope.** En cada tramo el control no usa más etiquetas que escrituras hizo el banco de `mars_titan_m1` en el mismo ámbito, ventana y semilla (`admitted` en las métricas de su informe). La igualdad se fija en etiquetas y no en pasos. Una escritura del banco es una etiqueta madura y un paso del control usa hasta `block_rows` etiquetas, así que igualar pasos daría al control mucha más información que al banco cuando el bloque es mayor que uno. M1 admite cada etiqueta madura de una predicción emitida, de modo que el tope coincide con las etiquetas del tramo y el control ve las mismas. Las etiquetas que quedan después de la última actualización se cuentan aparte, porque ya no afectan a ninguna predicción.

La campaña A por etapas declara la regla en `online_controls` con los valores numéricos pendientes. Mientras haya valores pendientes, el lanzamiento queda bloqueado. Fijarlos es una decisión previa a ejecutar.

## Precisión

El control exige FP32 estricto antes de leer ninguna fuente: `float32_matmul_precision = highest`, sin TF32 en cuBLAS ni en cuDNN y sin autocast. PyTorch 2.14 tiene además la interfaz `fp32_precision`. Si pide TF32 en matmul, convolución o RNN, el control se detiene. Si las dos interfaces se mezclan, la lectura de los indicadores antiguos falla y también se detiene. Una petición genérica de TF32 (`torch.backends.fp32_precision`) se rechaza aunque cada submódulo la anule con `ieee`. En ese caso los indicadores antiguos se leen estrictos y solo la interfaz nueva muestra la petición. El control prefiere detenerse a depender de esa herencia.

## Campaña

El motor registra el ejecutor como `("neural", "online")`, con el informe `online.json` y sin reanudación de intentos. `JobRun.bank` lleva el intento elegido de `mars_titan_m1` y el ancla es el estado elegido de `transformer_compact` en la misma ventana y semilla. El recibo comprueba como en un traslado que el informe parte de ese estado. Los trabajos, con identificador `<ámbito>/<ventana>/transformer_compact_online/online-s<semilla>`, los declara el plan de la campaña A por etapas.

El control no forma parte de la cadena de predictores que alimenta la RL. Para los brazos en línea, el valor `labels_used_until` del recibo de ventana solo acota el estado de partida. La causalidad dentro de calibración y evaluación la comprueban las pruebas del ejecutor.

## Comprobaciones

`tests/training/test_online_reference.py` usa un ancla Transformer con los pesos iniciales de la semilla y una ventana M1 ajustada con el registrador de gradientes sobre un padre Titans-MAC. El optimizador del control solo registra pasos y normas.

- Con tope cero no hay pasos y las predicciones coinciden con las del Transformer congelado en calibración y evaluación.
- Cada paso usa etiquetas ya maduras de predicciones emitidas antes, con `update_every` 1 y 2, bloques de como máximo `block_rows` etiquetas y cada etiqueta una sola vez.
- El tope limita las etiquetas usadas.
- Una etiqueta adelantada al instante de su decisión o a un instante anterior detiene el control.
- Una etiqueta de una decisión que nunca se predijo detiene el control. Una decisión sin etiqueta en el tramo se predice y no se guarda.
- Las filas escritas son las de M1, con las mismas etiquetas y el mismo tope.
- Se rechazan TF32 por cualquiera de las dos interfaces, incluida la petición genérica, la precisión `high`, el autocast, una regla con valores pendientes, un banco de otra semilla, otra vista o otra escritura, otros índices, otro número de etiquetas, un ancla que no es Transformer y una salida existente.
- El motor resuelve el ancla y el banco de la misma ventana y semilla y se detiene entre instantes al pedir la parada.

Estas pruebas eliminan los 21 mutantes de la mutación dirigida sobre el ejecutor y el motor.

No se ha medido el coste. Cada tramo recorre una vez sus instantes con un forward por bloque de decisiones y un forward y backward por paso.
