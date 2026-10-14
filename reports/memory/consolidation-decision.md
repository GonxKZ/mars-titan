# Decisión de consolidación por repaso

Autor: Gonzalo García Lama. Decisión: 13 de octubre de 2026. Evidencia de MT-053 (#55).

## Decisión

El repaso de filas anteriores queda implementado, declarado en la matriz de adaptadores v3 y sin activar en la campaña. Ninguna etapa nombra la sección `replay_continuation` en `additional_controls`, así que los planes de A y A v2 no cambian. Activarlo es nombrar la sección en la etapa, sin tocar código ni la matriz. El [diseño y las comprobaciones](../../docs/engineering/replay-continuation.md) están en su documento de ingeniería.

No se activa todavía por tres motivos:

1. La issue condiciona el repaso a que la memoria mínima aporte evidencia y a que haya una pregunta con margen. MT-052 (#54), que mide retención e interferencia, sigue abierta. Sin una medida de olvido en la continuación, el repaso responde a un riesgo que nadie ha observado en estos datos.
2. El bloqueo de aprendizaje impide cualquier ajuste. No hay resultado predictivo que valorar, solo coste.
3. El coste no es pequeño: los tres controles suman unas 37 horas de GPU más en los adaptadores de A v2 (de 118 a 156, o de 180 a 218 con la cota de los objetivos de #128) con 16.000 muestras-época por segundo y 10 épocas efectivas, cuando el plan de A v2 ya necesita un caudal 7,9 veces mayor que el supuesto para caber en 580 h útiles (8,1 con esos objetivos).

## Corte y filas

- Ajuste de la ventana k: filas nuevas desde el final de la calibración del padre de k-1 hasta el final del tramo de ajuste de k, como la continuación completa.
- Repaso: sesiones anteriores a esa frontera cuyas filas ajustó todas el padre en su tramo `train` y cuyas etiquetas maduran antes de la frontera. La validación y la calibración del padre no entran.
- Selección: solo con la validación de la ventana, la misma para todos los casos. La reserva de 2024 no se abre.

## Control que descarta el repaso

`repeat_new_rows` hace las mismas actualizaciones adicionales con las filas nuevas. Si iguala a los dos repasos en validación, la ganancia es de cómputo y el repaso se descarta. Si el reparto por régimen no mejora al uniforme, la diversidad se descarta con él. La comparación de la etapa declara esa familia (`versus_repeat_new_rows`) y se informan todos los controles, también los que empeoren.

## Coste medido

Preparación y lectura en CPU sobre la ventana `fold-012` de 64 activos de EE. UU. ([recibo](../engineering/replay-continuation-cost-20261013.json), con el detalle en el [documento de ingeniería](../../docs/engineering/replay-continuation.md#coste)):

- 3.708 sesiones anteriores admitidas y 192 excluidas porque el padre no ajustó todas sus filas.
- Primera época con repaso: 5,7 s (uniforme) y 7,3 s (por régimen) frente a 0,35 s de la continuación completa, porque abre y lee las sesiones planificadas. Las siguientes, entre 0,50 y 0,65 s.
- Cada control hace un 50 % más de actualizaciones por época que la continuación completa. Es el coste principal y no se ha medido en GPU, porque el bloqueo impide ajustar. Con el caudal supuesto de la campaña, los tres controles suman 37 horas de GPU en A v2.

## Qué haría cambiar la decisión

- MT-052 mide olvido en la continuación completa (pérdida en sesiones antiguas tras ajustar con las nuevas) mayor que la dispersión entre semillas.
- El presupuesto de la campaña A, ya medido, deja horas para los ajustes adicionales sin quitarlas a la comparación principal.

Si las dos se cumplen, la etapa nombra la sección y la ejecución queda registrada como cualquier otra continuación. Si el repaso no supera a la repetición, la extensión se cierra con ese resultado negativo.
