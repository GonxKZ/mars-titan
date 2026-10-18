# Contraste de representaciones gráficas en el estudio de refinamientos

Este protocolo complementa el [estudio de desarrollo de las rutas de
refinamiento](refinement-development-v1.md). Compara la representación gráfica
original del corpus con una representación geométrica de las mismas ventanas.
Es un contraste de representación. La capacidad de elegir rutas se estudia
también dentro de cada representación con sus controles propios.

Se conserva el catálogo de
`configs/experiments/refinement-development-v1.json`: 64 activos US, semillas
42, 43 y 44, núcleos `linear_v1` y `mac_paper`, dos épocas por núcleo y lector,
lectores M1 con K1 y K4 y los controles ya fijados. Cada representación tiene
sus propios núcleos, lectores, políticas, calibradores y checkpoints. No se
transfieren pesos entre representaciones. Los parámetros comunes comparten
semillas y reglas de inicialización. Una diferencia de dimensiones debe
registrarse, con sus parámetros y coste, sin presentarla como una comparación
a igualdad exacta de tamaño.

## Datos y contraste

La cohorte se fija con el criterio de historia anterior a abril de 2021 del
protocolo principal. Ambas representaciones conservan activos, identidades de
fila, objetivos, disponibilidad, máscaras, cortes y épocas. La comprobación
de emparejamiento admite diferencias únicamente en `charts` y `chart_hash`.
Los recibos vinculan esa comprobación y ambas preparaciones mediante hashes.
Los recuentos iguales por sí solos no acreditan igualdad de filas.

Se mantienen `fold-017` para seleccionar núcleo y lector y `fold-018` para
ajustar y seleccionar la puerta en 2022, calibrarla en el último trimestre y
evaluarla en 2023. La evaluación de 2022 que publican los runners del padre
sigue declarada como exposición previa. El test final de 2024 y la campaña
completa permanecen cerrados. El año 2023 conserva su condición de desarrollo.

Se presentan dos diferencias separadas:

- El cambio de representación para el mismo núcleo, semilla, profundidad del
  lector y control, sobre las mismas sesiones.
- El cambio de la política de rutas respecto a K1 y los demás controles,
  dentro de cada representación.

Los resultados conservan el signo `MAE_variante − MAE_referencia`, los
intervalos por bloques temporales y la variabilidad entre semillas definidos
en el protocolo principal. La selección usa solo validación. El catálogo,
los cortes y los hiperparámetros no se cambian después de observar 2023.
Las comparaciones incompletas se identifican con sus fases pendientes y no
se mezclan con pares terminados. Este contraste exploratorio no acredita
superioridad general ni originalidad de una representación.

Esta ronda compara la representación original basada en ResNet con la
geométrica. No separa por sí sola los efectos de rasterización, preentrenamiento
externo y canales derivados. Un cambio de MAE no identifica cuál de esos
mecanismos lo causó. Los controles predictivos de tinta y duplicación de precios
quedan fuera de este catálogo. La comparación técnica que incluye tinta mide
ese recorrido de cálculo, pero no constituye un entrenamiento predictivo ni
una medida de su MAE. No se añaden candidatos a los cruces de proyección y
puerta ya fijados.

## Orden y presupuesto conjunto

La ejecución sigue el orden semilla, proyección del núcleo y etapa del
catálogo. Cada etapa alterna la representación original y la geométrica y
termina ambas antes de avanzar. Si una etapa necesita varios intentos, sus
intentos alternan dentro del mismo par. Los estados confirmados se reanudan
desde los recibos existentes, sin empezar de nuevo los contadores.

Se fija un límite conjunto de ocho horas de tiempo local de los procesos
hijos y se conserva el límite de seis horas de cada estudio. La contabilidad
conjunta incluye lanzamiento, importaciones del hijo, verificación, trabajo
y comprobación de su recibo. La espera previa por GPU, las importaciones y
verificaciones del supervisor y su escritura final de liquidación quedan
fuera de ese cronómetro. Los tiempos internos de cada fase se mantienen
separados. No se suman dos veces al total conjunto.

Antes de lanzar un hijo se reserva en disco todo el saldo conjunto. Al
confirmar su terminación se cobra el tiempo observado y se libera el resto.
Un corte sin cierre mantiene la reserva cargada y bloquea otra ejecución.
Las reservas abiertas de cada estudio conservan también su cargo. No se
recrea la preparación ni se vacía el libro mayor para recuperar presupuesto.
La reserva conservadora no representa tiempo de GPU medido.
Los intentos públicos deben corresponder a hijos registrados en la cola. Si
aparece una ejecución externa a ella, se detiene la supervisión para evitar
que ese coste quede fuera del total conjunto.

La supervisión solicita la parada de su propio árbol de procesos antes de
agotar el saldo conjunto y limita la espera de cierre a cinco segundos. El
comando `timeout`, ejecutado como proceso independiente, conserva ese límite
si el supervisor termina abruptamente.
Registra cualquier exceso observado durante el cierre. Una parada forzada
conserva el último estado confirmado y puede dejar una reserva pública sin
cerrar. Ese caso requiere revisar los recibos antes de reanudar. No se
detienen cargas ajenas para liberar recursos.

La primera pareja de núcleos produce una barrera técnica para revisar coste,
memoria y recuperación. Su extrapolación solo estima el coste de los núcleos
y declara la hipótesis de coste constante entre proyecciones y semillas.
Todavía no estima lectores, trazas, selección ni evaluación. La continuación
se registra sin cambiar candidatos según la calidad predictiva observada.
Si no cabe todo el catálogo, las combinaciones pendientes siguen enumeradas.

## Latencia posterior de los checkpoints reales

La futura medición de latencia utilizará los checkpoints seleccionados y
estados temporales obtenidos con pasado admisible. Los rangos existentes en
`MarsTitanInference._observe` separan `readout.snapshot`, `readout.prepare`,
`readout.apply` y `readout.emit`. `RefinementInference._predict_block`, dentro
de `readout.apply`, mide la consulta, la decisión de ruta y el lector sobre
un estado MAC ya preparado. Ese punto aislado no mide el predictor completo.

Para medir la predicción por lote hay que incluir la preparación MAC, la
consulta y los refinamientos, la cabeza y la emisión. Las transferencias de
entrada y la creación de la instantánea se medirán por separado y se
identificará si están incluidas en el total. La lectura de Parquet, la
construcción de índices y el calentamiento histórico permanecen en el coste
de fase. No se convertirán sus medias en percentiles de predicción.

Cada repetición debe partir del mismo estado confirmado y del mismo snapshot,
con las mismas filas por lote. Repetir una llamada que ya haya avanzado los
pesos rápidos cambia la carga medida. Se registrarán los tamaños reales de
lote, calentamiento técnico, repeticiones, sincronización CUDA, p50/p95/p99,
caudal y picos de RAM y VRAM. No se comparará una ruta ya preparada en GPU
con otra que incluya lectura y transferencias sin desglosar esa diferencia.

Este documento fija el contraste y las medidas pendientes. No contiene
resultados de entrenamientos ni latencias de checkpoints reales.
