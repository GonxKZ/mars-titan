# Escalado emparejado de 64 a 256 activos

Esta revisión amplía el [estudio de refinamientos](refinement-development-v1.md)
y su [contraste de representaciones](refinement-representations-v1.md) con dos
tamaños de cohorte. Cada tamaño conserva las representaciones original y
geométrica, las semillas 42, 43 y 44, los núcleos `linear_v1` y `mac_paper`,
las dos épocas y el catálogo de lectores, rutas y controles. No incorpora
otra búsqueda de hiperparámetros. Hay 72 trabajos por tamaño y representación.

Las dos cohortes usan la misma sal y el mismo ranking SHA256 entre activos
US con filas de ajuste anteriores a abril de 2021. Las 64 acciones son el
prefijo de las 256 en ese ranking. Ordenarlas por símbolo para serializarlas
no cambia esa selección. No se utilizan objetivos numéricos ni cobertura
posterior para elegirlas. Se conservan todas las filas admitidas de esos
activos desde 2000 y las ausencias explícitas, sin inventar historia anterior
a su disponibilidad.

## Configuración y conservación de estados

La revisión 2 de `configs/experiments/refinement-development-v1.json` declara
los tamaños 64 y 256. El tamaño predeterminado sigue siendo 64.
`benchmarks.refinement_study prepare` acepta `--cohort-size 64` o
`--cohort-size 256`, sin editar el JSON entre ejecuciones. `verify` acepta
el mismo argumento para exigir una cohorte concreta. `run` utiliza la
identidad confirmada en `prepared.json` y rechaza un cambio de tamaño en la
línea de órdenes.

Cada preparación registra tamaño, activos, su huella y la identidad de los
64 activos comunes. La configuración, las recetas y la auditoría de capacidad
forman parte del recibo. Los cuatro estudios requieren un runtime común nuevo
y directorios propios. Las preparaciones y runtimes anteriores se conservan
sin reescribir sus hashes. No se transfieren checkpoints entre tamaños ni
entre representaciones.

Los bloques de núcleo y lector quedan acotados a 256 filas en ambos tamaños.
No cambia la definición de actualización por instante ni la profundidad del
modelo. Los checkpoints conservan estado rápido, banco, pendientes, RNG,
optimizador y cursor según el contrato existente. Las identidades de las
recetas impiden confundir este recorrido con uno anterior de otra capacidad.

## Admisión por capacidad

`benchmarks/refinement_study_capacity.py` reutiliza los registros de
`financial_observations._records` para contar emisiones y maduraciones, sin
ejecutar el predictor. Las emisiones preceden a las maduraciones del mismo
instante. Las predicciones sin etiqueta siguen contándose hasta el cierre
de la fase. La comprobación no se limita a las filas finalmente puntuadas.

La colección conserva el máximo de 32.768 trazas pendientes y 131.072 filas
maduras. Si una fase no cabe, la preparación se detiene con sus recuentos,
sin descartar casos ni ampliar límites. El límite de trazas pendientes se
aplica a la colección de ajuste y validación de la puerta. Los pendientes
del padre y de otras fases también se registran, pero no son esa misma cola.

La [auditoría de la cohorte de 256](../../reports/engineering/refinement-scaling-capacity-20261018/verification.json)
observó un máximo de 512 trazas pendientes tanto en el ajuste como en la
validación de la puerta. El padre desde 2000 alcanzó 16.389 predicciones
pendientes y terminó con 16.133 sin resolver. La evaluación de desarrollo de
2023 tuvo 63.679 etiquetas, un máximo de 513 pendientes y 257 sin resolver
al cierre. El máximo de filas de una decisión fue 256. Son recuentos de
disponibilidad, no resultados predictivos.

El buffer de escritura sigue siendo de 1.024 filas. La lectura de tablas de
trazas conserva el límite de 64 MiB. La memoria de modelos, grafos, snapshots
y checkpoints se medirá durante la pareja piloto. Los recuentos anteriores
no acreditan por sí solos un pico de RAM o VRAM del entrenamiento.

## Presupuesto y contraste

El límite conjunto es de dieciséis horas locales, con ocho horas por tamaño
de cohorte y seis por estudio de representación. No se exige agotarlas. La
supervisión conserva los intentos y saldos al reanudar y aplica el menor
saldo disponible a cada hijo. El supervisor anterior de dos preparaciones
queda conservado. Esta ampliación requiere una revisión identificada para
las cuatro preparaciones.

La primera pareja de núcleos de cada tamaño mide el coste antes de ampliar
semillas. Las representaciones se alternan dentro de la misma etapa. La
revisión de coste no cambia los candidatos según sus resultados de 2023.
Las combinaciones no terminadas permanecen identificadas como pendientes.

El efecto de pasar de 64 a 256 se calcula también sobre las 64 acciones
comunes, con las mismas identidades de fila y objetivos. Se filtran las
predicciones ya emitidas. No se vuelve a ejecutar el modelo de 256 con un
banco que haya visto solo 64 activos. Así se mide el recorrido desplegado
con cada población, incluido su contexto y la competencia por la memoria.

Los resultados de toda la población de 256 se presentan aparte. Comparar
directamente su MAE con el de otra mezcla de 64 activos no aísla el efecto
del escalado. Se informan filas, épocas, actualizaciones, tiempo y memoria
para cada tamaño. Igual número de épocas no implica igual exposición a filas
ni igual coste. Se mantienen MAE por sesión, emparejamiento temporal e
intervalos por bloques. Ninguna fecha se convierte en varias observaciones
independientes por aparecer en varias semillas.

El test final de 2024 y la campaña completa siguen cerrados. Las restricciones
de atribución del contraste ResNet/geometría permanecen vigentes. Este
preregistro y la auditoría de capacidad no acreditan una mejora de MAE.
