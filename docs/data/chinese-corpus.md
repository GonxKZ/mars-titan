# Corpus chino compartido y ventanas temporales

`combine_chinese_samples` reúne ediciones completas producidas por
`prepare_chinese_samples`. Recibe sus directorios y un destino nuevo. La opción
`market_factors` permite indicar un descriptor CN común con el que recalcular
toda la supervisión. La unión reutiliza los vectores ya calculados, sin cargar
codificadores ni repetir inferencia.

Las ediciones deben compartir censo padre, calendario, contexto de 64 sesiones,
codificadores, conceptos contables CNY, catálogo y panel de 140 indicadores.
Cada símbolo aparece una sola vez y pertenece al censo declarado. El resultado
conserva los hashes de sus ediciones y etapas. El número de activos unidos se
distingue del tamaño del censo original.

El destino contiene copias independientes de los precios, noticias,
fundamentales y muestras codificadas. Los bytes de las modalidades se conservan.
La identidad queda fijada antes de copiar y un bloqueo excluye escrituras
simultáneas. Las copias confirmadas se reutilizan al recuperar una interrupción.
El recibo final se escribe después de verificar fuentes, configuración y
supervisión. Una edición completada que se altera se rechaza.

Los límites son 1.024 ediciones, un millón de muestras en total, 200.000 filas
por archivo y 64 MiB por archivo y por tabla descomprimida. Se procesan los
activos por separado. Estos límites permiten representar el censo de 892
candidatos, pero no demuestran que todos sean admisibles ni que se haya medido
la unión completa.

## Edición real comprobada

La unión de Ping An y Vanke conserva las 721 muestras de la
[historia contable revisada](chinese-fact-history.md). La
[ampliación del CSI 300 a 2021](csi300-market-factor.md#ampliación-con-los-boletines-de-2021)
recupera 46 etiquetas por calentamiento. La división anual preliminar tiene
266 filas de entrenamiento y 451 de validación. Quedan dos fronteras anuales
y dos cortes finales. El cambio de factor también modifica etiquetas anteriores,
por lo que la supervisión tiene otra identidad.

El [protocolo temporal CN](../../configs/evaluation/chinese-real-walk-forward.json)
conserva los parámetros estadounidenses y cambia el calendario de mercado.
Tiene entrenamiento expansivo, dos meses de validación, uno de calibración,
uno de evaluación y separación de una sesión. La maduración de la etiqueta
determina la admisión en cada partición. La recuperación del corte anual se
activa explícitamente y no recupera etiquetas que crucen la reserva de 2024.

| Ventana | Entrenamiento | Validación | Calibración | Evaluación |
| --- | ---: | ---: | ---: | ---: |
| 000 | 225 | 71 | 38 | 42 |
| 001 | 266 | 70 | 42 | 34 |
| 002 | 298 | 82 | 34 | 38 |
| 003 | 338 | 78 | 38 | 33 |
| 004 | 382 | 74 | 33 | 38 |
| 005 | 418 | 73 | 38 | 44 |
| 006 | 458 | 72 | 44 | 38 |
| 007 | 493 | 84 | 38 | 21 |
| 008 | 532 | 84 | 21 | 38 |
| 009 | 578 | 61 | 38 | 36 |

Las ventanas reúnen 715 muestras distintas. Las 362 filas de evaluación no se
repiten entre ventanas. Los recuentos de entrenamiento y otros roles se solapan
a lo largo del tiempo y no se suman como observaciones independientes.
El lector ha recorrido las cuarenta particiones y ha comprobado dimensiones,
valores finitos, disponibilidad de las entradas y las 140 máscaras macro activas.
Cuatro filas con etiqueta válida quedan fuera de todas las ventanas por sus
fronteras y separaciones temporales. Corresponden al 31 de octubre y al
30 de noviembre de 2023 en ambas empresas.

La comprobación CPU/CUDA recorre un lote de 64 filas de entrenamiento con
RNN, LSTM, GRU y DLinear. Las entradas transferidas coinciden exactamente, la
diferencia máxima del forward es 7,16 × 10⁻⁷ y todos los gradientes son finitos.
Se utilizan pesos iniciales, dropout cero y ninguna actualización del
optimizador. No son resultados predictivos de modelos entrenados. La campaña
activa se reanudó al terminar.

El proceso de reutilización y lectura de las cuarenta particiones tarda
7,86 segundos y alcanza 732.668 KiB de RAM. La comprobación CUDA tarda
4,00 segundos de proceso, con 1.268.452 KiB de RAM y picos de PyTorch de
109.006.848 bytes asignados y 148.897.792 reservados. Son ejecuciones
funcionales de una repetición. El recibo conserva los reintentos de las
comprobaciones y sus causas, sin afirmar una aceleración.

Pasan 253 pruebas relacionadas y siete mutaciones dirigidas de la unión.
La cobertura del módulo es del 89,88 % de sentencias y del 78,85 % de ramas.
La mayor complejidad se concentra en `_source`, que valida la procedencia de
cada edición, con CCN 87 y CRAP 100,24 según la convención registrada.
La cobertura no demuestra ausencia de defectos. La revisión focal comprobó
hashes obligatorios, pertenencia al censo, disponibilidad, contexto, máscaras
macro y rechazo de una supervisión completada que se haya corrompido.

El [recibo de verificación](../../reports/data/chinese-corpus-20261007.json)
distingue preparación y pruebas técnicas de entrenamiento científico. El
resultado sigue marcado `development_snapshot`, `cohort_complete=false` y
`training_ready=false`. Hay dos empresas revisadas, frente a 892 candidatos
originales y 810 instrumentos con cuatro fuentes potenciales. Queda ampliar
esa cobertura antes de ejecutar la comparación china completa. Los textos
mantienen la cohorte `original_audited`, sin atribuirles verificación editorial
externa. El factor sigue siendo retrospectivo y la simulación financiera con
posiciones reales requiere resolver la procedencia de los ajustes OHLC.
