# Matriz de experimentos y reglas de comparación

Estado: diseño de la comparación confirmatoria. Ninguna celda representa un resultado de eficacia observado. Las sondas MLP y GRU del [presupuesto experimental](../../reports/resources/campaign-budget.md) miden coste, no completan esta comparación.

## Familias y prioridad

| ID | Familia | Función en la comparación | Prioridad |
| --- | --- | --- | --- |
| B0 | Predicción residual cero y media histórica permitida | Referencias ingenuas que detectan si hay señal útil. | Obligatoria |
| B1 | Ridge con variables históricas | Referencia lineal regularizada, fácil de interpretar. | Obligatoria |
| B2 | Boosting tabular compacto | Contraste no lineal. Comenzar con HistGradientBoosting de scikit-learn. | Obligatoria |
| B3 | GRU compacta | Referencia neural secuencial con presupuesto comparable. | Obligatoria |
| B4 | DLinear | Control de complejidad de bajo coste, con adaptación documentada a la variable objetivo. | Deseable tras el piloto |
| B5 | TCN o PatchTST reducido | Contraste adicional si aporta una pregunta distinta. | Extensión |
| B6 | Memoria asociativa compacta con regla delta | Control simplificado, separado de Titans-MAC y del banco episódico. | Secundaria |
| B7 | Transformer compacto | Aislar el cambio de codificador frente a la GRU, con fusión y cabeza comunes. | Obligatoria |
| B8 | Titans-MAC adaptado | Atención cercana, memoria neuronal con momentum y olvido, y parámetros persistentes aprendidos. | Obligatoria |
| B9 | MARS-TITAN sobre Titans-MAC | Incorporar las modificaciones acordadas como componentes desactivables. | Obligatoria |
| M0 | Codificador y cabeza de MARS-TITAN sin memoria | Aísla el efecto de introducir memoria. | Obligatoria |
| M1 | Memoria global con escritura uniforme | Aísla la selección de eventos. | Obligatoria |
| M2 | Memoria global y escritura por error maduro | Contrasta sorpresa predictiva frente a combinación económica. | Obligatoria |
| M3 | Propuesta compacta con sorpresa, régimen e incertidumbre | Modelo principal de investigación. | Obligatoria |

Las familias B7–B9 corresponden a la [dirección arquitectónica del 8 de octubre](titans-mac-architecture.md) y no representan resultados ejecutados. B0–B9 son identificadores de familias de esta tabla. No fijan el baseline B de CM-v1, que necesita su identidad completa. M0–M3 conservan las políticas del banco episódico. Si una arquitectura incluye memoria neuronal, debe indicar que desactivar el banco no retira esa otra memoria.

La elección de boosting aprovecha las implementaciones existentes. Cada cambio de familia mantiene identidad y presupuesto propios. No se ejecutan modelos solo para ampliar la tabla de resultados.

HistGradientBoosting no debe reservar aleatoriamente una parte del panel para parada temprana. Se utilizará `early_stopping=False` con selección externa cronológica, o un conjunto de validación explícito cuando la versión fijada lo admita. La [documentación de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) describe la activación automática de parada y la reserva interna. Leer el dataset por bloques no convierte ese estimador en un algoritmo incremental.

El contraste principal de refinamiento usa K = 1. K = 2 o 4 pertenece a MT-054 y no cuenta las actualizaciones asociativas de Titans. La edición histórica utiliza todas las filas admisibles con sus máscaras y conserva el control estricto por separado. Las nuevas arquitecturas y sus ampliaciones siguen sin entrenamientos ni evaluaciones científicas mientras esté vigente el bloqueo histórico.

## Ablaciones emparejadas

| Contraste | Mantener constante | Confusor que se debe medir |
| --- | --- | --- |
| M3 frente a M0 | Entradas, cabeza, particiones, semillas y ajuste. | Diferencia de parámetros, tiempo y longitud efectiva de contexto. |
| Escritura selectiva frente a uniforme | Capacidad de memoria y presupuesto de actualización. | Número de escrituras. Comparar también escritura aleatoria con presupuesto igual cuando sea viable. |
| Sorpresa completa frente a error | Escalas y calibración obtenidas del pasado. | Disponibilidad de indicadores económicos y cantidad de etiquetas maduras. |
| Régimen frente a memoria global | Señales y capacidad total. | Número de bancos y parámetros extra. Regímenes filtrados, sin suavizado futuro. |
| Alternativas de codificación y agregación textual | Las cuatro modalidades, muestras comunes y disponibilidad temporal. | Cobertura de noticias, versión y coste del codificador. |
| Alternativas de representación contable y visual | Las cuatro modalidades, universo y reloj de decisión. | Redundancia del gráfico respecto a precios, revisiones contables y dimensión de las entradas. |
| Con/sin abstención | Predicciones y periodo. | Capital expuesto, cobertura, costes y número de operaciones. |
| Memoria congelada frente a adaptativa | Estado inicial y regla predefinida. | Uso de etiquetas durante evaluación. No confundir política online con reajuste retrospectivo. |

La memoria por mercado, sector y activo se añade solo después de validar el caso global. Si no cabe en el presupuesto, se documentará el límite y se contrastará una versión reducida, conservando los objetivos de la comparativa.

## Presupuesto inicial propuesto

Hasta 64 activos, ventanas de 64 sesiones, dimensión latente 64, lote inicial de 16 muestras y máximo 30 épocas con parada temprana. Estas cifras son **puntos de partida por medir**. El estado de memoria debe preservarse en orden temporal aunque se agrupen activos. No se puede mezclar aleatoriamente el eje temporal de una secuencia adaptativa.

Como límite de planificación, hasta diez configuraciones por familia y tres semillas en la comparación confirmatoria. Se registrará el tiempo total de búsqueda, no solo el entrenamiento del mejor modelo. Los codificadores de texto se precalcularán con pesos congelados cuando sea apropiado. La primera medición de memoria fijará el margen operativo por debajo de los 8 GB físicos, incluyendo activaciones, optimizadores y actualización interna.

## Adaptadores de postentrenamiento

[La matriz de adaptadores](../../configs/posttraining/adapter-matrix-v1.json) fija antes del primer ajuste qué partes de un padre congelado pueden cambiar. Combina cabeza, consulta y salida de la lectura y fusión en sus siete combinaciones de uno, dos y tres puntos, más un control de rango completo en la fusión. Cada semilla conserva tres controles: la salida del padre congelado, la corrección lineal residual inicializada a cero y la continuación de todos los parámetros. Todos los brazos tienen las mismas filas, semillas, lote, épocas, actualizaciones y validación temporal.

La pregunta es dónde conviene adaptar, no cuántos parámetros caben. Por eso la matriz registra parámetros entrenables y estados invalidados. Un punto solo quedaría justificado si mejora al padre y a la corrección lineal con la misma métrica, y esa comparación es la que permite descartarlo. La lectura solo existe en el Transformer compacto entre las referencias actuales. Titans-MAC y la lectura episódica tienen destinos declarados, pero su ajuste depende del entrenador cronológico. Los detalles técnicos están en el [postentrenamiento con la edición histórica](../engineering/masked-posttraining.md). No hay resultados de esta matriz.

## Identificador y evidencia

Cada ejecución futura tendrá `run_id`, hash de configuración, commit, hash del manifiesto, semillas, entorno, modo de memoria, corte temporal, horas GPU, VRAM máxima y motivo de finalización. Conservará predicciones por activo/instante, métricas por sesión y eventos de actualización. Las tablas resumidas se generan desde esas evidencias, sin transcribir cifras manualmente.

No se reemplaza una ejecución fallida por la siguiente sin registrarla. Antes del test final se fija qué comparaciones son confirmatorias y cuáles exploratorias. Las tablas de resultados deberán mostrar todas las familias obligatorias, el número de activos/sesiones válidos y el coste computacional.
