# Matriz de experimentos y reglas de comparación

Estado: diseño previo a la implementación. Ninguna celda representa un resultado ejecutado.

## Familias y prioridad

| ID | Familia | Función en la comparación | Prioridad |
| --- | --- | --- | --- |
| B0 | Predicción residual cero y media histórica permitida | Referencias ingenuas que detectan si hay señal útil. | Obligatoria |
| B1 | Ridge con variables históricas | Referencia lineal regularizada, fácil de interpretar. | Obligatoria |
| B2 | Boosting tabular compacto | Contraste no lineal. Comenzar con HistGradientBoosting de scikit-learn. | Obligatoria |
| B3 | GRU compacta | Referencia neural secuencial con presupuesto comparable. | Obligatoria |
| B4 | DLinear | Control de complejidad de bajo coste, con adaptación documentada al target. | Deseable tras el piloto |
| B5 | TCN o PatchTST reducido | Contraste adicional si aporta una pregunta distinta. | Extensión |
| M0 | Codificador y cabeza de MARS-TITAN sin memoria | Aísla el efecto de introducir memoria. | Obligatoria |
| M1 | Memoria global con escritura uniforme | Aísla la selección de eventos. | Obligatoria |
| M2 | Memoria global y escritura por error maduro | Contrasta sorpresa predictiva frente a combinación económica. | Obligatoria |
| M3 | Propuesta compacta con sorpresa, régimen e incertidumbre | Modelo principal de investigación. | Obligatoria |

La elección de boosting aprovecha una dependencia ya prevista y evita incorporar dos librerías equivalentes al inicio. XGBoost o LightGBM pueden sustituirla mediante una decisión registrada. No se ejecutarán todos solo para ampliar la tabla de resultados.

## Ablaciones emparejadas

| Contraste | Mantener constante | Confusor que se debe medir |
| --- | --- | --- |
| M3 frente a M0 | Entradas, cabeza, particiones, semillas y ajuste. | Diferencia de parámetros, tiempo y longitud efectiva de contexto. |
| Escritura selectiva frente a uniforme | Capacidad de memoria y presupuesto de actualización. | Número de escrituras. Comparar también escritura aleatoria con presupuesto igual cuando sea viable. |
| Sorpresa completa frente a error | Escalas y calibración obtenidas del pasado. | Disponibilidad de indicadores económicos y cantidad de etiquetas maduras. |
| Régimen frente a memoria global | Señales y capacidad total. | Número de bancos y parámetros extra. Regímenes filtrados, sin suavizado futuro. |
| Precios frente a precios+texto | Muestras comunes y misma disponibilidad temporal. | Selección por cobertura de noticias y versión del codificador. |
| Con/sin fundamentales o gráficos válidos | Universo y reloj de decisión. | Información redundante del gráfico respecto a precios. Revisiones contables. |
| Con/sin abstención | Predicciones y periodo. | Capital expuesto, cobertura, costes y número de operaciones. |
| Memoria congelada frente a adaptativa | Estado inicial y regla predefinida. | Uso de etiquetas durante evaluación. No confundir política online con reajuste retrospectivo. |

La memoria por mercado, sector y activo se añade solo después de validar el caso global. Si no cabe en el presupuesto, se documentará el límite y se contrastará una versión reducida, conservando los objetivos de la comparativa.

## Presupuesto inicial propuesto

Hasta 64 activos, ventanas de 64 sesiones, dimensión latente 64, minibatch inicial 16 y máximo 30 épocas con parada temprana. Estas cifras son **puntos de partida por medir**. El estado de memoria debe preservarse en orden temporal aunque se agrupen activos. No se puede mezclar aleatoriamente el eje temporal de una secuencia adaptativa.

Como límite de planificación, hasta diez configuraciones por familia y tres semillas en la comparación confirmatoria. Se registrará el tiempo total de búsqueda, no solo el entrenamiento del mejor modelo. Los codificadores de texto se precalcularán con pesos congelados cuando sea apropiado. La primera medición de memoria fijará el margen operativo por debajo de los 8 GB físicos, incluyendo activaciones, optimizadores y actualización interna.

## Identificador y evidencia

Cada ejecución futura tendrá `run_id`, hash de configuración, commit, hash del manifiesto, semillas, entorno, modo de memoria, corte temporal, horas GPU, VRAM máxima y motivo de finalización. Conservará predicciones por activo/instante, métricas por sesión y eventos de actualización. Las tablas resumidas se generan desde esas evidencias, sin transcribir cifras manualmente.

No se reemplaza una ejecución fallida por la siguiente sin registrarla. Antes del test final se fija qué comparaciones son confirmatorias y cuáles exploratorias. Las tablas de resultados deberán mostrar todas las familias obligatorias, el número de activos/sesiones válidos y el coste computacional.
