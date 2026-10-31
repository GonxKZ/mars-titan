# Fuente financiera madura para N1

`NeuromodulationSource` implementa el estado de señales previsto para [N1](dual-neuromodulation-v1.md). Recibe filas `MatureWrite` y produce un snapshot por flujo. La conexión con el registro y la publicación del ejecutor nativo sigue pendiente. El contador recibido y un dataclass no prueban por sí solos que una predicción fue emitida ni que un resultado no se repite históricamente. Los consumidores N1 continúan sin habilitarse automáticamente.

## Estadísticas y orden

Cada fila contiene flujo, fecha de predicción, disponibilidad, punto emitido y residuo maduro. Toda la disponibilidad se valida antes de leer valores financieros. Se rechazan fechas futuras, filas repetidas en el instante, cortes que retroceden, generaciones no consecutivas o un contador aplicado que no concilia con los nuevos resultados. El registro nativo debe confirmar después ese contador y la procedencia de las filas.

Las filas se ordenan por disponibilidad, decisión y flujo. Se agrupan por mercado y corte de publicación. Para cada grupo se calculan RMS del residuo, RMS del error `target-issued` y media del error absoluto. Se utiliza escalado antes de sumar, para evitar que cuadrados o sumas intermedias desborden cuando sus resultados serían finitos. Un error financiero no representable se rechaza y no modifica el estado previo.

La ventana guarda los últimos W instantes con resultados de cada mercado. No cuenta filas, días vacíos ni sesiones de calendario. Se conserva un historial de W+1 parejas normalizadas para comprobar el control pasado al restaurar. W está entre 2 y 256 y el instante admite hasta 40960 resultados, con un límite configurable menor. No se construye una matriz de distancias ni se guarda toda la historia.

Para la señal esperada se usa el RMS de los RMS de residuos de la ventana. Se transforma en `e = rms/(rms+expected_scale)` con una evaluación que evita cocientes desbordados. Es una proxy elegida para la variante, no una desviación típica centrada ni una identificación de ruido aleatorio. Puede ser alta aunque las predicciones sean perfectas.

Para el componente inesperado, con escala del error de la ventana anterior y un piso positivo:

```text
scale = max(scale_floor, rms_error_previous_window)
z = clip(mae_current/scale, 0, k+H)
k = cusum_reference + cusum_slack
S_next = clip(S_previous + z - k, 0, H)
u = S_next/H
```

El estado empieza sin señal y espera `min_events` para disponer del pasado requerido. H es `cusum_limit`. Es un CUSUM saturado y normalizado propio. La [referencia clásica de CUSUM](https://doi.org/10.1093/biomet/41.1-2.100) es un antecedente, no la demostración de esta regla. Se comprobó la ficha bibliográfica, no se reprodujo el algoritmo ni sus garantías. La variante no hereda probabilidades de falsa alarma, tiempos medios de detección o optimalidad. Ventana, escala, piso, referencia y límite son parámetros que deben fijarse antes de evaluar.

## Publicaciones del mismo instante

El estado conserva la base anterior al instante y los resultados admitidos en ese corte. Si llegan varias generaciones con el mismo corte, la fuente recompone un único grupo por mercado desde esa base y todas sus filas. Una generación vacía no añade un evento estadístico.

Un snapshot para el propio instante siempre lee la base anterior. Solo una predicción posterior puede leer las estadísticas recién publicadas. Las filas reordenadas y la partición en generaciones del mismo instante conservan las señales siguientes. Los números de generación y sus huellas pueden diferir, porque representan publicaciones distintas.

El snapshot contiene parejas inmutables por flujo, configuración, generación, corte y huella de estado. `array()` devuelve una copia FP64. Modificarla no cambia el estado. Las huellas y la metadata son procedencia, no entradas predictivas.

## Control pasado y recuperación

`shuffled_past_prefix` mantiene las estadísticas de los resultados observados y sustituye solo la pareja consultada por una de las W parejas anteriores, excluyendo la actual. El generador deriva de semilla, mercado y contador y no altera el RNG global. La semilla se conserva en la configuración. No es una permutación global ni conserva la distribución contemporánea o la recencia.

El estado tiene esquema versionado, configuración, generación, corte, contador aplicado, base y resultados del instante. El estado corriente se reconstruye al cargar. Se comprueban tamaño, forma, finitud, ventanas, historial y concordancia de señales. Las exportaciones no comparten listas con el estado. El checksum detecta cambios de bytes y no autentica una historia falsa autoconsistente. La cadena de confianza y la unicidad histórica deben proceder del ejecutor nativo.

## Evidencia y límites

El [recibo](../../reports/engineering/neuromodulation-source-20261031/verification.json) conserva 172 casos CPU correctos, cero omisiones y un CUDA anterior excluido. Incluye las señales previas y el núcleo N1. Se comprueban cálculos manuales, mercados separados, datos futuros, publicación parcial, eventos vacíos, recuperación, presupuestos, control pasado, RNG y valores extremos.

La primera pasada detectó 18 de 19 mutaciones. La sustitución de error por residuo sobrevivió porque la serie del fixture era constante y su escala ocultaba la diferencia. Se cambió ese caso a un salto del residuo con predicción perfecta. El error debe seguir siendo cero y no activar el CUSUM. La repetición detectó el defecto con G1/G2 verdes y M1 rojo. Las dos pasadas se conservan y el resultado final cuenta 19 defectos distintos.

El módulo cubre 189 de 204 sentencias y 53 de 66 ramas. Las veinte funciones nuevas están medidas y el CRAP máximo es 30,29 en la validación del estado base. La cobertura de errores de validación es parcial. No se han medido latencia, RAM/VRAM ni calidad predictiva y no se ha ejecutado aprendizaje.

El siguiente enlace debe contrastar las filas con predicciones nativas realmente emitidas, conservar los snapshots del evento y confirmar el estado con memoria y cola pendientes. También debe reproducir la señal siguiente al recuperar y detectar referencias de otra configuración. Hasta entonces esta fuente no completa la variante financiera N1.
