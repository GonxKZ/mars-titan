# Fuente financiera madura para N1

`NeuromodulationSource` implementa el estado de señales previsto para [N1](dual-neuromodulation-v1.md). Recibe filas `MatureWrite` y produce un snapshot por flujo. `neuromodulation_session` lo conecta al enlace ordinario de Titans en `FinancialSession`. El contador recibido y un dataclass no prueban por sí solos la procedencia. El enlace obtiene las filas de resultados nativos y comprueba su concordancia al recuperar. N1 sigue desactivado por defecto.

## Estadísticas y orden

Cada fila contiene flujo, fecha de predicción, disponibilidad, punto emitido y residuo maduro. Toda la disponibilidad se valida antes de leer valores financieros. Se rechazan fechas futuras, filas repetidas en el instante, cortes que retroceden, generaciones no consecutivas o un contador aplicado que no concilia con los nuevos resultados. La recuperación contrasta ese contador y las filas con el registro nativo de la generación confirmada.

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

## Publicación con la sesión nativa

La sesión conserva dos referencias, `neuromodulation_before` y `neuromodulation`. La primera reproduce el snapshot usado para predecir. La segunda incorpora los resultados que el ejecutor aplicó después de emitir. Se guardan con el banco, los pesos rápidos y la cola pendiente en la misma generación. No se encadena una copia completa de todas las generaciones.

El contrato identifica la configuración y los módulos de fuente y enlace. Cada bloque recibe una vista de las mismas parejas por identidad de flujo. Cambiar el tamaño de bloque no recalcula estadísticas ni permite que un activo vea resultados procesados durante la predicción de otro. El calentamiento no inventa predicciones emitidas. Una liquidación sin observaciones actualiza resultados y no crea un snapshot de decisión.

Al recuperar se comprueban generación, corte y traza anterior. El enlace vuelve a construir las filas desde `record.feedback`, incluidos el punto emitido y el objetivo, y exige el contador `snapshot.applied` del ejecutor. Recalcula el estado posterior completo y lo compara con el conservado. Esto detecta una modificación del punto emitido aunque se haya recalculado el checksum de la fuente. Los checksums siguen sin ser autenticación frente a quien pueda sustituir de forma coordinada todos los artefactos del ejecutor.

La poda conserva las referencias anterior y posterior que necesitan los checkpoints retenidos. Un fallo antes de confirmar no publica el avance. Se comprueban recuperación, repetición del siguiente evento, orden de activos, bloques de una y cuatro filas, admisión M0–M3 y control pasado. Con coeficientes nulos, las predicciones y los estados rápidos coinciden exactamente con la referencia sin N1. Una señal madura no nula cambia los pesos rápidos y deja intactos los parámetros compartidos.

Esta integración no completa todos los cruces de variantes. El enlace de mecanismos y E2 todavía necesitan registro propio. No se han ejecutado sus mezclas con N1, la ruta CUDA, medidas de coste integral ni comparaciones predictivas.

El [recibo de la integración nativa](../../reports/engineering/neuromodulation-native-20261101/verification.json), del 1 de noviembre, conserva 317 identificadores de caso CPU distintos. La selección principal pasa 309 casos y excluye 54 CUDA. La ampliación pasa 28 y la inspección inicial uno, con 21 identificadores compartidos. Son 338 ejecuciones, no 338 casos distintos. No hubo omisiones en esas selecciones.

Se detectaron 18 mutaciones distintas con G1 y G2 verdes y M1 rojo. La primera pasada detectó 13 de 14 y dejó inconcluyente la supresión de la comprobación de generación: otra validación seguía rechazando el estado. El caso reforzado cambia también la traza para que coincida con la generación alterada y detecta el defecto por separado. Se añadieron cuatro defectos de registro, configuración e inspección inicial. La repetición no cuenta como un defecto nuevo. No hubo intentos de optimizador y la protección de aprendizaje conservó su huella.

`neuromodulation_session.py` cubre sus 42 sentencias y 10 ramas. Los cinco archivos seleccionados cubren 1216 de 1555 sentencias y 476 de 724 ramas. Las 29 funciones modificadas están medidas. El CRAP máximo es 376,97 en la función preexistente `FinancialSession._prepare_event`, con cuatro líneas ejecutables nuevas cubiertas. La cobertura del resto de esa función sigue siendo parcial. Estas cifras no sustituyen la comprobación funcional ni demuestran ausencia de errores. Se reutilizó el binario nativo existente, sin modificarlo ni recompilarlo.
