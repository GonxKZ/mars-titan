# Verificación de la campaña de comparación

Fecha: 13 de noviembre de 2026. El código prepara la campaña de [#836](https://github.com/GonxKZ/mars-titan/issues/836). La campaña principal no se ha ejecutado. El piloto usa el origen 2018 y conserva cerrado el test final de 2024.

La [evidencia de verificación](verification.json) recoge los resultados, las versiones y las huellas de las fuentes. Las [medidas del piloto](pilot.json) conservan los errores, tiempos y límites de cada recorrido. El [dimensionado](budget.json) separa los supuestos de los tiempos medidos.

## Comprobaciones

- Última pasada de campaña, datos, controles y lectores con el enlace nativo: 374 pruebas superadas y tres casos CUDA excluidos de esa pasada CPU.
- Revisión final de parada cooperativa, identidad de dependencias y recuperación: 116 pruebas superadas.
- Escalas, gradientes y recuperación del predictor en `cuda:0`: 32 pruebas superadas, sin omisiones ni pasos de optimizador.
- Guardas e inventarios: 198 pruebas superadas. El inventario estático se volvió a comprobar en la pasada final.
- Ventanas cronológicas: 68 pruebas superadas y una omitida por la protección de aprendizaje.
- Modelos y selección en CPU: 4.039 pruebas superadas y 1.131 omisiones por CUDA, recursos externos o aprendizaje bloqueado. La verificación CUDA se informa por separado.
- Mutación dirigida: 41 defectos introducidos detectados. Cada caso conserva controles en verde antes y después y cero intentos de optimizador.
- Ensayo inicial del motor: 28 de 28 modelos terminados, con 64 activos y una época.
- Ensayo trazado: 13 trabajos terminados. Los 195 archivos relevantes abiertos están incluidos en el paquete de 710 archivos de esa cohorte.
- LightGBM con datos reales de 64 activos y tres semillas: la ejecución continua y las llamadas reanudables por semilla dejaron exactamente los mismos informes, modelos y predicciones. Ambas conservaron las 1.000 rondas del control.

Las pasadas se solapan. No se suman como pruebas únicas. La regresión inicial detectó una aserción que exigía el mismo límite de historia para la criba y para las ventanas generales. La criba conserva doce meses y las ventanas generales admiten hasta 300. La prueba se corrigió para comprobar esos dos comportamientos y su archivo completo pasó después.

## Cobertura y complejidad

Se usaron coverage.py 7.16.1, con ramas, y Radon 6.0.1. La complejidad se calcula sobre las regiones de función que identifica coverage.py, con la convención de [`summarize_python_quality.py`](../../../scripts/summarize_python_quality.py). CRAP se calcula como CC²·(1 − cobertura)³ + CC, con la cobertura de sentencias de cada función.

En los archivos analizados se recorrieron 4339 de 4542 sentencias y 1559 de 1690 ramas. Entre las líneas ejecutables modificadas se cubrieron 1276 de 1304. Se midieron 140 funciones modificadas y ninguna quedó sin medir. El mayor CRAP fue 59.40, en `FinancialPredictor._prepare`. Sus dos líneas modificadas están cubiertas. El validador nuevo de la declaración tiene CC y CRAP de 58, aun con todas sus sentencias recorridas.

Las regiones y las líneas son convenciones distintas y sus recuentos no se mezclan. Antes de combinar las coberturas se retiraron los datos antiguos de los archivos que habían cambiado de nuevo. La cobertura y CRAP localizan complejidad que merece revisión, pero no prueban ausencia de errores.

## Fallos encontrados con datos reales

La preparación de LightGBM alcanzó los límites de 11 y 10 GiB del proceso y el sistema registró `oom-kill`. Se separó la carga de los tramos y se liberaron los bloques que ya estaban concatenados. El control de 1.024 activos terminó tras recuperar una semilla confirmada. En la campaña, cada intento ajusta como máximo una semilla nueva y la emisión se hace después, en otro proceso. Se conservan los modelos y la ronda seleccionada junto a sus predicciones.

El puente necesitaba 145.825.792 bytes según su estimación de repetición diferenciable, por encima de los 134.217.728 declarados. El presupuesto del lector pasa a 256 MiB. Una prueba construye todos los lectores declarados con el lote y el banco completos. El recorrido del puente con 1.024 activos terminó con esa configuración.

Se corrigieron también los reintentos de procesos que terminaban antes de escribir un registro, la parada que llega durante la creación de un proceso, el código de salida de una ejecución limitada a un nivel, el permiso para código sin confirmar y la importación de paquetes. La cola tiene una fecha límite y pide una parada recuperable al alcanzarla.

## Límites

Las pruebas se ejecutaron localmente. No se ha comprobado WSL2 ni el rendimiento real de la RTX 5060 Laptop del segundo equipo. El entorno fijado incluye soporte `sm_120`, pero el segundo equipo debe completar `doctor` y la comprobación CUDA indicada en la guía.

El piloto neuronal usa una ventana y una semilla. Las diferencias de error no sustituyen el análisis temporal de la campaña. No se midieron energía, coste económico ni percentiles de latencia. La última revisión del cambio fue local, sin un segundo revisor.

El piloto tabular de 1.024 activos se ejecutó con la retención anterior: conserva sus predicciones y una copia privada del modelo de la semilla 42. La versión preparada conserva un modelo por semilla, junto con la ronda elegida y su huella.
