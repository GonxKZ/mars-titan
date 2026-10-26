# Lesiones de lectura sobre la misma trayectoria de memoria

`lesion_protocol_v1` (B4 de [#750](https://github.com/GonxKZ/mars-titan/issues/750)) comprueba cuánto depende una predicción de las áreas y coordenadas del corrector maduro. La supresión solo afecta a lecturas diagnósticas. La trayectoria original conserva sus emisiones, sus escrituras y, cuando se usa E1, los componentes y pesos originales de la mezcla.

Están implementados los helpers de lectura y el registro causal recuperable. La integración del diagnóstico en la ventana y en la sesión nativa sigue pendiente. Tampoco hay una orden de evaluación científica de B4 ni curvas sobre datos reales. Estos módulos no completan todavía el protocolo ni permiten afirmar que el predictor distribuye recuerdos útiles de forma resistente a lesiones.

## Intervención y controles

El [protocolo](../../src/mars_titan/memory/lesion_protocol.py) fija semillas y fracciones antes de observar errores del tramo. La configuración inicial genera veinte condiciones: lectura intacta, corrección cero, doce supresiones de áreas y seis de coordenadas. Usa las semillas 0 y 1 y las fracciones racionales 1/4, 1/2 y 3/4. El máximo permitido es de 24 condiciones.

La selección elimina un prefijo de una permutación fija. Las máscaras de una misma semilla son anidadas y no se eligen por el error observado. PCG64 recibe cuatro palabras, dominio `0xB4`, eje, semilla e índice del área. No consume el RNG global de NumPy ni el de PyTorch. La identidad conserva los parámetros, las huellas de las máscaras y el número de áreas y coordenadas que permanecen activas.

Una supresión de áreas corta sus consultas, sus lecturas y sus mensajes de entrada y salida en B2. Se ofrecen dos denominadores: media entre áreas activas y media entre todas las áreas originales, con las eliminadas a cero. La segunda lectura permite distinguir parte del cambio de ganancia que introduce la primera. La eliminación total devuelve corrección cero sin cambiar la regla antigua que exige al menos un área en una lectura ordinaria.

Para las coordenadas, se interviene después de DPFP. Si P_a es la máscara diagonal del área a, los mapas del puente se restringen en ambos extremos, P_a T_a P_b. Se seleccionan las submatrices antes del producto, de modo que una salida eliminada no se calcula primero para descartarla después. Las consultas permanecen en el soporte de su máscara durante todos los pasos. B2 conserva su normalización de filas, con vector cero definido como cero. Tras eliminar coordenadas también debe aplicarla con ganancia cero, porque deja de ser válido el atajo de suponer una consulta inicial de norma uno. Las rutas antiguas sin coordenadas conservan su comportamiento.

El [helper del puente](../../src/mars_titan/memory/area_query_bridge.py) se contrasta con operadores restringidos explícitos en uno, dos y cuatro pasos. La prueba incluye salidas eliminadas que desbordarían al calcular el producto completo y gradcheck de las consultas. Las máscaras no son una poda que libere almacenamiento ni una técnica de entrenamiento.

El control monolítico usa un área con cuatro desplazamientos DPFP frente a cuatro áreas con uno. Iguala los coeficientes W y permite comparar fracciones de coordenadas. No iguala por sí solo los mapas adicionales de B2 ni el tiempo de cálculo. El redondeo de una fracción de áreas puede eliminar cero áreas cuando solo existe una. Se informa la fracción realizada y no se presenta ese punto como una supresión equivalente.

La corrección cero conserva el pronóstico del núcleo. Es distinta del predictor de retorno residual cero. Aquí no se lesionan los parámetros compartidos, los pesos rápidos de Titans ni el banco episódico. Una prueba sobre este corrector no acredita resistencia de toda la arquitectura ni reproducción de una memoria biológica.

## Emisión, maduración y recuperación

`LesionReadout.read` devuelve correcciones de forma `[filas, condiciones]`. La primera columna coincide bit a bit con la corrección original que recibe. El consumidor debe sumar el mismo pronóstico del núcleo a todas las columnas antes de registrarlas.

El [registro](../../src/mars_titan/memory/lesion_ledger.py) conserva copias de los pronósticos emitidos sin su grafo de autograd. `begin` fija el corte, `issue` registra decisiones contemporáneas y `receive` acepta únicamente resultados disponibles después de su decisión y antes o en el corte actual. No vuelve a leer la memoria al madurar una etiqueta. `discard` retira exclusiones y finalizaciones sin inventar resultados.

Cada condición reutiliza `SessionErrors`: primero promedia los activos del mismo mercado e instante y después los grupos de sesión. No sustituye esa métrica por la media global de filas. Las actualizaciones de los casos se preparan antes de confirmarlas juntas, por lo que un desbordamiento o un límite excedido no publica parcialmente los otros casos.

El estado versionado contiene reloj, contadores, predicciones pendientes, sumas por sesión y el último evento. Sus límites máximos son 40.960 pendientes y 50.000 sesiones. La recuperación valida identidades, dimensiones, tipos, finitud, orden, maduración, censuras y conciliación de contadores, además de la huella. Las pruebas incluyen estados con una huella recalculada que conservan errores semánticos. Una huella coherente no sustituye esas comprobaciones.

La salida por condición contiene pronósticos puntuales. No se crean cuantiles ni intervalos a partir de esas columnas. Los contrastes predictivos y su incertidumbre temporal requieren completar la integración, fijar el protocolo y ejecutar únicamente los tramos autorizados. El test final y el bloqueo de aprendizaje permanecen cerrados.

## Antecedentes y alcance

Michel, Levy y Neubig estudian la retirada de cabezas de atención durante inferencia en *Are Sixteen Heads Really Better than One?*, [arXiv:1905.10650v3](https://arxiv.org/abs/1905.10650v3), revisado el 4 de noviembre de 2019. Danihelka et al. proponen memoria asociativa redundante en *Associative Long Short-Term Memory*, [arXiv:1602.03032v2](https://arxiv.org/abs/1602.03032v2), revisado el 19 de mayo de 2016. Son antecedentes de ablación y redundancia, no garantías de esta intervención financiera. En esta revisión se verificaron sus metadatos y resúmenes, sin atribuir una lectura completa de los manuscritos ni una comprobación Lean.

Las máscaras, la restricción de los mapas, los controles de denominador y el registro B4 son decisiones del proyecto. No se presentan como una arquitectura nueva demostrada ni como evidencia de consciencia, memoria holográfica física o superioridad predictiva.

## Comprobaciones y coste del núcleo técnico

La [verificación del 26 y 27 de octubre de 2026](../../reports/engineering/lesion-core-20261027/verification.json) reúne 264 casos de regresión y 69 del núcleo actual, 333 casos CPU distintos. Quince configuraciones conservan estados, lecturas y recuperación bit a bit frente a 3e89cc5df. Los cinco casos CUDA anteriores siguen sin ejecutarse y no se ha probado una integración B4 en GPU.

Los dos módulos nuevos cubren 272/285 sentencias y 107/120 ramas con coverage.py 7.16.1. El [CRAP](../../reports/engineering/lesion-core-20261027/quality.json) máximo por función es 28,1, calculado con Radon 6.0.1 y cobertura de sentencias. La validación de restauración se separó en comprobaciones de predicciones, sesiones, resultados y censuras. Las cifras corresponden a los helpers y al registro, no a una canalización B4 integrada.

Se detectaron nueve [mutaciones dirigidas](../../reports/engineering/lesion-core-20261027/mutation-results.json), con referencia verde antes y después y sin intentos de pasos de optimizador. La batería inicial dejó vivo el mutante que quitaba la máscara del puente, porque el consumidor ya había enmascarado sus consultas. Se añadió una prueba de su entrada directa. Ese superviviente y la corrección de la cobertura de pruebas permanecen registrados. No se atribuye una mutación exhaustiva del proyecto.

El [benchmark](../../benchmarks/lesion_read_cost.py) usa estados escritos con 64 resultados de un fixture y consultas de 64, 256 y 1.024 filas. Cada caso tiene tres calentamientos y quince repeticiones, con dos hilos CPU y FP64. Se mide la lectura original frente a la original más veinte diagnósticos, incluyendo codec y máscaras. Se excluyen predictor, cálculo del régimen, registro durable, colas y transferencias GPU. Las [dos](../../reports/engineering/lesion-core-20261027/benchmark-1.json) [pasadas](../../reports/engineering/lesion-core-20261027/benchmark-2.json) conservan muestras, versiones, huellas, carga del equipo y percentiles.

| Perfil, 256 filas | Original, pasadas 1 / 2 (ms) | Con veinte condiciones, pasadas 1 / 2 (ms) | Bytes del estado original |
| --- | ---: | ---: | ---: |
| B1, cuatro áreas | 13,346 / 4,690 | 195,430 / 101,479 | 6.144 |
| Un área, iguales coeficientes W | 6,100 / 2,332 | 88,741 / 55,697 | 4.608 |
| B2, un paso | 10,542 / 10,112 | 191,289 / 181,270 | 530.432 |
| B2, cuatro pasos | 23,430 / 32,994 | 337,616 / 358,031 | 530.432 |
| B3 sobre B2 | 7,404 / 11,656 | 141,865 / 152,436 | 532.992 |
| E1 sobre B3+B2 | 32,207 / 20,124 | 485,101 / 405,819 | 1.593.880 |

Hay variación apreciable entre pasadas. Estos valores no permiten ordenar perfiles por pequeñas diferencias. El diagnóstico añade trabajo y no es una mejora de velocidad de la predicción ordinaria. Los procesos completos tardaron 211,95 y 187,10 segundos, con picos RSS de 651.200 y 652.096 KiB. Los bytes de la tabla incluyen coeficientes e índices del estado original. No incluyen máscaras, objetos Python, temporales ni el registro. No se mide energía ni VRAM. Quince muestras no permiten establecer un objetivo fiable de latencia de cola.

El [esquema](../diagrams/variants/b4-lesion-protocol.png) muestra la separación entre lectura diagnóstica y maduración. Tiene nueve comprobaciones de composición correctas y evidencia de navegador. La integración con los consumidores sigue pendiente. No hay una diferencia de MAE ni una mejora de generalización medida de B4.
