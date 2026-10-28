# Lesiones de lectura sobre la misma trayectoria de memoria

`lesion_protocol_v1` (B4 de [#750](https://github.com/GonxKZ/mars-titan/issues/750)) comprueba cuánto depende una predicción de las áreas y coordenadas del corrector maduro. La supresión solo afecta a lecturas diagnósticas. La trayectoria original conserva sus emisiones, sus escrituras y, cuando se usa E1, los componentes y pesos originales de la mezcla.

Los helpers y el registro están conectados a la ventana y a la sesión nativa, con comprobaciones CPU sobre fixtures. Hay configuración guardable y órdenes para ejecutar una ventana existente y analizar sus curvas por sesión. El paso durable tiene medidas CPU con fixtures compactos. Faltan las comprobaciones CUDA, las medidas con formas del corpus real y las ejecuciones científicas autorizadas. La integración técnica no demuestra que el predictor distribuya recuerdos útiles de forma resistente a lesiones.

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

La salida por condición contiene pronósticos puntuales. No se crean cuantiles ni intervalos a partir de esas columnas. Los contrastes predictivos y su incertidumbre temporal requieren completar la validación de la integración, fijar el protocolo y ejecutar únicamente los tramos autorizados. El test final y el bloqueo de aprendizaje permanecen cerrados.

## Conexión con la ventana y la sesión

`CorrectionInference` y `run_correction_window` aceptan el argumento opcional `lesions`, de tipo `LesionProtocol`. Sin él conservan las salidas y campos anteriores. Con B4, `prediction_rows` añade una columna `lesion_<condición>` por pronóstico diagnóstico, junto a la predicción y los cuantiles originales. Las métricas de cada condición quedan bajo `lesions`. El recorrido de la ventana y su regeneración mantienen el mismo padre y la misma corrección, y la petición registra la identidad del protocolo y las huellas de sus fuentes. Un traslado con fuentes incompatibles se rechaza antes de preparar entradas.

`FinancialSession` admite el mismo argumento. El [adaptador](../../src/mars_titan/memory/lesion_session.py) guarda el registro en un artefacto de la misma generación nativa. La confirmación comprueba que sus pendientes coinciden con los originales y que la condición intacta conserva el valor realmente emitido. La poda mantiene las referencias de las dos generaciones recuperables. Un corte antes de publicar conserva la generación anterior y permite repetir sin duplicar resultados.

`lesion_event()` devuelve copias del último evento confirmado y `lesion_metrics()` sus errores acumulados por sesión. Estas consultas se utilizan entre pasos de la misma sesión. No exponen una instancia mutable del registro ni autorizan ejecutar dos pasos concurrentes sobre la misma sesión. La identidad y las fuentes del diagnóstico forman parte del contrato y no pueden cambiar al reanudar.

Los cierres de fase pueden retirar hasta el límite completo de pendientes, aunque supere el tamaño de un lote de observaciones. Las finalizaciones no crean etiquetas ni errores. El pronóstico original, los pesos y el reloj de E1 y los cuantiles del núcleo se comprueban frente al recorrido sin B4.

La [verificación de la integración](../../reports/engineering/lesion-integration-20261027/verification.json) registra 405 pruebas aprobadas y seis comprobaciones CUDA pendientes. Tras adelantar el control de fuentes se repitieron 66 casos focalizados, incluidos en los anteriores. Las sesiones directa y recuperada reproducen las emisiones y los diagnósticos de la ventana. La regeneración por la API completa conserva los Parquet, con un padre preparado mediante un registrador de gradientes que no cambia pesos. Son fixtures técnicos, no entrenamientos ni resultados financieros reales.

Cinco [mutaciones dirigidas](../../reports/engineering/lesion-integration-20261027/mutation-results.json) detectan omitir el núcleo en los diagnósticos, sustituir la emisión real, ignorar pendientes desalineados, aceptar otra predicción original y perder B4 al regenerar. Los controles pasan antes y después de cada mutación, sin intentos de pasos de optimizador. También se reprodujo y corrigió la apertura de entradas antes de comprobar fuentes incompatibles.

La [calidad de los helpers](../../reports/engineering/lesion-integration-20261027/quality.json) cubre 322/336 sentencias y 124/138 ramas, con coverage.py 7.16.1 y Radon 6.0.1. La [cobertura de los dos módulos grandes del recorrido](../../reports/engineering/lesion-integration-20261027/quality-integration.json) se informa aparte: 938/1.260 sentencias y 285/500 ramas en la comprobación focalizada. No son una cobertura global del proyecto ni una validación CUDA. La medida posterior del paso durable con persistencia se informa más adelante.

## Configuración y análisis por sesión

La [configuración B4](../../configs/experiments/lesion-protocol-v1.json) conserva las máscaras anteriores y sigue desactivada. Declara MAE por sesión como criterio del proyecto, Rank IC con al menos tres activos, confianza del 95 %, 2.000 réplicas y bloques temporales de 16 días observados, con sensibilidad a 8 y 32. La semilla del remuestreo es 20261027. No son valores optimizados sobre resultados de B4.

La orden [run_lesion_window.py](../../scripts/run_lesion_window.py) reutiliza `run_correction_window`. Recibe `--view`, `--parent`, `--study`, `--arm`, `--lesions`, `--search-case`, `--seed` y `--output`. `--study` puede ser el [estudio B2](../../configs/experiments/area-query-bridge-v1.json), con la variante `bridge` y el caso de receta `eta5e-2`. La ruta de su receta se resuelve desde la raíz del repositorio. `--lesions` apunta a la configuración B4 y `--device` vale `cuda:0` por defecto. Conserva la reserva de GPU y la interrupción del runner existente. No hay otro entrenamiento del padre.

Antes de ejecutar, guarda `lesion-study.json` con protocolo, análisis, variante, semilla, caso y huellas de las dos declaraciones y de la receta. La huella de la declaración de variantes corresponde a los bytes leídos para seleccionar sus componentes. Una reanudación debe conservarlo. No añade esa declaración retrospectivamente a una salida existente. La protección de aprendizaje se consulta antes de leer entradas, crear la salida o reservar CUDA.

La orden [analyze_lesion_curves.py](../../scripts/analyze_lesion_curves.py) recibe `--report`, `--study`, `--partition` y `--output`. Admite `run.json` y `carry.json` completados, con las predicciones B4 del tramo indicado. Solo permite validación, calibración o evaluación anteriores a 2024. Verifica tamaño, huella, filas, tipos, columnas, identidad de máscaras y coincidencia de `lesion_intact` con `prediction`. Rechaza cambios de las fuentes durante el cálculo. Si existe la declaración anterior a la ventana, debe coincidir con el análisis solicitado. Si la API original no la guardó, informa su ausencia y no atribuye un preregistro a esa ejecución.

La lectura y la puntuación reutilizan el [lector común de archivos por sesiones](../engineering/prediction-file-scoring.md). Conservan los controles de columnas, la guarda científica y la huella jerárquica de población. La comparación técnica con la versión anterior mantiene exactamente las tablas por sesión y sus resúmenes. Este cambio no añade una evaluación predictiva ni demuestra una aceleración.

El análisis ordena con DuckDB, dos hilos, 256 MiB de memoria interna y hasta 1 GiB de spill en disco junto al destino. Arrow reúne sesiones completas aunque crucen lotes o row groups. Admite hasta 8.192 filas por sesión y 50.000 sesiones. Conserva estadísticos por sesión en lugar de cargar todos los pronósticos en RAM. Estos límites no incluyen la memoria de Python, Arrow ni los estadísticos y no constituyen una medida del pico total. La huella de cohorte combina las huellas de las sesiones y sus grupos internos, con una convención distinta de la del panel completo de filas.

La salida nueva contiene `curves.json` y `sessions.parquet`. Reutiliza `SessionScores` para MAE, MSE, RMSE, dirección con empates y abstenciones explícitos y Rank IC. Cada condición se compara con la intacta. Añade intacta y núcleo sin corrector frente a `residual_zero`. Las diferencias de MAE son variante menos control. La mejora relativa es positiva cuando baja el error y queda sin definir si el control tiene error cero. El remuestreo circular conserva juntas todas las sesiones y activos del mismo día. La corrección simultánea se aplica dentro de cada familia métrica, sin afirmar cobertura conjunta entre MAE y Rank IC. Un intervalo que incluye cero no demuestra equivalencia. Los cuantiles originales no se alteran ni se puntúan en este análisis.

No se eligen máscaras ni arquitecturas con estas curvas. Una ventana y semilla no acreditan generalización entre semillas y el análisis no comprueba por sí solo la disponibilidad temporal de las entradas originales. El campo `seconds` mide el cálculo y la preparación de la salida después de verificar las entradas. No sustituye un benchmark completo de la canalización.

Estas órdenes de ayuda y pruebas están comprobadas desde la raíz del repositorio:

```bash
PYTHONPATH=src:. uv run python scripts/run_lesion_window.py --help
PYTHONPATH=src:. uv run python scripts/analyze_lesion_curves.py --help
uv run pytest tests/memory/test_lesion_study.py tests/evaluation/test_lesion_curves.py
```

La [verificación de configuración y curvas](../../reports/engineering/lesion-curves-20261027/verification.json) registra 137 pruebas aprobadas sin omisiones, con fixtures y sin pasos de optimizador. Incluye una salida real de `CorrectionInference`, sesiones desordenadas y de distinto tamaño, bloques insuficientes, MAE base cero y recibos de regeneración. Detecta seis mutaciones dirigidas con referencias verdes antes y después. Los cuatro archivos nuevos cubren 253/264 sentencias y 50/62 ramas, con CRAP máximo 18 según coverage.py 7.16.1 y Radon 6.0.1. No es una mutación exhaustiva ni una cobertura global. No hay curvas científicas, mejora predictiva ni validación CUDA añadidas por estas pruebas.

## Medición de la sesión durable

El [benchmark de sesión](../../benchmarks/lesion_session_cost.py) utiliza `FinancialSession.step` completo con parámetros compartidos congelados de Titans-MAC, memoria por áreas y B4 desactivado o con veinte condiciones. Compara B1, B2 y E1 sobre B2. Cada configuración parte de una sesión nueva, procesa tres eventos de prefijo, cierra y recupera ese estado y mide el cuarto evento con memoria ya escrita. La siguiente transición se contrasta entre repeticiones directas y recuperadas.

Son fixtures manuales de 64, 256 y 1.024 activos. Sus anchuras de precios, noticias, gráficos, fundamentales y macro son 5, 2, 1, 3 y 3, con presencia de precios y gráficos. El padre tiene dimensión oculta 32 y bloques físicos de 64 filas. No son las formas del corpus completo ni una medida de un predictor entrenado. No se estiman etiquetas residuales y un gancho rechaza cualquier intento de paso de optimizador.

El intervalo principal incluye los controles de entrada y prefijo, codec, predictor, corrector, consultas diagnósticas, registro, publicación nativa, comprobación del estado confirmado y poda de artefactos. La preparación del fixture, creación del modelo y sesión, prefijo, reapertura, comprobaciones externas y cierre se registran aparte. Los perfiles instrumentados con cProfile quedan fuera de las muestras de latencia.

Los identificadores nativos y las referencias de artefactos incorporan el contrato de cada configuración. La paridad entre configuraciones se comprueba por activo y fecha, valores emitidos, contenido de los pesos rápidos, memoria correctora y contadores. Dentro de cada configuración se exige además reproducir las huellas completas de snapshots y la siguiente transición. No se ignoran diferencias numéricas para acomodar identidades distintas.

Se registran las muestras sin filtrar, su calentamiento, mediana, mínimo, máximo, caudal y bytes de archivos persistidos. El tiempo corresponde a la cohorte completa. Dividir filas entre tiempo da caudal amortizado y no la latencia individual de cada activo. Los contadores de `/proc/self/io` se conservan con su alcance de proceso y no se presentan como una medida directa del dispositivo. El pico RSS corresponde al proceso completo, incluida la preparación. No permite atribuir un pico transitorio separado a B4. La medida CPU no acredita latencia, consumo o rendimiento GPU. Tres repeticiones no permiten estimar de forma fiable p95 o p99 ni ordenar diferencias pequeñas.

La orden necesita un enlace nativo existente y una carpeta de salida nueva fuera del repositorio. No compila ni descarga dependencias. Su interfaz y sus pruebas técnicas se consultan así:

```bash
PYTHONPATH=src:. uv run python benchmarks/lesion_session_cost.py --help
uv run pytest tests/tooling/test_lesion_session_cost.py
```

Las [dos fases ejecutadas](../../reports/engineering/lesion-durable-cost-20261027/verification.json) terminaron con 72 recorridos, 18 de calentamiento y 54 medidos. Hay tres repeticiones por combinación de tamaño, perfil y diagnóstico. Las medianas del paso completo fueron las siguientes:

| Activos | Perfil | Sin B4 (ms) | Con B4 (ms) | Diferencia (ms) | Bytes retenidos adicionales |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | B1 | 463,7 | 643,7 | 180,0 | 92154 |
| 64 | B2 | 604,2 | 610,2 | 6,0 | 93178 |
| 64 | E1 sobre B2 | 722,5 | 1002,3 | 279,9 | 94522 |
| 256 | B1 | 1625,1 | 1832,5 | 207,4 | 318714 |
| 256 | B2 | 1478,0 | 1688,7 | 210,7 | 319738 |
| 256 | E1 sobre B2 | 1724,5 | 2025,9 | 301,4 | 321082 |
| 1024 | B1 | 7080,4 | 7643,5 | 563,1 | 1224954 |
| 1024 | B2 | 7419,1 | 8655,7 | 1236,6 | 1225978 |
| 1024 | E1 sobre B2 | 8106,9 | 10191,5 | 2084,5 | 1227386 |

La primera fase tardó 482,66 segundos de proceso y la segunda 1.661,00. Sus picos RSS fueron 838.644 y 921.416 KiB. Estos picos incluyen preparación y pruebas de integridad, con ambas configuraciones en el mismo proceso. No son una diferencia de memoria entre activar y desactivar B4. Los archivos de sesión y los fixtures pesados permanecen fuera del repositorio. El [resumen legible por máquina](../../reports/engineering/lesion-durable-cost-20261027/summary.json) conserva las estadísticas. Los recibos de [64 y 256 activos](../../reports/engineering/lesion-durable-cost-20261027/matrix-small.json) y de [1.024](../../reports/engineering/lesion-durable-cost-20261027/matrix-large.json) contienen las muestras y su procedencia.

En B2 con 256 activos, el incremento de la mediana es de 210,7 ms y se retienen 319.738 bytes adicionales. Hay variación de carga y caché entre repeticiones. La diferencia pequeña de B2 con 64 activos no demuestra que su coste sea despreciable. Estas medidas cuantifican trabajo diagnóstico adicional, sin demostrar un beneficio predictivo.

Un [perfil instrumentado separado](../../reports/engineering/lesion-durable-cost-20261027/profile-64-bridge.txt), con el mismo guion, localiza trabajo repetido en la comprobación de fuentes y la validación de artefactos. En el caso B2 de 64 activos aparecen 1.792 comprobaciones de firma y 19.712 llamadas `lstat`. Sus tiempos acumulados se solapan y la instrumentación añade coste. No se usan como una descomposición aditiva de la latencia sin instrumentar. El [ensayo posterior de firmas de archivo](../engineering/prefix-path-checks.md) no superó el margen prefijado en ambos modos. La implementación de producción se conserva.

El benchmark y sus comprobaciones acumulan 39 pruebas aprobadas sin omisiones y cuatro mutaciones dirigidas detectadas con referencias verdes antes y después. La cobertura del guion es de 256/268 sentencias y 51/58 ramas. Su CRAP máximo es 31,12, con coverage.py 7.16.1 y Radon 6.0.1. Es evidencia focalizada de la herramienta y su recuperación, sin cobertura global ni validación CUDA. No se cambió código productivo para obtener estos resultados.

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

El [esquema](../diagrams/variants/b4-lesion-protocol.png) muestra la separación entre lectura diagnóstica y maduración. Tiene nueve comprobaciones de composición correctas y evidencia de navegador. Las medidas anteriores corresponden al núcleo de lectura. El apartado de sesión durable recoge la medida posterior con persistencia. No hay una diferencia de MAE ni una mejora de generalización medida de B4.
