# Contexto y libro de fallos maduros

S2 de [#790](https://github.com/GonxKZ/mars-titan/issues/790) propone recordar fallos reales fuera de muestra y reducir la confianza ante contextos parecidos. Es una variante experimental propia. No se ha medido su efecto predictivo.

## Contexto implementado

`memory.failure_context` captura una emisión sin recibir su etiqueta. Conserva identidad y fecha de la muestra, disponibilidad y huella de la entrada, clave del codec congelado, valores macro, máscaras, logaritmos de edades, presencia de modalidades, participación de áreas y mecanismo usado. Su diseño identifica el codec, el contrato de entrada y el catálogo con las doce familias macro. No mezcla versiones de representación.

La actividad de las áreas reproduce la máscara estructural de H1. Un área pertenece a su modalidad y una ranura compartida participa cuando existe alguna modalidad. Esta máscara no mide atención ni importancia causal. Los datos se conservan como bytes inmutables. Modificar los arrays originales no cambia el contexto. La consulta y la serialización comprueban su huella, y la recuperación valida diseño, tamaños, tipos y actividad.

Las edades redondeadas no permiten reconstruir fechas exactas de publicación. Las fechas individuales quedan desconocidas salvo que una fuente las aporte. Una fecha conocida no puede superar la disponibilidad de la entrada ni acompañar a un concepto ausente. Se conserva el contrato histórico de ausencias y se rechaza el test reservado de 2024.

`memory.failure_ledger` implementa el libro y `memory.failure_ledger_state` su recuperación. `memory.failure_session` lo compone con `FinancialSession`, desactivado por defecto. Disponer de estas piezas no completa S3, el meta-etiquetado ni el recuerdo durante entrenamiento. El estudio científico y las comprobaciones pendientes de S2 siguen separados.

## Libro y publicación por corte

El libro recibe el punto que el productor declara como emisión y lo conserva junto a su contexto. Solo admite un fallo cuando recibe un resultado disponible, posterior a la decisión y con error absoluto estrictamente mayor que el umbral G. La configuración identifica los manifiestos del umbral y del predictor y exige que sus ajustes terminen antes de la decisión. Esta comprobación temporal no acredita por sí sola la veracidad de esos manifiestos. El productor de la sesión debe verificar su procedencia fuera de muestra y conciliar el punto con la predicción realmente emitida. Esa conexión sigue pendiente.

Los pendientes quedan fuera de la consulta. Resolver un resultado modifica la retención, pero no la instantánea visible durante el corte. También se registra cuándo se recibió el resultado. Una etiqueta tardía con disponibilidad antigua no reescribe la instantánea. El siguiente corte publica el estado nuevo. El orden de los activos no cambia la retención ni los desempates, y el control desordenado usa un generador propio.

La recuperación conserva pendientes, cursores por flujo, contadores, retención e instantánea. Cada contexto se serializa una sola vez, aunque siga referenciado por ambas colecciones. Se validan esquemas, límites, identidad de decisión, maduración y conciliación antes de devolver el estado recuperado. El caso de una instantánea que todavía referencia un fallo desalojado de la retención tiene una comprobación específica.

Capacidad, pendientes, flujos, bytes lógicos y tamaño de bloque de consulta tienen límites explícitos. La capacidad se denomina `capacity` en esta configuración y cuenta fallos, no lecturas K ni pesos rápidos de Titans. Los buffers de esta versión usan el orden de bytes del host. No se ha comprobado su recuperación en equipos con otro orden de bytes.

## Claves nulas del candidato

El codec GRU de 128 dimensiones admite claves nulas. El contexto puede conservarlas con `allow_zero_key=True`, una política identificada que no cambia la identidad del modo original. Se conservan la fila, la emisión y el error maduro. La clave nula no recibe una dirección inventada ni participa en la semejanza. Para una consulta nula, riesgo cero y confianza uno significan que S2 no atenúa, no que exista certeza predictiva.

Los fallos sin clave direccional siguen sujetos a la capacidad y retención del libro. Pueden desplazar otros fallos, un coste que debe contrastarse. No se reserva una caché normalizada adicional para ellos. El [recibo de esta política](../../reports/engineering/failure-zero-keys-20261031/verification.json) registra 118 casos CPU sin omisiones, 14 defectos nuevos detectados y la repetición de los 57 anteriores. Las identidades y los bytes del contexto anterior se contrastan con referencias fijas. No se ha medido una mejora financiera.

El registro congelado de H1 de #783 se incorpora como dependencia experimental de [#826](https://github.com/GonxKZ/mars-titan/pull/826). Sus pruebas CUDA y su regresión amplia posterior siguen pendientes. El coste anterior del libro corresponde al código de la caché previo a esta política y no mide el productor ni el recorrido de sesión nuevo.

## Productor desde el consumidor real

`memory.failure_producer` obtiene el codec y el grafo del enlace financiero real. Admite los consumidores escalares de Titans, retención espaciada y GRU, además del consumidor de mecanismos en modo de confianza. Comprueba sus identidades, la congelación de pesos, el catálogo macro y su huella, la estructura del grafo y las cinco modalidades de `DecisionRow`. Recalcula la huella de las entradas antes de capturar el contexto. No conserva una referencia al modelo ni reejecuta el predictor o la proyección de clave.

El catálogo se lee con un límite de 2 MiB y 682 filas. Sus indicadores deben ser únicos y pertenecer a las doce familias admitidas. La actividad se contrasta con los buffers estructurales del consumidor H1/H2/H3 y sus áreas o ranuras. Las fechas macro individuales siguen desconocidas salvo que una fuente verificada las aporte. No se deducen del logaritmo de la edad.

La [verificación del productor](../../reports/engineering/failure-producer-20261031/verification.json) registra 163 casos CPU sin omisiones, 45 del productor y 118 del núcleo, con los codecs reales de 64 y 128 dimensiones. Sus 16 defectos dirigidos se detectan con controles verdes. Las seis funciones de producción nuevas están medidas, con CRAP máximo 20,18. Este diagnóstico no garantiza ausencia de errores.

## Ciclo financiero implementado y límites

La opción `failure_memory` de `FinancialSession` recibe `FailureSessionConfig`. `None` conserva la ruta sin S2. La política identifica el consumidor congelado, el catálogo y las fronteras declaradas de ajuste del predictor y del umbral G, anteriores al inicio de decisiones. La vinculación de identidades no demuestra por sí sola cómo se ajustó el modelo. Esa procedencia debe contrastarse con el manifiesto auditado.

`confidence_only` registra la señal y conserva el punto. `shrink_point` multiplica el punto de entrada a S2 por su confianza heurística. `constant_shrink` ofrece una contracción constante predeclarada, de 0,75 por defecto, como control distinto. No se ha elegido ese valor con una evaluación científica. El control guarda los puntos de entrada y los factores usados, sin sustituir la predicción finalmente emitida al calcular el error maduro.

El libro se publica dentro del mismo bundle que los estados y las predicciones. Su pendiente debe coincidir con `NativePrediction.value` al resolver resultados, exclusiones y finalizaciones. El calentamiento no emite registros S2 y puede preceder al ajuste declarado. El centinela de cierre de 2024 solo permite liquidar y restaurar el estado. Las predicciones, consultas y etiquetas de 2024 continúan fuera de esta variante de desarrollo.

Los buffers se transportan como tensores `uint8` CPU de almacenamiento propio y tamaño exacto. Al recuperarlos vuelven a ser bytes inmutables. Esto permite reutilizar el serializador ordenado y su carga restringida, sin ampliar la lista de clases permitidas ni usar `weights_only=False`. Antes de decodificar se comprueban esquema, capacidad, tipo, forma y tamaño.

La [verificación de sesión](../../reports/engineering/failure-session-20261031/verification.json) registra 188 casos CPU sin omisiones y tres controles financieros originales. Incluye los 163 anteriores, así que las selecciones no se suman como muestras independientes. Sus 16 defectos dirigidos se detectan con controles verdes. Se comprueban la recuperación de Titans y GRU, paridad sin atenuación, error del punto real, orden de activos, instantánea común, calentamiento, cierre y replay tras un fallo antes del commit. Las 23 funciones cambiadas están medidas. `_prepare_event` mantiene CRAP 548,09 por su complejidad y recorridos aún no cubiertos en esta selección. No se afirma que toda la clase esté comprobada.

La atenuación combinada con B4, E2 o el consumidor de mecanismos se rechaza hasta definir su composición sin romper diagnósticos, cuantiles ni mediana. Faltan las comprobaciones CUDA de esta ruta, la regresión amplia correspondiente, el coste completo y el estudio científico. No hay mejora predictiva medida y la variante no se activa por defecto.

## Confianza con acciones registradas

`action_source="mechanism_trace_v1"` permite componer el libro con el consumidor de mecanismos en `confidence_only`. El valor por defecto, `fixed_predictor/no_external_tool`, conserva la configuración escalar anterior. Un origen que no corresponde al enlace real se rechaza antes de publicar. El origen forma parte de la identidad y no puede cambiar durante una sesión ni al recuperar su estado.

`memory.failure_actions` concilia cada flujo, sus metadatos y su punto con la traza de emisión. La etiqueta identifica el modo de contexto, las rutas ejecutada y emitida y la calibración. La acción solicitada no sustituye a la aplicada cuando hubo un retorno a la referencia. Las filas se concilian por identidad, incluso si cambia su orden. Se rechazan trazas incompletas, duplicadas o ajenas, tipos incorrectos y puntos diferentes o no finitos.

La propuesta de mecanismos conserva los campos auxiliares ya preparados y publica el libro en la misma generación. La confianza queda como diagnóstico, sin introducir un nuevo rasgo en el selector, cambiar acciones ni alterar puntos o cuantiles. La calibración en línea y la recogida de alternativas conservan sus contratos. Esta composición tampoco amplía por sí sola el registro de grafos del consumidor de mecanismos.

El [recibo de la composición](../../reports/engineering/failure-mechanisms-20261031/verification.json) registra 245 casos CPU sin omisiones, incluidos 39 de acciones y los controles anteriores. Se detectaron 18 defectos nuevos y se repitieron los 16 de sesión con controles verdes. La función nueva tiene sus 17 sentencias y ocho ramas cubiertas, con CRAP 18. Las ocho funciones cambiadas están medidas. La preparación financiera mantiene CRAP 411,60 en esta selección, que no cubre toda la clase. No se compara ese valor con otras selecciones como una reducción de complejidad.

Se comprobaron paridad de puntos y cuantiles, calibración en línea, orden de activos, error maduro de la emisión real, fallback, selector automático, colección de alternativas, recuperación de la siguiente predicción y rollback antes de publicar. Los coeficientes del selector de prueba son suministrados, no ajustados. La procedencia auditada de los ajustes, las atenuaciones indicadas, el coste completo, la regresión amplia y CUDA siguen pendientes. No hay una mejora predictiva medida.

## Exportación analítica implementada

`memory.failure_session.confirmed_book(session)` obtiene una copia verificada del libro publicado. Cambiar esa copia no modifica la sesión. `memory.failure_export.failure_reader` ofrece lotes Arrow de hasta 512 filas. Se elige entre la instantánea anterior al feedback del corte y la retención posterior. Las etiquetas pendientes nunca se exportan. Un `as_of` distinto del reloj actual se rechaza, porque necesita su propio checkpoint y no un filtro sobre una selección posterior.

La salida conserva la predicción emitida, el resultado maduro, sus fechas, la representación, las máscaras y las huellas. Las fechas macro desconocidas son valores nulos. Los metadatos declaran `history_complete=False`, el alcance elegido, los pendientes excluidos y el total admitido. Por tanto, este archivo no sustituye al historial completo ni al checkpoint para reanudar.

`write_failures` reutiliza la escritura Parquet por lotes y publica un nombre derivado de la huella. Rechaza archivos existentes, incluso si otro escritor publica mientras se prepara la salida. `failure_counts` calcula conteos por mercado y mecanismo con DuckDB, un hilo, 64 MiB y sin archivos de desbordamiento. Estos conteos describen el conjunto exportado, no el error del predictor completo.

El [recibo de exportación](../../reports/engineering/failure-export-20261031/verification.json) registra 43 casos CPU sin omisiones, incluidos 17 del exportador y 26 de sesión. Se detectaron 14 defectos del exportador y dos del accesor con controles G1/M1/G2 verdes. Las siete funciones nuevas están medidas, con CRAP máximo 8. El conjunto de fuente y pruebas registra 585 de 590 sentencias y 74 de 80 ramas cubiertas. No se suman estas selecciones solapadas con las anteriores como casos únicos.

El [coste técnico](../../reports/engineering/failure-export-20261031/cost.json) usa fixtures con claves de 64 dimensiones, 140 indicadores y 17 marcas de actividad. Se hicieron tres repeticiones por tamaño en CPU, sin calentamiento separado, con hilos numéricos limitados a uno y otras comprobaciones en curso.

| Fallos | Mediana de lectura Arrow, ms | Mediana de escritura Parquet, ms | Mediana del agregado DuckDB, ms | Pico de RSS del proceso, KiB |
| --- | --- | --- | --- | --- |
| 64 | 40,75 | 45,22 | 74,10 | 629 736 |
| 512 | 274,68 | 334,27 | 296,48 | 639 820 |
| 4096 | 2208,03 | 2433,09 | 2416,35 | 674 072 |

El pico incluye imports y construcción de los fixtures. El primer uso puede incluir inicialización de bibliotecas. No se han medido la latencia completa de sesión, energía, VRAM ni transferencias GPU. Estas cifras no demuestran una aceleración ni una mejora predictiva.

## Regla implementada y control contrastable

Sean q y c claves normalizadas, k(q,c) = exp(−‖q−c‖² / (2h²)), con h positivo. La regla max_e k(q,c_e) no cambia al permutar los mismos contextos. Por tanto, ese desorden no permite contrastar la correspondencia entre contexto y fallo.

La implementación pondera cada contexto por s_e = 1 − τ/|error_e|, solo para errores maduros estrictamente mayores que un umbral τ positivo, fijado con datos G admisibles. El riesgo es max_e s_e k(q,c_e). La confianza c = 1 − strength × riesgo permanece entre cero y uno para strength entre cero y uno. Es una puntuación heurística, no una probabilidad calibrada. La similitud histórica tampoco demuestra que el siguiente resultado vaya a repetirse.

El control permuta los contextos entre las severidades maduras elegibles del mismo mercado y mecanismo. Si todas las severidades coinciden, vuelve a ser equivalente. Dos fixtures verifican esa invariancia y un caso con severidades distintas que sí cambia. No son resultados financieros.

Con una secuencia fija de emisiones se puede conservar el mismo libro al comparar geometrías. Si la señal atenúa el punto emitido, también cambia el error que madurará después y pueden divergir las admisiones. Esa comparación estudia dos dinámicas completas. No permite atribuir toda diferencia únicamente a la permutación.

## Fuente y límites

[Reflexion, arXiv:2303.11366v4](https://arxiv.org/html/2303.11366v4), de Shinn y colaboradores, usa un actor, un evaluador y reflexiones textuales guardadas para intentos posteriores. La lectura del 31 de octubre de 2026 comprende la ficha, el resumen, el algoritmo 1, la sección 3 y la sección 5. Su memoria práctica está acotada. La sección 5 advierte de óptimos locales y limitaciones de las comprobaciones.

S2 adapta la idea de conservar feedback. Su contexto numérico, la semejanza ponderada y la maduración financiera son decisiones propias. No incorpora un generador de reflexiones textuales ni reproduce los resultados del artículo. No se ha ejecutado Lean ni una comparación científica de esta propuesta.

## Comprobación del contexto, 31 de octubre de 2026

Pasan 46 casos CPU sin omisiones. Incluyen aislamiento de buffers, futuras disponibilidades, máscaras, fechas desconocidas, normalización de claves extremas, recuperación, cambios de diseño y alteración del sello. Las 19 mutaciones dirigidas se detectan con controles G1 y G2 en verde. Se cubren las 147 sentencias y 42 ramas del módulo de producción. CRAP máximo 16, con coverage.py y complejidad de Radon según `scripts/summarize_python_quality.py`. Son diagnósticos de estas comprobaciones y no garantías de ausencia de errores.

Los casos usan fixtures identificados, sin aprendizaje, pasos de optimizador ni acceso al test final. Esta pieza NumPy captura metadatos en CPU. No ejercita una ruta de predictor CUDA ni aporta una medida de latencia de la variante completa.

## Coste técnico del libro aislado

La referencia del libro pasa 107 casos que incluyen los 46 del contexto. La selección con caché pasa 112, sin omisiones. Sus cinco casos adicionales comprueban también paridad exacta de señales y reconstrucción al recuperar. Se detectan 57 defectos dirigidos distintos, 19 del contexto, 30 del libro y ocho de la caché, con controles verdes. Los 30 del libro se repiten tras añadir la caché y no se suman como defectos nuevos. El [recibo de verificación](../../reports/engineering/failure-ledger-20261031/verification.json) registra cobertura, CRAP y huellas. Ninguna de las 53 funciones de producción cambiadas queda sin medir. CRAP máximo 16 en la selección, sin que esa puntuación garantice ausencia de errores.

El perfil de la referencia mostró normalización y verificación repetidas de claves históricas en cada consulta. La opción `cache_keys=True` captura claves normalizadas como bytes inmutables y las reconstruye a partir de contextos verificados al recuperar. Mantiene la geometría FP64 y contabiliza la memoria adicional. El modo sin caché sigue disponible y es la referencia por defecto mientras falta medir la integración completa.

Se hicieron tres repeticiones de cada configuración, alternando su orden. Cada una usa cinco consultas de calentamiento, 50 consultas medidas, cinco exportaciones y tres recuperaciones, además de construir el libro. Los fixtures tienen claves de 64 dimensiones, 140 indicadores macro y 17 marcas de actividad. Estas marcas no representan 17 modelos ejecutados. Las medidas se hicieron en un Ryzen 9 8945HS, con hilos numéricos limitados a uno, perfil power-saver y otras comprobaciones CPU en curso.

| Fallos | Mediana de consulta sin caché, ms | Mediana con caché, ms | Bytes adicionales de claves |
| --- | --- | --- | --- |
| 64 | 1,76 a 2,60 | 0,226 a 0,283 | 32 768 |
| 512 | 13,34 a 20,95 | 1,00 a 1,67 | 262 144 |
| 4096 | 91,62 a 153,79 | 9,76 a 13,41 | 2 097 152 |

Los intervalos de la tabla son el mínimo y el máximo de las tres medianas, no intervalos de confianza. Con 4096 fallos, el trabajo completo medido del libro pasa de 10,15 a 13,91 s sin caché a 5,35 a 7,64 s con caché. No incluye la carga inicial de Python. El [recibo de coste](../../reports/engineering/failure-ledger-20261031/cost.json) registra también tiempo de proceso, dispersión, percentiles y pico de RSS. RSS incluye imports y fixtures, y no se presenta como memoria atribuible solo al libro. No se han medido energía, transferencias GPU ni latencia de `FinancialSession`.

El productor y la conexión operativa con el predictor están implementados. Las composiciones indicadas, la procedencia auditada de los ajustes, el coste completo y el estudio predictivo siguen pendientes. No hay una mejora financiera medida.

## Coste de la sesión durable, 31 de octubre de 2026

El [coste integral CPU](../../reports/engineering/failure-cost-driver-20261031/cost-cpu.json) usa el arnés `benchmarks.failure_session_cost` del commit `7483e0327`. Terminaron 32 casos con 64 y 256 activos, cuatro configuraciones por tamaño, un calentamiento excluido y tres repeticiones medidas. Se comparan por separado el predictor escalar y el consumidor de mecanismos, con S2 activado y desactivado. Todos usan banco M1 de 64 episodios y lector K1. El libro S2 tiene capacidad igual al número de activos y claves en caché.

Se comparte la misma fuente manual dentro de cada tamaño. Los hashes de predicciones, estado neuronal, episodios retenidos y cuantiles coinciden entre los controles de cada familia, también en la siguiente transición tras recuperar. Los parámetros compartidos y el RNG permanecen iguales. No hubo pasos de optimizador y la protección de aprendizaje conserva su huella.

| Activos | Consumidor | Decisión sin S2, mediana en s | Con S2, mediana en s | Sobrecoste de la mediana |
| --- | --- | --- | --- | --- |
| 64 | Escalar | 0,622 | 2,130 | +242,27 % |
| 64 | Mecanismos | 0,951 | 1,966 | +106,73 % |
| 256 | Escalar | 1,650 | 5,117 | +210,23 % |
| 256 | Mecanismos | 1,940 | 5,607 | +188,96 % |

Son tiempos de `FinancialSession.step` con lotes CPU ya preparados. Incluyen la emisión y el feedback de la cuarta decisión. No incluyen adquisición ni codificación de noticias e imágenes originales. La ejecución usa CPU FP64, un hilo numérico y el perfil power-saver, con otras aplicaciones presentes. Las tres muestras de 256 activos van de 1,604 a 2,004 s sin S2 y de 4,682 a 5,604 s con S2 en el predictor escalar. En mecanismos van de 1,939 a 2,329 s y de 5,075 a 6,492 s. Estos rangos no son intervalos de confianza ni estimaciones fiables de las colas de latencia.

La mediana de reapertura pasa de 0,633 a 2,175 s en el escalar de 256 activos y de 0,981 a 2,186 s en mecanismos. El libro añade 260 992 bytes lógicos con 64 activos y 1 043 968 con 256. No incluyen toda la memoria de los objetos Python ni equivalen al RSS. Los bytes persistidos por caso y las muestras completas constan en el recibo.

El bucle medido y sus comprobaciones duraron 555,13 s. El tiempo externo del comando fue de 560,06 s. El pico RSS registrado dentro del proceso fue de 864 400 KiB y el máximo externo del comando fue de 944 700 KiB. Se conservan ambos contadores sin atribuir su diferencia a una causa no medida. Ninguno representa el pico individual de una variante. No se midieron energía ni VRAM.

El fixture tiene modalidades compactas y un solo indicador macro. Los padres no están entrenados. La medida muestra un sobrecoste de esta implementación y justifica perfilarla antes de optimizar. No demuestra latencia sobre el corpus completo ni mejora predictiva. Las variantes siguen desactivadas por defecto.

El comando requiere el binario nativo verificado y un `TMPDIR` en disco. Crea una salida externa nueva y conserva los protocolos y las mediciones. Las carpetas propias de los casos técnicos terminados se retiran, mientras que las fuentes y los casos fallidos se conservan.

```bash
uv --no-cache run --no-sync python -m benchmarks.failure_session_cost \
  --repository "$PWD" \
  --native "${MARS_TITAN_EPISODIC_NATIVE:?}" \
  --output "$(mktemp -d "${TMPDIR:?}/failure-cost.XXXXXX")/run" \
  --sizes 64 256 --warmup 1 --repeats 3 --seconds 1800 --run
```

El [perfil diagnóstico posterior](../../reports/engineering/failure-cost-driver-20261031/profile-cpu.json) recorre cuatro casos de 256 activos con instrumentación. Registra 84 restauraciones del libro, con 16,60 s acumulados, y 52 806 reconstrucciones de contexto, con 9,57 s. Las consultas geométricas suman 1,33 s en 2560 llamadas. También aparecen verificaciones de procedencia y serialización entre los recorridos costosos compartidos. Los tiempos acumulados se solapan y no corresponden solo al intervalo de decisión.

El perfil motivó evitar reconstrucciones repetidas del mismo estado confirmado dentro de una generación, conservando la verificación de huellas y el aislamiento de las propuestas. La opción y su comparación posterior se describen a continuación.

## Caché de decodificación optativa

`decoded_cache_bytes=0` conserva la referencia sin caché y es el valor por defecto. Un presupuesto positivo permite guardar una sola decodificación durante el evento. La clave incluye contrato, diseño, configuración y referencia completa del artefacto. Cada acierto relee y verifica el archivo antes de reutilizarla. Las propuestas reciben copias independientes de diccionarios, contextos, emisiones y fallos. Solo se comparten bytes y escalares inmutables. La entrada se descarta al terminar el evento, también si falla, y no se serializa.

El presupuesto cuenta objetos retenidos mediante `sys.getsizeof`. No es un límite del RSS ni de toda la memoria transitoria de las copias. Si no cabe una entrada, se usa la restauración completa. El origen y el presupuesto forman parte de la identidad de la configuración cuando se activa la opción.

El [recibo](../../reports/engineering/failure-decoded-cache-20261031/verification.json) conserva la selección de 270 casos y las 18 pruebas del arnés actualizado, ambas sin omisiones y con solapamiento explícito. Se detectaron once defectos de caché y dos del protocolo de presupuestos con controles verdes. Las quince funciones cambiadas están medidas.

La [comparación emparejada](../../reports/engineering/failure-decoded-cache-20261031/cost-cpu.json), con código `2d18bd63d`, terminó sus 64 casos. Hay 16 calentamientos excluidos y 48 casos medidos, con tres repeticiones por configuración. Se comparten fuente y modelos por tamaño entre los presupuestos de cero y 8 MiB, con controles adicionales sin S2. Coinciden las predicciones, estados neuronales, episodios y cuantiles entre ambos presupuestos, incluida la siguiente transición tras recuperar. Los parámetros y el RNG no cambian y no hubo intentos de optimizador.

| Activos | Consumidor con S2 | Sin caché, mediana en s | Caché de 8 MiB, mediana en s | Cambio de la mediana |
| --- | --- | --- | --- | --- |
| 64 | Escalar | 1,372 | 1,213 | −11,54 % |
| 64 | Mecanismos | 1,696 | 1,359 | −19,86 % |
| 256 | Escalar | 4,921 | 3,902 | −20,70 % |
| 256 | Mecanismos | 5,518 | 4,253 | −22,93 % |

Las doce diferencias emparejadas con S2 fueron negativas, aunque la dispersión es considerable. En el escalar de 256 activos, los cambios por repetición fueron del −27,79 %, −23,99 % y −0,50 %. Los controles sin S2 también fluctúan aunque la opción de caché no interviene en su recorrido. Sus medianas cambian entre un −17,36 % y un +4,50 % al cambiar la etiqueta del presupuesto. Los resultados son descriptivos y no justifican atribuir toda la diferencia a la caché ni generalizar la magnitud a otras cargas.

El bucle completo duró 967,47 s y el comando externo 971,72 s. Los picos globales fueron 869 720 KiB dentro del proceso y 948 132 KiB en la medida externa. No se midió el pico por configuración, por lo que no se afirma ahorro de RAM. El coste de recuperación, los rangos y las muestras constan en el recibo. La caché es temporal, por lo que una diferencia de reapertura no demuestra reutilización entre procesos.

Se conserva el protocolo CPU FP64, un hilo, banco M1 de 64 episodios, lector K1 y fixtures compactos con un indicador macro y padres sin entrenar. Había otras aplicaciones activas. Durante las últimas repeticiones también hubo inventario de disco y retirada de temporales, sin aislar el almacenamiento. No se midieron energía ni VRAM. La opción sigue desactivada por defecto y no hay mejora predictiva medida.

Para repetir la comparación, el comando anterior admite `--decoded-cache-bytes 8388608 --compare-decoded-cache`, siempre con una salida nueva. Los datos científicos y el test final permanecen fuera de esta comprobación técnica.
