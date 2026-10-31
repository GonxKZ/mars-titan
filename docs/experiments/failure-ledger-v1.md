# Contexto y libro de fallos maduros

S2 de [#790](https://github.com/GonxKZ/mars-titan/issues/790) propone recordar fallos reales fuera de muestra y reducir la confianza ante contextos parecidos. Es una variante experimental propia. No se ha medido su efecto predictivo.

## Contexto implementado

`memory.failure_context` captura una emisión sin recibir su etiqueta. Conserva identidad y fecha de la muestra, disponibilidad y huella de la entrada, clave del codec congelado, valores macro, máscaras, logaritmos de edades, presencia de modalidades, participación de áreas y mecanismo usado. Su diseño identifica el codec, el contrato de entrada y el catálogo con las doce familias macro. No mezcla versiones de representación.

La actividad de las áreas reproduce la máscara estructural de H1. Un área pertenece a su modalidad y una ranura compartida participa cuando existe alguna modalidad. Esta máscara no mide atención ni importancia causal. Los datos se conservan como bytes inmutables. Modificar los arrays originales no cambia el contexto. La consulta y la serialización comprueban su huella, y la recuperación valida diseño, tamaños, tipos y actividad.

Las edades redondeadas no permiten reconstruir fechas exactas de publicación. Las fechas individuales quedan desconocidas salvo que una fuente las aporte. Una fecha conocida no puede superar la disponibilidad de la entrada ni acompañar a un concepto ausente. Se conserva el contrato histórico de ausencias y se rechaza el test reservado de 2024.

`memory.failure_ledger` implementa el libro y `memory.failure_ledger_state` su recuperación. El consumo de la señal en `FinancialSession` está pendiente. Disponer de estas piezas no completa S2, S3, el meta-etiquetado ni el recuerdo durante entrenamiento.

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

`memory.failure_producer` obtiene el codec y el grafo del enlace financiero real. Admite los consumidores escalares de Titans, retención espaciada y GRU. Comprueba sus identidades, la congelación de pesos, el catálogo macro y su huella, la estructura del grafo y las cinco modalidades de `DecisionRow`. Recalcula la huella de las entradas antes de capturar el contexto. No conserva una referencia al modelo ni reejecuta el predictor o la proyección de clave.

El catálogo se lee con un límite de 2 MiB y 682 filas. Sus indicadores deben ser únicos y pertenecer a las doce familias admitidas. La actividad se contrasta con los buffers estructurales del consumidor H1/H2/H3 y sus áreas o ranuras. Las fechas macro individuales siguen desconocidas salvo que una fuente verificada las aporte. No se deducen del logaritmo de la edad.

La [verificación del productor](../../reports/engineering/failure-producer-20261031/verification.json) registra 163 casos CPU sin omisiones, 45 del productor y 118 del núcleo, con los codecs reales de 64 y 128 dimensiones. Sus 16 defectos dirigidos se detectan con controles verdes. Las seis funciones de producción nuevas están medidas, con CRAP máximo 20,18. Este diagnóstico no garantiza ausencia de errores. La integración con emisión, maduración, recuperación y publicación atómica de `FinancialSession` todavía está pendiente. Tampoco está integrado el exportador analítico Arrow/DuckDB ni el estudio científico.

## Regla propuesta y control contrastable

Sean q y c claves normalizadas, k(q,c) = exp(−‖q−c‖² / (2h²)), con h positivo. La regla max_e k(q,c_e) no cambia al permutar los mismos contextos. Por tanto, ese desorden no permite contrastar la correspondencia entre contexto y fallo.

La siguiente fase propone ponderar cada contexto por s_e = 1 − τ/|error_e|, solo para errores maduros estrictamente mayores que un umbral τ positivo, fijado con datos G admisibles. El riesgo sería max_e s_e k(q,c_e). La confianza c = 1 − strength × riesgo permanece entre cero y uno para strength entre cero y uno. Es una puntuación heurística, no una probabilidad calibrada. La similitud histórica tampoco demuestra que el siguiente resultado vaya a repetirse.

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

La conexión con el predictor, el productor del contexto real, los controles de consumo, su coste completo y el estudio predictivo siguen pendientes. No hay una mejora financiera medida.
