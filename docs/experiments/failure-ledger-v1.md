# Contexto y libro de fallos maduros

S2 de [#790](https://github.com/GonxKZ/mars-titan/issues/790) propone recordar fallos reales fuera de muestra y reducir la confianza ante contextos parecidos. Es una variante experimental propia. No se ha medido su efecto predictivo.

## Contexto implementado

`memory.failure_context` captura una emisión sin recibir su etiqueta. Conserva identidad y fecha de la muestra, disponibilidad y huella de la entrada, clave del codec congelado, valores macro, máscaras, logaritmos de edades, presencia de modalidades, participación de áreas y mecanismo usado. Su diseño identifica el codec, el contrato de entrada y el catálogo con las doce familias macro. No mezcla versiones de representación.

La actividad de las áreas reproduce la máscara estructural de H1. Un área pertenece a su modalidad y una ranura compartida participa cuando existe alguna modalidad. Esta máscara no mide atención ni importancia causal. Los datos se conservan como bytes inmutables. Modificar los arrays originales no cambia el contexto. La consulta y la serialización comprueban su huella, y la recuperación valida diseño, tamaños, tipos y actividad.

Las edades redondeadas no permiten reconstruir fechas exactas de publicación. Las fechas individuales quedan desconocidas salvo que una fuente las aporte. Una fecha conocida no puede superar la disponibilidad de la entrada ni acompañar a un concepto ausente. Se conserva el contrato histórico de ausencias y se rechaza el test reservado de 2024.

El almacenamiento de pendientes y fallos maduros, su consulta por semejanza y el consumo de la señal en `FinancialSession` están pendientes de integración. Disponer del contexto no completa S2, S3, el meta-etiquetado ni el recuerdo durante entrenamiento.

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
