# Lectura histórica de resultados recibidos al mismo corte

La lectura as-of rechazaba una recepción nativa válida cuando una decisión y el `settlement` posterior compartían corte. Ahora admite esa igualdad únicamente después de `warmup` o `decision`, conserva el límite `before` exclusivo y comprueba que el tipo histórico coincide con el registro nativo. El orden siguiente utiliza el tipo nativo verificado. No se modifican las reglas del ejecutor nativo.

La [verificación](verification.json) reúne 126 casos únicos: 71 de orden y vistas individuales y 55 de ventanas e integración. Los tres casos repetidos para completar la instrumentación ya están incluidos. Se comprueban recepción tardía, invariancia ante sufijos futuros, límites exclusivos, snapshot y RNG intactos, cierre y reapertura, con integración CPU y CUDA. Las fechas de disponibilidad y recepción siguen separadas.

El [resumen de calidad](quality.json) registra 184/184 sentencias y 34/34 ramas en dos fuentes seleccionadas, con 13 funciones y CRAP máximo de 12. Se usaron coverage.py 7.16.1 y Radon 6.0.1. Es cobertura de ese alcance, no de todo el repositorio ni de todas las combinaciones booleanas. Los tiempos instrumentados de los recibos no son medidas de latencia o rendimiento del lector.

Se detectaron los [cuatro objetivos de mutación](mutations.json) que alteran el límite inferior, la exclusividad de `before`, la admisión de cortes iguales y la concordancia histórico/nativo. Cada uno tiene un fallo semántico en CALL con referencia y restauración correctas. Es una selección dirigida, no una comprobación de todas las mutaciones posibles.

Se conservan los [intentos no aprobados](attempts.json): la corrida que agotó el límite de 240 segundos por proceso, el fallo de compatibilidad del diagnóstico «futuro» y el fallo de colección del intento V2 por un helper omitido. El diagnóstico se corrigió conservando la prueba. El cierre de fixtures y el hook se corrigieron por separado, sin contar errores de colección como mutantes detectados.

Esta corrección aporta infraestructura para #711. No implementa el ciclo exterior de propuestas ni demuestra una mejora predictiva. No se ejecutaron ajustes o entrenamientos. La campaña y el test final de 2024 siguen cerrados.
