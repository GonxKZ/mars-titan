# Observación de escrituras compartidas y plasticidad

La ampliación de #448 registra las escrituras compartidas de PT4 y las ganancias de plasticidad que la memoria ya ha calculado. Separa sus contadores y pérdidas de los pasos por flujo. El análisis JSON y Markdown conserva esa distinción. No cambia la regla de actualización ni activa la instrumentación por defecto.

Se comprobaron 359 casos distintos, incluidos 54 CUDA. La primera pasada aprobó 357 y dejó pendientes dos fixtures que ajustan parámetros. La complementaria ejecutó los dos originales, con cuatro actualizaciones de AdamW por configuración en FP32 y FP64, sin corpus financiero. No quedan omisiones de esta selección. Las comprobaciones cubren valores, estados, gradientes, RNG, recuperación del cursor, cadencia, límite de bytes, escrituras tras reunir los bloques de una sesión y rechazo de entradas inválidas. No constituyen toda la batería del repositorio.

La mutación dirigida en CPU detectó los 17 cambios introducidos en una copia separada. Pasan la referencia inicial y la restauración. Incluye doble división de la pérdida compartida, mezcla de contadores, captura sin cascada, pérdida de estadísticas, grafos retenidos y resúmenes incorrectos. Esta selección no es un análisis de todos los mutantes posibles.

La cobertura de ramas, medida con coverage.py, registra 605 de 619 sentencias y 241 de 258 ramas en los tres módulos del informe. El cálculo de CRAP utiliza la complejidad de Radon y cobertura de sentencias por función. La mayor puntuación es 32,37 en `NeuralMemory.validate_state`. `_write` conserva complejidad 30, con sus 34 sentencias y 22 ramas observadas. Estas medidas localizan riesgos y no demuestran ausencia de errores.

Los [datos de verificación](verification.json), [mutaciones](mutations.json) y [cobertura, complejidad y CRAP](quality.json) registran fuentes, huellas, recuentos y límites. La medición de coste de esta ampliación todavía está pendiente. La paridad y el sobrecoste completos de todas las familias de v3.1 continúan en #448. No se ha abierto el test final.
