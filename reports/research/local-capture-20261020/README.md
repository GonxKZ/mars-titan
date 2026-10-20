# Captura local de plasticidad

La variante `capture_scope="row_local"` sustituye únicamente la coherencia media entre filas por la coherencia de la propia fila al modular la captura PT5. Conserva parámetros, número de estados, RNG y huellas anteriores cuando no se selecciona. Su [contraste de desarrollo](development-v1/README.md) completó 15 ajustes y obtuvo un 7,20 % más MAE medio que la captura global. Las tres configuraciones pierden frente al residual cero. La variante permanece desactivada.

Se ejecutaron 350 casos distintos en dos selecciones, 279 y 71, con 51 casos CUDA, sin fallos, errores u omisiones. Se comprobaron gradientes FP32/FP64, contraste analítico, independencia condicionada entre filas y flujos, ambos sentidos del intercambio, paridad unitaria, bytes persistentes, RNG, recuperación y compatibilidad histórica. Los dos fixtures originales de paridad aplicaron 16 pasos de AdamW en total. No utilizaron el corpus financiero ni abrieron el test final.

La primera mutación dirigida detectó 13 de 14 cambios. Una mezcla entre flujos sobrevivió porque el caso comparaba filas con la misma señal. Se amplió la prueba del Jacobiano con dos flujos distintos y se repitieron las 14 mutaciones en una copia separada. Todas quedaron detectadas, con referencia inicial y restaurada aprobadas. Se conservan el [primer resultado](initial-mutations.json) y el [resultado corregido](mutations.json). Esto no es una campaña exhaustiva de todos los operadores mutables.

La cobertura con ramas de coverage.py, resumida mediante Radon y las regiones de función, registra 828/961 sentencias y 307/404 ramas en cinco módulos. La configuración de plasticidad cubre 21/21 sentencias y 18/18 ramas en su validador, con CRAP 14. La actualización de la regla cubre 25/25 y 4/4, con CRAP 5. El control nuevo de inicialización cubre 18/18 y 10/10, con CRAP 10. Las convenciones y límites están en [quality.json](quality.json).

El máximo CRAP de la selección es 204,48 en `FinancialConfig.__post_init__`. Es un validador histórico con otras combinaciones que esta selección no cubre completamente. Esa cifra se conserva como diagnóstico. No se afirma cobertura completa del predictor ni ausencia de errores. Una prueba anterior del diagnóstico de sincronización de PyTorch emitió su advertencia sobre el carácter provisional de esa API.

Pasaron Ruff, formato y el verificador del repositorio. El [recibo](verification.json) identifica las fuentes y las dos selecciones. Las salidas XML, cobertura completa e intentos intermedios se conservan fuera del repositorio. Los controles sintéticos comprueban el mecanismo y no prueban mejora financiera, estabilidad global o superioridad frente al estado del arte.

Se añadieron seis casos específicos de permutación y causalidad, tres CPU y tres CUDA. Cubren los flujos del MAC con proyecciones del artículo, la causalidad del prefijo y la permutación conjunta del gradiente y su historia por filas internas de plasticidad. Esta última propiedad corresponde a la regla estadística, no a toda la red.

Pasaron sin omisiones y los cinco mutantes de alineamiento quedaron detectados. La primera ampliación detectó cuatro de cinco: el orden elegido conmutaba con invertir las filas. Se sustituyó por un ciclo de tres filas y se repitió la campaña completa, con referencia inicial y restaurada aprobadas. Se conservan el [primer intento](causality-initial-row-mutations.json), el [recibo anterior](causality-verification-v1.json), las [mutaciones anteriores](causality-mutations-v1.json), el [recibo ampliado](causality-verification.json) y las [mutaciones finales](causality-mutations.json).

Junto con los 13 casos del materializador, la verificación reúne 369 casos distintos, 54 CUDA y 22 mutantes dirigidos. Las repeticiones no aumentan esos totales. El [coste del predictor y de la memoria](../../engineering/local-capture-20261020/README.md) también se midió, sin ajustes adicionales.

La [comprobación previa a la integración](integration-verification.json) repitió 57 de esos casos sobre el árbol combinado con `develop`, con CUDA, sin fallos ni omisiones y sin pasos de optimizador. Pasaron también Ruff, formato y el verificador del repositorio.
