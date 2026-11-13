# Actualización de Soup Sieve

Fecha: 13 de noviembre de 2026. [Issue #838](https://github.com/GonxKZ/mars-titan/issues/838).

El bloqueo pasa de Soup Sieve 2.9.2 a 4.0. Esta versión corrige los avisos [GHSA-mmhr-3c5m-f64f](https://github.com/advisories/GHSA-mmhr-3c5m-f64f) y [GHSA-4j63-6c4x-wfg4](https://github.com/advisories/GHSA-4j63-6c4x-wfg4), relacionados con el coste de ciertos selectores sobre HTML no confiable. La excepción a la fecha general de resolución se limita a este paquete y permite la versión publicada el 3 de noviembre. El resto de versiones del bloqueo permanece igual.

Las 255 pruebas seleccionadas de noticias, documentos macroeconómicos y adquisición pasaron sin omisiones. Se contrastaron dos formas de selector con tres tamaños y tres repeticiones. El selector descendiente conservó el resultado vacío. Soup Sieve 4.0 rechaza el selector complejo dentro de `:has()` ejercitado en el diagnóstico. Los lectores del proyecto no usan esa forma y sus pruebas conservaron los resultados esperados.

En una copia aislada, `uv lock --check` aceptó el bloqueo actualizado, rechazó la restauración completa del bloqueo anterior y volvió a aceptar el actualizado. Este control comprueba coherencia con las restricciones del proyecto, no la autenticidad de cualquier edición manual del archivo.

La [evidencia](verification.json) conserva versiones, resultados y tiempos del diagnóstico. Son casos pequeños, no una garantía de tiempo máximo para cualquier HTML. No cambian PyTorch, NumPy, PyArrow ni LightGBM y no se ejecutaron pasos de optimizador.

`make check TEST_WORKERS=2` terminó con 14.338 pruebas superadas, 401 omisiones y 16 advertencias en 412,66 s. Las omisiones y advertencias coinciden con las de la comprobación anterior de la campaña.
