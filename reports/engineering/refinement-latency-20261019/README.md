# Comprobación del medidor de latencia

El medidor conserva predicciones, parámetros y RNG en los casos comprobados.
Se ejecutaron 47 pruebas CPU y tres CUDA en invocaciones separadas. La parte
CUDA incluye un lector M1 con cuatro refinamientos y banco de episodios
maduros, tanto en FP32 como en FP64. Las predicciones y los contadores
coinciden exactamente entre las pasadas instrumentadas y las de referencia.
Son fixtures técnicos, sin pasos de optimizador ni datos reales de mercado.

Las pruebas CPU también reconstruyen cuatro pasadas del lector y la política
serializada, comprueban los sellos y rechazan predicciones alteradas. Una prueba
refleja las predicciones guardadas alrededor del objetivo sin cambiar el recibo
de métricas, para comprobar que se contrastan las predicciones emitidas. Se comprueban el
restablecimiento de los métodos tras fallos, los límites de registros y tiempo,
la exclusión del calentamiento de los percentiles y el rechazo de otra carga
GPU antes de inicializar CUDA.

Las 14 mutaciones dirigidas fueron detectadas. Se ejecutaron en copias
aisladas y las fuentes originales conservaron sus huellas. El conjunto cubre
cambios de parámetros, contador de actualizaciones, RNG, restauración tras
fallos, límite de eventos, tiempos de calentamiento, llamadas anidadas, coste
amortizado, procedencia del checkpoint, selección, predicciones, protección de
aprendizaje y presupuesto. No es una medida de todos los mutantes posibles del
repositorio.

La cobertura se tomó en la invocación CPU con coverage.py 7.16.1 y ramas
activadas. El módulo de instrumentación tiene 106 de 109 sentencias y 29 de 32
ramas ejecutadas. El ejecutor tiene 110 de 146 sentencias y 28 de 38 ramas. Las
pruebas CUDA no forman parte de esta medida de cobertura.

El CRAP usa la complejidad de Radon 6.0.1 y la cobertura de sentencias por
región de coverage.py, mediante el script común del proyecto. El máximo es
25,0191 en `measure_pass`. Esta puntuación orienta la revisión y no demuestra
ausencia de errores. La carga real de la partición y el recorrido satisfactorio
completo de la CLI CUDA todavía necesitan la repetición de los checkpoints
seleccionados.

Las fuentes comprobadas están en `2efdc5c24`. El recibo legible por máquina
recoge huellas de código, tests, logs y resultados en
[verification.json](verification.json). La cobertura y la mutación están en
[quality.json](quality.json) y [mutations.json](mutations.json).

Órdenes ejecutadas con el entorno local y las bibliotecas nativas configuradas:

```bash
uv run --no-sync coverage run --branch --source=benchmarks.refinement_latency,benchmarks.refinement_latency_study -m pytest -q tests/tooling/test_refinement_latency.py tests/tooling/test_refinement_latency_study.py -m 'not cuda'
uv run --no-sync pytest -q tests/tooling/test_refinement_latency.py -m cuda
uv run --no-sync python -m benchmarks.refinement_latency_study --help
uv run --no-sync python scripts/check_repository.py
```

Estas comprobaciones no miden una mejora predictiva ni acreditan menor latencia
en los datos del estudio. Esa comparación está pendiente de finalizar la
selección y ejecutar las pasadas fijadas en el
[contrato de medida](../../../docs/engineering/refinement-latency.md).
