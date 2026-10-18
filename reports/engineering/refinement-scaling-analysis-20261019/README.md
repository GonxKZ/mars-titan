# Verificación del resumen de cohortes

El resumen de 64 y 256 activos superó 27 pruebas CPU. Comprueban diferencias e
interacción calculadas a mano, ponderación por sesión con distinto número de
activos, población común fijada, predicciones repetidas, cambios de objetivo,
semillas incompletas y discrepancias de fechas. Los casos de integración leen
Parquet de fixtures, verifican sus huellas y escriben los resultados y las
tablas por sesión en destinos nuevos.

La revisión corrigió dos defectos de la primera versión. La media de semillas
aceptaba modelos o poblaciones adicionales sin avisar. La carga de una fase no
contrastaba su huella de preparación. Los tests reprodujeron ambos fallos antes
de aplicar las correcciones.

Las 14 mutaciones dirigidas se detectaron en copias aisladas. Cubren pérdida de
emparejamiento de objetivos, sustitución de MAE por MSE, signos de delta e
interacción, media de semillas incompletas, elección de una sola semilla, fechas,
selección, preparación, fase confirmada, recuento de filas, código compartido,
protección de aprendizaje y retirada del control cero. Las fuentes originales
no cambiaron durante estas ejecuciones. El denominador corresponde a estos
fallos concretos, no a todos los mutantes posibles.

Coverage.py 7.16.1 registró 172 de 182 sentencias y 85 de 90 ramas del módulo. El
CRAP máximo es 19,0183 en `score_populations`, usando la complejidad de Radon
6.0.1 y cobertura de sentencias por región. La CLI se comprobó con `--help`.
Las comprobaciones de Ruff y formato también pasaron. No hay una ruta CUDA
nueva en este resumen estadístico.

La comprobación de las preparaciones reales confirmó 64, 64, 256 y 256 activos,
con sus huellas originales. Solo leyó metadatos y manifiestos. No abrió
predicciones de evaluación ni calculó resultados de modelos.

Las fuentes comprobadas están en `51efb6396`. Los recibos se conservan en
[verification.json](verification.json), [quality.json](quality.json) y
[mutations.json](mutations.json). La
[descripción de los contrastes](../../../docs/engineering/refinement-scaling-comparison.md)
fija su alcance. La ejecución sobre grupos científicos completos queda pendiente.

```bash
uv run --no-sync coverage run --branch --source=benchmarks.refinement_scaling_analysis -m pytest -q tests/tooling/test_refinement_scaling_analysis.py
uv run --no-sync python -m benchmarks.refinement_scaling_analysis --help
```
