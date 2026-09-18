# Comprobaciones del repositorio

Las pruebas actuales de `tests/tooling/` comprueban utilidades de gestión de la biblioteca de referencias. Incluyen validación de identificadores, metadatos, integridad de archivos y tratamiento de accesos fallidos. No evalúan modelos, kernels, rentabilidad ni hipótesis científicas.

Desde la raíz del repositorio:

```bash
uv run --locked pytest tests/tooling
```

Estas comprobaciones pueden ejecutarse en una máquina con CPU. Su resultado no acredita disponibilidad de GPU ni reproducibilidad de un futuro entrenamiento.

Cuando se implemente la parte científica, cada prueba deberá proteger una propiedad concreta. Entre ellas estarán la disponibilidad de información por fecha, la maduración de las etiquetas, los reinicios de memoria, el cálculo de métricas y la concordancia con referencias numéricas. Las pruebas que necesiten CUDA se distinguirán de las utilidades y no se sustituirán silenciosamente por una ejecución en CPU.

No se añaden pruebas científicas vacías ni modelos de ejemplo para aparentar cobertura. Los experimentos y sus resultados se documentarán cuando se hayan ejecutado.
