# Comprobaciones del repositorio

Las pruebas actuales de `tests/tooling/` comprueban la biblioteca de referencias y las capturas de fuentes públicas. Incluyen validación de identificadores, metadatos, integridad de archivos, formatos, límites de descarga y tratamiento de accesos fallidos. Se ejecutan sin red y verifican que una actualización no sobrescriba capturas anteriores ni habilite datos para el benchmark. No evalúan modelos, kernels, rentabilidad ni hipótesis científicas.

Desde la raíz del repositorio:

```bash
uv run --locked pytest tests/tooling
```

Estas comprobaciones pueden ejecutarse en una máquina con CPU. Su resultado no acredita disponibilidad de GPU ni reproducibilidad de un futuro entrenamiento.

Cuando se implemente la parte científica, cada prueba deberá proteger una propiedad concreta. Entre ellas estarán la disponibilidad de información por fecha, la maduración de las etiquetas, los reinicios de memoria, el cálculo de métricas y la concordancia con referencias numéricas. Las pruebas que necesiten CUDA se distinguirán de las utilidades y no se sustituirán silenciosamente por una ejecución en CPU.

No se añaden pruebas científicas vacías ni modelos de ejemplo para aparentar cobertura. Los experimentos y sus resultados se documentarán cuando se hayan ejecutado.
