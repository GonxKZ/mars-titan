# Registro de experimentos

Este directorio reservará índices y configuraciones de ejecuciones, sin almacenar checkpoints ni predicciones voluminosas. No contiene experimentos ejecutados.

Un registro deberá enlazar `run_id`, configuración, commit, manifiesto de datos, partición, semilla, estado, tiempo, VRAM y ruta de artefactos. La matriz que determina qué ejecutar está en [experimentos previstos](../docs/research/experiment-matrix.md). Los archivos de resultados irán en `artifacts/`, que no se versiona.
