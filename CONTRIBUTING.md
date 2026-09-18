# Desarrollo y revisión

Este repositorio apoya un proyecto individual. Cualquier cambio debe mantener la trazabilidad de la contribución y respetar las condiciones académicas de la titulación.

1. Abrir una rama coherente con el cambio: `feat/temporal-contract`, `fix/news-availability`, `docs/experimental-protocol`, `test/memory-ordering` o `chore/dependencies`.
2. Usar uv y el entorno bloqueado. Mantener secretos, datos y artefactos grandes fuera de Git.
3. Escribir pruebas para comportamiento nuevo con riesgo de error. Priorizar orden temporal, maduración de etiquetas, estado de memoria y métricas. No escribir pruebas que solo copien constantes de la implementación.
4. Ejecutar los controles del README y revisar el diff. Registrar también experimentos fallidos.
5. Crear commits atómicos en inglés conforme a Conventional Commits, por ejemplo `fix(data): reject records published after prediction time`.
6. En la revisión explicar problema, cambio, evidencia, objetivo afectado y limitaciones. Cerrar hitos solo cuando exista evidencia experimental.

Los nombres de código y ramas se escriben en inglés. La documentación científica, las explicaciones y los mensajes orientados al usuario se redactan en español natural. `main` debe permanecer reproducible. Los modelos no se seleccionan con el resultado del test final.
