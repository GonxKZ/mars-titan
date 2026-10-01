# Desarrollo y revisión

Este repositorio desarrolla un proyecto individual de investigación. Cada cambio debe conservar su trazabilidad, sus pruebas y los límites de la evidencia que aporta.

1. Consultar la issue y su tarjeta del Project. Reutilizar la tarea adecuada o crear una sin duplicados, moverla a En curso y abrir una rama asociada desde `develop`, por ejemplo `feat/mt-005-data-preparation`.
2. Usar uv y el entorno bloqueado. Mantener secretos, datos y artefactos grandes fuera de Git.
3. Escribir pruebas para comportamiento nuevo con riesgo de error. Priorizar orden temporal, maduración de etiquetas, estado de memoria y métricas. No escribir pruebas que solo copien constantes de la implementación.
4. Ejecutar localmente los controles del README y revisar el diff. Actions se limita al despliegue de GitHub Pages, sin pruebas ni entrenamientos. Registrar también experimentos fallidos.
5. Crear commits atómicos en inglés conforme a Conventional Commits, por ejemplo `fix(data): reject records published after prediction time`.
6. En la revisión explicar problema, cambio, evidencia, objetivo afectado y limitaciones. Abrir la PR hacia `develop` y pasar la tarjeta a En revisión. Tras verificar la integración y sus criterios, cerrar la issue y moverla a Hecho. La promoción de `develop` a `main` es una revisión posterior. Cerrar hitos científicos solo cuando exista evidencia experimental.

La documentación, los comentarios del código y los mensajes propios orientados al usuario se redactan en español natural. Los identificadores técnicos, nombres de código y ramas conservan sus convenciones compatibles. Los títulos bibliográficos, licencias y datos de terceros mantienen su forma original. `main` debe permanecer reproducible. Los modelos no se seleccionan con el resultado de la prueba final.
