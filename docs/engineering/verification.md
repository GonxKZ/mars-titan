# Verificación de la preparación

Fecha: 18 de septiembre de 2026. Esta revisión corresponde a la base documental, la configuración y la planificación de MARS-TITAN. No acredita la ejecución de experimentos científicos.

Registro histórico de la preparación inicial. Las cifras de referencias, pruebas y tareas de esta página describen aquella entrega. El inventario posterior y sus límites se recogen en la [verificación de la ampliación](research-verification.md).

## Repositorio y seguimiento

El [repositorio privado](https://github.com/GonxKZ/mars-titan) tiene `main` como rama principal, licencia MIT, autoría y cita de Gonzalo García Lama y plantillas de revisión. Los commits se redactan en inglés conforme a Conventional Commits.

El [Project MARS-TITAN](https://github.com/users/GonxKZ/projects/4) está vinculado al repositorio. Incluye una vista Kanban agrupada por estado y una vista tabular del plan completo. Las 46 issues están asignadas a Gonzalo, distribuidas en seis hitos y tres tareas transversales sin hito. Tienen prioridad, tamaño, criterios de aceptación y 124 relaciones nativas de dependencia.

Los estados del tablero son Pendiente, En curso, En revisión, Bloqueado y Hecho. Solo se cierra la preparación documental MT-001. Las 45 tareas restantes siguen pendientes de trabajo científico o técnico. El [mapa remoto](../../.github/planning/remote-map.json) conserva identificadores, enlaces y fecha de comprobación. Las listas y casillas de las issues se comprobaron mediante el renderizado Markdown de GitHub.

## Comprobaciones realizadas

| Comprobación | Resultado y alcance |
| --- | --- |
| Entorno uv | Instalación de las herramientas de desarrollo desde `uv.lock` en Python 3.12. |
| Pruebas de mantenimiento | 13 pruebas correctas sobre identificadores, conservación de archivos, procedencia, cambio de URL, rechazo de HTML y descargas incompletas. |
| Descarga real y caché | Una descarga nueva de Titans produjo recibo, tamaño y hash. La segunda ejecución reutilizó el mismo archivo y conservó el hash. |
| Ruff | Revisión de código y formato correctas. |
| Documentación | Enlaces locales, JSON, YAML, TOML, BibTeX, secciones de issues y dependencias del catálogo válidos. |
| Bibliografía | 38 identificadores únicos y coincidentes entre catálogos y BibTeX. |
| Biblioteca | 28 PDF interpretables por `pdfinfo`, con hashes registrados. Se inspeccionaron visualmente las portadas de las referencias principales. |
| Fuentes recibidas | Siete hashes de documentos y catálogos pequeños coinciden con los originales. No es una auditoría de todos los archivos de FinMultiTime. |
| CMake | Configuración completada con C/C++ y con CUDA activada en el equipo local. No existen kernels ni algoritmos científicos que evaluar todavía. |
| GPU | RTX 4070 Laptop, 8.188 MiB, PyTorch compartido `2.14.0+cu130` y operación mínima en `cuda:0` comprobados. No se entrenó ningún modelo. |
| Exclusiones de Git | Dataset, PDF locales, entorno virtual, pesos y resultados grandes excluidos. |

La ejecución histórica de [Quality sobre la base publicada](https://github.com/GonxKZ/mars-titan/actions/runs/35376345437) terminó correctamente en el commit `b3fb31d8e8fe2f8b6096e2d08a8141833248b914`. Posteriormente se desactivó GitHub Actions por decisión del proyecto y se retiró su configuración. Este enlace conserva una evidencia anterior, no una automatización vigente. Las revisiones posteriores se realizan localmente.

## Configuración de GitHub y límites pendientes

Se habilitaron las alertas de dependencias. La configuración inicial fijaba las acciones por SHA y programaba su mantenimiento. Esa automatización se ha retirado junto con GitHub Actions. El repositorio permite squash merge y solicita borrar ramas después de la integración. Los cambios se organizan en ramas convencionales.

GitHub rechazó la activación de protección de `main` con HTTP 403 porque el plan actual no la permite en repositorios privados. La respuesta exige GitHub Pro o visibilidad pública. El repositorio conserva su visibilidad privada y no se afirma que la protección esté activa. Los controles locales y las reglas documentadas sí están disponibles.

El formato final del documento y el calendario de redacción siguen pendientes de concretar. La fuente LaTeX es una base de trabajo. No se ha comprobado su compilación porque no hay una distribución LaTeX instalada. Sus cinco claves de cita se han contrastado con la bibliografía.

La preparación no garantiza que la arquitectura supere a las referencias. El trabajo posterior deberá aportar evidencia experimental y revisar sus condiciones antes de publicar conclusiones.
