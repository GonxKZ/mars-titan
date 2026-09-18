# Verificación de la preparación

Fecha: 18 de septiembre de 2026. Esta revisión corresponde a la base documental, la configuración y la planificación de MARS-TITAN. No acredita la ejecución de experimentos científicos.

## Repositorio y seguimiento

El [repositorio privado](https://github.com/GonxKZ/mars-titan) tiene `main` como rama principal, licencia MIT, autoría y cita de Gonzalo García Lama y plantillas de revisión. Los commits se redactan en inglés conforme a Conventional Commits.

El [Project MARS-TITAN](https://github.com/users/GonxKZ/projects/4) está vinculado al repositorio. Incluye una vista Kanban agrupada por estado y una vista tabular del plan completo. Las 46 issues están asignadas a Gonzalo, distribuidas en seis hitos y tres tareas transversales sin hito. Tienen prioridad, tamaño, criterios de aceptación, referencias a la rúbrica y 124 relaciones nativas de dependencia.

Los estados del tablero son Pendiente, En curso, En revisión, Bloqueado y Hecho. Solo se cierra la preparación documental MT-001. Las 45 tareas restantes siguen pendientes de trabajo científico o académico. El [mapa remoto](../../.github/planning/remote-map.json) conserva identificadores, enlaces y fecha de comprobación. Las listas y casillas de las issues se comprobaron mediante el renderizado Markdown de GitHub.

## Comprobaciones realizadas

| Comprobación | Resultado y alcance |
| --- | --- |
| Entorno uv | Instalación de las herramientas de desarrollo desde `uv.lock` en Python 3.12. |
| Pruebas de mantenimiento | 13 pruebas correctas sobre identificadores, conservación de archivos, procedencia, cambio de URL, rechazo de HTML y descargas incompletas. |
| Descarga real y caché | Una descarga nueva de Titans produjo recibo, tamaño y hash. La segunda ejecución reutilizó el mismo archivo y conservó el hash. |
| Ruff | Revisión de código y formato correctas. |
| Documentación | Enlaces locales, JSON, YAML, TOML, BibTeX, secciones de issues y dependencias del catálogo válidos. |
| Bibliografía | 38 identificadores únicos y coincidentes entre catálogos y BibTeX. |
| Biblioteca | 28 PDF interpretables por `pdfinfo`, con hashes registrados. Se inspeccionaron visualmente la rúbrica y portadas de referencias principales. |
| Fuentes recibidas | Siete hashes de documentos y catálogos pequeños coinciden con los originales. No es una auditoría de todos los archivos de FinMultiTime. |
| CMake | Configuración completada con C/C++ y con CUDA activada en el equipo local. No existen kernels ni algoritmos científicos que evaluar todavía. |
| GPU | RTX 4070 Laptop, 8.188 MiB, PyTorch compartido `2.14.0+cu130` y operación mínima en `cuda:0` comprobados. No se entrenó ningún modelo. |
| Exclusiones de Git | Dataset, PDF locales, entorno virtual, pesos y resultados grandes excluidos. |

La ejecución de [Quality sobre la base publicada](https://github.com/GonxKZ/mars-titan/actions/runs/35376345437) terminó correctamente en el commit `b3fb31d8e8fe2f8b6096e2d08a8141833248b914`. Las siguientes revisiones documentales vuelven a ejecutar ese mismo flujo. El trabajo puede consultarse en [GitHub Actions](https://github.com/GonxKZ/mars-titan/actions/workflows/quality.yml).

## Configuración de GitHub y límites pendientes

Se habilitaron las alertas de dependencias. Las acciones están fijadas por SHA y el mantenimiento de sus versiones queda configurado con Dependabot. El repositorio permite squash merge y solicita borrar ramas después de la integración. Los cambios se organizan en ramas convencionales.

GitHub rechazó la activación de protección de `main` con HTTP 403 porque el plan actual no la permite en repositorios privados. La respuesta exige GitHub Pro o visibilidad pública. El repositorio conserva su visibilidad privada y no se afirma que la protección esté activa. El flujo de calidad y las reglas documentadas sí están disponibles.

La plantilla del aula, las instrucciones específicas de formato y el calendario académico siguen pendientes de aportación y revisión con la dirección. La fuente LaTeX es una base de trabajo y no una plantilla oficial. No se ha comprobado su compilación porque no hay una distribución LaTeX instalada. Sus cinco claves de cita se han contrastado con la bibliografía.

La preparación no garantiza una calificación ni que la arquitectura supere a las referencias. La matriz de rúbrica recoge qué evidencia deberá producir el trabajo posterior y las condiciones que deben revisarse antes de cualquier entrega.
