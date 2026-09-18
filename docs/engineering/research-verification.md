# Verificación de la ampliación de investigación

Autor: Gonzalo García Lama. Fecha: 18 de septiembre de 2026.

Esta revisión recoge la ampliación posterior a la [preparación inicial](verification.md). Distingue los documentos y herramientas comprobados de los modelos y experimentos que siguen pendientes. Las cifras anteriores se conservan como historial, no como inventario actual.

## Investigación y alcance

La documentación incluye aprendizaje y memoria biológica, aprendizaje continuo, recurrencia interna, predicción financiera, macroeconomía y eficiencia. La [arquitectura candidata](../research/candidate-architecture.md) separa parámetros compartidos, estado persistente y refinamiento de una consulta. La [revisión adversarial](../research/adversarial-review.md) y el [registro de hipótesis](../research/novelty-ledger.md) recogen antecedentes, objeciones y pruebas capaces de descartar las propuestas.

El diseño se ha corregido para mantener compatibles las claves de memoria con el codificador, esperar a la maduración de las etiquetas y contar todas las lecturas recurrentes. La inspiración en el cerebro no se presenta como una reproducción biológica. Las propuestas tampoco acreditan novedad, ausencia de olvido o superioridad predictiva antes de experimentar.

El alcance propuesto es un piloto de hasta 64 activos y una comparación principal de hasta 128. La [lectura por bloques](full-dataset-training.md) contempla recorrer el conjunto completo sin materializar todas sus ventanas. El [inventario físico](../../reports/resource-inventory.json) no valida sus filas para entrenamiento. La ampliación del universo depende del coste medido en los 32 GB de RAM y la RTX 4070 Max-Q de 8 GB disponibles.

La política de [recuperación](checkpoint-recovery.md) incluye optimizador, RNG, memoria adaptativa, cursor confirmado y etiquetas pendientes. Está especificada para futuras ejecuciones, pero todavía no existe un entrenador ni se han generado checkpoints de pesos. El funcionamiento 24/7 no sustituye el perfilado sostenido ni un calendario de entregas, que sigue sin confirmar.

## Fuentes y datos comprobados

| Elemento | Evidencia disponible | Límite |
| --- | --- | --- |
| Bibliografía | 111 identificadores concordantes entre nueve catálogos y sus BibTeX. | Revisión dirigida con fecha de corte, no lectura íntegra de todas las publicaciones. |
| Biblioteca local | 80 PDF descargados, incluidos cinco libros, y 31 referencias sin PDF. | Los originales no se redistribuyen. Descarga no equivale a validación científica. |
| Integridad de PDF | Los 80 archivos son interpretables por `pdfinfo`. Se inspeccionaron muestras visuales. | Engram conserva avisos de sintaxis en metadatos. El [registro](../../reports/reference-validation.json) los documenta. |
| Contexto macro | 140 candidatos con dependencias válidas, de los que 64 tienen metadatos de serie contrastados, 70 son fórmulas derivadas y seis siguen pendientes de identificación completa. | No se ha calculado ni habilitado un panel de 140 entradas de entrenamiento. |
| Primera captura pública | Nueve archivos válidos de ocho proveedores, 5.149.883 bytes y hashes registrados. | Ninguno es admisible automáticamente en el benchmark. La disponibilidad histórica sigue pendiente de auditoría. |
| Posts aportados | [Ocho enlaces revisados](../references/social-followup.md), con acceso, recursos y límites documentados. | Las promesas de ganancias y las demostraciones sin operaciones reproducibles no son evidencia de rendimiento. |

El [catálogo público](../../data/catalogs/public-sources.json) conserva catorce fuentes, incluidas las que no se pudieron obtener. El [manifiesto inicial](../../data/manifests/public-snapshots.json) conserva quince registros de intento. Los errores SEC, GDELT, XLSX y el bloqueo de Stooq no se presentan como descargas satisfactorias. El acceso gratuito tampoco se confunde con permiso de redistribución.

## Herramientas y comprobaciones

Las utilidades ejecutables se limitan a biblioteca, verificación y adquisición de fuentes. No implementan muestras científicas, modelos, entrenamiento o evaluación de rentabilidad.

El [actualizador público](../data/public-source-updates.md) ofrece ocho fuentes renovables por defecto y selección explícita del PDF fijo de Apple. Conserva versiones, hashes y fallos, limita tiempo y bytes, valida formatos y suspende el host tras HTTP 403 o 429. No instala tareas programadas ni mezcla capturas nuevas con FinMultiTime. La prueba real se limita al RSS monetario oficial, además de las nueve adquisiciones iniciales ya registradas.

Las 54 pruebas sin red de biblioteca y capturas pasan en el entorno local y en una exportación limpia del árbol preparado para publicar. Ruff comprueba código y formato sin incidencias. El verificador del repositorio valida enlaces locales, JSON, YAML, TOML, BibTeX, catálogo macro y dependencias de tareas. CMake configura la estructura C/C++ sin fuentes científicas. La detección de CUDA y una operación mínima ya constan en la [revisión del entorno](reproducibility.md), pero no son una medición de latencia del predictor.

La captura real corregida del RSS se conserva en `data/external/20260918T194800.921923Z/manifest.json`, con HTTP 200, 15 entradas y 9.645 bytes. Sus marcas respetan inicio, adquisición y fin. Las capturas previas permanecen intactas. La limitación de precisión temporal de la primera prueba del actualizador se documenta en su guía, sin reescribir aquella evidencia.

Comandos de comprobación desde la raíz:

```bash
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/check_repository.py
cmake -S native -B native/build -DMARS_TITAN_ENABLE_CUDA=OFF
```

El flujo [Quality del actualizador](https://github.com/GonxKZ/mars-titan/actions/runs/35388152845) terminó correctamente sobre el commit `06b960eb9eb4efa8fa46984c353acd52db4adf1e`. Los resultados posteriores se conservan en [GitHub Actions](https://github.com/GonxKZ/mars-titan/actions/workflows/quality.yml). Debe consultarse el commit de cada ejecución, ya que una comprobación correcta no valida cambios posteriores.

## Seguimiento contrastado en GitHub

La comprobación independiente del repositorio y del Project confirma:

- 62 tareas canónicas y 62 elementos en el tablero.
- 61 tareas Pendiente y MT-001 en Hecho por la preparación original.
- 177 dependencias nativas coincidentes con el grafo local, sin ciclos.
- 20 etiquetas del catálogo aplicadas por contenido, seis hitos y 62 asignaciones a Gonzalo.
- Cinco duplicadas cerradas como `not_planned`, con historial conservado y fuera del tablero activo.

La [revisión del catálogo](../research/backlog-review.md) explica las consolidaciones. El [mapa remoto](../../.github/planning/remote-map.json) conserva identificadores y enlaces. Cerrar una duplicada no significa haber ejecutado su experimento. La existencia de un capturador tampoco completa por sí sola la tarea de integrar datos actualizados con control temporal.

## Pendientes que no se ocultan

No se ha entrenado ni comparado ningún modelo. No hay cifras propias de precisión, latencia, rentabilidad o eficiencia energética. Los objetivos de latencia y memoria son presupuestos de diseño que deberán medirse en el equipo real.

La memoria LaTeX no se ha compilado y no sustituye la plantilla oficial del aula. Falta concretar esa plantilla y el calendario con la dirección. La cobertura máxima de la rúbrica dependerá de las evidencias científicas y de la defensa, no del volumen de documentación.

El repositorio y el tablero permanecen privados. La protección de `main` no está activa porque GitHub la rechazó con el plan actual. Se conservan el flujo de calidad, las convenciones y la limitación documentada, sin cambiar la visibilidad ni contratar servicios.
