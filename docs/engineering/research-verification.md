# Verificación de la ampliación de investigación

Autor: Gonzalo García Lama. Revisión final: 19 de septiembre de 2026, Europe/Madrid.

Esta revisión recoge el estado de preparación de esta entrega. Distingue documentos y herramientas comprobados de modelos y experimentos pendientes. La [verificación inicial](verification.md) conserva el historial de la primera base publicada.

La [guía de implementación](implementation-guide.md) y el [tablero](../research/task-board.md) recogen 64 guías y 183 dependencias. La biblioteca incorpora TRA como antecedente adicional y reúne 112 referencias y 81 PDF locales. Estos recuentos describen preparación, no ejecución científica.

El [observatorio](https://gonxkz.github.io/mars-titan/) también está publicado y verificado. Su [informe](observatory.md) distingue 112 pruebas Python, 21 JavaScript y comprobaciones de navegador de cualquier resultado científico. La conexión del futuro entrenador, el historial completo y las ejecuciones reales siguen en sus tareas, no se dan por realizados por publicar una interfaz.

## Investigación y alcance

La documentación incluye aprendizaje y memoria biológica, aprendizaje continuo, recurrencia interna, predicción financiera, macroeconomía y eficiencia. La [arquitectura candidata](../research/candidate-architecture.md) separa parámetros compartidos, estado persistente y refinamiento de una consulta. La [revisión adversarial](../research/adversarial-review.md) y el [registro de hipótesis](../research/novelty-ledger.md) recogen antecedentes, objeciones y pruebas capaces de descartar las propuestas.

El diseño se ha corregido para mantener compatibles las claves de memoria con el codificador, esperar a la maduración de las etiquetas y contar todas las lecturas recurrentes. La inspiración en el cerebro no se presenta como una reproducción biológica. Las propuestas tampoco acreditan novedad, ausencia de olvido o superioridad predictiva antes de experimentar.

El alcance propuesto es un piloto de hasta 64 activos y una comparación principal de hasta 128. La [lectura por bloques](full-dataset-training.md) contempla recorrer el conjunto completo sin materializar todas sus ventanas. El [inventario físico](../../reports/resource-inventory.json) no valida sus filas para entrenamiento. La ampliación del universo depende del coste medido en los 32 GB de RAM y la RTX 4070 Max-Q de 8 GB disponibles.

La política de [recuperación](checkpoint-recovery.md) incluye optimizador, RNG, memoria adaptativa, cursor confirmado y etiquetas pendientes. Está especificada para futuras ejecuciones, pero todavía no existe un entrenador ni se han generado checkpoints de pesos. El funcionamiento 24/7 no sustituye el perfilado sostenido ni un calendario de entregas, que sigue sin confirmar.

## Fuentes y datos comprobados

| Elemento | Evidencia disponible | Límite |
| --- | --- | --- |
| Bibliografía | 112 identificadores concordantes entre nueve catálogos y sus BibTeX. | Revisión dirigida con fecha de corte, no lectura íntegra de todas las publicaciones. |
| Biblioteca local | 81 PDF descargados, incluidos cinco libros, y 31 referencias sin PDF. | Los originales no se redistribuyen. Descarga no equivale a validación científica. |
| Integridad de PDF | 80 comprobaciones iniciales con `pdfinfo` y la verificación incremental de TRA, con muestras visuales. | Engram conserva avisos de sintaxis en metadatos. El [registro](../../reports/reference-validation.json) diferencia ambas revisiones. |
| Contexto macro | 140 candidatos con dependencias válidas, de los que 64 tienen metadatos de serie contrastados, 70 son fórmulas derivadas y seis siguen pendientes de identificación completa. | No se ha calculado ni habilitado un panel de 140 entradas de entrenamiento. |
| Primera captura pública | Nueve archivos válidos de ocho proveedores, 5.149.883 bytes y hashes registrados. | Ninguno es admisible automáticamente en el benchmark. La disponibilidad histórica sigue pendiente de auditoría. |
| Posts aportados | [Ocho enlaces revisados](../references/social-followup.md), con acceso, recursos y límites documentados. | Las promesas de ganancias y las demostraciones sin operaciones reproducibles no son evidencia de rendimiento. |

El [catálogo público](../../data/catalogs/public-sources.json) conserva catorce fuentes, incluidas las que no se pudieron obtener. El [manifiesto inicial](../../data/manifests/public-snapshots.json) conserva quince registros de intento. Los errores SEC, GDELT, XLSX y el bloqueo de Stooq no se presentan como descargas satisfactorias. El acceso gratuito tampoco se confunde con permiso de redistribución.

## Herramientas y comprobaciones

Las utilidades ejecutables cubren biblioteca, verificación, adquisición de fuentes y seguimiento del observatorio. No implementan muestras científicas, modelos, entrenamiento o evaluación de rentabilidad.

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

Se retiraron el flujo de calidad y su actualización automática por decisión del proyecto. No había ejecuciones en curso ni en cola al desactivarlo. Posteriormente se autorizó una excepción limitada al despliegue de GitHub Pages. La validación se mantiene local y se repite antes de publicar. Los registros históricos no se eliminan ni se presentan como pruebas automáticas vigentes.

## Seguimiento contrastado en GitHub

La comprobación del catálogo final y del Project confirma:

- 64 tareas canónicas y 64 elementos en el tablero.
- 61 tareas Pendiente y dos entregas de preparación en Hecho, MT-001 y MT-069. MT-068 está Bloqueado por la decisión sobre los automatismos internos de GitHub, aunque su web ya está publicada y verificada.
- 183 dependencias nativas coincidentes con el grafo local, sin ciclos.
- 20 etiquetas del catálogo aplicadas por contenido, seis hitos y 64 asignaciones a Gonzalo.
- Cinco duplicadas cerradas como `not_planned`, con historial conservado y fuera del tablero activo.

La [revisión del catálogo](../research/backlog-review.md) explica las consolidaciones. El [mapa remoto](../../.github/planning/remote-map.json) conserva identificadores y enlaces. Cerrar una duplicada no significa haber ejecutado su experimento. La existencia de un capturador tampoco completa por sí sola la tarea de integrar datos actualizados con control temporal.

## Pendientes que no se ocultan

No se ha entrenado ni comparado ningún modelo. No hay cifras propias de precisión, latencia, rentabilidad o eficiencia energética. Los objetivos de latencia y memoria son presupuestos de diseño que deberán medirse en el equipo real.

La memoria LaTeX no se ha compilado y no sustituye la plantilla oficial del aula. Falta concretar esa plantilla y el calendario con la dirección. La cobertura máxima de la rúbrica dependerá de las evidencias científicas y de la defensa, no del volumen de documentación.

El repositorio se hizo público posteriormente por autorización expresa, tras revisar archivos versionados e historial. El tablero conserva su acceso privado. El rechazo anterior de protección de `main` correspondía al repositorio privado y al plan disponible entonces. No se ha contratado ningún servicio. Los datos, pesos y PDF de terceros siguen fuera de Git.

Tras el cambio de visibilidad se activó la protección de `main`, incluida para administradores. Exige PR, historial lineal y conversaciones resueltas, y bloquea force-push y borrado de la rama. No exige comprobaciones de Actions ni aprobaciones de otro colaborador en este trabajo individual. Las pruebas siguen siendo locales y el único workflow versionado publica Pages. La [verificación posterior de los automatismos internos](observatory.md#límite-de-los-automatismos-de-github) identifica una limitación pendiente de decisión. No se confunde la ausencia de otros YAML propios con la desactivación de todos los trabajos gestionados por GitHub.
