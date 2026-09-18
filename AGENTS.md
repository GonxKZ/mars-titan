# Convenciones permanentes de MARS-TITAN

## Autoría y redacción

- El autor y responsable del proyecto es **Gonzalo García Lama**.
- Escribir en español de España, con un tono natural, preciso y propio de un estudiante de máster. Evitar frases grandilocuentes, traducciones literales y afirmaciones de perfección.
- La documentación describe el trabajo, sus decisiones y sus evidencias. Evitar comentarios sobre el proceso de asistencia en el README y en la documentación técnica.
- Evitar texto de relleno, frases hechas, listas repetitivas y un estilo artificial. Cada párrafo debe aportar una decisión, explicación o evidencia concreta.
- Usar siempre MARS-TITAN como nombre del proyecto, del repositorio y del tablero. Las referencias académicas se describen como proyecto, memoria o trabajo de máster.
- No usar punto y coma en la prosa ni rayas para introducir incisos. Preferir frases claras, comas y paréntesis. Esta norma no altera la sintaxis necesaria del código ni los títulos bibliográficos originales.
- Respetar los requisitos académicos explícitos recogidos en `docs/academic/requirements.md`, incluida la revisión con el director antes de la entrega. No inventar autorizaciones, resultados, lecturas completas, contribuciones ni garantías de originalidad.
- Diferenciar siempre propuesta, implementación, experimento ejecutado y resultado observado. Una mejora esperada no es un resultado.
- Mantener citas verificables y correspondencia con la bibliografía. Usar APA en la memoria. Los derechos de terceros no quedan cubiertos por la licencia MIT del proyecto.

## Investigación

- Trabajo individual de **Tipo 3: comparativa de soluciones**. Los seis objetivos son hitos de investigación. Las carpetas se organizan por responsabilidad.
- La preparación inicial comprende repositorio, configuración, documentos e investigación. No iniciar implementaciones científicas ni entrenamientos sin que esa fase de trabajo se solicite.
- Antes de cambiar alcance, consultar `docs/research/protocol.md`, `docs/research/roadmap.md` y `docs/academic/rubric-matrix.md`.
- Registrar decisiones, configuraciones, semillas, versiones y fallos. Un resultado negativo bien evaluado también es una aportación.
- Mantener el test final cerrado durante selección de modelos. No ajustar transformaciones, umbrales, regímenes o residualizadores con el futuro.
- Dimensionar el trabajo para 32 GB de RAM y RTX 4070 Max-Q de 8 GB. El equipo puede estar encendido 24/7, pero las fechas de entrega siguen sin confirmar.
- Usar el piloto de hasta 64 activos para medir y una comparación principal propuesta de hasta 128. Escalar a 256, al universo completo o a China solo si el presupuesto lo permite. La calidad de validación tiene prioridad sobre ampliar filas o variantes.
- Toda ejecución larga debe poder recuperarse desde un checkpoint coherente que incluya estado de memoria, etiquetas pendientes, RNG, optimizador y cursor confirmado. Ver `docs/engineering/checkpoint-recovery.md`.
- Contrastar las propuestas con antecedentes y objeciones. No afirmar novedad, ausencia de olvido, eliminación total de ruido, precisión perfecta o superioridad antes de aportar evidencia.
- Toda modalidad necesita evidencia de disponibilidad temporal. `period_end` no equivale a fecha de publicación.
- «Causal» significa aquí respetar el orden de información. No afirmar identificación de causas económicas o contrafactuales sin un diseño adicional que la sostenga.

## Desarrollo

- Usar **uv** para paquetes, entornos y ejecución de Python. No usar pip, Poetry o Conda directamente.
- Para cargas de aprendizaje profundo, comprobar `nvidia-smi` y `torch.cuda.is_available()`, seleccionar `cuda:0` y registrar memoria y versiones. Nunca pasar silenciosamente a CPU.
- En tareas independientes compatibles, preferir `/home/gonzalo/.venvs/pytorch-cuda`. El proyecto mantiene un entorno local reproducible con `uv.lock`.
- Buscar archivos y texto con `rg`. No modificar ni mover `dataset/` sin necesidad explícita.
- Mantener lógica reutilizable en `src/mars_titan/`, exploración en `notebooks/`, configuración en `configs/` y optimización nativa en `native/`.
- C++/CUDA solo tras perfilado y con comparación numérica frente a una referencia Python. No añadir kernels sin evidencia de necesidad.
- Comprobar calidad con `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest` y `uv run python scripts/check_repository.py`.
- Ejecutar las comprobaciones localmente. No configurar ni activar GitHub Actions. Mantener las pruebas y la revisión previa a cada publicación sin automatizaciones en GitHub.
- No versionar datasets, secretos, pesos, cachés, resultados voluminosos ni PDF de terceros sin permiso de redistribución.

## Git y GitHub

- Todos los commits siguen **Conventional Commits**, en inglés: `type(scope): short description`.
- Usar ramas breves en inglés acordes al trabajo: `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`, `chore/...`.
- Crear commits atómicos y significativos. Revisar el diff y los archivos incluidos antes de cada commit.
- Durante trabajo prolongado, procurar bloques revisados de unos cuatro a siete minutos cuando encaje con la tarea. Mantener las fechas reales y no dividir artificialmente un cambio solo para aparentar actividad.
- Revisar las issues y el Kanban antes de crear nuevas tareas. Unir solapes por resultado esperado, conservar dependencias y enlazar cualquier consolidación. Aplicar etiquetas por contenido y etapa con un vocabulario coherente.
- Escribir las issues con contexto, trabajo delimitado, criterios comprobables y evidencia, usando el tono natural de un estudiante de máster que investiga aprendizaje automático. No usar punto y coma en la prosa.
- No reescribir historia compartida ni borrar datos. Conservar cambios previos del usuario.
- Al cerrar un objetivo, enlazar evidencia real con la matriz de rúbrica. No marcarlo completado por haber redactado su plan.
