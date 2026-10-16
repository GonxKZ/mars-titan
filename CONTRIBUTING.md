# Contribuir a MARS-TITAN

MARS-TITAN es un proyecto individual de investigación de un trabajo de máster. Cada cambio debe conservar su trazabilidad, sus pruebas y los límites de la evidencia que aporta. Esta guía basta para preparar un clon limpio, ejecutar las comprobaciones y publicar un cambio. El detalle de cada parte está en los documentos enlazados.

## Preparar un clon limpio

Requisitos: Linux x86-64, Git, [uv](https://docs.astral.sh/uv/), GNU Make y ripgrep. uv instala Python 3.12 y las dependencias de `uv.lock`, incluido PyTorch con su índice CUDA 13.0. La compilación nativa necesita además CMake 3.24 o posterior, Ninja y Clang. `make docs` necesita Node.js y una distribución LaTeX con `latexmk`.

```bash
git clone https://github.com/GonxKZ/mars-titan.git
cd mars-titan
make setup
make check
```

`make setup` crea `.venv` con `uv sync --locked --all-extras`. `make check` ejecuta Ruff, el formato, `scripts/check_repository.py` y las pruebas rápidas. En un clon limpio del 13 de octubre de 2026, con la caché de uv ya llena y el equipo cargado (carga media de 13 a 17 en 16 hilos), `make setup` tardó 1,2 s y `make check` unos cinco minutos. `test-fast` superó 6.510 pruebas, omitió 343 con su motivo y dejó fuera 3.241, con un pico de memoria de 1,7 GB. Las pruebas añadidas después sin medida alargaron `test-fast` hasta 67 minutos el 15 de octubre. Con la lista de lentas corregida ([#635](https://github.com/GonxKZ/mars-titan/issues/635)), tardó 4 min 24 s en serie en una plaza `memslot suite`, con otra suite en la segunda plaza y una carga media de 4 a 9. Superó 7.463 pruebas, omitió 377 y dejó fuera 4.075, con un pico de 1,4 GB de memoria residente. Desde [#491](https://github.com/GonxKZ/mars-titan/issues/491), `test-fast` reparte los archivos entre tres procesos de pytest-xdist (`-n 3 --dist loadfile`). En tres rondas dentro de una plaza `memslot suite` de 6 GiB, la mediana bajó de 205 s en serie a 92 s, con un máximo de 2,5 GiB de memoria anónima y el mismo resultado en cada prueba ([informe](reports/engineering/test-suite-parallelism-20261015/README.md)). Sin caché, `make setup` tiene que descargar las ruedas de un entorno que ocupa unos 6,5 GB, sobre todo por PyTorch con CUDA.

Los datos de FinMultiTime, las ediciones derivadas, los pesos y los resultados voluminosos no están en el repositorio. Las pruebas usan datos sintéticos o fixtures pequeños, y las que necesitan un artefacto local lo dicen al omitirse.

## Órdenes habituales

El [Makefile](Makefile) es el punto de entrada. `make` sin objetivo lista los disponibles.

| Orden | Qué hace | Qué necesita |
| --- | --- | --- |
| `make setup` | Entorno bloqueado con todos los extras | uv |
| `make lint` | `ruff check`, `ruff format --check` y `scripts/check_repository.py` | El entorno |
| `make test-fast` | Pruebas en CPU sin las marcas `slow`, `native` ni `cuda`, repartidas por archivo entre `TEST_WORKERS` procesos | El entorno y unos 2,5 GiB de memoria |
| `make check` | `lint` y `test-fast`. Es la comprobación mínima antes de abrir una PR | El entorno |
| `make test` | Suite CPU completa por carpetas, con los binarios nativos si están compilados | Unas horas y hasta 6 GiB de memoria en `tests/training` |
| `make native-build` | Compila `native-release`, `native-ppo-release` y `native-candidate-cuda` en `build/native/` | CMake, Ninja, Clang y el PyTorch del entorno |
| `make test-native` | Pruebas `native` en CPU y CTest de los tres presets, sin las etiquetas `optimizer-steps` ni `cuda` | `make native-build` |
| `make test-cuda` | Pruebas `cuda` en `cuda:0` con `MARS_TITAN_REQUIRE_CUDA=1` | Una GPU NVIDIA libre para esta única carga |
| `make docs` | Enlaces y órdenes documentadas, pruebas de la web del observatorio y PDF de la memoria en `build/docs/` | Node.js y LaTeX |

Las variables `THREADS` (por defecto 2, por proceso), `TEST_WORKERS` (3, y 0 para un solo proceso), `NATIVE_JOBS` (2) y `RUN` (`uv run --locked`) se pueden cambiar en la llamada, por ejemplo `make test-fast TEST_WORKERS=0`. Solo `test-fast` usa varios procesos. Con la lista de lentas atrasada, las doce ejecuciones en paralelo de la primera medida superaron los 6 GiB de la plaza, así que `make test` y las pruebas lentas siguen en serie. Todas las comprobaciones y entrenamientos se ejecutan en local. GitHub Actions solo publica GitHub Pages.

## Pruebas

| Marca | Significado |
| --- | --- |
| `slow` | La suma de los casos de la función alcanza un segundo en las medidas de [tests/slow_tests.txt](tests/slow_tests.txt) |
| `native` | Necesita un binario o enlace nativo. Sin él se omite y el motivo dice qué preset lo compila y qué variable lo declara |
| `cuda` | Necesita `cuda:0`. Sin CUDA se omite y nunca pasa a CPU |
| `unguarded_fit` | Ajusta sin pasar por un punto de entrada protegido. Se omite mientras rige el bloqueo de aprendizaje |
| `native_binding`, `external_reference` | Con el modo estricto, su omisión cuenta como fallo |

Se combinan con `-m`, por ejemplo `uv run --locked pytest -m "native and not cuda" tests/memory`. Cada omisión explica qué falta y cómo conseguirlo, y `-rs` lo muestra al final. La [guía de pruebas](tests/README.md) describe la protección del aprendizaje, el modo estricto, la suite completa y cómo regenerar la lista de pruebas lentas.

Mientras exista la protección local de aprendizaje, las pruebas que intentan un paso de optimizador se omiten antes de modificar pesos. Sin ese archivo, como en un clon limpio de otro equipo, el aprendizaje está permitido y esas pruebas ajustan modelos pequeños con datos sintéticos.

## Dónde está cada cosa

| Carpeta | Contenido |
| --- | --- |
| [src/mars_titan](src/mars_titan/README.md) | Lógica reutilizable por dominio: datos, modelos, memoria, entrenamiento, postentrenamiento, evaluación y simulación. El README separa los módulos que solo reproducen ediciones anteriores |
| [native](native/README.md) | C++20 y CUDA con CMake: simulación, PPO y KLPO, banco episódico y GRU candidata |
| [configs](configs/README.md) | Declaraciones de campañas, matrices y recetas. Su huella forma parte de las identidades |
| [scripts](scripts/README.md) | Órdenes de línea de comandos que envuelven módulos de `src/` |
| [tests](tests/README.md) | Pruebas por responsabilidad, con la misma estructura que `src/` |
| `benchmarks/` | Medidas de rendimiento reproducibles, sin resultados predictivos |
| [reports](reports/README.md) | Informes, recibos y evidencias publicadas |
| [docs](docs/README.md) | Investigación, ingeniería, datos y bibliografía |
| [site](site/README.md) | Observatorio estático de las campañas |
| [thesis](thesis/README.md) | Memoria de trabajo en LaTeX |

Las [variables de entorno](docs/engineering/environment-variables.md) están en una tabla única, con su significado y su valor por defecto. La [auditoría de la estructura](docs/engineering/repository-structure.md) recoge la reorganización prevista de las carpetas.

## Tareas, ramas y commits

1. Revisar la issue y su tarjeta del Project MARS-TITAN. Reutilizar la tarea adecuada o crear una con la plantilla, sin duplicar objetivos. Moverla a En curso.
2. Crear la rama desde `develop` con un nombre breve en inglés que diga el tipo de trabajo y la tarea, por ejemplo `feat/mt-068-observatory`, `fix/...`, `docs/...`, `refactor/...`, `test/...` o `chore/...`.
3. Escribir pruebas para el comportamiento nuevo con riesgo de error. Priorizar orden temporal, maduración de etiquetas, estado de memoria y métricas. No escribir pruebas que solo copien constantes de la implementación.
4. Crear commits atómicos en inglés con [Conventional Commits](https://www.conventionalcommits.org/), `type(scope): short description`, por ejemplo `fix(data): reject records published after prediction time`. Revisar el diff y los archivos incluidos antes de cada commit.
5. Ejecutar `make check` y, si el cambio toca código nativo o CUDA, `make test-native` o `make test-cuda`. Registrar también los experimentos fallidos.

No se versionan datos, secretos, pesos, cachés, resultados voluminosos ni PDF de terceros sin permiso de redistribución. Tampoco instrucciones privadas del entorno de trabajo ni rutas personales. Los informes publicados usan rutas relativas o marcadores.

## Revisión e integración

La PR va hacia `develop` con la plantilla del repositorio: problema y cambio, criterios comprobables, evidencia, objetivo afectado y limitaciones. La tarjeta pasa a En revisión cuando la evidencia está lista. Tras integrar y comprobar sus criterios, la issue se cierra y pasa a Hecho. Una tarea científica no se cierra por completar su documentación, y un hito solo se da por cumplido con evidencia experimental.

## Publicación a `main`

`main` recoge versiones revisadas de `develop`. Cada publicación es una PR de `develop` a `main` que se integra con un commit de merge, nunca con squash ni rebase, para conservar el historial completo. Después `develop` avanza hasta ese commit de merge, de modo que las dos ramas quedan en el mismo punto y la siguiente publicación solo contiene trabajo nuevo:

```bash
git fetch origin
git switch develop
git merge --ff-only origin/main
git push origin develop
```

Si `develop` recibió cambios mientras se revisaba la PR, se publican en la siguiente. `main` debe permanecer reproducible y los modelos no se seleccionan con el resultado de la prueba final.

## Redacción

La documentación, los comentarios del código y los mensajes orientados al usuario se escriben en español natural. Los identificadores técnicos, nombres de código y ramas conservan sus convenciones. En la prosa se evitan calcos del inglés como *arm* u *oracle*: lo comparado es un «modelo», una «variante» o una «configuración», y se habla de «referencia», «óptimo exacto» o «cota con información futura» según el caso. `scripts/check_repository.py` rechaza esos calcos fuera de la evidencia generada y de las copias conservadas. Los títulos bibliográficos, licencias y datos de terceros mantienen su forma original. Se distingue siempre propuesta, implementación, experimento ejecutado y resultado observado.
