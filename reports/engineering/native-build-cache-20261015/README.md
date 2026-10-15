# Caché de compilación entre worktrees y tamaños de las variantes (15 de octubre de 2026)

Este informe completa [#487](https://github.com/GonxKZ/mars-titan/issues/487) y [#486](https://github.com/GonxKZ/mars-titan/issues/486) después de las [variantes de compilación del 10 de octubre](../native-build-variants-20261010/README.md). Recoge tres repeticiones de las compilaciones limpia, incremental y con ccache, el motivo por el que ccache reutilizaba poco entre worktrees, su corrección y el tamaño de las salidas de base, LTO, PGO y LTO con PGO. Ninguna medida ejecuta pasos de optimizador y la protección local de aprendizaje siguió activa.

## Por qué ccache reutilizaba poco entre worktrees

La identidad compilada tenía dos huellas como definiciones `PUBLIC` de `mars_titan_financial`:

- `MARS_TITAN_NATIVE_SOURCE_SHA256` resume el contenido de las fuentes nativas y se calcula al configurar, con `CMAKE_CONFIGURE_DEPENDS` sobre esas fuentes.
- `MARS_TITAN_NATIVE_BUILD_SHA256` resume la configuración de los objetivos, incluidas las rutas absolutas de fuentes y cabeceras.

Las dos viajaban en la orden de compilación de todas las unidades de los objetivos que enlazan con `mars_titan_financial`, 55 de las 79 de `native-ppo-release`, aunque solo 6 las usan. Otro worktree con las mismas fuentes tiene otras rutas y, por tanto, otra huella de compilación en esas 55 órdenes. El modo directo de ccache, que resume la orden sin preprocesar, falla en todas ellas. El modo preprocesado acierta en las 49 que no usan las huellas, pero antes ejecuta el preprocesador en cada una, y las 6 restantes se compilan de nuevo. Además, editar cualquier fuente nativa cambia la huella de fuentes, CMake vuelve a configurar y Ninja vuelve a lanzar esas 55 órdenes.

La corrección mueve las seis definiciones de la identidad (versión, las dos huellas, compilador, versión del compilador y tipo de compilación) a `native/src/build_identity.cpp`, que las expone con funciones de `mars_titan::build` (`native/include/mars_titan/build_identity.hpp`). Las seis unidades que las usaban llaman a esas funciones. Las huellas no cambian de contenido: la de compilación sigue incluyendo las definiciones de cada fuente de los objetivos de la identidad, también las de `build_identity.cpp`. Tras el cambio, solo la orden de esa unidad lleva la identidad.

## Medidas

Todas las compilaciones son de `native-ppo-release` con Clang 21.1.8, CMake 4.2.3, Ninja 1.13.2 con dos trabajos, ccache 4.12.3 y LibTorch del entorno compartido, en el Ryzen 9 8945HS con el perfil de ahorro y cada una en su trabajo `memslot heavy`. El equipo tenía otras cargas y la carga media y la temperatura de cada compilación quedan en su recibo. Con carga, el mismo trabajo llega a tardar el doble, así que solo se comparan medidas tomadas una junto a otra.

### Primera tanda (52802c61, madrugada del 15 de octubre)

Tres repeticiones, con carga media de 9 a 19 y temperaturas de 46 a 80 °C. La compilación limpia tiene 133 pasos de Ninja y 77 unidades de compilación.

| Medida | Ronda 1 | Ronda 2 | Ronda 3 |
| --- | ---: | ---: | ---: |
| Limpia, sin ccache | 919 s | 990 s | 1.190 s |
| Limpia, ccache vacía (77 fallos) | 666 s | 1.376 s | 869 s |
| `financial_batch.hpp` tocada, sin ccache (66 pasos) | 257 s | 511 s | 458 s |
| La misma cabecera tocada, con ccache (27 aciertos directos) | 42 s | 18 s | 19 s |

Tocar una cabecera sin cambiar su contenido relanza las 27 unidades que la incluyen, y ccache las resuelve todas en modo directo, porque la cabecera tiene el mismo resumen. Con una caché vacía, ccache no añade un coste que se distinga de la variación entre rondas.

Esta tanda incluía también, en cada ronda, la compilación en un segundo worktree con la caché ya llena. Esas tres rondas no sirven. El guion de medida de los dos worktrees era el del primero, y compila las fuentes de la raíz donde está el guion. Así que compilaron otra vez el primer worktree, con `CCACHE_BASEDIR` apuntando al segundo, y sus 77 fallos no dicen nada sobre ccache entre worktrees. La segunda tanda repite esa medida con el guion de cada árbol.

### Segunda tanda (identidad en una unidad, tarde del 15 de octubre)

Cada pareja compila un worktree con la caché vacía y después otro worktree con las mismas fuentes y la caché que dejó el primero, cada uno con su propio guion y `CCACHE_BASEDIR` en su raíz. La base es `develop` en 06d8e4e2, con 79 unidades. La rama tiene una unidad más, `build_identity.cpp`. Después se añade una línea de comentario a `native/src/ppo_policy.cpp`, se compila, se deshace el cambio y se vuelve a compilar.

| Medida | Base (06d8e4e2) | Rama, ronda 1 | Rama, ronda 2 | Rama, ronda 3 |
| --- | ---: | ---: | ---: | ---: |
| Segundo worktree, caché llena | 116,4 s | 8,5 s | 10,7 s | 13,7 s |
| Aciertos directos, preprocesados y fallos | 24, 49 y 6 | 79, 0 y 1 | 79, 0 y 1 | 79, 0 y 1 |
| Objetos distintos entre los dos worktrees | 6 de 79 | 1 de 80 | 1 de 80 | 1 de 80 |
| Una línea añadida a `ppo_policy.cpp` | 89,3 s, 94 pasos | 11,0 s, 36 pasos | | |
| Aciertos y fallos de esa edición | 49 preprocesados y 6 fallos | 1 preprocesado y 1 fallo | | |
| Vuelta atrás | 11,0 s, 55 aciertos directos | 10,3 s, 2 aciertos directos | | |

Los objetos que difieren entre worktrees solo contienen otra identidad: los 6 que usaban las huellas en la base y `build_identity.cpp.o` en la rama. Los ejecutables y bibliotecas enlazados difieren en 35 de 46 salidas en los dos casos, porque llevan dentro la identidad y la huella del enlazador. En la rama, la edición vuelve a compilar `ppo_policy.cpp`, que ccache resuelve en modo preprocesado porque un comentario no cambia la salida del preprocesador, y `build_identity.cpp`, cuya huella de fuentes cambia. Los otros 34 pasos no compilan.

Las compilaciones en frío de esta tanda no se comparan entre sí: la de la base tardó 985 s con el procesador a 80 °C y carga media de 8, y las de la rama 352, 456 y 524 s con 46 a 60 °C y carga de 3 a 6. La corrección no cambia el trabajo de una compilación limpia, solo añade una unidad pequeña.

Con la caché compartida, montar un worktree nuevo para una rama pasa de unos dos minutos a unos diez segundos, y editar una fuente nativa deja de relanzar 55 compilaciones.

## Tamaños (#486)

El 10 de octubre se midieron los tiempos y las huellas de base, LTO, PGO y LTO con PGO en el recorrido de la etapa RL ([informe](../native-build-variants-20261010/README.md)), y no se adoptó ninguna variante. Faltaba el tamaño de las salidas. `native_build_variants.py` registra ahora `linked_bytes`, el tamaño de cada ejecutable y biblioteca enlazados. Las cuatro variantes se compilaron en el mismo árbol, 52802c61 con ese registro.

El perfil de PGO es nuevo. Las cintas del 10 de octubre son anteriores a #524 y el binario las rechaza, así que se regeneraron con `benchmarks/rl_stage_tapes.py` del mismo árbol: US, tres tramos de ajuste de 2019 a 2021 y uno de validación de 2022, con 128 activos por tramo y puntuaciones sintéticas que no proceden de ningún modelo. El generador escribe también la cinta de evaluación de 2023, que el perfil no usa, y ninguna llega al test de 2024. `mars-titan-policy-benchmark`, compilado con `--pgo generate`, recorrió las tres cintas de ajuste y la de validación en CPU y en `cuda:0` con 512 pasos. Los dos recibos declaran `none_no_optimizer_steps`.

| Salidas enlazadas (bytes) | Base | LTO | PGO | LTO con PGO |
| --- | ---: | ---: | ---: | ---: |
| Las 45 | 33.996.384 | 20.266.928 (−40,4 %) | 33.505.272 (−1,4 %) | 20.149.072 (−40,7 %) |
| `mars-titan-ppo` | 2.320.224 | 2.092.416 | 2.341.856 | 2.117.896 |
| `mars-titan-klpo` | 2.661.688 | 1.743.840 | 2.701.256 | 1.764.536 |
| `mars-titan-policy-benchmark` | 2.029.248 | 1.595.768 | 1.941.512 | 1.536.208 |
| 35 ejecutables de pruebas | 25.533.504 | 13.626.088 | 25.069.408 | 13.493.240 |
| Las otras 10 salidas | 8.462.880 | 6.640.840 | 8.435.864 | 6.655.832 |

LTO reduce las salidas un 40 %, sobre todo los ejecutables de pruebas, que enlazan mucho código que no usan. En `mars-titan-ppo` la reducción es del 10 %. PGO apenas cambia el tamaño. Las compilaciones tardaron 1.190 s la base (carga media de 14 y 75 °C), 282 s LTO (carga de 8 y 46 °C), 553 s PGO y 480 s LTO con PGO (carga de 7 a 13 y 44 °C). Con cargas tan distintas, esos tiempos no permiten comparar el coste de compilar cada variante. El pico de memoria de los procesos hijos fue de unos 687 MiB en las cuatro. Todos los objetos de LTO y 73 de los 77 de PGO difieren de la base, como se espera de esas opciones.

La decisión del 10 de octubre no cambia. El tamaño no limita nada en este equipo, y ninguna variante mejoró el recorrido de forma reproducible en las dos tandas de entonces. Release mantiene `MARS_TITAN_ENABLE_IPO` y `MARS_TITAN_PGO` desactivadas. Con el perfil nuevo no se han repetido los tiempos del recorrido.

## Caché de los presets de desarrollo

`native-ppo-dev` y `native-candidate-cuda-dev` fijan ahora la caché de ccache en `~/.cache/mars-titan/ccache`, con sus temporales dentro y un límite de 4 GiB (`CCACHE_DIR`, `CCACHE_TEMPDIR` y `CCACHE_MAXSIZE`), y `CCACHE_BASEDIR` en la raíz del worktree, para que las rutas dentro de él no entren en la clave. Así no ocupa `/tmp`, que es memoria en este equipo, ni entra en el repositorio.

## Reproducción

Cada compilación usa [`benchmarks/native_build_variants.py`](../../../benchmarks/native_build_variants.py) del árbol que se compila, desde su raíz, con `CCACHE_BASEDIR` en esa raíz, `CCACHE_DIR` y `CCACHE_TEMPDIR` fuera de `/tmp` y `MARS_TITAN_TORCH_ENVIRONMENT` apuntando al entorno con PyTorch. Una pareja de worktrees con la caché vacía al empezar:

```bash
python benchmarks/native_build_variants.py build --preset native-ppo-release \
  --build-dir build/native/ppo-ccache --label primero --launcher ccache \
  --define MARS_TITAN_TORCH_ENVIRONMENT=ENTORNO --output primero.json
# Desde la raíz del segundo worktree, con su propio guion y la misma CCACHE_DIR:
python benchmarks/native_build_variants.py build --preset native-ppo-release \
  --build-dir build/native/ppo-ccache --label segundo --launcher ccache \
  --define MARS_TITAN_TORCH_ENVIRONMENT=ENTORNO --output segundo.json
python benchmarks/native_build_variants.py compare \
  --first primero.json=PRIMERO/build/native/ppo-ccache \
  --second segundo.json=SEGUNDO/build/native/ppo-ccache --output comparacion.json
```

La edición añade una línea de comentario al final de `native/src/ppo_policy.cpp` y repite `build` con `--keep` sobre la misma carpeta. La vuelta atrás restaura el archivo con `git checkout` y repite `build --keep`. La cabecera tocada de la primera tanda usa `touch --build-dir CARPETA --header native/include/mars_titan/financial_batch.hpp`. Las variantes de #486 añaden `--ipo`, `--pgo generate --pgo-data DIRECTORIO` o `--pgo use --pgo-data PERFIL.profdata` a `build`. El perfil de PGO sale de `mars-titan-policy-benchmark` compilado con `--pgo generate`, en CPU y en `cuda:0`, sobre las cuatro cintas de ajuste y validación de la última ventana de política de US que escribe [`benchmarks/rl_stage_tapes.py`](../../../benchmarks/rl_stage_tapes.py), y se une con `llvm-profdata-21 merge`. Ningún paso aplica pasos de optimizador.
