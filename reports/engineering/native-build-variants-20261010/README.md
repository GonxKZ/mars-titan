# Variantes de compilación nativa (10 de octubre de 2026)

Este informe responde a #486 y #487. #486 pide medir LTO y PGO en la etapa RL nativa con un perfil representativo y adoptarlos en el preset solo si ganan de forma reproducible con los mismos bits. #487 pide medir ccache y mold sobre la compilación real de `native-ppo-release` y `native-candidate-cuda` y usarlos solo en desarrollo, sin cambiar la identidad de Release.

Ninguna medida entrena. `mars-titan-policy-benchmark` recorre la recogida PPO, la inferencia, las ventajas GAE, una oleada KLPO y el forward y backward de los objetivos sobre cintas reales reconstruidas, sin ningún paso de Adam. Al terminar exige que los parámetros conserven su huella y que el contador de pasos de optimizador siga a cero. No se ejecutaron las pruebas de CTest que aplican Adam (`ppo_policy`, `ppo_training` y `ppo_gru_packing`).

## Entorno

| Elemento | Valor |
|---|---|
| CPU | AMD Ryzen 9 8945HS, 8 núcleos y 16 hilos |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8.188 MiB, controlador 595.91.07 |
| Memoria | 32 GB, compartida con suites y compilaciones de otras ramas |
| Herramientas | Clang 21.1.8, CMake 4.2.3, Ninja 1.13.2, GNU ld 2.46 (enlazador por defecto de Clang aquí), mold 2.40.4, ccache 4.12.3, llvm-profdata 21.1.8 |
| Energía | Perfil `power-saver` sin cambios. No se tocaron relojes, ventiladores ni límites |

La carga media de un minuto estuvo entre 6 y 26 durante las compilaciones y entre 11 y 24 durante los recorridos. El guardián térmico congeló el cálculo una vez, de 14:01:08 a 14:01:28, en mitad de la primera compilación base de `native-ppo-release`. Su tiempo (270,9 s) se descarta y se usa la repetición `ppo-base-r2` (192,1 s). Ninguna otra medida coincidió con una congelación.

## Método

`benchmarks/native_build_variants.py` tiene cinco órdenes con salida JSON:

- `build` configura y compila un preset en su propio directorio con `-j 2` y registra tiempos, estadísticas de ccache, la identidad compilada y la huella SHA-256 de cada objeto y de cada salida enlazada.
- `touch` cambia solo la fecha de una cabecera muy incluida (`financial_batch.hpp`) y recompila, como al volver a una rama.
- `relink` borra las salidas enlazadas de un directorio al día y mide solo el enlazado.
- `compare` clasifica cada archivo que difiere entre dos compilaciones: `identity_only` si coincide al sustituir por ceros la huella de identidad de cada lado, `identity_and_link_metadata` si además solo cambian secciones del enlace dinámico (RUNPATH con otro directorio o build-id) y `content` en cualquier otro caso. Los objetos con LTO son bitcode de LLVM, sin tabla de secciones, y cuentan como `content`.
- `run` ejecuta `mars-titan-policy-benchmark` de cada variante sobre las mismas cintas, con 512 pasos y 32 de calentamiento, en rondas cuyo orden rota, y falla si alguna variante no da las mismas huellas de contenido (recogidas, entorno, muestreador y registros) que las demás.

Las fuentes medidas son las de develop en d355df5f. Hasta ee8da126 solo cambia `native/tests/rl_variety_reference.py`, que no se compila. `native-ppo-release` compila 68 objetos y 39 salidas enlazadas. `native-candidate-cuda` se compiló con la extensión episódica de Python y la simulación (36 objetos y 19 salidas), igual que la usa la suite.

## ccache y mold (#487)

| Compilación | `native-ppo-release` | `native-candidate-cuda` |
|---|---:|---:|
| Base, sin ccache | 192,1 s (carga 6) | 117,1 s (carga 14 a 20) |
| ccache en frío | 245,6 s (carga 10 a 16), 68 fallos | 132,8 s (carga 13 a 15), 36 fallos |
| ccache en caliente, directorio nuevo | 9,1 s, 68 aciertos | 3,2 s, 36 aciertos |
| Cabecera tocada, base | 119,0 s, 63 pasos | Sin medir |
| Cabecera tocada, ccache | 8,7 s, 26 aciertos | Sin medir |
| mold sin ccache | 316,9 s (carga 16 a 19) | 138,8 s (carga 19 a 20) |
| ccache y mold, otro directorio | 47,0 s, 62 aciertos y 6 fallos | 11,5 s, 33 aciertos y 3 fallos |

| Reenlazado completo | GNU ld | mold |
|---|---:|---:|
| `native-ppo-release`, 39 salidas | 6,32 a 9,36 s, mediana 7,28 (5 medidas) | 2,33 a 3,24 s, mediana 2,56 (4 medidas) |
| `native-candidate-cuda`, 19 salidas | 2,07 a 3,08 s, mediana 2,64 (5 medidas) | 0,61 a 1,25 s, mediana 0,83 (4 medidas) |

Las compilaciones completas no están emparejadas y su carga cambia mucho, así que solo se leen las diferencias grandes. ccache en frío no acelera y en el candidato, con cargas parecidas, tardó un 13 % más. La ganancia está en volver a compilar fuentes con el mismo contenido, que es lo que pasa entre worktrees y ramas: de 192 s a 9 s en PPO y de 117 s a 3 s en el candidato. Una cabecera editada de verdad no tendría aciertos. mold reduce el enlazado a un tercio, que es lo que cuenta en cada iteración de desarrollo.

En cuanto a los bits, con ccache los objetos coinciden con base salvo la huella de identidad, que cambia porque `BuildIdentity.cmake` registra el lanzador (62 iguales y 6 `identity_only` en PPO, 33 y 3 en el candidato). Las salidas enlazadas solo difieren en la identidad, la RUNPATH y el build-id. Con mold los objetos son los mismos, pero las 39 y 19 salidas enlazadas cambian de contenido porque mold dispone el ejecutable de otra forma. Por eso Release sigue con el enlazador y la identidad de siempre.

La decisión es añadir `native-ppo-dev` y `native-candidate-cuda-dev`, que heredan de los presets de Release y solo añaden `CMAKE_C_COMPILER_LAUNCHER=ccache`, `CMAKE_CXX_COMPILER_LAUNCHER=ccache` y `CMAKE_LINKER_TYPE=MOLD`. Los dos se compilaron con sus presets y sus objetos coinciden con los de las variantes medidas con mold salvo la identidad (62 y 6 en PPO, 33 y 3 en el candidato). Las salidas enlazadas que llevan RUNPATH (33 y 10) difieren por disposición, porque la RUNPATH incluye el directorio de compilación y su longitud cambia con el nombre del preset. Su identidad compilada registra `ccache` y `MOLD`, así que se distingue de la de Release. No deben usarse en la campaña, aunque todavía ninguna comprobación lo impide.

## LTO y PGO (#486)

El perfil de PGO sale de `mars-titan-policy-benchmark` compilado con `MARS_TITAN_PGO=generate` y recorrido sobre las cintas reales de US (tres tramos de ajuste, de `fold-014` a `fold-016`, y la validación de `fold-017`) en CPU y en `cuda:0`. `llvm-profdata` unió los cuatro `.profraw` en un perfil de 777 funciones (SHA-256 `cb65ee17…d023`). Las cintas de CN quedan fuera del perfil y sirven para ver si la ganancia se generaliza.

Con `-Werror`, Clang rechaza PGO por dos avisos: `-Wprofile-instr-unprofiled` en las unidades que el recorrido no ejecuta y `-Wprofile-instr-out-of-date` en los `main` de otros ejecutables, que comparten nombre con el del recorrido. Para medir se desactivaron solo esos dos avisos en las variantes con PGO. Adoptarlo exigiría tratarlos en `Instrumentation.cmake` y mantener un perfil versionado que queda desfasado con cada cambio de fuentes.

Todas las variantes dieron las mismas huellas de contenido que base en todas las rondas, escenarios y tandas. Los objetos y ejecutables de LTO y PGO sí cambian, como corresponde a otro código generado.

La regla de adopción se fijó tras la primera tanda y antes de ver la segunda: una variante gana si, en las dos tandas y en los cuatro escenarios, la mediana de su razón de tiempo total frente a base de la misma ronda queda por debajo de 1 y por debajo de la del control mold, que no cambia el código.

| Tanda y escenario | Rondas | LTO | PGO | LTO y PGO | mold (control) |
|---|---:|---:|---:|---:|---:|
| 1, US en CPU | 3 | 1,008 | 1,071 | 1,008 | 1,020 |
| 1, US en `cuda:0` | 3 | 0,939 | 0,925 | 0,910 | 0,999 |
| 1, CN en CPU | 3 | 0,946 | 0,866 | 0,912 | 0,972 |
| 1, CN en `cuda:0` | 3 | 0,988 | 1,015 | 1,010 | 0,997 |
| 2, US en CPU | 6 | 1,023 | 0,962 | 0,967 | 1,011 |
| 2, US en `cuda:0` | 6 | 0,960 | 0,980 | 0,985 | 1,031 |
| 2, CN en CPU | 6 | 0,987 | 0,995 | 1,004 | 1,011 |
| 2, CN en `cuda:0` | 6 | 0,996 | 0,990 | 1,015 | 0,993 |

PGO gana en 6 de 8 combinaciones, LTO en 5 y LTO con PGO en 4. Ninguna cumple la regla. Una de las derrotas de PGO es US en CPU de la primera tanda, que forma parte de su propio perfil. Las razones de una misma variante van de 0,75 a 1,24 entre rondas, y el control mold, que solo cambia la disposición, de 0,88 a 1,20. Con la máquina compartida, una diferencia de un 2 % a un 5 % en la mediana no se distingue del ruido.

Las compilaciones con LTO y PGO tardaron entre 324 y 337 s con cargas de 14 a 24, frente a los 192 s de base con carga 6. No se puede atribuir esa diferencia a la variante.

La decisión es no adoptar LTO ni PGO. Release los mantiene desactivados y las opciones `MARS_TITAN_ENABLE_IPO` y `MARS_TITAN_PGO` siguen disponibles para repetir la medida con la máquina libre.

## Reproducción

Las rutas locales se sustituyen por marcadores. `<cintas>` son las cintas reconstruidas de la etapa RL, `<entorno>` el entorno de Python con LibTorch y `<salida>` el destino.

```bash
B="python benchmarks/native_build_variants.py"
memslot suite -- $B build --preset native-ppo-release --build-dir build/native/ppo-base \
  --label ppo-base --define MARS_TITAN_TORCH_ENVIRONMENT=<entorno> --output <salida>/ppo-base.json
memslot suite -- $B build --preset native-ppo-release --build-dir build/native/ppo-mold \
  --label ppo-mold --linker MOLD --define MARS_TITAN_TORCH_ENVIRONMENT=<entorno> \
  --output <salida>/ppo-mold.json
memslot suite -- $B build --preset native-ppo-release --build-dir build/native/ppo-pgo \
  --label ppo-pgo --pgo use --pgo-data <salida>/policy.profdata \
  --define MARS_TITAN_TORCH_ENVIRONMENT=<entorno> \
  --define "CMAKE_C_FLAGS=-Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date" \
  --define "CMAKE_CXX_FLAGS=-Wno-profile-instr-unprofiled -Wno-profile-instr-out-of-date" \
  --output <salida>/ppo-pgo.json
$B compare --first <salida>/ppo-base.json=build/native/ppo-base \
  --second <salida>/ppo-pgo.json=build/native/ppo-pgo --output <salida>/compare.json
memslot light -- $B relink --build-dir build/native/ppo-mold --label ppo-mold-relink \
  --output <salida>/ppo-mold-relink.json
memslot gpu --max 8G -- $B run --variant base=build/native/ppo-base \
  --variant pgo=build/native/ppo-pgo --variant mold=build/native/ppo-mold \
  --tapes <cintas> --market US --device cuda:0 --rounds 6 --output <salida>/US-cuda.json
python reports/engineering/native-build-variants-20261010/runtime_ratios.py <razones>.json \
  t1=<salida>/tanda1/ t2=<salida>/tanda2/
```

El perfil se genera con `--pgo generate --pgo-data <perfiles>`, recorriendo después `mars-titan-policy-benchmark` de esa compilación sobre `<cintas>` en CPU y en `cuda:0` y uniendo los archivos con `llvm-profdata merge`.

## Archivos

| Archivo | Contenido |
|---|---|
| `runtime_ratios.py` | Razones emparejadas por ronda, control mold y regla de adopción |
| `evidence/builds.json` | Recibos de compilación, cabecera tocada y reenlazado, con identidad, entorno y carga |
| `evidence/compare-*.json` | Clasificación de objetos y salidas enlazadas entre dos compilaciones |
| `evidence/runtime-t1-*.json` y `runtime-t2-*.json` | Las dos tandas de recorridos por mercado y dispositivo |
| `evidence/runtime-ratios.json` | Razones, medianas y veredicto de cada variante |

Los recibos guardan el número de huellas de cada compilación y las comparaciones guardan cada archivo distinto. Los directorios de compilación, el perfil y la caché de ccache no se versionan.

## Límites

- La máquina estaba compartida. Las compilaciones completas no están emparejadas y los recorridos solo se comparan dentro de la misma ronda.
- El perfil representa la etapa RL sin actualizaciones. Las rondas con Adam no se pueden medir mientras siga el bloqueo de aprendizaje.
- La cabecera tocada mide un cambio de fecha sin cambio de contenido. Una edición real recompila las unidades afectadas sin aciertos de ccache.
