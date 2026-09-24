# Núcleo nativo de simulación

`mars_titan_simulation` es una biblioteca compartida con una interfaz C. `mars_titan_financial` contiene la sesión financiera pura y `mars-titan-sim` permite ejecutarla desde un programa C++ autónomo. CMake mantiene C17 para los consumidores de la interfaz C y C++20 para la implementación. Los perfiles usan Ninja y dos trabajos de compilación. CUDA continúa desactivada por defecto.

La configuración requiere CMake 3.24 o posterior, Ninja y compiladores de C y C++. Desde `native/`:

```bash
cmake --preset native-debug
cmake --build --preset native-debug
ctest --preset native-debug
```

Los binarios y `compile_commands.json` se guardan en `build/native/`, fuera de este directorio. En Linux, la biblioteca de Release se genera en `build/native/native-release/libmars_titan_simulation.so`, respecto a la raíz del repositorio.

```bash
cmake --preset native-release -DMARS_TITAN_BUILD_RUNNER=ON
cmake --build --preset native-release
ctest --preset native-release
../build/native/native-release/mars-titan-sim --help
```

Los perfiles habituales de Clang y GCC activan `MARS_TITAN_BUILD_RUNNER` para construir la sesión y el ejecutable. `MARS_TITAN_BUILD_RUNNER=OFF` con `MARS_TITAN_BUILD_FINANCIAL=ON` permite compilar solo la sesión y sus pruebas. El ejecutable requiere UNIX por sus bloqueos y escrituras confirmadas. Las bibliotecas pueden configurarse por separado en MSVC. Una fuente requerida que falta produce un error de configuración.

El programa utiliza Arrow y Parquet C++, OpenSSL Crypto y [nlohmann_json 3.12.0](https://github.com/nlohmann/json/releases/tag/v3.12.0). La descarga de JSON se verifica con el SHA-256 publicado. CMake busca primero los paquetes del SDK Arrow/Parquet. En Linux puede localizar las cabeceras y bibliotecas C++ de PyArrow mediante `uv` durante la configuración. El ejecutable enlaza `libarrow` y `libparquet`, sin `arrow_python` ni `libpython`. JSON y las dependencias de archivos quedan fuera de la sesión pura.

La identidad compilada separa la huella de fuentes y cabeceras de la huella de compilación. Esta última incorpora compiladores y versiones, sistema y arquitectura, C++20, configuración, flags, opciones de los objetivos, endurecimiento STL, sanitizadores, cobertura, LTO y PGO. Cuando se usa PGO, incluye también el contenido de sus perfiles. Debug, Release y GCC tienen identidades distintas. `native-build-identity-Release.txt`, o el archivo equivalente de cada configuración, permite consultar los valores utilizados. Es una identidad de configuración y fuentes, no un hash del binario.

La huella de fuentes incluye `accurate_sum.hpp` y los módulos de CMake que preparan la compilación. Los cambios en esos archivos hacen que CMake vuelva a calcular ambas huellas antes de compilar. El ejecutable registra también el identificador y la versión del compilador y el tipo de build.

## Perfiles y diagnósticos

| Perfil | Finalidad |
| --- | --- |
| `native-debug` | Clang, símbolos, avisos como errores y análisis de lifetime disponible. |
| `native-release` | Release con paridad de coma flotante y endurecimiento STL. |
| `native-asan-ubsan` | AddressSanitizer, UndefinedBehaviorSanitizer y detección de fugas cuando la plataforma la admite. |
| `native-tsan` | ThreadSanitizer en una compilación independiente. |
| `native-msan` | MemorySanitizer con libc++ y libc++abi instrumentadas. |
| `native-static-analysis` | clang-tidy y analizador de rutas de Clang. |
| `native-fuzz` | libFuzzer con ASan y UBSan, entradas y tiempo acotados. |
| `native-coverage` | Cobertura LLVM de líneas y ramas, separada de Release y los sanitizadores. |
| `native-gcc-debug`, `native-gcc-release` | Compilación alternativa con GCC. |
| `native-gcc-analysis` | Analizador de rutas de GCC mediante `-fanalyzer`. |
| `native-profile` | RelWithDebInfo con símbolos y punteros de pila. |

`native-asan` conserva el nombre del perfil anterior y equivale a `native-asan-ubsan`. Los perfiles `native-msvc-debug`, `native-msvc-release`, `native-msvc-asan` y `native-msvc-analysis` están disponibles en Windows, desde un entorno de desarrollo de MSVC con Ninja. Incluyen las opciones compatibles `/W4`, `/permissive-`, `/sdl`, `/fp:strict` y `/analyze` cuando se solicita. Estos perfiles de Windows no se han probado en este equipo.

CMake comprueba el soporte de los avisos antes de activarlos. Incluye conversiones, cambios de signo, ocultación de nombres, formatos, desreferencias nulas y otros diagnósticos de C++. En Clang intenta primero Lifetime Safety y después la combinación experimental `-Xclang -fexperimental-lifetime-safety -Wexperimental-lifetime-safety`. GCC utiliza los avisos de referencias y punteros colgantes que admita.

libstdc++ utiliza `_GLIBCXX_ASSERTIONS`. Cuando se detecta libc++ con modos de endurecimiento, Release utiliza el modo rápido y los perfiles de verificación el extensivo. Las opciones se aplican a los objetivos propios mediante `mars_titan_configure_target`.

## Análisis estático

El perfil de análisis exige una instalación ejecutable de clang-tidy. Con Clang 21, la herramienta utilizada se puede instalar en el entorno del usuario:

```bash
uv tool install clang-tidy==21.1.6
cmake --preset native-static-analysis
cmake --build --preset native-static-analysis
ctest --preset native-static-analysis
```

También puede indicarse su ruta mediante la variable CMake `MARS_TITAN_CLANG_TIDY`. Se comprueba que clang-tidy comparte la versión principal del compilador Clang. `.clang-tidy` selecciona análisis de rutas, errores, rendimiento, portabilidad y comprobaciones de las C++ Core Guidelines compatibles con el estándar del proyecto.

El objetivo `static-analysis` utiliza `clang-check` y la base de comandos de compilación para analizar las fuentes propias activadas, incluidas la sesión y el ejecutable. Devuelve un error cuando detecta un problema. En GCC y MSVC, el análisis forma parte de la compilación del perfil correspondiente. Una herramienta solicitada que falta o no acepta las opciones produce un error de configuración.

## Sanitizadores y pruebas

```bash
cmake --preset native-asan-ubsan -DMARS_TITAN_BUILD_RUNNER=ON
cmake --build --preset native-asan-ubsan
ctest --preset native-asan-ubsan
```

Los tests `simulation`, `concurrency` y `c_abi` comprueban la contabilidad, las llamadas concurrentes y un consumidor C17 real. La sesión añade `financial_session`. El perfil de fuzzing añade `fuzz_smoke` y, cuando la sesión está activada, `fuzz_session_smoke`. Si también está activado el ejecutable, `fuzz_serialization_smoke` comprueba la lectura JSON acotada y la conservación de snapshots al serializarlos y recuperarlos. Su semilla se copia a `fuzz-json-corpus` en el directorio de build y las entradas generadas se guardan allí. Cada ejecución utiliza 1000 entradas, semilla 42, un máximo de 4096 bytes por entrada y un límite de 2048 MiB. Cada test tiene un tiempo máximo.

Los perfiles instrumentados conservan símbolos y punteros de pila. CTest solicita trazas simbolizadas y utiliza `llvm-symbolizer` cuando está disponible. Las búsquedas de símbolos por red quedan desactivadas. ASan y UBSan se combinan entre sí. TSan y MSan requieren sus propios directorios y runtimes.

MSan no se configura con una biblioteca estándar sin instrumentar. `MARS_TITAN_MSAN_STDLIB_ROOT` debe señalar una instalación de libc++ y libc++abi compiladas con MSan, con cabeceras en `include/c++/v1` y bibliotecas en `lib`. CMake comprueba esas rutas y las referencias a la instrumentación. La presencia del runtime de Clang por sí sola no basta. En ausencia de estas dependencias, el perfil termina con un error explícito.

La [preparación de LLVM para MSan](cmake/msan-runtime.md) fija la versión, el commit y las opciones de la instalación local.

```bash
cmake --preset native-msan -DMARS_TITAN_BUILD_FINANCIAL=ON \
  -DMARS_TITAN_MSAN_STDLIB_ROOT="${MSAN_STDLIB_ROOT:?Define la instalación instrumentada}"
```

El núcleo y la sesión pura se comprueban con MSan. Ese perfil excluye explícitamente `mars-titan-sim` porque Arrow y OpenSSL son bibliotecas precompiladas sin instrumentación MSan. El ejecutable sigue disponible para ASan, UBSan, TSan y análisis del código propio.

La ejecución de los sanitizadores depende también del sistema operativo y de su espacio de direcciones. Un fallo de inicialización del runtime no equivale a una prueba superada. No se cambian permisos, ASLR ni opciones del sistema para ocultarlo. MSVC dispone de ASan en un perfil separado de las comprobaciones `/RTC` de Debug.

Las pruebas locales de `tests/native/test_diagnostics.py` usan programas temporales para comprobar avisos, accesos inválidos y herramientas ausentes. Se ejecutan desde la raíz:

```bash
uv run --locked pytest tests/native/test_diagnostics.py
```

## Release y medición

Release utiliza `-fno-fast-math` y `-ffp-contract=off` en GCC y Clang. MSVC utiliza `/fp:strict` cuando lo admite. LTO y PGO están desactivados. La instrumentación de sanitizadores, el perfil de rendimiento y CUDA se configuran por separado.

`MARS_TITAN_ENABLE_IPO=ON` solicita LTO para Release y comprueba su disponibilidad. `MARS_TITAN_PGO=generate` y `MARS_TITAN_PGO=use` son opciones explícitas. GCC utiliza un directorio de perfiles. Clang genera archivos `.profraw` y consume un archivo `.profdata` combinado con `llvm-profdata`. La ruta se indica mediante `MARS_TITAN_PGO_DATA`. Estos modos necesitan una carga representativa y una comparación de paridad y tiempo total antes de adoptar sus resultados.

La medición del núcleo debe incluir la preparación de entradas y el enlace con Python, además del tiempo interno. Deben registrarse las versiones, la configuración, las formas de los datos, las repeticiones, la memoria y el error frente a la referencia. Los tiempos de los perfiles instrumentados describen esas comprobaciones, no el rendimiento de Release.

## Cobertura de líneas y ramas

```bash
cmake --preset native-coverage -DMARS_TITAN_BUILD_RUNNER=ON
cmake --build --preset native-coverage
ctest --preset native-coverage
cmake --build --preset native-coverage --target coverage-report
```

El perfil utiliza contadores atómicos para las pruebas concurrentes. `llvm-profdata` combina los perfiles y `llvm-cov` genera un resumen y `build/native/native-coverage/coverage/native-coverage.json`, respecto a la raíz del repositorio. El informe excluye pruebas, cabeceras del sistema y dependencias externas. Los nombres de fuentes del proyecto quedan relativos al repositorio. El informe rechaza perfiles anteriores a los binarios para evitar mezclar ejecuciones de versiones distintas.

Las pruebas externas del ejecutable pueden escribir en el mismo directorio mediante `LLVM_PROFILE_FILE`. La cobertura registra líneas y ramas observadas. Las pruebas de comportamiento y de mutación deben comprobar qué errores detectan esos recorridos.

Referencias: [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html), [UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html), [MemorySanitizer](https://clang.llvm.org/docs/MemorySanitizer.html), [ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) y [clang-tidy](https://clang.llvm.org/extra/clang-tidy/).
