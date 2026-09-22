# Configuración nativa de MARS-TITAN

Este directorio prepara CMake para C17 y C++20. CUDA es opcional y está desactivada por defecto. El objetivo de interfaz `mars_titan::native_options` agrupa los estándares y los avisos de compilación para futuros objetivos. No genera una biblioteca, un ejecutable ni una extensión de Python.

La configuración requiere CMake 3.24 o posterior y compiladores de C y C++. Desde la raíz del repositorio:

```bash
cmake -S native -B build/native-cpu -DMARS_TITAN_ENABLE_CUDA=OFF
```

Para comprobar las herramientas de CUDA en el equipo documentado:

```bash
cmake -S native -B build/native-cuda \
  -DMARS_TITAN_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.4 \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

`89` corresponde a la capacidad de cómputo 8.9 observada en la RTX 4070 Laptop. En otro equipo se debe comprobar la arquitectura y ajustar las rutas. Al activar CUDA, CMake exige tanto el compilador como el conjunto de herramientas. Una ausencia o incompatibilidad produce un error visible, sin cambiar automáticamente a una compilación con CPU.

Conviene usar directorios de configuración distintos para CPU y CUDA. CMake conserva la elección de compilador en su caché. Estos directorios contienen archivos generados y quedan fuera del código fuente.

## Alcance de la comprobación

Configurar con éxito verifica que CMake reconoce las herramientas y puede realizar sus comprobaciones de compilador. No verifica kernels, resultados numéricos, gradientes, rendimiento o compatibilidad de una futura extensión con PyTorch. No hay fuentes científicas ni binarios del proyecto que compilar o ejecutar.

La estructura reserva [include/](include/README.md) para contratos públicos, [src/](src/README.md) para implementaciones C/C++ y [cuda/](cuda/README.md) para posibles kernels. Su contenido se decidirá después de perfilar una implementación de referencia en Python. No se añaden opciones de cálculo aproximado ni opciones de optimización numérica antes de medir su efecto.

El conjunto de herramientas local y el entorno de ejecución usado por una distribución de PyTorch pueden tener versiones distintas. Antes de compilar una extensión habrá que comprobar la compatibilidad de PyTorch, el conjunto de herramientas, el compilador anfitrión y la ABI. Los detalles del entorno están en [reproducibility.md](../docs/engineering/reproducibility.md).

## Diagnóstico por objetivo

Los objetivos C++ propios llaman a `mars_titan_configure_target(nombre)`. La función aplica C++20, los avisos del proyecto y la base de comandos de compilación. Las opciones de análisis se aplican a ese objetivo y no reescriben las opciones de compilación de bibliotecas de terceros. C17 se mantiene para fuentes C que se incorporen de forma explícita.

C++20 es la base común con CUDA. Una necesidad concreta de C++23 debe justificar el cambio de estándar y comprobar el soporte del compilador anfitrión, el conjunto de herramientas y la extensión de PyTorch. No se adopta un estándar superior solo por su fecha.

```cmake
add_library(operacion src/operacion.cpp)
mars_titan_configure_target(operacion)
```

El ejemplo describe la integración futura. No existe todavía esa biblioteca.

Desde `native/` se pueden configurar perfiles separados:

```bash
cmake --preset native-debug
cmake --preset native-asan
cmake --preset native-profile
```

| Opción | Comprobación o finalidad |
| --- | --- |
| `MARS_TITAN_WARNINGS_AS_ERRORS=ON` | Convierte avisos propios en errores. Incluye conversiones y ocultación de nombres en C++. |
| `MARS_TITAN_ENABLE_CLANG_TIDY=ON` | Ejecuta análisis estático, comprobaciones de errores, rendimiento y portabilidad. |
| `MARS_TITAN_SANITIZER=address-undefined` | Instrumenta memoria y comportamiento indefinido en CPU. |
| `MARS_TITAN_SANITIZER=thread` | Configuración alternativa para comprobar concurrencia CPU. |
| `MARS_TITAN_PROFILE=ON` | Conserva punteros de pila en un perfil de medición sin sanitizadores. |

Los sanitizadores de CPU y los perfiles de rendimiento no se combinan. Tampoco se activan automáticamente sobre CUDA. El coste de instrumentación no se publica como rendimiento de una versión optimizada.

clang-tidy y clang-format deben corresponder a herramientas disponibles en el
entorno. Se puede pasar su ruta sin modificar la configuración del sistema:

```bash
cmake -S native -B build/native/tidy \
  -DMARS_TITAN_ENABLE_CLANG_TIDY=ON \
  -DMARS_TITAN_CLANG_TIDY=/ruta/a/clang-tidy
clang-format --dry-run --Werror --style=file:native/.clang-format archivo.cpp
```

Una herramienta solicitada que no existe produce un error, no un análisis omitido silenciosamente. El archivo `.clang-tidy` selecciona comprobaciones concretas y `.clang-format` fija un formato común. No sustituyen las pruebas de comportamiento.

## Medición y depuración

Primero se registra una carga representativa, sus entradas, la precisión, la versión y la salida de referencia. El informe separa lectura, conversiones, transferencia, cálculo, sincronización y escritura. Debe medir latencia, caudal, RAM y VRAM máximas, además del tiempo completo. La mejora teórica máxima depende de la fracción del recorrido que realmente se acelere.

El paralelismo también forma parte del experimento. Se comparan cantidades
acotadas de trabajadores, hilos de BLAS y compilación, tamaño de lote y precarga.
No se multiplican estos niveles sin medir la sobresuscripción, el ancho de banda
y las copias. Los perfiles de compilación usan dos trabajos como punto de partida,
no como máximo demostrado del equipo. La carga científica conservará colas y
cachés limitadas, cancelación y errores visibles.

Para CPU se emplean símbolos de depuración, `perf` cuando el entorno lo permita y
Valgrind para perfiles o errores de memoria. No se cambian permisos del kernel ni
parámetros de seguridad para habilitar un contador. Para CUDA, las herramientas
previstas son Nsight Systems para el recorrido, Nsight Compute para kernels y
Compute Sanitizer para memoria y sincronización. Solo se usarán cuando exista un
ejecutable concreto y una carga autorizada.

```bash
perf stat -- ejecutable argumentos
valgrind --tool=memcheck --error-exitcode=1 ejecutable argumentos
nsys profile --trace=cuda,nvtx,osrt -o perfil ejecutable argumentos
ncu --set basic ejecutable argumentos
compute-sanitizer --tool memcheck --error-exitcode=1 ejecutable argumentos
compute-sanitizer --tool racecheck --error-exitcode=1 ejecutable argumentos
compute-sanitizer --tool synccheck --error-exitcode=1 ejecutable argumentos
```

Son órdenes para futuros ejecutables, no experimentos ejecutados. No se activa
`fast-math`, precisión reducida, copias asíncronas o kernels propios sin contrastar
sus errores y el beneficio completo frente a las bibliotecas existentes.

## Comprobación de las herramientas

Las [pruebas locales](../tests/native/test_diagnostics.py) crean programas C++
temporales. Verifican compilación correcta, avisos como errores, un acceso fuera
de límites detectado por AddressSanitizer y una desreferencia nula detectada por
clang-tidy. También prueban configuraciones incompatibles y herramientas ausentes.

```bash
uv run --locked pytest tests/native/test_diagnostics.py
```

La prueba de clang-tidy acepta su ruta mediante `MARS_TITAN_CLANG_TIDY`. Si falta,
se informa como omitida. En la prueba del fallo de memoria se desactiva únicamente
la simbolización externa para evitar búsquedas de símbolos por red. Se exige la
salida de error real del sanitizador, no basta con que el proceso termine por señal.

Referencias: [AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html),
[UndefinedBehaviorSanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html),
[clang-tidy](https://clang.llvm.org/extra/clang-tidy/) y
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
