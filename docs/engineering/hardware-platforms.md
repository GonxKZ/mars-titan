# Plataformas de hardware: RTX 4070 y DGX GB10

Autor: Gonzalo García Lama. Estado del 10 de octubre de 2026 ([#500](https://github.com/GonxKZ/mars-titan/issues/500)).

MARS-TITAN debe funcionar en el portátil de desarrollo (x86-64 con una RTX 4070 Laptop de 8 GiB) y en una DGX Spark con GB10 (aarch64, GPU Blackwell de capacidad 12.1 y 128 GB de memoria unificada). Este documento recoge qué se ha preparado para la segunda máquina, qué se ha comprobado sin ella y qué queda pendiente de comprobar allí. Ninguna comprobación de esta página se ha ejecutado todavía en la GB10.

## Dependencias resueltas para las dos arquitecturas

`pyproject.toml` declara en `[tool.uv] environments` y `required-environments` Linux x86-64 y Linux aarch64. `uv.lock` resuelve así las ruedas CUDA 13.0 de PyTorch y torchvision, CuPy y XGBoost para ambas arquitecturas. Las demás plataformas quedan fuera del lock, así que salen de él los paquetes exclusivos de macOS y Windows y las variantes de PyTorch para CPU de esos sistemas. La resolución x86-64 no ha cambiado. Al evaluar los marcadores de `uv export --all-extras --all-groups` para Linux x86-64, las 183 versiones coinciden con las del lock anterior, y cada paquete del lock tiene una rueda aarch64 o una rueda pura. La instalación aarch64 se comprobó con `uv sync --dry-run --python-platform aarch64-manylinux_2_28`, sin instalar las ruedas aarch64 con sus bibliotecas NVIDIA.

Las ruedas aarch64 se inspeccionaron con `cuobjdump` para saber si traen código para la capacidad 12.1. Un cubin `sm_120` se ejecuta en una GPU 12.1 porque comparte la versión mayor, y el PTX se compila al cargar en cualquier capacidad igual o superior.

| Rueda aarch64 | Bibliotecas con código CUDA | Código para la GB10 |
| --- | --- | --- |
| torch 2.14.0+cu130 | 2 | cubin `sm_120` en las dos y `sm_121a` en una |
| torchvision 0.29.0+cu130 | 2 | cubin `sm_120` y `sm_121`, PTX `sm_120` y `sm_121` |
| cupy-cuda13x 14.2.0 | 3 | cubin `sm_120` y PTX `sm_120` en las tres |
| xgboost 3.3.0 | 1 | cubin `sm_120` y PTX `sm_120` |

Esta inspección muestra que el código existe. No demuestra que los núcleos se ejecuten bien en la GB10, cosa que comprueba la orden de la última sección.

## Compilación nativa cruzada

`native/cmake/toolchains/aarch64-linux-gnu-clang.cmake` compila para aarch64 con Clang y lld sobre las cabeceras, libstdc++ y crt de `g++-aarch64-linux-gnu`. Es el mismo compilador que los perfiles PPO nativos, así que la ABI coincide con la de LibTorch. No fija ninguna extensión de ISA y el código nativo no tiene despacho propio por ISA. Las únicas rutas vectorizadas son las de LibTorch, que elige su núcleo al cargar.

El preset `native-aarch64-release` compila PPO, KLPO, el ejecutable financiero y el enlace episódico con `-Werror`. Toma LibTorch y PyArrow de un entorno uv aarch64 y OpenSSL, zlib y zstd de una raíz aarch64 adicional:

```bash
export MARS_TITAN_AARCH64_ENVIRONMENT=<entorno uv aarch64 con PyTorch CPU>
export MARS_TITAN_AARCH64_SYSROOT=<raíz con OpenSSL, zlib y zstd aarch64>
cmake --preset native-aarch64-release
cmake --build --preset native-aarch64-release -j 2
ctest --preset native-aarch64-release
```

La configuración consulta el intérprete aarch64 bajo qemu-user, así que su límite de tiempo es `MARS_TITAN_PROBE_TIMEOUT`. Las consultas de LibTorch y PyArrow exigen que el paquete sea de la misma arquitectura que el destino, de modo que un entorno x86-64 se rechaza al configurar y no al enlazar. Bajo emulación, la capacidad CPU que informa PyTorch se registra como `emulated:...`, porque no es la del procesador real.

CTest ejecuta los binarios con qemu-user y multiplica cada límite por `MARS_TITAN_TEST_TIMEOUT_SCALE`. El preset excluye la etiqueta `optimizer-steps`, que llevan las pruebas que aplican pasos de optimizador sobre datos sintéticos. Una comprobación técnica en otra máquina no debe aprender.

`MARS_TITAN_ENABLE_CUDA` declara ahora `CMAKE_CUDA_ARCHITECTURES` como `89-real;121-real`, solo SASS para la RTX 4070 y la GB10. Una GPU distinta falla al cargar en lugar de compilar PTX en ejecución sin aviso. La identidad de compilación registra las arquitecturas, la compilación cruzada y las raíces de búsqueda.

## Comprobaciones emuladas en el portátil

Todo lo siguiente se ejecutó el 10 de octubre en el portátil, con Clang 21.1.8, lld, el runtime de `g++-aarch64-linux-gnu` 15.2 y qemu-user. Comprueba que el código compila y se comporta igual en aarch64 en CPU. No mide rendimiento, no ejercita CUDA y no usa el procesador Grace real.

| Comprobación | Resultado |
| --- | --- |
| Compilación `native-aarch64-release` con `-Werror` | 126 objetivos sin avisos. La primera compilación con `-j 2` tardó 311 s y usó 797 MB de memoria residente |
| CTest bajo qemu-user | 30 de 30 superadas en dos ejecuciones, de 145 y 223 s. Quedan fuera las 5 pruebas con la etiqueta `optimizer-steps` |
| Las mismas opciones en x86-64, con `native-release` y LibTorch en CPU | Sin avisos y las mismas 30 pruebas superadas en 24 s |
| `nvcc` 13.4 con `89-real;121-real` | `cuobjdump` muestra un cubin `sm_89` y otro `sm_121` |
| Pytest con el intérprete aarch64 | 169 superadas y 4 omitidas por el bloqueo de aprendizaje, en `tests/hardware`, seis módulos de contratos de datos de `tests/data` y el enlace episódico nativo aarch64 con `MARS_TITAN_REQUIRE_NATIVE=1` |

El compilador cruzado GCC 15 no sirve con `-Werror`. Rechaza `native/src/ppo_training.cpp` por una conversión de signo y da un aviso `-Wnull-dereference` dentro de las cabeceras del optimizador Adam de LibTorch, que parece un falso positivo. Por eso el archivo de toolchain usa Clang, igual que los perfiles PPO de x86-64.

PyArrow para aarch64 enlaza con libzstd, así que la raíz aarch64 tomó `libzstd1` 1.5.7 de los repositorios de Ubuntu para arm64. El paquete se comprobó con la firma de `InRelease` y el SHA-256 del índice antes de copiarlo.

CTest encontró una diferencia real en `markov_filter`. El control de cancelación usaba el épsilon de `long double`, que en x86-64 es el formato extendido de 64 bits de mantisa y en aarch64 es binary128. Con la misma entrada, aarch64 aceptaba una cancelación que x86-64 rechaza. El filtro usa ahora la cota 2^-63, o el épsilon propio si el `long double` de la plataforma es menos preciso. En x86-64 la regla no cambia, aarch64 aplica la misma y calcula con más precisión de la exigida. En aarch64 la aritmética binary128 se resuelve por software, así que su coste en la GB10 está pendiente de medir.

## Identidad de la plataforma

`mars_titan.hardware.platform_identity` registra sistema, arquitectura, procesador (`lscpu`), memoria, PyTorch, CUDA, cuDNN y cada GPU (`nvidia-smi`). Las GPU no salen de `torch.cuda` para no crear un contexto CUDA en el proceso de la campaña, que solo cuenta ese contexto tras su primer trabajo CUDA. Bajo qemu-user el intérprete dice aarch64 y `lscpu`, que es un programa del anfitrión, dice x86-64. La plataforma queda entonces marcada como emulada.

La huella `platform_sha256` cubre lo que puede cambiar la aritmética: arquitectura, modelos de CPU y su capacidad en PyTorch, versiones de PyTorch, CUDA y cuDNN, y modelo y capacidad de cada GPU. El controlador NVIDIA, el núcleo, la memoria total y el identificador físico de la GPU se registran fuera de la huella.

La campaña con máscaras guarda la identidad en su resumen y en cada recibo nuevo, fuera de la identidad del trabajo, para que los recibos ya confirmados sigan siendo válidos. Al publicar las fuentes de una comparación exige una única huella entre los recibos. Los recibos anteriores a este registro solo se admiten con `--unrecorded-platform <perfil>`, que los atribuye a un perfil, y entonces los recibos con plataforma deben corresponder a ese mismo perfil. El manifiesto de fuentes (versión 2) y el informe de la comparación conservan la huella y esa atribución.

## Perfiles de hardware

`configs/hardware/rtx4070-laptop.json` y `configs/hardware/dgx-gb10.json` declaran cada máquina con su arquitectura, GPU y capacidad, modelo de memoria y límites para MARS-TITAN. El perfil se elige siempre por su nombre y se compara con la plataforma detectada antes de empezar. Cualquier diferencia detiene la ejecución.

| Perfil | Memoria | Límites de MARS-TITAN |
| --- | --- | --- |
| `rtx4070-laptop` | dedicada | 24.576 MiB de RAM, 7.680 MiB de VRAM, 16 hilos |
| `dgx-gb10` | unificada | 65.536 MiB entre anfitrión y GPU, 8 hilos |

En la GB10 corre una aplicación de producción. Con memoria unificada el perfil exige además que el proceso corra en un cgroup con `memory.max` no superior al límite, por ejemplo con `systemd-run --user --scope -p MemoryMax=64G`. Está pendiente de comprobar si las reservas del controlador de la GPU cuentan en ese cgroup, así que el presupuesto de VRAM de la declaración de ejecución se suma dentro del mismo límite. La elección de 8 hilos es provisional hasta medir la carga de la aplicación de producción.

La declaración de ejecución de la campaña pasa a la versión 2 y nombra el perfil en el que se midieron sus estimaciones de memoria. Sus presupuestos deben caber en los límites del perfil. `historical-masked-campaign-execution.json` nombra `rtx4070-laptop`, así que en la GB10 se rechaza. Antes de lanzar allí la campaña hace falta una declaración propia con memoria medida en esa máquina. Las órdenes sin declaración de ejecución, como las comprobaciones CUDA de M2 y M3, toman el perfil de `MARS_TITAN_HARDWARE_PROFILE`.

## Comprobación pendiente en la GB10

`mars_titan.hardware.profile_check` compara la máquina con su perfil y escribe un recibo JSON. No ajusta nada, no ejecuta pasos de optimizador y no lee datos de la campaña. Comprueba el perfil, que PyTorch traiga código para la capacidad de la GPU, un producto de matrices en FP32 estricto frente a FP64, CuPy y XGBoost con CUDA y, si se indica, CTest de una compilación nativa sin la etiqueta `optimizer-steps`.

```bash
systemd-run --user --scope -p MemoryMax=64G -p CPUQuota=800% \
  uv run --no-sync python -m mars_titan.hardware.profile_check \
  --profile dgx-gb10 --output gb10-check.json \
  --native-build build/native/native-ppo-release
```

La orden está probada con identidades y PyTorch sustitutos en el portátil. Su ejecución en la GB10, el nombre exacto que da `nvidia-smi` a esa GPU y la memoria que informa están pendientes.
