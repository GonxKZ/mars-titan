# Configuración nativa de MARS-TITAN

Este directorio prepara CMake para C17 y C++20. CUDA es opcional y está desactivada por defecto. El target de interfaz `mars_titan::native_options` agrupa los estándares y los avisos de compilación para futuros objetivos. No genera una biblioteca, un ejecutable ni una extensión de Python.

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

`89` corresponde a la capacidad de cómputo 8.9 observada en la RTX 4070 Laptop. En otro equipo se debe comprobar la arquitectura y ajustar las rutas. Al activar CUDA, CMake exige tanto el compilador como el toolkit. Una ausencia o incompatibilidad produce un error visible, sin cambiar automáticamente a una compilación con CPU.

Conviene usar directorios de configuración distintos para CPU y CUDA. CMake conserva la elección de compilador en su caché. Estos directorios contienen archivos generados y quedan fuera del código fuente.

## Alcance de la comprobación

Configurar con éxito verifica que CMake reconoce las herramientas y puede realizar sus comprobaciones de compilador. No verifica kernels, resultados numéricos, gradientes, rendimiento o compatibilidad de una futura extensión con PyTorch. No hay fuentes científicas ni binarios del proyecto que compilar o ejecutar.

La estructura reserva [include/](include/README.md) para contratos públicos, [src/](src/README.md) para implementaciones C/C++ y [cuda/](cuda/README.md) para posibles kernels. Su contenido se decidirá después de perfilar una implementación de referencia en Python. No se añaden opciones de cálculo aproximado ni flags de optimización numérica antes de medir su efecto.

El toolkit local y el runtime usado por una distribución de PyTorch pueden tener versiones distintas. Antes de compilar una extensión habrá que comprobar la compatibilidad de PyTorch, toolkit, compilador anfitrión y ABI. Los detalles del entorno están en [reproducibility.md](../docs/engineering/reproducibility.md).
