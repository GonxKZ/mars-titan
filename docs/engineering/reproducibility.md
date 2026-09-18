# Reproducibilidad de MARS-TITAN

Autor: Gonzalo García Lama. Estado comprobado el 18 de septiembre de 2026.

El repositorio prepara la documentación, el entorno y las herramientas necesarias para iniciar la comparación. No contiene todavía una implementación científica ni resultados de entrenamiento. Las comprobaciones descritas aquí deben distinguirse de la reproducción de un experimento.

## Entorno del proyecto

Python se gestiona con uv. El entorno local será `.venv/`, con Python 3.12 como versión de trabajo. `pyproject.toml` declara los grupos de dependencias y `uv.lock` conserva la resolución. La opción `package = false` indica que el repositorio aún no se construye ni se instala como un paquete Python.

Para preparar las herramientas de documentación y calidad:

```bash
uv sync --locked
```

El grupo `dev` contiene las utilidades de comprobación. Los extras `data`, `research`, `cuda` y `notebooks` quedan disponibles para el trabajo posterior. No hace falta instalar PyTorch en el entorno local para revisar documentos, comprobar la biblioteca de referencias o configurar CMake.

Cuando se inicie la implementación científica, la instalación de sus dependencias se hará explícitamente:

```bash
uv sync --locked \
  --extra data \
  --extra research \
  --extra cuda \
  --extra notebooks
```

El extra `cuda` obtiene PyTorch desde el índice CUDA 13.0 declarado en `pyproject.toml`, para Linux x86-64. No se ha instalado ese conjunto completo en esta preparación. La resolución del lockfile no sustituye una comprobación de importación y funcionamiento en el entorno donde se vaya a ejecutar.

Las herramientas actuales se ejecutan desde la raíz:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked python scripts/check_repository.py
```

La integración continua con CPU se limita a estas herramientas y a la documentación. No ejecuta entrenamientos ni sustituye las comprobaciones de CUDA.

## Equipo y entorno compartido

La comprobación local ha observado:

| Elemento | Valor observado |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| Memoria total informada por `nvidia-smi` | 8188 MiB |
| Driver NVIDIA | 595.91.07 |
| Capacidad de cómputo | 8.9 |
| Compilador CUDA | `/usr/local/cuda-13.4/bin/nvcc`, versión 13.4.92 |
| CMake | 4.2.3 |
| Compiladores C y C++ | GNU 15.2.0 |
| Entorno compartido | `/home/gonzalo/.venvs/pytorch-cuda` |
| Python compartido | 3.14.4 |
| PyTorch compartido | `2.14.0+cu130` |
| Runtime CUDA informado por PyTorch | 13.0 |
| `torch.cuda.is_available()` | `True` |

Los valores corresponden a esta máquina y fecha. La memoria libre depende de otros procesos. Detectar el dispositivo no acredita que un modelo concreto quepa en memoria ni que sus operaciones sean deterministas.

El entorno compartido puede servir para tareas independientes compatibles. No reproduce automáticamente la resolución de `uv.lock` ni la versión de trabajo Python 3.12 del proyecto. Se consulta sin modificar sus dependencias:

```bash
nvidia-smi
uv run --directory /tmp --no-project \
  --python /home/gonzalo/.venvs/pytorch-cuda/bin/python python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible en el entorno compartido")

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Dispositivo:", device)
print("GPU:", torch.cuda.get_device_name(device))
print("Capacidad:", torch.cuda.get_device_capability(device))
PY
```

El comando usa `--no-project` y un directorio externo para evitar que uv sincronice este repositorio sobre el entorno compartido. No instala dependencias ni ejecuta un entrenamiento. Para el desarrollo del proyecto se mantendrá `.venv/` con su lockfile.

Antes de una carga de aprendizaje profundo se repetirán `nvidia-smi` y la comprobación de PyTorch en el entorno elegido. Se seleccionará `cuda:0` de forma explícita y se registrarán versiones y memoria. Un fallo de CUDA deberá dejar un error visible con su motivo. Cualquier ejecución con CPU requerirá una decisión explícita y quedará identificada en el registro del experimento.

## Toolkit, runtime y configuración nativa

La máquina dispone de toolkit CUDA 13.4 y el entorno compartido usa una distribución de PyTorch para CUDA 13.0. No se exige igualdad exacta entre ambos números para usar esa distribución de PyTorch. El toolkit local se necesita para compilar código CUDA propio. El driver debe soportar el runtime que se utilice.

Una extensión nativa introduce una comprobación adicional. Antes de compilarla se revisarán las versiones admitidas por PyTorch, el toolkit, el compilador C/C++ anfitrión, la ABI y la arquitectura de GPU. Detectar CUDA en PyTorch no demuestra esa compatibilidad de compilación. La documentación de [extensiones de PyTorch](https://docs.pytorch.org/docs/stable/cpp_extension.html) describe los requisitos del toolkit para extensiones CUDA.

[native/](../../native/README.md) declara C17, C++20 y CUDA opcional. La opción `MARS_TITAN_ENABLE_CUDA` es `OFF` por defecto. Al activarla se requieren el compilador CUDA y el toolkit. Solo existe un target de interfaz para compartir opciones, sin fuentes científicas, kernels o binarios.

La configuración local se ha completado con CUDA desactivada y activada, en directorios temporales separados. En la segunda comprobación se usaron `nvcc` 13.4.92, GNU 15.2.0 y la arquitectura 89.

Las órdenes de configuración están en el README nativo. Un resultado correcto de CMake acredita la detección de herramientas y sus comprobaciones de compilador. No verifica un kernel, su concordancia numérica o su rendimiento. Estas pruebas se diseñarán si el perfilado justifica una implementación nativa.

## Registro de futuros experimentos

Cada ejecución científica conservará la revisión del código, el lockfile, la configuración efectiva, las semillas y el entorno utilizado. También registrará el dispositivo, la precisión, los límites de memoria y los ajustes que puedan afectar al determinismo.

Los datos tendrán identificadores de versión, hashes y reglas de disponibilidad temporal. Los cortes de entrenamiento, validación, calibración y test se guardarán junto con los estados iniciales y las reglas de actualización de memoria. Una predicción deberá poder asociarse con la información disponible cuando se produjo.

Las métricas se calcularán a partir de predicciones guardadas y etiquetas maduras, siguiendo el [protocolo](../research/protocol.md). El registro incluirá fallos, pruebas descartadas y cambios de configuración. Una semilla fija por sí sola no garantiza resultados idénticos entre versiones, dispositivos o operaciones.

Los datos originales, pesos y resultados voluminosos permanecerán fuera de Git. Se versionarán su procedencia, las condiciones de uso y los metadatos necesarios para reconstruir el experimento cuando sea posible. Un recurso inaccesible o una restricción de licencia se documentará como límite de reproducción.
