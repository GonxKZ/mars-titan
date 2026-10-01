# Reproducibilidad de MARS-TITAN

Autor: Gonzalo García Lama. Estado actualizado el 19 de septiembre de 2026.

El repositorio contiene preparación temporal de datos y sondas MLP y GRU para medir coste. La arquitectura MARS-TITAN y la comparación confirmatoria siguen pendientes. La [preparación ejecutada](../data/preparation.md) y el [presupuesto experimental](../../reports/resources/campaign-budget.md) distinguen implementación, datos admitidos y entrenamientos observados.

## Entorno del proyecto

Se requieren Git, uv y ripgrep. Para comprobar la configuración nativa se necesitan CMake y compiladores C/C++. En Debian o Ubuntu, las dependencias de sistema pueden prepararse con:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends git ripgrep cmake build-essential
```

Las herramientas de sistema se comprueban en el equipo local. uv gestiona las dependencias de Python por separado. No se configura integración continua científica en GitHub. Actions se limita a la publicación de GitHub Pages autorizada por el responsable del proyecto.

La captura de [fuentes públicas](../data/public-source-updates.md) utiliza `curl`. La inspección de documentos PDF utiliza `pdfinfo`, incluido en `poppler-utils` en Debian y Ubuntu. Son requisitos adicionales de esas operaciones, no del entrenamiento ni de las pruebas sin red. El actualizador no instala programas, crea cuentas o activa tareas periódicas.

Python se gestiona con uv. El entorno local es `.venv/`, con Python 3.12 como versión de trabajo. `pyproject.toml` declara las dependencias y `uv.lock` conserva la resolución. El paquete se construye con Hatchling y se instala en modo editable durante el desarrollo. La orden `mars-data` expone las operaciones de preparación.

Para preparar las herramientas de documentación y calidad:

```bash
uv sync --locked
```

El grupo `dev` contiene las utilidades de comprobación. Los extras `data`, `research` y `notebooks` añaden herramientas opcionales. `cuda` y `encoders` se han usado en la preparación multimodal y las mediciones locales. No hace falta PyTorch para revisar documentación o validar formatos. Las pruebas que lo necesitan se omiten de forma explícita si no está instalado.

Para reproducir la preparación y sus comprobaciones con GPU:

```bash
uv sync --locked \
  --extra cuda \
  --extra encoders
```

El extra `cuda` obtiene PyTorch desde el índice CUDA 13.0 declarado en `pyproject.toml`, para Linux x86-64. Se han comprobado importación, operación en `cuda:0`, extracción de representaciones y entrenamiento supervisado. La resolución del lockfile no sustituye estas comprobaciones al cambiar de equipo.

Las herramientas actuales se ejecutan desde la raíz:

```bash
uv run --locked --extra cuda --extra encoders ruff check .
uv run --locked --extra cuda --extra encoders ruff format --check .
uv run --locked --extra cuda --extra encoders pytest
uv run --locked --extra cuda --extra encoders python scripts/check_repository.py
```

Las comprobaciones se ejecutan localmente. La suite incluye una integración breve en CUDA con datos sintéticos, independiente de los ensayos temporizados con datos reales. Si no hay GPU, esa prueba se omite con una explicación, no se transforma en un entrenamiento CPU. La publicación de la web no ejecuta estas pruebas ni los experimentos.

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

El [registro del entorno local](../../reports/resources/environment.json) conserva la comprobación del 19 de septiembre, con Python 3.12.14, las versiones efectivamente importadas, una operación en CUDA y la huella de `uv.lock`. La tabla anterior describe también herramientas del entorno compartido, no afirma que su intérprete sea el utilizado por el proyecto.

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

[native/](../../native/README.md) utiliza C17 en su interfaz pública y C++20 en el núcleo y el ejecutable financiero. `mars-titan-sim` lee Parquet y ejecuta escenarios sin iniciar Python. Arrow y Parquet se enlazan como bibliotecas C++. La opción `MARS_TITAN_ENABLE_CUDA` sigue siendo `OFF` por defecto y el simulador contable no incorpora kernels CUDA.

La configuración local se ha completado con CUDA desactivada y activada, en directorios temporales separados. En la segunda comprobación se usaron `nvcc` 13.4.92, GNU 15.2.0 y la arquitectura 89.

Las órdenes del README nativo separan Release de los perfiles de avisos, análisis estático, vida útil, sanitizadores y fuzzing. La verificación contrasta los resultados con la referencia Python y comprueba recuperación y concurrencia. Cada informe debe identificar el binario y las herramientas realmente ejecutadas. Detectar una opción del compilador no acredita por sí solo la corrección ni una mejora de rendimiento.

## Registro de futuros experimentos

Cada ejecución científica conservará la revisión del código, el lockfile, la configuración efectiva, las semillas y el entorno utilizado. También registrará el dispositivo, la precisión, los límites de memoria y los ajustes que puedan afectar al determinismo.

Los datos tendrán identificadores de versión, hashes y reglas de disponibilidad temporal. Los cortes de entrenamiento, validación, calibración y test se guardarán junto con los estados iniciales y las reglas de actualización de memoria. Una predicción deberá poder asociarse con la información disponible cuando se produjo.

Las métricas se calcularán a partir de predicciones guardadas y etiquetas maduras, siguiendo el [protocolo](../research/protocol.md). El registro incluirá fallos, pruebas descartadas y cambios de configuración. Una semilla fija por sí sola no garantiza resultados idénticos entre versiones, dispositivos o operaciones.

Los datos originales, pesos y resultados voluminosos permanecerán fuera de Git. Se versionarán su procedencia, las condiciones de uso y los metadatos necesarios para reconstruir el experimento cuando sea posible. Un recurso inaccesible o una restricción de licencia se documentará como límite de reproducción.
