# Plataforma de hardware: portátil con RTX 4070

Autor: Gonzalo García Lama. Estado del 13 de octubre de 2026 ([#561](https://github.com/GonxKZ/mars-titan/issues/561)).

MARS-TITAN se ejecuta en una sola máquina: el portátil de desarrollo, x86-64 con un AMD Ryzen 9 8945HS de 16 hilos, 30 GiB de RAM útiles y una GeForce RTX 4070 Laptop de 8 GiB (capacidad 8.9). Todas las recetas, presupuestos y medidas del proyecto se tomaron en ella. Este documento recoge cómo se declara esa máquina, cómo se registra en cada ejecución y cómo se comprueba antes de una campaña.

## Dependencias

`pyproject.toml` declara en `[tool.uv] environments` y `required-environments` solo Linux x86-64. `uv.lock` resuelve para esa plataforma las ruedas CUDA 13.0 de PyTorch y torchvision, CuPy y XGBoost, y deja fuera los paquetes exclusivos de otros sistemas.

## Compilación nativa

`MARS_TITAN_ENABLE_CUDA` declara `CMAKE_CUDA_ARCHITECTURES` como `89-real`, solo SASS para la RTX 4070. Una GPU distinta falla al cargar en lugar de compilar PTX en ejecución sin aviso. Las consultas de LibTorch y PyArrow al configurar exigen que el paquete sea de la misma arquitectura que el destino, de modo que un entorno equivocado se rechaza al configurar y no al enlazar. La identidad de compilación registra las arquitecturas y las raíces de búsqueda.

El control de cancelación de `markov_filter` usa la cota 2^-63, que es el épsilon del `long double` extendido de x86-64. Un `long double` menos preciso conservaría su propio épsilon.

## Identidad de la plataforma

`mars_titan.hardware.platform_identity` registra sistema, arquitectura, procesador (`lscpu`), memoria, PyTorch, CUDA, cuDNN y cada GPU (`nvidia-smi`). Las GPU no salen de `torch.cuda` para no crear un contexto CUDA en el proceso de la campaña, que solo cuenta ese contexto tras su primer trabajo CUDA. Si la arquitectura del intérprete no coincide con la del procesador que describe `lscpu`, la plataforma queda marcada como emulada y el perfil la rechaza.

La huella `platform_sha256` cubre lo que puede cambiar la aritmética: arquitectura, modelos de CPU y su capacidad en PyTorch, versiones de PyTorch, CUDA y cuDNN, y modelo y capacidad de cada GPU. El controlador NVIDIA, el núcleo, la memoria total y el identificador físico de la GPU se registran fuera de la huella. Aunque todo corra en el portátil, una actualización de PyTorch o de CUDA cambia la huella.

La campaña con máscaras guarda la identidad en su resumen y en cada recibo nuevo, fuera de la identidad del trabajo, para que los recibos ya confirmados sigan siendo válidos. Al publicar las fuentes de una comparación exige una única huella entre los recibos. Los recibos anteriores a este registro solo se admiten con `--unrecorded-platform <perfil>`, que los atribuye a un perfil, y entonces los recibos con plataforma deben corresponder a ese mismo perfil. El manifiesto de fuentes (versión 2) y el informe de la comparación conservan la huella y esa atribución.

## Perfil de hardware

`configs/hardware/rtx4070-laptop.json` declara la máquina con su arquitectura, GPU y capacidad, memoria y límites para MARS-TITAN: 24.576 MiB de RAM, 7.680 MiB de VRAM y 16 hilos. Los límites dejan 6 GiB de RAM al sistema y 508 MiB de VRAM al escritorio. El perfil se elige siempre por su nombre y se compara con la plataforma detectada antes de empezar. Cualquier diferencia detiene la ejecución.

La declaración de ejecución de la campaña (versión 2) nombra el perfil en el que se midieron sus estimaciones de memoria, y sus presupuestos deben caber en los límites del perfil. `historical-masked-campaign-execution.json` nombra `rtx4070-laptop`. Las órdenes sin declaración de ejecución, como las comprobaciones CUDA de M2 y M3, toman el perfil de `MARS_TITAN_HARDWARE_PROFILE`.

## Comprobación de la máquina

`mars_titan.hardware.profile_check` compara la máquina con su perfil y escribe un recibo JSON. No ajusta nada, no ejecuta pasos de optimizador y no lee datos de la campaña. Comprueba el perfil, que PyTorch traiga código para la capacidad de la GPU, un producto de matrices en FP32 estricto frente a FP64, CuPy y XGBoost con CUDA y, si se indica, CTest de una compilación nativa sin la etiqueta `optimizer-steps`.

```bash
uv run --no-sync python -m mars_titan.hardware.profile_check \
  --profile rtx4070-laptop --output laptop-check.json \
  --native-build build/native/native-ppo-release
```

Conviene repetirla antes de una campaña y después de cualquier cambio del controlador, de PyTorch o del lock.
