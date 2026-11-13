# Campaña de comparación en dos equipos: guía de uso

Estado: guía operativa de la campaña declarada en `configs/experiments/comparison-campaign-v1.json` (#836). Describe órdenes y comprobaciones. No contiene resultados. El diseño experimental, las medidas del piloto y el dimensionado están en [la declaración de la campaña](../experiments/comparison-campaign-v1.md). Fecha: 13 de noviembre de 2026.

## Qué hace cada equipo

La campaña reparte los modelos en dos colas con la misma edición de datos, cohortes, ventanas y semillas. Los ajustes neuronales comparten receta y parada. Los lectores tienen su presupuesto común y un control sin banco. Ridge y LightGBM conservan las diferencias que recoge el diseño experimental.

| Cola | Equipo | Modelos |
| --- | --- | --- |
| `references` | El portátil que tiene la edición de datos | Transformer compacto, GRU, LSTM, RNN, DLinear, Ridge, la regla de momentum y reversión y LightGBM |
| `mars_titan` | El segundo equipo, con 8 GB de VRAM | Titans-MAC y sus dos controles de memoria, MARS-TITAN con las políticas M0 a M3, la red de áreas y sus ablaciones, los grupos de activos y su control, la atención entre activos y su control, los refinamientos K = 2 y K = 4 y el puente de evidencia |

Cada trabajo es un modelo en un origen y una semilla. La cola ejecuta primero los orígenes 2019 a 2023 por niveles de prioridad (`core`, `controls`, `networks`, `refinements` y `bridge`) y después repite esos niveles con los años añadidos 2017, 2016 y 2015. La fecha límite declarada es el 5 de diciembre de 2026 a las 00:00 de Madrid. La cola pide a los trabajos que terminen su siguiente barrera y deja pendiente lo que no haya completado. El orden da prioridad a la primera etapa sobre los años añadidos.

El estado de la campaña vive fuera del repositorio, en `~/.local/state/mars-titan/comparison-campaign-v1`: cohortes, cola, registros, salidas y permiso. La opción `--state` o la variable `MARS_TITAN_COMPARISON_STATE` lo cambian de sitio.

## Reglas que conviene tener presentes

- Los dos equipos ejecutan el mismo commit. El permiso guarda también la huella de la declaración y de los cambios locales en el código, incluidos los archivos nuevos. La cola se niega a continuar si cambia esa identidad. Durante la campaña se trabaja en el código desde otra copia. Un estado que ya tiene trabajos no admite autorizar otra identidad, para no mezclar resultados. Una ejecución distinta necesita otra carpeta con `--state`.
- La protección local del aprendizaje (`training-hold-2000.json`) no se toca. La campaña tiene su propio permiso, que registra quien la lanza con la orden `authorize`.
- El test final de 2024 permanece cerrado. La declaración no admite orígenes posteriores a 2023.
- Nada de esto sube datos al repositorio. El paquete de datos viaja en un disco y solo contiene derivados locales de la edición.

## Equipo de las referencias

Todas las órdenes se ejecutan desde la raíz del repositorio, con el entorno ya creado (`make setup`).

```bash
# 1. Cohortes de cada origen. Solo lee y escribe manifiestos.
uv run python -m mars_titan.training.comparison_campaign prepare

# 2. Comprobaciones previas: vistas, datos, GPU, memoria y disco.
uv run python -m mars_titan.training.comparison_campaign doctor --machine references

# 3. Paquete de datos para el segundo equipo, en una carpeta vacía de un disco externo.
read -r -p "Carpeta nueva para el paquete en el disco externo: " export_dir
test -n "$export_dir" && uv run python -m mars_titan.training.comparison_campaign export --destination "$export_dir"

# 4. Permiso expreso de quien lanza la campaña en este equipo.
uv run python -m mars_titan.training.comparison_campaign authorize --note "Campaña de comparación, equipo de las referencias"

# 5. Cola de este equipo. La misma orden continúa después de una parada.
uv run python -m mars_titan.training.comparison_campaign run --machine references
```

El diagnóstico comprueba 12 GiB de RAM disponible para las referencias y 6 GiB para MARS-TITAN. LightGBM ajusta como máximo una semilla nueva por proceso y después emite en otro proceso. Cada semilla conserva su modelo en `model.json` y el campo `rounds` indica el prefijo elegido para predecir.

El paquete ocupa unos 20,1 GiB. `export` crea la carpeta de destino si falta. Con `--part-gib 3.5`, cada tar ocupa como máximo 3,5 GiB, incluidas sus cabeceras, y cabe en un disco FAT32. Si un archivo individual no cabe, la exportación informa del motivo. El portátil necesita unos 5 GiB libres para las salidas de su cola. La orden `doctor` avisa si no los hay, y la cola se detiene sin perder trabajos cuando queda menos de 1 GiB.

Se puede exportar antes del commit final si los datos y las cohortes ya están fijados. Al importar aparece un aviso si el paquete declara otro commit o declaración. Las cohortes deben coincidir y los dos equipos deben ejecutar la misma versión final del código.

## Segundo equipo con Windows 11

El segundo equipo ejecuta la campaña dentro de WSL2 con Ubuntu 26.04, la misma distribución que el portátil. El enlace episódico compilado que usan M1, M2 y M3 viaja en el paquete de datos y busca sus bibliotecas en rutas absolutas, así que el usuario de Linux y la ruta del repositorio deben coincidir con los del portátil.

### Preparar Windows y WSL2

1. Instalar un controlador de NVIDIA para Windows compatible con CUDA. Dentro de WSL no se instala el controlador de Linux, como indica la [guía de NVIDIA para WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
2. En PowerShell como administrador, consultar `wsl --list --online` e instalar Ubuntu 26.04 con el nombre que muestre esa lista. Si aparece `Ubuntu-26.04`, la orden es `wsl --install -d Ubuntu-26.04`. Reiniciar cuando lo pida Windows y comprobar con `wsl --list --verbose` que usa la versión 2. La [guía de Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install) recoge las órdenes y la importación de distribuciones que no figuren en la lista.
3. Al crear el usuario de Ubuntu, darle el mismo nombre que tiene en el portátil (el que muestra allí `whoami`), para que la carpeta personal tenga la misma ruta en los dos equipos.
4. WSL2 recibe por defecto la mitad de la memoria del equipo. Cada trabajo necesita unos 6 GiB. Si el equipo tiene 16 GB, crear `%UserProfile%\.wslconfig` con este contenido. Aplicarlo con `wsl --shutdown` antes de empezar la campaña, ya que esa orden detiene todas las distribuciones abiertas. Microsoft documenta ambos ajustes en [la configuración de WSL](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

   ```ini
   [wsl2]
   memory=12GB
   swap=8GB
   ```

5. Evitar que Windows suspenda o reinicie el equipo: en Energía, no suspender nunca con el equipo enchufado, y en Windows Update, pausar las actualizaciones hasta después del 5 de diciembre.
6. El disco de Windows que aloja WSL necesita unos 60 GB libres para el entorno, los datos y las salidas.

### Preparar el repositorio y los datos

Dentro de Ubuntu:

```bash
sudo apt update && sudo apt install -y git curl tmux make
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash

mkdir -p ~/Escritorio && cd ~/Escritorio
git clone https://github.com/GonxKZ/mars-titan.git "TFM TITANS"
cd "TFM TITANS"
uv sync --locked --all-extras
nvidia-smi
```

El entorno fijado usa Python 3.12 y PyTorch 2.14.0 con CUDA 13.0. El wheel incluye `sm_120`, correspondiente a la familia RTX 5060 en [la tabla de NVIDIA](https://developer.nvidia.com/cuda/gpus). Esto no sustituye una comprobación en el equipo de destino.

La rama `main` del repositorio debe estar en el mismo commit que el portátil (`git rev-parse HEAD` en los dos). Con el disco externo conectado, Windows lo monta dentro de WSL en `/mnt/d`, `/mnt/e` o la letra que tenga:

```bash
# 1. Restaurar el paquete en las mismas rutas, con cada huella comprobada al escribir.
read -r -p "Carpeta del paquete de datos dentro de WSL: " data_dir
test -d "$data_dir" && uv run python -m mars_titan.training.comparison_campaign import --source "$data_dir"

# 2. Cohortes de este equipo y comprobaciones previas.
uv run python -m mars_titan.training.comparison_campaign prepare
uv run python -m mars_titan.training.comparison_campaign doctor --machine mars_titan
MARS_TITAN_REQUIRE_CUDA=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run pytest tests/models/titans/test_financial_learning_scale_cuda.py -q

# 3. Permiso expreso y cola, dentro de tmux para que cerrar la ventana no la detenga.
uv run python -m mars_titan.training.comparison_campaign authorize --note "Campaña de comparación, equipo de MARS-TITAN"
tmux new -s mars-titan
uv run python -m mars_titan.training.comparison_campaign run --machine mars_titan
```

La orden `doctor` debe terminar sin ninguna línea `FALTA`. Comprueba las cohortes del paquete, que `cuda:0` responde, que el enlace episódico carga y que hay memoria y disco. La línea «código y declaración» debe mostrar el mismo commit y la misma huella en los dos equipos, sin cambios locales. El aviso del permiso desaparece al ejecutar `authorize`.

La comprobación CUDA de esas órdenes verifica emisión, gradientes y recuperación sin pasos de optimizador.

Conviene dejar abierta la ventana de la terminal. WSL puede apagar la distribución cuando no queda ninguna sesión abierta, aunque haya procesos en segundo plano.

### Si el enlace episódico no carga en el segundo equipo

Los modelos `mars_titan_m1`, `m2`, `m3` y sus variantes necesitan el enlace. Si `doctor` informa de que no carga, hay que comprobar las versiones de Python y PyTorch y las rutas del paquete. Intercambiar las colas es una alternativa si el portátil dispone del espacio que necesita `mars_titan`, estimado en 14 GiB. En ese caso ejecuta `run --machine mars_titan` y el segundo equipo `run --machine references`. Antes se repite `doctor` para el nuevo papel de cada equipo.

## Seguir, parar y reanudar

```bash
# Recuento por nivel, horas gastadas y estimación de las restantes.
uv run python -m mars_titan.training.comparison_campaign status --machine mars_titan

# Tabla de errores por modelo con lo ya terminado.
uv run python -m mars_titan.training.comparison_campaign results --machine mars_titan

# Lista ordenada de los trabajos de la cola.
uv run python -m mars_titan.training.comparison_campaign plan --machine mars_titan
```

- Parar: `Ctrl+C` en la terminal de la cola. Las redes guardan su checkpoint. Ridge termina el alpha actual antes de parar. LightGBM conserva las semillas confirmadas y vuelve a ajustar la semilla incompleta al reanudar. Puede detenerse también entre particiones de la emisión.
- Reanudar: repetir la misma orden `run`. Los trabajos terminados no se repiten y el pausado continúa donde estaba. Un apagado inesperado se recupera igual.
- Fallos: un trabajo que falla se reintenta hasta tres veces sin detener la cola. Lo que sigue fallando queda anotado con su traza en `queue/` y su salida en `logs/`. La orden `retry --machine …` lo devuelve a la cola cuando la causa está resuelta.
- Antes de vaciar la cola, `run` arranca hasta su primera barrera cada modelo que no necesita un padre ajustado. Los lectores se comprueban cuando su padre está disponible.
- `--tier` limita la orden `run` a uno o varios niveles, por ejemplo `--tier core --tier controls`. Devuelve éxito cuando terminan los trabajos de esos niveles, aunque queden otros sin ejecutar.

`status` identifica los trabajos fallidos con su último error y los intentos abiertos con su hora de inicio. `results` muestra el MAE de evaluación, el MAE de cero sobre las mismas filas, su diferencia porcentual, la diferencia tras calibrar y las correlaciones de Pearson y de rangos por sesión. La escala se elige solo con la partición de calibración. La regla y LightGBM no tienen un MAE bruto comparable y esa columna queda vacía. La tabla resume lo terminado y no sustituye el análisis final con incertidumbre.

Cada origen construye su índice de ajuste una sola vez y lo comparte entre modelos. En el piloto, la primera construcción con 1.024 activos y toda la historia añadió unos cinco minutos.

En el portátil, el servicio de vigilancia térmica congela los trabajos por encima de 92 °C y los reanuda por debajo de 82 °C. Esas pausas alargan la campaña pero no alteran los resultados.

### Si el segundo equipo va justo de tiempo

La cola `references` es mucho más corta que la cola `mars_titan`. Cuando el portátil termine la suya puede hacerse cargo de niveles enteros de la otra, porque tiene todos los datos. Los niveles `controls` y `networks` no dependen de otros trabajos. Los lectores (`core`, `refinements` y `bridge`) necesitan en el mismo equipo el Titans-MAC de su origen y su semilla.

```bash
# En el segundo equipo: parar con Ctrl+C y relanzar sin el nivel que se cede.
uv run python -m mars_titan.training.comparison_campaign run --machine mars_titan --tier core --tier controls --tier refinements --tier bridge

# En el portátil: ejecutar ese nivel de la cola mars_titan.
uv run python -m mars_titan.training.comparison_campaign run --machine mars_titan --tier networks
```

Cada trabajo ocupa entre 20 y 27 MB. El nivel `networks` completo son unos 5 GB, así que el portátil necesita ese espacio libre antes de aceptarlo.

## Reunir los resultados

Al terminar, o en cualquier momento con la cola parada, las salidas del segundo equipo se llevan al portátil con un archivo tar. Las predicciones de validación, calibración y evaluación de cada trabajo están en `runs/`, y su estado en `queue/`.

```bash
cd ~/.local/state/mars-titan/comparison-campaign-v1
tar -cf /mnt/d/mars-titan-resultados-mars_titan.tar cohorts queue logs runs
```

En el portátil se extrae en una carpeta aparte, por ejemplo `~/.local/state/mars-titan/comparison-campaign-v1-mars_titan`, para no mezclar las dos colas.

## Qué comprobar si algo no encaja

| Síntoma | Causa habitual | Qué hacer |
| --- | --- | --- |
| `Falta el permiso expreso de esta declaración y de este código` | No se ejecutó `authorize`, o cambió el commit, la declaración o el código local | Recuperar la versión autorizada. Para otra ejecución, preparar y autorizar otro `--state` |
| `doctor` marca `FALTA` en los datos de una cohorte | El paquete no se importó entero | Repetir `import`. Los archivos ya restaurados se conservan |
| `import` dice que el paquete se exportó desde otra carpeta personal | El usuario de Linux no se llama como en el portátil | Crear en Ubuntu un usuario con ese nombre y trabajar con él |
| `torch.cuda.is_available() es falso` | Falta el controlador de NVIDIA en Windows o WSL no ve la GPU | Actualizar el controlador y ejecutar `wsl --shutdown`. La campaña nunca cambia a CPU |
| La cola se detiene por disco | Queda menos de 1 GB libre | Liberar espacio y repetir `run` |
| Un trabajo muere sin mensaje | Falta de memoria | Cerrar otros programas o subir `memory` en `.wslconfig` |
