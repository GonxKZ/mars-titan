# Coste de preparación de las etiquetas residuales

La preparación de etiquetas de 66 activos pasa de una mediana de **62,12 segundos
a 11,72 segundos**. Las tres repeticiones de cada motor producen los mismos
Parquet, comprobados mediante SHA-256. La mejora corresponde al recorrido de
lectura, cálculo, comprobación de huellas y escritura, no al entrenamiento.

La edición tiene 245 muestras de ajuste y 225 de validación. Se calculan las
8.565 decisiones del calendario de cada activo para conservar el historial y
los motivos de exclusión. No se abre el test final ni se cambian las etiquetas
de las campañas guardadas.

## Cambio y referencia

El perfil previo identifica miles de cortes de Series y comparaciones de fechas
de pandas por activo. Se convierten las fechas una sola vez y se utilizan vistas
de arrays para las ventanas. Se conservan las operaciones NumPy de la regresión
con intercepto, su orden, los 126 pares mínimos y la ventana de 252 sesiones.

La referencia de `budget_targets.py` permanece intacta. El motor `reference`
permite repetirla y `numpy` selecciona la ruta nueva. La preparación registra el
motor, las huellas de ambas implementaciones y las versiones de sus dependencias.
Un cambio de motor o de código exige otra salida y no reutiliza silenciosamente
las etiquetas de una edición anterior.

Las pruebas contrastan coeficientes, etiquetas, fechas, historial y exclusiones
con igualdad exacta. Incluyen precios ausentes, publicación posterior, NaT,
mercado constante, varianza casi nula y cambios de ventana. Una perturbación
posterior no altera las etiquetas anteriores. La comparación no se limita a
redondear los errores de entrenamiento.

## Medición completa

| Motor | Mediana de proceso | Intervalo de las tres repeticiones | Pico de RAM observado |
| --- | ---: | ---: | ---: |
| Referencia pandas | 62,12 s | 61,51 a 62,78 s | 202,57 a 213,40 MiB |
| Arrays NumPy | 11,72 s | 11,62 a 11,89 s | 211,32 a 224,39 MiB |

La razón entre medianas es **5,30**. El ahorro mediano es de 50,40 segundos. La
reducción de tiempo acepta un aumento pequeño de memoria en esta medición. No
se afirma que el consumo sea menor ni que exista una aceleración equivalente de
todo el entrenamiento.

Se utiliza un AMD Ryzen 9 8945HS, 32 GB de RAM, Linux x86-64 y Python 3.12.14.
OMP y OpenBLAS tienen un límite de cuatro hilos. Cada repetición arranca un proceso
nuevo y escribe en un directorio distinto, sin reutilizar etiquetas. El orden de
los dos motores se alterna. El tiempo de proceso incluye arranque e importaciones.
El pico de RAM se lee de `VmHWM` del propio proceso.

La caché del sistema operativo no se vació. Parte de la medición coincidió con
la campaña neuronal y con comprobaciones pequeñas en CPU. No es una prueba de
rendimiento máximo aislado. No se ha medido energía, tráfico de memoria física,
fallos de caché ni instrucciones por ciclo.

## Decisión sobre C++ y CUDA

El entrenamiento ya emplea los operadores compilados y kernels CUDA de PyTorch.
Parquet y las operaciones numéricas utilizan las implementaciones compiladas de
PyArrow y NumPy. El cambio elimina trabajo en Python sin sustituir esas bibliotecas.

No se incorpora un kernel propio para este tramo. Después del cambio, toda la
preparación de etiquetas ocupa unos 12 segundos en esta edición y una parte de
ese tiempo es lectura, huellas y escritura. La campaña neuronal de la misma
edición duró unos 25 minutos. No se ha demostrado que sustituir su cálculo OLS por
C++ compense otra implementación y su mantenimiento. La decisión se revisará con
una población materializada mayor y un perfil nuevo, sin alterar las campañas
anteriores.

La ruta temporal usa nanosegundos. Fechas artificiales fuera de su rango, como el
año 2500, fallan al convertirlas. No se reinterpretan como fechas históricas ni
se aceptan silenciosamente. El intervalo del corpus utilizado cabe en ese rango.

## Reproducción

La [receta de medición](../analysis/residual_preparation_benchmark.py) conserva las
medidas por repetición y rechaza una salida existente. La
[evidencia de recursos y calidad](residual-preparation-quality.json) recoge las
huellas, pruebas y límites.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python \
  reports/analysis/residual_preparation_benchmark.py \
  data/processed/training-edition-post-scan-20260923/materialized.json \
  data/processed/training-edition-post-scan-20260923/prepared \
  data/interim/residual-benchmark-repeated --repetitions 3
```
