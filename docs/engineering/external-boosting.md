# Boosting CUDA con memoria externa

`xgboost_external_cuda` es una referencia distinta de HistGradientBoosting.
Utiliza XGBoost 3.3.0, CuPy 14.2.0 y `ExtMemQuantileDMatrix`. Las versiones están
fijadas en el extra `boosting`. La ruta exige `cuda:0` y rechaza una sustitución
por CPU. HistGradientBoosting sigue disponible como control CPU explícito.

La inicialización carga PyTorch antes de CuPy. En el equipo de comprobación,
importar CuPy primero seleccionaba bibliotecas del CUDA 13.4 del sistema y
una capa lineal de PyTorch fallaba con `CUBLAS_STATUS_NOT_INITIALIZED`. PyTorch
usa sus bibliotecas CUDA 13.0. La regresión se prueba en un proceso nuevo y
no se modifica el controlador. La [guía de instalación de CuPy](https://docs.cupy.dev/en/stable/install.html)
explica la selección entre varias instalaciones de CUDA y las bibliotecas
distribuidas como paquetes. No se infiere compatibilidad por compartir el
número principal de versión.

Cada fila concatena la ventana de precios, texto, gráfico, fundamentales y
macro, en ese orden. El lector es `CorpusDataset`, compartido con RNN, LSTM,
GRU y DLinear. Solo se ajustan etiquetas de entrenamiento. La validación se
evalúa después del ajuste y la reserva final no se abre.

## Memoria y recorrido

Los bloques de entrada se comprueban y convierten a `float32`. Se rechazan
valores no finitos, desbordamientos de conversión, cambios de dimensión y
presupuestos insuficientes. La conversión reduce precisión respecto a la
referencia Ridge en `float64`. Esta diferencia queda identificada en el recibo.

El iterador vuelve a cortar los bloques en límites deterministas. XGBoost
3.3.0 necesita que las páginas se repitan con los mismos límites entre pasadas.
El orden y los valores también se contrastan mediante una huella. El recuento
de cada pasada debe coincidir con toda la población de ajuste. No se permite
resolver una falta de memoria reduciendo silenciosamente la población.

La caché de páginas puede residir en RAM o en disco. El control conservador
de RAM usa `N × (4F + 16)` bytes para `N` filas y `F` variables. Es un límite de
admisión, no una medida del pico completo del proceso. Hay memoria adicional
para índices, gradientes, histogramas, copias temporales y el contexto CUDA.
La ejecución completa necesita un límite de proceso y margen del sistema.

`max_batch_bytes` limita cada bloque y su representación convertida. No limita
todas las asignaciones internas de las bibliotecas. `max_quantile_batches` no
se usa como protección, porque la versión instalada avisa de que no tiene
efecto. La caché en disco puede reducir el consumo de RAM, pero añade tráfico
de almacenamiento y PCIe. No se presupone que sea más rápida.

La [documentación de memoria externa de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)
describe el iterador, las páginas externas y los requisitos de la ruta GPU.
La versión de la documentación puede diferir de la instalada. Las opciones
anteriores se han contrastado con la API y las pruebas de XGBoost 3.3.0.

## Ejecución y recuperación

Tras preparar el entorno con `uv sync --extra cuda --extra encoders --extra
research --extra boosting`, el módulo `mars_titan.training.external_corpus`
acepta `--manifest`, `--output`, `--rounds`, `--max-depth`, `--max-bin`,
`--learning-rate`, `--seed` y los presupuestos de lote y caché.
`--disk-cache` selecciona disco. `--resume` exige la identidad anterior de
datos, código, parámetros y versiones. No se sincronizan dependencias mientras
otro proceso científico usa ese entorno.

Se confirma un modelo UBJ cada diez rondas por defecto y al completar el
ajuste. El recibo conserva su huella, el número de filas, las rondas, las
pasadas completas y los parámetros. UBJ conserva además el presupuesto de
predicción y la auditoría. Una interrupción puede perder trabajo desde la
última ronda confirmada, pero no mezcla estados de rondas distintas.

El archivo y el directorio se sincronizan antes de publicar su referencia.
Se rechaza un modelo serializado mayor de 128 MiB antes de sustituir el recibo,
porque ese es también el límite de carga. La creación inicial de una ejecución
es exclusiva y su bloqueo impide dos escritores simultáneos.

La recuperación reconstruye la matriz desde los bloques y comprueba su
huella antes de continuar los árboles. No es aprendizaje incremental sobre
nuevas empresas ni una continuación sobre otra edición de los datos.
El muestreo de filas y columnas está fijado a uno. Las pruebas comparan
predicciones exactas entre ajuste continuo y ajuste interrumpido y recuperado.

Las predicciones de entrenamiento y validación se guardan en Parquet, con
identificador de muestra, activo, mercado, instante, objetivo y control cero.
Se comprueba la igualdad del modelo en memoria con el modelo restaurado.
MAE y MSE se informan por fila y por sesión, con desglose de mercado.

El tiempo medido pertenece a cada intento. El RSS es el máximo durante la
vida del proceso. La memoria CUDA registrada son observaciones de toda la
GPU, que incluyen otras aplicaciones. No se publica como pico exclusivo de
XGBoost ni como una medición de energía.

Las pruebas de integración usan una población sintética pequeña. No acreditan
una mejora predictiva ni rendimiento a escala completa. El perfil con millones
de filas y la búsqueda tabular siguen siendo comprobaciones distintas.

La [verificación local](../../reports/resources/external-boosting-quality.json)
incluye la batería completa, cobertura de sentencias y ramas, complejidad,
CRAP y cuatro mutaciones dirigidas. Estas medidas describen los casos
ejecutados, no garantizan ausencia de errores ni equivalencia entre versiones
de CUDA que no se hayan probado.
