# Fuentes de la variante KronLSTM

Revisión del 23 de octubre de 2026. La referencia es *Kronecker Delta Rule for Long Context
Linear Attention*, de Hartl, Zólyomi, List, Klambauer, Böck y Hochreiter,
[arXiv 2610.26834v1](https://arxiv.org/abs/2610.26834v1), presentada el 19 de octubre de 2026.
El código inspeccionado corresponde a [NX-AI/KronLSTM en
3d800c9a153a88ee9624fe6acf7f39a58017a17e](https://github.com/NX-AI/KronLSTM/tree/3d800c9a153a88ee9624fe6acf7f39a58017a17e).

Se han contrastado la [derivación del operador](https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/ops/kronlstm/README.md),
la [referencia recurrente](https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/ops/kronlstm/naive.py),
la [capa con puertas y caché](https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/layers/kronlstm.py)
y la [configuración del modelo](https://github.com/NX-AI/KronLSTM/blob/3d800c9a153a88ee9624fe6acf7f39a58017a17e/fla/models/kronlstm/configuration_kronlstm.py).
El código tiene licencia MIT, con avisos de FLA y NXAI conservados en
[la copia de la licencia](../../licenses/kronlstm-mit.txt).

El estado es una matriz completa por cabeza. La factorización de los productos de
Kronecker evita construir una transición cuadrática en el número de elementos del estado.
No restringe la memoria a rango uno. La capa publicada consulta después de actualizarla
y usa, por defecto, escritura y borrado desacoplados, valores normalizados y olvido vectorial.

Las hipótesis estacionarias del apéndice A3.5 se refieren a asociaciones, muestreo, rango
y paso especificados. No se han demostrado sobre las entradas y puertas de esta variante
financiera ni certifican el Jacobiano completo del predictor. Tampoco convierten una
mejora de un benchmark de lenguaje en una mejora del retorno residual.

La implementación compacta de MARS-TITAN es una adaptación experimental no canónica.
No utiliza el checkpoint lingüístico publicado ni reproduce su entrenamiento de 1.500
millones de parámetros y 100.000 millones de tokens. La comprobación de las ecuaciones,
la paridad numérica, el coste local y un efecto predictivo son evidencias distintas.

El backend propio `pytorch_reference_cpu_qkv64_v1` calcula la proyección QKV en FP64
cuando la entrada y los pesos son FP32 en CPU y devuelve su resultado en FP32.
La configuración del codificador y de cada bloque declara
`cpu_float32_qkv_accumulation='float64'`. CUDA FP32 y CPU FP64 conservan la proyección
nativa. Las puertas de olvido y objetivo siguen su cálculo FP32. Esta ruta de CPU
es una corrección numérica de la referencia y no una ejecución íntegramente FP32.

En una comprobación técnica con las semillas 0..7, la proyección nativa FP32 produjo dos
fallos de consistencia entre la ventana completa y su partición 3+5. La candidata
con QKV en FP64 superó los ocho casos con las tolerancias originales
`rtol=1e-5` y `atol=1e-6`. Frente a los mismos pesos convertidos a FP64, redujo el
error L2 de las ocho ventanas completas, pero el error máximo de la semilla 1
pasó de `6.11e-6` a `8.58e-6`. Esa referencia conserva las conversiones FP32 de
las puertas y tampoco constituye un forward íntegramente FP64.

La corrección cambia resultados respecto al backend anterior y tiene identidad
propia. La comprobación no demuestra una reducción universal del error, una aceleración
ni un efecto financiero. El estado de ventana y el contrato de memoria exterior
no cambian.
