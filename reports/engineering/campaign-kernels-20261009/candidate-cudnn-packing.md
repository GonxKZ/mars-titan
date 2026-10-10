# Pesos de cuDNN de la GRU candidata nativa

La GRU de precios de la candidata (`native/src/candidate.cpp`) llamaba a `at::gru` con cuatro pesos en almacenamientos separados. En CUDA, cuDNN necesita un bloque único y PyTorch lo compactaba en cada llamada, con el aviso «RNN module weights are not part of single contiguous chunk of memory» escrito en stderr. El cambio empaqueta los pesos una vez, como ya hacía `PolicyNetwork::pack_recurrent_weights` en la red de políticas.

## Cambio

`Candidate::pack_recurrent_weights` llama a `at::_cudnn_rnn_flatten_weight` sobre los cuatro parámetros de la GRU (modo GRU, entrada 5, oculto 128, una capa, `batch_first`). La función copia los valores al bloque de cuDNN y deja los parámetros registrados como vistas de ese bloque. Se ejecuta al final de `torch::nn::Module::to` dentro de `Candidate::to`, que es el punto por el que pasan la construcción en el dispositivo y `load_state`. En CPU, o sin cuDNN aceptable, no hace nada. La copia en el sitio de `candidate_run._load_parameters` conserva las vistas, así que el optimizador sigue viendo los mismos tensores.

El receptor de la llamada vive en `native/src/candidate_identity.cpp`, donde está `Candidate::to`. Es una línea. La rama `feat/native-policy-real-tapes` no toca ninguno de los tres archivos nativos modificados.

## Paridad

Se compararon el enlace de develop (`3f99e144…`) y el nuevo (`8780e628…`) en procesos separados, con el mismo bloque real de 128 filas de la vista CN `fold-012` (evento 3.519, enero de 2021), los mismos pesos iniciales y K = 1 y 4. En CPU y CUDA, FP32 y FP64, con la configuración numérica del ajuste y con `fp32_precision="ieee"`, los cuantiles, el estado, la fusión y los 21 gradientes son iguales bit a bit. La huella de parámetros no cambia. La comprobación CPU/CUDA existente (`cuda_candidate_check.py`) da los mismos errores máximos que su recibo anterior (1,5·10⁻⁶ en FP32 y 1,6·10⁻¹⁵ en FP64).

Un archivo guardado desde CUDA escribe ahora el bloque de la GRU como un solo almacenamiento y ocupa 631 bytes menos. Su recuperación reproduce las salidas bit a bit y la huella de parámetros coincide también al cargarlo en CPU.

## Medidas en `cuda:0`

Bloque de 128 filas, tres rondas alternas de cada enlace con 30 repeticiones de 20 llamadas. La CPU estaba compartida (carga media entre 12 y 16), por eso se alternan los enlaces.

| Llamada | Antes (ms) | Después (ms) | Cociente |
| --- | ---: | ---: | ---: |
| `encode_context` forward y backward | 4,38 ± 0,28 | 4,17 ± 0,25 | 1,05 |
| `encode_context` forward sin grafo | 2,18 ± 0,19 | 1,90 ± 0,17 | 1,14 |
| `forward` completo con backward | 6,07 ± 0,68 | 5,81 ± 0,38 | 1,04 |

El perfil de un forward con backward pasa de 4 copias DtoD, 10 `aten::copy_` y 39 `aten::empty` a ninguna copia DtoD, 6 `aten::copy_` y 34 `aten::empty`. El tiempo de los eventos CUDA coincide con el de reloj, porque la llamada está limitada por los lanzamientos.

En el recorrido del entrenador hasta el paso (8 tramos tras 2 de calentamiento, 68.655 observaciones) se compactaban los pesos 616 veces sin opciones de memoria y 1.232 con `accumulation_rows=128` y `recompute`, una por forward. Con el cambio no hay ninguna. El tiempo por tramo no permite ver la mejora: el cociente emparejado por tramo es 0,95 ± 0,26 y 0,91 ± 0,23, dentro del ruido de la carga concurrente, porque el tramo está dominado por el recorrido Python del entrenador. La memoria pico del asignador baja unos 11 MB sin opciones y 0,2 MB con ellas.

## Pruebas

- `tests/models/candidate/test_cudnn_packing.py`: 9 pasan en `cuda:0` (bloque único tras construir, copiar en el sitio y recuperar, ausencia del aviso, igualdad bit a bit con la ruta que compacta en forward y backward, FP32 y FP64) y 2 en CPU.
- Mutaciones: sin la llamada en `Candidate::to`, empaquetando copias en lugar de los parámetros, y con el enlace anterior. En los tres casos fallan 7 pruebas CUDA.
- Regresión CPU con el enlace nuevo: 157 pruebas de la candidata, su entrenador, su banco y la cabeza nativa pasan (7 omitidas por CUDA). `ctest` de la candidata: 8 de 8, incluida `candidate_cuda`. `clang-tidy` sin avisos.

Ninguna prueba ni medida aplica pasos de optimizador.

## Identidad

Cambian el `native_binary_sha256` del adaptador y la huella de fuentes de la compilación nativa, como en cualquier recompilación. No cambian la huella de parámetros, la representación ni los valores calculados.
