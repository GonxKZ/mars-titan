# Divergencia de la memoria de Titans-MAC entre CPU, CUDA y precisiones (9 de octubre de 2026)

Las comprobaciones CUDA de #374 observaron que, con residual y LayerNorm en la memoria, las trayectorias de MAC en CPU y CUDA se separan con el número de pasos (1,7 % de la norma de salida a 2.048 pasos y 4,2 % a 4.096), aunque CUDA repite bit a bit. Este estudio comprueba si la causa es un defecto del cálculo o la sensibilidad propia de la recurrencia, y qué precisión es aceptable para la memoria de Titans en la campaña. Solo ejecuta pasos hacia delante de MAC y la derivada interna de su regla asociativa. No crea optimizadores, no usa etiquetas y no cambia parámetros. No mide utilidad predictiva.

El [recibo](titans-divergence.json) contiene todas las cifras. El script es [`benchmarks/titans_divergence.py`](../../../benchmarks/titans_divergence.py) y sus pruebas en CPU están en `tests/models/titans/test_titans_divergence_benchmark.py`.

## Montaje

- MAC de la receta de campaña: dimensión 64, memoria de profundidad 2, cuatro cabezas, cuatro tokens persistentes, segmento de un token y semilla 42. El candidato `gate_bias_residual_layer_norm` coincide con el MAC inicial de `FinancialPredictor` de `configs/titans/chronological-training-historical-masked.json` (la extracción lo comprueba tensor a tensor). Como contraste se usan `gate_bias` sin residual ni LayerNorm y `v1_residual_layer_norm` sin bias de puertas.
- Cuatro flujos por recorrido. Tres corrientes sintéticas de `titans_gate_retention.py` (`fused_norm_1`, `unit_variance` y `recurring_unit`) de 4.096 pasos y una corriente real de 3.820 pasos: los tokens fusionados de los cuatro activos con más observaciones de ajuste de la vista CN fold-012 (`CN/600867.SS`, `CN/600868.SS`, `CN/600871.SS` y `CN/600893.SS`), calculados con el codificador inicial de `FinancialPredictor`. La extracción exige que el token reconstruido sea idéntico al de `prepare`. Su norma media es 1,82.
- Estudio de reproducción: parámetros y tokens FP64 como `titans_mac_output_scale.py`. Reproduce los dos recibos anteriores: CPU idéntico bit a bit y CUDA con diferencias de la norma de salida de hasta 2,3·10⁻¹⁶, en los 45 puntos comparados.
- Estudio de precisión: parámetros iniciales y tokens redondeados a FP32 y comunes a todas las precisiones, de modo que solo cambia la aritmética. La referencia es CPU FP64. Se comparan CUDA FP64, CPU FP32, CUDA FP32 con TF32 desactivado, CUDA con TF32 en matmul y cuDNN, y CUDA con BF16 emulado.
- BF16 emulado: el núcleo rechaza autocast, así que los operandos y el resultado de cada `linear`, `bmm` y SDPA se redondean a bfloat16 y el resto (LayerNorm, normalize, GELU, puertas y actualización) se calcula en FP32. Es una cota optimista del error de un autocast real.
- Exponente de Lyapunov máximo de la transición rápida en FP64 con el método de Benettin: perturbación inicial δ₀ = 10⁻⁸·‖z₀‖ del estado completo (pesos rápidos y momentum), renormalización cada 8 pasos, tres direcciones por flujo. Además, crecimiento libre sin renormalizar desde perturbaciones de 10⁻¹⁵ y 10⁻⁷.
- Piso de invariancia por filas: copias exactas de cada flujo en el mismo lote (12 filas), en CPU y CUDA, FP64 y FP32.

Entorno: RTX 4070 Laptop GPU, PyTorch 2.14.0+cu130 (CUDA 13.0, cuDNN 9.24), Python 3.12.14, dos hilos, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, perfil `power-saver` sin cambios y CPU compartida con otras cargas. Cada orden CUDA se ejecutó con el candado de GPU. Los tiempos no son medidas de rendimiento.

## Sensibilidad propia de la recurrencia

| Corriente | λ `gate_bias_residual_layer_norm` | λ `gate_bias` | λ `v1_residual_layer_norm` |
| --- | ---: | ---: | ---: |
| `fused_norm_1` | 0,0153 [0,0144, 0,0161] | −0,00073 | 0,604 |
| `unit_variance` | 0,0068 [0,0066, 0,0071] | −0,00066 | 0,477 |
| `recurring_unit` | 0,0133 [0,0125, 0,0137] | −0,00022 | 0,429 |
| real CN | 0,0370 [0,0311, 0,0403] | −0,00065 | 0,576 |

λ es el exponente máximo por paso en FP64, con el mínimo y el máximo por flujo y dirección entre corchetes. Sin residual ni LayerNorm la memoria contrae y los errores no crecen. Con residual y LayerNorm la transición es caótica: un error relativo u crece como u·e^{λt}.

La separación entre CPU FP64 y CUDA FP64 sigue exactamente ese crecimiento:

| Corriente | Pendiente CPU/CUDA FP64 | 1 % CPU/CUDA FP64 | 1 % desde perturbación FP64 de 10⁻¹⁵ |
| --- | ---: | ---: | ---: |
| `fused_norm_1` | 0,0174 | 1.834 | 1.841 |
| `unit_variance` | 0,0072 | 3.947 | 3.909 |
| `recurring_unit` | 0,0119 | 2.087 | 2.087 |
| real CN | 0,0553 | 483 | 483 |

Las columnas de 1 % indican el primer paso en que la distancia relativa de la salida alcanza el 1 %. Una perturbación de 10⁻¹⁵ en un recorrido FP64 exacto alcanza el 1 % prácticamente en el mismo paso que la separación CPU/CUDA, que parte del redondeo de FP64. Con `gate_bias` sin LayerNorm, CUDA FP64 sigue a 1,2·10⁻¹⁵ o menos de CPU durante los 4.096 pasos.

## Distancia a FP64 por precisión (candidato de campaña)

| Precisión | Error relativo de salida a 64 pasos (máximo de 4 corrientes) | Paso en que alcanza 1 % (sintéticas y real) | Horizonte previsto ln(10⁻²/u)/λ en `fused_norm_1` y real |
| --- | ---: | --- | --- |
| CUDA FP64 | 1,1·10⁻¹⁴ | 1.834, 3.947, 2.087 y 483 | 2.094 y 869 |
| CPU FP32 | 5,0·10⁻⁶ | 612, 1.070, 590 y 227 | 784 y 325 |
| CUDA FP32 | 7,7·10⁻⁶ | 612, 1.131, 536 y 224 | 784 y 325 |
| CUDA TF32 | 2,2·10⁻³ | 242, 393, 302 y 166 | 197 y 82 |
| CUDA BF16 emulado | 3,0·10⁻² | 2, 2, 2 y 1 | 61 y 25 |

CPU y CUDA en FP32 se alejan de FP64 al mismo ritmo y en el mismo paso, dentro de la variación entre corrientes. Ninguno de los dos dispositivos es más exacto que el otro: ambos son trayectorias de redondeo distintas de un sistema caótico, y más allá del horizonte ninguna trayectoria calculada, tampoco la FP64 de CPU, sigue a la solución exacta. La comparación útil a largo plazo es estadística.

Con `gate_bias` sin LayerNorm, los errores quedan acotados durante todo el recorrido: FP32 hasta 3,1·10⁻⁶ en el estado, TF32 entre 5·10⁻⁴ y 3·10⁻³ en la salida y BF16 emulado entre 0,7 % y 2 % en la salida.

### Clima del recorrido

Norma media de la salida y percentil 90 de |salida| en la segunda mitad del recorrido. La dispersión FP64 es la mayor diferencia de clima entre trayectorias FP64 exactas que solo difieren por una perturbación inicial de 10⁻⁷, con dos direcciones. Es una vara de medir ruidosa: la pareja CPU FP64 y CUDA FP64 da otra muestra de la misma dispersión.

| Corriente | Dispersión FP64 | CUDA FP64 | CPU FP32 | CUDA FP32 | CUDA TF32 | CUDA BF16 emulado |
| --- | --- | --- | --- | --- | --- | --- |
| `fused_norm_1` | 0,38 % y 0,17 % | 0,48 % y 0,04 % | 0,23 % y 0,48 % | 0,39 % y 0,03 % | 0,18 % y 0,13 % | 0,95 % y 0,30 % |
| `unit_variance` | 0,21 % y 0,44 % | sin separar | 0,04 % y 0,06 % | 0,06 % y 0,06 % | 0,12 % y 0,07 % | 0,08 % y 0,18 % |
| `recurring_unit` | 0,54 % y 0,60 % | 0,49 % y 0,42 % | 0,84 % y 0,61 % | 0,01 % y 0,05 % | 0,06 % y 0,39 % | 0,74 % y 1,20 % |
| real CN | 12 % y 9 % | 4,1 % y 0,3 % | 5,7 % y 0,6 % | 11 % y 6,1 % | 3,5 % y 7,7 % | 0,002 % y 2,7 % |

FP32 y TF32 quedan dentro de la dispersión de FP64 o cerca de ella. BF16 emulado llega a entre dos y dos veces y media la dispersión en `fused_norm_1` y `recurring_unit`. CPU FP32 también la supera 1,6 veces en una medida de `recurring_unit`, así que la diferencia de BF16 es moderada. Con la corriente real, la dispersión entre trayectorias FP64 ya es del 9 al 12 %, porque cuatro flujos y unos 1.900 pasos dan medias poco estables, así que esa corriente no distingue precisiones por su clima.

## Defectos buscados

- **Orden de reducción.** CPU y CUDA usan núcleos distintos y redondean distinto, pero en FP64 quedan por debajo de 10⁻¹³ durante los primeros 256 pasos de las corrientes sintéticas. La separación posterior crece con la pendiente del exponente FP64 y no aparece sin LayerNorm. No hay indicios de una reducción defectuosa.
- **Invariancia por filas.** Con las formas del estudio (12 filas, dimensión 64) las copias exactas de un flujo no se separan nunca en CPU ni en CUDA, en FP64 ni en FP32. En la prueba unitaria, con dimensión 8 y 6 filas en CPU, sí se separan por debajo de 10⁻¹³, porque la GEMM y SDPA de CPU redondean según la posición en el lote. Con una transición caótica, un cambio de composición de bloques o de tamaño de lote puede llevar a otra trayectoria tras unos cientos de observaciones, igual que un cambio de dispositivo.
- **LayerNorm mal condicionada.** La LayerNorm de la memoria no tiene afinidad y normaliza MLP(x). Cuando los pesos rápidos son pequeños, la varianza de su entrada se acerca a ε = 10⁻⁵ y la ganancia de su Jacobiano, 1/√(var + ε), amplifica cualquier perturbación hasta 316 veces. La tabla relaciona esa varianza con λ:

| Recorrido | Mediana de la varianza de entrada de LN | Llamadas con varianza < 10⁻⁴ | Ganancia máxima | λ |
| --- | ---: | ---: | ---: | ---: |
| campaña, `unit_variance` | 3,6·10⁻³ | 0 de 12.288 | 49 | 0,0068 |
| campaña, `fused_norm_1` | 1,3·10⁻³ | 24 de 12.288 | 117 | 0,0153 |
| campaña, `recurring_unit` | 6,9·10⁻⁴ | 76 de 12.288 | 165 | 0,0133 |
| campaña, real CN | 3,2·10⁻⁴ | 2.839 de 11.460 | 313 | 0,0370 |
| `v1_residual_layer_norm` (reproducción) | 2,0·10⁻⁵ a 4,8·10⁻⁵ | 7.705 a 8.806 de 12.288 | 316 | 0,43 a 0,60 |

Cuanto menor es la varianza de entrada de la LayerNorm, mayor es el exponente. Es un mal condicionamiento propio de la forma M(x) = x + LN(MLP(x)) de la sección 3.3 de las actas con pesos rápidos de escala pequeña, no un fallo de la implementación. Esta revisión no cambia el núcleo. Si se quisiera reducir la sensibilidad, habría que estudiar como variante identificada otro ε, una LN con afinidad o la escala inicial de los pesos rápidos.
- **normalize y subnormales.** La norma mínima de entrada de `F.normalize` fue 0,037 en todos los recorridos, lejos de su ε de 10⁻¹². No se midieron subnormales directamente. Las varianzas mínimas de entrada de la LayerNorm, 2,3·10⁻⁷ con el candidato de campaña y 1,5·10⁻¹¹ con `v1_residual_layer_norm`, quedan muy lejos del rango subnormal de FP32.

## Conclusión

La separación entre CPU y CUDA es la sensibilidad propia de la memoria online con residual y LayerNorm, no un defecto. Ambos dispositivos se alejan de FP64 al mismo ritmo, el que fija el exponente de Lyapunov, y en FP64 su separación reproduce la de una perturbación de 10⁻¹⁵. Con los tokens reales iniciales el exponente es 0,037 por paso, así que una trayectoria FP32 pierde la coincidencia del 1 % con FP64 tras unas 220 observaciones del flujo. Las predicciones de Titans online solo son reproducibles bit a bit con la misma configuración de dispositivo, precisión, núcleos y composición de bloques. CUDA lo cumple.

Las cifras corresponden a parámetros iniciales. El ajuste cambiará la escala de los pesos rápidos y, con ella, el exponente. No se ha medido después de entrenar porque el bloqueo de aprendizaje sigue vigente.

## Recomendación para la campaña

- **FP32 estricto en la memoria de Titans** (`allow_tf32=False` en matmul y cuDNN) como modo de referencia y recomendado. Es la única opción con error de salida por debajo de 10⁻⁵ a 64 pasos, que es la escala del BPTT de 8 instantes de la receta.
- **TF32** queda dentro de la dispersión de FP64 en el clima, pero su error a 64 pasos es de 1,4·10⁻³ a 2,2·10⁻³ y supera el 0,1 % desde el segundo o tercer paso. Solo es admisible como opción identificada, igual en todos los brazos que comparten la ruta, si su ganancia de caudal medida lo justifica.
- **BF16 no es aceptable en la actualización de la memoria ni en su lectura.** Su error alcanza el 1 % en uno o dos pasos y el 3 % a 64 pasos, incluso con la emulación optimista, y su clima llega a entre dos y dos veces y media la dispersión FP64 en dos corrientes. Este estudio no evalúa BF16 en las partes sin estado (codificador de precios y proyecciones de modalidades).
- **Tolerancias declaradas.** Las comparaciones punto a punto de la memoria online deben limitarse a horizontes cortos. A 64 pasos con el candidato de campaña: FP64 entre dispositivos 10⁻¹³ (observado hasta 1,1·10⁻¹⁴), FP32 frente a FP64 10⁻⁵ (observado hasta 7,7·10⁻⁶), TF32 5·10⁻³ (observado hasta 2,2·10⁻³) y BF16 emulado 5·10⁻² (observado hasta 3,0·10⁻²). Para recorridos largos solo tiene sentido comparar estadísticas, con la dispersión FP64 como referencia.

## Órdenes

```bash
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
B=benchmarks/titans_divergence.py
uv run --no-sync python $B extract --view <vista CN fold-012>/manifest.json \
  --recipe configs/titans/chronological-training-historical-masked.json --work <índices> --output <tokens.pt>
uv run --no-sync python $B trajectories --cache <caché> --precisions cpu_float64 cpu_float32
uv run --no-sync python $B trajectories --cache <caché> --real-tokens <tokens.pt> --studies precision \
  --precisions cpu_float64 cpu_float32
flock -x <candado GPU> uv run --no-sync python $B trajectories --cache <caché> [--real-tokens <tokens.pt>] \
  --precisions cuda_float64 cuda_float32 cuda_tf32 cuda_bf16_emulated
uv run --no-sync python $B lyapunov --cache <caché> --real-tokens <tokens.pt>
uv run --no-sync python $B floor --cache <caché> --real-tokens <tokens.pt>
flock -x <candado GPU> uv run --no-sync python $B floor --device cuda:0 --cache <caché> --real-tokens <tokens.pt>
uv run --no-sync python $B report --cache <caché> --real-tokens <tokens.pt> --output titans-divergence.json
```

Las siete pruebas de `tests/models/titans/test_titans_divergence_benchmark.py` pasan en CPU en unos 2 s y detectan las nueve mutaciones dirigidas probadas (orden de capas del estado, redondeo y restauración de la emulación BF16, base y renormalización de Benettin, pendiente, umbrales, redondeo de tokens y parámetros del estudio de precisión).
