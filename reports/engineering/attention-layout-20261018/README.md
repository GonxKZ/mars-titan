# Disposición explícita de QKV al congelar parámetros

La opción `attention_layout="sequence_contiguous_v1"` elimina la diferencia causada por los flags `requires_grad` en la reproducción comprobada de la mezcla PT1+PT5+PT6. Conserva exactamente las 15.936 predicciones originales de `cell7-42`, con parámetros congelados y sin congelar, sin pasos de optimizador ni cambios en los valores de los parámetros compartidos. Los pesos rápidos siguen la actualización asociativa habitual.

El valor predeterminado continúa siendo `batch_first_v1`. Se omite de las identidades para conservar los checkpoints y la ruta histórica. La opción nueva aparece en las identidades de MAC y del predictor, se transmite al lector y distingue sus estados y checkpoints. Las mediciones originales no se sustituyen por reproducciones con la disposición nueva.

## Localización de la diferencia

La primera diferencia aparece en `mac.attention`, en la primera observación del calentamiento del 3 de enero de 2022. Sus entradas y sus parámetros coinciden. La salida difiere como máximo en `2,682209e-7`. La recurrencia posterior amplifica esa perturbación y termina con una diferencia media absoluta de `0,00761683` y máxima de `0,06760360` en las predicciones congeladas de 2023.

En la versión instalada de PyTorch, la proyección QKV sin sesgo recibe una vista no contigua de forma `[6, 64, 64]` y strides `[64, 384, 1]`. Con pesos que requieren gradiente usa `mm`, con una copia intermedia. Con los pesos congelados usa `bmm`, incluso dentro de `no_grad`. El [código de `should_fold`](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/LinearAlgebra.cpp) contiene esa decisión. El [recibo de la primitiva](attention_primitive.json) registra las operaciones observadas en esta GPU.

La opción prepara `packed.transpose(0, 1).contiguous().transpose(0, 1)` antes de llamar a `MultiheadAttention`. La transposición interna recibe entonces una secuencia contigua y ambas configuraciones usan `mm`. No cambia los valores, el orden de los tokens, la máscara triangular, el número de actualizaciones ni los pesos. En el caso aislado, la proyección nueva coincide bit a bit con la ruta original que requiere gradiente. Su error máximo frente a la referencia FP64 es `7,957681e-7`, frente a `1,929879e-6` de la ruta congelada anterior. FP64 es una referencia más precisa, no una certificación numérica.

La prueba anterior con `TORCH_LINEAR_FLATTEN_3D=1` no resolvía este caso. Ese control pertenece al [recorrido lineal con sesgo](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/Linear.cpp), mientras que QKV en MAC no tiene sesgo. También se conserva el resultado negativo de quitar el grafo de las proyecciones locales.

## Verificación y coste

El [recibo de reproducción](replay.json) comprueba el camino implementado, después de la sonda inicial con hooks. La [frontera observada](attention_boundary.json) identifica el primer módulo distinto. El diagnóstico conserva filas, checkpoint de origen, precisión, huellas de implementación y parámetros intactos.

Las pruebas comprueban CPU/CUDA, FP32/FP64, modos online, frozen y disabled, gradientes a través de la atención y la pérdida asociativa, preservación de RNG, identidad histórica, recuperación, mezcla completa, PT4 y C. La perturbación de observaciones futuras conserva la predicción ya emitida. El lector reconstruye el padre con la nueva opción y recupera su estado tras un corte. Las [invocaciones](verification.json) se solapan y no se suman como pruebas distintas.

La [mutación dirigida](mutations.json) detecta nueve fallos de nueve probados, incluidos ignorar la disposición, alterar el valor predeterminado, perder la identidad y omitir su propagación. La fuente restaurada vuelve a pasar. El [informe de cobertura y CRAP](quality.json), con coverage.py 7.16.1 y Radon 6.0.1, registra 848 de 976 sentencias y 258 de 370 ramas en cinco módulos. CRAP usa `C² × (1 − cobertura de sentencias)³ + C`. Su máximo es 92,012 y quedan rutas previas sin cubrir. Estas cifras no prueban ausencia de fallos ni representan cobertura del repositorio completo.

Se repitió tres veces la reproducción completa, rotando el orden de las rutas. Cada tiempo incluye el proceso, la carga, el calentamiento y las predicciones de la ventana. Los valores son segundos.

| Ruta | Repetición 1 | Repetición 2 | Repetición 3 | Mediana |
| --- | ---: | ---: | ---: | ---: |
| Histórica, flags activos | 22,42 | 13,20 | 14,05 | 14,05 |
| Histórica, congelada | 16,66 | 15,90 | 12,80 | 15,90 |
| Contigua por secuencia, congelada | 17,21 | 12,99 | 12,54 | 12,99 |

La mediana de la ruta nueva es un 18,3 % menor que la congelada anterior, pero tres repeticiones con esta dispersión no acreditan una aceleración general. Hubo comprobaciones CPU concurrentes durante parte de la medición. La primera repetición de la ruta nueva fue más lenta que la congelada anterior. Los [recibos completos](timing-results.json) separan tiempo de proceso y cronómetro interno. Estas cifras no son percentiles de latencia por activo.

El pico de memoria asignada por PyTorch pasó de 93.999.616 a 94.097.920 bytes frente a la ruta congelada anterior, 98.304 bytes adicionales. Se usaron `cuda:0`, RTX 4070 Laptop, PyTorch 2.14.0+cu130, FP32 estricto y TF32 desactivado. No se cambiaron drivers, potencia ni aplicaciones ajenas. No se midió energía.

## Alcance y reproducción

La corrección elimina esta dependencia del despacho en la configuración, GPU y trayectorias comprobadas. No demuestra contracción de la memoria, equivalencia bit a bit entre equipos, insensibilidad a toda perturbación ni mejora predictiva. Los resultados FP64 y FP32 todavía pueden diferir. Las comparaciones futuras deben identificar la disposición elegida y conservar sus controles.

`uv run --no-sync python -m benchmarks.combined_memory_numerics --help` describe la reproducción. `--attention-layout sequence_contiguous_v1` declara la intervención después de cargar el checkpoint original. `--route fp32_tracked` y `--route fp32_frozen` distinguen los flags. `--work` recibe la preparación del estudio combinado y `--output` necesita un destino nuevo. Se crea un estado rápido nuevo y se registra `diagnostic_attention_layout`. No se reinterpreta un checkpoint de recuperación con otra identidad.

La implementación está en `932d8ab02`. La [ronda combinada anterior](../../research/combined-memory-20261018/README.md) conserva sus resultados y el diagnóstico inicial. La campaña completa y el test final permanecen cerrados.
