# Tabla de palabras congelada en CPU

`FrozenEncoders(word_embedding_placement="cpu")` mantiene la tabla de palabras FP32 de MiniLM en CPU. Consulta los IDs del lote antes de trasladarlo y entrega `inputs_embeds` a Bert. La suma de posiciones, la normalización, las capas de MiniLM y ResNet18 siguen en `cuda:0`. Se conservan pesos, tokenizador, fragmentos completos, máscaras y agregación. La opción predeterminada es `"cuda"` y ejecuta la consulta original en GPU.

La CLI incorpora `--word-embedding-placement`. La ubicación y el identificador `cpu_word_inputs_embeds_fp32_v1` forman parte de la especificación. También se registra el SHA del módulo de consulta. Las cachés y ediciones distinguen ambas ubicaciones y las versiones de código. El cambio no cuantiza pesos ni permite extraer representaciones sin CUDA.

La especificación registra los flags efectivos de TF32 para matmul y cuDNN, sin modificarlos. Se contrastan al construir y antes de cada llamada a texto o imágenes. Cambiarlos exige otra identidad. El constructor rechaza un dtype global distinto de FP32, comprueba los tipos de pesos y buffers y crea las constantes de imagen explícitamente en FP32. Un contexto autocast activo se rechaza. La comparación medida utiliza matmul TF32 desactivado y cuDNN TF32 activado, como la referencia anterior.

Antes de cada modalidad también se comprueban el modo de evaluación de todos sus módulos, la congelación de parámetros y los tipos FP32. Activar un Dropout interno o convertir el modelo a FP64 provoca un error antes de preparar sus entradas. Esta comprobación recorre metadatos de los tensores, sin copiarlos ni calcular un hash de los pesos por llamada.

La tabla debe estar congelada, en evaluación, en CPU y en FP32. Se rechaza `max_norm`, que podría modificar filas durante la consulta. Solo se admiten IDs int64 en CPU y consultas de hasta 32 fragmentos de 128 tokens. La inferencia desactiva gradientes. Si falla el traslado del modelo, se restaura su referencia a la tabla. Las pruebas verifican orden, repeticiones, padding, ausencia de alias mutable y errores de tipo, dispositivo y presupuesto.

## Memoria y alcance medido

La tabla fijada tiene forma `[250037, 384]` y ocupa 384.056.832 bytes. El resto de los tensores de MiniLM y ResNet18 ocupa 131.311.008 bytes. Con lotes de ocho fragmentos y dos imágenes, la consulta individual transfiere como máximo 1.572.864 bytes de activaciones. Los IDs no se transfieren en la ruta CPU.

La comparación técnica usa ocho textos reales admitidos y cinco imágenes sintéticas con ambos codificadores residentes. Alterna el orden de las ubicaciones, descarga cada instancia, calienta una vez y mide tres repeticiones sincronizadas. Los límites son 640 MiB de asignador y 768 MiB de reserva inicial para la referencia, y 224 MiB y 320 MiB para la tabla CPU. La reserva se comprueba después de inicializar CUDA. No se modifica si resulta insuficiente.

El pico asignado pasó de 549.734.400 a 165.955.584 bytes y el reservado de 578.813.952 a 192.937.984 bytes. Los cuatro pares medidos con las guardas de precisión dieron entre un 2,2 % y un 4,4 % más de tiempo en la ruta CPU. La comprobación posterior con la guarda completa de modo y dtype mantuvo las salidas exactas y esos picos. Sus medianas fueron 0,1680 s en CUDA y 0,1817 s con tabla CPU. Esta última medida es una pareja focal, no una estimación del coste del corpus.

El [recibo de verificación](../../reports/data/frozen-embedding-placement-20261008.json) separa el prototipo, el constructor final, las pruebas CPU y los fallos conservados. La reducción de VRAM no acredita aceleración del corpus. El tiempo de proceso incluye carga y comprobación de pesos y se distingue de las medidas de inferencia. El asignador de Torch no incluye toda la memoria del contexto CUDA. No se midió energía.

## Identidad de una edición nueva

La canalización actual exige una representación común, incluida la especificación de codificadores, en todos los activos. Una edición existente rechaza la ubicación o el código nuevos. La unión de corpus por mercado tampoco admite mezclar estas identidades. El comportamiento predeterminado se conserva, pero ejecutar código nuevo produce una identidad nueva incluso con `"cuda"`.

La alternativa disponible sin añadir compatibilidad heterogénea es materializar una edición nueva uniforme desde el mismo preparado y conservar el prefijo anterior con sus recibos. Repetir ese prefijo implica volver a calcular sus filas y conservar espacio para ambas salidas. La caché anterior no se hace pasar por caché de la nueva ubicación.

Reutilizar salidas entre identidades exigiría una derivación explícita que verificase los artefactos padres, registrase su procedencia efectiva y fuese admitida por los productores y lectores. Ese contrato no está implementado aquí. No se ha escrito corpus con esta opción ni se ha modificado la edición anterior. La comprobación técnica no habilita aprendizaje.
