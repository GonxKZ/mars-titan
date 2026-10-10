# Recuperación del estado financiero por bloques

`models.titans.financial_blocks` reúne estados persistidos para un lote del predictor y actualiza un índice de referencias. El censo puede contener hasta 8192 flujos. Cada estado tensorial sigue limitado a `FinancialConfig.max_batch`, con un máximo de 256. El módulo no carga archivos, publica checkpoints ni poda artefactos.

## Transporte y reunión

`predictor.export_state_cpu(state)` conserva el esquema de `export_state` y devuelve tensores CPU propios, contiguos y sin grafo. Los IDs y cortes se copian como tuplas. La precisión, configuración, parámetros, modo train/eval, momentum y contadores permanecen identificados. `export_state` conserva su comportamiento anterior. Con las proyecciones de la sección 4.4, el payload añade el campo `convolution` en la memoria y en MAC con las ventanas causales de cada flujo, que se copian, validan y reúnen igual que los pesos rápidos. Sin convolución los campos son los de siempre.

`predictor.restore_state(payload)` sigue exigiendo el dispositivo anterior de los tensores rápidos. `restore_state(payload, device=...)` admite un payload CPU y exige que el dispositivo solicitado coincida exactamente con el del predictor. No mueve el predictor ni convierte la precisión. Valida toda la fuente en CPU antes de trasladar W, momentum y steps. `observed_steps` permanece en CPU. Ambas rutas devuelven copias sin grafo y requieren haber restaurado antes los parámetros compartidos.

`predictor.gather_state(blocks, flow_ids, max_source_bytes=512*1024**2)` recibe un diccionario con exactamente los IDs pedidos. `flow_ids` debe ser una lista o tupla cuyo tamaño se valida antes de copiarla. Cada entrada contiene `(payload_CPU, fila)`. Varias entradas pueden compartir el mismo objeto payload, que se valida y contabiliza una sola vez. La fila debe contener el mismo ID pedido. Esta selección explícita permite que dos artefactos anteriores contengan un activo y que el índice indique cuál es su versión vigente.

La reunión comprueba esquema, identidades, cursores, tipos, formas, dispositivo, finitud y almacenamiento de cada bloque único. Rechaza fuentes con grafo, IDs desconocidos o repetidos y filas que no correspondan al ID pedido. Reúne las filas en el orden solicitado en CPU y después transfiere únicamente ese resultado al dispositivo del predictor. No restaura bloques completos en CUDA para seleccionar una fila. Los flujos nuevos deben proceder de `initial_state` explícito, sin fallback ni reinicio automático.

`max_source_bytes` admite entre 1 byte y 2 GiB y limita la suma de almacenamiento tensorial retenido y metadatos canónicos de las fuentes únicas. Dos objetos payload distintos se cuentan por separado aunque compartan almacenamiento. El resultado debe caber en `max_state_bytes`, antes de reunir sus tensores. Se aplican también los límites individuales de cada fuente. No se truncan flujos para cumplir el presupuesto.

Estas cuentas no son una cota de RSS. Los payloads ya están cargados por el consumidor. La validación necesita temporales y, durante un traslado, pueden coexistir las fuentes, el resultado CPU y su copia CUDA. Los parámetros del predictor y el almacenamiento de archivos tienen presupuestos independientes.

## Índice de referencias

`replace_state_references(predictor, references, previous, following, block_id)` devuelve un índice nuevo. Cada registro contiene `block_id`, `row`, `config_id`, `parameter_id`, `observed_steps`, `last_prediction_at` y `last_sample_id`. `block_id` tiene formato SHA256. Dos IDs no pueden apuntar a la misma fila de un bloque.

`previous` y `following` deben contener los mismos IDs y orden. Sus identidades y contadores se validan con el predictor. El estado anterior debe coincidir con los cursores del índice. Cada flujo avanza exactamente una observación y su corte debe ser posterior. Un flujo sin referencia previa solo se admite desde un estado inicial con contador cero y cursor vacío. El resultado reemplaza exactamente esos IDs y conserva las referencias ausentes, sin compartir diccionarios mutables con la entrada.

El índice admite hasta 8192 registros y un presupuesto de 16 MiB para sus metadatos. La cuenta suma los tamaños de contenedores, claves y valores con `sys.getsizeof` y su representación JSON canónica. Cuenta por separado valores repetidos y no estima el RSS de la validación. El índice no contiene W, momentum ni tensores de todo el censo.

El consumidor debe verificar el SHA del archivo confirmado y contrastar la fila, identidad y cursores de cada referencia con el payload recuperado antes de llamar a `gather_state`. El formato SHA256 por sí solo no prueba esa correspondencia. Al publicar un checkpoint, su manifiesto debe conservar las referencias anidadas vivas y también las del checkpoint anterior. Esta ampliación de publicación y poda corresponde al coordinador y no está implementada en este módulo.

El plan C se crea una sola vez sobre el grupo lógico completo, con la identidad del checkpoint y del grupo. Cada bloque recibe la misma selección y el mismo denominador. La reunión no crea otro plan ni ejecuta MAC. La lectura episódica posterior a MAC y K tampoco deben repetir esa escritura.

## Comprobación técnica

Las pruebas CPU comprueban transporte, almacenamiento propio, tipos incompatibles, NaN/Inf, presupuestos anteriores a la copia, cursores, reemplazo parcial y un censo ficticio de 5676 flujos en 23 bloques. Pasan 373 casos distintos, 54 propios del puente. La API sin argumentos nuevos conserva 32 pares exactos de predicciones, estados e identidades frente a la base. Se detectaron 13 mutaciones dirigidas. El [recibo](../../reports/research/financial-state-blocks-verification-20261008.json) recoge cobertura, complejidad, mediciones y límites. No se utiliza un corpus real ni un optimizador.

`tests/models/titans/cuda_financial_blocks_check.py` es una comprobación manual de los cuatro controles con FP32 y FP64. Los ocho casos pasaron en una RTX 4070 Laptop. La exportación CPU, restauración explícita en CUDA y reunión de una fila conservan exactamente el estado, la siguiente predicción y el gradiente de la cabeza. Los RNG permanecen iguales. La traza confirma transferencias de una sola fila y el pico Torch fue de 20.402.176 bytes, con un límite de 128 MiB. Apareció un aviso cuBLAS de creación del contexto primario durante backward. No se atribuye a esta comprobación una medida aislada de rendimiento GPU ni paridad de todos los gradientes del predictor.
