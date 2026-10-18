# Recuento de metadatos del estado de Titans

El recuento de bytes reutiliza ahora el tamaño de la parte invariable de los metadatos. En cuatro repeticiones del mismo microajuste, la mediana del proceso completo baja de 34,38 a 32,22 segundos, un 6,3 %. Los tensores del modelo y del optimizador seleccionados y las predicciones de las tres fases coinciden exactamente con el ajuste original. Es una medida con cuatro activos, sin extrapolación al universo completo.

La referencia es `832f32c5` y el cambio es `2b556622d`. El [recibo](verification.json) conserva medidas individuales, percentiles, huellas, comprobaciones y fallos. La tarea [#62](https://github.com/GonxKZ/mars-titan/issues/62) sigue abierta para completar el perfil y presupuesto de la campaña.

## Cambio y límites

`FinancialPredictor._metadata_bytes` conserva como máximo dos enteros, uno por modo train/eval. No retiene flujos, entradas ni estados anteriores. Sigue serializando los tres campos variables con `canonical`. El resto se calcula con esos campos vacíos y se descuenta el tamaño de su objeto vacío. Así se obtiene el mismo número de bytes que serializando el objeto completo, incluidos escapes y separadores. Una carga, un cambio de tipo o dispositivo y el sellado tras una actualización invalidan la caché.

El cambio no modifica operaciones numéricas, límites, orden de etiquetas ni el esquema de los checkpoints. El presupuesto cuenta el almacenamiento retenido por los tensores. Se comprueba que el límite exacto se admite y que un byte menos se rechaza. Los contratos de configuración permanecen inmutables, como exige ya la caché de identidad del predictor.

## Medidas

El equipo es una RTX 4070 Laptop de 8 GiB, con driver 595.91.07, PyTorch 2.14.0+cu130 y CUDA 13.0. Se usa `cuda:0`, FP32 estricto, dos hilos para PyTorch y BLAS y una carga propia de GPU. El perfil energético `power-saver` permanece sin cambios.

El perfil posterior repite el recorrido corto inicial. La reconstrucción completa de metadatos pasa de 290 a nueve llamadas. Sus tiempos instrumentados se conservan por separado de los benchmarks y no se usan para calcular aceleración.

| Medida | Referencia | Cambio | Reducción del tiempo |
| --- | ---: | ---: | ---: |
| Tramo cronológico, mediana de seis procesos por versión | 5,406 s | 5,166 s | 4,4 % |
| Mismo tramo, seis recorridos por versión dentro de un proceso | 5,651 s | 5,219 s | 7,7 % |
| Ajuste completo, mediana de dos procesos por versión | 34,376 s | 32,222 s | 6,3 % |

Los tramos cronológicos alternan A B B A tres veces. Cada medida recorre 1.280 observaciones y etiquetas maduras, con lectura, transferencia, forward y backward, tras cuatro segmentos de calentamiento. El optimizador sustituto conserva parámetros y solo registra llamadas. La primera serie tiene una tanda desfavorable al cambio. En la segunda, las tres tandas favorecen el cambio, con cocientes de tiempo entre 1,035 y 1,122. Los percentiles por segmento y todas las repeticiones están en el recibo. La variación entre procesos limita la precisión del porcentaje.

El ajuste completo repite `mac_online-42` del [microensayo](../../research/memory-microstudy-20261018/README.md) en orden A B B A, con su mismo manifiesto, receta, semilla y ventana. Cada repetición hace 119 actualizaciones, selecciona y guarda checkpoints y emite validación, calibración y evaluación. Son 476 actualizaciones de verificación, no cuatro experimentos científicos nuevos. Las identidades de índice cambian porque contienen la ruta del manifiesto fuente. Se comprueba que el contenido del índice y los datos coinciden al excluir únicamente esa ruta, y que el resto de los metadatos del modelo es idéntico.

El [benchmark del recuento](../../../benchmarks/titans_metadata_sizing.py), con identidades sintéticas y sin cálculo del modelo, baja la mediana de 82,02 a 4,65 µs para cuatro flujos, de 107,54 a 29,15 µs para 128 y de 269,34 a 192,39 µs para 1.024. Estas cifras solo describen el trabajo de serialización en CPU.

Los cuatro ajustes completos registran el mismo pico de PyTorch, 79.180.288 bytes asignados y 79.691.776 reservados. Eso no incluye toda la VRAM del proceso. El RSS varía entre 1.857.474.560 y 1.866.731.520 bytes y no acredita ahorro de RAM. No se midieron energía ni contadores de hardware. Tampoco se midió el efecto del cambio en una campaña completa, otras arquitecturas o miles de activos.

## Comprobaciones y reproducción

Pasan 146 casos únicos de contrato y recuperación, incluidos veinte nuevos y comprobaciones CUDA en FP32 y FP64. Las capturas cronológicas comparan exactamente 1.802 tensores de gradientes en 45 llamadas al optimizador sustituto. Las cuatro repeticiones reales conservan todos los tensores seleccionados de modelo y optimizador y las predicciones de validación, calibración y evaluación. Cuatro mutaciones del recuento quedan detectadas y el control sin mutar pasa.

Coverage.py 7.16.1 y Radon 6.0.1 registran las siete sentencias y las dos ramas de `_metadata_bytes`, con complejidad y CRAP de 2. El archivo completo tiene 369 de 457 sentencias y 124 de 194 ramas cubiertas por esta selección de pruebas. Estas medidas no demuestran ausencia de errores.

Se conservan los fallos iniciales de configuración de rutas y de importación de PyTorch durante la recogida de cobertura por nombre de módulo. La recogida posterior por ruta pasa. También se mantienen los avisos de PyTorch sobre su diagnóstico de sincronización y la creación del contexto CUDA en autograd. No se han suprimido comprobaciones para obtener resultados positivos.

`benchmarks/chronological_replay.py` reproduce el tramo con `--family titans --arm titans_mac_online --strict --start-fraction 0.5 --warmup 4 --segments 40`, además de las rutas de vista y trabajo. Ahora conserva también los tiempos individuales de segmento. `benchmarks/titans_metadata_sizing.py` recibe el destino mediante `--output` y mide el componente con veinte repeticiones de calentamiento y doscientas medidas por tamaño. Los ajustes completos usan el ejecutor del microensayo, copias de sus manifiestos y destinos independientes. La campaña completa y el test final permanecen cerrados.
