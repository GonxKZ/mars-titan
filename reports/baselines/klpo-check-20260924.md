# Comprobación de KLPO del 24 de septiembre de 2026

Se completaron 36 ajustes del adaptador sobre un padre Ridge congelado y la
edición histórica de 470 muestras. Esta ejecución comprueba integración,
recuperación y registro. No sustituye la comparación sobre el corpus completo.

Cada ajuste recorrió cinco épocas de 245 muestras de entrenamiento. La validación
tuvo 225 muestras. Hubo 180 pasadas completas y 44.100 visitas de entrenamiento
entre los 36 ajustes. Como cada partición de entrenamiento cabe en un lote,
cada ajuste realizó cinco actualizaciones. No se abrió el test final ni se
añadieron muestras sintéticas al corpus.

Los [resultados por ejecución](klpo-check-20260924.json) conservan configuración,
huellas, checkpoint seleccionado, errores y recursos. El diseño incluye tres
semillas y los controles REINFORCE, pérdida esperada y MAE. Las variantes KLPO
Full-KL, MC y varianza exacta usan β de 0,03, 0,1 y 0,3.

## Error observado

El MAE por sesión del padre fue 0,0321803 y el del retorno cero, 0,0147404.
El menor MAE por sesión de los 36 ajustes fue 0,0313105, en KLPO MC con semilla
42. En esta ejecución, los tres valores de β dieron esa misma puntuación para
esa semilla. El conjunto de comprobación no permite atribuir una ventaja a una
intensidad de regularización ni afirmar superioridad de KLPO. Ningún ajuste
superó la referencia de retorno cero.

El checkpoint se seleccionó con esa misma validación. Su puntuación no es una
estimación independiente después de seleccionar modelo y parámetros. Tampoco
se ha contrastado significación estadística con una muestra tan pequeña.
La política discreta y el centro continuo se registran por separado para
evitar atribuir a KLPO una diferencia debida a la cuantización.

## Coste de la comprobación

El proceso completo, incluidas preparación, evaluación y escritura, duró
36,80 segundos según GNU `time -v`. El máximo residente fue de 1.844.928.512
bytes, aproximadamente 1,72 GiB. No hubo uso de intercambio registrado.

El máximo asignado por PyTorch en una ejecución fue de 10.351.104 bytes y el
máximo reservado, 25.165.824 bytes. Estas cifras no incluyen toda la memoria
del contexto CUDA, las bibliotecas ni los demás procesos de la GPU. La máquina
tenía otras cargas activas, por lo que estos tiempos no son un benchmark aislado
ni una estimación del coste de 1,5 millones de muestras.

Se usó CUDA en la RTX 4070 Laptop, con parámetros en FP32, probabilidades y
objetivos en FP64 y `CUBLAS_WORKSPACE_CONFIG=:16:8`. El adaptador recibió 1.682
componentes por muestra, incluidas las cuatro modalidades, macro y el padre.
Las versiones exactas y las huellas del código están en el archivo JSON.

## Alcance de las pruebas

Las pruebas sintéticas separadas contrastan el gradiente con enumeración y
diferencias finitas, el óptimo de Gibbs, la independencia del muestreo y una
señal lineal definida. No representan noticias, documentos contables o retornos
observados. Se usan para detectar defectos de implementación.

El [registro de calidad](../resources/klpo-quality.json) recoge 1.387 pruebas
generales correctas y 16 comprobaciones adicionales de boosting. Quedaron sin
ejecutar la extracción real de codificadores y una comprobación de `clang-tidy`
no configurada. Las 45 pruebas nuevas incluyen siete regresiones para fallos de
recuperación y cambios de identidad. Cuatro mutaciones dirigidas fueron
detectadas. La cobertura de los ocho módulos registrados es del 88,07 % de
sentencias y del 74,88 % de ramas. La cobertura y CRAP no demuestran ausencia
de defectos.

La [cola completa](../../docs/engineering/klpo-posttraining.md) exige que terminen
las campañas padre. Sus 216 ajustes previstos conservan la población común
admitida. No se han ejecutado esos 216 ajustes en esta comprobación.
