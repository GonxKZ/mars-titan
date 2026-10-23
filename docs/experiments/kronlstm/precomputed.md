# Ejecución escalar precomputada

`FinancialConfig.temporal_encoder="kronlstm_precomputed_v1"` activa una ejecución
opcional de la misma regla escalar y los mismos parámetros de `kronlstm_v1`.
Precomputa términos que no dependen del estado recurrente. Conserva el orden de
olvido, revisión, escritura y lectura, junto con los diagnósticos de cada paso.
El backend predeterminado sigue siendo `reference`.

`CompactPriceKronLSTM` admite `execution="reference"` y `execution="precomputed"`.
La configuración y los checkpoints de referencia conservan sus campos anteriores.
La ejecución nueva declara `pytorch_precomputed_cpu_qkv64_v1`. Sus cachés y
checkpoints requieren ese mismo backend. Las proyecciones QKV con entrada FP32
acumulan en FP64 en CPU y devuelven FP32, como en la referencia. CUDA conserva la
proyección nativa. Los manifiestos Kron incluyen ahora `precomputed.py`, por lo que
la identidad de código cambia explícitamente.

La comparación del [predictor completo](../../../reports/engineering/kron-precomputed-20261023/README.md)
conserva el protocolo, las medidas y el resultado negativo del control vectorial.
El prototipo medido está archivado en el commit
`6a56f194c62e2ab8d77a4bc4ea24d84f5214c3a4`. La opción financiera
`vector_delta_precomputed_v1` se retiró tras incumplir el criterio de rendimiento.
El operador genérico conserva la regla vectorial como referencia numérica para
reproducir las comprobaciones. El control financiero `vector_delta_v1` sigue usando
su ejecución de referencia.

La receta independiente
[`window-month.json`](../../../configs/experiments/kronlstm-precomputed/window-month.json)
fija enero de 2015 para ajuste y febrero para validación, 59 días entre ambos.
Solo cambia `temporal_encoder` respecto a la receta Kron del piloto. Las seis
recetas anteriores se conservan. Esta receta está preparada y validada, sin un
ajuste financiero ejecutado. Una nueva ejecución necesita su protocolo conjunto,
admisión CUDA y presupuesto antes de arrancar. El test final permanece cerrado.

Las pruebas contrastan salidas, estados, gradientes, RNG, recuperación de cachés y
rechazo de identidades incoherentes en CPU y CUDA con FP32 y FP64. Las medidas de
latencia son técnicas y no establecen una mejora del error predictivo.
