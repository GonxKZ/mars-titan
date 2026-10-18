# Latencia del predictor con refinamiento

`benchmarks.refinement_latency_study` vuelve a ejecutar modelos seleccionados y
compara cada predicción, su identidad, orden y bits con el Parquet confirmado de
la evaluación. También exige los mismos contadores. Un cambio invalida la
medida, aunque conserve el MAE agregado. No ajusta pesos ni elige checkpoints.

La entrada es un estudio preparado de 64 o 256 activos, con la proyección
`linear_v1`, semilla 42 y lector K1. Antes de cargar modelos exige los sellos de
selección de ambos lectores y la evaluación completa. Las opciones `k1`, `k4cc`
y `policy` comparan dos rutas fijas y la política ya seleccionada del mismo
lector. La protección local de aprendizaje también se comprueba en este punto
de entrada. El test final permanece cerrado.

Cada invocación reconstruye cuatro veces el modelo y el estado inicial. La
primera pasada completa carece de instrumentación por evento. Las otras tres
usan sincronización CUDA antes y después de observar, resolver etiquetas y
publicar episodios. Se conserva todo el calentamiento histórico. Las primeras
cinco decisiones puntuables se identifican como calentamiento técnico y solo
se excluyen de los percentiles de latencia estable. Sus predicciones siguen en
la comparación exacta.

La observación incluye preparación del lote, transferencia, avance de Titans,
consulta episódica, selección de ruta, refinamiento, predicción y registro de
pendientes. La resolución de etiquetas y la admisión tienen medidas propias.
Los tiempos exclusivos descuentan llamadas anidadas para evitar sumarlas dos
veces. Los p50, p95 y p99 se calculan por evento. Dividir su tiempo entre activos
produce un coste amortizado, no la latencia de una petición individual.

El tiempo de la fase comprende lectura, calentamiento, decisiones y maduración.
Las auditorías adicionales y la carga se registran aparte. El tiempo total de
la función incluye admisión de recursos, carga, índices, repetición, auditorías
y escritura. Un supervisor externo debe contabilizar también el arranque del
proceso. La razón entre pasadas instrumentadas y la primera pasada incluye
efectos de caché y orden de ejecución, por lo que no aísla por sí sola el coste
de instrumentar.

La memoria registrada distingue el pico RSS durante la vida del proceso y los
picos del asignador CUDA durante la fase, con los modelos residentes incluidos.
No equivale a la memoria total del dispositivo. No se estiman transferencias,
consumo energético ni contadores de hardware sin una medida correspondiente.

La CLI admite `--work`, `--output`, `--case` y `--max-seconds`. El destino debe ser
nuevo y estar fuera del estudio de entrada. El dispositivo es `cuda:0`, FP32
estricto, sin TF32, y se conserva una sola carga propia GPU. El límite por
invocación no puede superar 1.200 segundos. El protocolo de la ronda limita a
veinte minutos la suma de estas comprobaciones, por lo que el supervisor debe
pasar el saldo disponible y conservar su contador entre casos. Un caso que no
quepa queda sin medir.

Las funciones inferiores admiten CPU explícita para fixtures técnicos. La CLI
no cambia a CPU cuando falta CUDA. Los tiempos permanecen acotados a 32.768
registros por defecto. Una excepción restaura los métodos instrumentados y
conserva un recibo de fallo si la ejecución ya había creado su salida.

La [verificación inicial](../../reports/engineering/refinement-latency-20261019/README.md)
cubre el medidor y la repetición con fixtures. La medición de los checkpoints
reales seleccionados sigue pendiente.
