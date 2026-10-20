# Piloto neuronal con contrato v1

Esta ejecución describe `row_broadcast_v1`. Su control con ganancias unitarias no reprodujo la referencia después del ajuste, por lo que las diferencias entre configuraciones no se atribuyen al mecanismo neuronal. Los datos y resultados se conservan. La revisión `scalar_controls_v2` tiene otra identidad y otro manifiesto.

Se completaron once configuraciones sobre 64 activos estadounidenses, con semilla 42, una época y 685 actualizaciones externas por modelo. Las 15.936 predicciones de desarrollo corresponden a las mismas 249 sesiones de 2023. Se utilizaron 203.350 filas para ajustar, 8.000 para validar y 3.968 para calibrar. El test final de 2024 permaneció cerrado.

| Configuración | MAE residual por sesión |
| --- | ---: |
| Residual cero | 0,01584945 |
| Titans-MAC original | 0,01747972 |
| Escritura por coherencia | 0,01985341 |
| Escritura constante sin PT5 | 0,02090347 |
| PT5 original | 0,02547209 |
| PT5 con ganancias unitarias | 0,03106864 |
| PT5 con escritura modulada | 0,02349177 |
| PT5 con captura modulada | 0,02519505 |
| PT5 con ambas modulaciones | 0,02217319 |
| PT5 con escritura constante | 0,02443523 |
| PT5 con captura constante | 0,02414986 |
| PT5 con ambas ganancias constantes | 0,02430453 |

Todas las configuraciones tienen más error que el residual cero en esta ventana. La combinación presenta una diferencia de −0,00329890 frente a PT5, con intervalo temporal del 95 % de [−0,00511559, −0,00149755]. Es una diferencia observada entre estas implementaciones. El fallo del control unitario impide atribuirla a las modulaciones.

El diagnóstico encontró que expandir la puerta escalar por filas cambia la reducción de su gradiente, aunque el forward sea idéntico. En el fixture CUDA, la diferencia máxima del gradiente fue 3,05176·10⁻⁵ en FP32 y 4,26326·10⁻¹⁴ en FP64. Esas cifras no acotan la divergencia de una trayectoria de entrenamiento.

Dos repeticiones con la misma semilla reprodujeron exactamente los archivos de predicciones de la referencia y del control unitario, cada uno respecto a su primera ejecución. No son nuevas semillas ni observaciones temporales independientes. La referencia y el control partieron de los mismos parámetros compartidos, pero terminaron con tensores y predicciones distintos en validación, calibración y desarrollo.

El [resultado legible por máquina](results.json) conserva métricas, contrastes, intervalos, hashes y costes. Los trece primeros intentos, incluidos dos fallos sin actualizaciones, consumieron 1.543,436 segundos externos. Las dos repeticiones añadieron 270,329 segundos. El coste acumulado fue de 1.813,765 segundos. Los tiempos internos están incluidos en los externos.

La ejecución es un piloto de una semilla sobre desarrollo reutilizado. No confirma superioridad general, no acredita retención semántica y no justifica activar las variantes por defecto. La [repetición preespecificada](../../../../docs/research/neural-plasticity-protocol.md#repetición-con-controles-escalares) mantiene los datos y las decisiones de selección.
