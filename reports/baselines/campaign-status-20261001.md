# Estado de las campañas al 1 de octubre de 2026

La serie histórica terminó sus 216 ajustes a las 16:13 CEST. Sus seis padres tienen 36 casos completos cada uno. Esta edición conserva su cobertura macro incompleta y su presupuesto original. No se mezcla con la edición de 140 indicadores.

La edición estricta mantiene las 160 ejecuciones neuronales ya terminadas. La cola posterior completó los 17 casos tabulares de la primera ventana y 50 de sus 132 postentrenamientos. Quedó en pausa con un estado recuperable. Las otras tres ventanas y las evaluaciones posteriores siguen pendientes.

En esos 50 postentrenamientos, 47 terminaron antes del máximo de cinco épocas y 38 seleccionaron el estado inicial. Ningún estado elegido supera el MAE de validación de su estado inicial. Esta propiedad procede de la selección y no demuestra generalización fuera de muestra. Los [recibos saneados](campaign-status-20261001.json) conservan configuraciones esenciales, épocas, métricas de validación y huellas.

La comparación financiera sintética terminó 63 casos de siete variantes y tres semillas en tres etapas: 21 pilotos, 21 ajustes principales y 21 auditorías de estados congelados. No son 63 entrenamientos. La consolidación auxiliar no se activó porque no cumplió su condición de validación. Los mundos simulan tres conceptos macro y no acreditan la cobertura de 140 indicadores ni resultados financieros reales. Las métricas reservadas de auditoría no forman parte de este informe.

## Reglas de selección y parada

| Edición o método | Regla aplicada |
| --- | --- |
| Referencias neuronales estrictas | Máximo de 30 épocas, paciencia de cinco validaciones completas y mejora mínima de 0,00001 en MAE por sesión. |
| Sus 96 continuaciones supervisadas | Cinco épocas y selección que incluye el estado inicial. Se conserva el presupuesto del par MAE/MSE. |
| Nuevas continuaciones reales | Máximo de cinco épocas, paciencia de dos validaciones completas y selección del estado inicial. |
| Adaptadores históricos | Cinco épocas fijas y selección del mejor estado posterior. No se incorpora retroactivamente una parada diferente. |
| XGBoost tabular | 200 rondas fijas y selección de configuración por validación. No evalúa parada temprana durante el ajuste de esta edición. |
| Ridge | Solución regularizada y selección de regularización por validación. No tiene épocas de aprendizaje. |
| Comparación financiera sintética | Interacciones predeclaradas y selección de la mejor política, incluido el estado inicial. La edición de presupuesto igual no utiliza parada temprana independiente. |

Estas reglas no permiten afirmar que todos los modelos hayan alcanzado su mejor solución posible. La [tarea de convergencia](https://github.com/GonxKZ/mars-titan/issues/190) prepara una edición separada con un mínimo de aprendizaje, paciencia posterior y motivos de finalización que distingan falta de mejora y presupuesto agotado. Las configuraciones y resultados anteriores permanecen como evidencia de sus respectivos protocolos.

El test real de 2024 permanece cerrado. El candidato MARS-TITAN sigue sin implementar ni entrenar.
