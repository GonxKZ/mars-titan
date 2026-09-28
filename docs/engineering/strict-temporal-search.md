# Búsqueda sobre las ventanas con 140 indicadores macro

La [configuración de búsqueda](../../configs/baselines/strict-temporal-search-us.json) declara dos candidatos por familia antes de ejecutar comparaciones. Conserva RNN, LSTM, GRU y DLinear. Los índices 0 y 10 del diseño existente corresponden a estas configuraciones:

| Candidato | Anchura | Capas | Dropout | Pérdida | Tasa de aprendizaje |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 32 | 1 | 0 | MAE | 0,0001 |
| 10 | 64 | 1 | 0,2 | MSE | 0,0001 |

Se comparan ocho candidatos con la semilla 42 en cada ventana. La configuración de cada familia se elige por MAE de sesión en validación, con desempate por identificador. Se añaden las semillas 43 y 44 para esa configuración y se reutiliza la ejecución ya completada de la semilla 42. Cada referencia recorre todas sus filas de entrenamiento, usa lotes de 512 y un contexto de 64 sesiones. El máximo es de treinta épocas, con paciencia de cinco evaluaciones completas y mejora mínima de 0,00001 en el MAE de sesión.

Cada referencia seleccionada tiene dos continuaciones supervisadas, una con MAE y otra con MSE. Ambas parten del mismo checkpoint y realizan cinco épocas con tasa de aprendizaje 0,0001 y un AdamW nuevo. La selección permite conservar la época 0. Se evalúa de nuevo al padre sobre las entradas de esa ejecución antes de actualizar parámetros. Si ningún ajuste supera el criterio declarado, las predicciones finales proceden del padre.

La paciencia de las continuaciones no puede ser menor que su presupuesto de épocas. Así se mantienen las mismas actualizaciones en los controles pareados, aunque el estado elegido sea anterior. La parada temprana de las referencias y la selección del mejor estado de las continuaciones cumplen funciones distintas. Ninguna de estas reglas garantiza por sí sola ausencia de sobreajuste.

Cada ventana comprende ocho candidatos, ocho repeticiones adicionales y veinticuatro continuaciones. Las cuatro ventanas suman 160 ejecuciones, de las cuales 64 son referencias y 96 son continuaciones. Las configuraciones históricas mantienen su versión y recuentos anteriores.

## Ejecución y recuperación

`uv run python -m mars_titan.training.temporal_search --help` describe la ejecución. Requiere la configuración, el directorio de las vistas y un destino nuevo. `--resume` comprueba la identidad de la preparación y continúa las ejecuciones pendientes. Un resumen incompatible se rechaza sin sobrescribir su evidencia.

El controlador comprueba que estén todas las ventanas del protocolo, con poblaciones positivas y huellas coincidentes. Usa la admisión CUDA existente para mantener una única carga científica y conservar margen de memoria. Cada ventana publica su registro y el resumen exterior actualiza el número de ejecuciones realmente completadas. Las métricas por época permanecen en el `run.json` correspondiente.

El checkpoint seleccionado se sustituye cuando mejora el criterio declarado. Los estados de recuperación conservan el cursor confirmado, el optimizador y los RNG. La evaluación inicial del padre y la selección de época 0 también se recuperan después de una pausa. El entrenador utiliza la retención existente de dos estados recientes y la referencia al mejor estado, sin un archivo permanente por época.

Los datos de calibración y evaluación quedan separados de esta selección. El test de 2024 continúa reservado. La configuración no declara una mejora predictiva antes de observar resultados. La cohorte de noticias y los límites de procedencia siguen identificados en las [vistas de entrada](../data/temporal-corpus.md).

Las comprobaciones de selección, recuperación, recuentos y orquestación se registran en [strict-temporal-search-quality.json](../../reports/resources/strict-temporal-search-quality.json). Incluyen una búsqueda técnica en CUDA, continuidad tras una pausa en la validación inicial y recuperación de directorios creados antes de confirmar su primer recibo.
