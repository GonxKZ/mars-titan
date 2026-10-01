# Mínimo de aprendizaje y paciencia posterior

La edición de convergencia conserva la selección por `session_mae`. Una época reemplaza el mejor estado cuando su error baja más que `min_delta` respecto al último mejor error aceptado. La selección opera desde la primera evaluación y, en las continuaciones, incluye el padre como época 0.

`minimum_epochs` retrasa únicamente el contador de paciencia. Mientras `epoch <= minimum_epochs`, el contador permanece en cero. A partir de la siguiente evaluación completa, una mejora lo reinicia y una evaluación sin mejora lo incrementa. Con mínimo 5 y paciencia 8, una curva plana desde el padre se detiene después de la época 13. Una mejora posterior puede desplazar esa parada. El mínimo debe ser entero, no negativo e inferior al máximo de épocas.

| Configuración nueva | Mínimo | Paciencia posterior | Mejora mínima | Máximo |
| --- | ---: | ---: | ---: | ---: |
| [convergence-temporal-search-us.json](../../configs/baselines/convergence-temporal-search-us.json) | 10 | 10 | 0,00001 | 100 |
| [real-continuations-v3.json](../../configs/baselines/real-continuations-v3.json) | 5 | 8 | 0,00001 | 50 |

La búsqueda usa `schema_version=3`, conserva las cuatro familias, los candidatos 0 y 10, las semillas 42, 43 y 44 y las cuatro ventanas del protocolo temporal estricto. Sus controles neuronales MAE y MSE siguen ejecutando cinco épocas completas, con selección del padre y paciencia de cinco. Este presupuesto fijo conserva la igualdad de actualizaciones de esos controles. La continuación larga de cada referencia neuronal pertenece a la configuración de postentrenamiento separada.

La configuración real usa `selection.version=3` y solo admite `condition=real`. Incluye los seis objetivos del adaptador y los controles `neural_mae` y `neural_mse` en los padres neuronales. Las condiciones con remuestreo o síntesis mantienen la edición emparejada de presupuesto fijo. Los archivos de configuraciones anteriores y sus contratos siguen disponibles.

## Recibos y recuperación

Al completar una ejecución de la política nueva, `stop_reason` toma el valor `validation_plateau` si se ha consumido la paciencia, o `budget_exhausted` si termina por presupuesto. Si ambas condiciones coinciden en la última época, se registra `validation_plateau`. `last_epoch_improved` indica si la última época aceptó una mejora según `min_delta`. Llegar al máximo no acredita convergencia ni un óptimo global.

Una pausa mantiene `status=paused` y no publica un motivo de finalización. Solo las evaluaciones completas avanzan la selección. El estado conserva mejor época, mejor error, contador de paciencia, última época e indicador de parada, junto con pesos, optimizador, cursor, estadísticas e información de RNG. Una pausa durante la evaluación del modelo seleccionado conserva el estado recuperable del último optimizador.

La reanudación sigue exigiendo la identidad científica completa. No convierte una ejecución antigua en una ejecución de esta edición. La importación de padres para postentrenamiento usa el contrato de inferencia existente, con comprobación de arquitectura, dimensiones, fuentes y hashes de inferencia. No se relajan las comprobaciones de identidad de las referencias ni se reescriben recibos anteriores.

El postentrenamiento conserva dos estados recientes y el mejor si es distinto. La referencia mantiene su política existente de checkpoints. El mejor modelo y el estado usado para recuperar tienen funciones distintas.

## Comprobaciones

Las pruebas del selector comprueban mejoras antes del mínimo, recuperación tras una caída inicial, reinicio de paciencia y selección de época 0. Las pruebas CPU del postentrenamiento comparan la ejecución continua y la recuperada, incluidos pesos, optimizador y RNG, con las ocho variantes. Las pruebas CUDA de referencias usan un corpus técnico de 18 filas y comprueban los motivos de parada y la igualdad de las predicciones tras recuperar.

La integración CUDA de postentrenamiento comprueba `klpo_mc` y `neural_mae` sobre una vista temporal pequeña, con dos filas de entrenamiento y una de validación. Usa las puntuaciones reales y pausa después de un paso de AdamW. El estado recuperado y las predicciones coinciden con la ejecución continua, y la validación del checkpoint seleccionado coincide con el mejor error observado, incluida la época 0.

Estas comprobaciones validan contratos con cargas pequeñas. No son resultados de la comparación científica ni demuestran una mejora predictiva de los nuevos presupuestos. El test final sigue cerrado.
