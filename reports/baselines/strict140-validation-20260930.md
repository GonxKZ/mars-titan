# Validación de la edición temporal con 140 indicadores

Revisión del 30 de septiembre de 2026 de las 160 ejecuciones terminadas el 28 de septiembre. Son 64 referencias neuronales y 96 continuaciones supervisadas. Los [recibos y métricas saneados](strict140-validation-20260930.json) conservan la semilla, configuración, ventana y huellas de cada resultado. La revisión concilia los informes JSON. No vuelve a ejecutar los modelos ni recalcula sus predicciones.

Cada muestra admitida contiene las cuatro modalidades y los 140 indicadores. Las cuatro ventanas amplían el entrenamiento desde noviembre de 2022. Sus validaciones abarcan junio y julio, julio y agosto, agosto y septiembre, y septiembre y octubre de 2023. Las ventanas consecutivas comparten historia y no son réplicas independientes.

Las referencias tienen un máximo de 30 épocas, paciencia de cinco validaciones completas y mejora mínima de 0,00001 en MAE por sesión. Cuarenta y tres de los 48 finalistas terminan antes de ese máximo. Las continuaciones comparan MAE y MSE durante cinco épocas, con un optimizador nuevo y selección del padre como época 0.

| Pérdida | Casos | Mejoran la validación del padre | Conservan la época 0 | Última época peor que el padre |
| --- | ---: | ---: | ---: | ---: |
| MAE | 48 | 37 | 11 | 29 |
| MSE | 48 | 22 | 26 | 38 |

Se seleccionaron 59 continuaciones con mejora y 37 padres sin cambios. Ninguna selección empeora la métrica de validación del padre. Esa propiedad procede de la regla de selección. No demuestra por sí misma generalización.

En 67 casos, conservar la última época habría empeorado al padre. En 75, la última época es peor que el estado seleccionado. Estos resultados respaldan conservar el mejor checkpoint confirmado y permitir que el ajuste termine sin sustituir al padre.

Los 160 informes contienen predicciones de entrenamiento y validación. Los recuentos de calibración y evaluación no acreditan que esos bloques se hayan evaluado. La [continuación temporal](../../docs/engineering/temporal-posttraining.md) incorpora esa evaluación por separado, después de congelar los estados. El test real de 2024 permanece cerrado. No se han establecido superioridad fuera de muestra, rentabilidad ni resultados del candidato MARS-TITAN.
