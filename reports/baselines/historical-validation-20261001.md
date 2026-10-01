# Validación de la serie histórica terminada

Se conciliaron los 216 recibos y sus predicciones de validación, sin casos fallidos ni pendientes en la comprobación. Esta edición conserva cobertura macro incompleta y cinco épocas por ajuste. Sus resultados se analizan por separado de las ventanas con 140 indicadores.

De los 216 ajustes, 21 reducen el MAE por sesión frente a su padre y 195 lo aumentan. La comparación utiliza la mediana de la política discreta frente a la predicción del padre. Esa diferencia no aísla el sobreajuste ni el efecto de discretizar la salida. Esta edición seleccionaba estados posteriores al ajuste y no incluía el estado inicial entre los candidatos.

| Padre | Casos | Reducen MAE | Aumentan MAE | Delta medio al padre |
| --- | ---: | ---: | ---: | ---: |
| dlinear | 36 | 0 | 36 | 0.00071582 |
| gru | 36 | 0 | 36 | 0.00077874 |
| lstm | 36 | 0 | 36 | 0.00078200 |
| ridge | 36 | 21 | 15 | 0.00010262 |
| rnn | 36 | 0 | 36 | 0.00072981 |
| xgboost | 36 | 0 | 36 | 0.00082408 |

El ajuste con menor MAE observado es `rnn/mae-s44`, con 0.01501183. Su delta al padre es positivo (0.00003405), por lo que tampoco mejora a su propio padre. La selección de este mínimo es descriptiva y no constituye una elección confirmatoria basada en test.

Las medias agrupan configuraciones y semillas. No se presume independencia, significación estadística ni una mejora general de un método. Los [resultados por caso](historical-validation-20261001.json) conservan configuración, semilla, métricas y huellas de los recibos.

La [edición de convergencia](../resources/convergence-verification-20261001.md) añade selección del estado inicial, mínimos y paciencia. La comparación con aquella edición requerirá respetar sus poblaciones, periodos y definiciones de salida. Los datos y resultados históricos no se sobrescriben. El test final permanece cerrado.
