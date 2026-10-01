# Métricas predictivas y agregación

La evaluación conserva el error por fila y el error por sesión. El primero
da más peso a las fechas con más observaciones. El segundo calcula el error
dentro de cada sesión y promedia después esas medias.

Para una sesión observada $s$, con $n_s$ muestras:

$$
\operatorname{MAE}_s = \frac{1}{n_s}\sum_{i\in s}|\hat y_i-y_i|,
\qquad
\operatorname{MSE}_s = \frac{1}{n_s}\sum_{i\in s}(\hat y_i-y_i)^2.
$$

La métrica primaria prevista para seleccionar las referencias es:

$$
\operatorname{MAE}_{\mathrm{sesiones}}
=\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}\operatorname{MAE}_s.
$$

Cada sesión se identifica por mercado e instante de decisión. Dos mercados
no se fusionan por coincidir en una fecha. En una comparación conjunta, cada
par observado tiene el mismo peso. También se conservan las medias de cada
mercado. La definición no exige que ambos calendarios tengan el mismo número
de sesiones.

Por ejemplo, los errores absolutos 1 y 3 en una sesión, y 6 en otra, producen
MAE por fila de 10/3 y MAE por sesión de 4. Una prueba integrada con cinco
observaciones en tres sesiones obtiene 0,072 y 0,07 respectivamente. Son
ejemplos de comprobación, no resultados bursátiles.

## Implementación y límites

`SessionErrors` conserva tres acumuladores por sesión. No guarda todos los
errores individuales. Admite lotes de hasta 4.096 observaciones y un presupuesto
predeterminado de 50.000 sesiones. Un lote inválido no modifica el estado ya
confirmado. Se rechazan valores no finitos, desbordamientos, fechas ausentes,
longitudes incompatibles y pérdida de precisión temporal al convertir a
microsegundos. Una colección vacía devuelve métricas indefinidas y su motivo,
nunca un error cero artificial.

Durante la evaluación de cada modelo se crea un acumulador nuevo. No se mezclan
épocas, semillas ni folds. El lector supervisado comprueba las claves y los
recuentos de muestras antes de entregar los lotes. Los resultados informan
`samples`, `session_count`, `session_mae`, `session_mse` y `by_market_session`.
El MAE conserva las unidades del retorno residual y el MSE usa su cuadrado.

La curva de entrenamiento sigue mostrando el error por fila de las predicciones
obtenidas durante las actualizaciones. La evaluación final de entrenamiento y
validación incluye ambas agregaciones. Cambiar la partición física de los
archivos no cambia su definición, salvo el redondeo propio de float64.

Los diagnósticos existentes de `models/baselines/diagnostics.py` calculan además
sesgo, RMSE, cuantiles de error, dirección y correlaciones para sus entradas.
No sustituyen la media por sesión definida aquí. El bootstrap anterior pondera
filas y no debe describirse como un intervalo de esta métrica primaria sin la
adaptación correspondiente.

La pérdida pinball, los intervalos predictivos, NLL y ECE quedan fuera de este
acumulador. NLL requiere una densidad especificada y ECE probabilidades de un
evento definido. No se deducen esas salidas de una predicción puntual.

La [verificación de esta agregación](../../reports/resources/session-evaluation-quality.json)
registra ejemplos conocidos, cobertura y mutaciones dirigidas. No presenta esos
ejemplos como precisión obtenida sobre datos de mercado.
