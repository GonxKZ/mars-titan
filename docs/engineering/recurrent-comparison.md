# Comparación recurrente y diagnóstico de error

RNN simple y LSTM reutilizan la preparación, la fusión y el entrenador de las
referencias GRU y DLinear. La secuencia de 64 sesiones de precios entra en el
codificador temporal. Noticias, gráficos, fundamentales y macro aportan sus
representaciones disponibles en la decisión. Por tanto, la recurrencia se aplica
a precios y la fusión posterior conserva las demás entradas. No se afirma que
los cinco bloques sean secuencias recurrentes independientes.

La [RNN de PyTorch](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html)
utiliza una capa unidireccional de 32 unidades y activación tangente hiperbólica.
La [LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html) tiene la
misma dimensión oculta, sin proyección adicional. En ambas se utiliza el último
estado oculto. En LSTM no se confunde con el estado de celda. Los estados iniciales
son nulos en cada ventana y no se arrastran entre activos ni ventanas solapadas.

Cada bloque contextual se proyecta a 32 posiciones. La concatenación de las cinco
representaciones tiene 160 posiciones y la cabeza produce un retorno residual.
Igualar la dimensión oculta no iguala el número de parámetros de las familias.
Cada ejecución registra su capacidad y su coste, sin atribuir a la arquitectura
una diferencia que pueda deberse a la muestra o al presupuesto.

## Configuraciones y continuaciones

La [rejilla recurrente](../../configs/baselines/recurrent-variants.json) define
36 ajustes iniciales y 12 continuaciones. Cruza dos familias, pérdidas MSE,
L1 y Huber, tasas `1e-4` y `1e-3` y semillas 42, 43 y 44. Cada ajuste inicial
recorre 30 épocas. Los lotes tienen 16 muestras, sin procesos adicionales de
lectura, y se usa AdamW en `cuda:0`. El informe registra las opciones numéricas
efectivas de PyTorch y cuDNN.

Las continuaciones parten de MSE con tasa `1e-3`, elegido antes de observar su
error. Para cada familia y semilla se comparan cinco épocas con L1 y otras cinco
con MSE, ambas con tasa `1e-4` y optimizador nuevo. Cambiar el objetivo se contrasta
así con añadir el mismo número de pasos. Reutilizar pesos no equivale a reanudar
el optimizador y no aporta datos nuevos.

La [rejilla ampliada](../../configs/baselines/expanded-reference-variants.json)
aplica la misma regla a RNN, LSTM, GRU y DLinear, con 96 ejecuciones. Solo procede
sobre una intersección multimodal admitida y materializada. No combina en una
misma comparación modelos entrenados sobre cohortes distintas.

## Predicciones y comprobaciones

Se conservan los errores por época y las predicciones de los pesos finales.
`training-predictions.parquet` contiene predicciones dentro de muestra.
`predictions.parquet` contiene exclusivamente validación. Los dos archivos
conservan activo, instante, etiqueta, predicción del modelo y referencia cero.
La reserva de 2024 en adelante no se utiliza para ajuste ni evaluación.

La pérdida de entrenamiento de una época se mide mientras cambian los pesos.
No equivale al error del modelo final sobre ese mismo tramo. La evaluación final
separa ambas cantidades para describir el ajuste y la diferencia con validación.

Cada ejecución guarda checkpoints por época, comprueba la reproducción exacta
desde la primera época y contrasta las predicciones con el último checkpoint
restaurado. La recuperación de una trayectoria ya terminada es una comprobación.
No sustituye al ejecutor de reanudación operativa de una campaña interrumpida.

## Cómo interpretar el error

Para cada observación se define $e_i=\hat y_i-y_i$. MAE resume $|e_i|$ y RMSE
penaliza más los errores grandes. El sesgo medio, la mediana y los cuantiles de
error absoluto permiten detectar si una media oculta compensaciones de signo o
una cola de fallos. Las métricas relativas se comparan con un control explícito.
Una reducción frente a cero no demuestra rentabilidad.

Los desgloses por activo y periodo conservan su número de observaciones. Las
correlaciones agrupadas no se presentan como correlaciones de ordenación
transversal por sesión. Los objetivos constantes, la ausencia de señales
direccionales y las muestras insuficientes producen valores no estimables con
un motivo, no ceros que aparenten una medida.

El rango entre semillas mide sensibilidad a la inicialización. No es un intervalo
de confianza sobre el mercado. El remuestreo por bloques de fechas es exploratorio
y mantiene juntas las observaciones de una misma fecha. Con pocas fechas, ventanas
solapadas y selección dirigida de artículos, no demuestra generalización.

No se calcula MAPE sobre retornos próximos a cero. Una predicción puntual tampoco
permite acreditar calibración probabilística, cobertura de intervalos o puntuaciones
de una distribución inexistente. Sharpe, drawdown, costes y rotación necesitan una
simulación de operaciones definida y retornos negociables. No se deducen de los
residuales ni se rellenan con valores ficticios.
