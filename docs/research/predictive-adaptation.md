# Comparación de objetivos de adaptación predictiva

La adaptación añade una corrección lineal común a cada predictor de referencia,
cuyos pesos permanecen congelados. Recibe las cuatro modalidades, macro y la
predicción del padre. Las entradas se estandarizan con entrenamiento. Si `z`
es ese vector estandarizado, el centro adaptado es `μ = μ_padre + s(wᵀz + b)`,
con `w = 0` y `b = 0` al iniciar. La escala `s` procede de las etiquetas de
entrenamiento. La inicialización conserva exactamente el centro del padre.

Este experimento no modifica los pesos de RNN, LSTM, GRU o DLinear. Sus
[continuaciones neuronales](../engineering/reference-search.md) siguen siendo
otra comparación, con su capacidad y coste propios. El adaptador común permite
contrastar los objetivos sin introducir una arquitectura distinta en cada uno.

## Política y objetivos

La rejilla contiene las [21 acciones fijadas con entrenamiento](../engineering/causal-prediction-environment.md).
Para la acción `a_j`, se definen `z_j = a_j/s` y `m = μ/s`. La distribución es
`π_j = softmax(z_j m − z_j²/2)`. Es una gaussiana discretizada de anchura fija.
El término común `μ²` se omite porque se cancela al normalizar.

La alternativa `softmax(−|a_j − μ|/s)` tiene una región plana: cuando el centro
supera el mayor retorno de la rejilla, el término que contiene `μ` se cancela
en todas las acciones. Su gradiente respecto al centro es cero. La expresión
gaussiana evita esa cancelación matemática, aunque todavía puede saturarse en
coma flotante. Por eso se registran centros fuera de rejilla, masa en los
extremos, entropía y distribuciones próximas a una sola acción.

Para una etiqueta madura `y`, el coste es `c_j = |a_j − y|/s`. Se comparan:

- Muestreo del gradiente con REINFORCE. Se toma una acción de la política
  actual y se minimiza `stopgrad(c_A − b) log π_A`, con
  `b = stopgrad(Σ π_j c_j)`.
- Pérdida esperada exacta, `Σ π_j c_j`.
- MAE normalizado del centro continuo, `|μ − y|/s`.

REINFORCE es un estimador del gradiente del refuerzo esperado. La formulación
original corresponde a [Williams (1992)](https://doi.org/10.1007/BF00992696).
Aquí el baseline usa las pérdidas de todas las acciones. Es un experimento
con información completa y muestreo del gradiente, no una situación donde
solo se conoce la recompensa de la acción elegida.

Al enumerar las 21 acciones, `Σ π_j(c_j − b)∇log π_j = ∇Σ π_jc_j`, porque
el baseline no depende de la acción y `Σ π_j∇log π_j = 0`. Las pruebas
comprueban esa igualdad en CPU como referencia numérica y en CUDA. La
identidad corresponde al gradiente antes de AdamW y del recorte de norma,
que son transformaciones posteriores de la actualización.

El valor del objetivo auxiliar de REINFORCE no es una medida directa de
precisión. Su interpretación se explica en la
[derivación de optimización de políticas](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).
La comparación utiliza los errores observados, no el signo o la disminución
de esa función auxiliar.

## Población, preparación y recuperación

Cada padre debe tener un informe terminado, checkpoint y predicciones
verificables de la misma edición, particiones y objetivo. La huella del
manifiesto supervisado vincula también la definición de la etiqueta. El join
contrasta claves, activo, mercado, instante y objetivo, además de la cobertura
completa. Las modalidades no se copian para cada padre. Su caché conserva una
predicción `float64` por muestra y entrega tramos pequeños de solo lectura.

El normalizador se ajusta por bloques y solo con entrenamiento. Usa
[`StandardScaler.partial_fit`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
cuatro hilos durante ese cálculo y la ponderación del padre. Las columnas
constantes mantienen escala uno. Los tres controles reutilizan las mismas
estadísticas, entradas, padre, inicialización y tamaño de lote.

La [configuración](../../configs/baselines/predictive-adaptation.json) fija cinco
épocas y semillas 42, 43 y 44. Son nueve ajustes por padre. AdamW usa tasa
0,0001, decaimiento 0,01 y recorte de norma uno. No hay parada temprana que
reduzca el número de actualizaciones de un control. Se conserva el estado con
menor MAE por sesión de la mediana al finalizar épocas completas. Los empates
conservan la primera época.

El cursor, optimizador, generadores y pesos se confirman después de una
actualización completa. La recuperación distingue el estado de la última
actualización del estado seleccionado para predecir. Un bloqueo impide dos
escritores sobre una misma ejecución. Los cambios de datos, padre, código,
normalización o política numérica invalidan la recuperación.

Los parámetros del adaptador son `float32`. La normalización se calcula en
`float64` y su resultado se convierte a `float32` antes de la capa lineal.
El centro heredado, los logits y los objetivos mantienen `float64`.
No se hereda el tipo de parámetros de una opción global de
PyTorch. La ejecución del modelo exige `cuda:0`, sin sustitución por CPU.

## Evaluación y alcance

El ajuste es offline, posterior al corte de entrenamiento. Las predicciones
del padre sobre train son predicciones dentro de muestra. No representan una
política que ya existiese en 2018. Durante la evaluación de 2023 los pesos y
normalizadores permanecen congelados. Esa validación sirve para desarrollo
y selección, no constituye una prueba independiente tras comparar variantes.
La reserva final permanece cerrada.

Se guardan el centro continuo, el vecino más próximo de la rejilla y la
mediana inferior de la política. El vecino más próximo separa la cuantización
de la transformación probabilística. Para distancias calculadas iguales se
elige la acción de menor índice. También se comparan el padre y la predicción
cero, con MAE y MSE por fila, sesión y mercado.

La probabilidad de una acción no se interpreta como incertidumbre financiera
calibrada. La entropía y la masa extrema describen la política. La saturación
de etiquetas describe un límite de la rejilla. Los resultados de entrenamiento
por época se acumulan antes de cada actualización y no equivalen a evaluar
un único modelo congelado sobre todo train. Las predicciones finales sí usan
un estado seleccionado fijo.

El módulo `mars_titan.training.predictive_study` recibe `--config`, `--ordered`,
`--parent`, `--output` y, cuando corresponde, `--resume`. Este bloque no
implementa PPO financiero, ajuste conjunto ni MARS-TITAN. Las pruebas de
integración no se presentan como una campaña ejecutada sobre el corpus amplio.

La [verificación del bloque](../../reports/resources/predictive-adaptation-quality.json)
registra 44 pruebas nuevas, cinco mutaciones dirigidas detectadas y la batería
local completa de 1.299 pruebas correctas. Se ejecutó en dos procesos para no
acumular contextos CUDA con la GPU compartida. Incluye igualdad del gradiente,
rechazo de alineaciones incorrectas y recuperación exacta. Un fallo antes del
primer checkpoint permite reiniciar la preparación, pero la desaparición de
un estado confirmado se rechaza.
