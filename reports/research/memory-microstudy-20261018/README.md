# Microensayo de memoria y postentrenamiento

Ninguna de las configuraciones ajustadas supera el MAE del retorno residual cero en este microensayo. C+M reduce el error frente a su B, mientras PT1 empeora frente al núcleo `mac_online`. La comparación comprende cuatro activos, una época y una ventana de desarrollo. No permite elegir la mejor arquitectura para el universo completo ni acredita superioridad frente al estado del arte.

El [protocolo](../../../docs/research/memory-microstudy.md) se fijó en `a67706226`, antes de ajustar. Se completaron los 28 trabajos con 119 pasos cada uno, 3.332 actualizaciones en total. En todos entraron en la pérdida las 3.760 etiquetas admitidas de ajuste, sin etiquetas maduras descartadas por falta de grafo. Validación conserva 500 filas, calibración 252 y evaluación 1.004, agrupadas en 251 sesiones de 2005. El primer intento falló antes de actualizar por un modo de atención no declarado y permanece registrado.

La [comparación legible por máquina](results.json) contiene todos los modelos, métricas por sesión e intervalos. El [recibo de verificación](verification.json) conserva configuraciones, semillas, fuentes, checkpoints, recursos, pruebas y fallos. Las predicciones por muestra, tablas por sesión, índices y pesos permanecen en los artefactos locales del ensayo.

## Datos y controles

KT, WINA, ESP y TCX se eligieron por una regla fija sobre disponibilidad de ajuste, sin usar resultados de validación o evaluación. Se conservaron todas sus filas admitidas desde 2000 y las máscaras. Estas filas no tienen noticias ni fundamentales observados. Sí tienen precios, gráficos y contexto macro parcial, con una media de 66,43 indicadores observados de 140 en ajuste y 76,41 en evaluación. La presencia de los canales en el esquema no equivale a haber aprendido de cuatro modalidades observadas.

La configuración usa FP32 estricto, sin TF32, K = 1 y un banco de 32 episodios. Conserva el calentamiento de doce meses y la purga por intervalo de etiqueta. La selección puede retener el estado inicial. Las semillas 42, 43 y 44 se usan para el contraste PT1. Las demás configuraciones tienen una semilla y se identifican como exploratorias. La campaña completa y el test final de 2024 siguen cerrados.

## Error observado

Los valores son MAE residual por sesión. Los porcentajes expresan cambios de error, no tasas de acierto. Primero se promedian los activos de una sesión y después las sesiones.

| Configuración | MAE | Cambio frente a B | Reducción relativa frente a B |
| --- | ---: | ---: | ---: |
| B | 0,02521395 | 0 | 0 % |
| B+C | 0,01583059 | −0,00938337 | 37,21 % |
| B+M | 0,02518854 | −0,00002541 | 0,10 % |
| B+C+M | 0,01558730 | −0,00962665 | 38,18 % |
| Residual cero | 0,01320669 | No es un factor del contraste | No procede |

B mantiene la identidad de CM-v1, con el control C desactivado, SDPA Math y lector M1. Los dos núcleos B y B+C tienen la misma huella de parámetros iniciales. El resultado de C+M no sustituye esta referencia por otra elegida después de medir.

Con bloques de 16 sesiones y 2.000 réplicas, el intervalo temporal individual del 95 % de Δerror para C+M frente a B es [−0,01316717, −0,00667655]. El intervalo simultáneo de la familia publicada es [−0,01465490, −0,00459840]. Esto describe esta ventana y semilla, no variabilidad entre entrenamientos. El efecto aislado de M es inconcluyente: su intervalo individual es [−0,00014190, 0,00009406]. La interacción C/M es −0,00021787 y su intervalo [−0,00048779, 0,00003656] también incluye cero. No se ha demostrado una aportación adicional de M ni que C supere a otros controles convencionales de estabilidad.

| Núcleo, media de tres semillas | MAE |
| --- | ---: |
| `mac_online` | 0,03204301 |
| Inicialización sencilla de PT1 | 0,04232542 |
| PT1 | 0,03877956 |

PT1 aumenta el MAE un 21,02 % frente a `mac_online` y lo reduce un 8,38 % frente a su alternativa de inicialización. El intervalo individual de Δerror frente a `mac_online` es [0,00039864, 0,01309297]. El simultáneo incluye cero. No respalda la no inferioridad de PT1. Las tres semillas se promedian por fecha, sin convertir 251 sesiones en 753 observaciones temporales. La dispersión entre semillas se conserva por separado en el recibo.

La GRU episódica alcanza 0,01343789, el menor MAE de los modelos ajustados, aunque también supera al error del control cero. La referencia Transformer directa alcanza 0,06907537. Una época y una única tasa no permiten concluir que una familia sea superior después de un ajuste comparable de hiperparámetros.

Los lectores M0, M1, M2 y M3 producen MAE de 0,07113110, 0,07086848, 0,07033626 y 0,07169912. Ninguno mejora al padre congelado en este ajuste corto. En postentrenamiento, la continuación completa alcanza 0,02348041. El adaptador `head+fusion` alcanza 0,03261369, un 6,95 % menos que su referencia congelada de 0,03505106. Los casos `head` y `head+readout+fusion` seleccionan la época cero y coinciden bit a bit con esa referencia. No aportan cambios aprendidos en el estado seleccionado.

La referencia congelada difiere ligeramente de la emisión original del padre, cuyo MAE era 0,03507742, sin cambiar sus valores de parámetros. Por eso los adaptadores se comparan con el padre bajo las mismas condiciones de ejecución. El análisis inicial que no hacía esta distinción se conserva como antecedente corregido y no se usa para atribuirles una mejora.

## Estado, recuperación y coste

Quince reproducciones de los núcleos seleccionados conservan las predicciones bit a bit, los parámetros compartidos y los RNG de CPU y CUDA. El observador registra pesos rápidos y momentum durante el calentamiento y evaluación, sin pasos de optimizador. Estos valores no son extremos de entrenamiento.

En las tres semillas, el máximo observado de la norma del momentum de PT1 es de 1,54, 1,51 y 1,62, frente a 2,91, 3,51 y 3,08 de `mac_online`, tomando el máximo de las capas. La reducción del estado no se traduce aquí en menor MAE. La caja matemática sigue teniendo el alcance condicionado documentado. No certifica la contracción global del MAC ni el redondeo de su implementación.

Una ejecución adicional de recuperación se interrumpió en el paso 64 y continuó hasta el 119. Terminó con los mismos tensores de modelo y optimizador que la ejecución continua, y predicciones idénticas en validación, calibración y evaluación. La reapertura de un resultado completado ejecuta cero pasos y conserva las huellas de sus 30 archivos. Estas comprobaciones se contabilizan aparte de los 28 ajustes de la comparación.

| Sistema completo de CM-v1 | Tiempo de núcleo más lector | Pico asignado por PyTorch |
| --- | ---: | ---: |
| B | 71,19 s | 79.216.128 bytes |
| B+C | 95,72 s | 98.409.984 bytes |
| B+M | 72,59 s | 79.216.128 bytes |
| B+C+M | 95,63 s | 98.409.984 bytes |

El coste incluye ajuste, selección, predicciones y arranque de los procesos. C+M requiere un 34,33 % más de tiempo que B en esta ejecución. Los 28 trabajos y el intento inicial ocupan 959,40 segundos. El mayor pico RSS registrado es de 2.004.221.952 bytes. Se conservan 67 checkpoints, 104.793.372 bytes en total, entre estados recientes y seleccionados. El pico del asignador no representa toda la VRAM del proceso, porque no incluye todas las reservas directas del proveedor ni el contexto CUDA.

Los percentiles p50, p95 y p99 del paso de optimizador están en el recibo, separados del tiempo total. No se midieron energía, contadores de hardware ni escalado al universo completo. Tampoco se recogieron en este microensayo las curvas de distorsión J_t y retención de episodios raros de M. No se adopta un backend nuevo ni se interpreta esta medida como el límite del equipo.

## Comprobaciones RL con fixtures

Se ejecutaron ocho microajustes en `cuda:0` sobre las cintas sintéticas existentes, con 960 transiciones por algoritmo y una semilla. Las tres variantes PPO completaron 120 pasos de optimizador cada una, Double DQN 705 y KLPO, GRPO, Dr. GRPO y GSPO tres cada uno. Sus gradientes registrados son finitos y no nulos. Igualar transiciones no iguala actualizaciones de algoritmos distintos.

La selección retuvo la política inicial en los ocho casos. Las dieciséis auditorías posteriores, inicial y seleccionada para cada algoritmo, conservan exactamente las huellas de parámetros y decisiones. Los [recibos técnicos de RL](policy-fixtures.json) no son resultados financieros. El campo `domain=real` del binario identifica el recorrido de cinta reconstruida, pero estas cintas y puntuaciones son fixtures. Su procedencia se declara expresamente como sintética en este informe.

La etapa real de políticas requiere tres ventanas previas de predicciones fuera del ajuste. Este microensayo produce una sola ventana y no ejecuta esa comparación.

Se conserva una configuración rechazada antes de aprender porque declaraba dos entornos donde CUDA exige dieciséis. Otra versión usó una cota de 1.024 transiciones, que los controladores terminales redondearon a 960 para consumir oleadas completas. Antes de interpretar recompensas se fijaron las 960 transiciones comunes. No se utiliza la diferencia de exposición para ordenar algoritmos.

## Fuentes y comprobación matemática

La revisión de [SR-TTT v2](https://arxiv.org/html/2603.06642v2) confirma que sus mejoras iniciales se retiraron por fallos de evaluación y acceso al futuro. Se consultaron el resumen, la corrección y las secciones de defectos e implementación. [TTT-RM v1](https://arxiv.org/abs/2610.04701v1) y [Decision Titan v1](https://arxiv.org/abs/2610.01513v1) son antecedentes de memorias complementarias y RL offline, respectivamente. Sus resúmenes y versiones se comprobaron, sin reproducir sus experimentos. No se atribuye novedad general a combinar un banco y pesos rápidos.

Para una pérdida asociativa cuadrática, ∇ℓ = 2Jᵀr. La relación entre pérdida y norma de gradiente depende del Jacobiano. En una memoria lineal con clave de norma fija puede simplificarse, pero no debe trasladarse sin condiciones a una memoria no lineal. El [chequeo reproducible](../check_associative_surprise_scope.py) usa M(w) = w²: con (w, v) = (0, 1), la pérdida es 1 y el gradiente 0. Con (1, 3/2), la pérdida es 1/4 y el gradiente −2. Las fracciones exactas y autograd FP64 en CUDA concuerdan en estos casos ([recibo](associative-surprise-scope.json)). Es una delimitación elemental de alcance, no un algoritmo nuevo ni una mejora del predictor.

Los resultados mantienen separadas la formulación propia de PT1, su verificación matemática, la implementación y lo observado. La campaña confirmatoria, otras ventanas, la evaluación final y una comparación financiera real de políticas permanecen pendientes.

## Código y verificación

El [ejecutor](../../../benchmarks/memory_microstudy.py) prepara las copias y ejecuta un trabajo de la lista. El [analizador](../../../benchmarks/memory_microstudy_analysis.py) utiliza las métricas por sesión y el bootstrap del proyecto. La [reproducción de estado](../../../benchmarks/memory_microstudy_state.py) carga el checkpoint seleccionado y exige igualdad exacta, con un modo independiente para el control congelado. La protección global del aprendizaje no se modifica.

Las 22 pruebas nuevas comprueban selección sin futuro, fechas exactas, pin de PT1, límites, bloqueo de aprendizaje, emparejamiento, semillas y referencia congelada. Pasan también 160 regresiones de las guardas y dos pruebas del observador. Ocho mutaciones dirigidas quedan detectadas, con un control sin mutar que pasa. La primera ejecución de tres mutantes usó una ruta incorrecta y su recuento fue descartado antes de repetirlo.

Coverage.py 7.16.1 registra 388 de 443 sentencias y 101 de 138 ramas de los tres programas del microensayo. Radon 6.0.1 calcula CRAP como C² × (1 − cobertura de sentencias)³ + C. La cobertura incluye las pruebas, el análisis y comprobaciones CUDA de reapertura y reproducción. No incluye la instrumentación de los 28 ajustes originales. Los diagnósticos de cobertura no sustituyen sus comprobaciones de comportamiento ni acreditan ausencia de errores.
