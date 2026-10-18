# Combinaciones de memoria en una cohorte de desarrollo

Se completaron 44 de los 83 trabajos del [protocolo](../../../docs/experiments/combined-memory-v1.md), con 64 instrumentos, dos épocas y evaluación en 2023. El factorial de ocho combinaciones dispone de tres semillas por celda. CM-v1 dispone de sus cuatro variantes con una semilla. El plazo de ejecución terminó el 18 de octubre de 2026 a las 16:14:49 UTC. Un trabajo quedó interrumpido y 38 no llegaron a iniciarse. El test final de 2024 y la campaña completa siguen cerrados.

La combinación PT5+PT6 obtiene el menor MAE medio de las ocho celdas. La mezcla de los tres factores tiene un resultado inconcluyente frente al núcleo. Ninguna celda supera al retorno residual cero. Tampoco hay evidencia de una mejora general del banco episódico frente al control que ajusta el refinador sin banco. Estos resultados no justifican promover una arquitectura.

## Datos y alcance

La cohorte conserva todas sus filas admisibles desde 2000, con 169.674 filas de ajuste hasta marzo de 2022, 8.000 de validación, 3.968 de calibración y 15.936 de evaluación en 249 sesiones de 2023. Todas las comparaciones usan los mismos identificadores y objetivos. La selección de los 64 instrumentos utiliza únicamente duración y presencia de entradas del ajuste. Hay ocho instrumentos de cada uno de los ocho estratos declarados. La [auditoría de cohortes](cohorts.json) conserva la selección, los recuentos y las huellas.

Los estratos cruzan noticias y fundamentales presentes en al menos el 5 % de las observaciones del último año de ajuste con una historia de al menos 2.520 filas. Las ausencias permanecen explícitas. La presencia de una entrada macroeconómica codificada no demuestra que estén disponibles los 140 indicadores originales ni que todos tengan una disponibilidad histórica perfecta. La muestra contiene distintos tipos de instrumentos y no representa estadísticamente todo el mercado. La edición histórica completa no queda verificada por este ensayo.

El calentamiento usa doce meses de entradas pasadas. Cada recorrido reinicia los estados. El banco episódico empieza vacío y el calentamiento no escribe etiquetas financieras. Por tanto, el banco de evaluación no contiene automáticamente todos los episodios desde 2000. El pasado completo de ajuste interviene en el aprendizaje de los parámetros, con el alcance que permite esta cohorte.

## Factorial de PT1, PT5 y PT6

PT1 reúne la inicialización de puertas y las restricciones declaradas. PT5 añade la cascada de pesos rápidos con dos escalas. PT6 modula la fusión según la presencia de modalidades. Los controles de inicialización aislada, mezcla EMA y rama ancha quedaron sin ejecutar, así que esta ronda no separa todavía todos sus mecanismos.

La tabla promedia los errores por sesión de las semillas 42, 43 y 44. No es un ensemble de predicciones. La desviación típica describe la variación entre semillas.

| Configuración | MAE por sesión | Desviación entre semillas | Cambio relativo frente al núcleo |
| --- | ---: | ---: | ---: |
| Núcleo, `cell0` | 0,01482282 | 0,00294502 | Referencia |
| PT1 | 0,01551535 | 0,00336011 | −4,67 % |
| PT5 | 0,01369191 | 0,00059942 | +7,63 % |
| PT1 + PT5 | 0,01329856 | 0,00062362 | +10,28 % |
| PT6 | 0,01422116 | 0,00175690 | +4,06 % |
| PT1 + PT6 | 0,01573041 | 0,00280492 | −6,12 % |
| PT5 + PT6 | 0,01320484 | 0,00031049 | +10,92 % |
| PT1 + PT5 + PT6 | 0,01426156 | 0,00080189 | +3,79 % |
| Retorno residual cero | 0,01231802 | No procede | Control fuerte |

El contraste principal, mezcla completa menos núcleo, da ΔMAE = −0,00056126. Su intervalo temporal del 95 % es [−0,00132270, +0,00017460] y el simultáneo de la familia es [−0,00161453, +0,00049200]. El resultado es inconcluyente. La mezcla PT5+PT6 reduce el error observado frente al núcleo, con intervalo simultáneo [−0,00259963, −0,00063632], pero su MAE sigue siendo un 7,20 % mayor que el del retorno cero.

Añadir PT1 a PT5+PT6 aumenta el MAE en 0,00105671, con intervalo simultáneo [0,00091613, 0,00119729]. Añadir PT6 a PT1+PT5 también lo aumenta, en 0,00096300. La interacción PT1/PT5 con PT6 desactivado es negativa, mientras que la de PT1/PT6 con PT5 desactivado es positiva. No hay una regla según la cual acumular componentes mejore el predictor. Los efectos condicionados, promediados y de tercer orden están separados en [results.json](results.json).

Los intervalos usan 2.000 réplicas de bloques temporales circulares de 16 fechas UTC observadas y conservan juntos los activos de cada fecha. El archivo incluye la sensibilidad predefinida a bloques de 8 y 32. Los intervalos simultáneos se aplican dentro de cada familia, no a todos los resultados juntos. Condicionan sobre las semillas ejecutadas. La dispersión entre semillas se informa aparte, sin convertirlas en nuevas fechas. Dos épocas, una tasa fija y una ventana no acreditan convergencia ni generalización.

## Referencias y lectores episódicos

Con semilla 42, el Transformer directo obtiene 0,01287059, MAC sin actualización 0,01247382, MAC congelado 0,01298453, la variante con las proyecciones del artículo 0,01327591 y la GRU episódica 0,01252232. El núcleo `cell0-42` obtiene 0,01279036. Estas referencias tampoco superan al retorno cero en el promedio observado. No se ha repetido cada referencia con tres semillas.

Los lectores se comparan con el padre congelado, cuyo MAE es 0,01279036. K = 1 y la capacidad episódica es 128. El padre no recibe actualizaciones del optimizador del lector.

| Política | Inicialización original | Refinador inicializado a cero |
| --- | ---: | ---: |
| M0, sin banco | 0,01234973 | 0,01233728 |
| M1, conservación uniforme | 0,01235470 | 0,01235269 |
| M2, error maduro | 0,01236132 | 0,01235560 |
| M3, sorpresa completa | 0,01235403 | Evaluación interrumpida |

La mejora del lector frente al padre aparece también en M0. M1 con inicialización original aumenta el MAE en 0,00000497 frente a M0, con intervalo simultáneo que incluye cero. M2 aumenta el error en 0,00000662 frente a M1 original. La inicialización nula cambia poco los lectores completados y sus contrastes frente a la inicialización original son inconcluyentes. No se atribuye una mejora al banco por la reducción de error que consigue el refinador sin banco.

La inicialización residual nula tiene antecedentes. Aquí se integra como una opción que preserva inicialmente la salida del padre y puede desactivarse. Su paridad inicial, gradientes y recuperación están comprobados. No se reivindica como un descubrimiento general de aprendizaje profundo.

## Factorial de CM-v1

B conserva la identidad de CM-v1 de esta ronda. C penaliza una compresión `RᵀJR` del Jacobiano de la transición rápida, con rango 4, frecuencia 16, dos flujos, rejilla de 64 ángulos, umbral 1 y peso 0,1. M usa la retención heurística con centros fijos, frontera 8 y ocho candidatos nuevos. La compresión puede omitir direcciones expansivas. La heurística no hereda el factor del algoritmo publicado de metric k-median.

| Variante | MAE por sesión | ΔMAE frente a B | Cambio relativo | Intervalo simultáneo del ΔMAE |
| --- | ---: | ---: | ---: | --- |
| B | 0,01295901 | 0 | Referencia | No procede |
| B+C | 0,01304591 | +0,00008690 | −0,67 % | [−0,00005021, +0,00022400] |
| B+M | 0,01279317 | −0,00016584 | +1,28 % | [−0,00021115, −0,00012054] |
| B+C+M | 0,01290501 | −0,00005401 | +0,42 % | [−0,00021446, +0,00010644] |

M reduce el error observado frente a B en esta ventana y semilla. C+M frente a B es inconcluyente. La interacción `MAE_CM − MAE_C − MAE_M + MAE_B` es +0,00002494 y su intervalo simultáneo incluye cero. C+M mejora frente a C, pero no hay evidencia de que mejore frente a M. Las cuatro variantes mantienen un MAE mayor que el retorno cero. No se han reproducido pruebas Lean ni se afirma estabilidad global del predictor.

## Coste observado

Se utilizó `cuda:0`, una RTX 4070 Laptop de 8 GiB, PyTorch 2.14.0+cu130, CUDA 13.0, FP32 estricto, TF32 desactivado y dos hilos CPU. Una sola carga propia utilizó la GPU. La ronda conservó las aplicaciones y la configuración energética del equipo. Algunas comprobaciones CPU coincidieron con los ajustes, por lo que estos costes no constituyen un benchmark aislado de velocidad.

| Recorrido, semilla 42 | Tiempo interno del proceso |
| --- | ---: |
| Núcleo `cell0` | 179,88 s |
| Mezcla completa `cell7` | 229,21 s |
| Núcleo B de CM-v1 | 210,84 s |
| Núcleo C, suma de dos intentos | 1.157,88 s |
| Lector B | 260,21 s |
| Lector B+C | 253,55 s |
| Lector B+M | 282,58 s |
| Lector B+C+M | 275,07 s |

El coste completo de cada variante incluye su núcleo y su lector. C cuesta unas 5,49 veces el núcleo B en estos intentos. M añade unos 22,36 segundos al lector de B. No hay evidencia aquí de que ese coste compense en el contraste conjunto C+M. Se completaron 1.370 actualizaciones por ajuste de estas variantes. La selección puede conservar una época anterior sin reducir las actualizaciones de su control.

Los 47 recibos conservan 44 ejecuciones completas, una pausa reanudada, un fallo previo al ajuste y una terminación por plazo. Suman 10.801,64 segundos de cronómetros internos y, para la terminación sin recibo del hijo, del controlador. El límite absoluto de la cola fue de 12.600 segundos, más una reserva técnica de 1.800 dentro del presupuesto de cuatro horas. Los cronómetros internos incluyen carga, aprendizaje, selección, evaluación y escritura posteriores a su inicialización. No incluyen necesariamente el arranque del intérprete ni la espera de la GPU.

El mayor pico de PyTorch fue 264,86 MiB asignados y 614 MiB reservados. El mayor RSS registrado fue 2,10 GiB. No son medidas de toda la memoria ocupada por el sistema. Los percentiles p50/p95/p99 publicados corresponden únicamente a la llamada sincronizada al optimizador. No son latencias de predicción por activo ni de sesión. No se midieron energía, bytes de transferencia ni una traza completa de amplificaciones y consolidación `J_t`.

El [inventario de parámetros](parameters.json) registra 186.119 en el núcleo y 189.319 con FiLM. Un lector M1 añade 20.801 parámetros ajustables. M0 ajusta 12.417 y conserva las proyecciones no utilizadas como parámetros inertes. Los estados rápidos por flujo, el banco episódico y los codificadores previos del corpus no forman parte de ese recuento.

## Sensibilidad numérica detectada

Una reproducción sin optimizador conserva exactamente las predicciones originales cuando mantiene los flags de los parámetros. Al congelarlos en FP32, el núcleo cambia poco, con diferencia absoluta media de 0,000000157 y máxima de 0,00014461. En la mezcla completa, sobre las mismas 15.936 filas y el mismo checkpoint, esas diferencias aumentan a 0,00761683 y 0,06760360. Es un cambio material de trayectoria que impide tratar ambas rutas como intercambiables.

Las dos reproducciones FP64, con los valores FP32 del checkpoint promovidos exactamente, coinciden entre sí en las salidas almacenadas. FP64 no es una solución exacta. Su trayectoria también difiere de la original FP32. Quitar el grafo de las proyecciones locales no cambia la reproducción original y el control `TORCH_LINEAR_FLATTEN_3D=1` no elimina la diferencia entre flags en FP32.

El código de [PyTorch en el commit instalado](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/LinearAlgebra.cpp) tiene decisiones de multiplicación que dependen de `requires_grad`. El control de aplanado pertenece al [recorrido lineal con sesgo](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/Linear.cpp). Esa posibilidad motivó el diagnóstico, pero no se ha identificado la operación que inicia la divergencia de este modelo. Las [limitaciones numéricas documentadas](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) tampoco explican por sí solas su magnitud.

El [diagnóstico](numerics.json) conserva las nueve rutas, tiempos, hashes y diferencias, con cero pasos de optimizador. No modifica pesos, etiquetas, resultados anteriores ni backend predeterminado. La comparación de lectores usa su propio padre congelado. Antes de ampliar o promover estas memorias hay que localizar la primera divergencia y comprobar la sensibilidad de la trayectoria, con una tolerancia declarada y una referencia de mayor precisión.

## Evidencia y trabajo pendiente

[execution.json](execution.json) identifica cada trabajo, su checkpoint, selección, código y estado. [results.json](results.json) contiene errores, dirección, rangos, cuantiles, desgloses por cobertura e intervalos. Las métricas por sesión están en `sessions/` y sus huellas en [artifacts.json](artifacts.json). Las predicciones individuales y los checkpoints permanecen en el almacenamiento local. La implementación y su [verificación](../../engineering/combined-memory-20261018/README.md) se conservan separadas de los resultados predictivos.

El código científico se fijó en `41d3d0b96c7f5cb77af7003550be16b2ff0c0dba`. El lanzador del lector corrigió la representación del valor predeterminado K = 1 en `02b0cd1572977c3d56d519531fef7e19cc9cfc06`. No cambiaron módulos del modelo, datos ni configuración del estudio. Los intentos fallidos y reanudados consumen presupuesto y se conservan.

La última ejecución, M3 con refinador nulo sobre `cell0`, alcanzó las 1.370 actualizaciones, pero fue interrumpida antes de completar la publicación de su evaluación. Su checkpoint es recuperable y no se incorpora una métrica parcial. Permanecen sin ejecutar los lectores sobre `cell7`, K = 2 y K = 4, PT4 y sus lectores, los controles PT1 de inicialización, EMA y rama ancha, cuatro continuaciones o adaptadores y las ocho celdas de 2020. PPO/KLPO, nuevas calibraciones y el estudio confirmatorio no se ejecutaron en esta ronda. La lista exacta de pendientes está en el recibo.
