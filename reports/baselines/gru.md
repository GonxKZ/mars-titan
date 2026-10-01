# Sonda GRU con entradas multimodales estrictas

La sonda reutiliza `CostProbe`, una GRU unidireccional de 32 unidades para la
ventana de precios y proyecciones de noticias, gráficos, fundamentales y contexto
macro. La fusión termina en una salida escalar. No es todavía MARS-TITAN ni una
memoria persistente. El estado recurrente se reinicia en cada ventana de 64
sesiones, sin arrastrarlo entre ventanas solapadas.

Se usan las mismas entradas estrictas y etiquetas de las sondas Ridge y boosting.
El objetivo es el retorno residual de la siguiente sesión, calculado con la
historia autorizada. Entrenamiento hasta 2022, validación en 2023 y test final
cerrado. Los codificadores de texto e imagen permanecen congelados y sus salidas
se leen de los artefactos preparados.

## Ejecución y memoria

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  uv run --locked --extra research --extra cuda --extra encoders \
  python -m mars_titan.gru_probe \
  --prepared data/processed/news-expansion-20260921 \
  --samples data/processed/bounded-samples-20260921-v2/US \
  --output data/interim/gru-probe-new \
  --report reports/resources/gru-probe-new.json \
  --epochs 3
```

La orden exige CUDA, destinos nuevos, noticias completas verificadas y entre dos
y diez épocas. Usa lotes de 16, precisión float32, AdamW con tasa `0,0001` y semilla
42. No busca hiperparámetros ni elige la mejor época según esta validación pequeña.

La lectura reutiliza `SupervisedWorkload` y los bloques Parquet existentes. El
presupuesto admite hasta 64 activos y 100.000 etiquetas entre entrenamiento y
validación. No materializa una matriz con todas las ventanas. Cero trabajadores
adicionales es una referencia de coste para la muestra pequeña, no una afirmación
de concurrencia óptima. La [rejilla anterior](../resources/campaign-budget.md)
separa el beneficio por paso del coste de arrancar trabajadores.

## Checkpoints y medidas

Cada época confirma pesos, optimizador, estados aleatorios, configuración,
huellas y cursor de fin de época. Una interrupción dentro de una época obliga
a repetir esa época desde el último checkpoint confirmado. Los campos de memoria
persistente y etiquetas pendientes están vacíos porque esta sonda no tiene ese
comportamiento, no porque se haya comprobado ya la recuperación de MARS-TITAN.

La recuperación ejecutada aquí es una prueba interna. Este CLI acepta solo
destinos nuevos y no ofrece aún una orden para continuar una ejecución caída.
Los checkpoints se pueden cargar mediante `load_checkpoint`, que exige la misma
configuración y huellas. Una campaña larga necesitará además una entrada de
continuación que valide el directorio y su último cursor, antes de ejecutarse.

El ejecutor vuelve a cargar la primera época y repite las restantes. Exige pesos
finales exactamente iguales a los de la ejecución continua. También restaura el
último checkpoint en otro modelo y contrasta las predicciones. Es una comprobación
de recuperación en el mismo entorno, no una garantía entre versiones de CUDA.

Se conservan tiempo por época, latencias por paso, memoria CUDA asignada y
reservada, RSS/PSS muestreada, sensores y tiempo total. Cada paso se sincroniza
para medirlo, lo que introduce un coste y no representa un bucle optimizado con
solapamiento. El tiempo total incluye además preparación y prueba de recuperación.
No se suma ese ensayo repetido al tiempo de entrenamiento de la campaña principal.

El ajuste y la validación tienen su propio coste. La extracción inicial de
representaciones no está incluida. Tampoco se extrapolan 65 ejemplos a todo el
corpus ni se interpreta el error de unas pocas noticias seleccionadas como
evidencia predictiva concluyente.

El reloj interno comienza al entrar en el ejecutor, después de importar los
módulos. El tiempo de proceso completo debe medirse aparte. La preparación
compartida ahora incluye la validación inicial de rutas y manifiestos en el reloj
de las referencias. Los informes anteriores conservan su versión y sus tiempos.

## Ensayo ejecutado

El [registro del 21 de septiembre](../resources/strict-gru-probe.json) contiene
50 ejemplos de entrenamiento y 15 de validación de DECK y MNST. Se comprobaron
nueve huellas comunes de datos frente a Ridge y las 15 filas de activo, fecha y
objetivo coinciden exactamente. La red tiene 51.937 parámetros.

| Medida | Resultado observado |
| --- | ---: |
| Tres épocas de entrenamiento, suma | 0,546 s |
| Tres validaciones, suma | 0,074 s |
| Entrenamiento de la primera época | 0,382 s |
| Entrenamiento de las épocas segunda y tercera | 0,082 s cada una |
| Preparación dentro del ejecutor | 2,471 s |
| Verificación de recuperación | 0,460 s |
| Tiempo interno total | 4,145 s |
| Tiempo completo del proceso | 5,90 s |
| Pico de memoria CUDA asignada durante entrenamiento | 76,379 MiB |
| Pico de memoria CUDA reservada durante entrenamiento | 80 MiB |
| Máximo RSS observado por GNU time | 1.417.004 KiB |

El [recibo externo de tiempo](../resources/strict-gru-process-time.json) conserva
la medida del proceso completo, incluido arranque e importaciones. La memoria
asignada por PyTorch no representa toda la memoria del controlador ni de las otras
aplicaciones. La GPU estuvo compartida con una carga de unos 6,5 GB y actividad
alta. Estos tiempos no acreditan rendimiento máximo ni se extrapolan al corpus.

El MAE diagnóstico fue 0,088434 para GRU y 0,009620 para cero. El resultado de la
GRU fue peor. Se conserva sin cambiar las épocas ni buscar otra configuración
para mejorar este pequeño conjunto. El error de 15 filas dirigidas no permite
ordenar definitivamente los modelos ni afirmar utilidad financiera.

## Verificación

La suite completa pasó con 452 pruebas, sin omisiones, incluida la integración
de codificadores reales en CUDA. La recuperación desde la primera época reprodujo
exactamente los pesos finales y el último checkpoint conservó las predicciones.
La integración comprueba cambios de pesos en precios y en cada proyección modal.

Tres mutaciones fueron detectadas: eliminar el optimizador en ambas pasadas,
permitir el sufijo futuro y adelantar el cursor del checkpoint. Las modificaciones
se hicieron en memoria, sin cambiar archivos de producción ni los datos originales.

El [registro de calidad](../resources/strict-gru-quality.json) conserva herramientas,
alcance y límites. En los dos ejecutores, coverage.py 7.16.1 midió un 86,22 % de
sentencias y un 76,56 % de ramas. Radon 6.0.1 asignó complejidad 26 al ejecutor
GRU, con CRAP 26,173 usando cobertura de sentencias por función. El CLI queda fuera
de la cobertura automatizada y el ensayo real no se suma a ese porcentaje.

#26 permanece abierta para la referencia definitiva, la decisión sobre DLinear
y la campaña con particiones comparables y cobertura suficiente. La continuación
operativa de campañas largas corresponde a #67.
