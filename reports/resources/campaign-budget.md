# Coste experimental de las referencias multimodales

Medición del 19 de septiembre de 2026 en la RTX 4070 Laptop de 8 GB. Se entrenaron
MLP y GRU con precios, noticias, fundamentales, gráficos y contexto macro. Los
codificadores estaban congelados y sus representaciones ya calculadas.

Los registros corresponden al código de la revisión
`7d2e21d324104491db8c1d6589d08b799c7653cc`. Sus huellas se verifican contra esa
revisión. Las posteriores traducciones de mensajes no actualizan ni sustituyen
las mediciones históricas.

Se usó retorno residual apertura a cierre de la siguiente sesión, ajustado por SPY
mediante OLS con intercepto. Cada ajuste utilizó como máximo las 252 sesiones
anteriores, con un mínimo de 126 pares válidos. Las etiquetas futuras solo entraron
como objetivos una vez observadas, nunca como entradas del modelo.

Entrenamiento hasta diciembre de 2022 y validación de coste durante 2023. No se
abrió el test final. Son paneles técnicos para dimensionar trabajo, no una selección
destinada a demostrar eficacia predictiva.

## Tiempos observados

Tres épocas por caso, semilla 42, lote 64, AdamW, precisión `float32` y cero
trabajadores adicionales. Los tiempos de la tabla son medianas por época.

| Modelo | Activos del panel | Ejemplos de entrenamiento | Entrenamiento | Validación |
| --- | ---: | ---: | ---: | ---: |
| MLP | 4 | 6.246 | 0,400 s | 0,228 s |
| GRU | 4 | 6.246 | 0,460 s | 0,233 s |
| MLP | 11 | 12.067 | 0,808 s | 0,451 s |
| GRU | 11 | 12.067 | 0,905 s | 0,452 s |
| MLP | 22 | 21.552 | 1,425 s | 0,821 s |
| GRU | 22 | 21.552 | 1,533 s | 0,791 s |

En el panel de 22, entrenan 18 activos y validan los 22. Los otros cuatro no tienen
muestras admisibles en el tramo de entrenamiento. La validación contiene 2.759
ejemplos. Cada modelo comparte las mismas entradas y particiones dentro de su panel.

La MLP tiene 58.465 parámetros y la GRU 51.937. La memoria asignada máxima observada
durante estos ensayos fue de unos 88,1 MiB en CUDA. La PSS muestreada del proceso
alcanzó unos 1,44 GiB. Estas cifras no incluyen toda la memoria usada por el escritorio
o el controlador de la GPU y no equivalen a su consumo total.

El [registro completo](phase1-supervised-budget.json) conserva tiempos por época,
latencias por paso, memoria asignada y reservada, RSS/PSS, sensores disponibles,
exclusiones y 115 huellas de entrada. Las métricas de error se conservan como
diagnóstico de ejecución, no como resultados confirmatorios.

## Comprobación con veinte épocas

La [ejecución GRU de veinte épocas](phase1-supervised-budget-20epochs.json) sobre
el panel completo terminó en 59,86 segundos. Incluyó 10,83 segundos de preparación
de etiquetas, 31,08 de entrenamiento, 16,02 de validación, 0,32 de checkpoints y
otros costes de inicio y coordinación.

La proyección comparable desde las tres épocas era 46,68 segundos para inicio,
entrenamiento, validación y checkpoints. Se observaron 48,05 segundos, un 2,96 %
más. Esta comprobación respalda una estimación local para estas referencias y esta
carga. No valida una extrapolación al corpus completo.

## Qué se puede presupuestar

Con las representaciones disponibles, repetir estas redes pequeñas no supone una
restricción importante de VRAM. La adquisición, auditoría y extracción de
representaciones son costes diferentes que deben contabilizarse una vez y reutilizarse.
La codificación registrada mezcla trabajo nuevo y caché recuperada, por lo que no
se presenta su suma como un tiempo de primera preparación en frío.

Los informes incluyen proyecciones de 10, 20 y 30 épocas. Sus rangos proceden de
las épocas observadas, no son intervalos estadísticos de confianza. Preparación,
búsqueda, nuevas semillas, folds, calibración y análisis incrementan el coste total.

La comparación principal de hasta 128 activos sigue pendiente de medir con la
arquitectura candidata. No se multiplica el tiempo por número de empresas sin
conocer sus ejemplos válidos, longitud textual y cobertura. El ensayo tampoco mide
el coste de entrenar conjuntamente MiniLM o ResNet18, ni el de una memoria adaptativa.

Los ensayos son demasiado cortos y ligeros para acreditar rendimiento térmico
sostenido durante 24 horas. Temperatura y potencia se registraron cuando estaban
disponibles, sin cambiar la configuración energética del equipo.

## Lectura y concurrencia

Se repitió una [rejilla de 18 casos](../data/cost-profile.json) con lotes de
16, 32 y 64 y con cero, dos o cuatro trabajadores. Cada caso mide 50 pasos tras
calentamiento. Terminó sin errores de cierre de lectores. Usa entradas reales,
pero un objetivo artificial de coste, por lo que no se mezcla con el entrenamiento
supervisado anterior ni se interpreta su pérdida como precisión.

Con lote 64, la MLP pasó de unos 18.814 ejemplos por segundo sin trabajadores a
28.193 con dos. La GRU pasó de 17.772 a 33.220 con cuatro. Son tasas del tramo
medido, no del proceso completo. Arrancar los trabajadores costó alrededor de
un segundo por caso y la caché del sistema no estaba controlada. Solo se hizo
una repetición de esta rejilla final, por lo que no demuestra un óptimo general.

Los ensayos supervisados conservan cero trabajadores como referencia sencilla
de coste completo sobre estos paneles pequeños. Una campaña mayor deberá contrastar
el beneficio sostenido con la RAM del conjunto de procesos. La rejilla informa
RSS del padre, no atribuye esa cifra a todos los trabajadores.

## Repetición y recuperación

```bash
uv run --locked --extra cuda python scripts/train_budget_probe.py \
  --report reports/resources/phase1-supervised-budget.json \
  --output data/interim/budget-training/new-run
```

Cada época confirma un checkpoint con pesos, optimizador, estados aleatorios,
configuración y huellas. Se comprobó recuperación exacta de pesos en MLP y GRU
en el panel de cuatro activos. Los checkpoints permanecen fuera de Git. No se
atribuye a estas redes un estado de memoria adaptativa que todavía no tienen.
