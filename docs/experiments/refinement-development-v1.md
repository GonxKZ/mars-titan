# Estudio de desarrollo de las rutas de refinamiento

Este estudio mide si una política pequeña puede elegir entre parar, refinar con
los episodios ya recuperados o repetir la consulta antes de refinar. Conserva
un único avance de Titans-MAC por observación y una instantánea M1 común para
las seis alternativas. No cambia la identidad de la campaña ni selecciona un
nuevo modelo base.

La configuración es `configs/experiments/refinement-development-v1.json`.
Los ejecutores son `benchmarks/refinement_study.py`,
`benchmarks/refinement_study_traces.py`, `benchmarks/refinement_study_policy.py`
y `benchmarks/refinement_study_controls.py`.
La implementación del mecanismo y sus límites están en
[las rutas de refinamiento](../../src/mars_titan/models/titans/refinement_routes.py)
y [la política de ganancias firmadas](../../src/mars_titan/models/titans/refinement_policy.py).

## Datos y orden temporal

Se eligen 64 activos US ordenando una huella con sal fija. Solo son elegibles
los activos con al menos una fila aceptada en el ajuste de `fold-017`, antes del
1 de abril de 2021. La selección no usa valores objetivo ni cobertura de fechas
posteriores. Ambos folds conservan los mismos activos y los mismos archivos de
muestras. Cada fase utiliza todas sus filas admitidas desde 2000, con las
ausencias explícitas de la edición histórica. Esta cohorte acotada no acredita
que se haya preparado o evaluado el universo completo.

| Componente | Ajuste | Selección | Calibración | Evaluación de desarrollo |
| --- | --- | --- | --- | --- |
| Núcleo y lector | Desde 2000 hasta marzo de 2021 | Abril a septiembre de 2021 | Octubre a diciembre de 2021 | La ventana original de 2022 no elige la puerta |
| Política de rutas | Enero a marzo de 2022 | Abril a septiembre de 2022 | Octubre a diciembre de 2022 | 2023 |

`posttraining.chronological_windows.staged_window` define el nuevo tramo de
ajuste dentro de la partición `train` de `fold-018`. No se crean seis particiones
físicas. `posttraining.staged_rows.fit_rows_proof` comprueba que sus identidades
de fila no se solapan con ajuste, selección o calibración del padre y que las
etiquetas del padre ya habían madurado. La vista conserva las purgas por el
intervalo real de la etiqueta.

Cada fase reinicia los estados y utiliza el calentamiento previo de doce meses
solo con entradas. El test final de 2024 permanece cerrado. El año 2023 ya ha
participado en desarrollo, por lo que sus resultados no serán una confirmación
final independiente.

Los recorridos originales del núcleo y del lector de `fold-017` publican
predicciones de evaluación de 2022 antes de confirmar su `run.json`. Sus pesos
se eligen con la validación de 2021, pero esa evaluación ya recorre fechas que
después pertenecen al ajuste, selección y calibración de la puerta. Se conserva
ese contrato y se registra esta exposición previa. Los valores de esos
informes de evaluación no entran en el ajuste ni en la selección de la puerta.
No se afirma que las etiquetas de calibración de 2022 sigan sin leer.

El sello del estudio controla el cálculo y la publicación de las fases
posteriores, por proyección y semilla. Requiere que estén confirmadas las
selecciones de los lectores K1 y K4, vinculadas al mismo núcleo. Se comprueba
antes de generar o reutilizar las referencias de `fold-018` y sus resultados
posteriores. No es un sello global entre semillas ni entre estudios. Tampoco
es un control físico de acceso a los Parquet, que contienen varios tramos y
se leen para verificar hashes y contratos.

## Comparación fijada

Hay dos referencias del núcleo, `linear_v1` y `mac_paper`, con identidades
distintas. Ambas usan `sequence_contiguous_v1` para que congelar los parámetros
no cambie la ruta de proyección de atención. La comparación no atribuye a la
política una diferencia de codificador o de proyecciones de memoria.

Cada núcleo utiliza las semillas 42, 43 y 44 y dos épocas de presupuesto fijo.
Los lectores se ajustan también durante dos épocas, con capacidad M1 de 128
episodios e inicialización nula del refinamiento. Se distinguen lectores
ajustados con K1 y con K4. K4 conserva la primera selección de episodios durante
sus pasos. K1 usa su identidad existente, cuya única lectura coincide con esa
regla. Los parámetros comunes parten de las mismas semillas. El criterio de
selección del núcleo y del lector sigue siendo MAE por sesión de validación,
con el estado inicial elegible si los ajustes empeoran.

Las rutas son K1, K2C, K2R, K4CC, K4CR y K4RC. C reutiliza la selección y los
valores proyectados. R vuelve a consultar antes del bloque siguiente. Los dos
últimos pasos de K4 comparten la decisión tomada después de K2. La política
elige una sola ruta tras K1 y no accede a etiquetas financieras para decidir.
K cuenta refinamientos, no actualizaciones de memoria neuronal.
Las letras de las rutas describen consultas episódicas. El control de
estabilidad C de CM-v1 conserva su identidad y no interviene en este estudio.

La colección comparte prefijos entre alternativas y requiere nueve
refinamientos y hasta tres búsquedas por fila. Ese coste se registra por
separado del despliegue, donde se ejecuta únicamente la ruta elegida y hay
como máximo dos búsquedas. Una traza solo se publica después de madurar la
etiqueta. La selección de M1 no depende del error financiero de una alternativa.

La puerta usa ridge para estimar cinco diferencias firmadas de MAE respecto
a K1. También aprende ganancias negativas. Las sesiones tienen el mismo peso,
repartido entre sus activos. La normalización y los rangos marginales de cada
rasgo se ajustan solo con las trazas de enero a marzo de 2022. Fuera de esos
rangos se devuelve K1. Estar dentro no acredita pertenecer al soporte conjunto
de los datos. La comprobación no es un detector OOD calibrado y ridge no produce
una garantía de confianza.

El catálogo de regularización, ganancia mínima y costes está cerrado en el
JSON. Los costes de continuación se expresan en unidades de MAE, no en
milisegundos. Cada candidato se ejecuta realmente sobre validación, incluyendo
la agrupación de filas por ruta. Se elige primero por MAE por sesión, después
por el menor número medio de refinamientos y finalmente por el orden declarado.
La calibración nunca elige la puerta. El registro de selección y sus huellas
se confirman antes de ejecutar las fases posteriores del grupo.

Los controles interpretativos incorporados son las seis rutas fijas, K4 con una
consulta por paso, un lector M0 propio, el predictor congelado sin lector,
retorno residual cero, contracción escalar hacia cero, un umbral sencillo y una
asignación por hash con frecuencias fijadas en validación. Cada control necesita
su propio recibo de ejecución y, si emite intervalos, su calibración. Una
frecuencia fijada para el control aleatorio no garantiza igualdad exacta de
cómputo en una ventana posterior. La presencia de un control en este protocolo
no implica que ya exista un resultado suyo.

El control por umbral compara el ancho del primer intervalo con los cuantiles
0,5, 0,75 y 0,9 de ese rasgo en las trazas de ajuste. Se contrasta continuar por
K2C, K4CC o K4CR. Sus nueve candidatos se seleccionan en una familia separada
de los doce candidatos ridge. El control por hash usa los recuentos de rutas
del ridge elegido en validación, incluidas las emisiones que aún no habían
madurado al cerrar el tramo. No utiliza RNG global ni los resultados de la
evaluación para asignar una ruta.

M0 ajusta su propio lector K1 con el mismo núcleo, datos y dos épocas. No es un
lector M1 al que se le retire el banco después del ajuste. K4 por paso conserva
los pesos del lector elegido y cambia explícitamente su calendario de consulta,
sin otro entrenamiento. La contracción escalar aplica al predictor sin lector
uno de los cinco coeficientes del catálogo, elegido solo con validación.

## Recibos y recuperación

`prepare` recibe el directorio que contiene `fold-017/manifest.json` y
`fold-018/manifest.json`, más un destino nuevo. Proyecta manifiestos sin
modificar las fuentes, guarda las recetas y la prueba de filas y fija huellas
del código. `verify` comprueba que no han cambiado los artefactos ni los
manifiestos originales y que siguen las mismas versiones de Python, PyTorch,
NumPy y PyArrow. `run` recibe uno de los identificadores del catálogo
de `prepared.json`. La ayuda disponible es:

```bash
uv run --no-sync python -m benchmarks.refinement_study --help
```

Las fases de núcleo y lector reutilizan los checkpoints y el mejor estado de
los entrenadores existentes. La colección, selección y predicción confirman
recibos por fase. Una fase incompleta se vuelve a recorrer desde su reinicio
declarado. Las fases confirmadas se reutilizan solo si coinciden su identidad
y los hashes de sus archivos. Nunca se reconstruye un estado usando datos de
fases posteriores.

Los Parquet de trazas se escriben en bloques de hasta 1.024 filas. Una fase
admite como máximo 131.072 trazas maduras. Se conservan identidad, instantes,
rasgos disponibles, seis predicciones, cuantiles y resultado maduro. La
comprobación exige todas las filas y objetivos esperados, sin omisiones ni
duplicados. Los Parquet se publican de forma atómica al completar la fase.

La ronda inicial fija un presupuesto acumulado de seis horas locales, con un
límite cooperativo de treinta minutos por intento. La primera semilla sirve
para medir el coste antes de continuar con las restantes. Antes de inicializar
CUDA se reserva en disco todo el saldo disponible. Al cerrar el intento se
registra el tiempo real y se libera el resto. Si el proceso muere sin confirmar
su cierre, la reserva sigue cargada e impide reanudar automáticamente. Esa
reserva es una medida contable conservadora, no tiempo de cómputo observado.

El cronómetro del intento incluye la escritura de la reserva y la preparación
del runtime CUDA. Excluye las importaciones del lanzador, la verificación del
manifiesto preparado, la adquisición del cerrojo, la escritura final de la
liquidación y la liberación de la lease. No mide el tiempo total del proceso.
Los límites de treinta minutos y seis horas son cooperativos. Puede haber un
exceso hasta alcanzar la siguiente barrera, registrado en
`cooperative_limit_overrun_seconds`. Los intentos fallidos que cierran también
descuentan su tiempo medido.

La ejecución exige `cuda:0`, FP32 estricto para el predictor y una autorización
acotada compatible con la protección vigente. `GpuLease` se adquiere dentro
del proceso científico. Comprueba la carga de cómputo ajena y la memoria libre,
con un máximo de 6 GiB de VRAM y al menos 1 GiB de margen. El supervisor puede
consultar `nvidia-smi` sin inicializar CUDA. El ejecutor no modifica la protección
de aprendizaje ni permite la campaña completa.

## Métricas y evidencia necesaria

La métrica principal es MAE residual por sesión. Las diferencias se calculan
sobre los mismos activos e instantes. El remuestreo previsto usa bloques
temporales de 16 días, 2.000 réplicas e intervalos del 95 %, con sensibilidad
predefinida a bloques de 8 y 32 días. Las semillas se describen por separado
y no multiplican el número de fechas independientes.

Los cuantiles de la política compuesta se calibran con CQR al 80 % y 95 %.
El calibrador queda fijado antes de la evaluación y no cambia la mediana.
La dependencia temporal impide presentar la cobertura como garantía universal.

Los recibos distinguen tiempo total de fase, trabajo de colección y de
despliegue, búsquedas, refinamientos, proyecciones de valores, filas y picos de
RAM y memoria CUDA registrada por el asignador de PyTorch. La latencia por
solicitud y sus percentiles requieren además una
medición del recorrido efectivo con calentamiento y sincronización adecuados.
No se deducen de dividir el tiempo de fase por el número de filas.
Las búsquedas y proyecciones se cuentan por fila. El control por paso observa
sus lecturas no vacías. M0 y el predictor sin lector registran cero operaciones
episódicas. Estos recuentos no son números de kernels ni equivalen a latencia.

El estudio estará ejecutado solo en las fases que tengan recibos completos.
La implementación y las comprobaciones con fixtures no acreditan una mejora
predictiva, superioridad frente a modelos actuales ni novedad universal.
