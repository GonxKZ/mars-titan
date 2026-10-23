# Piloto de desarrollo sobre enero y febrero de 2015

`pilot-month-v3` preparó la vista real y terminó antes de iniciar el ajuste por
rechazo de admisión CUDA. Solo usa la ventana A:
ajuste `[2015-01-01, 2015-02-01)` y validación `[2015-02-01, 2015-03-01)`, 59 días
en total. Conserva los mismos 256 activos declarados por el padre, las seis
configuraciones, la semilla 42, dos épocas y los hiperparámetros anteriores. El
protocolo v1 y su ampliación siguen preparados y sin ejecutar. El registro
`pilot-month-v2` con chunk 2 se conserva sin cambios. La única modificación de las
recetas en v3 es `chunk_size=8` en las dos configuraciones Dust.

La reducción del tramo responde al coste técnico y a una ronda breve. Se registra
antes de observar métricas predictivas de un ajuste real. Se fija chunk 8 mediante
el criterio previo de paridad y coste, con población 16, lote 32 y pérdida MAE
sintética. Un perfil posterior comparó una referencia chunk 2 sin medir tiempo con
un calentamiento y tres medidas chunk 8. Todas conservaron parámetros, entradas,
RNG y fuentes, y pasaron `rtol=2e-4` y `atol=2e-6`. No hubo pasos de optimizador ni
objetivos financieros reales. Chunk 4 sigue rechazado por el fallo de paridad
anterior, cuyo registro se conserva sin relajar las tolerancias.

Chunk 8 midió 0,480665 s de mediana por estimación, con rango de 0,383458 a
0,736923 s y 28 forwards completos, frente a los 106 de chunk 2. Ambos evalúan
6720 filas por estimación. El perfil anterior de chunk 2 había medido 1,600015 s
de mediana, entre 1,390531 y 1,643335 s. Los perfiles se ejecutaron bajo contención.
No acreditan una aceleración universal ni una comparación controlada de cargas
CPU concurrentes. El pico Torch asignado pasó de 46 392 320 a 61 218 304 bytes,
14 825 984 bytes más (14,826 MB). El pico reservado fue 62 914 560 bytes en ambos.

## Configuraciones y exposición

Los archivos de `configs/experiments/window-variants-pilot-month-v3/` son objetos
puros `WindowVariantRecipe`, sin campos de metadatos adicionales.

| Archivo | Codificador | Estimador | Optimizador | Tasa |
| --- | --- | --- | --- | --- |
| `transformer-bp-adamw.json` | Transformer | BP | AdamW | 0,001 |
| `kronlstm-bp-adamw.json` | Kron escalar | BP | AdamW | 0,001 |
| `vector-delta-bp-adamw.json` | Revisión vectorial | BP | AdamW | 0,001 |
| `transformer-dust-adamw.json` | Transformer | Dust | AdamW | 0,001 |
| `transformer-bp-sgd.json` | Transformer | BP | SGD | 0,01 |
| `transformer-dust-sgd.json` | Transformer | Dust | SGD | 0,01 |

Todas usan H=32, una capa, lote 32, FP32 estricto, pérdida MAE y selección por MAE
de sesión. La mejora mínima es `1e-5`, con el estado inicial elegible y presupuesto
fijo de dos épocas. El decaimiento es cero, el clipping tiene norma máxima uno y
SGD usa momentum cero. Hay checkpoints cada 32 actualizaciones. Dust conserva
población 16, σ=0,01, generador privado con semilla 73, chunk 8, máximo de 512
forwards por estimación y límite lógico de trabajo de 512 MiB.

Kron sustituye el codificador de precios por su bloque completo con proyecciones,
convolución causal, puertas y matriz recurrente. El control vectorial cambia la
revisión escalar por una revisión delta vectorial y conserva los demás componentes.
La matriz y la convolución empiezan de cero en cada ventana independiente de
precios. Dust mantiene el Transformer completo y estima señales para todos sus
parámetros mediante perturbaciones de activación, crédito QKV y pullbacks locales.
El optimizador aplica después esas señales. No es una perturbación global de los
pesos. Las diferencias y licencias están en las auditorías de
[KronLSTM](kronlstm/source_audit.md) y [Dust](dust/source_audit.md).

BP y Dust emparejan el mismo Transformer y optimizador. Kron y su control vectorial
comparten parámetros y cambian la revisión. H=32 no iguala el número de parámetros
entre Transformer y Kron. Las proyecciones compatibles mantienen una inicialización
común. La igualdad de exposición exige que las seis configuraciones completen las
mismas filas y dos épocas. Una ejecución parcial no satisface esa condición.

## Presupuesto y recuentos de la vista

El límite global es de 1800 segundos y empieza antes de adquisición, preparación y
admisión. Incluye ajuste, comprobaciones, evaluaciones, selección, checkpoints y salida. No
se amplía al cambiar de proceso o configuración. Cada configuración tiene además
un límite interno de 600 segundos, 1 GiB de allocator CUDA, 256 MiB de caché y
256 MiB de salida. El supervisor global prevalece. Los seis límites internos no
son una estimación del coste total.

La vista real contiene 3249 filas de train y 3078 de validación. De los 256 activos
declarados, 171 aportan filas en cada partición. Con lote 32, el lector forma
`ceil(3249/32)=102` lotes por época. Completar las dos épocas requeriría 204
actualizaciones y 6498 exposiciones lógicas a filas de train por configuración.
Son cantidades proyectadas, no ejecutadas. Se contrastarán con los cursores si se
inicia otro intento. No se deducen filas a partir del universo ni del calendario.

Como proyección parcial, los lotes completos de las dos configuraciones Dust
costarían `0,48066481901332736 * 4 * floor(3249/32) = 194,189` segundos usando la
mediana técnica.
Faltan los posibles lotes parciales, todos los controles BP y el resto del recorrido.
Esta proyección no es una cota temporal certificada ni garantiza completar el piloto.
El coste total observado y la exposición efectivamente completada se registrarán
por separado. Un timeout no autoriza ampliar los 1800 segundos ni elegir un resultado
favorable entre ejecuciones incompletas.

## Datos y objetivo heredado

El padre tiene SHA-256
`55e48db0be364fffeb94471aa969d3f8e845943828a9a264e3fc05215134df9f`
y el catálogo
`bcd5f0209be87434b29e8c3a3f438dc5680bd44bba6037d7bdcd02c5edb5ae2c`.
La vista conserva IDs, máscaras y cinco modalidades para las seis configuraciones.
No usa un objetivo de modelado de lenguaje. Las representaciones de noticias y
gráficos ya están calculadas con MiniLM y ResNet18 congelados.

La etiqueta es el residual heredado
`r_activo,t+1 − α_t − β_t r_mercado,t+1`, con retorno `close/open − 1` de la siguiente
sesión oficial. Para cada t se ajusta OLS con intercepto sobre hasta 252 sesiones,
con al menos 126 pares cuyos retornos ya estén disponibles en t. La decisión ocurre
cinco minutos después del cierre. El objetivo madura con ambos retornos siguientes.
La vista no reajusta un residualizador con enero ni congela coeficientes para febrero.
El contexto de precios de 64 sesiones y el historial residual pueden empezar antes
del primer objetivo de ajuste. Los coeficientes se actualizan también en validación
con información madura y son distintos de los parámetros compartidos del predictor.

No se carga un escalador estadístico del padre. Los precios se transforman dentro de
su propio contexto y fundamentales y macro usan logaritmo con signo, bit observado
y antigüedad. La recuperación de la frontera 2022–2023 queda fuera de esta ventana.
El factor declara `point_in_time_verified=false` y ajustes retrospectivos sin
reconstruir. Las representaciones declaran `historical_simulation=false`. El universo
congelado tampoco acredita una selección histórica de activos. El tramo corto no
elimina estas limitaciones. La preparación puede recorrer artefactos de otros años,
aunque solo admita para el ajuste y la validación las filas del intervalo declarado.

## Estado del intento y auditoría de entradas

La etapa `prepare` terminó con código 0 en 30,293 segundos y produjo la vista con
SHA-256 `b4e970cc67c629e2579c6ff050fd514ba0311ae46dc96ad8f8141c55757caa8f`.
El intento completo duró 32,085 segundos. La admisión mediante `GpuLease` rechazó
el inicio con `another_compute_process` y código 75. No se inició el proceso de
ajuste, ninguna configuración produjo actualizaciones y el contador confirmado quedó en
cero. El permiso acotado del proceso quedó cerrado y la protección global de
aprendizaje permaneció intacta. No se detuvo ninguna carga ajena.

La auditoría posterior reconcilió todas las 3249 filas de train y 3078 de
validación. Conserva huellas de los IDs, inputs y objetivos de ambas particiones
y de las seis recetas. No construyó predictores, no emitió predicciones ni aplicó
actualizaciones. Duró 5,041 segundos internamente y 7,082 segundos supervisados.
Su pico RSS de vida de proceso fue 758120448 bytes. Esa memoria corresponde a la
auditoría de entradas, no a un entrenamiento. No hay MAE financiero observado ni
comparación de resultados entre configuraciones.

El [informe técnico](../../reports/engineering/kron-window-technical-20261023/README.md)
recoge las comprobaciones CUDA, las mutaciones dirigidas y los límites de las medidas.

## Comandos

La vista se preparó una sola vez. Los comandos siguientes describen la preparación
y el ajuste, pero la etapa `run` no se inició en el intento registrado. Las
variables de padre y catálogo deben identificar los manifiestos anteriores. Las
salidas de un nuevo intento deben ser nuevas y externas al repositorio.

```bash
uv run --no-sync scripts/run_window_variant.py prepare \
  --parent "${MARS_WINDOW_PARENT:?Falta el manifiesto padre}" \
  --catalog "${MARS_WINDOW_CATALOG:?Falta el catálogo}" \
  --recipe configs/experiments/window-variants-pilot-month-v3/transformer-bp-adamw.json \
  --output "${MARS_WINDOW_OUTPUT_ROOT:?Falta la salida externa}/view-pilot-month-v3"

uv run --no-sync scripts/run_window_variant.py run \
  --manifest "${MARS_WINDOW_OUTPUT_ROOT:?Falta la salida externa}/view-pilot-month-v3/views/fold-000/manifest.json" \
  --recipe configs/experiments/window-variants-pilot-month-v3/transformer-bp-adamw.json \
  --output "${MARS_WINDOW_OUTPUT_ROOT:?Falta la salida externa}/transformer-bp-adamw-pilot-month-v3"
```

Las otras configuraciones cambian el archivo de receta y la salida. Los comandos
individuales necesitan la supervisión global conjunta de 1800 segundos. `--resume`
es explícito y sigue dentro del presupuesto pendiente. El CLI no modifica la
protección de aprendizaje. La reserva CUDA conserva su comprobación de concurrencia,
sin detener cargas ajenas ni saltar el control. No se activan campañas, reanudaciones
automáticas ni el test final de 2024. No hay una mejora predictiva observada de este
piloto.

El CLI consulta y reserva CUDA antes de admitir la vista. El ejecutor comprueba todas
las filas antes de construir el predictor y aplicar el primer paso del optimizador.
Si se interrumpe la admisión de una reanudación, `admission-attempt.json` conserva el
último intento. El informe anterior y los checkpoints confirmados siguen intactos.
