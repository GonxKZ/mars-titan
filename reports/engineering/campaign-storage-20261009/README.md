# Presupuesto de disco de la campaña A

Fecha: 9 de octubre de 2026. Tarea [#363](https://github.com/GonxKZ/mars-titan/issues/363), rama `feat/campaign-storage-budget`.

Este informe estima el disco que ocuparían la campaña A y sus etapas posteriores, mide alternativas de retención sin pérdida para los consumidores y deja implementada una guardia de disco. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto, y no se ha leído ningún objetivo, muestra ni modelo. De las vistas publicadas solo se leen los manifiestos, con las filas de cada activo y tramo. Las cifras usan GB decimales (10⁹ bytes).

## Resumen

- A las 16:44 UTC quedaban 55,7 GB libres en `/`. Con el margen declarado de 8 GiB (8,6 GB) quedan 47,1 GB utilizables.
- Con el diseño declarado hoy (4.680 ajustes de la campaña base con la declaración ampliada, 4.185 predicciones de la ablación de modalidades, 3.915 ajustes de adaptadores y la etapa de políticas), todo ocuparía 884 GB con el formato actual y 294 GB con la opción más compacta que conserva lo que leen los consumidores. Ninguna opción cabe.
- Solo la campaña base ocupa 456,7 GB sin liberar nada, 264,5 GB con la liberación al confirmar que implementa esta rama y 63,9 GB con la opción más compacta. Su pico mínimo es 94,5 GB.
- Hay dos límites que no resuelve ningún formato: la caché de páginas de XGBoost en la ventana US+CN más poblada (31,9 GB mientras dura el ajuste) y el corpus ordenado de la etapa de adaptadores en esa misma ventana (unos 160 GB transitorios).
- El usuario ha cambiado el diseño a un modelo conjunto US+CN desde 2004 con controles separados para tres familias. La [fórmula](#fórmula-para-el-diseño-conjunto) y la orden `storage --counts` recalculan todo en cuanto se publiquen los recuentos.

## Qué se ha medido

| Componente | Medida |
| --- | --- |
| Tablas por fila de la campaña base | Parquet sintético por ventana y tramo con el esquema, los tipos, el orden y los grupos de filas de cada escritor (`training/storage_measurements.py`) y las filas reales de cada activo de las 45 ventanas publicadas. Las claves y los instantes reproducen la estructura real. Los decimales son aleatorios, así que su compresión es una cota superior |
| Disposiciones alternativas | Grupos grandes (1.048.576 filas, ZSTD 3, BYTE_STREAM_SPLIT en decimales y diccionario solo en texto e instantes), tabla común por ámbito, ventana y tramo con decimales propios por ajuste, y la misma con índice de fila. Cada archivo se relee y se compara bit a bit con el original antes de contarlo |
| Agregados por sesión y estrato | Tabla por sesión, mercado y cuatro estratos de presencia más el total, con recuentos, errores absolutos y cuadráticos, matriz direccional 2×2, pinball de los cinco cuantiles, cobertura al 80 y 95 % y Rank IC del total |
| Índices de observaciones | Muestras y etiquetas maduras de la ventana más poblada de cada ámbito ordenadas con DuckDB (grupos de 2.048 filas, ZSTD) |
| Adaptadores | Esquema de `posttraining.heldout` con el centro y los cuantiles en float64 y el padre en float32 |
| Estados | 12,0 bytes por parámetro más 38.423 de cabecera, medidos en un estado real de 50.881 parámetros con AdamW (648.995 bytes). XGBoost usa la cota del inventario para profundidad 6 y 2.000 rondas (9,4 MB por modelo), sin medir |
| Caché de XGBoost | Páginas ELLPACK de `models/baselines/external_boosting.py` (hipótesis de bits locales) y caché de validación de 6.744 bytes por fila con tope de 16 GiB |
| Cintas de políticas | Cinta de 252 sesiones y 128 activos con el esquema de `simulation/storage.py` |

PyArrow 25.0.1, DuckDB 1.5.5, NumPy 2.5.3 y Python 3.12.14. Las medidas por ventana están en [`window-measurements.json`](window-measurements.json), los coeficientes en [`row-bytes.json`](row-bytes.json), las medidas fuera del ajuste base en [`extras.json`](extras.json) y la estimación completa en [`estimate.json`](estimate.json).

## Bytes por fila

Máximo entre las 45 ventanas. Orden de cada celda: validación, calibración y evaluación.

| Escritor | Actual | Grupos grandes | Tabla común | Tabla común con índice de fila |
| --- | --- | --- | --- | --- |
| Referencias neuronales (cuantiles float32) | 55,00 / 53,76 / 57,12 | 31,29 / 31,58 / 30,81 | 17,76 / 17,81 / 17,69 | 22,03 / 22,38 / 21,10 |
| Ridge (float64) | 24,43 / 23,65 / 24,79 | 17,15 / 17,46 / 16,64 | 7,19 / 7,26 / 7,19 | 11,52 / 11,83 / 10,61 |
| XGBoost (float32) | 20,46 / 19,68 / 20,82 | 13,64 / 13,90 / 13,15 | 3,71 / 3,71 / 3,70 | 8,05 / 8,36 / 7,14 |
| GRU episódica | 48,02 / 47,88 / 48,21 | 32,43 / 32,39 / 32,47 | 17,71 / 17,82 / 17,70 | 22,58 / 22,60 / 22,52 |
| Titans-MAC, MARS-TITAN y CM-v1 | 55,27 / 55,08 / 55,43 | 32,77 / 32,81 / 32,84 | 18,23 / 18,48 / 18,14 | 22,97 / 22,98 / 22,91 |
| Tabla común (claves y objetivo), una por tramo | | | 9,99 / 10,23 / 9,48 | |

Otros coeficientes: agregados por sesión 0,59 bytes por fila sustituida, índice ordenado 4,16 bytes por evento (dos eventos por muestra) más 12,17 mientras se construye, adaptadores 99,0 bytes por fila con el formato actual, 65,0 con grupos grandes y 39,2 con la tabla común.

El formato actual escribe un grupo de filas por lote del escritor (256 filas en las referencias neuronales) con diccionario en todas las columnas. Esa es la mayor parte del exceso. La tabla común guarda una sola vez `sample_id`, `asset_id`, `market`, `prediction_at` y `target`. Cada ajuste guarda sus cinco cuantiles, o su predicción si es puntual, y se reconstruyen `prediction` (los mismos bits que la mediana) y `zero` (cero positivo, comprobado). Los escritores cronológicos ordenan por instante y los demás por activo, así que sin índice de fila la tabla reconstruida sale en orden canónico (`asset_id`, `prediction_at`). Todos los consumidores actuales ordenan o agrupan por clave. Si hiciera falta el orden físico original, el índice de fila cuesta entre 1,7 y 4,4 bytes más por fila. No se reduce la precisión de ningún escritor. La familia Titans guarda en float64 valores calculados en float32, y la tabla común los conserva en float64.

## Campaña base

4.680 ajustes en 45 ventanas (19 en US, 13 en CN y 13 en US+CN), 104 por ventana. Las filas reservadas suman 46,9 millones por brazo y semilla y 4.878 millones entre todos los ajustes. El pico recorre los trabajos en el orden del plan, uno a uno como ejecuta hoy la campaña.

| Escenario | Conservado (GB) | Pico (GB) |
| --- | ---: | ---: |
| Formato actual sin liberar nada | 456,7 | 470,1 |
| Formato actual con liberación al confirmar (implementada) | 264,5 | 288,5 |
| Grupos grandes en todas las tablas | 165,3 | 192,5 |
| Tabla común en todas las tablas | 102,1 | 131,3 |
| Filas solo donde las leen, grupos grandes, agregados en el resto | 92,7 | 122,3 |
| Filas solo donde las leen, tabla común, agregados en el resto | 63,9 | 94,5 |

Sin liberar nada, el formato actual ocupa 239,5 GB de tablas, 181,2 GB de índices de observaciones, 34,3 GB de estados y 1,7 GB de informes y resúmenes. La liberación borra los índices, que se reconstruyen desde la vista, y deja un estado por ajuste salvo en XGBoost, cuyos tres modelos (17,8 GB en total con la cota sin medir) siguen porque la reanudación de un intento completo vuelve a cargar el de recuperación. El pico de todos los escenarios es un finalista de XGBoost en US+CN fold-012, con 27,8 GB de páginas y 4,1 GB de caché de validación sobre lo ya conservado.

Por familia, en GB:

| Familia | Ajustes | Actual sin liberar | Actual | Tabla común | Filas necesarias con tabla común |
| --- | ---: | ---: | ---: | ---: | ---: |
| Referencias neuronales | 900 | 57,8 | 54,7 | 19,3 | 11,8 |
| Ridge y XGBoost | 765 | 34,7 | 34,7 | 21,5 | 18,9 |
| Titans-MAC | 720 | 89,5 | 42,8 | 15,3 | 9,4 |
| GRU episódica | 135 | 16,5 | 7,6 | 3,4 | 2,7 |
| MARS-TITAN | 1.080 | 127,9 | 62,0 | 20,9 | 11,9 |
| CM-v1 (brazos) | 720 | 85,3 | 41,4 | 13,9 | 7,9 |
| CM-v1 (núcleos auxiliares) | 360 | 45,0 | 21,5 | 7,8 | 1,2 |

Por ámbito, el formato actual con liberación ocupa 123,0 GB en US, 25,3 GB en CN y 116,3 GB en US+CN.

### Retención real de estados

| Modelo | Mientras corre | Tras el recibo, con liberación |
| --- | --- | --- |
| Referencias neuronales | Dos recientes, el mejor y la última época fijada (hasta 3) | El elegido |
| Titans-MAC, MARS-TITAN, CM-v1 y GRU episódica | Dos recientes y el mejor | El elegido |
| XGBoost | Dos modelos de recuperación y el elegido | Sin cambios |
| Ridge | Un modelo | Sin cambios |
| Adaptadores | Dos recientes y el mejor | Sin cambios (la etapa no libera) |

`release_recovery_states` comprueba la huella y el tamaño del estado elegido, publica un `latest.json` que solo lo nombra y después borra los demás estados de esa ejecución. Si no hay un estado íntegro no toca nada, así que nunca desaparece el único estado confirmado.

## Etapas posteriores

| Etapa | Escenario | Conservado (GB) | Pico propio (GB) |
| --- | --- | ---: | ---: |
| Ablación de modalidades, 4.185 predicciones de evaluación | Formato actual | 144,1 (131,8 sin índices) | 144,2 |
| | Grupos grandes | 86,7 (74,4 sin índices) | 86,8 |
| | Tabla común de la campaña base | 54,3 (42,0 sin índices) | 54,4 |
| Adaptadores, 3.915 ajustes | Formato actual | 471,8 | 611,7 |
| | Grupos grandes | 333,0 | 479,3 |
| | Tabla común | 227,6 | 378,6 |
| | Calibración y evaluación por fila, validación agregada | 183,9 | 337,0 |
| Políticas, 2.640 cintas | Cintas | 4,0 | |

La ablación evalúa 2.538 millones de filas. Sus traslados de Titans-MAC escriben un índice de la evaluación con su calentamiento (12,3 GB en total) que la etapa no libera y nadie vuelve a leer.

En los adaptadores el pico lo marca el corpus ordenado de US+CN fold-012: 66,0 GB del archivo ordenado de entrenamiento y validación (4.330 bytes por fila) y 93,8 GB de la entrada sin ordenar (6.409 bytes por fila) mientras se prepara. Con este disco la etapa no puede ejecutarse en las ventanas grandes sin otro diseño de su lectura. Los adaptadores para todas las familias que prepara `feat/adapters-all-families` multiplican además el número de ajustes.

Los ejecutores de ajuste de PPO y KLPO no existen todavía, así que sus estados y salidas no se pueden estimar.

## Total y margen

| Opción | Base | Ablación | Adaptadores | Políticas | Total (GB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Formato actual con liberación | 264,5 | 144,1 | 471,8 | 4,0 | 884,4 |
| Más compacta sin pérdida para la comparación | 63,9 | 42,0 | 183,9 | 4,0 | 293,8 |

Frente a 47,1 GB utilizables, ni la campaña base sola cabe. Con el diseño conjunto los recuentos bajarán, pero el pico de XGBoost depende de las filas de entrenamiento de la ventana más poblada y no del número de ajustes: en US+CN fold-012 exige 31,9 GB libres sobre el margen en el momento del ajuste.

## Fórmula para el diseño conjunto

Para cada ventana *w* con filas reservadas *n*ᵥ, *n*꜀ y *n*ₑ (validación, calibración y evaluación), filas de entrenamiento *t* y *f* activos, y para cada ajuste *j* de esa ventana con escritor *k* y modelo *m*:

- Conservado con el formato actual y liberación: Σⱼ [ Σₚ *b*ₖ,ₚ·*n*ₚ + *S*ₘ·*K*ₘ + 0,26 MB + 0,52 MB si es neuronal ], con *b* de la columna «Actual», *S*ₘ el estado de la declaración y *K*ₘ = 1 salvo XGBoost (3).
- Conservado con la tabla común y solo las filas necesarias: Σ_tramos leídos *s*ₚ·*n*ₚ + Σⱼ [ Σ_{p ∈ P(j)} *c*ₖ,ₚ·*n*ₚ + 0,59·Σ_{p ∉ P(j)} *n*ₚ + *S*ₘ·*K*ₘ + 0,26 MB ], con *c* de la columna «Tabla común», *s* la tabla común y P(j) = {calibración, evaluación} para el elegido de un brazo de cuantiles, {evaluación} para el elegido de un brazo puntual y ∅ para los casos no elegidos y los núcleos auxiliares.
- Pico: lo conservado hasta el trabajo más su término transitorio. XGBoost: ⌈*t*·1.686·⌈log₂(max_bin+1)⌉/8⌉ + mín(6.744·*n*ᵥ, 16 GiB). Familia Titans: 2·67.174·*f* bytes de estado a mitad de época y 2·12,17·*t* bytes del índice mientras se construye. Al compactar después de escribir, la tabla original del ajuste.

Como referencia, un ajuste en una ventana del tamaño de US+CN fold-012 (2,13 millones de filas reservadas) escribe:

| Escritor | Actual (MB) | Grupos grandes (MB) | Tabla común (MB) | Solo calibración y evaluación (MB) |
| --- | ---: | ---: | ---: | ---: |
| Referencia neuronal | 119,5 | 66,2 | 37,8 | 27,0 |
| Ridge | 52,3 | 36,0 | 15,3 | 11,0 |
| XGBoost | 43,8 | 28,6 | 7,9 | 5,6 |
| GRU episódica | 102,6 | 69,2 | 37,8 | 27,0 |
| Titans-MAC, MARS-TITAN y CM-v1 | 118,0 | 70,0 | 38,9 | 27,7 |
| Tabla común de los tres tramos, una por ventana | | | 20,8 | 14,7 |

La orden recalcula todas las etapas con la configuración nueva y un JSON de recuentos por ámbito y ventana (`train`, `validation`, `calibration`, `evaluation` y `flows`), sin volver a escribir tablas:

```bash
uv run --no-sync python scripts/run_masked_campaign.py storage \
  --campaign <campaña conjunta> \
  --storage configs/baselines/historical-masked-campaign-storage.json \
  --counts <recuentos.json> \
  --row-bytes reports/engineering/campaign-storage-20261009/row-bytes.json \
  --extras reports/engineering/campaign-storage-20261009/extras.json \
  --ablation-stage <ablación> --adapter-stage <adaptadores> --rl-stage <políticas> \
  --output <estimación.json>
```

Con vistas ya preparadas, `--views ÁMBITO=<vistas>` sustituye a `--counts` y mide las ventanas que falten con `--cache` y `--scratch`. `--extensions` solo hace falta mientras las familias sigan en la declaración ampliada.

## Qué leen los consumidores

| Consumidor | Qué lee | De qué trabajos |
| --- | --- | --- |
| Comparación walk-forward | Claves, objetivo, predicción y cuantiles de evaluación, y de calibración en los brazos de cuantiles para el calibrador CQR. Estratos de modalidades y bootstrap por bloques de sesiones sobre esas filas | El elegido de cada brazo, semilla y ventana |
| Ablación de modalidades (#423) | El estado elegido. Escribe su evaluación y la compara fila a fila con la evaluación base, corregida con el calibrador original | El elegido |
| Selección por validación | `session_mae` del informe del ajuste, no las filas | Todos los casos de búsqueda |
| Adaptadores | El estado elegido del padre. Cada adaptador lee sus propias tablas al confirmar | El elegido como padre |
| Etapa de políticas y KLPO | `asset_id`, `market`, `prediction_at` y `prediction` de la evaluación (`window_tapes.segment_predictions`) y los recibos de ventana | El predictor elegido |
| Recibos y reanudación | Huella de calibración y evaluación (`_Campaign.confirmed`). Los traslados y lectores de la familia Titans comprueban la huella de todas las predicciones del informe del padre, validación incluida (`titans_walk_forward._verify`) | Todos los confirmados y los padres |
| Nuevos consumidores en curso | Cartera long-short por cuartiles y adaptadores en la comparación (`feat/evaluation-completeness`) leen evaluación y calibración del elegido. La serie de patrimonio por sesión (`fix/rl-realism-reporting`) sale de las políticas | El elegido |

Para no tocar varios lectores si se compacta, se ha propuesto en [#363](https://github.com/GonxKZ/mars-titan/issues/363#issuecomment-6085070321) que todas las lecturas pasen por `walk_forward_comparison._read_predictions(record, columns)`.

## Filas necesarias según el protocolo

Citas literales que fijan qué filas lee la comparación:

- [Plan de la campaña](../../../docs/research/training-campaign-2000.md#ejecución-y-recuperación): «`sources` publica el manifiesto de un ámbito para `evaluation.walk_forward_comparison`. Elige para cada brazo, semilla y ventana el ganador de la búsqueda, el finalista o la predicción trasladada».
- [Comparación declarada](../../../configs/evaluation/historical-masked-2000-comparison.json): `"partition": "evaluation"` y, en la calibración, `"method": "cqr_symmetric_score_v1"` con `"partition": "calibration"`. Las semillas son `[42, 43, 44]` en todos los brazos salvo Ridge (`[42]`).
- [Protocolo](../../../docs/research/protocol.md#particiones-ajuste-y-calibración): «La validación se usa para hiperparámetros, selección de modelo y umbral de escritura. La calibración se reserva para intervalos y abstención.»
- `training/masked_campaign.py` registra en cada recibo solo `COMPARED = ("calibration", "evaluation")`.
- Coordinación de `feat/evaluation-completeness` en #363: «La comparación solo lee el caso elegido de cada brazo con 42, 43 y 44. [...] Los casos de búsqueda con solo la 42 no entran en la comparación. Quedan en los recibos de selección y en el registro de ensayos.»

Dos citas apuntan en sentido contrario y el usuario debe valorarlas:

- [Protocolo](../../../docs/research/protocol.md#particiones-ajuste-y-calibración): «Se registra cada intento, incluidos errores y descartes.» Los agregados conservan el registro de cada intento (informe, recibo, métricas por sesión y estrato), pero no sus filas.
- [Protocolo](../../../docs/research/protocol.md#memoria-y-orden-de-actualización): «Emitir y conservar las predicciones de **todos** los activos de esa sesión con ese estado.» Describe el orden de predicción y escritura en línea. Una lectura estricta pediría conservar las filas de cada ajuste, también de los no elegidos.

Con esas reglas, necesitan filas la evaluación del elegido de cada brazo, semilla y ventana y la calibración de los elegidos de brazos de cuantiles. No las necesitan los casos de búsqueda no elegidos, la validación de ningún ajuste ni los núcleos auxiliares de CM-v1. En la campaña base declarada hoy necesitan filas 2.880 ajustes y no las necesitan 1.800: 1.530 casos de búsqueda que no serán los elegidos (todos menos uno por búsqueda) y los 270 ajustes elegidos de los núcleos auxiliares de CM-v1. Como el ganador de una búsqueda no se conoce hasta confirmar todos sus casos, la estimación conserva sus filas hasta el último caso.

## Opciones para decidir

Ninguna de estas opciones se ha aplicado. La rama solo implementa la liberación de índices y estados de recuperación, que no cambia ningún valor leído.

1. **Tabla común y decimales propios para todo.** Campaña base 102,1 GB conservados y 131,3 de pico, ablación 42,0 sin índices y adaptadores 227,6. Las lecturas devuelven los mismos bits, comprobado por escritor en las pruebas y en cada medida. Cambian las huellas de archivo de los recibos y el orden físico. Hace falta implementar la escritura o la compactación tras el recibo, un campo `rows` en los recibos, la lectura en `_read_predictions`, `segment_predictions` y `_verify`, y la comprobación de las etapas que leen sus propias tablas. **No cambia lo que la comparación puede calcular.**
2. **Agregados por sesión y estrato en lugar de filas donde nadie las lee.** Combinada con la opción 1, la campaña base queda en 63,9 GB conservados y 94,5 de pico, y los adaptadores en 183,9. Con grupos grandes en lugar de la tabla común, 92,7 y 122,3. La comparación declarada no cambia, porque no lee esas filas. **Sí cambia lo que se puede calcular después** sobre los casos no elegidos y la validación: métricas o estratos nuevos, otra calibración, contrastes emparejados entre casos de búsqueda, una auditoría de la selección con un criterio por fila distinto de los agregados guardados y la comprobación bit a bit de esos archivos. `_verify` de la familia Titans dejaría de poder comprobar la validación del padre. Las dos citas anteriores del protocolo son la razón para que lo decida el usuario.
3. **Filas solo del elegido.** Es la regla que aplican los escenarios «filas necesarias». Si se adopta, conviene declararla antes de lanzar, en la configuración de la campaña y en `docs/research/training-campaign-2000.md`, porque afecta al registro de ensayos.

Otras decisiones con su efecto medido:

- **Caché de XGBoost.** 31,9 GB transitorios por ajuste en US+CN fold-012 con la hipótesis de bits locales, hasta 58,6 GB con la cota de bits globales. Hace falta disco libre en ese momento, otra ubicación para las páginas o un orden del plan que adelante XGBoost cuando aún hay espacio.
- **Modelos de recuperación de XGBoost.** Liberarlos tras el recibo ahorraría hasta 11,9 GB con la cota. Las predicciones trasladadas y la ablación solo leen el modelo elegido, pero la reanudación de un intento completo carga el de recuperación, así que habría que cambiar ese camino.
- **Índices de la ablación.** Liberarlos tras su recibo ahorraría 12,3 GB sin perder información, igual que en la campaña base.
- **Etapa de adaptadores.** El corpus ordenado de unos 160 GB por ventana grande exige otro diseño de lectura o un volumen distinto, sea cual sea el formato de sus tablas.
- **Ranuras GPU.** El pico supone trabajos de uno en uno. Con varias ranuras, la guardia reserva la huella de cada trabajo admitido hasta su recibo, pero dos ajustes de XGBoost simultáneos en las ventanas grandes no caben.

## Guardia de disco y liberación

`configs/baselines/historical-masked-campaign-storage.json` declara el margen (8 GiB), la frecuencia de consulta (5 s), la liberación al confirmar y los bytes medidos. `run_masked_campaign.py run` exige `--storage`:

- Antes de crear la salida, `run` recorre los trabajos pendientes en el orden del plan y se niega a empezar si el pico proyectado más el margen supera el espacio libre (`DiskBudgetError`).
- Antes de cada trabajo, si su huella completa más lo reservado por los trabajos en curso no deja el margen, la campaña se detiene en `paused` sin empezarlo (`DiskPaused`). Cada trabajo admitido reserva su huella hasta su recibo, para que varias ranuras no cuenten con el mismo espacio.
- Durante un trabajo, un espacio libre por debajo del margen activa la misma parada recuperable que una señal, en la siguiente barrera del ejecutor. La consulta se limita a una cada `check_seconds`.
- Tras cada recibo se borran los índices de observaciones, los estados de recuperación y los restos de cachés de XGBoost, y `released.json` registra lo liberado. Es idempotente.

El resumen de la campaña guarda la proyección del lanzamiento, el estado de la guardia y el motivo de una pausa.

## Coste de CPU

Escritura y relectura del tramo de evaluación de US+CN fold-012 (1.221.822 filas), mediana de cinco repeticiones con dos hilos, en nanosegundos por fila ([`layout-timing.json`](layout-timing.json)):

| Escritor | Escribir actual | Escribir grupos grandes | Separar, comprobar y escribir tabla común | Leer actual | Leer grupos grandes | Leer y reconstruir tabla común |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Referencia neuronal | 1.240 | 179 | 864 | 791 | 57 | 53 |
| Titans | 677 | 239 | 664 | 91 | 54 | 57 |
| Ridge | 365 | 123 | 678 | 150 | 42 | 44 |

Los grupos grandes escriben entre 2,8 y 6,9 veces más rápido que el formato actual y se leen hasta 14 veces más rápido. La tabla común cuesta alrededor de un segundo por tramo y ajuste, sobre todo por la comprobación de claves, y se lee tan rápido como los grupos grandes. Frente a las horas de un ajuste, es despreciable.

## Inventario de lo liberable

Solo informativo. No se ha borrado nada. Los tamaños exclusivos cuentan cada inodo una vez y solo cuando todos sus enlaces duros están en el conjunto medido, porque los `.venv` comparten archivos con `~/.cache/uv`. Detalle en [`liberable-inventory.json`](liberable-inventory.json).

| Elemento | `du` (GB) | Exclusivo (GB) | Estado |
| --- | ---: | ---: | --- |
| Worktrees de tarea fusionados y limpios sin referencias: `native-policy-real-tapes-20261009` (#420, 0,97), `modality-ablation-20261009` (#423), `rl-stage-pending-20261009` (#422), `later-stages-cn-rules` (#421) y `docs-sync-409-417` (#419) | 1,17 | 1,17 | Sin procesos, servicios ni referencias en `~/.local/state/mars-titan` |
| Worktrees fusionados y limpios con referencias o conservados expresamente: `research-learning-20261005`, `financial-session-v2-20261008`, `klpo-actor-updates-20261009`, `cuda-checks-20261009` y `targets-dictionary-sessions-20261009` | 1,57 | 1,57 | Aparecen en recibos o son runtimes de comprobaciones |
| 74 runtimes históricos con HEAD separado, sin servicios ni procesos. Los mayores: `synthetic-episodes` (1,40), `issuer-announcements-20261007` (0,82), `hpc-corpus-batches` (0,60), `episodic-query-20261004` (0,48) y `research-memory-20261005` (0,35) | 14,88 | 7,17 | Todos tienen referencias en recibos de estado |
| 6 runtimes con servicios de systemd o procesos (`analysis-runtime`, `klpo-memory-runtime`, `observatory-runtime`, `real-runtime`, `strict-completion-runtime` y `strict140-runtime`) | 7,94 | 0,42 | En uso |
| 14 worktrees de trabajos en curso | | | En uso |
| Caché de uv (`~/.cache/uv`) | 14,69 | 1,22 | El resto comparte inodos con los `.venv` |
| Los dos primeros grupos de worktrees, los runtimes sin servicios y la caché de uv juntos | 23,26 | 9,56 | |
| `/tmp/pytest-of-gonzalo` | 1,82 | | pytest conserva las tres últimas sesiones y otros agentes las usan |
| `tmp/` del repositorio principal | 0,29 | | |
| `views-failed-empty-rowgroup-20261009T1430Z` y `targets-v3-failed-categorical-20261009T1352Z` | 0,005 | | Intentos fallidos conservados |
| `prepared-v2` | 4,93 | | El manifiesto de la edición v3 lo referencia, no es liberable sin decisión |

`data/` del repositorio principal ocupa 119,5 GB y no se ha inventariado por dentro.

## Comprobaciones

- Pruebas nuevas en CPU, sin GPU ni pasos de optimizador: `tests/training/test_storage_measurements.py` (19), `tests/training/test_campaign_storage.py` (19) y `tests/training/test_storage_budget.py` (9). Cubren el esquema y el orden de cada escritor, la relectura bit a bit con ceros con signo, subnormales, extremos y nulos, el rechazo de tablas que no se pueden reconstruir, la contabilidad de cada escenario, la búsqueda abierta en el pico, la fórmula, la liberación sin perder el estado elegido, la reserva de la guardia y la campaña con dobles que rechaza el lanzamiento, pausa antes de un trabajo o a mitad de él y reanuda sin repetir trabajos.
- Mutación dirigida: 51 mutantes sobre la declaración, la huella, la guardia, la liberación, las disposiciones, la estimación y la integración en la campaña. Tras reforzar las pruebas, todos detectados ([`mutation.json`](mutation.json)).
- Áreas afectadas (campaña con máscaras, Titans, MARS-TITAN, CM-v1, GRU candidata, adaptadores, ablación, etapa de políticas, comparación y checkpoints): 569 correctas y 73 omitidas. La guarda global de las pruebas omitió el único paso de AdamW antes de modificar pesos. Fallan por el entorno y no por esta rama la variante `cuda:0` de `test_checkpoint_recovery`, porque la GPU estaba oculta a propósito, y nueve pruebas de MARS-TITAN y CM-v1 que necesitan el enlace episódico nativo compilado, ausente en este worktree.

No se ha ejecutado nada en `cuda:0`.
