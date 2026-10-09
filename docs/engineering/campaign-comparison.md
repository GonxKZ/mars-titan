# Analizar campañas congeladas

La comparación tiene tres pasos separados: admisión de recibos y predicciones, controles financieros sin aprendizaje y exportación de agregados. No carga pesos, no inicia entrenamientos y no abre la reserva final. Los resultados del 5 de octubre están en el [informe de campañas](../../reports/baselines/campaign-comparison-20261005.md).

## Predicción real

`evaluation/comparison_sources.py` enlaza los cierres, manifiestos temporales, selecciones, padres y archivos evaluados. Rechaza rutas externas no autorizadas, enlaces simbólicos, hashes discordantes, poblaciones diferentes y particiones distintas de calibración y evaluación. Comprueba que las rejillas proceden del entrenamiento del mismo manifiesto. El único metadato externo admitido es el manifiesto temporal explícito y verificado por su huella. No sigue sus rutas hacia datasets.

La política de entradas se declara y no se deduce. `predictive_sources`, `compare_campaigns` y `evaluate_campaign_reliability` reciben `input_policy` (en la línea de órdenes, `--input-policy`), que por defecto es la estricta `strict_inputs_v1`. Cada manifiesto debe adherirse exactamente a la política declarada y sus vistas deben tener la versión que le corresponde: 1 para la estricta y 2 para `historical_masked_2000_v1`. Una vista con máscaras se rechaza si no se declara su política y una vista estricta se rechaza bajo la política con máscaras. Con la política estricta las salidas no cambian. Con la de máscaras la procedencia añade `input_policy` y `mask_contract`. Esta ruta sigue esperando campañas con búsqueda de referencias y continuación completas. La comparación de la campaña desde 2000 usa la [evaluación walk-forward](../research/metrics.md#evaluación-walk-forward-de-la-edición-desde-2000).

`prediction_statistics.py` comprueba primero las fechas Parquet y después solicita objetivos y predicciones. El límite es un millón de filas y 512 MiB por archivo, también para su tamaño descomprimido declarado. Verifica identidades, valores finitos y unicidad de activo y sesión. Reordena por identificador para comprobar igualdad exacta de la población, sin depender del orden físico de los activos.

El análisis usa el [MAE por sesión](../research/metrics.md). Las métricas archivadas se reconstruyen con tolerancias absoluta de 10⁻¹² y relativa de 10⁻¹⁰. Para comparar inferencia del padre con diferentes lotes se declaran tolerancias de 10⁻⁶ y 10⁻⁵ en referencias neuronales y boosting, y de 10⁻⁹ y 10⁻¹⁰ en Ridge. En la campaña revisada la diferencia observada del padre fue cero.

La política inicial se reconstruye con la rejilla congelada y el centro continuo del padre. Para valores normalizados `z` y centro `m`, los logits son `z*m - z²/2`. Se usa `log_softmax` de SciPy y la mediana de `ActionGrid`, incluida su regla para empates. La caché utiliza una huella del vector completo y de la rejilla, devuelve matrices de solo lectura y retiene como máximo 32 entradas y 8 MiB de valores. `--initial-cache-mib 0` permite ejecutar la referencia sin reutilización.

Desde un checkout con acceso a las campañas locales:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync \
  python -m mars_titan.evaluation.campaign_comparison \
  --reference data/interim/strict140-us-convergence-20261001 \
  --completion data/interim/strict140-us-convergence-completion-20261001 \
  --output artifacts/analysis/comparison-20261005
```

La salida debe ser nueva y estar separada de las fuentes. Un fallo no reemplaza un resultado anterior. `comparison.json` se confirma al terminar y contiene las huellas de los CSV y del Parquet de pérdidas. Un directorio parcial sin ese recibo no acredita una comparación terminada.

`cases.csv` conserva identidad, selección, recursos disponibles y diagnósticos de cada estado. `methods.csv` contiene medias descriptivas por método, `folds.csv` sus valores por ventana e `intervals.csv` los contrastes por longitud de bloque. `session-errors.parquet` conserva pérdidas derivadas por sesión para repetir el cálculo, sin copiar las modalidades originales.

El remuestreo usa bloques móviles sin envolver el final del periodo. Con `n` sesiones y longitud `L`, sortea `ceil(n/L)` comienzos entre cero y `n-L`, concatena los bloques y recorta a `n` sesiones. Los pesos de repetición se calculan con NumPy y se multiplican por todas las columnas a la vez. Se procesan 256 réplicas por lote. El límite es 10.000 réplicas, cuatro millones de celdas de entrada y diez millones de celdas de trabajo. Las semillas comparten índices y se promedian sus errores antes de remuestrear. El resultado no es un ensemble ni un intervalo combinado de folds.

## Controles financieros C++20

`mars-titan-financial-controls` reutiliza Arrow C++, el lector y la sesión financiera existentes. No inicia Python ni enlaza LibTorch. Evalúa efectivo, compra inicial tras calentamiento y rebalanceos al 25 %, 50 %, 75 % y 100 %. El ranking de activos y la contabilidad son los mismos que los de los comparadores. La primera decisión elegible es la sesión 64 y la ejecución ocurre en la apertura siguiente.

La CLI requiere `--campaign`, `--scenarios`, `--output` y las tres huellas brutas `--campaign-sha256`, `--freeze-sha256` y `--identity-sha256`. Estas huellas autorizan un conjunto previamente revisado. No se confunden con la huella canónica del contenido JSON. El catálogo y los recibos de auditoría deben corresponder a esa congelación. La CLI rechaza una salida dentro de las fuentes, incluso con rutas relativas.

La ejecución carga un mundo cada vez y lo reutiliza para seis políticas y tres costes. Hay un trabajador, sin paralelismo adicional ni GPU. Los límites admiten hasta 512 mundos de auditoría y una salida de 16 MiB. SIGINT conserva los mundos completos y marca el recibo como interrumpido. Esta herramienta de pocos segundos vuelve a empezar sobre una salida nueva, no promete reanudar ese análisis. Las campañas de entrenamiento mantienen sus propios checkpoints recuperables.

Los perfiles `native-controls-release`, `native-controls-static-analysis`, `native-controls-asan-ubsan` y `native-controls-coverage` incluyen el objetivo. Para compilar y comprobar la ruta Release:

```bash
cmake --preset native-controls-release -S native
cmake --build build/native/native-controls-release --parallel 1 \
  --target mars-titan-financial-controls financial_controls_tests
ctest --test-dir build/native/native-controls-release -R '^financial_controls' \
  --output-on-failure
```

Los presets se encuentran en `native/CMakePresets.json`. Los comandos anteriores se ejecutan desde la raíz del repositorio.

`evaluation/financial_comparison.py` recibe `--campaign`, `--controls`, `--controls-sha256` y `--output`. Verifica selección, auditoría, mundos, semillas, costes, calentamiento y episodios completos antes de agregar. `aggregates.csv` conserva cada semilla. `paired_summary.csv` promedia primero las semillas dentro de cada mundo y calcula diferencias frente a cada control sobre la misma población. `paired_worlds.csv` permite un análisis posterior sin volver a leer trazas. No incluye intervalos financieros ni multiplica el tamaño muestral por semillas o costes.

## Exportación y comprobaciones

`scripts/export_campaign_comparison.py` requiere `--predictive` y una salida nueva `--output`. `--financial` es opcional, por lo que una comparación predictiva real puede publicarse por separado. `--reliability` incorpora los diagnósticos de `campaign_reliability` cuando pertenecen a la misma campaña, modelos, checkpoints y predicciones. No calcula otra vez las métricas ni sus intervalos.

Las ventanas, fechas, réplicas y poblaciones proceden de los recibos. El exportador contrasta todas las filas con sus agregados y comprueba la cobertura exacta de cada serie dibujada. Las cuatro ventanas anteriores y las diez nuevas utilizan la misma ruta. Los intervalos conservan sus límites asimétricos. Cuando alguno no está definido, el punto se representa sin barra y la figura lo indica. Las figuras financieras conservan las medias por semilla y obtienen sus cantidades de mundos de la procedencia.

La comparación añade `checkpoint_sha256` y `source_report_sha256` a `cases.csv`. Para enlazar la fiabilidad de una comparación antigua que no conserve esas columnas hay que regenerar el análisis en una salida nueva. Esto no requiere entrenar ni volver a evaluar modelos. La [comprobación de exportación](../../reports/resources/comparison-export-20261006.json) conserva la paridad de los agregados históricos y documenta las pruebas de pertenencia, corrupción y publicación.

Cada CSV admite hasta 32 MiB, 100.000 filas y 256 columnas. Se lee y comprueba una sola versión antes de exportar. Los CSV, las figuras SVG/PNG y el índice de evidencia se preparan en un directorio temporal y se publican juntos, sin sobrescribir una salida anterior. Un fallo de escritura o de representación no confirma una exportación parcial.

`scripts/benchmark_campaign_comparison.py` repite el recorrido predictivo completo con y sin caché, alternando el orden. Comprueba igualdad exacta de las tablas científicas y del Parquet, y compara los casos después de excluir su tiempo de comprobación. Incluye un calentamiento por condición y al menos tres repeticiones medidas. La [evidencia de rendimiento y pruebas](../../reports/resources/campaign-comparison-20261005.md) indica los resultados, herramientas, cobertura y límites realmente comprobados.

## Cierre automático de la campaña real

`scripts/finish_real_campaign.py` recibe `--state`, el resumen de `training.real_campaign`, y `--output`, una salida independiente. Espera a que terminen las etapas neuronal, de continuación y de fiabilidad. Comprueba sus recuentos, dominio, reserva cerrada y huellas antes de ejecutar la comparación y exportarla con la fiabilidad correspondiente. Mientras la campaña siga abierta devuelve el estado `waiting` y código 3, sin crear la salida ni leer predicciones. El cierre completado devuelve código 0.

La salida fija las fuentes, los parámetros y el código del analizador. Un bloqueo impide dos cierres simultáneos. Cada invocación admite como máximo un intento nuevo y conserva hasta tres intentos en carpetas separadas. Una interrupción deja el intento anterior registrado y permite repetir el análisis en otra carpeta. No se reanuda a mitad de una agregación ni se sobrescriben resultados anteriores. El resultado completo se reutiliza solo si sus artefactos e identidad siguen coincidiendo. Cambiar únicamente la fecha de observación del coordinador no obliga a repetirlo.

El comando utiliza CPU y los analizadores existentes. No inicia entrenamientos, no necesita cargar pesos y no publica resultados en GitHub. Puede programarse como comprobación periódica local. El código 3 debe tratarse como espera, no como fallo. La interpretación de resultados y su promoción siguen requiriendo la revisión de los artefactos producidos.

La [verificación del cierre](../../reports/resources/real-campaign-analysis-20261006.json) reproduce una campaña anterior de 756 modelos sin cambiar sus resultados científicos. Esta comprobación técnica no acredita la finalización de la campaña real ampliada.

La suite general necesita CUDA visible y las rutas del perfil nativo que se haya compilado. Con el perfil conjunto, desde la raíz:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MARS_TITAN_NATIVE_LIBRARY="$PWD/build/native/native-controls-release/libmars_titan_simulation.so" \
MARS_TITAN_SIM_EXECUTABLE="$PWD/build/native/native-controls-release/mars-titan-sim" \
MARS_TITAN_PPO_EXECUTABLE="$PWD/build/native/native-controls-release/mars-titan-ppo" \
uv run --no-sync pytest -q
```

Las rutas predeterminadas de otros perfiles no identifican automáticamente esos artefactos. Ocultar CUDA hace fallar las pruebas que la requieren y no constituye una comprobación completa de CPU.
