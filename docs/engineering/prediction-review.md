# Revisión periódica de predicciones

El módulo `mars_titan.evaluation.prediction_review` recalcula los errores de las predicciones ya guardadas. Lee únicamente las ejecuciones que un resumen confirma como terminadas, comprueba la huella de su informe y después la de cada Parquet. No carga pesos, no ejecuta inferencia ni modifica el entrenamiento. La evaluación consume CPU mediante NumPy y Arrow.

La configuración local contiene una lista `sources` de objetos con `id` y `summary`. Cada identificador distingue una campaña o un padre y cada ruta apunta a su `summary.json`. Se admiten los resúmenes neuronales, los intentos tabulares y los resúmenes de cada padre predictivo. La cola superior no sustituye a estos últimos porque su contador se confirma por familias completas.

Las campañas con validación cruzada temporal añaden `manifest` a cada fuente, con el manifiesto original de esa ventana. Su SHA-256 debe coincidir con la identidad del resumen. El revisor comprueba la ventana mediante `build_folds`, sin cargar el corpus. Cuando una ejecución utiliza una vista por mercado en `views/US.json`, `views/CN.json` o `views/US+CN.json`, verifica su huella contra el informe y exige el mismo origen y contrato temporal. Una vista se lee una vez por mercado y ciclo.

Cada fuente puede declarar `input_policy`. Sin declaración se aplica la política estricta y su vista de versión 1. Con `historical_masked_2000_v1` la fuente debe incluir `manifest`, el manifiesto y las vistas por mercado deben adherirse a esa política y la versión de la vista debe ser 2. El contrato temporal guardado en el estado añade entonces `input_policy`, de modo que el cambio de política invalida la caché. Las referencias con retención `heldout_full_train_sessions_v1` no guardan predicciones completas de entrenamiento, así que se revisan con `--partition validation`.

La revisión habitual se ejecuta desde el entorno del proyecto:

```bash
uv run --no-sync python -m mars_titan.evaluation.prediction_review \
  --config ~/.config/mars-titan/prediction-review.json \
  --state ~/.local/state/mars-titan/prediction-review/state.json \
  --max-jobs 8
```

Una tarea equivale a una partición de una ejecución. La validación tiene prioridad y después se revisa entrenamiento. `--partition validation` permite limitar un recorrido a validación. Los únicos nombres aceptados son `train` y `validation`. Los archivos deben llamarse `train-predictions.parquet` y `validation-predictions.parquet`, permanecer en el directorio de su ejecución y no ser enlaces simbólicos. La salida debe quedar fuera de todos los directorios fuente.

## Contrato de evaluación

Cada fila tiene mercado, instante UTC, objetivo y predicción. Una fuente sin manifiesto temporal conserva el contrato anterior: entrenamiento previo a 2023 y validación durante 2023. Con `manifest`, se aplican los intervalos de entrenamiento y validación de la ventana, con inicio incluido y final excluido. Por eso un entrenamiento puede contener fechas de 2023 sin mezclarse con su validación posterior. Se rechazan protocolos que desplacen el comienzo del test final más allá de enero de 2024. La auditoría de maduración de etiquetas, purga por sesiones y disponibilidad de las modalidades se hace sobre el corpus, porque esas columnas no están en los archivos de predicciones. El revisor no sustituye esa auditoría ni evalúa calibración, evaluación exterior o test.

MAE y MSE por filas dan el mismo peso a cada observación. `session_mae` y `session_mse` dan el mismo peso a cada par de mercado e instante, aunque tengan diferentes números de activos. Se reutiliza `SessionErrors`, que acumula en float64. Los recuentos deben coincidir exactamente. Las métricas deben coincidir con tolerancia absoluta `1e-12` y relativa `1e-10`, que permiten el cambio en el orden de las reducciones por lotes. La tolerancia no se ajusta a los resultados de cada modelo.

En las referencias se revisa `prediction` y se calcula el control cero. En los adaptadores se revisan las lecturas declaradas `median`, `center`, `nearest`, `parent` y `zero`. El campo `delta_session_mae` resta el control al resultado principal. Un valor positivo indica empeoramiento. La mediana discreta y el centro continuo son lecturas diferentes y no se mezclan. No se recalculan aquí las métricas de entropía o divergencia de la política.

Los recuentos de las predicciones se contrastan con los del informe y el resumen cuando estos los declaran. Un informe o archivo alterado, un valor ausente o no finito, una partición equivocada o una métrica discrepante impiden marcar el resultado como verificado. La detección de duplicados y la correspondencia con la población admitida requieren la auditoría de corpus asociada. La revisión de errores por sí sola no acredita procedencia externa, calidad de noticias o ajuste de precios.

## Memoria, caché y recuperación

La lectura selecciona columnas, utiliza lotes de hasta 4.096 filas y abre un iterador por grupo Parquet. Esta última condición evita retener buffers entre grupos. Cada archivo admite como máximo 512 MiB y cada bloque, 512 MiB descomprimidos. Se limita la configuración a 32 fuentes, el historial a 8.192 evaluaciones y cada lectura JSON a 32 MiB. No hay un proceso por modelo ni una copia de las modalidades.

La caché identifica el código del revisor, del acumulador y de las particiones, el informe, el SHA declarado del Parquet, su dispositivo, inode, tamaño, fechas de modificación y cambio y el presupuesto de archivo. Las fuentes temporales añaden ambas huellas de manifiesto, el identificador de ventana y sus límites. Incorporar o cambiar ese contrato invalida el resultado anterior. Una revisión sin cambios vuelve a comprobar los informes pequeños y los metadatos, sin recorrer los Parquet. Una sustitución o cambio de presupuesto invalida la entrada. No es una comprobación criptográfica continua de un archivo cuya firma del sistema de archivos permanezca intacta.

El estado se sustituye mediante escritura temporal, `fsync` y renombrado atómico al terminar el ciclo. Si el proceso se corta, permanece la instantánea anterior y se repite como máximo el trabajo del ciclo incompleto. No se crean checkpoints de modelos. Los errores se conservan con su identidad y no cuentan como evaluaciones válidas. Un error sin cambios en su identidad no se reintenta automáticamente. Tras reparar una fuente, su cambio de firma activa la revisión. Si se necesita repetir por un fallo transitorio sin cambios de archivo, se conserva el estado anterior y se usa una nueva ruta de salida.

Los resultados históricos siguen en el estado, pero los contadores vigentes solo consideran las ejecuciones confirmadas por las fuentes de ese ciclo. Una fuente futura aún ausente se informa en `unavailable_sources`. Los errores de una fuente se registran en `source_errors`. El comando devuelve estado 1 si hay fallos, aunque otras fuentes se hayan podido revisar.

## Ejecución local periódica

Las unidades `configs/systemd/mars-titan-prediction-review@.service` y `.timer` revisan hasta ocho tareas cada quince minutos, después de terminar el ciclo anterior. La instancia identifica una copia instalada del módulo y sus dependencias internas mediante el SHA del módulo principal. La copia incluye `session_metrics.py`, `splits.py` y `data/storage.py`. `prediction-review.env` fija `MARS_TITAN_REVIEW_PYTHON` a un entorno uv compatible y `prediction-review.json` conserva las rutas privadas de las fuentes. El directorio de la instancia se añade a `PYTHONPATH` y sus huellas se comprueban en el estado.

La unidad tiene 512 MiB de límite de memoria, un hilo en los runtimes numéricos, prioridad baja de CPU e I/O y diez minutos de tiempo máximo. Su directorio de estado es la única salida persistente. No modifica las cuotas de entrenamiento y no se solapa consigo misma. El monitor de salud del proceso sigue siendo independiente, con su cadencia de cinco minutos.

```bash
systemctl --user list-timers 'mars-titan-*'
journalctl --user -u 'mars-titan-prediction-review@*' -f
journalctl --user -u mars-titan-scientific-campaign.service -f
watch -n 2 nvidia-smi
```

Las métricas científicas y sus límites están en [la auditoría de población y resultados](../research/population-and-results-audit.md). Las pruebas y medidas del revisor están en [su evidencia técnica](../../reports/resources/prediction-review-quality.json). El test final permanece reservado, sin materializar ni evaluar en esta edición.

La comprobación de instalación del 27 de septiembre de 2026 confirmó el arranque automático tras reiniciar y 322 particiones verificadas de 161 ejecuciones, sin discrepancias. Una revisión sin novedades necesitó 0,10 segundos internos y no volvió a leer predicciones. En cinco repeticiones manuales, después de un calentamiento, la mediana de tiempo total fue de 0,242 segundos. Estas medidas incluyen la campaña científica activa y no expresan rendimiento de entrenamiento.

El 28 de septiembre se corrigió la configuración que seguía observando la campaña anterior y se instaló el revisor con contratos temporales. La nueva comprobación recalculó 202 particiones de 101 ejecuciones, sin discrepancias. La cuarta ventana aún no había comenzado y su resumen se registró como ausente. El monitor de salud consulta la campaña vigente cada cinco minutos y el revisor comprueba hasta ocho particiones cada quince minutos. Los estados de las campañas anteriores se conservan aparte.

Pasan 116 pruebas del revisor, de los cortes temporales y del monitor de salud, además de cinco mutaciones dirigidas. La [evidencia de esta corrección](../../reports/resources/temporal-prediction-review-quality.json) registra 246 de 267 sentencias y 112 de 132 ramas cubiertas, con el convenio de CRAP declarado. La auditoría independiente comprobó la huella del checkpoint seleccionado en 101 ejecuciones y la presencia y tamaño de los retenidos, sin cargar pesos. Quince continuaciones habían conservado el estado inicial. Estas observaciones no acreditan generalización ni excluyen todo sobreajuste.
