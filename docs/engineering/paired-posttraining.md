# Postentrenamiento con aumentos emparejados

`mars_titan.posttraining` compara ajustes predictivos sobre una población real completa y dos aumentos de entrenamiento. Reutiliza las pérdidas de `models/predictive_adaptation.py` y `models/klpo.py`, los episodios de `episodes/` y los checkpoints de `training/checkpoints.py`.

La comparación científica no se ha ejecutado. Las comprobaciones descritas aquí usan ejemplos pequeños en CPU. No aportan resultados predictivos sobre datos financieros reales ni medidas de rendimiento en GPU.

La [edición 2 de selección](continuation-selection.md) añade el estado inicial como candidato y una configuración separada de continuaciones reales con paciencia. Este documento conserva el diseño y las comprobaciones de la edición original.

## Comparación fijada

El diseño está en [paired-posttraining.json](../../configs/baselines/paired-posttraining.json). Fija cinco épocas, semillas 42, 43 y 44, AdamW nuevo con tasa `0.0001`, `weight_decay=0.01`, recorte de norma 1 y lotes de hasta 256 filas. KLPO utiliza `beta=0.1`, mezcla exploratoria `0.000001` y 128 muestras auxiliares independientes en MC.

| Condición | Filas reales por época | Filas adicionales |
| --- | --- | --- |
| `real` | Todas las admitidas | Ninguna |
| `real_resampled` | Todas las admitidas | Bloques reales con reposición |
| `real_synthetic` | Todas las admitidas | Bloques sintéticos con el mismo tamaño de cada cohorte remuestreada |

`augmentation_windows` redondea el 25 % hacia la siguiente cohorte completa. `paired_world` conserva ese total y la distribución de tamaños por cohorte. El lector divide cada cohorte en sublotes de hasta 256 filas. Los dos aumentos tienen exactamente el mismo número de actualizaciones. La condición real permite medir el efecto de añadir entrenamiento.

Las ventanas tienen hasta 16 decisiones y 16 cohortes previas de calentamiento. Estas cohortes previas no añaden actualizaciones. Las referencias neuronales reinician su estado en cada ventana de precios y no mantienen una memoria entre decisiones. Las cohortes reales se permutan cada época con una semilla común. Los bloques adicionales se permutan como bloques y conservan su orden interno.

Los seis objetivos residuales son REINFORCE, pérdida esperada exacta, MAE y KLPO Full, MC y exacto. Comparten un adaptador lineal inicializado a cero, la rejilla de 21 acciones ajustada en train y las predicciones del mismo padre congelado. La recompensa es el negativo del error absoluto normalizado. Las distribuciones y pérdidas se calculan en float64.

RNN, LSTM, GRU y DLinear tienen también continuaciones neuronales con MAE y MSE. Cada continuación recibe una copia independiente de los pesos seleccionados de su padre y un optimizador nuevo. No modifica el padre ni reutiliza el optimizador de otra continuación. Ridge y XGBoost solo reciben los seis ajustes residuales. El diseño completo contiene 396 casos.

## Procedencia y tiempo de información

La cola admite los seis padres seleccionados por las campañas completas de referencias y modelos tabulares. Comprueba el recibo terminado, la población, las dimensiones, el checkpoint seleccionado y el test cerrado. Los padres deben conservar las huellas del código de inferencia y de preparación de entradas. Un checkpoint neuronal sin selección debe corresponder a una época completa.

La supervisión identifica su manifiesto de codificación mediante `configuration.source_manifest_sha256`. La cola comprueba ese enlace y exige que `FrozenEncoders.spec` coincida exactamente con `configuration.encoders`. Los mundos sintéticos usan los mismos codificadores de noticias y gráficos, con dimensiones 384 y 512, y los mismos contratos de fundamentales y macro, con dimensiones 45 y 420. Sus datos brutos proceden del generador controlado, no de un modelo ajustado con validación.

La volatilidad del generador se calcula solo con etiquetas de train. El resto de sus mecanismos conserva los valores declarados en `WorldConfig`. La normalización del adaptador usa todas las filas originales de train y se comparte entre condiciones, sin incorporar filas sintéticas o de validación. Cada semilla materializa su aumento una vez y lo comparte entre padres y objetivos.

`ParentCache` identifica cada predicción por el checkpoint, la codificación, la fecha, los activos y las entradas efectivas. Una entrada sintética diferente obliga a predecir de nuevo. Cambiar solo una etiqueta no altera la predicción del padre. `FrozenParent.predict(inputs)` también puede evaluar una observación sin etiqueta. Devuelve un escalar por activo y procesa sublotes de hasta 256 filas.

El ajuste es offline, después del corte de train. Se rechazan entradas publicadas después de la decisión y etiquetas que crucen la partición o el límite del episodio. La validación principal recorre exclusivamente la partición real de validación. Selecciona la época con menor MAE de la mediana de la política, promediada por sesión y mercado. El informe conserva también los errores del centro continuo y del padre. Ningún recorrido admite la partición test.

## Ejecución y recuperación

La CLI se consulta con:

```bash
uv run python -m mars_titan.posttraining.queue --help
```

Requiere `--config`, `--reference`, `--tabular`, `--encoded` y `--output`. `--reference` y `--tabular` reciben los `summary.json` completos de ambas campañas. `--encoded` recibe el manifiesto de codificación al que apunta la supervisión. `--output` debe quedar fuera de los orígenes. `--arm` admite `US`, `CN` o `US+CN` y usa `US` por defecto.

Antes de lanzar el proceso debe estar configurado `CUBLAS_WORKSPACE_CONFIG=:4096:8`. La ejecución científica exige `GpuLease` y `cuda:0`. Si hay otro proceso de cómputo CUDA, la admisión falla antes de cargar pesos, codificadores o datos grandes. No existe una alternativa CPU en la CLI. La API permite diagnósticos explícitos con `diagnostic=True` y `device="cpu"`, limitados a 5000 filas por partición y por época aumentada.

La salida contiene el corpus ordenado, la calibración, los episodios por semilla y las ejecuciones por padre, condición y objetivo. Los manifiestos confirmados quedan ligados a la identidad de la cola. Repetir el comando recupera las preparaciones y casos pendientes. Los casos terminados se verifican mediante sus huellas y contratos antes de omitirlos.

Cada actualización avanza su cursor después de `optimizer.step()`. El cursor se guarda junto con el resto del estado en los checkpoints periódicos, de fin de época y de parada. Incluyen pesos, optimizador, generadores propios de acciones y auxiliares, estados globales de Python, NumPy y PyTorch, contadores, historial de validación y selección. Su identidad incluye padre, datos, episodios, codificadores, condición, normalización, rejilla, código y política numérica. La ausencia de un checkpoint previamente confirmado impide reiniciar silenciosamente desde cero.

SIGINT y SIGTERM solicitan una parada en la siguiente barrera segura. Una validación interrumpida se repite desde su inicio. El checkpoint de recuperación conserva el modelo y optimizador de la última época, aunque la predicción final use una época anterior seleccionada. No hay etiquetas pendientes ni memoria recurrente entre lotes porque el ajuste es offline y las referencias reinician su estado por ventana.

Los lectores mantienen una cohorte y un episodio sintético activo. Las modalidades y lotes tienen un límite de 64 MiB y la caché del padre limita su contenido a 64 MiB y 8192 entradas. Los checkpoints heredan el límite de 512 MiB. `GpuLease` aplica los presupuestos de RAM y VRAM existentes. Los recibos registran tiempo total por intento, actualizaciones, pico de RSS del proceso y pico de VRAM asignada. Estos registros no equivalen a una medición aislada de kernels o de energía.

## Comprobaciones locales

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run pytest tests/posttraining -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_repository.py
```

Las pruebas cubren filas y actualizaciones emparejadas, rechazo del futuro, procedencia de codificadores y padres, recuperación dentro de una cohorte, restauración exacta de pesos, AdamW y los dos generadores de muestreo, dropout en las continuaciones y una cola de 24 casos pequeños en CPU. Los servicios de admisión CUDA y la selección de campañas se sustituyen por entradas de prueba en esa integración. La carga científica de Ridge y XGBoost y la integración con los codificadores reales requieren una comprobación posterior en GPU.

La ejecución local terminó con 37 pruebas correctas. Coverage.py 7.16.1, con medición de ramas, registró 663 de 766 líneas ejecutables y 207 de 294 ramas del paquete. Radon 6.0.1 calculó la complejidad ciclomática. El diagnóstico CRAP usa `C² × (1 − cobertura_de_líneas)³ + C`, con las líneas ejecutables dentro del intervalo de cada función. El máximo observado fue 48 en `load_parent`, cuyo recorrido tabular CUDA quedó sin ejecutar. La cobertura combinada de líneas y ramas fue 82,08 %. Estas cifras no demuestran ausencia de errores.

Cuatro mutaciones dirigidas fueron detectadas por las pruebas: retirar el aumento, admitir etiquetas posteriores al corte, compartir pesos con el padre y perder el estado de AdamW al reanudar. Se aplicaron en memoria mediante un plugin temporal de pytest. La suite general terminó con 1309 pruebas correctas, 38 omitidas y 114 fallos. La repetición de esos 114 casos con la configuración de cuBLAS declarada confirmó que requieren CUDA. No se presenta la suite general como una validación completa.
