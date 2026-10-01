# Selección de continuaciones desde el estado inicial

La edición 2 evalúa el estado inicial antes de actualizar pesos. Una continuación puede devolver la época 0 si ninguna época completa mejora su validación. El criterio sigue siendo la MAE de la mediana discretizada, promediada por sesión y mercado. Los errores del centro continuo y del padre se conservan como medidas distintas.

| Configuración | Casos | Presupuesto | Parada |
| --- | ---: | --- | --- |
| [paired-posttraining-v2.json](../../configs/baselines/paired-posttraining-v2.json) | 396 | Cinco épocas completas | Presupuesto fijo |
| [real-continuations-v2.json](../../configs/baselines/real-continuations-v2.json) | 132 | Hasta cinco épocas reales | Dos épocas completas sin mejora |

Ambas fijan mejora mínima cero, semillas 42, 43 y 44 y los mismos parámetros de ajuste del [diseño original](paired-posttraining.md). La configuración real conserva sus seis objetivos residuales y los controles neuronales MAE/MSE. La comparación emparejada mantiene todas las filas reales por época y la igualdad exacta de filas adicionales y actualizaciones entre remuestreo y síntesis. No detiene cada aumento por separado.

`selection.version=2` identifica esta política en la configuración, la identidad y los recibos. `patience=null` significa presupuesto fijo. Una paciencia entera solo se admite con `condition=real`. Los empates conservan el estado anterior. Con una mejora mínima positiva, una nueva puntuación se acepta únicamente cuando baja más que ese margen respecto al último mejor estado aceptado.

La configuración original `paired-posttraining.json` se conserva. Un caso sin la política v2 mantiene la selección desde la primera época. Los resultados anteriores no se reinterpretan y las huellas de configuración y código impiden recuperarlos como si pertenecieran a esta edición.

## Recuperación y retención

Antes de la validación inicial se confirma un estado con cero actualizaciones. Solo después de recorrer toda la validación se guardan `baseline`, el estado de selección y el mejor checkpoint de época 0. Una interrupción durante esa lectura repite la validación sin consumir paciencia. Las validaciones de fin de época también se confirman completas.

El estado recuperable incluye pesos, AdamW, cursor, estadísticas, RNG globales y generadores propios, historial, validación inicial y selección. Al terminar el ajuste se carga el mejor modelo para generar predicciones. Si esa evaluación final se interrumpe, la recuperación conserva los últimos pesos junto con su propio optimizador y repite la evaluación seleccionada. Antes de rotar estados se comprueba que el checkpoint mejor coincide con la selección confirmada.

El almacenamiento existente conserva dos estados recientes rotativos y un mejor estado si es distinto. Hay como máximo tres archivos de estado confirmados por ejecución, sin estados fijados adicionales. El mejor solo se reemplaza al aceptar una mejora. La escritura temporal y la comprobación del sustituto preceden a la retirada de archivos anteriores. Cada estado conserva el límite de 512 MiB.

`global_step` recoge las actualizaciones ejecutadas y `total_steps` el máximo declarado. `stopped_early` indica si la parada dejó épocas sin ejecutar. `checkpoint` identifica el modelo seleccionado y `recovery_checkpoint` el estado para continuar. `predictions.validation.metrics` conserva las métricas del modelo seleccionado, incluidas las de época 0 cuando gana.

## Ejecución

Se utiliza la misma CLI:

```bash
uv run python -m mars_titan.posttraining.queue --help
```

Para el diseño emparejado se pasa `--config configs/baselines/paired-posttraining-v2.json` y una salida nueva, por ejemplo `--output data/interim/paired-posttraining-v2`. Para las continuaciones reales se usa `--config configs/baselines/real-continuations-v2.json` y otra raíz, como `--output data/interim/real-continuations-v2`.

También son obligatorios `--reference`, `--tabular` y `--encoded`, con los informes completos de las campañas y el manifiesto de codificación verificado. Repetir el comando recupera esa misma edición. La cola real no carga codificadores ni prepara calibración o episodios sintéticos. Comprueba el enlace con el manifiesto de codificación de los datos ya preparados.

La ejecución científica exige `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `cuda:0` y `GpuLease` antes de cargar padres. Solo se ajusta y normaliza con train. La validación usa todas las filas reales admitidas y el test final permanece cerrado. El padre se mantiene congelado y cada continuación neuronal parte de una copia independiente.

## Comprobación local

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
  uv run --no-sync pytest tests/posttraining -q
```

Las pruebas usan diagnósticos CPU pequeños. Cubren época 0, empates, mejora mínima, paciencia, recuperación de RNG y optimizador, interrupciones de validación inicial, intermedia y final, presupuestos emparejados y retención tras varios guardados y reanudaciones. Algunas pruebas fijan una curva de puntuaciones para comprobar las decisiones de selección mientras ejecutan realmente el modelo, el optimizador y los artefactos.

La [comprobación registrada](../../reports/resources/continuation-selection-quality.json) terminó con 65 pruebas correctas y detectó cinco mutaciones dirigidas. Coverage.py 7.16.1 midió 379 de 431 líneas ejecutables y 111 de 150 ramas en los tres módulos afectados. El módulo nuevo de selección alcanzó 21 de 21 líneas y 10 de 10 ramas. Radon 6.0.1 y la fórmula CRAP declarada en el informe sitúan el máximo en 47,55, dentro de la admisión del runner. Los recorridos CUDA permanecen sin comprobar. Estas medidas no sustituyen las pruebas de comportamiento.

Esta edición no aporta una comparación científica nueva ni resultados de entrenamiento en GPU. Elegir por validación reduce el riesgo de devolver una época que ya se sabe peor en esa medida, pero no garantiza una mejora futura ni elimina el ajuste a la propia validación. PPO y Double DQN conservan su presupuesto fijo y su evaluación separada, sin adoptar este criterio predictivo.
