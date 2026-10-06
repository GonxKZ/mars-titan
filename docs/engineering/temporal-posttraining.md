# Postentrenamiento de las ventanas temporales

Los ajustes posteriores y las referencias tabulares pueden reutilizar las vistas con entrenamiento, validación, calibración y evaluación. Los límites proceden del protocolo verificado de cada ventana, incluida la sesión de purga y la maduración de etiquetas. El corte fijo de enero de 2023 queda reservado al formato histórico de dos particiones.

La preparación ordenada exporta entrenamiento y validación. La rejilla de 21 acciones y la normalización se ajustan con entrenamiento. Los cuatro recuentos declarados se conservan para comprobar que el padre y el ajuste pertenecen a la misma población. La identidad incluye el manifiesto de origen y las implementaciones del calendario y las transformaciones temporales.

El lector original puede decodificar etiquetas de otras particiones al leer un archivo Parquet compartido. Esas filas no se entregan al ajuste. Una prueba cambia los valores de calibración y evaluación y comprueba que la rejilla y los archivos exportados de entrenamiento y validación conservan exactamente sus valores. No se presenta esta comprobación como ausencia de lectura física de columnas compartidas.

La recuperación rechaza cambios en el vínculo con la supervisión y conserva las particiones completas ya confirmadas. Las pruebas también rechazan cohortes dentro de la purga, poblaciones incompatibles y padres cuya implementación temporal no corresponde a sus huellas.

La serie histórica y las ventanas con 140 indicadores son ediciones distintas. No se mezclan sus poblaciones, métricas ni checkpoints. La continuación real de edición 2 utiliza el diseño de [selección con estado inicial y paciencia](continuation-selection.md). No modifica el protocolo de una campaña histórica en curso.

## Ejecución y recuperación

`mars_titan.posttraining.completion` verifica todas las ventanas neuronales declaradas, la admisión de 140 indicadores y el contrato de codificadores. La edición de cuatro ventanas programa 68 referencias tabulares y 528 ajustes reales. Cada ventana tiene 17 referencias tabulares y 132 ajustes. Los seis padres comparten los seis objetivos residuales y tres semillas. Las cuatro familias neuronales añaden los controles de continuación MAE y MSE. La [edición real ampliada](real-expanded-comparison.md) define diez ventanas y exige padres emparejados por semilla.

La cola usa procesos separados para liberar los recursos de cada etapa. Cada proceso científico necesita una concesión CUDA exclusiva. La opción `--after` espera un recibo completo de la campaña anterior. Una campaña pausada o fallida requiere revisión y no se interpreta como terminada.

```bash
uv run python -m mars_titan.posttraining.completion \
  --reference data/interim/strict140-us-temporal-search-20260928 \
  --encoded data/processed/original-audited-us-encoded-20260923/manifest.json \
  --tabular-config configs/baselines/tabular-search-us.json \
  --post-config configs/baselines/real-continuations-v2.json \
  --output data/interim/strict140-us-completion-20260930 \
  --after data/interim/original-audited-us-klpo-20260926-v2/summary.json
```

La misma orden recupera una ejecución interrumpida. No modifica las configuraciones ni sus fuentes. Los ajustes guardan dos estados recientes de recuperación y el mejor si es distinto. Una etapa se confirma junto con la huella de su recibo. Un corte entre la finalización del proceso hijo y esa confirmación vuelve a verificar el hijo al reanudar.

La parada envía SIGTERM al hijo propio y concede hasta 620 segundos para guardar su estado. Si no responde, se termina ese proceso y se recoge su salida. El siguiente intento utiliza el último checkpoint confirmado. No se terminan procesos ajenos.

## Evaluación y análisis

La evaluación comienza después de terminar todas las etapas de ajuste de las ventanas declaradas. En la edición de cuatro ventanas congela los recibos de los 756 modelos resultantes, incluidos los 160 neuronales, y evalúa cada uno sobre calibración y evaluación. Esas métricas no cambian la selección de modelos, la rejilla ni la normalización. Las ejecuciones de evaluación se cuentan por separado de los entrenamientos.

Las continuaciones recuperan el mejor estado declarado. Los adaptadores conservan el orden de características y la escala de la rejilla original. Las referencias publican predicciones continuas. Los ajustes mantienen como salida principal la mediana de la política discreta, con centro continuo y padre en columnas separadas. Las continuaciones neuronales originales se comparan con el checkpoint del que proceden.

Cada evaluación escribe Parquet y un recibo independiente con sus huellas. Una interrupción no confirma métricas parciales. Repetir una evaluación terminada verifica sus artefactos sin reservar GPU ni reescribirla. Al terminar, `analysis.json` y `analysis.md` conservan resultados por caso, ventana, semilla y método. Las medias por ventana son descriptivas, sin intervalos de confianza ni supuestos de independencia entre periodos compartidos.

Las pruebas locales comprueban paridad con la fórmula de validación, reconstrucción del estado seleccionado, pausa y recuperación, corrupción de predicciones, conservación de semillas y detención de un hijo que ignora SIGTERM. La edición de convergencia de cuatro ventanas terminó y sus resultados constan en la [comparación del 5 de octubre](../../reports/baselines/campaign-comparison-20261005.md). La nueva edición mantiene configuraciones y salidas independientes.

La comprobación del 30 de septiembre terminó con 163 pruebas correctas. Se omitió la prueba de paridad CUDA, que requiere activación y GPU exclusiva, y se excluyó una integración CUDA de la cola histórica. Esa integración había pasado antes de reanudar la carga científica. La nueva prueba CUDA queda como comprobación previa al lanzamiento de las ventanas pendientes.

Coverage.py 7.16.2 midió un 83 % conjunto de sentencias y ramas en `completion.py`, `heldout.py`, `analysis.py` y `partition_contract.py`, incluyendo la admisión real y las pruebas CPU. Radon 6.0.1 calculó complejidad ciclomática. Para CRAP se utilizó la fracción de sentencias ejecutables cubiertas de cada función y la fórmula `CC² × (1 − cobertura)³ + CC`. El control de etapas conserva ramas de fallo y ejecución científica sin recorrer en estas pruebas. Estas medidas no equivalen a cobertura de CUDA ni de todo el repositorio.

Cuatro mutaciones dirigidas fueron detectadas: admitir cohortes en la purga, omitir la comprobación de población, permitir validación en el evaluador reservado e invertir las dependencias de ejecución. Las mutaciones se aplicaron en procesos de prueba separados, sin editar el runtime científico.
