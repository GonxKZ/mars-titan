# Validación temporal y admisión de una nueva edición

La validación cruzada debe entrenar con pasado y evaluar en periodos posteriores. La reserva final no interviene en la búsqueda de configuraciones. Este orden corresponde a la [evaluación con origen móvil](https://otexts.com/fpp3/tscv.html). Ajustar hiperparámetros usando el test contaminaría su función de comprobación final, como explica la [documentación de selección de modelos](https://scikit-learn.org/stable/modules/cross_validation.html).

El contrato implementado está en `evaluation/splits.py`. `build_folds` produce ventanas expansivas comunes a todas las empresas de un mercado. `FoldPartitioner` prepara sus límites una vez y asigna lotes de metadatos con NumPy. No utiliza el valor del objetivo, el nombre de la empresa ni una permutación aleatoria para elegir la partición.

## Cortes y purga

`configs/evaluation/walk-forward.json` conserva una configuración de diseño con comienzo en 2009, primera validación en enero de 2013, seis meses de validación, tres de calibración y tres de evaluación. Avanza tres meses y reserva 2024 como test. Produce 41 ventanas nominales. Estas fechas no acreditan que exista una muestra admisible para ejecutarlas.

Cada ventana mantiene cuatro funciones. Entrenamiento ajusta pesos y transformaciones. Validación selecciona configuración y checkpoint. Calibración se reserva para los métodos que la necesiten. Evaluación compara el comportamiento posterior de la regla seleccionada. Si los resultados de esas ventanas se usan para decidir el modelo definitivo, forman parte del desarrollo y no sustituyen al test final.

Los límites son intervalos cerrados por la izquierda y abiertos por la derecha. Una etiqueta debe madurar estrictamente antes del final de su tramo. El margen de una sesión retira la última decisión de entrenamiento, validación y calibración, usando el calendario real del mercado. La evaluación no se recorta por ese margen, pero sí purga etiquetas que atraviesan su límite. El criterio se aplica igual a todos los activos. Las publicaciones posteriores a la decisión y las entradas incompletas quedan excluidas.

El calendario conserva festivos, cierres anticipados y cambios horarios. La partición final se marca `test_reserved`. Este contrato no abre sus objetivos ni ejecuta una evaluación final. Cambiar valores futuros o el orden de las empresas no modifica la pertenencia de las filas anteriores.

## Cobertura completa y condiciones de ejecución

La nueva edición exige todos los indicadores del catálogo. Las máscaras existentes no cuentan como observaciones. Un valor cero observado sí es válido. La [puerta de admisión macro](../data/macro-admission.md) conserva las causas de exclusión y produce un índice de sesiones completas.

El comando siguiente cruza ese índice con el protocolo:

```bash
uv run --no-sync python -m mars_titan.evaluation.split_readiness \
  --protocol configs/evaluation/walk-forward.json \
  --admission ~/.local/state/mars-titan/full-macro-cv-20260927/admission-final/report.json \
  --output ~/.local/state/mars-titan/full-macro-cv-20260927/temporal-readiness/report.json
```

Un resultado `blocked`, con salida 2, impide considerar listas esas ventanas. `macro_windows_ready` indica únicamente que hay cobertura macro suficiente en los cuatro tramos y que se cumple la historia mínima. Todavía deben comprobarse las filas multimodales, sus etiquetas reales y la correspondencia entre el panel nuevo y los vectores codificados. Por eso este informe no declara ninguna ventana ejecutable ni inicia modelos.

La comprobación sobre la edición anterior encuentra cero sesiones completas entre las 3.774 sesiones de 2009 a 2023. El contrato genera sus 41 ventanas nominales, pero ninguna queda habilitada. No se ha entrenado una campaña con validación cruzada ni elegido su mejor configuración.

La recuperación de un indicador no permite cambiar su fecha de publicación. Los índices que aparecieron después del comienzo del historial requieren limitar las fechas de la nueva edición o declarar otro indicador y otra comparación. La duración de las ventanas se fijará a partir de la cobertura efectiva y del coste medido, antes de observar resultados de modelos.

## Selección y componentes que deben ajustarse de nuevo

La métrica primaria sigue siendo MAE por sesión. MSE, RMSE, dirección, Rank IC por sesión y, para los modelos con cuantiles, pinball, cobertura, anchura y riesgo-cobertura son diagnósticos complementarios definidos en [métricas](metrics.md). Las semillas declaradas son 42, 43 y 44. Los controles sin aprendizaje y el estado inicial del padre deben poder superar a una continuación que empeore. La comparación de ajustes emparejados conserva el mismo presupuesto de actualizaciones y selecciona el mejor estado permitido por su protocolo.

Las comparaciones entre modelos se hacen sobre las mismas filas, comprobadas mediante la huella de la población, y con contrastes emparejados por sesión. La incertidumbre se estima con un bootstrap circular por bloques de días UTC que mantiene juntos todos los activos y ambos mercados de cada día y aplica los mismos índices a todos los modelos. La longitud de bloque, su sensibilidad, la semilla, la ponderación entre mercados y la familia de contrastes que se afirmarán a la vez se fijan con desarrollo antes de evaluar. Una afirmación dentro de la familia exige que su intervalo simultáneo excluya el cero. La [propuesta sobre la cabeza de cuantiles](quantile-head-decision.md) debe decidirse antes de entrenar, porque determina qué modelos tienen calibración y abstención comparables.

Cada ventana necesita sus propios normalizadores, rejilla de acciones, predictor padre, optimizador y calibrador. La rejilla y los padres actuales se ajustaron con datos hasta 2022. Utilizarlos en una ventana anterior introduciría información posterior al corte, aunque sus pesos permaneciesen congelados. Los valores derivados de entradas pasadas pueden reutilizarse cuando su cálculo y disponibilidad no dependan del futuro.

Esta entrega implementa y comprueba el contrato de particiones y la comprobación previa de cobertura. La materialización de nuevos conjuntos completos, el adaptador de los entrenadores a sus vistas, la búsqueda de configuraciones, los ajustes posteriores y la evaluación final requieren datos admitidos y una entrega posterior. La campaña anterior se conserva como referencia de su propia edición y no satisface automáticamente el nuevo requisito de cobertura.
