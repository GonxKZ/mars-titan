# Estado de la edición histórica y de su ejecución

La edición v3.1 está generada y sus archivos actuales se han vuelto a comprobar el 27 de octubre de 2026. El [recibo de esta revisión](../../reports/data/historical-readiness-20261027.json) registra las huellas y el alcance de cada comprobación. La revisión no entrena modelos, no calcula errores predictivos y no cambia la configuración de ejecución.

El catálogo conserva 18.698.976 muestras de 5.008 activos con datos no vacíos. El censo declara 5.676 candidatos: 5.023 registros con estado `encoded`, incluidos quince que declaran cero muestras, y 653 con `missing_required_prices`. Estas categorías no son nuevos datos que puedan sumarse al catálogo. Las etiquetas aceptadas siguen repartidas entre 17.142.187 filas de entrenamiento y 1.235.785 de validación en el manifiesto global.

La edición histórica admite ausencias explícitas y mantiene la misma población entre modelos. La comparación estricta de cuatro modalidades completas permanece separada. El bit de presencia macro indica que hay al menos un indicador admitido, no que estén observadas las 140 posiciones del catálogo. Las ausencias y sus máscaras se conservan.

## Qué se ha comprobado ahora

Se ha ejecutado el verificador existente de `edition_catalog` sobre los Parquet actuales. Recalcula las huellas de las muestras y etiquetas y compara claves, presencia, enlaces, particiones y fechas con el catálogo. Confirma las 18.698.976 filas sin diferencias. La ejecución tarda 175,26 s y alcanza 637,56 MiB de RSS. No materializa el valor del objetivo ni vuelve a estimar los coeficientes residuales.

Los manifiestos y los tres informes de v3.1 coinciden con las huellas del [recibo del 11 de octubre](../../reports/data/edition-v3-1-verification-20261011.json). Sus comprobaciones de vectores y recálculo de etiquetas conservan aquel alcance, sin atribuirles una reproducción nueva. También se han confirmado las huellas de los 5.008 archivos de precios que utiliza el corpus y de los dos factores. Estos últimos mantienen `point_in_time_verified=False`. Las huellas no autentican la publicación original ni reconstruyen versiones contemporáneas.

La validación de metadatos acepta las 51 ventanas de A v2: 19 US, 13 CN y 19 US+CN. Sus informes y la configuración actual coinciden con el [recibo del 14 de octubre](../../reports/data/campaign-a-v2-views-v31-20261014.json). Se han cotejado protocolos, ventanas, huellas y recuentos, sin repetir el contraste independiente de todas sus filas. Las vistas comparten historia y sus recuentos no se suman como observaciones independientes.

`cohort_complete=True` y `training_ready=False` se conservan en los manifiestos. El [contrato histórico con máscaras](historical-input-masks.md#identidad-y-reutilización) mantiene esa segunda marca en falso. Por sí sola, no demuestra que la edición siga pendiente de construir o verificar. El lector histórico comprueba la política, el censo, la representación y las fuentes con sus propias reglas. La revisión no modifica esas marcas.

El verificador reutilizado conserva el mismo código que el recibo del catálogo del 13 de octubre. Sus catorce pruebas actuales aprueban sin omisiones. Cuatro [mutaciones dirigidas](../../reports/data/historical-readiness-20261027/mutations.json) detectan omitir la huella del catálogo, la correspondencia de presencia, los índices de etiquetas o la disponibilidad de la fila de precios que cierra la ventana. Cada una tiene controles verdes antes y después y cero intentos de pasos de optimizador. La [cobertura y el CRAP](../../reports/data/historical-readiness-20261027/quality.json), con coverage.py 7.16.1 y Radon 6.0.1, registran 284/309 sentencias, 41/50 ramas y CRAP máximo 19. CRAP utiliza cobertura de sentencias. Son comprobaciones del verificador y no una prueba de calidad predictiva.

## Condiciones de ejecución que siguen abiertas

La aceptación de la campaña en [#363](https://github.com/GonxKZ/mars-titan/issues/363) exige medir el recorrido completo sobre las vistas v3.1, comprobar las regeneraciones pertinentes en CUDA y vincular el presupuesto final, la retención, la precisión y las versiones. La preparación de cohortes tampoco completa por sí sola el motor y la aceptación de la criba de variantes de [#746](https://github.com/GonxKZ/mars-titan/issues/746).

La observación de recursos de esta revisión encuentra 744 MiB libres en la RTX 4070 Laptop y dos procesos de cómputo ajenos. La admisión por defecto requiere conservar 1 GiB y disponer de al menos 256 MiB para la carga, además de respetar la exclusividad de cómputo. No se intentó reservar la GPU ni se detuvieron otras aplicaciones.

El disco tiene 28,49 GB libres (26,53 GiB) en esa observación. La [proyección del 14 de octubre con retención v3](../../reports/engineering/campaign-storage-a-v2-v31-v3-20261014/README.md) estimaba un pico de 92,6 GB bajo regeneraciones idénticas. Es una proyección histórica y condicional, no una medida nueva ni un presupuesto actual aceptado. La comprobación presente de datos no valida por sí sola el espacio necesario para ejecutar todas las etapas.

La campaña conserva su declaración previa y la revisión no la lanza ni la reanuda. La protección de aprendizaje y la reserva del test final permanecen activas. El estado de los datos, la comprobación del código y las condiciones de ejecución se registran por separado.
