# Comparación sobre la edición real ampliada

La edición en preparación el 6 de octubre de 2026 amplía los datos disponibles para los comparadores RNN, LSTM, GRU, DLinear, Ridge y XGBoost. Exige precios, noticias, gráficos y fundamentales en cada muestra, junto con los 140 conceptos macro. Cada modelo y sus ajustes utilizan la misma representación y las mismas particiones. El candidato MARS-TITAN sigue fuera de esta ejecución.

## Fuentes y población

La cohorte de origen contiene 1.816.369 muestras codificadas de 2.226 activos estadounidenses. El reparto anterior por años admitía 1.809.376 etiquetas. Al exigir los 140 indicadores de aquella edición quedaban 264.131 muestras, de las que 261.879 aparecían en la unión de las cuatro ventanas. Sumar las filas de distintos folds contaría repetidamente las mismas observaciones.

La nueva admisión macro contiene 397 sesiones completas, desde el 1 de junio de 2022 hasta el 28 de diciembre de 2023, frente a las 257 sesiones anteriores. Este recuento corresponde al panel macro, no a muestras multimodales ni a etiquetas. El cruce con las otras modalidades y la purga temporal determinan qué se puede entrenar.

Se han aplicado estos cambios a una edición independiente:

- Historia publicada del índice de estrés: STLFSI3 desde su primera publicación acreditada en enero de 2022 y STLFSI4 desde noviembre de 2022. El catálogo identifica la composición. No se igualan sus niveles ni se atribuyen los valores de la versión 4 a fechas anteriores a su publicación.
- Recuperación de comunicados oficiales de seis conceptos chinos desde abril de 2022. El panel combinado conserva 119 documentos y 358 observaciones de versiones. Las 1.808 filas numéricas de la edición anterior permanecen idénticas.
- Retardos diarios calculados sobre observaciones numéricas válidas. La política `valid_observations` cambia explícitamente la definición del retardo frente a `source_records`. Conserva fechas, revisiones y ausencias de las fuentes, sin interpolar valores ni comprimir periodos mensuales o trimestrales.

El catálogo previo y sus resultados no se modifican. La disponibilidad acreditada de GSCPI limita el inicio conjunto a junio de 2022. La ausencia de una entrada del spread Brent–WTI excluye el 29 de diciembre de 2023. Las noticias originales preparadas terminan el 18 de diciembre y su ventana de cinco sesiones limita también las muestras del final de ese año. Un indicador reconstruido hoy no se presenta como una publicación histórica de entonces.

La cohorte conserva la categoría `original_audited`: usa los contenidos distribuidos con FinMultiTime y sus controles documentados. Esa categoría no acredita la verificación editorial externa de todos los artículos. China mantiene un bloqueo adicional por falta de fechas de publicación acreditadas para sus hechos contables. Recuperar macro chino no habilita por sí solo ese mercado bursátil.

La representación contable ampliada separa los conceptos monetarios USD y CAD. No realiza conversiones de moneda. Permite revisar activos que tenían hechos contables reales pero no encajaban en los canales USD del formato anterior. La edición final debe conciliar cada activo admitido y cada exclusión antes del lanzamiento.

## Particiones y selección

El [protocolo ampliado](../../configs/evaluation/real-expanded-walk-forward.json) define diez ventanas expansivas. El primer entrenamiento comienza en junio de 2022, la primera validación ocupa diciembre de 2022 y enero de 2023, y la primera evaluación corresponde a marzo de 2023. Las evaluaciones mensuales continúan hasta diciembre, sin solaparse. Cada ventana reserva dos meses de validación, uno de calibración y uno de evaluación. La reserva final comienza en enero de 2024 y permanece cerrada.

Cada partición exige que la etiqueta madure antes del límite siguiente. Entrenamiento, validación y calibración conservan además una sesión de margen. La opción `--recover-annual-boundaries` vuelve a comprobar los objetivos que el corpus original excluyó únicamente por el corte de enero de 2023. Solo admite objetivos finitos desde la última sesión de 2022 hasta la primera de 2023. Después aplica las fronteras nuevas y conserva `source_reason` como procedencia. No recupera etiquetas ausentes ni cruces con el test final.

La [búsqueda neuronal](../../configs/baselines/convergence-temporal-search-us.json) usa dos configuraciones por familia, semilla 42 y repetición de los finalistas con 43 y 44. El máximo es de 100 épocas, con diez épocas mínimas, paciencia de diez evaluaciones completas y mejora mínima de 0,00001 en MAE por sesión. Las continuaciones iniciales MAE/MSE tienen cinco épocas y selección del mejor estado, incluido el padre. [Ridge y XGBoost](../../configs/baselines/tabular-convergence-us.json) conservan su búsqueda predefinida. XGBoost tiene hasta 2.000 rondas, mínimo de 200 y paciencia de 100.

El [postentrenamiento emparejado](../../configs/baselines/real-matched-posttraining.json) usa exclusivamente datos reales. Cada semilla se vincula con el padre correspondiente. Ridge se identifica como padre determinista compartido. Se comparan REINFORCE, pérdida esperada exacta, adaptación MAE y tres variantes KLPO. Las familias neuronales añaden continuaciones MAE/MSE.

Estos ajustes recorren 50 épocas con el mismo tamaño de lote y presupuesto por condición. La paciencia de 50 no acorta ese presupuesto. La validación selecciona el mejor estado, incluida la época cero. Esta regla conserva el emparejamiento de actualizaciones, aunque un método deje de mejorar antes que otro. El informe distingue este presupuesto fijo de la parada adaptativa de los modelos base. Seleccionar el mejor estado no garantiza ausencia de sobreajuste.

Las configuraciones producen 40 casos neuronales, 17 tabulares y 132 ajustes por ventana. Son 1.890 casos de entrenamiento o continuación en diez ventanas y otras 1.890 evaluaciones de sus estados congelados. Estos recuentos proceden de las configuraciones, no de ejecuciones ya terminadas. Todas las filas reales admitidas de entrenamiento se recorren en cada época. Validación, calibración y evaluación no se añaden al ajuste de esa ventana.

## Qué mide la evaluación

El criterio principal sigue siendo MAE por sesión. Se conservan MSE, correlaciones, controles de predicción cero y la separación entre el efecto de discretizar la salida y el efecto del aprendizaje posterior.

`mars_titan.evaluation.campaign_reliability` empareja calibración y evaluación por ventana, modelo y checkpoint. Calcula precisión y recuperación por signo, acierto, abstenciones y frecuencia de emitir una predicción direccional. Cada porcentaje conserva su numerador y denominador. La dirección es la del retorno residual frente al mercado, no la subida o bajada bruta del precio.

Los intervalos del 90 % y 95 % usan el estadístico de orden de los errores absolutos de calibración. La evaluación posterior mide cobertura y anchura sin reajustar el radio. Solo se emite una señal de signo cuando todo el intervalo queda a un lado de cero. Las medias por sesión identifican cuántas sesiones tienen denominador definido. No se mezclan semillas o ventanas como observaciones independientes ni se afirma una garantía de cobertura bajo dependencia temporal.

El RL predictivo conserva sus 21 acciones y la recompensa basada en error. La simulación financiera real con posiciones persistentes sigue pendiente de acreditar ajustes OHLC, acciones corporativas y fechas de pago. Un retorno residual no se utiliza como beneficio de una cartera. Los resultados sintéticos anteriores continúan separados.

## Ejecución y comprobaciones

Las nuevas ediciones guardan hashes de fuentes, catálogos, fórmulas y código. El recálculo solo publica cuando la base SQLite está consolidada y no ha cambiado durante la lectura. Los archivos WAL forman parte del estado de una base activa, como describe la [documentación de SQLite](https://www.sqlite.org/wal.html). Las pruebas reproducen cambios concurrentes y exigen rechazar la publicación bajo una identidad antigua.

La preparación, el entrenamiento y la evaluación conservan recibos recuperables. El mejor checkpoint cambia solo cuando mejora el criterio declarado. Los estados de recuperación rotan con un límite pequeño, sin conservar uno por cada época. Las huellas y los cursores confirmados impiden continuar con fuentes o configuraciones cambiadas.

El lector reutiliza Arrow C++ mediante su proyección por columnas. Deja de decodificar el vector macro antiguo cuando la ventana proporciona su sustituto. La [prueba sobre 187.246 muestras](../../reports/resources/real-corpus-preparation-20261006.json) redujo los bytes decodificados por época un 30,4 %. Dos parejas completas de una época CUDA pasaron de 62,35 a 61,18 segundos de mediana, con igualdad exacta de pesos, AdamW, RNG, cursores y predicciones. El ajuste aislado no mejoró de forma consistente. No se atribuye a ese resultado una mejora predictiva ni una reducción del consumo energético.

La campaña utiliza una sola carga científica CUDA. Los presupuestos de RAM, VRAM y caché son límites de admisión, no objetivos que deban llenarse. El tamaño de lote y la concurrencia se cambian únicamente después de medir el recorrido completo y comprobar paridad o tolerancias numéricas declaradas.
