# Banco de integridad de la evaluación

Un resultado de la campaña solo se publica si supera las comprobaciones de este banco. Cada una responde a una pregunta concreta sobre si una cifra puede creerse: si las métricas están bien calculadas, si todos los modelos se evalúan con las mismas filas, si alguna información futura se ha colado o si una política explota un fallo del entorno. El banco desarrolla [#28](https://github.com/GonxKZ/mars-titan/issues/28) y se sigue en [#442](https://github.com/GonxKZ/mars-titan/issues/442).

El código vive en `src/mars_titan/integrity/`. No sustituye a las comprobaciones que ya hacen la comparación walk-forward, el contrato de recibos ni la auditoría de los entornos. Las repite por otro camino o las reúne en un informe común.

## Segunda implementación de las métricas

`integrity/independent_scores.py` recalcula las métricas por sesión de la comparación a partir de su especificación en [métricas predictivas](metrics.md), sin importar `evaluation/forecast_scores.py` ni `evaluation/forecast_panel.py`.
- **Agrupación.** Usa pandas sobre el par (mercado, instante) en lugar de los índices de sesión y `bincount` del código principal.
- **Probabilidad implícita de subida.** Recorre cada fila buscando el último cuantil que no supera el cero, en lugar de la fórmula vectorial.

Si las dos implementaciones coinciden, un error de agregación o de convención tendría que estar repetido en las dos para pasar desapercibido.

Cubre:
- MAE y MSE por sesión;
- dirección con la convención de abstención, y sus aciertos a cada cobertura de la curva riesgo-cobertura cuando el panel trae `asset_id`;
- precisión y exhaustividad de cada signo;
- Rank IC con sus motivos de indefinición;
- pinball y frecuencia por nivel;
- cobertura, anchura y puntuación de los intervalos centrales del 80 % y del 95 %;
- Brier del signo y recuentos por intervalo de probabilidad para el ECE.

`integrity/score_recheck.py` aplica esa implementación a una comparación ya publicada:
1. Comprueba las huellas de `comparison.json`, `sessions.parquet`, la configuración y el manifiesto de fuentes.
2. Vuelve a leer las predicciones crudas de cada modelo, semilla y ventana, también con sus huellas. Antes de puntuar quita las filas que el diseño conjunto excluye de la ventana, con la regla declarada que aplica también la identidad de filas.
3. Coteja la tabla de sesiones columna a columna y reconstruye el resumen de cada vista con las reglas del [registro de métricas](#registro-de-métricas).

El resultado cuenta las sesiones que faltan o sobran y las discrepancias por columna, y sale con código distinto de cero si alguna comprobación falla.

```bash
uv run python -m mars_titan.integrity.score_recheck \
  configs/evaluation/historical-masked-2000-comparison.json SOURCES.json US+CN COMPARISON_DIR \
  --output RECHECK.json
```

**Tolerancias.** Los recuentos y los motivos deben coincidir exactamente. Los valores reales usan una tolerancia relativa de 1e-10 y absoluta de 1e-13, que cubre la diferencia de orden de suma entre las dos implementaciones. Un valor indefinido solo coincide con otro indefinido.

El control cero no tiene archivo de predicciones. Se recalcula con predicción nula sobre los objetivos de la ventana, que la identidad de filas exige comunes a todos los modelos, y se coteja igual que los demás, también en los agregados de cada vista.

**Límite declarado.** Las métricas con los cuantiles calibrados no se recalculan todavía, porque el calibrador común no tiene segunda implementación. El informe de cada recálculo indica la exclusión.

**Pruebas.** `tests/integrity/` incluye:
- los ejemplos de la especificación;
- la regla de la probabilidad de subida con empates en cero;
- la coincidencia con el código principal en paneles con objetivos y predicciones nulos, sesiones de un activo y dos mercados;
- seis corrupciones de valores publicados que deben detectarse.

También hay dos mutaciones del código principal, una en la media por sesión y otra en la puntuación de intervalo. El recálculo las señala columna a columna sobre un estudio de juguete completo.

## Registro de métricas

`configs/evaluation/metric-registry.json` fija, antes de evaluar, cada cifra que publica el resumen de una vista: definición, unidad, sentido, regla de agregación entre sesiones y sección de [las métricas](metrics.md) donde se explica. Son 85 entradas, desde el MAE hasta cada punto de la curva de fiabilidad. Las unidades distinguen el retorno residual simple del objetivo (0,01 es un 1 %), su cuadrado, fracciones, porcentajes, correlaciones y recuentos.

Los huecos siguen reglas comunes declaradas en el mismo archivo:
- una sesión sin denominador o sin estado definido queda fuera de la media y se cuenta, nunca vale cero;
- una vista sin sesiones definidas publica la cifra nula;
- una fila que falta en un modelo hace fallar la identidad de filas y bloquea la publicación;
- las filas que excluye el diseño conjunto se predicen y quedan fuera de la calibración y de las métricas.

`integrity/metric_registry.py` lo comprueba de dos formas:
1. **Contra el código.** `load_registry` exige que las ponderaciones, la agregación de semillas, las filas excluidas, los estados de la correlación de rangos, las coberturas de la abstención, los intervalos de probabilidad y las convenciones de dirección, abstención y probabilidad de subida sean las de la comparación. Cada serie que usan los contrastes debe apuntar a una cifra registrada con el mismo sentido, y cada sección citada debe existir en la documentación.
2. **Contra cada informe.** `check_summary` reconstruye el resumen en bruto de cada modelo, semilla y vista desde la tabla de sesiones publicada, solo con las reglas del registro y la media de `independent_scores`. Falla con una cifra publicada sin registrar, una registrada que no se publica o un valor distinto. Del resumen calibrado solo se exige que cada cifra esté registrada, porque el calibrador común no tiene segunda implementación.

En una vista de un mercado se reconstruyen 187 cifras con cuantiles y 97 sin ellos, y cada mercado más de la vista conjunta añade las 14 cifras puntuales de ese mercado. Quedan sin recalcular, con su motivo en el registro, los compromisos de signo de los intervalos (la tabla de sesiones no los guarda), el número de periodos, las filas con predicción nula y objetivo nulo, la igualdad entre predicción puntual y mediana y las identidades. Reconstruir una vista cuesta 0,12 s con 6.000 sesiones y 0,24 s con 12.000, medido con la máquina compartida. Convertir una sola vez por vista las columnas de listas de la abstención y de la fiabilidad lo bajó desde 1,21 s, porque esas conversiones eran el 92 % del tiempo. El recálculo y el informe de integridad de cada comparación guardan la huella del registro. La declaración se añade al [registro previo](#registro-previo-y-desviaciones) junto a la configuración de la campaña.

**Pruebas.** `tests/integrity/test_metric_registry.py`, con las dos ponderaciones:
- todas las vistas del estudio de juguete se reconstruyen sin diferencias;
- en un panel con sesiones de 1 a 8 filas, ceros, empates y dos mercados, el resumen del código coincide con el registro, y ahí la media por fila y la cobertura de llamadas agrupada sí difieren de sus versiones por sesión (en el estudio todas las sesiones tienen las mismas filas y no las distinguen);
- veinte cifras cambiadas una a una (RMSE, porcentajes, exhaustividad, una cifra de un mercado, medias por fila, cobertura de llamadas, recuentos, la curva de abstención, pinball, frecuencias, intervalos, Brier, ECE y fiabilidad) y cuatro convenciones o listas cambiadas se detectan, cada una con su nombre y sin arrastrar otras;
- una cifra nueva sin registrar, un nivel de cuantil de más, una cifra que desaparece y una cifra calibrada sin registrar hacen fallar la comprobación;
- el control cero publica nula la exactitud condicionada, y un cero en su lugar se detecta;
- 23 registros alterados se rechazan al cargarlos, desde una convención o una cobertura distintas hasta una serie con el sentido cambiado o una sección inexistente.

## Mismas filas en todos los modelos

La comparación ya exige que todos los modelos evalúen las mismas filas, con `ForecastPanel.cohort_sha256`. `integrity/row_identity.py` lo comprueba por otro camino:
1. Ordena las filas de cada archivo de predicciones por mercado, activo e instante con Arrow.
2. Calcula una huella SHA-256 propia con los bytes de esas tres claves y los bits exactos del objetivo en float64.
3. Agrupa las huellas por ventana y tramo (calibración y evaluación) y exige que todos los pares modelo-semilla compartan una sola.

Un bit distinto en un objetivo, una fila de más o una de menos separan al modelo afectado en su propio grupo, y el informe lo nombra.

Con el diseño conjunto de A v2, la comparación quita de cada ventana las filas de los mercados que aún no son elegibles (China antes de `fold-006`) y, en un modelo prestado del ámbito conjunto, las del mercado ajeno al ámbito. La comprobación aplica de nuevo esa regla declarada antes de calcular la huella y cuenta por mercado las filas quitadas. Las de cualquier otro mercado siguen siendo un defecto.

Además, cuenta en cada archivo:
- las filas de la reserva de 2024;
- las filas fuera del tramo declarado de su ventana;
- las de mercados fuera del ámbito;
- las claves duplicadas;
- los objetivos no finitos.

Antes de leer cada archivo comprueba su huella frente al manifiesto de fuentes.

```bash
uv run python -m mars_titan.integrity.row_identity \
  configs/evaluation/historical-masked-2000-comparison.json SOURCES.json US+CN --output ROWS.json
```

**Coste medido.** Entre 0,6 y 0,9 s por archivo de 1,25 millones de filas en CPU, con dos hilos. Es el orden de una ventana de evaluación US+CN.

**Alcance.** Cubre los archivos de calibración y evaluación que entran en la comparación. Las predicciones de validación con las que cada runner elige su punto de parada no pasan por este manifiesto. Su comprobación contra la reserva de 2024 queda en la lista siguiente.

**Pruebas.** Sobre el estudio de juguete de la comparación:
- un estudio fiel pasa en US y en US+CN;
- se detectan un solo bit cambiado en un objetivo, una fila que falta en un modelo y una fila movida a 2024;
- se cuentan las claves duplicadas y los objetivos no finitos;
- se rechaza un archivo cambiado tras publicar el manifiesto;
- la huella no depende del orden de las filas ni de su partición en bloques.

## Escáner de fugas en las entradas

`integrity/leakage_scan.py` busca columnas de entrada que contengan el objetivo futuro, sin entrenar ningún modelo. Una entrada que lleva el objetivo, aunque sea con ruido o con una transformación monótona, ordena los activos de cada sesión casi igual que él. Las señales reales sobre rendimientos residuales diarios tienen correlaciones de rangos por sesión del orden de centésimas.

Para cada columna y cada sesión (mercado e instante) calcula dos correlaciones de Spearman:
- con el objetivo de la fila;
- con un placebo pasado, que es el objetivo del mismo activo `lag` decisiones antes. Con `lag` igual o mayor que el horizonte de la etiqueta, ese placebo ya ha madurado en el momento de la decisión.

Las ausencias se excluyen columna a columna, y el objetivo se ordena solo entre las filas en las que la columna está presente. Una columna se marca si la media de |IC| supera 0,2 o si alguna sesión con al menos 20 activos supera 0,95. El placebo no decide nada: sirve para distinguir una entrada que copia rendimientos pasados (legítima, alta con el placebo) de una que copia el futuro.

```bash
uv run python -m mars_titan.integrity.leakage_scan INPUTS.parquet --lag 1 --output LEAKAGE.json
```

**Límites declarados.**
- Los umbrales se fijan antes de mirar los datos reales. No se estiman.
- Que no marque nada no demuestra que no haya fugas sutiles. Una fuga débil y repartida entre muchas columnas puede quedar por debajo de los umbrales.
- Se aplica sobre la matriz tabular de la campaña, que comparte con las familias neuronales la construcción de entradas de cada ventana, y solo sobre filas de validación.

**Pruebas.** Con paneles escritos en la prueba y fugas inyectadas a propósito:
- se marcan el objetivo con ruido, una transformación monótona, el rango invertido y una fuga en una sola sesión;
- no se marcan una señal débil realista, el ruido ni el rendimiento pasado;
- con ausencias, la media coincide con la correlación de Spearman directa de pandas sobre las filas presentes;
- una columna constante y las sesiones pequeñas quedan sin correlación definida.

Tres mutaciones dirigidas hacen fallar las pruebas: ordenar el objetivo también en las filas ausentes, quitar el mínimo de activos y quitar el umbral por sesión.

### Matriz real de cada ventana

`integrity/window_leakage.py` aplica el escáner a la validación de cada vista de un ámbito de la campaña. Recorre la partición con `CorpusDataset` y construye la matriz con `tabular_corpus._matrix`, la misma que reciben Ridge y XGBoost, con las ventanas de precios ya filtradas por su canal de presencia. Las familias neuronales leen las mismas entradas con el mismo lector, así que una fuga en la construcción de entradas de la ventana aparecería también aquí. No se lee calibración, evaluación ni el test reservado.

- **Columnas.** Cada columna tiene nombre propio (`prices.<sesión>.<canal>`, `news.<i>`, `charts.<i>`, `fundamentals.<i>`, `macro.<i>` y `presence.<bloque>`), para localizar una fuga en su modalidad y su posición.
- **Bloques ausentes.** Con la política de máscaras, un bloque ausente vale 0 en la matriz. El escáner lo trata como ausente, porque ese relleno empataría a muchos activos de la sesión y diluiría una fuga presente en las demás filas. Los bits de presencia se escanean como columnas, así que una fuga escondida en la ausencia de un bloque también se vería.
- **Memoria.** La matriz de una ventana conjunta ocupa varios GB en float32, así que se escribe por columnas en un archivo temporal junto al informe y se escanea en bloques de 32 columnas con `leakage_scan.scan_blocks`. Las correlaciones coinciden exactamente con las de la tabla completa en memoria. El archivo se borra al terminar, también si el escaneo falla.
- **Placebo.** El objetivo es el residual apertura-cierre de la sesión siguiente, así que el placebo usa un retardo de dos decisiones, que ya ha madurado al decidir.
- **Informe y reanudación.** Cada ventana publica su informe con la huella de su vista, y el resumen del ámbito enumera las ventanas, sus columnas marcadas y las cinco con mayor |IC| medio. Una ventana con informe de la misma vista no se repite, y una vista distinta se vuelve a escanear. El ámbito solo pasa si están todas las ventanas, ninguna tiene columnas marcadas y todas han comprobado al menos una columna.

```bash
uv run python -m mars_titan.integrity.window_leakage \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json --scope US+CN \
  --views <vistas A v2>/US+CN --lag 2 --output <escáner>
```

**Pruebas.** Con un corpus técnico de 64 activos y objetivos aleatorios:
- la matriz coincide con la tabular en las filas presentes, tiene los bloques ausentes vacíos y conserva claves, objetivos y orden del lector;
- una columna de gráficos con el objetivo y un poco de ruido queda marcada, y otra con solo ruido no;
- el resumen del ámbito reanuda por huella de vista, rechaza otro retardo y ventanas ajenas, y no aprueba ventanas marcadas, incompletas o sin columnas comprobadas;
- escanear en bloques de cualquier anchura da los mismos resultados que la tabla completa.

Con sesiones de 64 activos, una columna sin relación tiene un |IC| por sesión con desviación típica de 0,126, y el umbral de 0,2 sobre la media no queda tan lejos. Con una de las semillas probadas, un bit de presencia llegó a 0,208 por azar. En las ventanas reales casi todas las sesiones tienen cientos o miles de activos y esa dispersión baja a unas centésimas, pero una modalidad que empieza a tener cobertura puede llegar con pocas filas por sesión, como las noticias de fold-005.

**Revisión de una marca.** `review_window` revisa una modalidad marcada de una ventana sin cambiar los umbrales ni el resultado del escaneo. Reúne las filas que cuentan para la regla (sesiones con al menos 20 filas con la modalidad presente) y describe cada sesión por activos y vectores distintos, porque muchos activos con el mismo vector reducen la muestra efectiva. Después repite la puntuación del escáner, que debe marcar exactamente las mismas columnas que el informe, y la compara con un nulo por permutación (`leakage_scan.permutation_null`): el objetivo se permuta dentro de cada sesión, lo que conserva la composición de las sesiones y los empates de cada columna, y se cuenta cuántas columnas marcaría la regla y cuál sería la mayor media de |IC|. Las 1.000 permutaciones usan la semilla 20261014, fijada al escribir la revisión. Permutar el objetivo no cambia los rangos de las columnas, así que se calculan una vez por bloque de 32 columnas y cada permutación solo vuelve a ordenar el objetivo. Cada bloque repite los órdenes con la misma semilla y el resultado coincide byte a byte con puntuar todas las columnas en cada permutación. En los precios de una ventana de China (83.312 filas y 384 columnas) cada permutación pasó de 4,7 s a 0,25 s. El nulo solo dice si el azar basta para explicar la marca. Una asociación real con el objetivo queda fuera de él tanto si es una fuga como si es una exposición legítima a un factor, así que la decisión sobre la marca necesita además la fecha de las entradas y el placebo.

```bash
uv run python -m mars_titan.integrity.window_leakage \
  --campaign configs/baselines/historical-masked-campaign-a-v2.json --scope US+CN \
  --views <vistas A v2>/US+CN --lag 2 --output <escáner> --review fold-005 --modality news
```

### Resultado en las vistas v3.1 de US+CN

El escaneo recorrió las 19 ventanas de A v2 sobre las vistas v3.1 conjuntas el 13 y 14 de octubre de 2026: 7.985.418 filas de validación entre abril de 2004 y septiembre de 2022, 1.783 columnas por ventana y entre 831 y 1.286 comprobadas, según las modalidades con cobertura. Las ventanas sumaron 6,2 horas repartidas en tres procesos paralelos (2,2 horas de reloj), con 3,8 GiB de memoria como máximo por proceso. El resumen ([`window-leakage-20261014`](../../reports/engineering/window-leakage-20261014/summary.json)) está completo y no pasa: 17 ventanas salen sin marcas y dos tienen columnas marcadas. En las dos, la modalidad marcada aparece por primera vez en la validación.

**fold-005, noticias de US (validación de abril a septiembre de 2009).** Las noticias de US empiezan el 8 de abril de 2009 y en esta validación solo tres sesiones tienen al menos 20 activos con noticias: el 15 y el 16 de abril y el 25 de junio, con 25 activos cada una y 206 de las 384 columnas de noticias marcadas. Los 25 activos son ETF y comparten 3 o 4 vectores distintos por sesión, porque cada noticia es un artículo general asociado a muchos símbolos: «Leveraged ETFs Beat Margin» (fecha de origen 8 de abril, 17 ETF apalancados e inversos), «Metals ETFs: Precious and Base» (14 de abril, 8 ETF de metales) y «Top Steal Stock to Buy Now» (12 de abril, WOR) en abril, y en junio «Fundamental ETFs and Alternative Weighting», «ProShares Expands ETF Lineup», «Are Leveraged and Short ETFs Right for You?» y «U.S. Treasury ETFs and Inflation», del 18 al 24 de junio. Todas tienen fecha de origen entre uno y siete días antes de la decisión y llegan por la regla declarada de disponibilidad al cierre de la sesión siguiente, y las dos sesiones de abril comparten los mismos artículos. La [revisión](../../reports/engineering/window-leakage-20261014/fold-005.review-news.json) reproduce las marcas del informe sin ninguna diferencia. El 23,1 % de las permutaciones marca 206 columnas o más (mediana 68,5 y percentil 95 de 347), así que el número de columnas marcadas cabe en el azar con 75 filas y tres o cuatro grupos por sesión. En cambio, ninguna permutación alcanza la media de |IC| de 0,459 de la columna más fuerte (percentil 95 del nulo, 0,367). Como cada artículo da el mismo vector a todos sus ETF, esa asociación solo puede venir de qué ETF agrupa cada artículo, sobre todo metales frente a renta variable apalancada, y esa pertenencia se conoce antes de decidir. La lectura que encaja con la evidencia es una exposición a la clase de activo en dos cortes transversales, no información posterior a la decisión.

**fold-018, fundamentales de China (validación de abril a septiembre de 2022).** Se marcan las tres columnas de valor de los conceptos chinos (`cn-reported` activos, pasivos y patrimonio en CNY, posiciones 23 a 25 del catálogo conjunto), con una media de |IC| de 0,207 a 0,225 en 123 sesiones de China de entre 24 y 49 activos (49 en 87 de ellas), cada activo con su propio vector. Es la primera ventana con fundamentales chinos en la validación. El placebo con el objetivo de dos decisiones antes da prácticamente lo mismo (0,208 a 0,223) y la media con signo es casi cero (0,009 a 0,026): el signo cambia de una sesión a otra. Es lo que se espera de una medida de tamaño de la empresa en el mercado de acciones A, donde el factor tamaño cambia de dirección de un día a otro, y lo contrario de una copia del objetivo, que tendría el mismo signo y un placebo mucho menor. La [revisión](../../reports/engineering/window-leakage-20261014/fold-018.review-fundamentals.json) reproduce las marcas del informe sin ninguna diferencia. Ninguna de las 1.000 permutaciones marca ninguna columna, y la mayor media de |IC| del nulo tiene un percentil 95 de 0,137 frente al 0,225 observado, así que el azar no explica la asociación. El nulo no separa una fuga de una exposición legítima. Lo que las separa es el placebo: una columna que copiara el objetivo de t perdería casi toda su relación con el objetivo de dos decisiones antes, y estas la conservan entera. La lectura que encaja con la evidencia es una exposición al tamaño de la empresa, que se conoce antes de decidir.

Las dos marcas quedan revisadas con una explicación que no requiere información posterior a la decisión, pero el resultado del escaneo no cambia: la regla declarada marca estas columnas y el ámbito no pasa. Los umbrales no se han tocado. Cómo trata la puerta de publicación una marca revisada se decidirá al integrarla, y esa regla se escribirá conociendo estos resultados.

## Cota con información futura en el entorno financiero

`integrity/oracle_bound.py` pone en escala los resultados de las políticas y detecta trampas por el resultado. El entorno decide al cierre de t y ejecuta en la apertura de t+1. La posición se mantiene hasta la apertura de t+2, donde la decisión siguiente puede cambiarla, y la última se valora al cierre final. El rendimiento de esa tenencia es `open[t+2] / open[t+1] - 1` (en la última decisión, `close[T-1] / open[T-1] - 1`), con los desdoblamientos, dividendos y bajas que la cartera aplica dentro de ella. Una baja con precio de salida cambia las acciones por su cobro, una baja sin recuperación lo pierde todo y una baja sin precio de salida deja la tenencia sin rendimiento, porque el episodio que la mantenga termina sin patrimonio. Una apertura ausente no se negocia y la posición pasa a la sesión siguiente: esa tenencia se reparte entre las dos decisiones valorándola al cierre de su misma sesión, y el producto de los dos tramos es el rendimiento real.

Con ese rendimiento hay dos cotas:
- **Cota exacta.** Suma en cada decisión `log1p(max(0, max_i r_i))`. Sin apalancamiento, cortos ni costes, el patrimonio de una tenencia crece como mucho lo que su mejor activo, y la caja no rinde. Relaja lotes, límites de precio, liquidez, impuestos y el tamaño de las órdenes, así que ningún episodio sin costes puede superarla salvo por redondeo (1e-9 en logaritmo). El entorno no la alcanza.
- **Política con información futura.** Usa el rendimiento como puntuación con la asignación, los costes y la liquidez del entorno y exposición completa. Reparte por igual entre el cuartil superior de los rendimientos positivos, así que no es el máximo de la regla: una política que concentra en uno o dos activos puede superarla sin ninguna fuga.

`bound` evalúa con el evaluador de políticas fijas que ya usa la simulación:
- la política de la cinta con exposición completa;
- la caja;
- la política con información futura.

Después informa de la fracción del crecimiento logarítmico de la política con información futura que alcanza la cinta y de la cota exacta. Una fracción de 0,5 o más se marca como sospechosa, porque casi con seguridad indica información futura en las puntuaciones, aunque no lo demuestra.

**Por qué hace falta.** La cinta comprueba las marcas de tiempo de las predicciones, no su contenido. Una cinta con puntuaciones del futuro y marcas correctas pasa su contrato. La cota ataca ese hueco por el resultado.

**Límites declarados.**
- La política con información futura es miope, de un paso, y reparte con la regla del entorno. Hasta el 14 de octubre el texto la presentaba como el máximo sin costes. La revisión de #584 lo refutó con el paseo de 60 sesiones y 8 activos de las pruebas: sin costes, la regla crece 1,901 en logaritmo y unas puntuaciones que solo dejan positivo el mejor activo de cada decisión llegan a 2,079. La cota exacta de esa cinta es 2,156. Desde entonces `exceeds_bound` se compara con la cota exacta y la política con información futura solo sirve de escala para la fracción sospechosa. Con costes rota en cada sesión y no optimiza la rotación: comprar y mantener puede superarla.
- La cota exacta suelta todas las restricciones del entorno. Es un máximo seguro, pero holgado, y por eso no sirve de escala: una política honesta queda muy lejos de ella.
- Hasta el 13 de octubre puntuaba con `close[t+1] / open[t+1] - 1`, que ignoraba el hueco nocturno que la posición sigue soportando. Se cambió al conectarla con la etapa de políticas (#365).
- Es un diagnóstico. Su cinta lleva un padre propio (`oracle_one_step_future_returns_diagnostic`) y el informe declara `excluded_from_comparisons`.
- Nunca se ejecuta sobre el test de 2024. La etapa de políticas calcula las dos cotas sobre la cinta de evaluación walk-forward de cada trabajo, enlaza su archivo en cada recibo y el informe bloquea cualquier episodio que supere la exacta o alcance la fracción sospechosa ([octava fase de la integridad de los entornos](../engineering/rl-environment-integrity.md#octava-fase-cota-con-información-futura-en-cada-recibo)).

**Pruebas.** Con cintas escritas en la prueba:
- la puntuación es el rendimiento de la tenencia que abre cada decisión y solo cambia con las aperturas que cruza;
- un desdoblamiento y un dividendo dentro de la tenencia entran en su rendimiento, una baja sin recuperación lo pierde todo, una baja con precio de salida paga su cobro y una baja sin él deja la tenencia sin rendimiento;
- una apertura ausente se valora en el cierre de su sesión, el producto de los dos tramos es el rendimiento real y no se puede comprar en ella;
- la cota exacta suma el mejor activo o la caja de cada decisión y vale cero en un mercado plano;
- ninguna política sin costes la supera en cuatro paseos aleatorios, con puntuaciones que concentran en uno o dos activos, la información futura completa o ruido, y con participación del 1 % o completa. Tampoco la superan una tenencia que cobra una baja con prima ni otra que cruza una apertura ausente, que sin esos dos tratamientos la superarían;
- el caso de la revisión de #584, con el mejor activo de cada decisión, supera a la política con información futura y queda por debajo de la cota exacta;
- la cinta de la política con información futura tiene identidad propia y marcada;
- en un mercado plano la política con información futura no gana nada y una política de compra constante pierde los costes;
- con 60 sesiones y 8 activos en paseo aleatorio y el coste por defecto del entorno, la política con información futura crece 1,798 en logaritmo, mientras que puntuaciones de ruido alcanzan 0,039, un 2,2 % de la cota, sin marcarse;
- unas puntuaciones con la información futura alcanzan el 100 % y se marcan.

## Registro previo y desviaciones

`integrity/preregistration.py` fija qué se va a comparar antes de producir resultados. El registro es un archivo JSON Lines que solo crece, con dos tipos de entrada:
- **Declaración.** Ruta y huella SHA-256 de un documento, por ejemplo la configuración de la comparación walk-forward, con sus modelos, métricas, familias de contrastes y parámetros del bootstrap. Va acompañada de una nota.
- **Desviación.** Cambio posterior sobre una declaración, con el motivo y la huella del documento que la sustituye. La declaración original no se borra.

Cada entrada guarda la huella canónica de la anterior. Si se edita, se borra o se reordena una línea, la cadena se rompe, aunque se recalcule la huella de la línea editada, porque la entrada siguiente apunta a la huella antigua. Las entradas inválidas se rechazan antes de escribir: desviaciones sin declaración, un documento declarado dos veces o fechas que retroceden.

`check_report` comprueba un informe de comparación:
1. Busca la declaración cuya huella coincide con `configuration.sha256` del informe.
2. Exige que la fecha de la declaración sea anterior a la del informe.
3. Lista las desviaciones registradas sobre esa declaración.
4. Si recibe el repositorio, busca el primer commit que añade la entrada, exige que sea anterior al informe e indica si ese commit ya está en un remoto.

```bash
uv run python -m mars_titan.integrity.preregistration declare \
  configs/evaluation/preregistration.jsonl configs/evaluation/historical-masked-2000-comparison.json \
  --note "Comparación principal de la campaña A"
uv run python -m mars_titan.integrity.preregistration check \
  configs/evaluation/preregistration.jsonl COMPARISON_DIR/comparison.json --repository . --output PREREG.json
```

**Límite declarado.** Las fechas locales y las de commit las fija el propio autor, así que por sí solas no prueban nada frente a terceros. La prueba externa es publicar el commit de la declaración en GitHub antes de lanzar la evaluación: el informe lo indica en `published_before_report`. La cadena tampoco impide declarar muchas variantes y publicar solo la favorable. Por eso todas las declaraciones y desviaciones del registro se publican con los resultados.

**Uso previsto.** La configuración definitiva de la campaña se declara cuando esté fusionada en `develop` y antes de generar ninguna predicción de evaluación. Hasta entonces el registro del repositorio no existe, para no acumular desviaciones sobre configuraciones que siguen cambiando.

**Pruebas.**
- Una declaración y su desviación forman una cadena válida.
- Se detectan una entrada editada (también con la huella recalculada), una eliminada y dos reordenadas.
- Las entradas inválidas no llegan al archivo.
- Un informe anterior a su declaración, o sin declarar, falla.
- Con un repositorio git temporal y un remoto local, se distinguen el commit local, el commit publicado y el commit posterior al informe.

## Informe de integridad de la publicación

El paso final de la campaña (`training/campaign_publication.py`) no publica ninguna cifra sin este informe. `integrity/publication_gate.py` lo calcula en la carpeta provisional, después de escribir comparaciones, matriz y comparación postentrenada y antes de darle el nombre del destino. Si alguna comprobación falla, `integrity.json` se queda en la carpeta provisional para revisarlo y la publicación termina con un error. El recibo de una publicación terminada guarda la huella del informe.

**Comparaciones.** Para cada comparación walk-forward y cada comparación postentrenada publicadas, `integrity/comparison_integrity.py` comprueba:
- la identidad de filas y la ausencia de 2024 de la sección anterior;
- las filas quitadas por el diseño conjunto, contadas por separado y cotejadas con las que publica la comparación en cada modelo, semilla, ventana y tramo;
- las sesiones en bruto recalculadas con la segunda implementación, control cero incluido, y las métricas agregadas de cada vista.

Con la retención v2 las filas de una ventana se liberan en cuanto se guardan sus agregados, así que al publicar ya no existen. Por eso la fase `aggregates` del recorrido escribe, junto a cada agregado, un registro `<ventana>.integrity.json` con esas mismas comprobaciones sobre las filas de la ventana, la huella de los agregados y su identidad (fuentes, configuración y código). Si el registro no está superado, el recorrido se detiene antes de liberar nada. Al publicar desde agregados, el informe exige el registro superado de cada ventana con la huella y la identidad de los agregados actuales, que las sesiones publicadas coincidan sin tolerancia con las de los agregados y que las métricas del informe se vuelvan a obtener desde ellas.

**Ejecuciones declaradas.** `integrity/declared_runs.py` vuelve a planificar la campaña y su etapa de adaptadores desde sus declaraciones. Cada trabajo planificado debe tener su recibo confirmado, o su selección en la cadena, y se cuentan todos sus intentos, también los que fallaron o se interrumpieron, junto con la parada temprana que declare su informe. Un recibo de un trabajo que el plan no declara hace fallar el informe, y también una comparación publicada que no tenga exactamente las semillas y ventanas declaradas de cada modelo, o que publique un modelo sin declarar.

**Selección.** `integrity/window_selection.py` lee los recibos confirmados, sin predicciones por fila:
- en cada ventana reentrenada, la búsqueda elegida es la de menor MAE de validación, con empate por identificador, y esa puntuación es la del informe del ajuste;
- cada finalista repite el caso elegido con su semilla;
- cada recibo de ventana de un modelo comparado nombra el estado elegido y declara una última etiqueta usada anterior a la evaluación;
- los tramos de cada ventana van en orden y la comparación evalúa la evaluación con la calibración como único tramo previo.

Con el walk-forward por etapas, o con una etapa de adaptadores por etapas publicada, añade el verificador de disjunción de [#437](https://github.com/GonxKZ/mars-titan/issues/437) sobre vistas, recibos base y selecciones de la cadena, sin duplicarlo.

**Presupuesto.** `integrity/fit_budget.py` compara el presupuesto de cada modelo ajustado con el de los modelos propios (MARS-TITAN, Titans-MAC y CM-v1), que fija el primer modelo propio declarado. Reúne del plan los casos de búsqueda, las semillas y la forma de iterar: unidad (épocas, rondas, actualizaciones en línea o solución cerrada), entrenador (lotes de filas o recorrido cronológico), máximo, regla de parada y paciencia. Cada campo distinto debe figurar con su motivo en `budget_differences` de la declaración de la publicación. Una diferencia sin declarar, o una declarada que la campaña ya no tiene, impide cargar la declaración y hace fallar el informe. En A v2 se declaran los tabulares (rejilla propia, una semilla y sin épocas), las referencias neuronales (mismas épocas y regla, con lotes de filas) y el control en línea (rejilla de tasas y actualizaciones limitadas por el banco). Después, para cada ajuste confirmado:
- repite la regla de selección sobre su historial de validación y exige que el ajuste termine donde la regla lo manda (todas las épocas con presupuesto fijo, la primera meseta con parada individual y la época común con parada conjunta) y que entregue la época que la regla elige;
- un ajuste sin padre que entrega su estado inicial, la época 0 que evalúan los entrenadores cronológicos, o que no declara actualizaciones, no está ajustado. Una referencia sin ajustar no cuenta como control y un modelo propio sin ajustar tampoco cuenta como modelo;
- con presupuesto fijo, los ajustes del mismo entrenador en el mismo ámbito y ventana aplican las mismas actualizaciones, y los de un grupo de parada conjunta terminan en la misma época;
- XGBoost no supera sus rondas, elige una ronda ajustada y declara por qué para antes, y el control en línea no usa en ningún tramo más etiquetas que las escrituras del banco que lo limita.

La etapa de adaptadores ya rechaza en su ejecutor un número de actualizaciones distinto del de su diseño. El informe todavía no repite esa comprobación.

**Abstención.** Una predicción cero es una abstención. La exactitud de dirección ya la cuenta como fallo, pero las precisiones de subida y de bajada solo juzgan las llamadas, así que un modelo que casi nunca llama puede mostrar una precisión alta con muy pocas filas. Las reglas están fijadas antes de evaluar en `evaluation/abstention_coverage.py` y cada comparación walk-forward y cada matriz las publican (detalle en [métricas](metrics.md#abstención-con-su-cobertura)):
- la curva riesgo-cobertura de dirección de cada semilla y vista, con la exactitud entre las ceil(c · n) filas con objetivo no nulo y mayor |predicción| de cada sesión para c de 0,1 a 1, empates por la identidad de fila y la abstención contada como fallo;
- el valor a la cobertura fija 0,5 (`direction_accuracy@0.5`), declarado junto a la curva;
- la cobertura de llamadas de cada modelo en cada vista, sumada sobre sus semillas;
- en las precisiones de subida y de bajada, cada contraste con su cobertura y marcado no elegible si algún modelo llama en menos de la mitad de sus filas juzgables. Se publica, pero no puede presentarse como ganado.

`integrity/abstention_marks.py` lo repite desde la tabla de sesiones publicada, sin el código de la comparación. Recalcula la cobertura de llamadas y exige las reglas declaradas, la marca y la cobertura de cada contraste condicionado a las llamadas, en la comparación y en cada vista de la matriz, y las filas retenidas y los aciertos de cada punto de la curva. La segunda implementación de las métricas recalcula los aciertos de cada cobertura con su propia ordenación por identidad de fila, y el recálculo los coteja sesión a sesión.

**Preprocesado.** Normalizaciones, residualizadores, umbrales de régimen y estratos solo pueden usar datos anteriores al tramo en que se aplican. `integrity/past_only_preprocessing.py` lo comprueba en cuatro partes:
- **Código.** Recalcula las dos implementaciones de las etiquetas residuales (`data/residual_arrays.py` y la referencia `data/budget_targets.py`) y las rutas del calendario de regímenes con precios deterministas del propio módulo, después de multiplicar alternativamente por 1,7 y 0,6 los precios posteriores a cada uno de tres cortes. Alfa, beta, historia y motivo de cada decisión hasta el corte, los objetivos ya maduros y la ruta, los activos y los rendimientos de cada sesión hasta el corte deben quedar iguales bit a bit. Una auditoría que no compara ninguna fila no pasa.
- **Edición.** Las etiquetas y el calendario se calculan fuera de la campaña. Cada edición que leen las vistas de una comparación publicada debe tener la huella que fijan esas vistas y declarar el mismo código de etiquetas que se audita, y el calendario de regímenes el mismo código que `regime_calendar`. Con la política `real_edition_only` de A v2, una edición o un calendario sin código declarado también fallan. El manifiesto de las etiquetas v3.1 declara las tres huellas y coinciden con el código actual.
- **Ajustes.** Ridge y XGBoost calculan medias, escalas y cortes de histograma exactamente con las filas de entrenamiento. Las escalas de escritura de M3 salen del índice de entrenamiento de la ventana y de su edición, con sus mismos límites de decisión, y esas etiquetas maduran antes de la primera decisión de validación. Las demás familias no estiman estadísticas con datos y lo declaran con su motivo, y una familia sin inventariar falla. En la etapa de adaptadores, la normalización de cada padre recorre solo las filas reales de entrenamiento, maduras antes de validación.
- **Reglas.** Cada sección de la configuración de la comparación está en un inventario con su regla causal: calibración en su propio tramo, presencia de modalidades de la muestra de la decisión, cuartiles de la misma sesión, calendario con ventanas que terminan en la decisión y estratos de liquidez con umbrales fijos sobre la edición sin ajustar hasta la última sesión publicada al emitir. Una sección nueva hace fallar el informe hasta que se declare su regla.

**Pruebas.** Cada mutación dirigida rompe el código principal o un artefacto y debe hacer fallar el informe:
- una fila de 2024 en las predicciones con la comprobación de tramos desactivada en la comparación;
- una media por sesión sesgada en el código principal, al publicar desde filas y al guardar los agregados de una ventana, que entonces no se libera;
- el registro de una ventana ausente, fallido, de otros agregados o de otro código;
- un ulp de diferencia entre las sesiones publicadas y las de los agregados, también al publicar por la orden de la campaña;
- una métrica publicada que ya no sale de sus sesiones, al publicar desde filas y desde agregados;
- filas quitadas mal contadas, un recuento publicado que no coincide con el recontado y un mercado elegible quitado a todos los modelos en el diseño conjunto;
- un trabajo sin recibo, un recibo sin declarar y una comparación sin una semilla, sin una ventana o con un modelo de más;
- una puntuación de búsqueda que no es la de su validación, una última etiqueta que alcanza la evaluación, un recibo de ventana que no nombra el estado elegido y tramos fuera de orden;
- un fallo del verificador de disjunción;
- en un estudio de juguete donde un modelo solo llama una vez por sesión y acierta siempre, con precisión al alza perfecta, sus contrastes de precisión salen no elegibles en la comparación y en la matriz, su exactitud de dirección y su exactitud a la cobertura fija quedan por debajo de las de los modelos que llaman y el control cero no acierta a ninguna cobertura. Una marca, una cobertura, una regla o un punto de la curva cambiados hacen fallar el informe, y también una marca cambiada en la matriz;
- una diferencia de presupuesto sin declarar o declarada sin existir, un ajuste truncado, otro que sigue tras su meseta o su época común, una época entregada distinta de la que elige la regla, un estado inicial entregado sin padre, actualizaciones distintas entre ajustes del mismo entrenador y los límites de XGBoost y del control en línea. Una prueba con las cuatro variantes reales de Titans-MAC y un optimizador que no cambia pesos espera que el informe las marque sin ajustar, porque ninguna época mejora la validación inicial. Esa prueba y la de la referencia neuronal real ejecutan los entrenadores, así que se omiten mientras rige el bloqueo y todavía no se han ejecutado. El resto de `tests/integrity/test_fit_budget.py` usa el plan real de A v2 con recibos e historiales escritos en la prueba.
- en el preprocesado, cada ajuste deliberado con el futuro: un residualizador con la beta de toda la muestra y otro que escala el objetivo con la desviación de todos los objetivos, rutas de régimen con la ventana adelantada cinco sesiones o con el umbral de turbulencia en la mediana de toda la serie, estadísticas tabulares que recorren también validación, escalas M3 de otro índice, de otra edición, de otras decisiones o con etiquetas que maduran en validación y una normalización de un padre con otro tramo, otras filas o etiquetas tardías. También una edición que no es la de sus vistas o de otro código, un calendario de otro código, código sin declarar con la política real y una sección sin inventariar o con una regla no causal.

Las pruebas que recorren la campaña reducida con su etapa de adaptadores necesitan rebajar la protección de aprendizaje y se omiten mientras rige el bloqueo (inventario de [#546](https://github.com/GonxKZ/mars-titan/issues/546)). `tests/integrity/test_declared_runs.py`, `tests/integrity/test_window_selection.py` y `tests/integrity/test_publication_gate.py` repiten las mismas comprobaciones y la conexión del informe con la publicación, el recorrido y la etapa sobre recibos escritos en la prueba, sin ejecutar ninguna campaña.

## Comprobaciones pendientes

Siguen en #442, en este orden:
1. Comprobaciones sin ajuste:
   - aplicar el escáner de fugas a la matriz tabular real de cada ventana de los ámbitos US y CN, cuando se preparen sus vistas v3.1 (el ámbito conjunto está en [Matriz real de cada ventana](#matriz-real-de-cada-ventana));
   - ninguna fila de 2024 en las predicciones de validación y en los demás artefactos de selección;
   - invariantes de caja, posiciones, costes y acciones imposibles en las cintas reales con políticas fijas;
   - cota con información futura sobre las cintas reales de cada ventana. La etapa de políticas ya la aplica a la evaluación y al estado elegido en validación. Falta ejecutarla sobre las cintas reales, que la etapa calcula al confirmar cada trabajo.
2. Falsaciones que ajustan modelos, solo declaradas mientras dure el bloqueo de aprendizaje: placebo con etiquetas permutadas dentro de cada sesión, desplazamiento temporal de las entradas, predicción constante y aleatoria y estabilidad entre semillas. Cada una con su coste y su criterio de éxito o fracaso escritos de antemano.
