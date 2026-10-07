# Datos y evaluación conjunta de US y CN

El brazo `US+CN` utiliza una representación común y conserva el calendario de cada mercado. La preparación parte de dos ediciones reales con cuatro modalidades y 140 indicadores macro. Exige los mismos codificadores, longitudes de contexto, orden de indicadores y regla de agregación de noticias.

La edición estadounidense aporta 402.826 muestras de 2.198 empresas. La edición china revisada aporta 5.374 muestras de 25 empresas. Su unión contiene 408.200 muestras y sigue siendo una edición parcial, porque la cobertura china está pendiente de ampliación. Estos recuentos describen las muestras disponibles antes de aplicar las ventanas temporales. No son filas de entrenamiento de cada modelo ni se suman entre folds.

## Representación y procedencia

`mars_titan.data.joint_corpus` prepara una edición Parquet compartida. Los 23 conceptos estadounidenses y los tres conceptos chinos ocupan canales distintos. El vector contable contiene 26 valores, 26 máscaras y 26 edades. Los conceptos que no corresponden al mercado tienen valor, máscara y edad cero. No se convierten monedas ni se equiparan taxonomías contables.

Los precios y las etiquetas se copian con igualdad de bytes. En las muestras solo cambia la disposición del vector contable. Noticias, gráficos, macro, fechas, índices de precios y demás columnas deben permanecer iguales. La [proyección de contextos](../data/joint-accounting-contexts.md) conserva los bits de los valores originales y utiliza bloques de memoria acotados. La preparación reutiliza los vectores ya calculados y no ejecuta nuevos codificadores.

Cada activo conserva las huellas de sus tres artefactos originales, la del manifiesto de origen y la de su representación. La configuración identifica el código de transformación. Un recibo por activo permite recuperar una interrupción sin volver a proyectar los activos confirmados. Las fuentes y la configuración se vuelven a comprobar antes de publicar el informe final y al reutilizar una salida terminada.

La admisión rechaza dimensiones incompatibles antes de descomprimir los vectores. También comprueba valores finitos, las 140 máscaras macro, disponibilidad anterior a la decisión, índices y ventanas OHLCV válidos y correspondencia entre fechas, posiciones y etiquetas. Un índice de precios inválido o una etiqueta trasladada a otro año no puede producir un recibo de finalización válido.

## Ventanas, consumidores y comparaciones

`mars_titan.training.joint_temporal_corpus` prepara las ventanas de US y CN por separado y después reúne sus filas. Los protocolos deben compartir intervalos mensuales, semillas, margen entre particiones y reserva final. Cada fila conserva la disponibilidad macro, el calendario y el límite efectivo de su mercado. Una sesión de China no desaparece porque ese día sea festivo en Estados Unidos.

Los manifiestos conjuntos declaran `temporal_views` con contratos US y CN. El formato anterior de un solo mercado conserva `temporal_view`. Declarar ambos formatos a la vez provoca un error. Los lectores de entrenamiento, las cohortes ordenadas, la caché de predicciones padre y los consumidores de postentrenamiento seleccionan el contrato que corresponde a cada activo.

Cada ventana separa entrenamiento, validación, calibración y evaluación. El test final de 2024 permanece cerrado. La recuperación de etiquetas que cruzaban el antiguo corte anual requiere una opción explícita y conserva la maduración y los cortes del nuevo protocolo.

Las diez ventanas preparadas admiten 405.694 filas distintas, con 400.367 de US y 5.327 de CN. Las evaluaciones reúnen 215.181 filas sin repeticiones entre meses, de las que 2.844 son chinas. Se ha comprobado que todas las etiquetas de los diez folds son idénticas byte a byte a las de las ediciones locales anteriores. La mayor ventana contiene 321.610 filas de entrenamiento, 43.185 de validación, 21.936 de calibración y 17.961 de evaluación. Sus cuatro particiones se han recorrido completamente con el lector conjunto.

El análisis conjunto informa errores e intervalos por mercado. El bootstrap temporal utiliza las sesiones de un mercado en cada contraste. Los radios de los intervalos predictivos se ajustan por modelo, fold, semilla y mercado usando su partición de calibración. Si falta información en un mercado, el resultado declara su denominador o intervalo indefinido. No toma observaciones del otro para completarlo.

`cases.csv` conserva una fila por estado y partición e incluye sus recuentos locales. Los agregados, intervalos y diagnósticos de fiabilidad incluyen `market` en su identidad. Las figuras separan los mercados en paneles. El exportador comprueba que los recuentos, estados congelados y huellas de predicciones coinciden entre los informes.

Para atribuir diferencias al mercado añadido, los tres brazos deben partir de esta misma representación, presupuesto y protocolo. La campaña estadounidense anterior conserva su representación de 69 componentes contables. Compararla directamente con un modelo de 78 componentes requiere declarar también ese cambio. Esta preparación no completa la comparación científica entre los tres brazos ni acredita una mejora predictiva.

El candidato MARS-TITAN continúa en diseño. La simulación histórica con posiciones persistentes mantiene su condición de admisión sobre precios, ajustes y acciones corporativas.

## Comprobaciones y recursos observados

La [evidencia del 7 de octubre](../../reports/resources/joint-market-corpus-20261007.json) identifica fuentes, huellas, poblaciones, pruebas y límites. El productor y sus consumidores pasan 102 pruebas CPU focales. La revisión del productor incluye ocho mutaciones dirigidas que detectan las regresiones de disponibilidad, etiquetas, recuperación y configuración. La fiabilidad por mercado tiene cuatro mutaciones adicionales. Las suites se solapan y sus recuentos no se suman como pruebas distintas.

En CUDA se ejecutaron 23 pruebas. El caso conjunto completó 30 ajustes técnicos GRU en diez ventanas, con interrupción y recuperación. Los checkpoints confirmados conservaron sus huellas y fechas. La evaluación de calibración y evaluación mantuvo las filas de ambos mercados. También se comprobaron las rutas anteriores de CN, la evaluación de padres y los lectores predictivos. Estos ensayos usan fixtures y no acreditan una campaña científica conjunta.

La materialización corregida tardó 137,39 segundos, con un pico de RAM de 310 MiB. La preparación temporal tardó 180,57 segundos y alcanzó 786 MiB. Son medidas de una ejecución con la campaña US activa. La preparación temporal coincidió con la comprobación de paridad. Los artefactos comprimidos de entrada ocupan 2.283.983.893 bytes y los de salida 2.301.996.743 bytes. No se midieron energía, transferencias CPU/GPU ni el pico de VRAM de las pruebas.

El recorrido emplea las operaciones nativas de Arrow y NumPy. La [comparación de la proyección](../../reports/resources/joint-context-projection-20261007.json) comprueba la copia de canales frente a una referencia y separa tamaños pequeños y grandes. Estas medidas no demuestran una aceleración del entrenamiento completo ni justifican un nuevo kernel propio.
