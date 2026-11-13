# Campaña de comparación en dos equipos

Este documento registra el diseño de la campaña de comparación de [#836](https://github.com/GonxKZ/mars-titan/issues/836). Se fija antes de ejecutar ningún trabajo de la campaña. La [declaración](../../configs/experiments/comparison-campaign-v1.json) contiene todos los valores y el [motor](../../src/mars_titan/training/comparison_campaign.py) la lee sin añadir decisiones. Las órdenes de cada equipo están en la [guía de uso](../engineering/comparison-campaign-two-machines.md).

Nada de la campaña se ha ejecutado. Las únicas medidas con pasos de optimizador son las del piloto acotado que se describe más abajo, hecho con la ventana de origen 2018, que no forma parte de la campaña. El test de 2024 sigue cerrado y la protección local del aprendizaje no se ha modificado: la campaña usa un permiso propio que registra quien la lanza.

## Pregunta

La campaña responde al objetivo general de la memoria: si MARS-TITAN y la red de áreas con grupos de activos reducen el MAE residual por sesión frente a sus controles y frente a la predicción cero. Para poder responderla vuelve a ajustar, con la misma población, las mismas filas y el mismo presupuesto, las referencias de la memoria, el núcleo Titans-MAC, MARS-TITAN con sus políticas de escritura y las variantes que se prueban por separado.

Un resultado negativo también es un resultado. La segunda entrega de la memoria recoge que ninguna referencia ni variante exploratoria había mejorado a la predicción cero con señal propia. La campaña no presupone que eso cambie.

## Reparto entre los dos equipos

| Cola | Modelos | Trabajos |
| --- | --- | --- |
| `references` | Transformer compacto, GRU, LSTM, RNN, DLinear, Ridge, regla de momentum y reversión, LightGBM | 144 |
| `mars_titan` | Titans-MAC, MARS-TITAN M0 a M3, controles de memoria, red de áreas, grupos de activos, atención entre activos, refinamientos y puente de evidencia | 480 |

El primer equipo ajusta los modelos que la memoria ya presentaba como referencias. El segundo ajusta solo Titans-MAC, MARS-TITAN y sus variantes. Los lectores episódicos necesitan en su mismo equipo el Titans-MAC de su origen y su semilla, así que el núcleo y todo lo que depende de él van juntos.

## Población y orígenes

Cada origen tiene su propia cohorte de 1.024 activos de EE. UU., la misma para todos los modelos y semillas de ese origen. La regla es la de [`comparison_cohorts`](../../src/mars_titan/training/comparison_cohorts.py): se ordenan por una huella SHA-256 con sal fija los activos que tienen alguna etiqueta admitida en el tramo de ajuste del origen y se toman los primeros. Ni el objetivo, ni el tamaño del activo, ni su cobertura posterior intervienen.

| Origen | Vista | Papel | Ajuste | Validación | Calibración | Evaluación | Activos elegibles | Filas de ajuste |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | `fold-014` | Desarrollo | 2000 a mar. 2018 | abr. a sep. 2018 | oct. a dic. 2018 | 2019 | 3.474 | 2.819.756 |
| 2020 | `fold-015` | Desarrollo | 2000 a mar. 2019 | abr. a sep. 2019 | oct. a dic. 2019 | 2020 | 3.728 | 2.837.753 |
| 2021 | `fold-016` | Desarrollo | 2000 a mar. 2020 | abr. a sep. 2020 | oct. a dic. 2020 | 2021 | 3.973 | 2.892.903 |
| 2022 | `fold-017` | Desarrollo | 2000 a mar. 2021 | abr. a sep. 2021 | oct. a dic. 2021 | 2022 | 4.162 | 3.021.675 |
| 2023 | `fold-018` | Comprobación posterior | 2000 a mar. 2022 | abr. a sep. 2022 | oct. a dic. 2022 | 2023 | 4.174 | 3.274.785 |
| 2017 | `fold-012` | Año añadido | 2000 a mar. 2016 | abr. a sep. 2016 | oct. a dic. 2016 | 2017 | 3.158 | 2.625.627 |
| 2016 | `fold-011` | Año añadido | 2000 a mar. 2015 | abr. a sep. 2015 | oct. a dic. 2015 | 2016 | 2.950 | 2.564.272 |
| 2015 | `fold-010` | Año añadido | 2000 a mar. 2014 | abr. a sep. 2014 | oct. a dic. 2014 | 2015 | 2.753 | 2.436.887 |

El ajuste recorre toda la historia disponible de cada ventana, desde 2000 hasta el corte de su validación. Cada evaluación son unas 250 sesiones y unas 256.000 predicciones por semilla.

Los orígenes 2019 a 2022 son de desarrollo. La declaración llama `confirmation` a 2023, pero ese año ya se utilizó en estudios anteriores del proyecto. Aquí es una comprobación posterior con los contrastes fijados, no una confirmación independiente ni un test sin consultar. Los tres años añadidos amplían la evidencia con ventanas anteriores y se ejecutan en una segunda etapa, cuando la primera ha recorrido todos sus niveles. Si el plazo se agota, lo que queda sin ejecutar son esos años en los niveles menos prioritarios.

El origen 2018 (`fold-013`) queda fuera de la campaña porque es la ventana del piloto que eligió la receta. Los años anteriores a 2015 quedan fuera por presupuesto. La declaración no admite orígenes posteriores a 2023.

Las semillas son las tres del protocolo: 42, 43 y 44.

## Receta común

Todos los modelos neuronales comparten la receta [`comparison-campaign-core-v1`](../../configs/titans/comparison-campaign-core-v1.json). Parte de la receta histórica con la disposición contigua de la atención: 64 unidades ocultas, una capa, cuatro tokens persistentes, cabeza de cuantiles con pérdida pinball, truncamiento de 8 sesiones, bloques de 1.024 filas, AdamW con ritmo de aprendizaje 0,001 sin decaimiento de pesos, recorte del gradiente a norma 1 y FP32 estricto. Añade una sola pieza, la escala de aprendizaje.

### Escala de aprendizaje

El diagnóstico de #836 encontró tres problemas de escala que afectaban por igual a todas las familias neuronales. La edición entrega los fundamentales y los indicadores macro como niveles en logaritmo con signo, la ventana de precios anclada a su primer cierre y una etiqueta residual del orden de una centésima. Con esas escalas un predictor recién construido emitía con una amplitud varias veces mayor que la etiqueta, una parte de las columnas macro del año evaluado quedaba fuera del rango visto al ajustar y el optimizador gastaba sus primeras épocas en reducir la amplitud de la salida.

[`learning_scale`](../../src/mars_titan/models/learning_scale.py) declara tres piezas, todas estimadas solo con filas del tramo de ajuste de cada ventana:

- La ventana de precios se reescribe como rendimientos por sesión.
- Cada columna de entrada se centra y se escala con su mediana y la semidistancia entre sus cuantiles 16 y 84, con un suelo para las columnas casi constantes, y se recorta a cinco escalas. El recorte acota cualquier deriva posterior al ajuste.
- La salida de la cabeza se multiplica por la escala de las etiquetas del ajuste y la mediana empieza en cero, de modo que el modelo recién construido emite la predicción cero y su intervalo inicial coincide con el de las etiquetas.

Las estadísticas se calculan con una pasada de lectura sobre el índice de ajuste ([`learning_statistics`](../../src/mars_titan/training/learning_statistics.py)), se guardan junto a ese índice y su huella entra en la identidad de cada ajuste. Todos los modelos y semillas de una ventana leen el mismo archivo. Ninguna fila de validación, calibración o evaluación interviene. Sin declarar la escala, la configuración y el estado del predictor conservan su formato y cálculo. Las huellas de ejecución incluyen el código y cambian con esta revisión. Las ejecuciones históricas se recuperan con su versión de código correspondiente.

### Referencias recurrentes y DLinear

GRU, LSTM, RNN y DLinear se ajustan dentro del mismo predictor que el Transformer compacto. La opción `reference_encoder` sustituye solo el codificador de la ventana de precios. La fusión de modalidades, la cabeza, la pérdida, la escala de aprendizaje, el recorrido cronológico y la regla de parada son los mismos. Así la diferencia entre dos referencias es su codificador y no su canalización.

## Parada y selección

| Ejecutor | Épocas | Estado inicial | Regla |
| --- | --- | --- | --- |
| Ajuste desde cero | 8 | Medido y no emitido | Época con menor MAE por sesión de validación |
| Lector episódico | 4 | El del padre, que puede emitirse | Época con menor MAE por sesión de validación, o el padre si ninguna lo mejora |

Todos los ajustes recorren su presupuesto completo, de modo que los controles emparejados conservan el mismo número de actualizaciones. Cada época hace una actualización cada 8 sesiones, 528 por época en el origen 2018 del piloto. El número depende de las sesiones de cada origen.

La regla del protocolo exigía una mejora mínima de 0,00001 y permitía emitir el estado inicial. Con la mediana inicial en cero, el estado inicial de un ajuste es la predicción cero. En la ventana del piloto su MAE de validación es 0,010768 y las mejoras alcanzables son del orden de 0,000006, menores que la mejora mínima. Con aquella regla casi todos los ajustes habrían emitido su estado inicial y la campaña habría comparado la predicción cero consigo misma, como ya ocurrió en 334 de los 528 ajustes de posentrenamiento que recoge la memoria. Por eso los ajustes desde cero usan mejora mínima cero y miden su estado inicial sin emitirlo: la época emitida es siempre un estado ajustado, y la predicción cero se compara aparte, con las mismas filas.

Un lector continúa a un padre ya ajustado. Su estado inicial es ese padre y sigue siendo elegible, como exige la regla del proyecto para las continuaciones.

## Modelos

### Cola `references`

| Modelo | Ejecutor | Qué es |
| --- | --- | --- |
| `transformer` | Ajuste | Transformer compacto sin memoria neuronal (`transformer_direct`) |
| `gru`, `lstm`, `rnn`, `dlinear` | Ajuste | El mismo predictor con otro codificador de precios |
| `ridge` | Ridge | Regresión Ridge con todas las filas de ajuste, estandarizada con sus momentos. Alphas 0,1, 1, 10, 1.000, 100.000 y 10.000.000, elegido con el MAE por sesión de validación |
| `momentum_reversal` | Regla | Regla sin aprendizaje sobre la ventana de precios |
| `lightgbm` | Tabular | LightGBM con los parámetros de Qlib sobre las mismas entradas, 12 meses de ajuste y el residuo normalizado por sesión como etiqueta |

### Cola `mars_titan`

| Nivel | Modelo | Qué es |
| --- | --- | --- |
| `core` | `titans_mac` | Titans-MAC con la memoria neuronal activa (`mac_online`) |
| `core` | `mars_titan_m0` | Lector sin banco episódico, con el mismo ajuste |
| `core` | `mars_titan_m1` | Banco con episodios maduros elegidos al azar |
| `core` | `mars_titan_m2` | Cupos de 50, 25 y 25 % entre uniforme, mayor error y más recientes |
| `core` | `mars_titan_m3` | Como M2, con error, anomalía y relevancia en el índice selectivo |
| `controls` | `titans_mac_disabled` | Titans-MAC sin lectura de memoria |
| `controls` | `titans_mac_frozen` | Titans-MAC con la memoria congelada |
| `networks` | `area_network` | Red de 19 áreas con puente compartido |
| `networks` | `area_network_membership_only` | Ablación: solo la pertenencia del catálogo |
| `networks` | `area_network_hidden_only` | Ablación: 23 grupos ocultos sin catálogo |
| `networks` | `asset_groups` | Grupos de activos calculados con la ventana pasada |
| `networks` | `asset_groups_shuffled` | Control con las filas barajadas |
| `networks` | `area_network_asset_groups` | Red de áreas con grupos de activos |
| `networks` | `cross_asset_attention` | Atención entre los activos de la sesión |
| `networks` | `cross_asset_attention_self` | Control con la misma capa y atención solo al propio activo |
| `refinements` | `mars_titan_m3_k2`, `mars_titan_m3_k4` | M3 con 2 y 4 refinamientos |
| `bridge` | `mars_titan_m3_bridge_self`, `_ring`, `_complete` | M3 con el puente de evidencia entre tres vistas y sus tres topologías |

Los modelos de un mismo ejecutor comparten receta, presupuesto y regla. Ninguno recibe más épocas que su control.

## Contrastes declarados

| Familia | Variante | Control |
| --- | --- | --- |
| Referencias | `ridge`, `rnn`, `lstm`, `gru`, `dlinear`, `transformer` | Predicción cero |
| Memoria neuronal | `titans_mac` | `transformer`, `titans_mac_disabled` y `titans_mac_frozen` |
| Guardar episodios | `mars_titan_m1` | `mars_titan_m0` |
| Elegir episodios | `mars_titan_m2` y `mars_titan_m3` | `mars_titan_m1` y, para M3, `mars_titan_m2` |
| MARS-TITAN | `mars_titan_m3` | `titans_mac`, `transformer` y la predicción cero |
| Refinamientos | `mars_titan_m3_k2`, `mars_titan_m3_k4` | `mars_titan_m3` |
| Puente de evidencia | `mars_titan_m3_bridge_ring`, `_complete` | `mars_titan_m3_bridge_self` y `mars_titan_m3` |
| Red de áreas | `area_network` | `titans_mac`, `area_network_membership_only` y `area_network_hidden_only` |
| Grupos de activos | `asset_groups` | `asset_groups_shuffled`, y este frente a `titans_mac` |
| Áreas con grupos | `area_network_asset_groups` | `titans_mac`, `area_network` y `asset_groups` |
| Atención entre activos | `cross_asset_attention` | `cross_asset_attention_self`, y este frente a `titans_mac` |

Los contrastes principales son los que nombra el objetivo general: `mars_titan_m3` y `area_network_asset_groups` frente a `titans_mac` y frente a la predicción cero. Los demás explican de dónde viene una diferencia si aparece.

## Análisis declarado

- **Métrica principal.** MAE residual por sesión en el año de evaluación de cada origen. Primero se promedian los errores absolutos de cada sesión y después las sesiones. La predicción cero se calcula con las mismas filas.
- **Emparejamiento.** Cada contraste se calcula sesión a sesión, con la media de las tres semillas de cada modelo, y se resume por origen y en conjunto.
- **Incertidumbre.** Bootstrap circular por bloques de 16 días, con 8 y 32 días como sensibilidad, 2.000 réplicas y confianza del 95 %. Cada familia se corrige con Romano-Wolf y se informa además Holm.
- **Papeles.** Los orígenes 2019 a 2022 describen y, si hay que elegir entre variantes, eligen. En esta campaña, 2023 se evalúa con los contrastes ya declarados. Su reutilización anterior impide tratarlo como un test independiente. Los años añadidos se informan aparte como evidencia adicional.
- **Medidas secundarias.** IC por sesión, cobertura de los intervalos de la cabeza de cuantiles y MAE tras calibrar la escala de la predicción con la partición de calibración. La regla y LightGBM predicen un orden y no una magnitud, así que solo se comparan por IC y por ese MAE calibrado.
- **Sensibilidad.** La regla del protocolo, con el estado inicial elegible, se reconstruye con la historia de validación de cada ajuste y se informa junto a la regla de la campaña.

La lectura calibrada aplica λ·predicción, con λ en [0, 1] elegido exclusivamente en la calibración anterior. Se obtiene mediante la mediana ponderada que minimiza el MAE por sesión, implementada en [`scale_diagnostics.market_shrinkage`](../../src/mars_titan/evaluation/scale_diagnostics.py). Se informa junto al error bruto y conserva el mismo emparejamiento, los intervalos por bloques y la corrección de comparaciones. El IC de Pearson y el IC de rangos (Spearman) son medidas distintas y se identifican por su nombre.

La orden `results` resume esas medidas de los trabajos terminados. Cada trabajo terminado pesa lo mismo, de modo que una tabla parcial puede contener cantidades distintas de semillas y orígenes entre modelos. El análisis con bootstrap se ejecuta al cerrar la campaña sobre las predicciones guardadas de los dos equipos.

## Piloto acotado

El piloto utiliza el origen 2018, 1.024 activos de EE. UU. y la semilla 42. El ajuste contiene 2.726.802 etiquetas y recorre 2.791.261 observaciones por época. La validación tiene 125 sesiones, la calibración 62 y la evaluación 250, con 255.750 etiquetas evaluadas. El MAE de cero en evaluación es 0,01193174797. Son medidas de desarrollo de una ventana y una semilla, no resultados de la campaña.

La regla fijada antes de comparar recetas elige el menor MAE de validación de la mejor época. Una diferencia menor de 0,000001 favorece AdamW constante con tasa 0,001. La receta elegida fue esa misma. Su mejor puntuación llegó en la quinta época, por lo que se fija un presupuesto de ocho épocas para la campaña. Los lectores conservan cuatro.

| Receta | Épocas | Mejor época | MAE de validación | MAE bruto frente a cero en evaluación | MAE calibrado frente a cero | IC Pearson | IC Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW 0,001 con escala | 5 | 5 | 0.010757877 | -0.0446 % | -0.0504 % | +0.03518 | +0.01995 |
| Schedule-free | 5 | 5 | 0.010764530 | -0.0329 % | -0.0485 % | +0.01902 | +0.01904 |
| Media exponencial de pesos | 5 | 5 | 0.010762764 | -0.0475 % | -0.0644 % | +0.02527 | +0.01678 |
| AdamW 0,0001 con escala | 5 | 5 | 0.010766513 | -0.0180 % | -0.0297 % | +0.01353 | +0.01695 |
| Receta anterior | 2 | 2 | 0.011383365 | +2.9840 % | -0.0082 % | -0.00768 | +0.00802 |

La media exponencial tiene menor error observado en evaluación que la receta elegida. Eso no cambia la selección, que solo usa validación. La receta anterior recibió dos épocas, por lo que su fila no es un contraste con el mismo presupuesto que las otras. A las dos épocas, el MAE de validación con las escalas nuevas era 0,010763843 frente a 0,011383365 de la receta anterior.

Las cuatro recetas con escala reducen ligeramente el error bruto frente a cero en este origen. Para la receta elegida, la diferencia relativa es −0,0446 % y pasa a −0,0504 % al calibrar. Estas diferencias no establecen superioridad fuera del piloto. No se han calculado aquí los intervalos por bloques ni la corrección de comparaciones de la campaña.

| Recorrido con 12 meses y una época | Ajuste (s) | Evaluaciones (s) | Total (s) | Pico asignado por PyTorch (MiB) |
| --- | ---: | ---: | ---: | ---: |
| Transformer compacto | 51.0 | 206.2 | 383.8 | 841.8 |
| Titans-MAC | 59.3 | 185.7 | 301.3 | 2420.9 |
| Red de 19 áreas | 52.5 | 180.2 | 291.8 | 3387.4 |
| Grupos de activos | 58.0 | 205.1 | 320.6 | 2421.7 |
| Lector M0 | 55.3 | 231.3 | 327.2 | 277.6 |
| Lector M1 | 88.8 | 330.0 | 460.1 | 283.9 |
| Lector M3 | 83.5 | 332.3 | 500.1 | 283.8 |
| Lector M3 con K = 2 | 92.8 | 348.6 | 536.3 | 284.1 |

Los tiempos de evaluación agrupan la validación inicial, la de la época y las emisiones finales de validación, calibración y evaluación. Cada fila corresponde a una ejecución. No son percentiles de latencia ni una medida de energía. La memoria de la tabla es la asignada por PyTorch, que no incluye toda la reserva del controlador y sus bibliotecas.

La receta elegida con toda la historia tardó 3.132,8 s, de los que 2.338,6 s fueron de ajuste. Coincidió al principio con el ensayo inicial y construyó el índice de ajuste. Las otras tres recetas con cinco épocas tardaron entre 2.105,7 y 2.166,9 s y reutilizaron el índice. Esta diferencia de condiciones impide interpretar sus tiempos como una aceleración atribuible al optimizador.

En los lectores de una época, M0, M1 y M3 dan errores muy próximos. La mejora frente a su padre también aparece sin banco, por lo que esta pasada no permite atribuirla a la recuperación episódica. Sus factores de calibración rondan 0,145 a 0,148. En Titans-MAC tras una época, el factor es 0,0867. Con toda la historia y cinco épocas, la receta elegida necesita 0,6358. El IC de rangos de la primera pasada de Titans-MAC fue 0,01538 y su IC de Pearson 0,00880.

Los primeros recorridos de Transformer, áreas y grupos permitían elegir el estado inicial y emitieron cero. Sus costes se conservan, pero no se usan para atribuir aprendizaje a esa emisión. La campaña registra el estado inicial de los ajustes por separado y exige seleccionar una época ajustada.

Ridge completó sus seis alphas en 812,4 s, con un pico residente de 2,73 GiB y 113,9 MiB asignados por PyTorch. La validación eligió 10.000.000. Su MAE de evaluación fue 0,01193983224, todavía por encima de cero. La regla de momentum y reversión tardó 50,6 s. Su IC de Pearson fue 0,00741 y la calibración eligió λ = 0.

Las comprobaciones del motor ejecutaron los 28 modelos con 64 activos y una época. El ensayo trazado posterior ejecutó las ocho referencias y los cinco modelos del núcleo de MARS-TITAN. Los 195 archivos relevantes abiertos están entre los 710 archivos de ese paquete. El recuento corresponde a dependencias de datos y bibliotecas. El código y las recetas de la campaña tienen su propia identidad en Git.

El control LightGBM terminó las tres semillas y seleccionó 30 rondas. Su IC medio de Pearson en evaluación fue 0,02279 y el de Spearman 0,01162. La lectura calibrada tuvo un 0,00068 % más MAE que cero. Su última ejecución tardó 1.334,2 s, además de los intentos previos registrados, y alcanzó 11,25 GiB de memoria residente. La campaña divide ahora su ajuste entre procesos para no acumular las estructuras de varias semillas.

El lector M3 con puente `ring` tardó 518,3 s, con 89,3 s de ajuste y 327,3 s de evaluaciones. Su pico asignado por PyTorch fue 288,7 MiB. Tuvo un 0,3834 % más MAE bruto que cero y un 0,0124 % menos tras calibrar. Esta única pasada no demuestra una ventaja del puente sobre M3. El presupuesto de trabajo de los lectores se fijó en 256 MiB después de comprobar el lote y el banco completos.

## Dimensionado

Las proyecciones se calculan con las medidas de este portátil y los recuentos de cada origen. La [evidencia de dimensionado](../../reports/engineering/comparison-campaign-20261113/budget.json) conserva sus coeficientes. No se ha medido la velocidad de la RTX 5060 Laptop del segundo equipo.

| Cola | Primera etapa, 2019 a 2023 | Años añadidos, 2015 a 2017 | Total proyectado |
| --- | ---: | ---: | ---: |
| Referencias | 78 h | 41 h | Unas 120 h |
| MARS-TITAN, escala por observaciones | 265 h | 141 h | Unas 406 h |
| MARS-TITAN, escala por etiquetas | 340 h | 179 h | Unas 519 h |

Los dos escenarios de MARS-TITAN reflejan la incertidumbre al extrapolar el lector desde doce meses a toda la historia. El primero escala su coste de ajuste por observaciones y el segundo por etiquetas. No son intervalos de confianza ni cotas garantizadas. M2 se aproxima con M1, K = 4 con un factor de 1,2 sobre M3 y las tres topologías de puente con la medida de `ring`. El coste de los ajustes neuronales se aproxima con la mediana de las recetas de cinco épocas, ampliada a ocho y a las filas de cada origen. Cada índice de ajuste se construye una sola vez, con unos cinco minutos adicionales por origen en el piloto.

Entre el 13 de noviembre a las 17:00 y el comienzo del 5 de diciembre hay 511 horas. Si se comienza a las 21:00 hay 507. La primera etapa tiene prioridad sobre los años añadidos. En el escenario más lento no cabe la segunda etapa completa en ese plazo. La cola pide una parada al llegar a `2026-12-04T23:00:00+00:00`, equivalente a las 00:00 del 5 de diciembre en Madrid, y conserva los trabajos incompletos en su siguiente barrera. No se incluyen como terminados ni se rellenan sus resultados.

El paquete de datos ocupa unos 20,1 GiB. Las salidas se presupuestan en 5 GiB para `references` y 14 GiB para `mars_titan`, con un margen mínimo libre de 1 GiB durante la ejecución. El diagnóstico comprueba que haya al menos 12 GiB de memoria disponible por trabajo de referencias y 6 GiB por trabajo de MARS-TITAN. La cola usa un solo trabajador por equipo. Si el segundo equipo va más lento, la guía permite ceder niveles completos al portátil una vez terminadas sus referencias y siempre que disponga de espacio.


## Posentrenamiento y refuerzo

La campaña no incluye posentrenamiento ni aprendizaje por refuerzo. La pregunta de la memoria es predictiva y antes hay que saber si algún modelo mejora a la predicción cero con señal propia. Los entornos de refuerzo consumen predicciones fuera de muestra, así que su lugar natural es después de esta campaña, con las predicciones que deja guardadas y con un criterio financiero validado propio. El posentrenamiento de la entrega anterior se quedó con el estado inicial en 334 de sus 528 ajustes, y repetirlo sin una señal previa gastaría el plazo en comparar estados iguales.

## Desviaciones y límites

- **Árboles potenciados.** La memoria presenta XGBoost ajustado con toda la población. Repetirlo con la historia completa necesita una matriz en disco mayor que el espacio libre del portátil. Lo sustituye LightGBM con las mismas entradas y 12 meses de ajuste, el control declarado en la criba de variantes, que se compara por IC.
- **Ridge.** Se ajusta con todas las filas y sin el recorte de la escala de aprendizaje. La suma de productos de su ajuste no se divide por el número de filas, así que una rejilla de 0,1 a 10 regulariza poco cuando hay millones de observaciones. La rejilla ampliada conserva esos valores y añade 1.000, 100.000 y 10.000.000. Solo la validación elige el alpha. DLinear es la referencia lineal que comparte la escala de los modelos neuronales.
- **Capacidad del banco.** El banco episódico guarda como mucho 1.024 episodios, un límite del enlace nativo compilado. Con 1.024 activos por sesión equivale a los episodios de una sesión. Ampliarlo exige recompilar el enlace y repetir su verificación, y queda fuera de esta campaña.
- **Un mercado.** Solo EE. UU. China no entra por presupuesto.
- **Tres semillas.** Son las del protocolo. La incertidumbre se apoya sobre todo en las sesiones.
- **Dos GPU distintas.** Cada modelo se ajusta entero en un equipo. No se afirma reproducibilidad bit a bit entre equipos.
- **Pausas térmicas.** El portátil congela los trabajos por encima de 92 °C. Alargan la campaña sin alterar sus resultados.

## Motor y comprobaciones

El [motor](../../src/mars_titan/training/comparison_campaign.py) convierte la declaración en una cola ordenada y ejecuta cada trabajo en un proceso propio con los ejecutores ya existentes. No ajusta nada por sí mismo. Sus piezas:

- Permiso propio ligado a la declaración, al commit y a los cambios locales del código, incluidos los archivos nuevos. Una ejecución con otra identidad necesita otro estado si ya hay trabajos iniciados.
- Comprobación inicial de los modelos que no necesitan un padre ajustado, protegida por el mismo cerrojo que sus trabajos. Los lectores se comprueban cuando su padre está disponible.
- Reanudación con la misma orden, reintentos acotados y guarda de disco que detiene la cola sin perder trabajos.
- Paquete de datos con huellas para el segundo equipo ([`comparison_data`](../../src/mars_titan/training/comparison_data.py)) y diagnóstico previo de cada equipo.

La [verificación local](../../reports/engineering/comparison-campaign-20261113/README.md) recoge las 374 pruebas de la última pasada, las 32 comprobaciones CUDA, las 41 mutaciones detectadas y los límites de cobertura. Las pasadas anteriores y sus omisiones se conservan por separado. El segundo equipo aún debe comprobar sus rutas, bibliotecas, memoria y GPU con `doctor`.
