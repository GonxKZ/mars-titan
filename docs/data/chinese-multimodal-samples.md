# Muestras chinas con las cuatro modalidades

La primera edición codificada de Ping An contiene 169 muestras reales, entre
el 10 de marzo y el 29 de diciembre de 2023. Usa los seis [hechos revisados](china-provenance.md),
los precios y resúmenes de la preparación original y el [panel macro CN](china-macro-edition.md).
Todas las muestras tienen precios, noticias, fundamentales y gráficos, además
de los 140 indicadores. El [factor CSI 300](csi300-market-factor.md) permite
asignar 168 etiquetas. La última queda excluida porque su resultado cruza el corte
anual.

`prepare_chinese_samples` conserva la referencia al manifiesto de 892 candidatos
y selecciona el activo con hechos reconciliados. Publica una preparación
derivada, vectores y etiquetas en directorios separados. La selección se declara
`development_snapshot`, con `cohort_complete=false` y `training_ready=false`.
Esta edición comprueba el recorrido de un activo y no acredita cobertura completa
de China.

La representación contable mantiene tres conceptos CAS en CNY: activos, pasivos
y patrimonio con minoritarios. Cada muestra contiene los tres valores
transformados mediante logaritmo con signo, sus máscaras y las edades desde la
disponibilidad. Son nueve componentes. Un cero observado conserva su máscara.
No se convierten importes a USD ni se aplican fórmulas empresariales US-GAAP.
El cursor elige los saldos de 2022 y conserva la publicación de marzo de 2023
para los comparativos de 2021.

| Entrada | Representación usada por el lector |
| --- | --- |
| Precios | 64 sesiones consecutivas de OHLCV, reconstruidas desde un índice sobre el Parquet preparado. |
| Noticias | 384 componentes, agregando todos los eventos admitidos en las últimas cinco sesiones. |
| Gráficos | 512 componentes de gráficos regenerados con los precios anteriores a la decisión. |
| Fundamentales | Nueve componentes CAS/CNY con valores, máscaras y edades. |
| Macro | 420 componentes para 140 indicadores, máscaras y edades. |

La admisión macro verifica el catálogo, la fuente y las 387 decisiones completas
antes de cargar los modelos. Cualquier muestra fuera de esas decisiones se
excluye antes de codificar. Las disponibilidades se conservan en UTC y no pueden
superar la decisión correspondiente.

La edición real utiliza MiniLM multilingüe y ResNet18 congelados en `cuda:0` y
float32. Sus pesos, tokenizador, versiones y huella de código coinciden con los
de la edición estadounidense. Se utilizó la implementación fijada del codificador
con SHA256 `e45398cb8aa06997915fcd36e1ab11eb521599bf0f36a0128d11300b227fde5e`.
La fábrica inyectable conserva esa procedencia en cada recibo. Procesar texto
chino y obtener valores finitos no demuestra utilidad predictiva en ese mercado.

La [comprobación CUDA](../../reports/data/chinese-multimodal-samples-20261007.json)
ejecutó 255 codificaciones de texto y 169 de gráficos. Contrastó valores,
máscaras y edades contables, reconstruyó los gráficos de ambos extremos y recorrió
las 168 etiquetas mediante `CorpusDataset`. La reutilización no volvió a ejecutar
los codificadores y conservó contenido y fechas de modificación. Dos ejecuciones
CUDA produjeron los mismos bytes de muestras y etiquetas.

La codificación tardó 5,44 segundos y la reutilización 1,87. El proceso completo,
incluidos modelos y comprobaciones, tardó 15,46 segundos y alcanzó 1.990.584 KiB
de RAM. PyTorch registró un máximo de 570.390.528 bytes asignados y 612.368.384
reservados en CUDA. Son contadores del asignador de PyTorch, no una medida de
toda la memoria del controlador. No se midieron energía ni transferencias.

La primera comprobación detectó que la supervisión reescribía un manifiesto
idéntico. Tras corregirlo, la segunda terminó con salida cero. La campaña
estadounidense se pausó desde un estado recuperable y se reanudó al cerrar cada
ventana. Sus fuentes y su código científico permanecieron fijados.

Pasan 138 pruebas relacionadas. Se detectan cinco mutaciones del contrato CNY y
otras cinco de alcance, admisión, población, identidad y reutilización. El recibo
registra cobertura y CRAP por función. La complejidad máxima de las funciones
modificadas está en `encode_corpus`, con CCN 84 y CRAP 108,27. Estos diagnósticos
señalan límites del código y no demuestran ausencia de errores.

Las particiones anuales preliminares contienen cero muestras de entrenamiento
y 168 de validación, porque esta revisión contable solo admite fechas de 2023.
Antes de entrenar hacen falta más cobertura y unas ventanas temporales adecuadas,
con validación, calibración y evaluación separadas. La comparación conjunta
necesita una representación común que conserve los conceptos de cada moneda.
El candidato MARS-TITAN sigue sin implementarse ni entrenarse y el test de 2024
permanece cerrado.

La [ampliación de historia contable](chinese-fact-history.md) conserva esta primera
edición y añade publicaciones anteriores de Ping An y dos informes de Vanke.
Sus ediciones independientes contienen 721 muestras y 671 etiquetas, con filas
de entrenamiento de 2022. La comparación china aún necesita un corpus común y
ventanas temporales separadas.
