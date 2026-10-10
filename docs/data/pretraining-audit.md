# Procedencia de los codificadores congelados

## Configuración usada

La extracción mantiene MiniLM multilingüe para texto y ResNet18 para imágenes.
Son representaciones congeladas para medir y comparar modelos compactos. No se
entrenan estos codificadores ni se interpreta su uso como reproducción histórica
estricta del conocimiento disponible en cada fecha del mercado.

| Elemento | Decisión y evidencia |
| --- | --- |
| Texto | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, 384 dimensiones. |
| Tokenización | Se fijan `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` y `config.json` mediante SHA-256. También se identifica el tokenizador cargado en memoria. |
| Texto largo | Fragmentos de hasta 126 tokens de contenido, con tokens especiales explícitos. Se procesa el documento completo y se agregan los fragmentos ponderando por longitud. |
| Imagen | `resnet18.IMAGENET1K_V1`, pesos identificados por URL y SHA-256, 512 dimensiones después de retirar el clasificador. |
| Transformación visual | Gráfico regenerado de 224 × 224 píxeles, RGB y normalización fijada. No se recortan los extremos de la ventana. |
| Ejecución | `cuda:0`, FP32, modo evaluación y gradientes desactivados. Sin alternativa silenciosa en CPU. |

Los dos modelos se publicaron después de buena parte del periodo evaluado. La edición
congelada mantiene `historical_simulation = false`: representa cada noticia y cada gráfico
con un modelo que pudo leer textos posteriores a la fecha de la decisión. Las secciones
siguientes registran qué consta de sus corpus y sus fechas hasta donde se ha podido
verificar, qué conocimiento podrían contener y cómo se medirá si ese conocimiento cambia la
comparación.

## Corpus de preentrenamiento y fecha de los pesos

### MiniLM multilingüe

La fecha de los pesos se puede acotar con el historial del repositorio. El
`model.safetensors` de la revisión fijada se añadió el 27 de marzo de 2024 como conversión de
formato. Se comprobó que sus 200 tensores son idénticos, uno a uno, a los del
`pytorch_model.bin` que se subió en el [commit `e62509716f` del 23 de junio de 2021](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e62509716f15c5fd03a6fd3156a4bc5e43f83f26),
cuyo SHA-256 (`16cc9e54…ac59`) es también el del `pytorch_model.bin` de la revisión fijada. Su
`config.json` se guardó con Transformers 4.7.0. Ningún texto posterior al 23 de junio de 2021
pudo influir en estos pesos. La fecha de creación del 2 de marzo de 2022 que daban los
metadatos de la API no es la de la primera subida, porque el historial empieza el 2 de junio
de 2021.

El contenido del corpus solo consta en parte:

| Etapa | Qué consta | Fuente | Fecha verificable |
| --- | --- | --- | --- |
| Método | Destilación multilingüe: un estudiante aprende a reproducir los vectores de un profesor monolingüe sobre frases traducidas | [Reimers y Gurevych (2020)](https://arxiv.org/abs/2004.09813), EMNLP 2020 | Abril de 2020 |
| Profesor | `paraphrase-MiniLM-L12-v2`, entrenado con AllNLI, sentence-compression, SimpleWiki, altlex, msmarco-triplets, quora_duplicates, coco_captions, flickr30k_captions, yahoo_answers_title_question, S2ORC_citation_pairs, stackexchange_duplicate_questions y wiki-atomic-edits | [Datos de paráfrasis de Sentence Transformers](https://sbert.net/examples/training/paraphrases/README.html) | No consta la fecha de cada volcado |
| Datos paralelos | Frases traducidas de más de 50 idiomas | [Modelos preentrenados de Sentence Transformers](https://sbert.net/docs/sentence_transformer/pretrained_models.html) | No se publica la lista exacta de esta versión |
| Estudiante | 12 capas, 384 dimensiones y 250.037 piezas de vocabulario. Coinciden con [Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384), destilado de XLM-R Base con los mismos corpus de su profesor ([Wang et al., 2020](https://arxiv.org/abs/2002.10957)). La ficha del modelo no lo nombra, así que es una inferencia por la arquitectura | `config.json` local y configuración de Microsoft | No consta |
| Corpus de XLM-R | CC-100: CommonCrawl filtrado en 100 idiomas, un volcado para inglés y doce para el resto | [Conneau et al. (2020)](https://arxiv.org/abs/1911.02116) | Artículo enviado en noviembre de 2019 |

Ninguno de esos corpus contiene las etiquetas de este proyecto ni se usó para predecir
rendimientos. El modelo aprende a acercar frases con el mismo significado. Pero CommonCrawl,
Wikipedia, las preguntas y respuestas y los resúmenes científicos reúnen texto general de hasta
2019 o 2021 que describe lo ocurrido después de 2000, incluidas noticias financieras con
precios, quiebras, fusiones, crisis y la evolución de empresas concretas. La representación de
una noticia de 2007 sobre un banco puede quedar cerca de textos sobre su caída posterior. El
predictor solo podría aprovecharlo si esa geometría se relacionara con el rendimiento residual
futuro, y ese es el riesgo que mide la [sensibilidad declarada](#sensibilidad-declarada).

### ResNet18

Los pesos `IMAGENET1K_V1` de la [ficha de torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
son el clasificador de las 1.000 clases de ImageNet-1K de [He et al. (2016)](https://arxiv.org/abs/1512.03385),
reproducido por torchvision con una receta sencilla y un 69,758 % de acierto top-1.
ImageNet-1K es el conjunto de fotografías de objetos de la competición ILSVRC de 2012. No
contiene gráficos bursátiles ni precios, y los gráficos del proyecto se dibujan solo con los
precios de la ventana pasada. Un efecto de anticipación a través de las imágenes exigiría que
fotografías de objetos codificaran resultados de mercado, algo poco plausible que la
sensibilidad no separa del texto porque cambia los dos codificadores a la vez.

El rendimiento publicado en clasificación de imágenes no demuestra utilidad sobre gráficos
bursátiles. El uso y los derechos de los datos originales de preentrenamiento son distintos de
la licencia del código de MARS-TITAN.

## Identidad de caché

La huella del codificador incluye pesos, revisión, artefactos de tokenización,
configuración, transformación, precisión, dispositivo, código y versiones de
ejecución. La clave de texto añade contenido y regla de disponibilidad. La imagen
se identifica por los bytes del gráfico pasado que realmente se codifica.

Estas huellas se incorporan a las claves nuevas. Cambiar un artefacto invalida la
reutilización de su representación. No se sobrescriben ni se certifican
retroactivamente las cachés antiguas de las sondas de coste. Esos ensayos siguen
asociados a su configuración y revisión originales.

Los ejemplos almacenados conservan además la decisión temporal y la procedencia
de las modalidades. Compartir una representación idéntica no autoriza compartir
una muestra si cambia su disponibilidad o deja de ser admisible.

## Control sin preentrenamiento posterior

`data/pretraining_free_encoders.py` implementa el control con la misma interfaz que los
codificadores congelados y sin ningún parámetro aprendido. Conserva las cuatro modalidades, la
misma población admitida y los anchos de 384 y 512 valores, así que precios, fundamentales,
macro y el resto de la canalización no cambian.

| Modalidad | Regla fija | Biblioteca |
| --- | --- | --- |
| Texto | Normalización NFKC, minúsculas y n-gramas de 2 a 4 caracteres dentro de cada palabra. Cada n-grama suma ±1 en una de 384 posiciones según MurmurHash3 de 32 bits con semilla 0 ([hashing con signo](https://arxiv.org/abs/0902.2206)) y el vector se normaliza con L2. El texto se procesa completo | `HashingVectorizer` de scikit-learn, del extra `research` |
| Gráfico | El mismo PNG de 224 × 224 que recibe ResNet18 se reduce a la media de la tinta roja y verde (1 − canal / 255) en bloques de 14 × 14 píxeles: 16 × 16 bloques por canal y 512 valores. La tinta roja separa las velas alcistas de las bajistas y la verde marca cualquier vela | NumPy y Pillow |

Los n-gramas de caracteres funcionan igual con el inglés y con el chino, que no separa
palabras con espacios. El hashing no aprende vocabulario ni pesos del corpus, y el vector de
una entrada solo depende de ella, por eso la identidad declara `historical_simulation = true`.
La identidad registra las reglas, las versiones de NumPy, scikit-learn y Pillow, la huella del
código y la de los vectores de tres entradas fijas, de modo que un cambio de biblioteca que
alterara el resultado también cambia la identidad. Se calcula en CPU en FP64 y se redondea
una vez a FP32. `strict_fp32_spec` la acepta porque no interviene PyTorch ni TF32.

```bash
uv run --extra research python -m mars_titan.data.corpus_encoding \
  --encoders pretraining_free --prepared <edición preparada> --output <edición de control> ...
```

La opción rechaza las pasadas que reutilizan o completan vectores en GPU (`--reuse-only`,
`--collect`, `--encode-pending` y `--text-carry`), porque el control calcula todos sus
vectores en una sola pasada. La edición resultante tiene su propia identidad de codificador,
así que nunca comparte vectores ni caché con la congelada.

La propuesta anterior preveía fijar dimensión y normalización con entrenamiento y validación.
Se fijaron sin datos para conservar los anchos del codificador congelado, porque elegirlos con
la validación sería ajustar el control. También se descartó una red pequeña entrenada en
cada ventana para los gráficos: añadiría capacidad aprendida, que es justo lo que el control
debe quitar, y necesitaría entrenar. No se atribuirá toda diferencia al preentrenamiento,
porque también cambian capacidad y representación.

## Sensibilidad declarada

La [declaración](../../configs/encoders/pretraining-free-sensitivity.json) se fijó el 10 de
octubre de 2026, antes de calcular ninguna edición de control ni ajustar ningún brazo.
`evaluation/encoder_sensitivity.py` la valida y compara dos comparaciones walk-forward
publicadas que solo difieren en los codificadores: misma configuración, ámbito, ventanas,
semillas, filas por sesión y signos de los objetivos, y ediciones distintas.

Para cada brazo y mercado, Δ de una sesión es el MAE con el control menos el MAE con los
codificadores congelados, con las semillas promediadas sesión a sesión. Positivo indica que
los congelados ayudan. El corte es el 1 de julio de 2021, el primer mes completo después de
publicar los pesos de MiniLM, y los tramos tienen 30 meses:

| Estadístico | Cálculo |
| --- | --- |
| `effect` | Media de Δ en todas las sesiones evaluadas |
| `break` | Media de Δ de enero de 2019 a junio de 2021 menos la de julio de 2021 a diciembre de 2023 |
| `placebo_break` | Media de Δ de julio de 2016 a diciembre de 2018 menos la de enero de 2019 a junio de 2021 |

Si la ventaja de los congelados viniera de conocer el futuro, existiría antes del corte y
desaparecería después. Una ventaja que solo decae con el tiempo, o una mejora general del
control con más datos de ajuste, produce también un salto entre los dos tramos anteriores,
que el codificador pudo conocer por igual. La anticipación se considera sospechosa si `break`
y `break − placebo_break` quedan por encima de cero con intervalos simultáneos. En otro caso no
se sostiene. Con menos de 100 sesiones en algún tramo la decisión queda sin tomar. Los
intervalos usan el bootstrap circular por bloques de días de la comparación (bloques de 16
días, 2.000 réplicas, semilla 20261009 y 95 %), con una familia max-t por mercado sobre los
brazos y los dos estadísticos.

Se declaran la GRU de referencia y `titans_mac_online` en US y CN. Volver a ajustarlos con la
edición de control cuesta lo mismo que esos brazos en la campaña: 2 brazos por 3 semillas en
19 ventanas de US y 13 de CN, 192 trabajos de ventana y semilla. Es una partida separable del
presupuesto, que se decidirá en el resumen previo al entrenamiento. No mide cuál de los dos
codificadores lleva el efecto, ni separa un cambio de régimen que coincida con el corte, ni
el conocimiento del mundo que no altera el error.

```bash
uv run python -m mars_titan.evaluation.encoder_sensitivity \
  --declaration configs/encoders/pretraining-free-sensitivity.json \
  --frozen <comparación congelada> --control <comparación de control> --output <informe nuevo>
```

## Relación con la ablación de modalidades

La [ablación de modalidades en inferencia](../research/metrics.md#ablación-de-modalidades-en-inferencia)
([#414](https://github.com/GonxKZ/mars-titan/issues/414) y
[#423](https://github.com/GonxKZ/mars-titan/pull/423)) mide cuánto cambia el error de cada
brazo cuando las noticias o los fundamentales se leen como ausentes, con el mismo estado. Si
un brazo apenas depende de las noticias, el conocimiento posterior de MiniLM tampoco puede
mover mucho sus conclusiones. La sensibilidad responde a otra pregunta: si la parte que sí
aporta el texto podría venir del futuro. Las dos se leen juntas y ninguna sustituye a la
otra.

## Límite de la preparación

Comprobar huellas y repetibilidad no verifica el contenido de las noticias. La
incidencia de correspondencia entre título y cuerpo descrita en la [política de
noticias](news-policy.md) sigue condicionando la extracción definitiva. No se
vuelve a codificar el corpus ni se inicia entrenamiento confirmatorio con esta
incorporación.

## Comprobación de los codificadores congelados

La suite completa pasó con 364 pruebas, incluida una extracción real en CUDA.
Se repitieron dos textos sintéticos, uno de ellos de varios fragmentos, y un
gráfico sintético. En los tres casos la diferencia máxima fue cero, dentro de
las tolerancias fijadas de `rtol = atol = 10⁻⁶`. Las huellas de todos los pesos y
buffers fueron iguales antes y después. No se usaron etiquetas financieras.

El [recibo de la ejecución aislada](../../reports/data/encoder-provenance.json)
registra hashes y versiones. En ese proceso, la inicialización tardó 3,87 segundos
y la repetición conjunta de los dos textos y el gráfico, 17,81 milisegundos.
Solo hay una observación temporizada. No es una estimación de percentiles ni del
coste por artículo del corpus. El pico fue de 546.401.792 bytes asignados y
578.813.952 reservados en CUDA, con 2.584,48 MiB de RSS del proceso, incluida la
comprobación de los estados completos.

Las pruebas de caché comprueban los cuatro artefactos por separado. Una mutación
que sustituye su SHA-256 por un valor constante se detectó en los cuatro casos.
Las diez pruebas dirigidas, incluida CUDA, cubrieron 114 de 121 sentencias y
23 de 30 ramas del módulo, un 90,73 % combinado con coverage.py 7.16.1.
La función nueva de huellas tiene complejidad 2, cobertura de sentencias del
100 % y CRAP 2, según Radon 6.0.1. Son controles acotados, no una certificación
de que cualquier entrada producirá una representación útil.

```bash
MARS_ENCODER_INTEGRATION=1 HF_HUB_OFFLINE=1 uv run --locked \
  --extra cuda --extra encoders pytest \
  tests/data/test_encoder_provenance_integration.py -q -s
```

La comprobación requiere los pesos ya descargados. Sin la variable de activación
queda omitida de la suite habitual. Al activarla, la falta de CUDA produce un
error, no una ejecución alternativa en CPU.

## Comprobación del control

El [recibo del 10 de octubre de 2026](../../reports/engineering/pretraining-free-encoders-20261010/README.md)
codificó dos veces, con instancias distintas, 2.000 noticias reales de US y 2.000 de CN y unos
950 gráficos por mercado dibujados con `charts.chart_png`, sin modelos ni GPU. Los vectores
coinciden bit a bit entre instancias, son finitos y los de texto tienen norma 1. También
comprobó los pesos de MiniLM frente a los de junio de 2021 y midió el coste. Las pruebas
comparan el texto con un hashing calculado aparte con `murmurhash3_32` y la tinta con
imágenes de valor conocido. Las 57 pruebas detectan las 39 mutaciones dirigidas del mismo
recibo.

