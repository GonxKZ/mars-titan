# Representación geométrica de 64 velas

`candle_geometry_linear64_v1` convierte los OHLC de una ventana en 512 valores FP32, sin
dibujar una imagen ni utilizar parámetros aprendidos. Es una representación experimental
desactivada por defecto. Cambia los vectores y tiene una identidad distinta de ResNet18 y del
control de tinta. No modifica la edición v3.1 ni demuestra una mejora predictiva.

La pregunta de [#665](https://github.com/GonxKZ/mars-titan/issues/665) es si conservar la
geometría antes de redondear a píxeles resulta útil frente a las representaciones visuales.
Es distinta de [#499](https://github.com/GonxKZ/mars-titan/issues/499), que estudia reducir el
coste de la misma imagen y de sus vectores sin cambiarlos. La procedencia del codificador
continúa formando parte de [#10](https://github.com/GonxKZ/mars-titan/issues/10).

## Fórmula y límites

Se compararon dos alternativas. La lineal conserva la escala vertical del gráfico existente.
La logarítmica expresaría los precios respecto al primer cierre observado y representaría
cambios proporcionales. Solo se implementa la lineal para no mezclar el efecto del rasterizado
con otro cambio de escala. El control de tinta fija ya existe en
`data/pretraining_free_encoders.py` y se reutiliza.

Sean M el máximo de los máximos observados y m el mínimo de los mínimos. Cada precio P se
representa como

\[
z(P)=\frac{P/M-m/M}{\max(1-m/M,10^{-9})}.
\]

Dividir antes de restar evita formar un rango absoluto que pierda precisión por
subdesbordamiento. La aritmética usa FP64 y redondea una sola vez a FP32. Por cada sesión se
guardan apertura, máximo, mínimo, cierre, cuerpo firmado, mecha superior, mecha inferior y
presencia:

\[
[z(O),z(H),z(L),z(C),z(C)-z(O),
z(H)-\max(z(O),z(C)),\min(z(O),z(C))-z(L),p].
\]

El orden es temporal, con ocho valores consecutivos por sesión. Son 64 × 8 = 512 componentes,
2.048 bytes por vector, igual dimensión que el vector visual actual. Los significados son
distintos. Los tres canales derivados son redundantes y su utilidad debe contrastarse.

Con todos los OHLC idénticos, los siete valores geométricos son cero y la presencia vale uno.
Una posición ausente contiene ocho ceros positivos. La máscara procede de `present_slots`
y del contrato de calendario declarado por la edición, no se estima a partir de un precio cero. El último
instante debe estar observado y la ventana necesita al menos dos observaciones.

Se rechazan precios no positivos, NaN, infinitos, formas incorrectas y desorden OHLC que
supere la tolerancia de la auditoría. Los conflictos pequeños admitidos se conservan. Una
mecha puede resultar negativa por ese redondeo y no se recorta para ocultarlo.
Si máximos y mínimos coinciden pero apertura o cierre difieren dentro de la tolerancia,
se conserva esa diferencia dividida por el suelo relativo de 10⁻⁹. Por eso el rango declarado
nulo no implica un vector nulo cuando existen esos conflictos. La normalización puede
amplificar el redondeo en ventanas casi planas y ese efecto forma parte del contraste pendiente.
Solo se convierten e inspeccionan como máximo las 64 filas observadas necesarias. Cambiar
precios anteriores a la ventana o posteriores al cierre elegido no cambia el vector.

## Identidad e integración

[candle_geometry.py](../../src/mars_titan/data/candle_geometry.py) contiene la fórmula, la
huella y `GeometryEncoders`. Esta composición delega el texto en el codificador ya elegido.
Conserva sus contratos y mantiene la limitación histórica del texto, si la tiene. La
identidad de la imagen declara CPU, aritmética, dimensiones, canales, versión de NumPy y
huellas del código de geometría y de validación de precios.

La huella identifica el nombre del algoritmo, la máscara de 64 posiciones y los OHLC
observados en FP64 little endian y orden temporal. No es el hash de un PNG. Se conserva el
nombre de columna `chart_hash`, pero su definición queda fijada por `image_policy`. El
codificador completo forma parte de la clave de caché, por lo que no se mezclan entradas de
distintas representaciones. Dos ventanas que producen los mismos píxeles pueden tener
geometrías distintas y necesitan huellas distintas.

La materialización existente recibe OHLC y máscara y calcula directamente el vector cuando
se pide esta variante. Las condiciones de admisión, disponibilidad, cohortes, contexto macro
y orden de las filas no cambian. La configuración anterior sigue dibujando los mismos PNG.
Una edición existente con otra identidad rechaza la nueva configuración.

`encode_corpus` admite `chart_representation="candle_geometry_linear64_v1"`. En la CLI se
activa con `--chart-representation candle_geometry_linear64_v1`, conservando el valor de
`--encoders` de su control. La ruta actual requiere una pasada directa y rechaza recogida de
PNG pendientes, codificación diferida, reutilización exclusiva, herencia separada de textos
y sustitución de una edición anterior. Los vectores de una ejecución directa pueden seguir
usando su caché y la recuperación normal por activo.

Con `--encoders frozen`, la composición reutiliza `FrozenEncoders`, que carga también su
ResNet18 aunque la nueva ruta no lo invoque. Ese coste de preparación se conserva y debe
contabilizarse. Con `--encoders pretraining_free`, tanto la composición como su control de
tinta se ejecutan en CPU. No se atribuye a la imagen una diferencia que provenga de cambiar
también el codificador de noticias.

Para una edición ya codificada se utiliza
[candle_geometry_edition.py](../../src/mars_titan/data/candle_geometry_edition.py):

```bash
uv run --no-sync python -m mars_titan.data.candle_geometry_edition \
  --source data/derived/encoded-history-v3.1 \
  --output data/derived/encoded-history-geometry-v1 --batch-rows 128
```

Esta ruta copia todas las filas y columnas de la edición confirmada y sustituye únicamente
`charts` y `chart_hash`. No instancia ningún codificador, no abre las cachés de noticias y no
ejecuta el contrato `text_carry`. Sus metadatos siguen identificando la procedencia original
del texto. La configuración, los recibos y la representación visual reciben identidades nuevas.
Los factores empresariales se copian sin cambiar sus bytes.

El texto sigue ocupando 384 valores FP32 y las velas 512. ResNet18 sin la clasificación final
tiene 11.176.512 parámetros, que ocupan 44.706.048 bytes en FP32, más sus buffers. Es el tamaño
de parámetros inspeccionado en CPU, no una medida del ahorro de RAM o VRAM del proceso.
La derivación evita cargarlo, pero sigue leyendo y comprimiendo las columnas conservadas al
escribir otro Parquet completo. No elimina ese coste de almacenamiento.

El origen debe ser un `materialized_corpus` histórico completo con contrato de ventanas v3.1.
Se verifican las huellas de configuración, manifiestos, muestras, precios y factores antes
y después. También se comprueban identidad, tamaño y fechas de modificación de esos archivos,
para detectar una reescritura aunque al terminar vuelvan a contener los mismos bytes.
`price_end_index`, sesión, hora de decisión y disponibilidad de precios deben corresponder
al mismo cierre del calendario. No se admiten decisiones desde 2024 ni huecos particulares
de un activo que no estén declarados como ausencias de todo el mercado.

La carpeta provisional `.<destino>.geometry-staging` permite reanudar activos confirmados.
Un recibo solo se guarda después de confirmar su Parquet. La recuperación verifica cada
archivo confirmado y rechaza una fuente, configuración o representación distinta. Un candado
evita dos escrituras sobre el mismo destino. El directorio final aparece mediante renombrado
atómico sin sustitución, tras el cotejo completo. No se borra ni sobrescribe una edición previa.
`comparison-manifest.json` conserva exactamente el manifiesto que se cotejó antes de añadir
el informe de derivación al manifiesto final.

El lote predeterminado tiene 128 filas y puede configurarse entre 1 y 512. Se limita a 16 MiB
decodificados. Cada grupo Parquet y la tabla de precios de un activo tienen un presupuesto
de 64 MiB, con un máximo de 200.000 filas por activo y 10.000 candidatos. Estos límites acotan
las entradas, no constituyen una cota de la RAM total de Python y Arrow. Se procesa un activo
cada vez. El informe separa lectura Parquet, preparación de precios, geometría y empaquetado,
escritura con confirmación, copia de factores, comparación y verificación final del origen.
`before_publication_seconds` termina antes de escribir el manifiesto final y renombrar el
directorio. El benchmark mide también la llamada completa, incluida esa publicación.

La opción `--selection` recibe un JSON con `assets`, una lista de objetos con `market` y
`symbol`. Rechaza duplicados e identidades ajenas al manifiesto fuente completo. Conserva
la solicitud, su hash, la selección ordenada y las huellas del manifiesto y configuración
originales. Las reglas de selección quedan como procedencia, sin acreditar por sí mismas
su validez científica. Cada activo conserva todas sus filas, sin filtrar fechas ni modalidades.

```bash
uv run --no-sync python -m mars_titan.data.candle_geometry_edition \
  --source data/derived/encoded-history-v3.1 \
  --output data/derived/geometry-study \
  --selection data/derived/study-selection.json --estimate-only --reserve-gib 16
```

Con una selección, la salida declara `scope=development_snapshot`, `cohort_complete=false`
y `training_ready=false`. `selection_complete=true` solo se publica al terminar todos sus
candidatos y cotejar las filas correspondientes contra el origen completo. La opción nunca
permite usar un origen incompleto. No cambia el estado de la edición histórica completa.

`--estimate-only` lee JSON y pies Parquet, sin decodificar muestras ni crear el destino.
Calcula un presupuesto con el mayor tamaño entre bytes del archivo y páginas sin comprimir,
añade los nuevos vectores completos, un 20 % y margen por grupos, activos y metadatos. Es una
estimación, no una cota certificada de compresión. La reserva predeterminada es 16 GiB en el
sistema de archivos del destino. Se comprueba antes de escribir y antes de cada activo. La
recuperación descuenta los activos ya confirmados. La estimación no sustituye la comprobación
posterior de hashes, índices, calendario y fechas. El benchmark de fixtures usa una reserva
explícita de 64 MiB para poder ejecutarse también en un directorio temporal pequeño.

## Comparación sobre las mismas filas

Se prepara un control y una edición geométrica en destinos nuevos, usando el mismo
manifiesto de preparación, codificador de texto, catálogo macro, factores, calendario y
resto de opciones. Solo cambia la opción de representación. No se añade un precio ni se
retira una fila por carecer de una modalidad opcional.

`benchmarks.candle_geometry compare` exige la misma preparación y contratos efectivos.
Comprueba las huellas de muestras antes y después de leer y compara lotes de 128 filas.
Todas las columnas salvo `charts` y `chart_hash` deben coincidir, incluidos sesión,
`price_end_index`, disponibilidad y máscaras. El límite es 16 MiB por lote decodificado y
64 MiB por grupo Parquet. La salida identifica ambos manifiestos y configuraciones.
La implementación de este cotejo se comparte con la derivación en el módulo de datos.

Este cotejo es específico del contraste de representación. `edition_comparison` se diseñó
para comparar v3 con v3.1 y presupone ventanas comunes con 64 filas observadas consecutivas.
No se utiliza aquí, porque ambas ediciones pueden conservar el mismo hueco de calendario.
La igualdad de columnas tampoco sustituye la prueba causal del extractor.

El contraste predictivo pendiente conserva MAE residual por sesión como criterio principal,
las mismas filas, cortes, parámetros del predictor y presupuesto de selección. Incluye el
control visual de ResNet, el de tinta fija y un control de reutilización de los mismos precios
para separar un efecto de duplicación. Debe informar parámetros totales, coste de preparación
y coste de predicción. Una representación más barata no acredita menor error.

## Benchmark y verificación

El benchmark admite fixtures técnicos reproducibles o un NPZ con `windows` de forma
`[n,64,4]` y `presence` booleana `[n,64]`. Las posiciones ausentes se rellenan con cero.
El archivo se limita a 16 MiB comprimidos y sin comprimir y a 256 ventanas. No acredita
por sí solo la procedencia temporal de esas entradas.

```bash
uv run --no-sync python -m benchmarks.candle_geometry measure \
  --device cpu --samples 64 --repeats 9 --warmup 2 \
  --output data/derived/candle-geometry/cpu-64.json
```

La opción `--device cuda` añade el control ResNet18 en `cuda:0`, con `GpuLease` y FP32
estricto. La CLI fija esa precisión antes de cargar los pesos. Falla si la GPU no está
disponible o está ocupada y no cambia a CPU. `--ordering-rtol` transmite la misma tolerancia
de orden a todas las representaciones y la registra, con cero como valor predeterminado.
Una edición auditada debe usar su tolerancia ya declarada. La medida no corrige los precios.
Los destinos existentes se rechazan antes de medir o comparar.

Cada variante usa las mismas ventanas. Se alterna el orden de medida, se hace una pasada
de referencia y después el calentamiento indicado. Se verifican los mismos bits entre
repeticiones de cada variante, sin exigir igualdad entre representaciones distintas.
Los tiempos incluyen dibujo cuando corresponde, representación y hash de entrada. Los
percentiles describen lotes completos. RAM y VRAM del proceso no se atribuyen a un módulo
aislado. No se miden energía, coste monetario ni calidad predictiva.

Las pruebas cubren fórmula, escala positiva, rango nulo, ausencias, extremos, tolerancia de
orden, hashes, identidad, texto delegado, ventanas futuras, comparación de filas, recuperación
y rechazo de opciones incompatibles. Las pruebas de integración conservan el hueco declarado
y todas sus filas admisibles. La variante no ejecuta pasos de optimizador.

La [comprobación del 18 de octubre](../../reports/engineering/candle-geometry-20261018/README.md)
conserva la primera comprobación de 164 pruebas CPU y la ampliación a 200 tras añadir la
derivación. La selección y el presupuesto elevan la batería a 217 pruebas y 23 mutaciones
dirigidas, con cobertura y CRAP registrados. También conserva los costes de lectura,
preparación y escritura de fixtures de 128 y 512 filas de la versión anterior a la selección.
La ampliación del informe incluye la derivación de 235.482 filas de 64 activos, otra de
943.780 filas de 256 activos y una medida de extracción con 256 ventanas reales de los
64 activos iniciales, incluidas ResNet y la reserva CUDA. No contiene una comparación
predictiva. La [revisión de las fuentes](../references/candle-and-negotiation-sources.md)
explica qué se tomó de los antecedentes y qué queda fuera del contraste.

Las vistas supervisadas emparejadas se preparan con `benchmarks.candle_geometry_views`.
El cotejo de la edición acredita que solo cambian `charts` y `chart_hash`. Se conservan
los mismos Parquet de etiquetas, sus hashes, configuración y fingerprints originales.
La representación nueva se identifica por separado y recibe su propio padre temporal.
Ambas vistas conservan `development_snapshot`, `cohort_complete=false` y `training_ready=false`.
No se transfiere a la geometría un checkpoint ajustado sobre ResNet.
