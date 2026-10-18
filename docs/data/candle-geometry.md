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

Con rango nulo, los siete valores geométricos son cero y la presencia vale uno. Una posición
ausente contiene ocho ceros positivos. La máscara procede de `present_slots` y del contrato
de calendario declarado por la edición, no se estima a partir de un precio cero. El último
instante debe estar observado y la ventana necesita al menos dos observaciones.

Se rechazan precios no positivos, NaN, infinitos, formas incorrectas y desorden OHLC que
supere la tolerancia de la auditoría. Los conflictos pequeños admitidos se conservan. Una
mecha puede ser ligeramente negativa por ese redondeo y no se recorta para ocultarlo.
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
estricto. Falla si la GPU no está disponible o está ocupada y no cambia a CPU. Esa medición
queda preparada para su ejecución coordinada. No se ha ejecutado en esta tarea.

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
registra 164 pruebas CPU, cobertura, CRAP, nueve mutaciones y costes de preparación. No contiene
resultados CUDA ni una comparación predictiva.
