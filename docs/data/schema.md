# Contrato de archivos preparados

Los originales se leen desde `dataset/`. Los derivados no cambian su formato ni
sus etiquetas originales, se escriben en almacenamiento separado.

| Artefacto | Unidad | Identificación y control |
| --- | --- | --- |
| Inventario SQLite | Archivo original | Ruta en bytes, tamaño, SHA-256, mercado, modalidad y estado de inspección. |
| Precios Parquet | Activo y sesión | OHLCV, sesión local y disponibilidad UTC. Sesiones duplicadas o inválidas se excluyen. |
| Noticias Parquet | Texto y disponibilidad | Huella, fecha fuente, disponibilidad, URL, línea y evidencia de asociación. |
| Fundamentales Parquet | Concepto, periodo y presentación | Unidad, valor, `filed`, `accn`, disponibilidad y archivo de origen. |
| Archivo macro | Indicador, periodo y versión | Intervalos originales, unidades históricas, ajuste y hash del ZIP. |
| Panel macro | Indicador y decisión | Valor o ausencia, periodo, unidad histórica, disponibilidad y huellas de insumos. |
| Muestras Parquet | Activo y decisión | Índice de ventana, cuatro modalidades, macro y procedencia. |

El activo queda identificado por la partición y el símbolo del archivo original.
Su identidad lógica es `finmultitime:mercado:símbolo`. Es una identidad del archivo,
no un identificador jurídico que resuelva cambios históricos de símbolo.

El manifiesto de cada activo contiene hashes de fuentes y salidas, política de
calendario, versión de preparación, recuentos y auditorías. El manifiesto de muestras
añade codificadores, huella macro, contexto, tiempo y recursos. Un manifiesto no
sustituye la comprobación de los archivos a los que remite.

Las dimensiones actuales son 384 para texto, 512 para imagen, 24 para fundamentales
y 420 para macro. En los bloques numéricos se conservan valores transformados,
máscaras y antigüedad. `macro_units` guarda las unidades nativas de cada decisión.
No se presupone homogeneidad entre cambios de año base.

El precio se entrega bajo demanda como una ventana de 64 × 5. La transformación
usa precios relativos dentro de esa ventana y volumen relativo a su propia media.
No ajusta parámetros con validación o test. Los valores originales y los cálculos
macroeconómicos se conservan con su precisión antes de construir tensores `float32`.

Los índices de noticias se refieren al orden por disponibilidad y huella dentro
del activo. Las huellas también se conservan para reconstruir la relación sin
depender únicamente de una posición. No se exportan textos al repositorio.

`cost_profile_ready` indica que existen entradas completas para las mediciones.
`training_ready` permanece falso hasta fijar las etiquetas y la cohorte del estudio.
Las etiquetas usadas para medir coste se generan aparte y no se añaden al bloque
de entradas. El retorno futuro de SPY nunca se entrega al predictor.

El cursor del lector identifica la siguiente fila confirmada por el consumidor.
La lectura anticipada no confirma trabajo. Las particiones de trabajadores son
disjuntas. Los checkpoints del ensayo breve se toman en fronteras de época, no
afirman recuperación de una memoria adaptativa todavía no implementada.
