# Proyecciones de la sección 4.4 de Titans

Recibos del 10 de octubre de 2026 para [#449](https://github.com/GonxKZ/mars-titan/issues/449). La descripción de las ecuaciones, del estado y de las desviaciones está en el [núcleo de memoria](../../../docs/engineering/titans-memory-core.md#proyecciones-de-la-sección-44).

| Archivo | Contenido |
| --- | --- |
| [`parity-cpu-previous-core.json`](parity-cpu-previous-core.json) | Arnés de paridad con `lucidrains/titans-pytorch@1d40c445` en CPU sobre esta rama, con la configuración anterior del núcleo |
| [`parity-cuda-previous-core.json`](parity-cuda-previous-core.json) | El mismo arnés en `cuda:0` |
| [`cost-cuda.json`](cost-cuda.json) | Forward y backward de medida en `cuda:0` con las formas de la campaña y entradas reales, sin optimizador |

Ningún recibo ejecuta pasos de optimizador ni construye un optimizador. Todos usan FP32 estricto, sin TF32 ni autocast.

## Paridad con la configuración anterior

El arnés de [#447](https://github.com/GonxKZ/mars-titan/pull/447) (`benchmarks/titans_reference_parity.py`, SHA-256 `5d90338d…`) se ejecutó sobre esta rama con el mismo entorno aislado, las mismas referencias y la misma orden que el [recibo original](../titans-reference-parity-20261009/README.md). En CPU coinciden exactamente todas las cifras, las tolerancias, los casos y los resúmenes. Solo cambian la fecha, la duración y los hashes de `neural_memory.py` y `config.py`. En `cuda:0` coincide todo salvo el máximo del asignador, 454.464.000 bytes frente a 454.387.200. FP64 pasa los 5 casos en ambos dispositivos y FP32 repite los 3 casos sin residual dentro de la tolerancia y los 2 con residual y LayerNorm fuera de ella, con las mismas diferencias que ya explicaba el recibo original. La configuración nueva no tiene equivalente en lucidrains, que no implementa la convolución y normaliza q y k con RMSNorm.

## Coste sin entrenar

`benchmarks/titans_paper_projections_cost.py` trabaja en dos fases para no ocupar la GPU con lectura de datos. El recibo registra el commit 771f64f6, que contiene el núcleo medido. El script se añadió en el commit siguiente sin cambios respecto al ejecutado (SHA-256 `faa389a0…`).

1. `materialize`, con `memslot light` y sin GPU, abre la vista `US+CN/fold-012` de la campaña A con `CorpusDataset`, que comprueba el SHA-256 de todos los archivos de la edición, y copia las entradas de 128 activos de EE. UU. en 8 sesiones comunes del tramo de ajuste, del 15 al 25 de febrero de 2011. Usa las mismas funciones de decodificación que el lector por bloques y no lee objetivos. Tardó 16 min 52 s, casi todo en la comprobación de unos 58 GB, con 0,83 GB de memoria residente. El archivo resultante ocupa 7,1 MB y queda fuera del repositorio. El recibo conserva su hash.
2. `measure`, con `memslot gpu`, recorre la receta de la campaña (`hidden_size` 64, `persistent_tokens` 4, cabeza de cuantiles, `gate_bias` y memoria residual, FP32 estricto y sin fastpath de atención) en un tramo diferenciable de 8 instantes con 128 flujos, que es `truncation` y `block_rows` de la receta. Cada repetición hace el forward con grafo, una pinball frente a ceros solo para obtener un backward, el backward y una inferencia sin grafo del mismo tramo. No se construye ningún optimizador y los gradientes se descartan. Hay 3 repeticiones de calentamiento y 20 medidas.

Medianas en una RTX 4070 Laptop (CUDA 13.0, torch 2.14.0), con el percentil 95 entre paréntesis:

| Variante e identidad | Forward con grafo | Backward | Inferencia | Pico asignado | Filas por segundo con backward |
| --- | --- | --- | --- | --- | --- |
| `mac_online`, `linear_v1` | 90,4 ms (99,9) | 57,1 ms (69,6) | 85,7 ms (100,1) | 433,1 MiB | 7.105 |
| `mac_online`, `titans_mac_paper_projections_v2` | 105,1 ms (126,4) | 67,7 ms (94,9) | 97,8 ms (120,9) | 437,1 MiB | 5.931 |
| `mac_frozen`, `linear_v1` | 63,2 ms (77,1) | 31,1 ms (46,5) | 59,6 ms (99,6) | 319,1 MiB | 10.596 |
| `mac_frozen`, `titans_mac_paper_projections_v2` | 78,2 ms (89,0) | 39,9 ms (52,5) | 72,2 ms (98,5) | 320,6 MiB | 8.890 |

En `mac_online` el núcleo nuevo multiplica la mediana del forward por 1,16, la del backward por 1,19 y la de la inferencia por 1,14, y el pico de memoria por 1,009 (4 MiB más). Añade 768 parámetros (tres núcleos `64 × 1 × 4`) a los 186.055 del predictor y 2.304 bytes por flujo al estado. La GPU no estaba aislada. Otro proceso ocupaba unos 0,6 GB antes de empezar y la carga del sistema era alta, así que las cifras sirven para comparar las dos identidades medidas en la misma plaza, no como tiempos absolutos del recorrido completo.

El sobrecoste relativo es mayor que el cálculo añadido, unas pocas multiplicaciones por canal. Una explicación posible, que todavía no se ha perfilado, es que con 128 flujos por instante pesan más los lanzamientos de núcleos pequeños y las comprobaciones de finitud del estado, que sincronizan con la CPU, y cada ventana añade varias de ambas. Si el coste importa en la campaña, el siguiente paso sería perfilar el tramo y agrupar las comprobaciones o fusionar la convolución, con una referencia numérica.
