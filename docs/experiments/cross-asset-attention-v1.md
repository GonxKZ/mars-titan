# Atención entre activos de la misma sesión

`cross_asset_attention_v1` es la variante C5 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751)). Después de MAC, cada flujo combina su representación con las de los demás activos de su mercado en el mismo instante de decisión. Está implementada en `src/mars_titan/models/titans/cross_asset_attention.py` y declarada en [su configuración](../../configs/experiments/cross-asset-attention-v1.json). Solo se ha comprobado con pruebas técnicas sobre fixtures. No se ha entrenado y no hay ninguna mejora medida.

## Motivación y alcance

`mac_online` predice cada activo con su propio estado. Los activos de una sesión comparten el mercado, el sector y las noticias del día, pero el predictor no ve esas relaciones salvo a través del contexto macro y del contexto de sesión PT4. C5 pregunta si leer las representaciones de los otros activos en el momento de emitir mejora la predicción de cada uno.

La referencia más cercana es MASTER ([Li et al., 2024](https://arxiv.org/abs/2312.15235)). MASTER codifica cada acción a lo largo del tiempo, aplica atención entre acciones en cada instante de su ventana y después resume el tiempo con la última posición como consulta. En MARS-TITAN el estado temporal de cada flujo es el de MAC y solo existe la representación de la emisión, así que C5 mezcla una única vez, en el instante de decisión. C5 tampoco incorpora la puerta guiada por el estado del mercado que MASTER aplica a las características de entrada.

## Ecuación ejecutada

Sea $e_i$ la representación de la fila i tras MAC y $G(i)$ el conjunto de filas del bloque con el mismo mercado y el mismo instante de decisión que i. Con H cabezas de dimensión $d_h$:

$$
n_i=\operatorname{LayerNorm}(e_i),\qquad
a^{h}_{ij}=\operatorname{softmax}_{j\in G(i)}\!\left(\frac{(W^{h}_{q}n_i)\cdot(W^{h}_{k}n_j)}{\sqrt{d_h}}\right),
$$

$$
e'_i=e_i+W_o\Bigl[\textstyle\sum_{j\in G(i)}a^{h}_{ij}\,W^{h}_{v}n_j\Bigr]_{h=1..H}.
$$

$W_o$ empieza en cero, de modo que el predictor con C5 emite al construirse lo mismo que sin ella. La semilla de C5 se deriva de la del predictor con SHA-256 y no consume el generador global, así que la variante y su control parten de los mismos pesos.

Todas las entradas de $G(i)$ son representaciones de decisiones del mismo instante, calculadas con entradas disponibles al emitir. La mezcla no lee etiquetas ni instantes posteriores.

## Control

El control `self_only` tiene los mismos parámetros y la misma inicialización, pero $G(i)=\{i\}$. Calcula la misma matriz de puntuaciones y la máscara deja solo la diagonal, de modo que la mezcla se reduce a $W_oW_vn_i$. Así separa dos efectos: la información de los otros activos (variante frente a control) y añadir una capa residual con más parámetros (control frente a la referencia sin C5). La declaración incluye ambos contrastes.

## Requisitos de ejecución

C5 necesita todas las filas de una sesión en el mismo bloque. El predictor la marca como operación por sesión (`session_level`), igual que los rangos de entrada y el objetivo IC de [la PR #760](https://github.com/GonxKZ/mars-titan/pull/760). `prepare` rechaza un bloque declarado incompleto y el entrenador repite sesiones completas al acumular gradientes. Con `block_rows` menor que las filas de un instante, o con `accumulation_rows` que parta los flujos de un segmento, el recorrido se detiene con un error en lugar de mezclar sesiones partidas.

El consumidor congelado todavía no registra C5 y la rechaza al construirse. Dust también la excluye de sus cambios admitidos. El código fuente de C5 entra en la identidad del entrenador solo cuando la variante está activa.

## Desviaciones respecto a MASTER

| MASTER | C5 | Motivo |
| --- | --- | --- |
| Atención entre acciones en cada instante de la ventana | Una mezcla en el instante de decisión | MAC no conserva una representación por instante pasado |
| Puerta guiada por el estado del mercado | Sin puerta | Aislar la atención entre activos. La puerta sería otra variante |
| Bloque Transformer con capa feed-forward | Atención residual con normalización previa y $W_o$ a cero | Paridad exacta con la referencia al construirse |
| Universo CSI300 y CSI800 | Activos de US y CN de FinMultiTime, por mercado | Las sesiones de cada mercado tienen calendarios distintos |

## Comprobación técnica

{{VERIFICACION}}

## Coste

La medida CPU del 28 de octubre llama solo al módulo `CrossAssetAttention` con representaciones aleatorias en FP32, H = 64 y cuatro cabezas. Las filas se reparten en sesiones de igual tamaño con dos mercados y un instante. Se usaron los ocho hilos por defecto de PyTorch, tres calentamientos y veinte repeticiones en cada uno de los dos órdenes de los modos. El equipo tenía otras cargas, con una media de carga de un minuto de 7,5 al empezar y 7,0 al terminar, así que los percentiles son descriptivos. `forward` es la inferencia sin gradiente y `backward` añade la propagación de una suma de cuadrados hasta los parámetros de C5, sin optimizador. El [recibo](../../reports/engineering/cross-asset-attention-20261028/cost-cpu.json) conserva todas las muestras.

| Filas | Sesiones | Forward C5 | Forward control | Backward C5 | Backward control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 0,30 ms (0,35) | 0,33 ms (0,43) | 0,98 ms (1,36) | 0,74 ms (1,23) |
| 256 | 1 | 0,93 ms (1,00) | 1,01 ms (1,77) | 2,24 ms (2,87) | 1,90 ms (3,50) |
| 1.024 | 4 | 5,67 ms (10,29) | 5,74 ms (9,54) | 10,19 ms (15,19) | 9,84 ms (17,16) |

Cada celda da la mediana y, entre paréntesis, el percentil 95. Los dos modos tienen 16.512 parámetros. El control calcula la misma matriz de puntuaciones y solo cambia la máscara, por eso cuesta casi lo mismo. Con 256 activos, el milisegundo del forward es pequeño frente a los 0,57 s del forward del predictor completo medido el mismo día para [A4](opponent-heads-v1.md) con cuatro hilos, aunque esa relación no se ha medido en un mismo proceso. No se han medido la memoria, la VRAM ni la energía.

```bash
CUDA_VISIBLE_DEVICES=-1 uv run python -m benchmarks.cross_asset_attention_cost --run \
  --output "${MARS_CROSS_ASSET_COST:?}"
```

## Pendiente

Faltan las comprobaciones CUDA, el registro en el consumidor congelado y la comparación predictiva con las mismas filas, cortes y presupuesto. La combinación con el lector episódico no está declarada. El aprendizaje sigue bloqueado y la variante no se activa por defecto.
