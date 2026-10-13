# Memoria de mercado compartida (PT4): coste y mutación

Recibos del 13 de octubre de 2026 para [#456](https://github.com/GonxKZ/mars-titan/issues/456).
La especificación está en [market-memory.md](../../../docs/engineering/market-memory.md).
Ningún paso lee objetivos ni el año sellado, no se ha usado la GPU y no se ha ejecutado ningún
paso de optimizador.

## Coste

[`cost.json`](cost.json) es el recibo de `benchmarks/titans_variant_cost.py` con la
[declaración](../../../configs/evaluation/market-memory-comparison.json) y el código del commit
`8fdb788d`, sobre las mismas entradas que los recibos de PT5 y PT6 (128 flujos de EE. UU. y
8 instantes de la vista de la campaña, huella `275365fa…`, sin objetivos). El control sectorial
lee las dos fichas de descripción del conjunto local, que no se versionan, y el recibo guarda su
identidad con la huella de cada ficha y del reparto.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/titans_variant_cost.py \
  --data <entradas>.npz --declaration configs/evaluation/market-memory-comparison.json \
  --output <recibo>.json --repetitions 15 --warmup 2 --threads 2 --sector-source <conjunto>
```

Las medianas, la optimización de la identidad del reparto y su lectura están en la
especificación. El proceso, con los tres modelos, alcanzó un pico de 0,96 GB de memoria
residente.

## Mutación dirigida

[`mutations.json`](mutations.json) aplica 32 mutaciones a la lectura y la escritura por sesión,
a la memoria compartida, al reparto sectorial, al predictor y al recorrido cronológico, cada una
sobre una copia aislada y con `tests/models/titans/test_market_memory.py`. Las 32 se detectan
en el commit `8fdb788d`.

La primera pasada dejó cuatro vivas: la consulta sin normalizar, la confirmación que conservaba
el grafo sin pedirlo, la barrera de truncamiento sin cortar el contexto y la recuperación sin
restaurarlo. Las dos últimas mostraron que en el corpus de prueba la cabeza del predictor sin
ajustar apenas cambia con el contexto, así que comparar solo las predicciones no bastaba. Las
pruebas del recorrido comparan ahora el contexto confirmado tras cada sesión, después de la
recuperación y en la barrera, y dos pruebas nuevas fijan la consulta y la confirmación sin grafo.
