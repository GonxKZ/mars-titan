# Memoria multiescala (PT5): coste y mutación

Recibos del 13 de octubre de 2026 para [#457](https://github.com/GonxKZ/mars-titan/issues/457).
La especificación está en [multiscale-memory.md](../../../docs/engineering/multiscale-memory.md).
Ningún paso lee objetivos ni el año sellado, no se ha usado la GPU y no se ha ejecutado ningún
paso de optimizador.

## Coste

[`cost.json`](cost.json) es el recibo de `benchmarks/titans_variant_cost.py` con la
[declaración](../../../configs/evaluation/multiscale-memory-comparison.json) y el código del
commit `be418f1f`, sobre las mismas entradas que el recibo de PT6 (128 flujos de EE. UU. y
8 instantes de la vista de la campaña, huella `275365fa…`, sin objetivos).

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/titans_variant_cost.py \
  --data <entradas>.npz --declaration configs/evaluation/multiscale-memory-comparison.json \
  --output <recibo>.json --repetitions 15 --warmup 2 --threads 2
```

Las medianas y su lectura están en la especificación. El proceso, con los tres modelos, alcanzó
un pico de 1,13 GB de memoria residente.

## Mutación dirigida

[`mutations.json`](mutations.json) aplica 18 mutaciones a la cascada, al banco, a la lectura, a la
condición de contracción, al presupuesto y al transporte de los niveles, cada una sobre una copia
aislada y con `tests/models/titans/test_multiscale_memory.py`. En el commit `be418f1f` se
detectan 17 y una no se aplica.

La primera pasada, sobre el código anterior, dejó tres vivas:

- La cascada leyendo la media de sus niveles. Se añadió
  `test_the_cascade_reads_only_its_first_level`.
- El presupuesto previo sin los niveles. La validación del estado ya rechazaba el exceso después
  de reservar los tensores, así que la prueba de presupuesto comprueba ahora también la cuenta
  previa (`_check_size`).
- `contraction` sin el punto α = 1 − g_{1,2}. Era una mutación equivalente: la única fila que
  depende de α es convexa y su máximo está en α_lo o en 1. El código evalúa ahora solo los dos
  extremos y por eso esa mutación figura como no aplicada.
