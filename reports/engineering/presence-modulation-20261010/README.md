# Fusión condicionada por la presencia (PT6): coste y mutación

Recibos del 13 de octubre de 2026 para [#458](https://github.com/GonxKZ/mars-titan/issues/458).
La especificación está en [presence-modulation.md](../../../docs/engineering/presence-modulation.md).
Ningún paso lee objetivos ni el año sellado, no se ha usado la GPU y no se ha ejecutado ningún
paso de optimizador.

## Coste

[`cost.json`](cost.json) es el recibo de `benchmarks/titans_variant_cost.py` con la
[declaración](../../../configs/evaluation/presence-modulation-comparison.json) y el código del
commit `f7ba79c7`. Las entradas son las del recibo de las proyecciones del artículo: 128 flujos
de EE. UU. y 8 instantes del tramo de ajuste de la vista de la campaña (huella `275365fa…`), con
sus bits de presencia reales y sin objetivos.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python benchmarks/titans_variant_cost.py \
  --data <entradas>.npz --declaration configs/evaluation/presence-modulation-comparison.json \
  --output <recibo>.json --repetitions 15 --warmup 2 --threads 2
```

Las medianas y su lectura están en la especificación. El proceso, con los tres modelos, alcanzó
un pico de 0,95 GB de memoria residente.

## Mutación dirigida

[`mutations.json`](mutations.json) aplica 15 mutaciones a `presence_modulation.py`, a la fusión
común y a la declaración en `financial.py`, cada una sobre una copia aislada y con
`tests/models/titans/test_presence_modulation.py`. Las 15 se detectan en el commit `f7ba79c7`.
La primera pasada dejó tres vivas: la fila completa pasando por la fórmula (solo cambiaba el signo
de un cero), la rama ancha sorteando con el generador global y la rama sin SiLU. Se añadieron las
pruebas `test_film_keeps_complete_rows_bit_for_bit_including_signed_zeros` y
`test_wide_branch_draws_its_expansion_from_its_own_seed_and_applies_silu`.
