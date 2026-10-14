# Fusión común de eventos: mutación dirigida

Recibo del 13 de octubre de 2026 para [#17](https://github.com/GonxKZ/mars-titan/issues/17).
La fusión y la relación de cada criterio con su evidencia están en la
[arquitectura](../../../docs/engineering/architecture.md#codificación-y-fusión-de-eventos). No se
ha usado la GPU ni se ha ejecutado ningún paso de optimizador.

[`mutations.json`](mutations.json) aplica 6 mutaciones a `fuse_modalities` y a su uso en
`FinancialPredictor`, cada una sobre una copia aislada y con
`tests/models/titans/test_common_fusion.py` y `tests/models/test_masked_reference_fusion.py`.
En el commit `0191b532` se detectan 5.

La que sobrevive multiplica la representación de precios por su bit de presencia. Es equivalente
para cualquier entrada admitida: `validate_historical_vectors` rechaza un lote sin precios o sin
gráficos, así que ese bit siempre vale 1 y la salida no cambia. La fusión no lo usa precisamente
por eso, como indica su docstring.
