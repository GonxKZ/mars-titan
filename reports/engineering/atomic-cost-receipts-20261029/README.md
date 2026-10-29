# Recibos de coste escritos con `atomic_json`

Issue: [#802](https://github.com/GonxKZ/mars-titan/issues/802). Comprobación del 29 de octubre de 2026 en CPU, sin pasos de optimizador y sin medir nada nuevo.

## Qué cambia

La medida de coste de C2 perdió una ejecución completa porque escribía su recibo con `write_text` en una carpeta que todavía no existía. Quince medidas de `benchmarks/` seguían el mismo patrón. Ahora todas escriben con `mars_titan.data.storage.atomic_json`, que crea la carpeta, escribe en un temporal, lo confirma con `os.replace` y rechaza NaN e infinitos. Cada una comprueba antes de medir que el destino es nuevo.

- Cinco separan ya el protocolo de la medida con `--run`: `cross_asset_attention_cost.py`, `holographic_network_cost.py`, `online_arbiters_cost.py`, `regional_variants_cost.py` y `nested_equal_depth_cost.py`.
- Las otras diez medían dentro de `main`. Ahora separan una función `run(args)`, que mide y devuelve el recibo, y `main` queda con los argumentos, la comprobación del destino, la escritura y el resumen impreso.

Los recibos publicados no cambian. Los nuevos se escriben con sangría de dos espacios y con los caracteres no ASCII escapados, que es el formato de `atomic_json`. El contenido leído con `json.load` es el mismo.

## Evidencia

[`verification.json`](verification.json) reúne las 56 pruebas de las líneas de órdenes con cobertura de ramas, la regresión de las 105 pruebas de `tests/benchmarks` y 60 mutaciones dirigidas, cuatro por medida. Las cuatro quitan la comprobación del destino, miden antes de rechazarlo, escriben un recibo con NaN sin rechazarlo y escriben sin crear la carpeta. Las 60 se detectaron con fases en verde antes y después.

## Límites

Las pruebas sustituyen `run` para no medir, así que el cuerpo de las diez funciones `run` sigue sin cubrir, igual que estaba dentro de `main` antes del cambio. Las medidas de C2 (`axial_time_region_cost.py`) y de H4 (`asset_groups_cost.py`) todavía están en sus ramas y se pasarán a `atomic_json` cuando lleguen a `develop`.
