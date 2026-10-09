# Mantenimiento del repositorio

Estas utilidades conservan la biblioteca, revisan el repositorio y capturan fuentes públicas. No implementan preparación de muestras, entrenamiento, modelos ni evaluación financiera.

```bash
uv run python scripts/fetch_references.py
uv run python scripts/check_repository.py
uv run python scripts/refresh_public_sources.py --list
uv run python scripts/refresh_public_sources.py --source fed_monetary_rss
```

El descargador bibliográfico usa los catálogos de referencias, conserva archivos previos y genera un registro local. El verificador comprueba enlaces locales, metadatos, correspondencia BibTeX, planificación y archivos excluidos.

El actualizador público requiere `curl` y escribe nuevas instantáneas en `data/external/`, fuera de Git. Su selección por defecto incluye ocho fuentes renovables. Usa límites de tiempo y tamaño, valida el contenido y suspende un proveedor tras HTTP 403 o 429. Los PDF fijos requieren selección explícita y `pdfinfo`. La [guía de actualización](../docs/data/public-source-updates.md) explica sus límites y la separación respecto a los datos experimentales.

## Objetivos residuales con un solo proceso

`prepare_corpus_targets.py` envuelve `mars_titan.training.corpus_targets.prepare_corpus_targets`. Sin `--execute` solo comprueba. Lee el manifiesto materializado, valida su política de entradas, cuenta activos por mercado, recibos ya confirmados y el estado del cerrojo, y no escribe ningún archivo ni calcula etiquetas.

```bash
uv run --no-sync python scripts/prepare_corpus_targets.py \
  --manifest <manifiesto materializado> --prepared <preparados> --output <supervisión> \
  --input-policy historical_masked_2000_v1 --target-factors <revisión de factores>
```

Con `--execute` toma un cerrojo exclusivo en `.<destino>.lock`, junto al destino y fuera de él, y lo mantiene durante toda la preparación. Un segundo proceso sobre el mismo destino termina con código 3 sin tocar nada. La reanudación por recibos de la función no cambia. Esta utilidad no se ha ejecutado sobre la edición real. Generar los objetivos reales corresponde a la etapa 2 de la [campaña](../docs/research/training-campaign-2000.md) y sigue sujeto a la verificación de la edición.

## Campaña con máscaras desde 2000

`run_masked_campaign.py` envuelve `mars_titan.training.masked_campaign`, `mars_titan.training.campaign_throughput` y `mars_titan.training.campaign_extensions`. `check` cuenta los trabajos de una variante sin leer datos, `prepare` crea las vistas por ámbito, `run` ejecuta y reanuda los trabajos con recibos, `sources` publica el manifiesto de fuentes de un ámbito, `throughput` mide el caudal de cada familia sin pasos de optimizador y `extensions` comprueba los recuentos de la declaración preparada de la GRU candidata, MARS-TITAN y CM-v1 sin activarla. `run` respeta el bloqueo de aprendizaje antes de empezar y antes de cada trabajo. La [campaña](../docs/research/training-campaign-2000.md#ejecución-y-recuperación) describe sus órdenes. Ninguna orden se ha ejecutado sobre la edición real.

```bash
uv run --no-sync python scripts/run_masked_campaign.py check \
  --campaign configs/baselines/historical-masked-campaign-a.json
```

## Control de la cabeza de cuantiles

`run_quantile_head_control.py` envuelve `mars_titan.training.quantile_head_control`. `check` cuenta los trabajos del [control declarado](../configs/baselines/quantile-head-control-us.json) sin leer datos, `run` ajusta o reanuda cada par del Transformer compacto con recibos y respeta el bloqueo de aprendizaje, y `decide` aplica la regla de retroceso de [#22](https://github.com/GonxKZ/mars-titan/issues/22) con las predicciones de validación confirmadas. La [guía de la cabeza](../docs/engineering/quantile-head.md#ejecutor-del-control) describe ventanas, contraste y coste estimado. Ninguna orden de ajuste se ha ejecutado.

```bash
uv run --no-sync python scripts/run_quantile_head_control.py check \
  --plan configs/baselines/quantile-head-control-us.json
```

## Simulación de la parada temprana

`simulate_early_stopping.py` aplica la selección de `mars_titan.training.selection` a los historiales de validación de ejecuciones ya confirmadas, sin modificarlas, con varias paciencias, mejoras mínimas y épocas mínimas, en modo individual y conjunto. Agrupa los ajustes por ventana, ámbito, etapa, semilla y caso, cuenta como censurado un ajuste que necesitaría épocas no registradas y escribe en la salida estándar épocas recorridas, pérdida relativa del estado elegido y huella de las fuentes. No entrena ni lee datos. Su resultado sobre las referencias de la edición ampliada anterior está en el [recibo de la simulación](../reports/engineering/early-stopping-history-20261009.json) y se discute en la [parada temprana opcional](../docs/research/walk-forward-2000.md#parada-temprana-opcional).

```bash
uv run --no-sync python scripts/simulate_early_stopping.py \
  data/interim/real-expanded-references-20261006 > early-stopping.json
```

Las pruebas de `tests/tooling/` protegen estos comportamientos de mantenimiento sin realizar peticiones de red.
