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

Las pruebas de `tests/tooling/` protegen estos comportamientos de mantenimiento sin realizar peticiones de red.
