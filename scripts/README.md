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

Las pruebas de `tests/tooling/` protegen estos comportamientos de mantenimiento sin realizar peticiones de red.
