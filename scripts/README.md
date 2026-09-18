# Mantenimiento del repositorio

Estas utilidades sirven exclusivamente para conservar y revisar la documentación. No implementan preparación de muestras, entrenamiento, modelos ni evaluación financiera.

```bash
uv run python scripts/fetch_references.py
uv run python scripts/check_repository.py
```

El descargador usa los catálogos de referencias, conserva archivos previos y genera un registro local. El verificador comprueba enlaces locales, metadatos, correspondencia BibTeX, planificación y archivos excluidos. Las pruebas de `tests/tooling/` protegen estos comportamientos de mantenimiento.
