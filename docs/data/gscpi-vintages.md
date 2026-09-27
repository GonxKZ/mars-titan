# GSCPI y disponibilidad de sus versiones mensuales

El CSV oficial de la Reserva Federal de Nueva York conserva una columna por mes de versión. La estimación actual del Excel no sustituye a esas columnas. El índice se publica mensualmente desde el 18 de mayo de 2022 y su calendario habitual indica el cuarto día hábil del mes. [Anuncio de publicación](https://www.newyorkfed.org/newsevents/news/research/2022/20220518), [página del índice](https://www.newyorkfed.org/research/policy/gscpi).

Las etiquetas mensuales del CSV no contienen una hora ni un día exactos de difusión. La política `NYFED_MONTHLY_VINTAGES` interpreta la etiqueta como el mes de la versión publicada y retrasa su admisión hasta la sesión siguiente al final de ese mes. Es una regla conservadora de disponibilidad, no una reconstrucción del instante de publicación. Los recibos conservan `publication_timestamp_verified=false`. Esta interpretación del archivo mensual debe permanecer visible al comparar resultados.

Se excluyen las columnas preliminares de enero a abril de 2022. La publicación regular comienza en mayo y la primera sesión estadounidense admitida por esta regla es el 1 de junio de 2022. Un valor correspondiente a 1997 no se convierte por ello en información disponible en 1997.

`macro_gscpi.parse_gscpi_vintages` conserva cada versión y su etiqueta. Comprueba fechas, orden, duplicados, meses ausentes, valores no finitos y periodos futuros. Solo convierte a números las columnas del intervalo solicitado. Los huecos del proveedor siguen siendo huecos. El campo de ajuste identifica un índice compuesto tal como se publica, sin atribuir un tratamiento estacional uniforme a sus componentes.

El cambio mensual utiliza la fórmula del catálogo sobre los dos periodos de la misma versión. No resta una estimación actual a una publicación antigua. El cálculo se realiza en float64, conservando la precisión de los valores que contiene el CSV.

```bash
uv run --no-sync python -m mars_titan.data.macro_gscpi \
  --catalog data/catalogs/macro-indicators.csv \
  --output data/processed/gscpi-monthly-20260927 \
  --market US --start 2009-01-01 --end 2023-12-31
```

La salida contiene el CSV descargado, el panel Parquet y un informe con sus huellas, catálogo, periodo y regla de disponibilidad. Se publica en un directorio nuevo, sin sustituir una edición anterior.

La ejecución de 2009 a 2023 produjo 7.548 filas para 3.774 sesiones. Hay 796 valores admitidos entre el índice y su cambio mensual. Las 7.548 comparaciones con una selección SQL independiente coinciden, con tolerancias absoluta y relativa de `1e-12`. Esa prueba comprueba la regla implementada, no demuestra una hora de publicación que el archivo no proporciona.

La recuperación no acredita una población con todos los macros ni una mejora de los modelos. Los datos descargados permanecen fuera del repositorio. Se conservan las [condiciones del proveedor](https://www.newyorkfed.org/privacy/termsofuse) y la atribución a Federal Reserve Bank of New York, Global Supply Chain Pressure Index.
