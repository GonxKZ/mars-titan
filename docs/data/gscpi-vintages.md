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

## Contraste de las publicaciones preliminares

La [revisión del 6 de octubre](../../reports/data/gscpi-release-audit-20261006.json) compara los adjuntos actuales de tres publicaciones con las columnas mensuales del CSV. Los valores se redondean a los dos decimales del CSV antes del contraste. Una fecha de artículo no basta para fechar el contenido que devuelve hoy su enlace.

| Publicación | Último periodo del adjunto actual | Diferencias frente a la columna mensual | Límite de disponibilidad |
| --- | --- | --- | --- |
| [4 de enero de 2022](https://libertystreeteconomics.newyorkfed.org/2022/01/a-new-barometer-of-global-supply-chain-pressures/) | Diciembre de 2021 | 0 de 292 frente a `Jan-22` | El comentario del proveedor del 7 de enero da valores de octubre y noviembre distintos de los del adjunto actual. Este declara un guardado posterior, del 10 de enero. |
| [3 de marzo de 2022](https://libertystreeteconomics.newyorkfed.org/2022/03/global-supply-chain-pressure-index-march-2022-update/) | Febrero de 2022 | 44 de 294 frente a `Mar-22` | El contenido y la fecha interna son compatibles con el artículo, pero no acreditan cuándo se difundió esa serie concreta. |
| [18 de mayo de 2022](https://libertystreeteconomics.newyorkfed.org/2022/05/global-supply-chain-pressure-index-may-2022-update/) | Mayo de 2022 | 275 de 292 frente a `May-22` | El texto solo presenta datos hasta abril. El adjunto actual contiene además mayo y no puede heredar la fecha del artículo. |

Con los otros 138 indicadores de la edición ampliada, resolver GSCPI y su cambio mensual podría recuperar siete sesiones, del 20 al 31 de mayo de 2022. Es un diagnóstico de cobertura macro, no un recuento de muestras multimodales admitidas. Para usar el adjunto de marzo falta una copia histórica o confirmación del proveedor que vincule su serie global con una publicación anterior al 20 de mayo.

La consulta a Internet Archive devolvió HTTP 429 en tres intentos y otra consulta agotó su tiempo de espera. Esto no demuestra que no exista una copia. No se han admitido sesiones adicionales ni cambiado los datos de la campaña activa.

El contraste posterior con FRASER descargó las copias de [marzo](https://fraser.stlouisfed.org/title/9884/item/734953/content/xlsx/frbny_libertystreet_20220303) y [mayo](https://fraser.stlouisfed.org/title/9884/item/734967/content/xls/frbny_libertystreet_20220518). Ambas coinciden byte a byte con los adjuntos actuales de New York Fed. Los registros de catálogo se crearon el 24 de agosto de 2026 y no indican una captura contemporánea a los artículos.

La copia de mayo conserva también el periodo de mayo de 2022, posterior al artículo del día 18 que solo presenta datos hasta abril. Por tanto, la fecha bibliográfica de FRASER tampoco acredita cuándo estuvieron disponibles esos valores concretos. Las huellas y los metadatos constan en `fraser_archive_check` del informe. Este contraste no incorpora nuevas sesiones.
