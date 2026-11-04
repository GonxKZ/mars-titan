# Cohortes por origen de la criba

La preparación de `screening_cohorts` crea una selección independiente de hasta 256 activos para cada origen de 2019, 2020, 2021 y 2022. Parte de las vistas históricas completas de EE. UU. con máscaras. Los archivos de precios, representaciones y etiquetas siguen en sus ubicaciones originales.

Un activo es elegible cuando el manifiesto de su origen declara al menos una etiqueta admitida en el tramo de ajuste. Los candidatos se ordenan por SHA-256 de la cadena formada por la sal fija, el mercado y el símbolo. El símbolo resuelve un posible empate de huellas. Se toman los primeros activos y se publican ordenados por símbolo. No se usa el valor del objetivo, la cantidad de historia por encima de una fila ni la cobertura futura para ordenar o excluir candidatos.

| Origen de evaluación | Vista | Información usada para elegir, anterior a |
| --- | --- | --- |
| 2019 | `fold-014` | 1 de abril de 2018 |
| 2020 | `fold-015` | 1 de abril de 2019 |
| 2021 | `fold-016` | 1 de abril de 2020 |
| 2022 | `fold-017` | 1 de abril de 2021 |

La cohorte de un origen no sustituye la de otro. Un activo elegido puede no tener filas en la validación o evaluación posterior y permanece seleccionado. La selección conserva todas las referencias, máscaras, muestras y límites de sus activos. El ajuste de uno a doce meses se aplica después mediante `fit_months`, sin cambiar esta regla de elegibilidad histórica. Los recuentos publicados son los del manifiesto temporal completo, no los de ese ajuste reducido.

La regla no reconstruye el universo que cotizaba en cada fecha ni corrige revisiones retrospectivas de las fuentes. Su garantía se limita a no usar cobertura ni valores futuros para elegir entre los candidatos del manifiesto recibido. Las limitaciones del universo y de la disponibilidad temporal siguen siendo las del corpus.

El comando recibe un directorio con `fold-014/manifest.json` hasta `fold-017/manifest.json` y un destino nuevo. Las rutas se pasan mediante las variables `MARS_SCREENING_SOURCES` y `MARS_SCREENING_COHORTS`:

```bash
uv run python -m mars_titan.training.screening_cohorts prepare \
  --sources "${MARS_SCREENING_SOURCES:?}" \
  --output "${MARS_SCREENING_COHORTS:?}"

uv run python -m mars_titan.training.screening_cohorts verify \
  --work "${MARS_SCREENING_COHORTS:?}"
```

La publicación confirma las cuatro vistas y su recibo en una sola operación. Rechaza un destino existente, también si aparece durante la preparación. Comprueba las huellas de los manifiestos de origen antes de publicar. `verify` vuelve a comprobar las huellas y a calcular la selección. Una modificación de la fuente, de una vista o de su selección invalida el recibo. No se sustituyen resultados anteriores.

Esta operación solo lee metadatos JSON. El recibo declara `verification_scope=manifests_only` y `row_files_opened=false`. No acredita haber comprobado los bytes de los Parquet, el contenido de cada fila ni la disponibilidad de los datos que estos referencian. El lector del corpus debe hacer sus comprobaciones antes de utilizarlos. Tampoco habilita aprendizaje, ejecuta optimizadores o abre el test final.

Las pruebas ejercitan la invariancia al orden y a la cobertura futura, los límites, los recuentos incompatibles, la alteración de archivos, las rutas y la publicación concurrente. El [recibo de comprobación](../../reports/engineering/screening-cohorts-20261026.json) recoge la cobertura, CRAP, las mutaciones dirigidas y la preparación realizada sobre los manifiestos locales. La criba científica y su comparación de variantes siguen pendientes.
