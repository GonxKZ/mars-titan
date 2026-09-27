# Admisión de cobertura macro completa

`assess_macro_completeness` comprueba el panel largo de indicadores antes de
seleccionar muestras de una nueva edición. Cada sesión del mercado debe contener
exactamente una fila por cada identificador del catálogo, con un valor finito,
disponibilidad anterior o igual a la decisión y procedencia SHA-256. Un cero
observado es válido. Una máscara presente no sustituye un valor ausente.

Las series originales necesitan su unidad histórica y su ajuste estacional.
La unidad histórica puede diferir de la unidad orientativa del catálogo. Los
derivados deben conservar la unidad de su fórmula y una lista de huellas de sus
fuentes. Las filas con una ausencia declarada, un periodo futuro o una política
de catálogo que excluya la fuente no se admiten.

La comprobación utiliza el cálculo ya materializado. No vuelve a calcular los
rezagos ni reconstruye publicaciones a partir del panel diario. Esa evidencia
corresponde a la adquisición y al motor de versiones históricas. No introduce
imputaciones ni reemplaza series por indicadores parecidos.

```python
from pathlib import Path
from mars_titan.data.macro_coverage import assess_macro_completeness

report = assess_macro_completeness(
    Path("data/processed/phase1/macro-US.parquet"),
    Path("data/catalogs/macro-indicators.csv"),
    Path("/tmp/mars-titan/macro-admission"),
    start="2009-01-01",
    end="2023-12-31",
    market="US",
)
```

El periodo incluye ambos días y cuenta todas las sesiones del reloj del mercado,
también las que no tienen filas. La lectura consulta primero las fechas de cada
grupo Parquet. Si un grupo cruza el corte final, solo materializa el prefijo
anterior a ese corte como lote entregado al validador. Un orden que intercale
filas futuras provoca un error antes de solicitar ese lote. Arrow puede leer
páginas o diccionarios internos que compartan datos a ambos lados del corte.
La comprobación no inspecciona ni agrega filas futuras. Las huellas SHA-256
recorren los bytes del archivo completo para identificar la fuente, sin
interpretar sus valores.

El directorio de salida debe ser nuevo y estar fuera de los directorios de las
fuentes. Se publican juntos dos archivos después de volver a comprobar las
huellas del panel y del catálogo:

- `complete-decisions.parquet` contiene `prediction_at` y `macro_available_at`
  como timestamps UTC, junto con `indicator_count`. Solo contiene decisiones
  admisibles.
- `report.json` conserva las huellas de la fuente, el catálogo, el periodo y el
  Parquet de salida. Incluye sesiones totales, completas y excluidas, los motivos
  de invalidez y los recuentos por indicador.

La publicación utiliza `renameat2` con `RENAME_NOREPLACE` en Linux. Si no está
disponible, falla sin sustituir la salida. Se comprueban las rutas reales de los
orígenes, también cuando se reciben mediante enlaces simbólicos. El prefijo
temporal conserva sus diccionarios dentro de un presupuesto de 64 MiB. Se valida
en fragmentos de hasta 512 filas, comprobando su expansión antes de convertirlos
en arrays densos. El número de motivos de ausencia distintos está limitado a 4.096.

`missing_by_indicator` cuenta las sesiones en las que el indicador no tiene
exactamente una fila válida. `absent_rows_by_indicator` cuenta únicamente las
sesiones sin fila. Los motivos de invalidez pueden coincidir en la misma fila y
sus recuentos no deben sumarse como exclusiones independientes.

`population_ready` solo indica que existe alguna decisión con cobertura macro
completa. No acredita las otras modalidades, el tamaño de los cortes temporales
ni la admisión de una campaña. Con cero decisiones completas es siempre `false`
y el Parquet conserva su esquema, aunque no contenga filas.

La ejecución del 27 de septiembre de 2026 sobre el panel estadounidense, entre
2009 y 2023, seleccionó 528.360 filas de 3.774 sesiones. Ninguna sesión tenía los
140 indicadores válidos. Quince indicadores estaban ausentes en todo el periodo.
Se admitieron como valores observados 11.299 ceros, sin convertirlos en ausencias.
La fuente tenía SHA-256
`73952d78a67a2976ce188e84f68fa13786c6b79479b8752e8837acff01da50a0`
y el catálogo
`f5cb54532bdeecc35bc5ed43a1c7c9f0453c428cd493611bcf21d25518ba2981`.
