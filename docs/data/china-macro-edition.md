# Contexto macro sobre las sesiones chinas

La [edición del 6 de octubre de 2026](../../reports/data/china-macro-edition-20261006.json)
contiene 67.760 filas para 484 sesiones chinas de 2022 y 2023. La puerta de
cobertura admite 387 sesiones con los 140 indicadores, desde el 2 de junio de
2022 hasta el 29 de diciembre de 2023, sin duplicados. Las otras 97 sesiones
conservan sus ausencias documentadas.

El cálculo utiliza las adquisiciones ALFRED conservadas, las versiones publicadas
de los índices de condiciones y estrés financiero, el CSV mensual de GSCPI y
los 119 comunicados NBS y PBOC ya recuperados. No descarga fuentes nuevas ni
traslada directamente las decisiones del panel estadounidense. Cada fuente
conserva su zona y su límite documental, seguido del próximo cierre estricto
chino. Con esta política, GSCPI `May-22` entra el 2 de junio de 2022.

Se mantienen los retardos `source_records` del panel base y la política
`valid_observations` de las cuatro variaciones diarias ampliadas. Esa segunda
política se aplica a petróleo WTI, dólar amplio, CNY por USD y tipo a diez años.
STLFSI3 y STLFSI4 mantienen sus tramos de publicación, sin equiparar sus niveles.
El catálogo comparte los conceptos y las reglas del estadounidense. Solo difiere
la fecha administrativa `verified_on` de la fila de historia de estrés.

## Historia anterior al intervalo emitido

Un calendario que empieza en 2022 sitúa publicaciones anteriores en su primera
sesión. La comprobación inicial detectó 1.097 fechas de disponibilidad truncadas
en 85 indicadores base. Por ejemplo, la observación de cobre de noviembre de
2021 pasaba del 8 de diciembre de 2021 al 4 de enero de 2022. Eso recortaba su
antigüedad en 27 días durante 157 decisiones. Ese primer cálculo se conservó como
diagnóstico y no se admitió como edición final.

`macro_recalculation --history-start` separa el calendario de disponibilidad del
intervalo de salida. La edición CN usa calendario desde 2000 y emite únicamente
2022–2023. `calculate_macro(decision_start=...)` procesa los eventos necesarios,
con sus revisiones y expiraciones, y evita construir las filas anteriores a la
salida. Las fórmulas y los valores de los retardos permanecen iguales a los de
la referencia completa.

Ejemplo para recalcular el componente base desde la adquisición local existente:

```bash
PYTHONPATH=src uv run --no-sync python -m mars_titan.data.macro_recalculation \
  --source data/external/phase1-macro \
  --catalog data/catalogs/macro-indicators.csv \
  --output data/processed/cn-macro-base-20261006 \
  --market CN --history-start 2000-01-01 \
  --start 2022-01-01 --end 2023-12-31
```

Este componente todavía requiere las sustituciones y la admisión. Las huellas
de los ocho componentes, de la composición y de la verificación constan en el
informe de la edición. GSCPI permite `--source-edition` para reutilizar el CSV
y el recibo de una preparación anterior. Conserva la fecha de adquisición y
comprueba de nuevo ambos archivos antes de publicar.

## Comprobaciones y uso pendiente

Una consulta independiente en DuckDB reconstruye la composición y contrasta
las nueve columnas mediante diferencias de multiconjuntos en ambos sentidos.
No encuentra discrepancias en las 67.760 filas. También obtiene las mismas
387 sesiones con valores finitos, fechas admisibles y procedencia. Este contraste
no sustituye la revisión de las fórmulas ni certifica de nuevo las fuentes.

La [medición de la ventana histórica](../../reports/resources/macro-history-window-20261006.json)
compara la referencia que emite 814.240 filas y filtra después con la ruta que
emite las 67.760 solicitadas. Los Parquet coinciden exactamente. En tres parejas
alternadas después de la comprobación inicial, la mediana del proceso pasa de
85,38 a 26,17 segundos. El pico de RAM permanece alrededor de 1,05 GiB. La campaña
de entrenamiento seguía activa durante las medidas. No se midió energía ni se
atribuye esta reducción al tiempo de entrenamiento.

Pasan 400 pruebas de la batería macro y una comprobación adicional de la nueva
opción CLI. La revisión focal pasa 187 casos y detecta seis mutaciones dirigidas.
El recibo distingue el alcance de cada comprobación y su convención de cobertura.

Las 169 fechas potenciales del caso contable de Ping An conservan macro completo.
La [codificación CNY](chinese-multimodal-samples.md) y el [factor CSI 300](csi300-market-factor.md)
permiten leer 168 muestras etiquetadas anteriores al corte anual.
La admisión de contexto macro no acredita por sí sola muestras con las cuatro
modalidades ni etiquetas utilizables. La campaña estadounidense conserva su
edición y su runtime, y el test final permanece cerrado.
