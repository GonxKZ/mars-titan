# Auditoría de precios e identidad

## Ejecución sobre la copia completa

Se recorrieron los 5.023 CSV de precios con el código de `d451c1b`, sin reutilizar
resultados de una pasada anterior. Las huellas de las fuentes se comprobaron antes
y después de cada lectura. Los recuentos coinciden con la auditoría anterior.

| Mercado | Archivos | Filas originales | Admitidas | Excluidas |
| --- | ---: | ---: | ---: | ---: |
| Estados Unidos | 4.213 | 17.448.737 | 17.385.328 | 63.409 |
| China | 810 | 3.139.412 | 3.133.858 | 5.554 |

Se escribieron 20.092 Parquet que ocupan 885.496.734 bytes. Contienen los precios
admitidos, 68.963 exclusiones localizables, 223.934 registros con hechos corporativos
declarados o valores que requieren revisión y 87.793 filas de cobertura por activo
y año. No son 223.934 acciones verificadas de forma independiente.

La ejecución tardó 332,46 segundos y alcanzó 189,40 MiB de RSS del proceso. Es una
medida de esta auditoría CPU y de sus escrituras, no del entrenamiento ni de toda
la memoria usada por el sistema operativo. El [informe de ejecución](price-audit-detailed.json)
conserva cifras exactas, política e identificadores de estado e instantánea.

La [segunda ejecución](price-audit-resume.json) verificó y reutilizó los 5.023
archivos y sus 20.092 derivados en 11,54 segundos. La huella del estado fue idéntica,
con 161,06 MiB de RSS máxima. Esa comprobación acredita la reanudación por archivo,
no la recuperación de un entrenador ni el consumo de las otras modalidades.

## Qué identifica cada serie

`finmultitime:mercado:símbolo` es un identificador estable dentro de la instantánea
original. No acredita continuidad de la entidad legal ni pertenencia histórica a
un índice. No se unen símbolos distintos por similitud ni se interpreta la última
fila como una baja verificada. El catálogo original carece de un mapa temporal de
cambios de símbolo, fusiones y reutilizaciones que permita reconstruir esa identidad.

El piloto se elige con cobertura anterior a su fecha de corte. No exige que el
instrumento siga existiendo al final del periodo de evaluación. Aun así, el archivo
original es retrospectivo y esa selección no elimina el sesgo de supervivencia de
su adquisición. La prueba de selección modifica información futura y comprueba
que no cambia el orden elegido.

## Reglas de admisión

Una barra necesita una sesión identificable, OHLC finitos y positivos, volumen no
negativo y `Low <= Open, Close <= High`. Un volumen cero no se transforma en una
exclusión automática. Si se repite una sesión, se excluyen todas sus filas para
no elegir una versión arbitraria. Los motivos pueden solaparse y su suma no tiene
por qué ser igual al número de filas excluidas.

`source_row` es el ordinal del registro CSV después de la cabecera. Cada exclusión
conserva ese ordinal, el campo `Date` y todos sus motivos. El estado relaciona la
partición con la ruta y el SHA-256 del original. Así se recupera la fila completa
sin copiar su contenido al repositorio público.

## Cobertura y ausencias

Se cuantifican filas observadas, admitidas y excluidas por activo y año. Dentro del
intervalo observado también se cuentan las sesiones esperadas sin fila. Una sesión
con una fila inválida no se confunde con una sesión ausente. Las fechas ilegibles
se contabilizan aparte, porque no se les puede asignar un año verdadero.

No se rellenan huecos. Una ausencia puede deberse a cobertura incompleta, falta de
negociación u otras causas que este archivo no permite distinguir. Tampoco se
extrapola cobertura antes de la primera fecha ni después de la última.

## Splits, dividendos y retornos

Las columnas `Dividends` y `Stock Splits`, cuando existen, se conservan en una tabla
separada con sus registros originales. Se señalan valores ausentes, no numéricos,
negativos o no finitos. No se vuelven a ajustar precios por esas columnas. Su
presencia no documenta por sí sola cómo se ajustaron los OHLC distribuidos.

El retorno apertura-cierre usado por las sondas es `Close / Open − 1`, tanto para
el activo como para SPY. Si un ajuste multiplicativo fuera idéntico en ambos precios
de la sesión, se cancelaría en ese cociente. Esa propiedad algebraica no demuestra
que el proveedor aplicase siempre un ajuste coherente. Los niveles históricos y
las señales entre sesiones siguen expuestos a la información retrospectiva.

Se localizaron tres casos en los originales: AAPL, registro 5.199 del 31 de agosto
de 2020, con `Stock Splits = 4`, TSLA, registro 3.062 del 25 de agosto de 2022, con
`Stock Splits = 3`, y SPY, registro 6.028 del 15 de diciembre de 2023, con dividendo
declarado. Se usa en los tres el mismo cociente apertura-cierre, sin aplicar otra
vez la acción declarada. Los hashes de esos archivos constan en el estado de la
auditoría. Estos casos localizables no certifican todos los ajustes del proveedor.

## Derivados y recuperación

La opción `audit-prices --details` genera cuatro Parquet por activo: precios
admitidos, exclusiones, hechos corporativos y cobertura anual. Procesa una serie
cada vez. Las tablas de auditoría tienen tipos explícitos incluso si están vacías.
El archivo de estado guarda tamaños, filas y SHA-256 de cada partición.

La reanudación solo reutiliza fuentes y derivados cuyas huellas coinciden. Rechaza
cambios de política, instantánea o directorio, así como derivados alterados. Un
directorio ajeno no se adopta como salida. Se confirma el estado cada 64 archivos,
de modo que una interrupción puede exigir repetir como máximo ese bloque.

```bash
uv run --locked mars-data audit-prices \
  --state data/interim/price-detail-state-v2.json \
  --details data/processed/price-audit-v2 \
  --report reports/data/price-audit-detailed.json
```

Los archivos de datos y el estado quedan fuera de Git. La comprobación de ambos
mercados no autoriza entrenamiento multimodal en China, cuya disponibilidad
fundamental sigue sin acreditarse.

## Verificación local

La suite completa pasó con 358 pruebas. Se comprobaron motivos superpuestos,
fechas nulas y no canónicas, sesiones ausentes, conservación de precios ante
splits y dividendos, destinos protegidos y recuperación con huellas verificadas.
La revisión detectó dos problemas de fechas, reproducidos mediante pruebas
fallidas y corregidos antes de la ejecución definitiva.

En `prices.py` y `audit.py`, coverage.py 7.16.1 midió 140 de 162 sentencias y 47 de
62 ramas cubiertas, un 83,48 % combinado. Ese alcance incluye la auditoría china
de fundamentales, que no forma parte de este cambio. Radon 6.0.1 midió complejidad
47 en `audit_prices`, con CRAP 50,030 usando cobertura de sentencias por función
y `C² × (1 − cobertura)³ + C`. Es la función más compleja de este cambio. Su
cobertura no constituye una garantía de corrección por haber superado las pruebas.

Cuatro mutaciones dirigidas se detectaron: admitir fechas no canónicas, contar
fechas desconocidas como sesiones, omitir la huella de derivados y permitir una
instantánea distinta durante la recuperación. Se aplicaron en memoria. No se
presentan como una campaña exhaustiva de mutación.
