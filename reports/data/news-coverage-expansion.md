# Ampliación de noticias completas contrastadas

Se revisaron 28 registros adicionales del piloto, entre 2018 y 2023, sin consultar
rentabilidades futuras. El registro acumulado pasa de 8 a 36 revisiones: 13 artículos
admitidos, 12 no verificables y 11 rechazados. Se mantienen los originales y las
revisiones anteriores. No se sustituye ningún cuerpo por su titular.

La selección fue dirigida por activo, año y posibilidad de identificar el medio,
con orden por SHA-256 de `42:source_record_hash` dentro de cada grupo. No es una
muestra representativa para estimar una tasa de errores. Se consultaron 32 páginas
de artículos entre ambas partes de esta ampliación. Los bloqueos de acceso no se
eludieron.

## Qué se considera comprobado

Los artículos admitidos conservan todos los párrafos editoriales y las declaraciones
de posiciones. La fuente visible permite contrastar el cuerpo, su fecha y la
relación sustantiva con el activo. No demuestra una instantánea histórica inmutable.
Las promociones y los avisos del distribuidor se excluyen mediante intervalos exactos,
con SHA-256 del texto seleccionado. No se reescriben frases.

Se excluyeron menciones incidentales, una colisión entre el símbolo de CSG Systems
y el de Credit Suisse, cuerpos atribuidos a otras empresas y un artículo incompleto.
También queda fuera a su fecha antigua un artículo de enero de 2019 cuya fuente
declara una actualización en abril y no proporciona la versión anterior.

El [registro de revisiones](../../data/manifests/news-reviews.json) conserva cada
localizador y enlace de evidencia. Por ejemplo, la [comparación entre Nike y
Deckers](https://www.fool.com/investing/2022/11/02/better-buy-nike-vs-deckers-outdoor/)
analiza directamente el negocio de Deckers. La [noticia sobre NetEase y UGG](https://www.fool.com/investing/2018/04/28/netease-chases-alibaba-and-jdcom-with-a-knockoff-m.aspx)
aporta información sobre competencia y propiedad intelectual de una marca del
activo. Una mención aislada en una lista no recibe el mismo tratamiento.

## Cobertura materializada

| Activo | Artículos admitidos | Muestras con cuatro modalidades y macro |
| --- | ---: | ---: |
| MNST | 6 | 30 |
| DECK | 7 | 35 |
| ABM | 0 | 0 |
| CSGS | 0 | 0 |

Las 65 muestras se escribieron en una preparación nueva con los codificadores
congelados. Los [resultados y huellas](news-coverage-expansion.json) conservan la
auditoría completa. No se modificó el manifiesto del piloto anterior ni su mínimo
de 252 muestras por activo. Esta ampliación permite probar recorridos de ejecución,
pero no ofrece todavía una cohorte suficiente para comparar precisión.

La GPU estaba compartida con otra aplicación que ocupaba unos 6 GB de VRAM.
La extracción coincidió parcialmente con comprobaciones locales. Por eso sus
tiempos no se utilizan como medida aislada de rendimiento ni para extrapolar una
campaña. No se cambió a CPU ni se detuvo la aplicación ajena.

## Verificación

`check_review_sources` vuelve a comprobar el archivo, ordinal, hash de línea,
metadatos y cuerpo de cada revisión. Detecta modificaciones del original y
revisiones con localizadores incompatibles. Comprueba las huellas completas antes
y después de leer las fuentes.

La suite completa pasó con 401 pruebas, incluida la comprobación CUDA. En el módulo
de revisiones, coverage.py midió 108 de 112 sentencias y 60 de 64 ramas, un 95,45 %
combinado. El contraste editorial y la cobertura limitada siguen siendo aspectos
distintos de estas pruebas de software.

Se añadió después una regresión para cambios del título que conservan cuerpo y
metadatos. Una mutación que elimina la comprobación del hash de línea fue detectada
por esa prueba. La suite habitual pasó entonces con 401 pruebas y una integración
CUDA omitida por requerir activación explícita. La función de contraste tiene
complejidad 22 y CRAP 22,010 con Radon 6.0.1 y cobertura de sentencias por función.

#7 continúa abierta porque aún falta cobertura suficiente. No se cierra la fase
de preparación por haber ampliado el registro.
