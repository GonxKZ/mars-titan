# FinanceDatabase como catálogo auxiliar

Revisión del 21 de septiembre de 2026, fijada en
[`d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e`](https://github.com/JerBouma/FinanceDatabase/commit/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e).
No se instaló el paquete, descargó el corpus completo ni modificó FinMultiTime.

## Decisión

Puede ayudar a comprobar identidades y organizar candidatos de revisión. No
amplía OHLCV, noticias, fundamentales o imágenes y no sustituye ninguna modalidad.
Su [README](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/README.md)
describe un catálogo de más de 300.000 símbolos, no una base histórica de precios
o estados financieros. Finance Toolkit consulta otros proveedores por separado.
Eso no acredita gratuidad, cobertura ni disponibilidad histórica de esos proveedores.

## Qué se comprobó

| Recurso primario | Observación | Consecuencia |
| --- | --- | --- |
| [Acceso a datos](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/financedatabase/helpers.py) | Descarga archivos comprimidos desde `main/compression/` y usa pandas | Fijar versión del paquete no congela los datos. Usar commit y hashes |
| [Consulta de acciones](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/financedatabase/Equities.py) | Excluye bajas por defecto. La selección de cotización primaria usa una regla sobre el símbolo | No aplicar esos filtros a un universo histórico |
| [CSV pequeño](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/database/equities/AQS.csv) | Hay campos ausentes y una discordancia entre descripción británica y país estadounidense en una fila | El catálogo requiere contraste, no es autoridad única |
| [Actualización](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/.github/workflows/database_update.yml) | Cadencia semanal y ejecuciones por cambios. Agrega símbolos de otra fuente | Fecha del commit no equivale a fecha efectiva de un evento financiero |
| [Ejecución revisada](https://github.com/JerBouma/FinanceDatabase/actions/runs/35509379813) | La actualización del 20 de septiembre terminó con fallo en la comprobación de clasificación | No interpretar cada actualización como una instantánea totalmente validada |

Los CSV de acciones incluyen símbolo, nombre, divisa, sector, industria, bolsa,
MIC, país, identificadores ISIN/CUSIP/FIGI y un indicador de baja. No incluyen
intervalos de vigencia ni CIK. Un sector o estado de baja actual no se aplica
retrospectivamente como una característica disponible en 2014.

## Unión propuesta, todavía no implementada

La clave original sigue siendo `finmultitime:mercado:símbolo`. El catálogo externo
produciría un inventario separado marcado como `current_snapshot`.

1. Descargar únicamente la partición necesaria de una revisión fija. Registrar
   URL, fecha, SHA-256 y licencia. Evitar cargar todos los instrumentos con pandas.
2. Conservar símbolo bruto y espacio de nombres. No quitar sufijos indiscriminadamente.
3. Comparar símbolo exacto, mercado, MIC, divisa y nombre. Una coincidencia de texto
   aislada no resuelve la entidad ni la validez temporal.
4. Registrar `matched`, `ambiguous`, `rejected` o `unresolved`, con evidencia y motivo.
   Los identificadores incompletos permanecen incompletos.
5. Contrastar la asociación con una fuente adecuada antes de promoverla. No
   modificar modalidades, cohorte o hechos históricos mediante este enriquecimiento.

Una implementación posterior puede usar Python con uv, PyArrow o Polars para
leer por bloques y escribir Parquet. No requiere un servicio, Go, CUDA ni una
biblioteca nueva para un catálogo de consulta ocasional.

## Caso dirigido de CSGS

Una consulta acotada a
[NMS.csv](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/database/equities/NMS.csv)
identificó `CSGS` como `CSG Systems International, Inc.`, con MIC `XNAS`, ISIN
`US1263491094` y `delisted=True`. La línea, incluido el salto final, tiene SHA-256
`b386eed08e6168a8d88ec0eeded2b9f62cd2262c56be45edc71e6fb2dff21663`.

Sirve para contrastar la colisión observada en una noticia sobre Credit Suisse
que usa `CSGS.N`. No autoriza a transformar ese símbolo en CSG Systems. La revisión
editorial sigue siendo necesaria. El indicador de baja no permite fechar cuándo
dejó de cotizar ni eliminar sus observaciones históricas.

## Derechos y límites

La [licencia del repositorio](https://github.com/JerBouma/FinanceDatabase/blob/d0b95bd51f9c594c81bac0b20c7ab6cbd084c51e/LICENSE)
es MIT. Deben conservarse sus avisos en copias sustanciales. No se encontró
procedencia por fila que permita acreditar las condiciones de cada atributo
agregado. No se publicará una extracción sustancial sin revisar esa procedencia.

El uso recomendado es auxiliar y presente. Reconstruir constituyentes históricos,
rellenar modalidades o atribuir disponibilidad pasada exige otras fuentes. Esta
revisión no demuestra que la integración aumente muestras válidas o precisión.
