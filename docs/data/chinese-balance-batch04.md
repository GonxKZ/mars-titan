# Cuarto lote de balances chinos

El lote añade doce empresas, 72 hechos contables revisados y 1.676 muestras
multimodales. La unión china pasa de las 37 empresas del
[tercer lote](chinese-balance-batch03.md) a 49, con 9.596 muestras. Sus diez
ventanas utilizan 9.516 filas distintas y 5.087 filas de evaluación sin
repeticiones entre meses. La cobertura sigue parcial.

El [recibo de la ampliación](../../reports/data/chinese-balance-batch04-20261007.json)
conserva las huellas de los documentos, las conciliaciones y los manifiestos.
La ejecución CUDA ocurrió antes de la pausa posterior para revisar la
cobertura histórica. Esta entrega publica sus datos y comprobaciones, sin
iniciar entrenamientos ni ejecutar de nuevo los codificadores.

## Documentos y cifras

Se seleccionaron los siguientes informes completos de 2021 entre los candidatos
observados en dos catálogos parciales de CNINFO. El orden por símbolo y la
primera publicación observada se fijaron antes de la descarga. No se acredita
que sean las primeras versiones históricas ni una muestra representativa del
mercado. El lote operativo usa emisores de Shenzhen, sin excluir del censo los
candidatos de Shanghái.

| Emisor | Publicación | Hechos admitidos | Muestras añadidas | Documento |
| --- | --- | ---: | ---: | --- |
| 000008.SZ | 2022-04-29 | 6 | 54 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-29/1213213656.PDF) |
| 000036.SZ | 2022-04-29 | 6 | 116 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-29/1213201697.PDF) |
| 000100.SZ | 2022-04-28 | 6 | 286 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-28/1213177032.PDF) |
| 000156.SZ | 2022-04-29 | 6 | 94 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-29/1213210132.PDF) |
| 000400.SZ | 2022-04-26 | 6 | 175 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-26/1213108981.PDF) |
| 000410.SZ | 2022-04-16 | 6 | 48 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-16/1212943999.PDF) |
| 000415.SZ | 2022-04-30 | 6 | 82 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-30/1213265873.PDF) |
| 000420.SZ | 2022-05-28 | 6 | 102 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-05-28/1213526742.PDF) |
| 000422.SZ | 2022-04-09 | 6 | 217 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-09/1212858037.PDF) |
| 000425.SZ | 2022-04-19 | 6 | 221 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-19/1212958957.PDF) |
| 000498.SZ | 2022-04-26 | 6 | 143 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-26/1213116566.PDF) |
| 000503.SZ | 2022-04-30 | 6 | 138 | [Informe 2021](https://static.cninfo.com.cn/finalpage/2022-04-30/1213263840.PDF) |

Los hechos corresponden a activos, pasivos y patrimonio total incluidos los
minoritarios, para los cierres de 2021 y 2020. Se contrastaron el perímetro
consolidado, CAS, CNY, unidades y columnas. Las 72 cifras originan 114
coincidencias de campo con los registros originales, incluidas sus versiones
duplicadas. Hay 27 coincidencias Decimal exactas y 87 bajo la política explícita
`binary64_roundtrip`. Esta exige la misma representación binaria, precisión
publicada de al menos dos ULP y redondeo decimal al importe publicado. No se
aplica una tolerancia relativa general. Ningún hecho de este lote quedó
excluido.

Cuatro casos requieren conservar estos detalles:

- `000100.SZ` no tiene texto extraíble. Sus balances se transcribieron desde
  imágenes y están expresados en miles de CNY. El resumen menos el balance da
  +1.305 CNY en activos de 2021 y −113 CNY en activos de 2020. Se usan los
  importes del balance, con doce coincidencias Decimal exactas en los originales.
  No se corrige la fuente ni se atribuye una causa a esas diferencias.
- `000415.SZ` también publica en miles de CNY. Se aplica el multiplicador 1.000.
- `000410.SZ` presenta patrimonio total negativo en 2021. Se conserva el signo.
- `000420.SZ` tiene minoritarios negativos en 2021 y publicación el 28 de mayo
  de 2022. La disponibilidad queda en el cierre de la siguiente sesión CN más
  cinco minutos, el 30 de mayo a las 07:05 UTC.

Las notas que introducen ajustes de arrendamientos en la apertura de 2021 se
mantienen separadas del cierre de 2020. Los comparativos conservan la fecha de
publicación del documento de 2022 y no se adelantan al periodo al que se refieren.

La revisión inicial conserva 84 páginas visuales. Una revisión independiente
volvió a renderizar e inspeccionar 72 páginas pertinentes, con píxeles idénticos,
y verificó por hash las 84 imágenes originales. Recalculó las igualdades
contables, las coincidencias Decimal y binary64, la publicación y las 72 filas
Parquet sin llamar al conciliador de producción. Las fuentes permanecieron
intactas.

## Codificación anterior a la pausa

MiniLM y ResNet18 mantienen la identidad de los codificadores US. La comprobación
CUDA validó las 1.676 muestras nuevas, las cuatro modalidades y las 140 máscaras
macro. Los vectores contables coinciden con una referencia Decimal de 50 dígitos
convertida a `float32`. La recuperación reutiliza las salidas sin nueva inferencia.

La ventana comenzó el 7 de octubre a las 17:15:45 UTC y duró 66,38 segundos.
El proceso de codificación y comprobación tardó 56,54 segundos y reservó como
máximo 612.368.384 bytes de VRAM. La campaña US se reanudó al terminar esa ventana.
Posteriormente quedó pausada de forma recuperable para revisar la cobertura
histórica desde 2000. El estado verificado a las 17:26 UTC conserva 511
operaciones terminadas, con el servicio deshabilitado y sin reanudación
automática. No se ejecutó otra carga GPU para preparar esta publicación.

Son medidas de una ejecución funcional. No miden aceleración, energía ni
transferencias CPU/GPU, ni aportan resultados predictivos.

La partición anual reúne 3.305 filas de entrenamiento y 6.248 de validación.
Las diez ventanas mantienen el protocolo US con calendario CN y sus cortes
propios. La mayor partición de entrenamiento tiene 7.491 filas. Las cuarenta
particiones se recorrieron y conservan 9.516 muestras distintas y 5.087 filas
de evaluación sin repeticiones. La unión mantiene los bytes anteriores y su
recuperación no reescribe los manifiestos confirmados.

Pasan 206 pruebas CPU en dos invocaciones, 173 de fuentes, hechos, preparación,
historia, muestras y unión, y 33 del lector CNY. No se modifica lógica de
producción ni se presenta una nueva medición de cobertura o mutación. El test
final de 2024 sigue cerrado.

## Captura de anuncios separada

La [consulta por emisor](chinese-announcements.md) también se comprobó sobre los
37 emisores admitidos antes de este lote. Se conservaron 154 anuncios con IDs
únicos, a partir de 36 POST nuevos y una respuesta reutilizada del piloto.
Los cuerpos nuevos suman 96.843 bytes y el intervalo mínimo entre inicios fue
de 5,427795 segundos. Los 83 anuncios coincidentes con los catálogos anteriores
mantienen sus metadatos. Las 37 capturas se recuperaron sin red ni cambios de
hash o fecha de modificación.

Las consultas cubren la categoría anual y fechas de publicación de 2022–2023.
Los títulos señalan 77 candidatos de 2021 y 77 de 2022, incluidos resúmenes,
versiones en inglés y correcciones. Estos anuncios no son hechos admitidos ni
acreditan cobertura completa de los cierres o del censo. Este resumen describe
los 37 emisores anteriores, no las 49 empresas de la nueva edición contable.

La [unión US+CN anterior](../engineering/joint-market-campaigns.md) conserva sus
25 empresas chinas. Incorporar esta ampliación requiere congelar otra población
conjunta. Siguen pendientes otros emisores, cierres y versiones documentales.
El candidato base MARS-TITAN está en implementación y todavía no se ha entrenado.
