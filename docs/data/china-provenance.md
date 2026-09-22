# Procedencia contable del mercado chino

La copia local conserva 810 instrumentos chinos con las cuatro fuentes. Sus
tablas contables no incluyen una publicación acreditada para cada cifra. El
cierre del periodo no permite decidir cuándo podía usarse un dato.

Se ha comprobado una consulta pública de anuncios de CNINFO, sin credenciales,
con los campos que utiliza su propia página de búsqueda. El anuncio `1216072952`
identifica el [informe anual de 2022 de Ping An Bank, código 000001](https://static.cninfo.com.cn/finalpage/2023-03-09/1216072952.PDF).
No se confunde con el resumen que publicó su matriz bajo el código `601318`.

Las páginas físicas 136 y 137, numeradas 7 y 8 dentro del anexo contable, contienen
el balance consolidado en millones de yuanes. Para 2022 muestran activos de
5.321.514, pasivos de 4.886.834 y patrimonio total de 434.680. La columna de 2021
contiene 4.921.380, 4.525.932 y 395.448. Las seis cifras coinciden con los datos
locales después de multiplicar por 1.000.000. [Informe original](https://static.cninfo.com.cn/finalpage/2023-03-09/1216072952.PDF).

El [recibo del contraste](../../reports/data/china-pingan-reconciliation-20260922.json)
conserva las huellas, páginas y posiciones de los registros. Cada cifra aparece
en dos filas locales con distinto `update_flag`. Esa duplicación no crea dos
hechos económicos independientes ni fecha las versiones del original.

## Función de reconciliación

`reconcile_chinese_fact` compara un registro con una revisión explícita del
documento y de su anuncio. No certifica un PDF por reconocer el formato de su
huella. La revisión de la fuente se realiza antes y se conserva como entrada.

La función comprueba instrumento, periodo, moneda declarada, unidad, presentación
y correspondencia exacta del valor. Usa `Decimal` para convertir la escala sin
introducir una tolerancia que oculte diferencias. La precisión de la operación
se calcula a partir de los dígitos de sus operandos. Si el original declara
una norma o un intervalo incompatible, la coincidencia del valor no lo corrige.
Conserva una taxonomía propia
`cn-reported`, sin afirmar equivalencia completa entre normas chinas y
estadounidenses.

La entrada original de este contrato expresa importes en yuanes. No es un
conversor general de tablas de terceros ni deduce una moneda ausente por
coincidencia numérica. El contexto devuelto procede de la revisión primaria y
mantiene explícito el límite sobre los metadatos originales.

Patrimonio atribuible a la matriz y patrimonio que incluye participaciones no
dominantes son conceptos diferentes. Un saldo no tiene el mismo intervalo que
un flujo. Los flujos requieren inicio y fin del periodo. Si el original declara
una presentación individual, una cifra consolidada no puede sustituirla.

Las fechas previstas no acreditan publicación. Una fecha sin hora se conserva
como fecha, sin inventar medianoche. Si existe un instante explícito, se exige
zona horaria y se comprueba su correspondencia con el día del anuncio en China.
La disponibilidad de entrenamiento todavía debe pasar por el calendario y la
barrera temporal comunes.

La salida `reconciled` significa que el valor coincide bajo el contexto de la
revisión primaria. `original_context_recovered` permanece en `false`. No afirma
recuperar todos los metadatos eliminados del original ni admite automáticamente
una muestra multimodal. Los valores comparativos de 2021 extraídos de este
informe mantienen su publicación de 2023. No se les atribuye una divulgación
anterior que no se haya comprobado.

## Alcance pendiente

Esta entrega contiene un contrato de contraste y un caso real, no un descargador
contable universal. Faltan la recuperación por todas las empresas y periodos,
el lector de hechos admitidos y la materialización general en Parquet.

La batería completa pasa 961 pruebas sin omisiones, incluidas 39 específicas del
contrato. Se detectan seis mutaciones dirigidas. El [recibo de calidad](../../reports/resources/china-reconciliation-quality.json)
registra la cobertura, complejidad, convención de CRAP y límites de estas comprobaciones.

Las noticias chinas conservan resúmenes sin la procedencia necesaria para admitir
cuerpos completos. La [interfaz de noticias de Tushare](https://tushare.pro/document/2?doc_id=143)
declara permisos específicos, independientes de los puntos. No se presupone
disponer de ese acceso ni se contratan servicios. La vía pública de CNINFO
resuelve documentos contables concretos, no toda la modalidad textual.

También siguen pendientes el factor de mercado chino con apertura y cierre,
la armonización de conceptos y las series nacionales del contexto macro cuya
publicación histórica no esté acreditada. El brazo conjunto no se considera
completo utilizando únicamente datos estadounidenses.
