# Ficha inicial de FinMultiTime

Estado: inspección documental y estructural, **no auditoría completa de los 109 GiB**. Fecha de revisión: 18 de septiembre de 2026.

## Procedencia y versión

Referencia principal: Xu et al., [FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis](https://arxiv.org/abs/2506.05019v2). El artículo describe precios, texto, tablas e imágenes. Su [revisión crítica](../references/neural-review.md) distingue lo descrito por los autores de lo comprobado en esta copia.

La carpeta `dataset/` existía antes de preparar el repositorio. No contiene una revisión de Hugging Face fijada ni un manifiesto completo que permita acreditar la descarga original. Su README declara MIT y cita una versión del artículo. Debe registrarse la revisión de origen cuando se reconstruya la adquisición. No se inventa ese identificador.

## Inventario observado localmente

| Elemento | Observación comprobada | Límite de interpretación |
| --- | --- | --- |
| Tamaño total | `du -sh dataset`: aproximadamente 109 GiB. | Espacio ocupado, no tamaño lógico exacto ni medida de cobertura. |
| Series | Aproximadamente 2,2 GiB, 5.023 archivos CSV entre ambos mercados. | Un archivo no garantiza una serie válida, distinta o completa. |
| Texto | Aproximadamente 14 GiB, 5.586 archivos JSONL. | No equivale a 5.586 activos con noticias útiles. |
| Tablas | Aproximadamente 85 GiB. | Es necesario inspeccionar duplicaciones y versiones antes de cargarlas completas en RAM. |
| Imágenes | Aproximadamente 8,5 GiB. | Las ventanas y fechas deben comprobarse. No basta con asociarlas al ticker. |
| CSV descriptivo estadounidense | 4.213 registros leídos como CSV. | Son filas del catálogo. No una lista histórica auditada del S&P 500. |
| CSV descriptivo chino | 859 líneas incluida la cabecera, decodificables con GB18030. Falla lectura UTF-8 directa. | La codificación y el número de registros estructurados requieren fijarse en la auditoría. |
| Precio AAPL | Campos `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`. Primeras filas observadas de enero de 2000 y fechas con desplazamiento horario. | El nombre de la columna no documenta el ajuste por dividendos/splits ni el momento de disponibilidad. |
| Noticias AAPL | Primer registro muestreado contiene `Date`, `Url` y `Article`, y habla de Shiba Inu. | La carpeta del activo no garantiza relevancia semántica. Revisar etiquetado y entidades. |
| Tabla AAPL | Ejemplo con `filing_date` exterior y hechos internos que conservan `end`, `filed` y `accn`, incluidos años anteriores y revisiones. | No aplicar la fecha exterior ni la de cierre contable a todos los hechos internos. |
| Proxy de mercado | Existe `dataset/time_series/S&P500_time_series/spy.csv`. | No se han auditado cobertura, ajustes ni coherencia con cada activo. |

Las cifras del artículo, de la ficha pública y de los catálogos locales no son idénticas. Hay que explicar diferencias de versión, universo y modalidad antes de publicar un tamaño de muestra. Tampoco deben equipararse instrumentos de una carpeta denominada S&P500 con sus constituyentes históricos.

## Decisiones necesarias antes de preparar muestras

1. Fijar el origen y una instantánea local. Producir manifiestos parciales por modalidad y hashes de los archivos utilizados, sin exigir releer 109 GiB en cada ejecución.
2. Definir calendario, zona horaria, ajustes corporativos, unidades y significado de cada campo.
3. Separar identificador estable de activo, símbolo y pertenencia a mercado/sector. Registrar cambios y bajas cuando existan.
4. Estimar cobertura por activo y fecha sin seleccionar por el futuro de la ventana evaluada.
5. Reconstruir hechos contables por `accn`/`filed` y versión. Excluir los que no permitan una fecha defendible.
6. Revisar relevancia de noticias y duplicados. Usar un retardo conservador para fechas sin hora.
7. Regenerar gráficos a partir de ventanas históricas cerradas. Evitar reutilizar resúmenes que abarcan el futuro del instante de predicción.
8. Comprobar derechos de noticias y datos subyacentes. La etiqueta MIT de un catálogo no acredita todos esos derechos.

## Estado y uso permitido en esta preparación

No se han normalizado datos, creado muestras de entrenamiento, seleccionado activos por rendimiento ni entrenado modelos. La copia existente permanece intacta y fuera de Git. El siguiente paso científico será auditarla siguiendo el [contrato temporal](data-contract.md), con resultados de calidad separados de las suposiciones.
