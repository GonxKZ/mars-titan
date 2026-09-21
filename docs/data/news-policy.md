# Admisión y revisión de noticias

La preparación distingue una noticia que cumple el contrato de campos y fechas
de una noticia cuyo contenido histórico ha sido verificado. No son garantías
equivalentes. La revisión del contenido sigue abierta por las incoherencias
observadas en la copia original.

## Reglas implementadas

Cada registro necesita texto, publicación y un `Stock_symbol` explícito que
coincida con el activo. El nombre del archivo no sustituye ese campo. Se conserva
la normalización conocida de los sufijos chinos `.SH` y `.SS`, sin inventar
correspondencias entre empresas o cambios de símbolo.

Una fecha `YYYY-MM-DD` no adquiere una hora ficticia. `event_at` y su alias
compatible `published_at` permanecen desconocidos. `available_at` se sitúa en el
cierre de la siguiente sesión estrictamente posterior, con el margen de cinco
minutos ya definido. El retardo alternativo de dos sesiones permite comprobar
sensibilidad. Ambos usan el calendario existente.

Una publicación con hora necesita segundos explícitos y zona `Z` o desplazamiento
`±HH:MM`. Se convierte a UTC. Una hora sin zona o con desplazamiento desconocido
`-00:00` queda excluida. También se excluyen precisiones que no pueden conservarse
con los microsegundos del contrato, en vez de redondear información futura hacia
una decisión anterior.

Los minutos del desplazamiento deben estar entre 00 y 59 y las horas entre 00 y
23. Una fecha que desborda el rango al convertirse a UTC se excluye como registro
inválido, sin interrumpir la lectura del archivo.

`event_at` representa publicación cuando está acreditada en los campos de la
fuente. No se presenta como fecha de ocurrencia del hecho económico descrito.
El idioma se conserva cuando lo declara la fuente. Si falta, queda desconocido.
No se traduce ni se infiere a partir del mercado.

## Identidad y conservación

`event_id` identifica activo, texto normalizado, URL y publicación normalizada.
Dos horas que expresan el mismo instante son equivalentes. Una revisión con otro
contenido o publicación no sustituye a la anterior. El identificador no cambia
por elegir un retardo distinto para disponibilidad.

Cada registro conserva archivo, número de registro y SHA-256 de su línea
decodificada, sin terminador de línea. El archivo completo mantiene otra huella.
Esto permite revisar una exclusión sin copiar artículos al repositorio. El texto
normalizado no sustituye el original. Los duplicados entre archivos también se
identifican durante la auditoría del panel.

Los motivos incluyen `missing_symbol_evidence`, `symbol_mismatch`,
`missing_text`, `missing_publication`, `unverified_timezone`,
`unsupported_timestamp_precision`, `invalid_record`, `duplicate` y
`duplicate_across_files`. Son identificadores estables del contrato.

## Auditoría anterior de campos y fechas

| Panel | Registros activo-fuente | Admitidos por campos y tiempo | Duplicados excluidos |
| --- | ---: | ---: | ---: |
| Piloto de cuatro activos | 4.367 | 4.357 | 10 |
| Panel técnico de 22 activos | 29.957 | 29.890 | 67 |

Los recuentos de admisión coinciden con los derivados previos en estos paneles.
Todos los registros admitidos declaran símbolo y tienen fecha sin hora. El paso
de una a dos sesiones desplaza su disponibilidad entre 21 y 120 horas según el
calendario, no siempre 24. No es una medición de efecto predictivo.

Los informes [del piloto](../../reports/data/news-audit-pilot.json) y
[del panel técnico](../../reports/data/news-audit-us.json) registran fuentes,
calendario, código, exclusiones, tiempo y memoria. Los Parquet de noticias y
exclusiones se conservan localmente en directorios separados. La orden rechaza
destinos existentes y evita sobrescribir fuentes, el panel o las particiones
mediante la ruta del informe.

## Problemas de contenido observados

Se seleccionó un registro anterior a 2019 por activo del piloto, mediante el
menor SHA-256 de `42:event_id`. Esa selección pequeña comprueba ejemplos, no
estima la tasa de errores de todo el corpus.

| Registro local | Observación | Interpretación permitida |
| --- | --- | --- |
| MNST, línea 709, 2018-12-10 | El texto describe resultados y evolución de Monster Beverage. | Relación directa observada en este caso. |
| CSGS, línea 268, 2014-06-06 | Título y cuerpo describen el dividendo de CSG Systems. | Relación directa observada en este caso. |
| DECK, línea 1208, 2018-08-03 | El artículo se centra en Prestige Brands y menciona Deckers entre otras acciones. | Mención secundaria, no un evento exclusivo del activo. |
| ABM, línea 816, 2014-12-09 | El título y la URL anuncian dividendos, pero el cuerpo describe empresas sanitarias ajenas y contiene un copyright de 2016. | Incoherencia que impide dar el cuerpo por verificado para esa noticia. El copyright por sí solo no fecha su publicación. |

La revisión dirigida de AAPL, línea 1, fecha 2025-04-18, encuentra un artículo
sobre Shiba Inu que compara su capitalización con Apple. El ticker declarado
acredita una etiqueta del proveedor, no relevancia económica específica.

Las primeras consultas de ABM y DECK no permitieron recuperar sus páginas. El
contraste posterior encontró fuentes editoriales accesibles y otras contradicciones.
No se atribuye a una consulta fallida la verificación de un texto.

## Admisión estricta de artículos completos

La preparación usa únicamente cuerpos editoriales completos contrastados. No se
sustituye un cuerpo ausente por el titular o por un resumen. El [registro de
revisiones](../../data/manifests/news-reviews.json) liga cada decisión a la huella
de la línea original, el activo, la fecha, la URL y la evidencia consultada.

Se contrastaron ocho registros, una muestra dirigida que no estima la tasa de
error del corpus. Dos artículos de MNST coinciden con sus originales editoriales
de [diciembre](https://www.fool.com/investing/2018/12/10/why-monster-beverage-shares-rose-13-last-month.aspx)
y [noviembre de 2018](https://www.fool.com/investing/2018/11/08/why-monster-beverage-corp-stock-fell-today.aspx).
Se conservan todos los párrafos editoriales y las declaraciones de posiciones del
autor y del medio. Los intervalos de caracteres excluyen únicamente promoción y
avisos del distribuidor. El texto seleccionado mantiene su propio SHA-256 y no
se reescribe.

En el artículo de noviembre hay una discrepancia AM/PM entre los medios. Se
conserva la fecha sola del original y el retardo por sesiones. No se promociona
ninguna de esas horas a una publicación exacta.

Dos registros de ABM son incoherentes. El de 2014 incluye un hito de Acasti que
figura en un [comunicado de diciembre de 2018](https://www.globenewswire.com/news-release/2018/12/31/1679176/0/en/acasti-pharma-announces-trilogy-phase-3-trials-of-capre-in-patients-with-severe-hypertriglyceridemia-has-now-exceeded-65-randomization-and-more-than-100-patients-20-have-completed-.html).
El registro 599 de 2018 contiene empresas de cannabis, mientras que su
[página original](https://www.nasdaq.com/articles/abm-industries-abm-beats-q4-earnings-revenue-estimates-2018-12-19)
describe resultados de ABM. No se reparan esos cuerpos ni se cambia su fecha.
Los otros cuatro registros revisados siguen sin contraste completo.

Los motivos estrictos son `content_unreviewed`, `content_unverifiable`,
`content_rejected`, `content_review_mismatch` y `missing_full_article`. Que una
noticia no esté revisada no significa que sea falsa. Se mantiene fuera hasta
que exista evidencia suficiente.

La correspondencia editorial actual no acredita una instantánea histórica
inmutable. Los informes mantienen ese límite explícito. Además, filtrar por
páginas que siguen accesibles introduce una selección retrospectiva que no debe
confundirse con una regla disponible en tiempo real.

## Cobertura estricta observada

La comprobación inicial de este apartado se conserva como antecedente. La
[ampliación posterior](../../reports/data/news-coverage-expansion.md) alcanza 13
artículos contrastados y 65 muestras de dos activos. La limitación de cobertura
para la comparación principal sigue vigente.

La [auditoría estricta](../../reports/data/news-audit-strict-pilot.json) recorre los
4.367 registros del piloto. Admite dos y excluye 4.365. El [recorrido multimodal](../../reports/data/strict-multimodal-coverage.json)
produce diez muestras de MNST con las cuatro modalidades y macro. ABM, DECK y
CSGS no producen muestras estrictas con este registro de revisiones.

No es una cohorte suficiente para comparar modelos. #7 sigue abierta para ampliar
la verificación y fijar cobertura. Las sondas anteriores se conservan como
mediciones de coste con su preparación original, sin atribuirles ahora validación
editorial retroactiva. No se ha entrenado una campaña nueva con estas diez muestras.

## Reproducción

Usar nombres nuevos para no sustituir otra auditoría:

```bash
uv run --locked python -m mars_titan.data.news_audit \
  --panel data/manifests/pilot.json \
  --reviews data/manifests/news-reviews.json \
  --output data/interim/news-reproduction \
  --report data/interim/news-reproduction-report.json
uv run --locked mars-data prepare --panel data/manifests/pilot.json \
  --news-reviews data/manifests/news-reviews.json \
  --output data/interim/strict-preparation-reproduction
uv run --locked pytest tests/data/test_news_availability.py \
  tests/data/test_news_audit.py tests/data/test_news_reviews.py
```

La auditoría actual procesa un activo cada vez, pero conserva sus registros en
memoria. No equivale a una conversión incremental del corpus entero. Las pruebas
no certifican relevancia predictiva ni ausencia de revisiones de texto no
documentadas por el proveedor.

Los comandos usan el registro estricto por defecto. `--fields-only` en la auditoría
y `--unreviewed-profile` en la preparación permiten repetir diagnósticos de campos
o coste, con una etiqueta explícita de contenido no revisado. No habilitan
entrenamiento científico ni convierten la salida en un conjunto verificado.

El selector del piloto exige la misma política y transmite el registro de
revisiones a cada activo. Un registro vacío no aporta cobertura. La selección
estricta actual no reúne cuatro activos ni el mínimo de 252 muestras, por lo
que se detiene antes de modificar el piloto anterior. #12 vuelve a depender de
la ampliación de cobertura verificada en #7.

## Verificación de la admisión estricta

La suite completa pasó con 394 pruebas, incluida la extracción real en CUDA.
Las pruebas recorren la preparación estricta, rechazos, cambios de hash,
intervalos inválidos, ausencia de cuerpos, configuración incorrecta y selección
del piloto. Una integración con datos sintéticos reconstruye el mismo piloto
usando los adaptadores reales y cobertura anterior al corte.

En los cuatro módulos de noticias y selección, coverage.py 7.16.1 registró 290 de
313 sentencias y 112 de 132 ramas cubiertas, un 90,34 % combinado. Radon 6.0.1
midió complejidad 12 y CRAP 12 en `reviewed_body`, con cobertura de sentencias
del 100 %. El selector tiene complejidad 27, cobertura del 89,47 % y CRAP 27,850,
calculado con `C² × (1 − cobertura)³ + C`. Los casos no cubiertos siguen siendo
un límite de la verificación.

Cuatro mutaciones dirigidas se detectaron: omitir la huella del cuerpo, admitir
un recorte vacío, saltarse el registro en el comando de preparación y dejar de
transmitirlo al selector. No es una campaña exhaustiva.

La segunda comprobación real mantuvo diez muestras y no amplió las doce entradas
de la caché existente. Los campos `reviews_sha256` de ambos informes identifican
el índice de revisiones serializado con claves ordenadas. El informe multimodal
conserva además `reviews_file_sha256` para los bytes del manifiesto. No deben
confundirse ambas huellas.

## Verificación de la entrega anterior de fechas

La suite local completa pasó con 344 pruebas. En `news.py` y `news_audit.py`,
coverage.py 7.16.1 midió 155 de 169 sentencias y 57 de 66 ramas cubiertas.
La cobertura combinada fue del 90,21 %. Radon 6.0.1 midió complejidad 31 en
`audit_news_panel`. Su CRAP fue 31,458 usando cobertura de sentencias de esa
función y la fórmula `C² × (1 − cobertura)³ + C`. La auditoría concentra varias
validaciones y sigue siendo el punto más complejo de esta incorporación.

Tres mutaciones dirigidas se detectaron con pruebas fallidas: eliminar el control
del rango del desplazamiento, dejar escapar el desbordamiento UTC y admitir el
nombre del archivo como sustituto del símbolo. Se aplicaron en memoria, sin
cambiar archivos de producción. No constituyen una campaña exhaustiva.

El identificador `availability_rule` conserva `next_session_close` también cuando
se solicita un retardo de dos sesiones. El informe identifica esa sensibilidad,
pero antes de entrenar con una tabla alternativa deberá guardarse el retardo
explícito por registro. Las tablas publicadas por esta auditoría usan una sesión.
