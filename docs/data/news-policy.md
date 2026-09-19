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

## Comprobación ejecutada

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

Las dos páginas contrastadas mediante consulta pública no estuvieron accesibles.
No se atribuye a esas consultas una verificación de su contenido histórico. Los
originales se conservan y no se reconstruye texto por suposición. La decisión de
usar artículos completos comprobables o estudiar titulares verificables está
pendiente. Por ese motivo, #7 no se cierra con esta entrega y los informes marcan
la validación semántica y de versión histórica como incompletas.

## Reproducción

Usar nombres nuevos para no sustituir otra auditoría:

```bash
uv run --locked python -m mars_titan.data.news_audit \
  --panel data/manifests/pilot.json \
  --output data/interim/news-reproduction \
  --report data/interim/news-reproduction-report.json
uv run --locked pytest tests/data/test_news_availability.py tests/data/test_news_audit.py
```

La auditoría actual procesa un activo cada vez, pero conserva sus registros en
memoria. No equivale a una conversión incremental del corpus entero. Las pruebas
no certifican relevancia predictiva ni ausencia de revisiones de texto no
documentadas por el proveedor.

## Verificación de esta entrega

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
