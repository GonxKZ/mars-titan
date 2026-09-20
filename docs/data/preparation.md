# Preparación multimodal reproducible

La preparación produce entradas con precios, noticias, fundamentales y gráficos.
El contexto macro es adicional y obligatorio. No se sustituye una modalidad
ausente por un vector de ceros. Las ausencias dentro de una modalidad se distinguen
de los valores observados mediante máscaras y antigüedad.

## Qué se ha comprobado

La copia original contiene 216.453 archivos y 116.185.016.265 bytes, unos 108,2 GiB.
Dos pasadas de hashes confirman la misma instantánea de contenido, identificada por
`f206a659be9e19f2cd6913488843c480c06808ce089c19cef95cd62d672c1619`.
No se ha modificado ningún original. Hay 27.252 archivos adicionales cuyo contenido
coincide con otro archivo. Se registran, pero no se eliminan.

| Comprobación | Cobertura ejecutada | Resultado |
| --- | --- | --- |
| Inventario y cabeceras | Todos los archivos | Sin errores de lectura. Inventariar no equivale a validar todo el contenido. |
| Precios estadounidenses | 4.213 archivos, 17.448.737 filas | 17.385.328 filas admitidas por las reglas OHLCV y calendario. |
| Precios chinos | 810 archivos, 3.139.412 filas | 3.133.858 filas admitidas. Los motivos de exclusión pueden solaparse. |
| Publicación de fundamentales chinos | 2.430 archivos, 202.767 registros | Ninguno acredita publicación. No se admite entrenamiento chino. |
| Normalización detallada por activo | Paneles técnicos de 22 activos estadounidenses y 62 chinos | Derivados locales y manifiestos reanudables. |
| Intersección multimodal estadounidense | Panel técnico de 22 activos | 25.856 muestras con cuatro modalidades y macro. |

Los recuentos y tiempos están en [los informes de datos](../../reports/data/).
No se ha normalizado toda la modalidad textual o contable estadounidense. El
inventario sí cubre el corpus completo. La implementación permite ampliar el
panel por bloques, conservando las mismas comprobaciones.

## Reloj y reglas de admisión

Las decisiones se sitúan cinco minutos después del cierre real de la sesión.
Se usan calendarios XNYS y XSHG versionados, con festivos y cierres abreviados.
XNYS es una aproximación del calendario estadounidense del archivo, no una
reconstrucción certificada del mercado de cotización de cada instrumento.

Las noticias con fecha sin hora se admiten al cierre de la siguiente sesión.
Se conserva el texto original, su huella, URL y relación declarada con el símbolo.
Esa relación no demuestra por sí sola relevancia semántica. Las noticias del
panel se agrupan en una ventana pasada de cinco sesiones. No se inventan noticias
para cubrir días sin información.

Los fundamentales estadounidenses usan `filed` y `accn` de cada hecho interno.
No se asigna la fecha exterior del archivo a todas sus cifras. Los valores
ambiguos de un mismo concepto, periodo y publicación se excluyen de la instantánea.
La representación actual contiene ocho conceptos de balance en USD, sus máscaras
y su antigüedad. No equivale a utilizar todos los conceptos XBRL existentes.

Los gráficos se regeneran con 64 sesiones consecutivas válidas y terminan en la
decisión. Una fila de precios excluida rompe la ventana, no se salta para reunir
64 filas cualesquiera. La imagen transforma los mismos precios, no constituye una
fuente económica independiente.

Los precios ajustados retrospectivamente se conservan tal como se distribuyeron.
La representación de ventana es invariante a un factor multiplicativo uniforme
de precios o volumen. Esto no reconstruye todas las acciones corporativas ni
convierte el archivo en una simulación histórica de ejecución negociable.

## Macro y representaciones

Se adquirieron 59 series completas de ALFRED, incluidas las 20 principales del
catálogo. Las 70 fórmulas están implementadas. En el intervalo preparado, 125 de
los 140 indicadores tienen al menos un valor numérico. Los restantes conservan
una causa de ausencia, no una cifra inventada.

Cada revisión mantiene su periodo, intervalo de vigencia, unidad histórica y
evidencia de origen. No se etiquetan todas las versiones del PIB como dólares de
2017. Las fórmulas rechazan combinaciones de unidades o ajustes incompatibles.
Los cambios de base siguen siendo una limitación al comparar niveles nativos.
No se ha realizado un reescalado retrospectivo con información posterior.

WTI y Brent quedan excluidos porque un valor está fechado después del inicio de
disponibilidad declarado. Falta evidencia para reconciliar esas fechas y no se
corrigen por suposición.
GSCPI y los índices que requieren versiones del método conservan sus restricciones.
Los candidatos sin identificador no se descargan. La granularidad es la publicación
y revisión disponible, alineada después a cada decisión diaria. No se crea
información intradía a partir de datos mensuales.

MiniLM procesa el texto completo por fragmentos de 126 tokens más tokens especiales.
ResNet18 procesa gráficos de 224 × 224 sin recortar sus extremos. Ambos permanecen
congelados. La caché identifica contenido, revisión, pesos y transformación. Las
representaciones modernas se consideran retrospectivas, no una prueba de que esos
codificadores estuvieran disponibles en todas las fechas del archivo.

Los vectores se almacenan como listas de tamaño fijo en `float32`. Las ventanas de
precios se construyen al consumir cada muestra. No se materializa un tensor con
todas las ventanas del corpus. La lectura optimizada conserva paridad exacta en las
25.856 muestras verificadas. En una medición con caché del sistema caliente pasó
de unos 2,6 segundos a unos 0,7 segundos, sin GPU.

## Reproducción y recuperación

Ejecutar desde la raíz del repositorio:

```bash
uv sync --locked --extra cuda --extra encoders
uv run mars-data inventory --report reports/data/full-source-inventory.json
uv run mars-data audit-prices
uv run mars-data select --market US --output data/interim/panel-us.json
uv run mars-data prepare --panel data/interim/panel-us.json
uv run mars-data macro --market US
uv run --extra cuda --extra encoders mars-data encode --panel data/interim/panel-us.json
```

La orden macro necesita una adquisición local previa. El procedimiento de descarga
oficial está en [el catálogo macro](macro-catalog.md). Los originales, ZIP,
Parquet, cachés y checkpoints permanecen fuera de Git. Véanse los
[permisos](permissions.md) y el [contrato de archivos preparados](schema.md).

El inventario confirma archivos individualmente. La preparación confirma un
manifiesto después de escribir y comprobar sus artefactos. La caché confirma cada
representación. Una interrupción puede obligar a reconstruir el Parquet de un
activo, pero no a volver a extraer sus representaciones válidas. Los hashes
impiden reutilizar artefactos alterados.

El panel técnico se elige por sector actual y volumen para medir cargas. No se
presenta como una cohorte histórica válida para extraer conclusiones. El piloto
de verificación usa otra regla, con cobertura anterior a un corte fijo y sin
exigir supervivencia posterior. El universo original sigue siendo retrospectivo.

El [manifiesto del piloto](../../data/manifests/pilot.json) conserva la selección
de coste anterior al contraste editorial, con MNST, ABM, DECK y CSGS. No acredita
la cobertura estricta actual. La regla usa un orden por hash con semilla
42 y exige al menos 252 muestras completas anteriores al 31 de diciembre de 2018.
No usa sectores actuales, volumen futuro ni permanencia hasta el final del archivo.
Se inspeccionaron 12 candidatos en ese orden para reunir los cuatro. No se confunde
este piloto con el panel técnico de 22 activos ni con la futura muestra confirmatoria.

```bash
uv run python -m mars_titan.data.universe
```

El selector actual exige el registro de [noticias completas contrastadas](news-policy.md)
y lo propaga a la preparación. Con las dos noticias verificadas actuales se
detiene antes de escribir un piloto de cuatro activos. No rebaja el mínimo de
252 muestras para aparentar una cohorte suficiente ni sobrescribe el manifiesto
anterior al detectar esa falta de cobertura. La selección no revisada queda
reservada a un diagnóstico explícito con `unreviewed_profile=True` en la API,
etiquetado como tal.
