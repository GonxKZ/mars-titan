# Contrato de datos y disponibilidad temporal

Versión de diseño: 0.1. Este contrato define la futura tabla experimental. Todavía no hay un pipeline implementado.

## Unidad de observación

Una muestra corresponde a un activo estable, un instante de decisión, una versión de entradas y un horizonte. Su clave será `(asset_id, prediction_at, horizon, data_version)`. El símbolo visible no basta como identificador si puede cambiar o reutilizarse.

| Campo | Tipo lógico | Regla |
| --- | --- | --- |
| `asset_id`, `symbol`, `exchange` | Texto | Identidad estable separada de su símbolo y mercado. |
| `event_at` | Fecha/hora UTC | Momento del hecho descrito. Puede ser anterior a su publicación. |
| `published_at` | Fecha/hora UTC opcional | Publicación original cuando esté acreditada. |
| `available_at` | Fecha/hora UTC | Primer momento utilizable según evidencia y latencia adoptada. Obligatorio para incluir el dato. |
| `ingested_at` | Fecha/hora UTC | Momento de adquisición de la copia. No se confunde con publicación histórica. |
| `prediction_at` | Fecha/hora UTC | Corte de información usado para emitir la predicción. |
| `entry_at`, `exit_at` | Fecha/hora UTC | Intervalo negociable definido por el horizonte y el calendario. |
| `label_available_at` | Fecha/hora UTC | Momento en que todas las componentes de la etiqueta ya pueden conocerse. |
| `source_id`, `source_version`, `content_hash` | Texto | Procedencia, versión y huella de la entrada. |
| `availability_rule` | Categoría | Evidencia directa, retardo conservador o exclusión justificada. |
| `modality_mask`, `missing_reason` | Máscara y categorías | Distinguir ausencia, dato inválido y exclusión por riesgo temporal. |
| `raw_return`, `market_return`, `residual_target` | Número finito | Etiquetas separadas de las columnas de entrada. Nunca accesibles al codificador antes de madurar. |
| `residualizer_cutoff`, `residualizer_version` | Fecha/hora y texto | Historia máxima y regla de ajuste del objetivo. |
| `split_id`, `feature_version` | Texto | Partición y transformación que generaron la muestra. |

## Reglas por modalidad

**Precios.** Verificar OHLC, sesiones, suspensiones, unidades, dividendos y splits. Una fecha a medianoche en un CSV diario no significa que el cierre estuviera disponible a esa hora. El cierre entra después de finalizar la sesión y del margen de publicación adoptado. Evitar usar precios ajustados retrospectivamente como niveles sin estudiar los efectos de acciones corporativas futuras.

**Noticias.** Conservar URL y evidencia temporal original. Las noticias posteriores al corte pasan a la siguiente decisión. Con fecha sin hora se esperará al cierre de la primera sesión cuya fecha sea estrictamente posterior a la del texto, con sensibilidad a un retardo mayor. Eliminar duplicados usando contenido y origen sin consultar rendimientos futuros. La relación noticia–activo necesita comprobación independiente del nombre del archivo.

**Fundamentales.** `period_end` describe el periodo contable, no la disponibilidad. Seleccionar hechos publicados antes del corte y conservar revisiones separadas por presentación. Si existe fecha de filing sin hora, aplicar el margen conservador definido. Si se conoce la aceptación y difusión efectiva, conservar esa evidencia. No rellenar hacia atrás valores de fin de trimestre ni sobrescribir el pasado con la última revisión.

**Imágenes.** Cada gráfico debe registrar la ventana de precios que lo originó, con extremo derecho menor o igual al corte. Un gráfico derivado de precios transforma información existente. No se supone que añade una fuente independiente. Las imágenes de periodo completo no se reasignan a días internos.

**Representaciones externas.** Registrar modelo, revisión de pesos/tokenizador, idioma, fecha de publicación, corpus conocido y parámetros de extracción. El hash del embedding debe depender también del texto y de la regla temporal, no solo del ticker. Los modelos de texto modernos sobre periodos antiguos se describen como evaluación retrospectiva de representaciones, salvo que se acredite disponibilidad histórica.

## Invariantes y futuros casos de prueba

- Añadir o modificar datos posteriores al corte no cambia entradas, predicciones ni estado permitido de fechas anteriores.
- Ninguna entrada incluida tiene `available_at > prediction_at`. Ninguna actualización supervisada usa una etiqueta no madura.
- Un escalador, selector, imputador o detector de régimen no se ajusta con validación, calibración o test del mismo fold.
- Los intervalos de etiqueta del ajuste no cruzan al tramo siguiente. Se purga por intervalos, no por filas aleatorias.
- Permutar el orden de activos de la misma sesión conserva sus predicciones, el estado resultante y las predicciones posteriores. Las escrituras usan el orden canónico registrado en el protocolo.
- Repetir un fold desde el mismo estado inicial reproduce el resultado dentro de la tolerancia numérica declarada.
- Los eventos duplicados no generan escrituras duplicadas por accidente. Cada actualización lleva identificador y corte.
- Una ausencia de modalidad no se convierte en un valor observado cero sin máscara y justificación.

## Formatos y almacenamiento

Mantener originales inmutables en `dataset/` o en almacenamiento externo. Usar Parquet para derivados, particiones por mercado/fecha cuando favorezcan las consultas y claves estables. Evitar un archivo diminuto por activo/día y evitar cargar todas las tablas contables en memoria. Los esquemas, manifiestos y recuentos resumidos se versionan. Los datos derivados permanecen fuera de Git.

El manifiesto deberá registrar licencia, ruta lógica, tamaño, hash, versión de esquema, rango temporal observado, columnas y exclusiones. El acceso a una muestra debe poder explicarse desde la fuente hasta la característica final, incluido el tratamiento de faltantes.
