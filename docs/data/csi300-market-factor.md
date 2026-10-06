# Factor retrospectivo CSI 300

El factor chino utiliza la apertura y el cierre diarios publicados en la tabla
CSI 300 del [boletín mensual de SSE](https://www.sse.com.cn/aboutus/publication/monthly/index/).
La adquisición de 2022 y 2023 conserva 24 respuestas, sus fechas de descarga y
huellas. Las 484 sesiones coinciden con XSHG, 242 por año, sin duplicados ni
ausencias. La apertura no se reconstruye con el cierre anterior.

`csi300_factor` lee una adquisición terminada, contrasta la consulta
`COMMON_SSE_ZQZS_M_CSI300_INDEX_C` y decodifica su envoltorio JSONP sin ejecutar
código. Rechaza claves duplicadas, meses posteriores a 2023, precios no positivos
o incoherentes y archivos que superan los presupuestos. Los meses declarados se
contrastan con un calendario completo. Las sesiones ausentes se registran y no
se rellenan, incluso cuando falta un mes entre dos meses adquiridos.

La salida incluye `prices.parquet`, `market-factors.json` y `report.json`. El
descriptor puede pasarse a la codificación del corpus mediante
`--market-factors`. Identifica CN, el índice `000300`, el archivo y su SHA256.
Los puntos del índice se conservan en float64 con los dos decimales publicados,
sin añadir ajustes. Los originales fijados conservan su representación textual.
El Parquet real ocupa 17.477 bytes.

La disponibilidad al cierre más cinco minutos es una convención de
reconstrucción retrospectiva. La adquisición actual no acredita las versiones
diarias originales ni la fecha de publicación de cada mensual. Tampoco se ha
verificado una política de correcciones históricas. El informe declara
`point_in_time_verified=false`. El índice sirve para etiquetas predictivas, no
para simular una operación ejecutable. `financial_simulation_ready=false`
conserva esa distinción. La procedencia depende de las respuestas y recibos
previamente contrastados, que el lector no vuelve a autenticar contra la web.

El retorno factorial de una sesión es `close / open - 1`. El residualizador
existente estima alfa y beta con un máximo de 252 sesiones y un mínimo de 126
parejas cuyos cierres ya hayan madurado. El retorno del índice de la siguiente
sesión forma parte de la etiqueta y no entra en las características ni en ese
ajuste. Iniciar el factor en enero de 2022 no aporta calentamiento suficiente
para junio de 2022.

La [comprobación real](../../reports/data/csi300-factor-20261006.json) contrasta
las implementaciones de referencia y NumPy sobre precios preparados de Ping An.
Las 484 filas coinciden exactamente en coeficientes, etiquetas y exclusiones.
De sus 169 fechas candidatas con noticias, precios, fundamentales y macro,
168 tienen etiqueta admisible. La última se excluye porque cruza el corte anual.
Esto aún no acredita muestras codificadas ni un corpus listo para entrenar.

La creación, reutilización y contraste completo tardaron 0,55 segundos en un
proceso, con 146.596 KiB de RAM máxima. Es una ejecución funcional por método,
sin estimación de aceleración. Pasan 77 pruebas relacionadas y cinco mutaciones
dirigidas. La cobertura del módulo es del 90,86 % de sentencias y del 75,71 % de
ramas. El recibo declara herramientas, complejidad y convención de CRAP.

La CLI se ejecuta con `PYTHONPATH=src uv run --no-sync python -m
mars_titan.data.csi300_factor`. Requiere `--acquisition` y `--output`. Las opciones
`--calendar-start` y `--calendar-end` permiten fijar el calendario. El destino
debe estar separado de la adquisición y del dataset. Una interrupción anterior
a la publicación no deja una edición parcial visible. La reutilización vuelve
a comprobar fuentes, contenido y descriptor y rechaza identidades distintas.
