# Presupuesto declarado y consumido en la publicación (13 de octubre de 2026)

Este informe acompaña al segundo bloque de [#442](https://github.com/GonxKZ/mars-titan/issues/442). Recoge cómo el informe de integridad que bloquea la publicación compara el presupuesto de cada modelo con el de los modelos propios y con lo que cada ajuste confirmado consumió, y cómo se ha comprobado que detecta los atajos que persigue. No se ha ejecutado ningún entrenamiento, paso de optimizador, evaluación científica ni piloto.

## Qué se comprueba

`integrity/fit_budget.py` añade la sección `budget` a `integrity.json`. El detalle está en el [banco de integridad](../../../docs/research/integrity-bench.md#informe-de-integridad-de-la-publicación).

- **Declarado.** Del plan de la campaña salen, para cada modelo ajustado, los casos de búsqueda, las semillas y la forma de iterar. El primer modelo propio declarado (MARS-TITAN M0 en A v2) fija la referencia. Cada campo distinto debe figurar con su motivo en `budget_differences` de la declaración de la publicación, que pasa a la versión 2. Una diferencia sin declarar, o declarada sin existir, impide cargar la declaración.
- **Consumido.** Cada ajuste confirmado repite su regla de selección sobre su historial de validación: debe terminar donde la regla manda y entregar la época que elige. Un ajuste sin padre que entrega la época 0, o sin actualizaciones, no cuenta como ajustado. Con presupuesto fijo, los ajustes del mismo entrenador en el mismo ámbito y ventana aplican las mismas actualizaciones, y los de un grupo de parada conjunta terminan en la misma época. XGBoost no supera sus rondas y el control en línea no usa más etiquetas que las escrituras del banco.

## Diferencias declaradas en A v2

| Modelo | Campos distintos | Motivo resumido |
| --- | --- | --- |
| Ridge | búsqueda, semillas, iteración | tres valores de alpha, ajuste determinista y solución cerrada sin épocas |
| XGBoost | búsqueda, semillas, iteración | doce combinaciones de profundidad, bins y tasa, una semilla y rondas con selección por validación |
| Control Transformer en línea | búsqueda, iteración | rejilla de tasas con la tasa cero y actualizaciones limitadas por las escrituras del banco de M1 |
| RNN, LSTM, GRU, DLinear y Transformer compacto | iteración | mismas épocas y regla, con lotes de filas en lugar del recorrido cronológico |

Los motivos completos están en `configs/evaluation/historical-masked-publication-a-v2.json`. Describen lo que ya declaran las configuraciones de la campaña y conviene que Gonzalo los revise antes de lanzarla.

## Hallazgo

Los entrenadores cronológicos (Titans-MAC, MARS-TITAN, CM-v1 y la GRU episódica) evalúan la época 0 y pueden entregarla, mientras que la referencia neuronal sin padre empieza en la época 1. Un modelo propio desde cero podía quedarse con su estado inicial aleatorio, cercano a predecir cero, y competir así con referencias ajustadas. Con este informe ese ajuste queda marcado sin ajustar y la publicación se detiene. MARS-TITAN parte del estado de Titans-MAC en línea (`parent`), así que conservarlo no cuenta como estado sin ajustar.

## Mutación dirigida

Cada mutante sustituye una línea de `fit_budget.py`, de la carga de la declaración o del informe, ejecuta con `pytest -x` las pruebas que lo persiguen y restaura el archivo. Todas corren con la protección de aprendizaje vigente: el consumido se prueba con el plan real de A v2 y recibos e historiales escritos en la prueba. Las dos pruebas que leen informes reales de Titans-MAC y de la referencia neuronal ejecutan los entrenadores con la protección rebajada, así que se omiten mientras rige el bloqueo, todavía no se han ejecutado y no cuentan aquí (inventario de [#546](https://github.com/GonxKZ/mars-titan/issues/546)).

Resultado: 32 de 32 mutantes detectados ([registro](mutation.json)), también al repetirlos tras integrar la cabeza 0eda4174 de #550. B4, que deja de comparar la iteración, muere al cargar la declaración de A v2 porque sus diferencias declaradas dejan de existir. Una puntuación de mutación no garantiza que no haya otros fallos. Indica que cada decisión mutada tiene al menos una prueba que la exige.

## Límites

- La etapa de adaptadores ya rechaza en su ejecutor un número de actualizaciones distinto del de su diseño, pero el informe todavía no repite esa comprobación.
- El informe compara el presupuesto declarado y el consumido. No mide si la rejilla de cada referencia es la adecuada para ella, que es una decisión de diseño.
- La campaña no se ha ejecutado. No hay ninguna cifra de resultados.
