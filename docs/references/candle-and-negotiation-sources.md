# Antecedentes de representación de velas y control de decisiones

El 18 de octubre de 2026 se revisaron los posts de
[Quant Science](https://x.com/quantscience_/status/2111792698164121789) y
[bodila](https://x.com/51bodila/status/2111568573164392666). El acceso directo a X devolvió 403.
Sus textos y medios se recuperaron mediante la API pública de FxTwitter y se contrastaron
con las fuentes primarias. Las afirmaciones promocionales no se toman como resultados de
MARS-TITAN.

## Kronos y la representación numérica

*Kronos: A Foundation Model for the Language of Financial Markets* utiliza tokenización
de OHLCV y un Transformer autorregresivo. No necesita capturas de gráficos. El 93 % citado
en el post corresponde a una mejora relativa de RankIC en su comparación, no a una tasa
de acierto ni a una reducción equivalente de MAE. El apéndice D de la versión consultada
declara preentrenamiento hasta junio de 2024 y evaluación desde julio de ese año.
[Artículo](https://arxiv.org/html/2508.02739v1),
[publicación y DOI](https://ojs.aaai.org/index.php/AAAI/article/view/39730).

El código se fijó en `67b630e67f6a18c9e9be918d9b4337c960db1e9a`. Su predictor completa
volumen ausente con cero y puede aproximar importe negociado mediante volumen por la media
de OHLC. Esas reglas no equivalen al contrato de máscaras de MARS-TITAN. Sus pesos publicados
no se han utilizado como referencia histórica limpia para 2020/2023 y no se ha comprobado
su solapamiento exacto por activo.
[Código consultado](https://github.com/shiyu-coder/Kronos/blob/67b630e67f6a18c9e9be918d9b4337c960db1e9a/model/kronos.py).

La [geometría de 64 velas](../data/candle-geometry.md) es un control propio y determinista
de 512 componentes. Evita dibujar un PNG y cargar pesos visuales durante su derivación.
Conserva los mismos OHLC y añade relaciones redundantes entre ellos. No incorpora el modelo
Kronos ni añade información de mercado. Su coste de extracción está medido. Su utilidad
predictiva requiere ajustar un modelo propio y compararlo con las mismas filas.

## Figgie y el alcance de sus resultados

El documento enlazado es *Reinforcement Learning for Figgie: Learning Negotiation as a
First-Class Skill*, de Daniel Yang, fechado el 7 de junio de 2026. Estudia cuatro jugadores
con PPO, GRU, rasgos bayesianos, clonación de un agente y modificación de recompensas.
Evalúa 1.000 partidas por configuración. Los resultados pertenecen al juego y no acreditan
rentabilidad bursátil ni precisión de retornos residuales.
[Documento original](https://cs224r.stanford.edu/projects/pdfs/Daniel%20Li%20Yang%20submission_416100137/CS224R_Project_Report-5.pdf).

Las reglas oficiales describen un juego de cuatro o cinco participantes, con composición
oculta de cartas y pagos definidos. Esa estructura permite razonar sobre estados ocultos
concretos. Los OHLC diarios no proporcionan por analogía un libro de órdenes, prioridad
de cola, llenados ni un modelo de impacto.
[Reglas de Figgie](https://www.figgie.com/how-to-play.html).

El criterio trasladable al controlador es separar información, acción y resultado. Las
consultas se valoran por la reducción de error que se observa al madurar una predicción
y por su coste. Leer más o refinar más no recibe una recompensa por sí mismo. Los controles
de rutas fijas, umbral, asignación por hash y retorno residual cero permiten detectar
actividad adicional sin utilidad predictiva. No se crea un entorno de negociación nuevo
a partir de esa analogía.

Estas fuentes no acreditan originalidad mundial de la ampliación ni superioridad frente
al estado del arte. Los [recibos de extracción](../../reports/engineering/candle-geometry-20261018/README.md)
y los [del lector](https://github.com/GonxKZ/mars-titan/pull/668) son evidencia
técnica acotada. La comparación predictiva conserva sus propios modelos, protocolo y resultados.
