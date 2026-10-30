# Techo de señal con anomalías conocidas

Este documento describe el estudio de la fase 0b de [#747](https://github.com/GonxKZ/mars-titan/issues/747). Mide cuánta señal diaria extraen dos modelos conocidos con anomalías publicadas, en las mismas filas, ventanas, cohortes y semillas que la [criba de variantes](variant-screening-v1.md). La [declaración](../../configs/experiments/signal-ceiling-v1.json) fija todos los valores antes de ejecutar nada y la [orden](../../scripts/run_signal_ceiling.py) la comprueba antes de leer ningún dato.

El estudio no se ha ejecutado. Sus pruebas usan modelos dobles de predicción fija. Ajustar Ridge o LightGBM es aprendizaje, así que la orden comprueba la protección del aprendizaje antes de abrir ninguna fuente y hoy se detiene. El test de 2024 sigue cerrado. El estudio corre en CPU, sin redes ni optimizadores de PyTorch.

## Pregunta

La fase 0 ([#744](https://github.com/GonxKZ/mars-titan/issues/744)) no encontró ordenación robusta en los predictores neuronales de los pilotos. El estudio pregunta si modelos sencillos con estas características sí la encuentran, usando el IC por sesión como contraste principal. Repite cada ajuste con el retorno bruto para contrastar la residualización. Un resultado negativo también puede deberse a las limitaciones de esos modelos o características, así que no diagnostica por sí solo un fallo de los datos. Un resultado positivo aportaría una referencia que los predictores neuronales deberían comparar sobre las mismas filas y ventanas.

El techo es el de estos dos modelos con estas características y filas, no el de los datos. Un techo bajo no prueba que no exista señal y uno alto no confirma ninguna variante neuronal. El estudio no cambia ninguna decisión de la criba.

## Filas, ventanas y cohortes

Cada origen de la criba, de 2019 a 2022, usa su vista con su cohorte de 256 activos de EE. UU., que `screening_cohorts.verify` comprueba leyendo solo los manifiestos. Las filas de cada tramo son las que declara la vista:

- El ajuste usa las filas de entrenamiento decididas en los doce meses de ajuste de la criba, las mismas decisiones que la fase de ajuste de Titans-MAC.
- La validación elige el alpha de Ridge y las rondas de LightGBM con el IC por sesión.
- La calibración fija el λ por mercado del MAE calibrado.
- La evaluación es el bloque de S del origen y solo se mide.

La orden comprueba que la vista es la ventana del origen y que el test final sigue cerrado, y ninguna fila medida llega a 2024.

## Características

Para la sesión t de un activo, con C el cierre, V el volumen y r el retorno cierre a cierre, el módulo [`anomaly_features`](../../src/mars_titan/data/anomaly_features.py) calcula diez características en seis familias:

| Familia | Característica | Definición | Fuente |
| --- | --- | --- | --- |
| Reversión | `reversal_1`, `reversal_5` | C_t / C_{t−k} − 1 con k = 1 y 5 | Jegadeesh (1990) y Lehmann (1990) |
| Momentum | `momentum_12_1` | C_{t−21} / C_{t−252} − 1, con cada extremo en el último cierre de su sesión o de las cinco anteriores | Jegadeesh y Titman (1993) |
| Volatilidad idiosincrática | `idiosyncratic_volatility_20`, `idiosyncratic_volatility_60` | Desviación del residuo de MCO con intercepto de r del activo sobre r del factor de mercado en 20 y 60 sesiones, con n − 2 grados de libertad | Ang et al. (2006) |
| Volumen | `volume_shock_20` | log(1 + V_t) menos la media de log(1 + V) en las 20 sesiones anteriores | Gervais et al. (2001) |
| Volumen | `relative_turnover_20_252` | Media de log(1 + V) en 20 sesiones menos la media en 252 | Datar et al. (1998) |
| Máximo anual | `distance_to_52_week_high` | C_t entre el máximo cierre de 252 sesiones, menos uno | George y Hwang (2004) |
| Presentaciones | `sessions_since_filing`, `recent_filing_5` | Sesiones desde la última presentación conocida en la decisión y si son menos de cinco | Bernard y Thomas (1989), como aproximación |

Una ventana exige el 75 % de sus observaciones y una característica sin ellas queda ausente. Las sesiones son las del calendario declarado del mercado sin las ausentes en todo él, igual que en las ventanas de precios de la edición.

Las fuentes son tres. Los precios se leen de la vista con su huella comprobada. El factor de mercado es el que declara la vista para su residuo, SPY en EE. UU., también con su huella. Las presentaciones salen de los fundamentales preparados de cada activo, cuyo manifiesto declara su huella y la de los precios que comprueba la vista, así que la cadena llega hasta la cohorte. Un fondo cotizado no tiene presentaciones y su familia queda ausente.

El precio de la sesión t está disponible en su decisión, el cierre más cinco minutos, así que las características de t usan los cierres hasta t. Una presentación se conoce con su primer hecho disponible y su sesión es la de la primera decisión igual o posterior. La rotación no se divide entre las acciones en circulación, porque la edición distribuye volúmenes con los ajustes retrospectivos de su fuente y las acciones de cada presentación no los tienen. Mezclarlos introduciría desdoblamientos posteriores a la decisión. Los cocientes de precios no cambian con un factor de ajuste común. Las diferencias de log(1 + V), usadas para admitir volumen cero, sí pueden cambiar. Por ejemplo, con veinte volúmenes anteriores iguales a uno y el actual igual a tres, el choque es log 2. Si se duplica toda la serie, pasa a log(7/3). Se conserva la fórmula declarada y esta limitación de la fuente, sin atribuir disponibilidad histórica perfecta al volumen ajustado. Los infinitos se rechazan como errores y las ausencias se representan con NaN.

En cada sesión, cada característica pasa a rangos en [−1, 1] entre los activos de la cohorte con precio en esa sesión, como en Gu et al. (2020). Con n valores válidos, el rango medio r pasa a 2 (r − 1) / (n − 1) − 1, y una ausencia vale cero con su indicador. Los modelos reciben veinte columnas, los diez rangos y sus diez indicadores. La sección cruzada solo depende de los precios de la sesión, no de qué filas tienen después una etiqueta madura.

La issue pedía los días alrededor de los resultados con su fecha de disponibilidad real. La edición no tiene fechas de anuncio y la fecha de la siguiente presentación no se conoce en la decisión, así que solo se cuentan las sesiones desde la última presentación conocida. Usar la fecha real de la siguiente sería una fuga.

## Objetivos

Cada modelo se ajusta dos veces en las mismas filas:

- Con el residuo de la criba, el objetivo de la vista.
- Con el retorno bruto apertura-cierre de la sesión siguiente, `close / open − 1` del activo en la siguiente posición del calendario, leído de los mismos precios verificados.

Para ajustar, el objetivo se normaliza en cada sesión con su media y su desviación con ddof = 1, como el `CSZScoreNorm` de Qlib, y las sesiones sin desviación no entran.

## Modelos

Ridge (Hoerl y Kennard, 1970) usa `Ridge` de scikit-learn con intercepto sin penalizar y la solución de Cholesky, que no depende de semillas. Ajusta seis alphas, de 1 a 100.000 en potencias de diez, y la validación elige el de mayor IC medio por sesión. Un empate se queda con el alpha mayor, el que más contrae.

LightGBM (Ke et al., 2017) conserva los parámetros de Qlib sobre Alpha158 (Yang et al., 2020) en el commit fijado, la misma declaración del control común de la criba salvo su identidad, sus características y su etiqueta. Ajusta las cinco semillas de la criba hasta 1.000 rondas, puntúa la validación cada diez y la época común de las cinco semillas, con radio uno, fija las rondas de todas. Ninguno de los dos modelos estandariza las características, porque los rangos y los indicadores ya comparten escala.

## Medidas

La evaluación de cada origen da, para cada objetivo y modelo, estas series por sesión:

- El IC por sesión, la correlación de Pearson entre predicción y objetivo, y el IC de rangos. Las semillas de LightGBM se promedian sesión a sesión en las sesiones que todas definen, como en la criba.
- El MAE de la predicción sin escalar y el MAE calibrado, con el λ de cada mercado elegido en la calibración de cada semilla, junto al MAE del control cero, que predice un objetivo nulo.
- La importancia de cada familia, la caída del IC por sesión de la evaluación al permutar juntas sus columnas, rangos e indicadores, dentro de cada sesión (Breiman, 2001). La permutación es la misma para los dos modelos, sus semillas y los dos objetivos, con un flujo aleatorio por familia derivado de la semilla 20261028.

Una predicción constante en una sesión con al menos tres activos y objetivo variable no ordena y su IC vale cero. La criba deja esa sesión sin definir, y la diferencia solo cambia el resultado de un modelo degenerado, como un LightGBM que no encuentra ninguna partición con la regularización de Qlib. La misma convención puntúa la validación, para que ese modelo mida cero en vez de interrumpir el estudio.

El [informe](../../src/mars_titan/evaluation/signal_ceiling_report.py) une los cuatro orígenes en una serie de 2019 a 2022 en orden temporal y aplica la inferencia del análisis de la criba (`evaluate_family`): bootstrap circular por bloques de 16 días UTC, 2.000 réplicas con la semilla 20261023, intervalos al 95 % marginales y simultáneos, y valores p del paso descendente de Romano y Wolf y de Holm. Cada familia usa las sesiones que definen todos sus miembros. Hay tres familias de contrastes sobre las cuatro combinaciones de objetivo y modelo, el IC por sesión, el IC de rangos y el MAE calibrado menos el del control cero, y una familia de importancias por combinación. El IC por sesión se publica también con bloques de 8 y 32 días, sin cambiar la lectura principal.

El techo observado de cada objetivo es el modelo con mayor IC medio de validación, la media de los cuatro orígenes con el mismo peso, elegido sin mirar la evaluación. Un empate se queda con Ridge, el modelo más sencillo.

## Ejecución

```bash
uv run --locked python scripts/run_signal_ceiling.py
uv run --locked python scripts/run_signal_ceiling.py --help
```

La ejecución del estudio requiere `--run`, `--cohorts` con la carpeta de cohortes verificada y `--output` con la carpeta de resultados. Sigue sujeta al bloqueo de aprendizaje.

Sin `--run`, la orden comprueba la declaración e imprime sus orígenes, modelos y objetivos. Con `--run`, comprueba la protección, la declaración y las cohortes, y cada origen escribe `origins/origin-AÑO.json` al terminar. Un origen terminado con la misma petición, que incluye la huella de la vista, el origen, la huella de la declaración y las del código, se reutiliza, así que repetir la orden continúa tras un corte. La identidad incluye el lector del corpus, los auxiliares temporales, la estructura del panel y los módulos de lectura y publicación. Cambiar cualquiera de esas fuentes rechaza la reutilización antes de reabrir los datos. Un origen interrumpido se repite entero. Al final escribe `report.json`. El informe del estudio, con el techo observado y su interpretación prudente, irá a `reports/research/` cuando se ejecute.

El coste no se ha medido. Con unas 64.000 filas de ajuste por origen y veinte columnas, Ridge resuelve seis sistemas pequeños. LightGBM ajusta cinco semillas por objetivo y origen, cuarenta ajustes en total, con el mismo límite de rondas que el control común.

## Desviaciones y límites

- Las presentaciones solo aproximan los anuncios de resultados, que llegan días o semanas antes. No se calculan los días hasta el siguiente resultado.
- El residuo de la volatilidad idiosincrática es el del modelo de mercado, el único factor que declara la vista. Otras definiciones usan varios factores.
- La rotación es relativa a la del propio activo y no usa las acciones en circulación.
- Los rangos se calculan entre los 256 activos de la cohorte, no en todo el mercado.
- La convención del IC nulo de una predicción constante se aparta de la criba, como se explica arriba.
- La importancia por permutación es descriptiva. Con familias correlacionadas, como la reversión y el máximo anual, una familia puede repartir su importancia con otra, y no se ha usado para elegir características.
- LightGBM y Ridge no se han ajustado nunca con estas entradas. Las pruebas usan modelos dobles de predicción fija.

## Comprobaciones

`tests/data/test_anomaly_features.py` comprueba cada fórmula con respuestas conocidas, la volatilidad residual frente a `numpy.polyfit`, los mínimos de cada ventana, la tolerancia del momentum, la edad de las presentaciones, también sin ninguna, los rangos por sesión y que alterar los datos posteriores a un corte deja igual todo lo anterior. `tests/models/test_session_ridge.py` comprueba la declaración de Ridge, sus rechazos, la etiqueta por sesión, la elección y el empate del alpha y la protección, con un `Ridge` doble que no ajusta nada.

`tests/training/test_signal_ceiling.py` recorre la vista corta de las pruebas con un factor de mercado y presentaciones técnicas. Comprueba que las filas y el retorno bruto son los de la vista y sus precios, las filas de ajuste, la lectura de las fuentes con sus huellas, la permutación por sesión, la convención del IC nulo y un origen completo con Ridge y LightGBM dobles, su reutilización y sus rechazos. La prueba de causalidad altera precios, factor y presentaciones después de un corte y exige que ninguna característica anterior cambie. `tests/evaluation/test_signal_ceiling_report.py` y `tests/training/test_signal_ceiling_cli.py` comprueban la unión de los orígenes, las familias frente a `evaluate_family`, el techo con su empate y la orden con sus orígenes sustituidos.

La [verificación técnica del 30 de octubre](../../reports/engineering/signal-ceiling-20261030/verification.json) registra 451 casos CPU aprobados sin omisiones sobre la integración con `develop`. La cobertura de diez módulos suma 1.544/1.852 sentencias y 388/616 ramas. Se midieron las 72 funciones cambiadas de producción, con CRAP máximo 59,11. La selección no recorre todas las rutas previas de los módulos compartidos. Ruff, formato y el comprobador del repositorio pasan. Las pruebas reprodujeron cinco fallos del traslado de LightGBM y siete de entradas infinitas o reutilización del informe antes de corregirlos.

El recibo sigue incompleto: las 83 recetas de mutación actuales compilan, pero su ejecución todavía está en curso. Las 65 detectadas en la fuente anterior se identifican como históricas. La regresión de 518 módulos importadores está preparada en lotes de hasta ocho módulos y dos trabajadores para acotar los temporales, sin equipararla a una única sesión global. Ni estas pruebas ni las de predicción fija miden señal financiera. El estudio científico y el test final siguen bloqueados.
