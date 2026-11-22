# Cuaderno de investigación vivo de Titans

Este cuaderno recoge, en orden, qué se sabe, qué se ha probado y qué se va a probar para que los
Titans y sus variantes aprendan de verdad y predigan con la máxima precisión y fiabilidad. Cada
entrada distingue lo **medido** de lo que es **hipótesis**. Los resultados negativos se
conservan. El detalle de la campaña DGX está en
[`docs/engineering/dgx-us-campaign.md`](../engineering/dgx-us-campaign.md).

## Reglas del cuaderno

- Una sola variable por experimento, con su control declarado.
- El origen 2013 es el de desarrollo: todo lo de este cuaderno se decide con él. Los orígenes
  2014 a 2023 son la evidencia limpia y 2024 sigue cerrado.
- Un cambio solo entra en la base con contraste pareado, error típico con autocorrelación y,
  cuando el coste lo permite, tres semillas.
- Toda memoria que diga aprender en evaluación se mide contra la misma memoria congelada y
  contra un placebo (etiquetas barajadas dentro de la sesión).
- Lo que propone un modelo externo se anota como propuesta hasta que una medida propia lo
  confirme.

## Estado medido (22 de noviembre de 2026, tarde; origen 2013, semilla 42, IC de evaluación)

| Pieza | Medida | Lectura |
| --- | --- | --- |
| Mejor núcleo: memoria en línea, rangos por sesión, sin ganancia de la mediana | 0,1163 | +0,0041 sobre los árboles (0,1122). Es el mismo ajuste que `titans_mac_xrank` de la v8, bit a bit |
| Núcleo de la v10 (con ganancia de la mediana) | 0,1132 | −0,0031 respecto al anterior (t = −3,0); +0,0009 sobre los árboles (t = 0,4) |
| Memoria en el mejor núcleo: congelada, sin memoria | 0,1098, 0,1059 | Escribir en línea suma 0,0065 (t = 5,2); un 17 % viene de escribir en evaluación (gemelos, N = +0,0011, t = 3,5) |
| Término de IC, rama densa | −0,0054, −0,0137 al quitarlos | Las dos piezas que más suman |
| Atención entre activos | +0,0021 (t = 2,3) | Su control que no lee a los demás queda a 0,0007: es sobre todo la capa residual |
| Órganos de memoria con etiquetas maduras | +0,0023 a +0,0031 (t = 2,5 a 2,8) | Sobre el núcleo, la congelada y la atención; con la atención llega a 0,1175 |
| Lectores episódicos y núcleo cognitivo | igual que su núcleo | No añaden |
| Rendimiento operable (etiqueta más alfa) | núcleo v10 0,0787, mejor núcleo 0,0830, árboles 0,0837; media de rangos de tres 0,0877 | El alfa conocido explica 0,0813 del IC de la etiqueta. En lo operable el núcleo v10 queda por debajo de los árboles (t = −2,1) y el mejor núcleo los iguala (+0,0043 sobre el v10, t = 3,8); con sus órganos llega a 0,0858 (M-2). Es la medida más expuesta al sesgo de supervivencia (D-2) |
| Objetivo operable en la red tabular | +0,0040 a +0,0063 de IC operable con MAE más IC operable y +0,0077 a +0,0104 con el IC operable solo (tres semillas) | Opción nueva en la base v11, en la cola de la DGX con y sin ganancia. Devuelto a la etiqueta pierde entre 0,001 y 0,002. Restar el alfa exacto en la cabeza no añade (H6-b) |
| Combinación con los árboles (media de rangos de tres modelos) | 0,1185 | +0,0022 sobre el mejor individual (t = 3,3). Los pesos aprendidos en línea no la mejoran (A-1) |

## Hipótesis abiertas

### H1. La memoria no aprende porque no ve el resultado

**Hipótesis.** La escritura de la memoria es autoasociativa del propio paso: memoriza el
estado, cuya historia ya resumen los 46 rasgos, y no usa la señal que este problema regala,
las etiquetas que maduran cada día (unas 4.000 por sesión). Una memoria que escriba con esas
etiquetas seguiría la deriva de la relación entre rasgos y retornos, que los pesos externos no
ven durante los dos años que separan el ajuste de la evaluación.

**Antecedentes en el repositorio.** Ya existen, probados solo con fixtures y nunca ejecutados
con datos reales: la mezcla multiescala de correctores lineales con errores maduros
([E1](../experiments/multiscale-mature-mixture-v1.md)), su versión multifrecuencia
([R2](../experiments/mature-cms-v1.md)) y la señal de fiabilidad madura que modula las puertas
([`mature_reliability_signal_v1`](../experiments/mature-reliability-signal-v1.md)).

**Antecedentes externos.** Una reimplementación crítica de Titans señala que el artículo
original no aisló la aportación de las actualizaciones en evaluación
([Di Nepi, Siciliano y Silvestri, 2025](https://arxiv.org/abs/2510.09551)). TAFAS adapta un
predictor congelado en evaluación con la verdad parcialmente observada y un módulo de
calibración con puerta ([Kim et al., 2025](https://arxiv.org/abs/2501.04970)).

**Experimento H1-a (medido el 22 de noviembre, una semilla).** Con los 46 rasgos y una red
congelada ajustada hasta 2010 (validación 0,1157, calibración 0,1242, evaluación 0,1147), que
después recorre 2011 a 2013 en orden. Los hiperparámetros de cada memoria se eligen con el IC
de calibración y se informa la evaluación. Guion: `tab_mem.py` en los registros.

| Memoria | Evaluación | Frente a la estática | t |
| --- | ---: | ---: | ---: |
| Cabeza reestimada en línea, retardo de una sesión (semivida 63, contracción 0,3) | 0,1148 | +0,0001 | 0,05 |
| La misma con retardo de dos sesiones | 0,1144 | −0,0004 | −0,26 |
| Entradas en línea (92 entradas y la predicción) | 0,1153 | +0,0005 | 0,23 |
| Memoria por activo del residuo maduro (semivida 21, peso 0,1) | 0,1165 | +0,0017 | 2,42 |
| Placebo: cabeza en línea con etiquetas barajadas | −0,0190 | −0,1337 | −22,4 |

- **Aprender del mercado en línea no añade orden.** Reestimar la cabeza con las etiquetas
  maduras gana 0,003 en calibración, el tramo con el que se eligió, y nada en evaluación. Toda
  la rejilla queda entre 0,095 y 0,1155: las semividas cortas con poca contracción empeoran y
  las largas igualan a la red estática. El placebo confirma que la memoria usa las etiquetas.
- **La memoria por activo sí añade, poco:** +0,0017 en evaluación (t = 2,4), +0,0007 en
  calibración y −0,0004 en validación. Es frágil: falta repetirla con semillas.
- **Conclusión provisional.** En este origen la relación entre rasgos y retornos no deriva lo
  bastante como para que seguirla en línea rinda. Si la memoria aporta aprendiendo en
  evaluación, es por activo: lo que cada activo se desvía de forma persistente de lo que el
  modelo espera de él. Coincide con lo medido en los Titans (aprender en línea, +0,0013).

**Experimento H1-b (22 de noviembre, una semilla).** La memoria por activo con tres valores
guardados y su placebo, que lee la memoria de otro activo (permutación fija). Mismos datos y
misma red estática que H1-a; semivida 21 sesiones y peso 0,1, elegidos con calibración. Guion:
`tab_lab.py`.

| Valor guardado | Evaluación | Frente a la estática | t | En calibración (t) |
| --- | ---: | ---: | ---: | ---: |
| Residuo madurado | 0,1165 | +0,0017 | 2,42 | +0,0007 (0,67) |
| Placebo del residuo | 0,1137 | −0,0010 | −4,53 | −0,0018 (−4,34) |
| Etiqueta madurada sola | 0,1162 | +0,0015 | 3,02 | +0,0009 (1,06) |
| Placebo de la etiqueta | 0,1138 | −0,0009 | −4,76 | −0,0022 (−5,03) |
| Signo del residuo | 0,1165 | +0,0017 | 2,70 | +0,0007 (0,74) |
| Placebo del signo | 0,1136 | −0,0011 | −4,90 | −0,0023 (−5,20) |

- **Las tres variantes dan lo mismo** (+0,0015 a +0,0017) y la memoria de otro activo resta
  0,001: la información es del propio activo, no de la sesión ni de la escala.
- **No es un rasgo que falte.** La red ya recibe la media de las últimas 21 etiquetas maduras
  de cada activo (`realized_label_21`) y las rentabilidades de apertura a cierre a cinco
  horizontes. La memoria añade lo que la red, ajustada hasta 2010, pondera de menos en 2011 a
  2013: la persistencia propia de cada activo.
- **Se repite con tres semillas.** Con el residuo, semivida 21 y peso 0,1 (los mismos
  elegidos por calibración en las tres): +0,0017, +0,0018 y +0,0015 en evaluación (t = 2,4,
  2,7 y 2,1) y +0,0007, +0,0010 y +0,0009 en calibración. El placebo resta entre 0,0007 y
  0,0013 (t entre −3,9 y −5,9). Falta medirla sobre las predicciones de los Titans.

### H2. Dos años de obsolescencia de los pesos externos cuestan orden

**Hipótesis.** Reajustar con datos hasta el año anterior, a épocas fijas, o continuar en
línea con etiquetas maduras, recupera varios milésimos de IC.

**Medido (H1-a, una semilla, ocho épocas fijas): no.** La misma red ajustada también con 2011
da 0,1143 en 2013 (−0,0005, t = −0,9) y ajustada también con 2011 y 2012, 0,1128 (−0,0019,
t = −1,5). Con los rasgos solos, dos años de obsolescencia no cuestan orden en este origen.
Queda por medir en los Titans, comparando en un mismo año las predicciones de orígenes
distintos, y en años de ruptura (2008, 2020, 2022).

### H3. La macro solo puede ordenar a través de la sensibilidad de cada activo

**Hipótesis.** Los índices macro son iguales para todos los activos de una sesión. Solo
cambian el orden si se cruzan con la sensibilidad de cada activo a sus variaciones. Una
herramienta que calcule esas sensibilidades con datos pasados (o una memoria por activo que
las aprenda de las etiquetas maduras) es la forma de que el modelo aprenda cómo se relacionan.

**Experimento H3-a (22 de noviembre, una semilla): la variación diaria no añade orden.** El
choque de cada sesión es la diferencia de un indicador respecto a la sesión anterior, entre su
desviación en el ajuste. Solo **uno** de los 140 indicadores cambia en más de la mitad de las
sesiones; los demás se publican por semanas o meses y quedan fuera de este diseño.

| Variante | Evaluación | Frente a la estática | t | Placebo (choques barajados) |
| --- | ---: | ---: | ---: | ---: |
| Sensibilidad por activo con olvido (semivida 63, contracción 1, peso 0,03) | 0,1145 | −0,0002 | −1,80 | −0,0001 |
| Regresión lineal con interacciones rasgo × choque, frente a la lineal sin ellas | 0,0967 | −0,0001 | −0,51 | −0,0004 |
| Red más la parte de interacción (peso 0,1) | 0,1133 | −0,0014 | −1,86 | −0,0003 |

La parte de interacción gana 0,0010 en calibración y pierde 0,0014 en evaluación: ruido. Ese
diseño dejaba sin contrastar los 139 indicadores de publicación espaciada.

**Experimento H3-b (22 de noviembre, una semilla): tampoco por publicaciones.** Hay publicación
de un indicador cuando la edad de su dato vuelve a cero; la sorpresa es el valor nuevo menos
el anterior, tipificada con las sorpresas del tramo de ajuste. Por activo e indicador, una
regresión con contracción de la etiqueta madura sobre la sorpresa, actualizada solo con
publicaciones pasadas. Guion del enjambre de Gemini (`macro_eventos.py`, 21 pruebas), revisado
y ejecutado con los datos reales.

| Variante | Evaluación | Frente a la estática | t | En calibración |
| --- | ---: | ---: | ---: | ---: |
| Mejor configuración por calibración (sin olvido, contracción 100, retardo 3, peso 0,05) | 0,1144 | −0,0003 | −1,74 | +0,0004 |
| Placebo: fechas de publicación barajadas por indicador | 0,1147 | −0,0000 | −0,13 | −0,0005 |
| Placebo: sorpresas barajadas entre eventos | 0,1144 | −0,0003 | −1,58 | −0,0009 |

La lectura real no se separa de sus placebos. Con los dos diseños, la macro no ordena activos
a través de una sensibilidad lineal por activo en este origen; coincide con lo que las
ablaciones de modalidad dirán en los Titans (`v_no_macro`, `v_macro`). Es coherente con que
el mercado absorbe los anuncios dentro de la sesión y con que no hay consenso de analistas
con el que medir la sorpresa.

### H4. Los lectores no aportan por sus claves y por lo que guardan

**Hipótesis.** Las claves son una proyección aleatoria de las entradas crudas, el banco tiene
1.024 entradas (menos de un cuarto de una sesión) y la admisión por sorpresa lo llena de
saltos irrepetibles. Claves desde el estado del núcleo, valor igual al residuo maduro del
padre y admisión de lo reciente deberían cambiarlo. Antes hay que separar cuánto de la mejora
de los lectores con el término de IC viene del banco y cuánto de sus capas ajustadas.

**Experimento H4-a (22 de noviembre, una semilla).** Banco con todas las filas de las últimas
sesiones maduras, valor igual al residuo madurado recortado a ±3 y lectura de los 256 vecinos
de mayor coseno. Placebo: los valores del banco permutados.

| Claves | Evaluación | Frente a la estática | t | En calibración (t) | IC de la lectura sola |
| --- | ---: | ---: | ---: | ---: | ---: |
| Activaciones de la última capa oculta (21 sesiones) | 0,1154 | +0,0007 | 1,50 | +0,0010 (1,20) | 0,016 |
| Su placebo | 0,1141 | −0,0006 | −2,31 | −0,0009 (−1,87) | |
| Proyección aleatoria de las entradas (63 sesiones) | 0,1164 | +0,0016 | 2,78 | +0,0021 (2,00) | 0,019 |
| Su placebo | 0,1141 | −0,0006 | −2,42 | −0,0025 (−5,87) | |

- **Las claves no eran el problema.** La proyección aleatoria de las entradas rinde más que el
  estado de la red: el estado se parece entre activos que la red ya trata igual, y las
  entradas conservan qué activo es y cómo está.
- **Lo que cambia respecto a los lectores de la campaña es el banco y su contenido:** 63
  sesiones enteras (unas 250.000 filas, frente a 1.024 entradas), admisión de todo lo reciente
  en lugar de las sorpresas, y como valor el residuo madurado.
- **Se repite:** con la proyección aleatoria, +0,0016 en las semillas 42 y 43 (t = 2,8 y 2,9;
  +0,0021 y +0,0020 en calibración), con placebos de −0,0006 y −0,0014. Con el estado oculto,
  +0,0007, +0,0006 y +0,0011 en las tres semillas.

**Experimento H4-b (22 de noviembre, dos semillas): se suman en parte.** Las dos lecturas con
la configuración que eligió cada una, y sus dos pesos elegidos a la vez con calibración.

| Semilla | Las dos (evaluación, t) | En calibración | Sobre solo activo | Sobre solo vecinos | Acuerdo entre lecturas |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | +0,0024 (2,67) | +0,0025 (1,69) | +0,0007 (1,79) | +0,0008 (2,15) | 0,16 a 0,32 |
| 43 | +0,0029 (2,41) | +0,0028 (1,57) | +0,0011 (1,86) | +0,0012 (1,77) | 0,15 a 0,34 |

El acuerdo es la correlación de rangos entre las dos lecturas dentro de cada sesión: baja, así
que no leen lo mismo. Juntas valen unos 0,0025 de IC sobre una red estática, la mitad de lo
que dieron los rangos por sesión. Es la primera memoria con etiquetas maduras que se separa de
su placebo en todas las réplicas. Siguiente paso: medirla sobre las predicciones emitidas de
los Titans (el padre ya tiene memoria en línea) y, si sobrevive, convertirla en un órgano
lector de la campaña con sus controles.

### H5. Optimización en un recorrido cronológico no estacionario

**Hipótesis.** Con 400 a 1.000 actualizaciones por época, el segundo momento de Adam
(0,999) tarda de una a dos épocas en adaptarse; 0,95 lo haría seguir el régimen. Acumular
sesiones enteras en lugar de 8.192 filas daría pasos más estables.

**Experimento H5-a (22 de noviembre, una semilla, red tabular): ninguna ayuda.** Segundo
momento de 0,95: evaluación −0,0000 (t = −0,05) y calibración −0,0018 (t = −1,76). Cinco
sesiones por actualización en lugar de dos: −0,0011 (t = −1,39). Decaimiento de pesos
desacoplado de 0,01: −0,0001 (t = −1,20). Se conserva la configuración base.

## Registro de experimentos

| Id | Fecha | Pregunta | Variable | Control | Resultado | Decisión |
| --- | --- | --- | --- | --- | --- | --- |
| H1-a | 22-11 | ¿Aprender de las etiquetas maduras añade orden sobre una red congelada? | Cabeza reestimada en línea | Red estática; placebo | +0,0001 (t 0,05); por activo +0,0017 (t 2,4) | La memoria compartida en línea no se persigue; se sigue la memoria por activo |
| H2-a | 22-11 | ¿Reajustar con datos más recientes mejora la evaluación? | Fin del tramo de ajuste (2010, 2011, 2012) | Red ajustada hasta 2010 | −0,0005 y −0,0019 | Sin evidencia de coste por obsolescencia en 2013 |
| H1-b | 22-11 | ¿Qué debe guardar la memoria por activo y es propia de cada activo? | Valor guardado (residuo, etiqueta, signo) | Red estática; memoria de otro activo | +0,0015 a +0,0017 (t 2,4 a 3,0); placebo −0,0010 (t −4,5) | Es información del activo; repetir con semillas y medir sobre los Titans |
| H3-a | 22-11 | ¿La variación diaria de la macro, cruzada con la sensibilidad de cada activo, añade orden? | Sensibilidad por activo; interacciones rasgo × choque | Red estática; choques barajados | −0,0002, −0,0001 y −0,0014 | No, con el único indicador diario; falta la prueba por publicaciones (H3-b) |
| H4-a | 22-11 | ¿Qué limita a la memoria episódica: las claves o el banco? | Claves (estado oculto, proyección aleatoria) con banco de sesiones enteras | Red estática; valores permutados | +0,0007 (t 1,5) y +0,0016 (t 2,8); placebo −0,0006 | El banco y lo que guarda; las claves aleatorias bastan |
| H5-a | 22-11 | ¿Ayuda cambiar el optimizador? | Segundo momento 0,95; 5 sesiones por paso; AdamW 0,01 | Configuración base | −0,0000, −0,0011, −0,0001 | Se conserva la base |
| H4-b | 22-11 | ¿La memoria por activo y la de vecinos son la misma información? | Las dos lecturas a la vez, con sus pesos | Cada una sola | +0,0024 y +0,0029 juntas; +0,0007 a +0,0012 sobre cada una | Son complementarias en parte; pasan a medirse sobre los Titans |
| H3-b | 22-11 | ¿La sorpresa de cada publicación macro, cruzada con la sensibilidad de cada activo, añade orden? | Regresión por activo en tiempo de eventos | Red estática; fechas barajadas; sorpresas barajadas | −0,0003 (t −1,7); placebos −0,0000 y −0,0003 | No. Con H3-a, la sensibilidad lineal a la macro queda descartada en este origen |
| E-1 | 22-11 | ¿Quitar la repetición del tramo acelera el ajuste sin cambiarlo? | `accumulation_rows` ausente | Recetas con repetición, a igual carga | No acelera (85–102 frente a 107–126 pasos/min) y la memoria congelada se frena (38 → 23) | Retirado a los 35 minutos; pendiente de un banco de pruebas con los tres modos de memoria |
| F-1 | 22-11 | ¿Qué hace la memoria por dentro al entrenar? | Trazas de `titans_mac` | `titans_mac_frozen` | Tasa de escritura 0,049 a 0,025; olvido 0,003 a 0,02; momento 0,50 a 0,2; pérdida asociativa entre 16 y 48 sin converger | La memoria está viva, pero no resuelve su objetivo interno |
| F-2 | 22-11 | ¿Tiene la memoria en línea la firma de un aprendizaje en evaluación? | Trazas del ajuste de 13 variantes en línea de la v8: pendiente de la pérdida asociativa a lo largo de la pasada (efecto fijo de época, error con autocorrelación) y relación entre la norma de los pesos rápidos y esa pérdida | Su control congelado, en validación | La pérdida asociativa sube a lo largo de la pasada en 10 de 13 (base: +0,65 en logaritmos, t 5,6); más norma de pesos rápidos no baja la pérdida (ρ de −0,03 a +0,49); la ventaja en validación sobre la congelada (+0,0005 a +0,0041 con el término de IC; negativa sin él) no guarda relación con la pérdida asociativa | La ventaja no viene de resolver el objetivo asociativo. Falta la prueba causal: la misma red evaluada sin escribir |
| D-1 | 22-11 | ¿Qué sesgos y defectos tiene la población preparada? | — | — | 4.200 de 4.202 activos de EE. UU. y los 810 de China llegan a diciembre de 2023; ningún precio ≤ 0 ni valor no finito; el 2,6 % de las filas de EE. UU. y el 3,9 % de las de China no tienen volumen (565 y 172 activos con más del 5 %); los fundamentales quedan disponibles al cierre de la sesión siguiente a su presentación (retraso mínimo de 1 día sobre el fin del periodo, mediana 131) | La supervivencia ya está declarada en el protocolo. La iliquidez de Amihud cuenta como cero los días sin volumen: se corrige en una versión nueva de los rasgos, no en la campaña en curso |
| P-1 | 22-11 | ¿Corta la paciencia de 3 épocas antes de tiempo? | Historias de validación de 63 ajustes de la v8 (ruido época a época por segundas diferencias; caída respecto al mejor al cortar) | — | El ruido época a época es de 0,0004 (mediana; 0,0007 en el percentil 90 de los modelos en línea) y al cortar la validación ha caído 0,0015 desde su mejor valor, unas cuatro veces ese ruido; solo el 19 % de los modelos en línea sube en la última época | La paciencia de 3 no corta por ruido. No se gasta GPU en un piloto de paciencia; el tope de 48 épocas ya cubre los ajustes que se quedaban sin terminar |
| E-1b | 22-11 | ¿Por qué quitar la repetición no aceleró y frenó a la memoria congelada? | Lectura del código y reproducción en CPU del segundo enjambre, con pruebas que fallan contra el código original | — | La memoria congelada se anclaba en cada tramo sobre todos los flujos vistos y, al reunir la sesión, autograd creaba un corte por cada hueco, de coste creciente con los flujos acumulados; el camino directo además espera a la GPU en cada comprobación y retiene los grafos de los dos instantes | Corregido en la rama `v11-candidatos` (anclaje por bloque y comprobaciones diferidas). La repetición sigue siendo la receta: falta medirlo en GPU con los tres modos |
| H1-c | 22-11 | ¿Las memorias supervisadas con etiquetas maduras suman sobre los Titans ya ajustados? | Memoria por activo, memoria por vecinos y las dos a la vez sobre las emisiones de 13 modelos de la v8, con sus pesos elegidos en calibración | Su padre; etiquetas barajadas; vecinos sin el propio activo | Las dos juntas: +0,0020 (t 2,2) sobre el mejor modelo (0,1163 → 0,1183), +0,0034 (t 2,6) sobre el núcleo y entre +0,0024 y +0,0046 en el resto. Los placebos quedan por debajo del padre en los 13 (de −0,0002 a −0,0169; t de −1,7 a −10,4). El acuerdo entre las dos lecturas es de 0,31 a 0,37. La ganancia es la misma con la memoria en línea, congelada o sin memoria | Es información que la memoria autoasociativa no recoge. Pasan a ser lectores de la base nueva; réplica con las semillas de la v10 y en los orígenes limpios |
| C5-p | 22-11 | ¿La atención entre activos aporta información de los demás activos, o es el efecto de añadir una capa? | v10, origen 2013, semilla 42: `cross_asset_attention` frente al núcleo y frente a su control `self_only`, que tiene los mismos parámetros y solo se lee a sí mismo | `titans_mac`; `cross_asset_attention_self`; sin memoria, `titans_mac_disabled` | Frente al núcleo: +0,0021 en evaluación (t 2,3) y +0,0005 en decisión (t 0,3). Frente a su control: +0,0007 en evaluación (t 1,0) y 0,0000 en decisión. Sin memoria resta: −0,0019 en evaluación (t −2,4) y −0,0039 en decisión (t −3,8) | Con una semilla, la mayor parte de la ganancia es de la capa residual y no de leer a los otros activos. La mezcla ocurre antes de que la rama técnica sume su residuo, así que no ve la señal que más ordena. La base v11 añade la posición después de la rama (`position="after_technical"`), apagada por defecto y con su control; entra en la criba v11 (`y1_xattn`). Falta repetir el contraste con las semillas 43 y 44 |
| I-1 | 22-11 | ¿Valen los supuestos del error estándar de los contrastes: mercados independientes y rezago fijo de Newey-West? | Emisiones reales: campaña conjunta v2 (16 modelos, origen 2013, semilla 42, 404 días con los dos mercados) y v10 (7 contrastes en sus tres tramos) | — | Correlación entre EE. UU. y China del IC del mismo día UTC: de +0,01 a +0,09 en niveles; en las diferencias pareadas entre modelos (120 pares), mediana +0,008, con p10 −0,033 y p90 +0,084 (el error de muestreo de una correlación con 404 días es 0,05). Autocorrelación de las diferencias pareadas: −0,03 de mediana en el rezago 1 (de −0,09 a +0,16) y ±0,02 en los rezagos 2 a 5; el error con el rezago automático es 0,98 veces el i.i.d. de mediana y con rezago 20 sube como mucho un 18 % | Los dos supuestos se sostienen: tratar los mercados como bloques independientes subestima el error en torno a un 1 %, y un ancho de banda dependiente de los datos no cambiaría las conclusiones. La correlación de 0,75 y el AR(1) de 0,75 que suponía la auditoría externa no se dan en estos datos. El resumen entre orígenes (`dgx_seed_contrasts.py` con varios `--year`) añade la prueba que no depende de ninguno de los dos: cada origen como réplica, con t de K − 1 grados |
| O-1 | 22-11 | ¿Lo ganado en la etiqueta se conserva en el rendimiento que se puede operar? | v10, origen 2013, semilla 42: IC por sesión de la etiqueta sin alfa con la predicción, por un factor elegido en calibración, más alfa; con todas las filas y en el tercio más líquido de cada sesión | Cada modelo con su control; el núcleo con `lightgbm_technical` | En lo operable el núcleo queda 0,0050 por debajo de los árboles (t −2,1; 0,0787 frente a 0,0837) y 0,0036 por encima en el tercio líquido (t 1,0). Escribir en memoria en línea suma +0,0056 frente a la memoria congelada (t 3,5) y +0,0072 frente a no tener memoria (t 4,1), más que en la etiqueta (+0,0014 y +0,0032). El término de IC, que suma 0,0054 en la etiqueta, suma +0,0002 (t 0,1). La atención entre activos, +0,0013 (t 1,3). En la v8, la receta con rangos y sin ganancia de la mediana daba 0,0830 con la regla de escala anterior | Lo que gana el término de IC está en la parte de la etiqueta que ya se conoce al predecir; lo que aporta la memoria en línea está en la parte operable. Los contrastes de semillas y de orígenes miden desde ahora las dos (`dgx_seed_contrasts.py --offset`). La receta final se elige mirando las dos, con el trío sin ganancia de la mediana en las tres semillas y en los orígenes limpios |
| H6-a | 22-11 | ¿Entrenar el término de IC sobre lo operable mejora el IC operable? | Red tabular (92 → 128 → 128 → 1, sin memoria), tres semillas, cuatro objetivos: IC, IC operable, MAE + 3·IC y MAE + 3·IC operable. El IC operable correlaciona predicción más alfa con los rangos normales de etiqueta más alfa | El mismo objetivo sobre la etiqueta, pareado por sesión | MAE + 3·IC reproduce al núcleo (etiqueta 0,1132, 0,1136 y 0,1112; operable 0,0806, 0,0798 y 0,0791). Con el término sobre lo operable: operable 0,0847, 0,0861 y 0,0846 (+0,0040, +0,0063 y +0,0054; t de 3,5 a 4,6) a su escala natural, y etiqueta 0,098. Devuelta a la etiqueta como k·(p + alfa) − alfa recupera 0,1117 y 0,1103 frente a 0,1136 y 0,1112. Como único objetivo: operable 0,0877, 0,0877 y 0,0892 (t de 5,6 a 6,4) | Opción `offset_column` y `offset_scale` del término de IC en la base v11, apagada por defecto (68 pruebas, 7 mutantes). Se mide en los Titans en la DGX con y sin ganancia de la mediana y en los tres modos de memoria |
| F-3 | 22-11 | ¿Aporta escribir en la memoria durante la evaluación, o la ventaja viene de haberse ajustado con ella? | Gemelos de escritura: la misma red ajustada en línea, evaluada escribiendo y sin escribir (paridad bit a bit en las anclas). Par sin rangos y par con rangos de la v8, origen 2013, semilla 42 | La red ajustada con memoria congelada | Sin rangos: escribir suma +0,00003 (t = 0,6). Con rangos, que es el mejor núcleo: +0,0011 (t = 3,5; pasa Holm entre los dos pares), sin decaer por trimestres (+0,0008, +0,0016, +0,0015, +0,0006), el 17 % de la ventaja sobre la congelada (intervalo de Fieller 8 % a 31 %); en calibración +0,0021 (t = 5,6), el 30 % | En el mejor núcleo la memoria aprende en evaluación, poco y de forma estable; la mayor parte de su ventaja está en el ajuste. Siguen el par de la v10 y el brazo que reinicia el estado al empezar la evaluación |
| M-1 | 22-11 | ¿Suman los órganos de memoria con etiquetas maduras sobre los Titans de la v10? | Memoria por activo y por vecinos emitidas fila a fila sobre cuatro modelos de la v10, con pesos elegidos en calibración | Su padre; placebos | Las dos juntas: +0,0023 (t = 2,8) al núcleo, +0,0031 (t = 2,5) a la congelada, +0,0029 (t = 1,6) a sin memoria y +0,0023 (t = 2,5) a la atención entre activos, que llega a 0,1175 (+0,0053 sobre los árboles, t = 3,0). Los ocho contrastes son positivos; tres pasan Holm | En el plan del relevo tras cada origen. Falta sobre el trío sin ganancia y sobre lo operable |
| H6-b | 22-11 | ¿Restar el alfa exacto en la cabeza (la red predice lo operable y emite la etiqueta por resta) mejora el objetivo operable respecto a darle el alfa como rasgo? | Red tabular, semillas 42, 43 y 44; tres objetivos con desplazamiento exacto p = f − alfa: MAE + 3·IC de la etiqueta, MAE + 3·IC operable e IC operable solo | Los mismos objetivos sin desplazamiento exacto (H6-a) y MAE + 3·IC, pareados por sesión | Con el IC de la etiqueta el desplazamiento exacto no cambia nada: operable a escala 0,0792, 0,0789 y 0,0799 (−0,0014, −0,0009 y +0,0008; t de −1,3 a 0,8). Con MAE + 3·IC operable: 0,0847, 0,0863 y 0,0865, igual que sin él (0,0847, 0,0861 y 0,0846), y etiqueta 0,107. Con el IC operable solo: 0,0885, 0,0882 y 0,0895 (+0,0077, +0,0084 y +0,0104 sobre MAE + 3·IC, t de 5,2 a 7,2; sin desplazamiento exacto 0,0877, 0,0877 y 0,0892), tercio líquido +0,0031, +0,0052 y +0,0076 (t de 1,8 a 3,2), y devuelta a la etiqueta 0,1105, 0,1118 y 0,1127 frente a 0,1127, 0,1127 y 0,1134 | El desplazamiento exacto no se adopta: queda implementado y probado en la rama `v11-known-offset`, fuera de la base. Lo que mueve el IC operable es el objetivo, no la parametrización. Ajustar para lo operable cuesta entre 0,001 y 0,002 de IC de la etiqueta al devolverla y gana cerca de 0,009 en lo operable: la criba v11 lo mide en los Titans, y sus contrastes deben incluir la etiqueta devuelta, k·(p + alfa) − alfa con k elegido en calibración |
| A-1 | 22-11 | ¿Mejora un agregador causal en línea a la media de rangos al combinar los Titans con los árboles? | Emisiones de la v10 en 2013, semilla 42: `titans_mac_nogain`, `cross_asset_attention` y `lightgbm_technical`; nueve agregadores con pesos decididos solo con etiquetas maduras e hiperparámetros elegidos en calibración (entrega del tercer enjambre; sus 25 pruebas pasan ejecutadas aquí y los IC individuales reproducen el informe de campaña) | Media de rangos; mejor individual; placebos de pesos barajados y de etiquetas permutadas | Media de rangos 0,1185 (+0,0022 sobre el mejor individual, t 3,3). Regresión contraída con ventana, con media exponencial y con covarianza, y los dos Hedge: de 0,1184 a 0,1185. AdaNormalHedge 0,1171 (−0,0013, t −2,5). Pesos por régimen 0,1182 (−0,0003, t −1,2). Los dos placebos, 0,1185 | Con tres modelos casi iguales y muy correlacionados los pesos aprendidos no aportan: se conserva la media de rangos. Queda por medir con los órganos de memoria como componentes (señal débil y distinta), encargado al programa de post-entrenamiento |
| A-2 | 22-11 | ¿Suma la señal de los órganos de memoria como cuarto componente del conjunto? | Lo mismo que A-1 más la señal del brazo combinado de los órganos de `titans_mac_nogain` (emisión menos padre), sin validación, selección en calibración y madurez exacta de la vista; 24 ensayos (programa de post-entrenamiento) | Media de rangos de los tres (0,1185 en la etiqueta y 0,0877 en lo operable); el brazo combinado solo; la señal barajada entre activos | Etiqueta: regresión contraída con covarianza 0,1196, +0,0011 (t 2,1; listón del máximo de 24 ensayos, 1,98); la media de rangos con la señal baja a 0,1174; con la señal barajada, 0,1185. Operable: 0,0889, +0,0012 (t 1,2). Peso medio de la señal: 10 % y 16 % | Suma poco y en el límite del azar. Los pesos barajados casi igualan: importa el peso medio de la señal y no su ritmo. Registrado como secundario S2 en los orígenes limpios |
| M-2 | 22-11 | ¿Suman los órganos de memoria en lo operable y sobre el trío sin ganancia? | Tabla operable de los órganos sobre siete modelos de la v10, 2013, semilla 42; factor elegido en calibración (programa de post-entrenamiento) | Su padre; `lightgbm_technical` (0,0837 en lo operable) | Sobre `titans_mac_nogain` (0,0830 en lo operable, 0,1163 en la etiqueta): brazo combinado 0,0858 en lo operable (+0,0027, t 2,9; +0,0021 sobre los árboles, t 1,1) y 0,1182 en la etiqueta (+0,0020, t 2,2; +0,0060 sobre los árboles, t 3,4). Los 14 contrastes brazo menos padre son positivos (t de 2,0 a 4,4; listón del máximo de 14, 1,7) | Los órganos suman también en lo operable. Siguen encadenados tras cada origen; C4 queda como está |
| D-2 | 22-11 | ¿Delata supervivencia la cobertura de noticias, o el universo? | Última sesión de precios de los 4.202 activos de EE. UU. y primera noticia de los 2.724 de la caché de 2013 | — | 4.200 de 4.202 activos llegan al final de 2023. 2.683 de 2.724 tienen noticias, que empiezan entre 2009 y 2013 (mediana 2011,7). La bandera de tener ya alguna noticia tiene un IC de +0,026 (t 14,8) en 2013: mide seguimiento y tamaño, no supervivencia, porque todos sobreviven | El universo es de supervivientes y se dice en el registro confirmatorio: afecta a todos los modelos, más a los niveles que a los contrastes y más a lo operable que a la etiqueta |
| H6-c | 22-11 | ¿Los dos términos de IC a la vez conservan el IC de la etiqueta y el operable? | Red tabular, semillas 43 y 44: MAE + 3·IC de la etiqueta + 3·IC operable; también con pesos 1,5 + 1,5 y el término operable solo con peso 10 | MAE + 3·IC de la etiqueta y MAE + 3·IC operable, pareados por sesión | Con los dos términos (3 + 3): operable +0,0057 y +0,0056 sobre la base (t 4,1 y 4,5), igual que con el operable solo (−0,0005 y +0,0002), y etiqueta devuelta 0,1133 y 0,1134 (la base, 0,1133 y 0,1119). Con 1,5 + 1,5, la mitad del operable. Con peso 10, operable +0,0086 y etiqueta devuelta 0,1097 | Es el punto que no sacrifica ninguna de las dos medidas. Opción `plain_weight` en la base v11 (bb43b642) y segunda criba en la cola de la DGX |
| R-1 | 22-11 | ¿Ayudan el decaimiento en coseno de la tasa de aprendizaje o la media móvil exponencial de los pesos? | Red tabular, semilla 43, sobre MAE + 3·IC | La misma red con tasa constante y pesos propios | Coseno: etiqueta −0,0021 (t −1,4), operable −0,0045 (t −2,4). Media móvil (0,999): +0,0001 (t 0,1) y +0,0006 (t 0,4) | Ninguno. En un recorrido cronológico los últimos pasos son los años más recientes y bajar la tasa les quita peso. No se repiten con la semilla 44 |
| D-3 | 22-11 | ¿Suman los rasgos nuevos calculados con lo que ya hay en disco (61 rasgos, encargo directo a Gemini, 25 pruebas suyas en verde aquí)? | Árboles (objetivo de rangos, 400 rondas) con los 46 rasgos más cada familia, calculada para los 2.724 activos de la caché de 2013 con el instante de decisión de la campaña | Los mismos árboles con los 46 rasgos (0,1111, 0,1184 y 0,1140 en validación, calibración y evaluación), pareados por sesión | Familia tipo Alpha158 (32): +0,0000 en evaluación (t 0,0) con el 22 % de la ganancia de ajuste. Recuento de noticias (8): −0,0005 (t −0,8). Contabilidad dinámica (18): −0,0035 (t −3,2), con cobertura del 4 al 8 % en el ajuste y del 50 % en 2013 | No entran. Las de precios son redundantes con los 46; las noticias miden seguimiento y tamaño; la contabilidad empeora porque su cobertura cambia entre el ajuste y la evaluación. Falta medir en China y las transformaciones por sector |
| T-1 | 22-11 | ¿Dónde ganan los árboles a la red? | Diagnóstico del tercer enjambre con las emisiones reales de 2013: `titans_mac_nogain` frente a `lightgbm_technical`, por deciles de cada rasgo, zonas de ausentes y extremos e interacciones 2×2, con cortes fijados en calibración | — | Etiqueta: la red gana 0,0040 (t 2,4) y la mezcla de rangos llega a 0,1180. Operable: empatan (0,0830 y 0,0837), correlación entre predicciones 0,71 y mezcla 0,0886. Los árboles ganan en las banderas de historia corta y de días sin negociar y en las colas de la volatilidad y del alfa | La mezcla vale más que afinar cualquiera de los dos. Hipótesis H7: incrustaciones por rasgo en la rama técnica para los umbrales y las colas |
| G-1 | 22-11 | ¿Está en las trazas reales la firma del mecanismo que el tercer enjambre deriva para la ganancia de la mediana y la memoria en línea? | Auditoría de las trazas de la v10 (2013, semilla 42): núcleo con y sin ganancia, y sus memorias congeladas | El núcleo sin ganancia | Con la ganancia, la pérdida asociativa de la memoria acaba 13 veces mayor (570 frente a 44), los pesos de la cabeza derivan 1,86 veces más y el gradiente de MAC es 1,9 veces mayor | Coherente con la simetría de escala entre la cabeza y la ganancia. Se contrasta con el decaimiento de pesos de 0,01 en la segunda criba |
| H7 | 22-11 | Protocolo, escrito antes de ajustar. ¿Las incrustaciones periódicas por rasgo (PLR lite: 16 frecuencias, 8 dimensiones) delante de la red tabular suben su IC? | Red tabular con MAE + 3·IC, semillas 43 y 44: PLR con frecuencias iniciales N(0; 0,1²) y N(0; 1) | La base sin incrustaciones y el control `lin_emb` (ReLU(w·x + b) por rasgo, misma anchura), pareados por sesión | Predicción: PLR sube el IC de evaluación al menos 0,0010 sobre la base y más que su control lineal. Si el control lineal iguala a PLR, la ganancia es de anchura y no de periodicidad | Regla: pasa a una criba de la DGX con el parche de la rama técnica si PLR − base ≥ +0,0010 en evaluación en las dos semillas y con la decisión del mismo signo; se abandona si queda por debajo de +0,0005 en las dos |
| D-4 | 22-11 | ¿Suman los rasgos nuevos en China? (la mejora de datos debe cubrir los dos mercados) | Tabla de China de la cohorte conjunta de 2013 (651 activos, 997.642 filas, 1.811 sesiones) con los 46 rasgos y los candidatos; árboles como en D-3. La contabilidad no se calcula (hechos en otra moneda y con otros conceptos) y no hay noticias de China | Los árboles con los 46 rasgos (0,1052, 0,1233 y 0,1170), pareados por sesión | Familia tipo Alpha158 (32), semilla 42: evaluación 0,1252, +0,0082 (t 3,9); decisión +0,0062 (t 2,4); se lleva el 43 % de la ganancia de ajuste. Por subfamilias: días desde máximos y mínimos +0,0033 (t 2,2), subidas +0,0027 (t 1,9), precio y volumen +0,0026 (t 1,9), variabilidad del volumen −0,0003. En EE. UU., todas las familias juntas: −0,0001 (t −0,1) | En China sí hay señal nueva y está repartida. Semillas 43 y 44 y la forma de la vela, en curso. Pasa a H8 |
| H8 | 22-11 | Protocolo, escrito antes de ver el resultado. ¿La familia tipo Alpha158 sirve también a la red en China? | Red tabular (dos capas de 128, MAE + 3·IC) en la tabla de China: 46 rasgos y sus rangos (92 entradas) frente a 78 y sus rangos (156), semillas 42 a 44, rasgos escalados con el tramo de ajuste | La misma red con los 46 rasgos, pareada por sesión | Predicción: al menos +0,0040 en evaluación con la calibración del mismo signo en dos de las tres semillas | Regla: con una media de +0,0030 o más en evaluación y calibración del mismo signo, la familia entra como versión nueva de los rasgos técnicos (extensión de entradas, sin recodificar la historia), se comprueba que es neutra en EE. UU. con la red y se criba en la campaña conjunta. Por debajo de +0,0015 la ganancia es solo de los árboles: la familia va a la referencia de árboles de China y las entradas de la red no cambian |
| H8-r | 22-11 | Resultado de H8: la familia tipo Alpha158 en la red, en China | Las tres semillas con los 32 rasgos de la familia y con los 16 que pasan la validación rasgo a rasgo (D-5) | La misma red con los 46 rasgos (0,1263, 0,1288 y 0,1245) | Con los 32: +0,0048, +0,0048 y +0,0054 (t 2,0 a 2,2), media +0,0050, con la calibración del mismo signo en las tres. Con los 16: +0,0073 (t 4,2), −0,0009 (t −0,5) y +0,0082 (t 3,6), media +0,0049; calibración +0,0081, +0,0044 y +0,0130 | La regla de H8 se cumple (media de +0,0030 o más y calibración del mismo signo): la familia entra como versión nueva de los rasgos técnicos. Los 16 dan la misma media con la mitad de columnas y más dispersión entre semillas (una semilla nula). La elección de los 16 no se apoya en la evaluación: sale de la validación con el tramo de ajuste |
| D-5 | 22-11 | ¿Qué rasgos de la familia entran, y da el código de la campaña los mismos números que el laboratorio? | Validación rasgo a rasgo con el tramo de ajuste de China: IC por sesión con t de Newey-West, IR por trimestres, estabilidad del signo y redundancia de rangos con los 46. Implementación `long_horizon_v3` en `technical_features.py` de la rama `v11-candidatos` (e3809397): los 46 de la v2 como prefijo bit a bit y 16 columnas centradas en su punto neutro, con tres cuartos de cada ventana como cobertura mínima | La familia medida en el laboratorio, con precios reales | Pasan 16 de 32 (\|t\| ≥ 3,5, \|IR\| ≥ 0,5, mismo signo en el 70 % de los trimestres, redundancia menor de 0,7). En 25 activos de China y 25 de EE. UU. (197.297 filas) las 16 columnas del módulo son iguales a las de la familia del laboratorio, tras su transformación declarada, hasta el último bit de float32. 17 pruebas y 11 mutantes a mano, todos muertos. En EE. UU. con árboles (semilla 42): los 16, decisión +0,0001 (t 0,1) y evaluación +0,0015 (t 3,3); los 32, +0,0016 (t 2,3) y +0,0018 (t 3,3) | Límite: la selección usa datos hasta el final del tramo de ajuste del origen 2013. Es causal para los orígenes de 2013 en adelante y no para los anteriores, así que un contraste confirmatorio de la v3 solo puede usar orígenes desde 2013. En EE. UU. esta tabla da una ganancia pequeña con árboles y la de D-3 daba +0,0000 con la misma familia: la diferencia es del orden del ruido entre ajustes y se resuelve con las semillas 43 y 44, en curso |
| H9 | 22-11 | Protocolo, escrito antes de ajustar. ¿Sirven los rasgos v3 al modelo completo con memoria en la campaña conjunta de EE. UU. y China? | Dos declaraciones con el mismo código (`v11-candidatos` cac92bf8), la misma cohorte y la misma receta: el trío sin ganancia de la mediana (en línea, congelado y sin memoria) con `long_horizon_v2` y con `long_horizon_v3` (rama técnica de 62 columnas). Origen 2013, semilla 42. Árboles de referencia con los rasgos de cada brazo. Contrastes pareados por sesión y leídos por mercado (`--market CN` y `--market US`), porque en la conjunta cada sesión de cada mercado pesa lo mismo | El mismo modelo con los rasgos v2, con el mismo código y la misma semilla | Predicción: en las sesiones de China, entre +0,002 y +0,005 de IC de evaluación para el modelo en línea, menos que la red tabular porque el modelo completo ya ve la ventana de precios; en las de EE. UU., entre −0,001 y +0,001 | Regla. Primario: la Δ del modelo en línea en las sesiones de China, en evaluación. (a) Con Δ > 0 y t ≥ 2 en China y Δ ≥ −0,002 en EE. UU., se ajustan las semillas 43 y 44 de los dos brazos, y la v3 pasa a ser la versión de rasgos de la declaración conjunta final si la media de las tres semillas en China es de +0,0015 o más, con al menos dos semillas positivas, y EE. UU. no pierde más de 0,001 de media. (b) Con Δ > 0 y t < 2 en China, una semilla más antes de decidir. (c) Con Δ ≤ 0 en China, la v3 no entra en la red y queda para la referencia de árboles si los árboles ganan. Secundarios, sin regla: el IC operable, la memoria (en línea menos sin memoria) en cada brazo y el modelo frente a los árboles de su brazo. La campaña de EE. UU. sola no cambia de rasgos salvo que EE. UU. gane +0,0015 o más con t ≥ 2 |
| D-6 | 22-11 | ¿Sirven en China los candidatos que se descartaron midiendo solo en EE. UU.? | Los 67 candidatos de la exploración del 21-11, calculados para la tabla de China con sus mismas funciones, sin tocarlas (`x4_mercado.py`). Árboles con los 15 que no necesitan el factor de mercado y no están ya en la v3 | Los árboles con los 46 rasgos, pareados por sesión | Semilla 42: +0,0037 (t 1,6) en evaluación y +0,0038 (t 1,6) en decisión. Sobre los 16 de la v3 añaden +0,0027, y sobre el modelo de mercado sustituto de D-7 con los 16, −0,0008 | No entran: lo que añaden no se distingue del ruido y desaparece cuando China tiene su modelo de mercado. De paso apareció D-7: 28 de los 67 candidatos salen vacíos en China |
| D-7 | 22-11 | ¿Cuánto pierde China sin su factor de mercado? | El factor que declara la vista para China (índice CSI 300, `…/history-2000-20261007/csi300-inherited-gaps-recovery/price-factor-2006-2023-v16/prices.parquet`, huella b96cd972…) no está en la DGX. `corpus_inputs.py` apaga entonces, sin aviso, los rasgos del modelo de mercado: `alpha_percent`, `alpha_t`, `beta`, `log_residual_volatility`, `has_market_model`, `realized_label_1`, `_5` y `_21`, `alpha_21_percent` y `alpha_63_percent` valen cero en todos los activos chinos. Se mide con un sustituto hecho solo con los datos del disco: la media simple del rendimiento intradía de los activos chinos con precio en cada sesión, entre 442 y 810 activos en 4.372 sesiones (`x5_cn.py`) | Los árboles con los 46 rasgos tal como salen hoy, pareados por sesión | Semilla 42: evaluación de 0,1170 a 0,1300, +0,0130 (t 4,4); decisión +0,0137 (t 4,0). Semilla 43: +0,0098 (t 3,6) y +0,0162 (t 4,9). Con el sustituto, los 16 rasgos de la v3 añaden +0,0019 en evaluación (sin él, +0,0058) y unos +0,005 en decisión | Es la mayor ganancia medida en China, más del doble que la de los rasgos nuevos, y parte de lo que daban esos rasgos era compensar lo que falta. No es un fallo del código: es un fichero que no se copió a este equipo y hay que traerlo de la máquina de origen. El sustituto no es el factor de la etiqueta y no se usa para entrenar. Hasta que llegue el fichero, toda la conjunta se ajusta en la DGX con el modelo de mercado de China apagado. Semilla 44 y red tabular, en curso |
| H9-n | 22-11 | Nota al protocolo de H9, escrita antes de ajustar ninguno de sus brazos | D-7 se descubrió después de fijar H9. Los dos brazos se ajustan en las mismas condiciones, con el modelo de mercado de China apagado si el fichero sigue sin estar; el relevo deja constancia de su presencia antes de cada brazo (`registra-factor.sh`, plan v40) | — | Predicción sin cambios para ese caso. Con el factor presente se espera menos de los rasgos v3 en China: entre 0 y +0,003 | Si el fichero llega con un brazo ya ajustado, la comparación no vale: se repiten los dos. Si llega después de H9, H9 se repite con el factor y esa es la lectura que decide la versión de rasgos de la conjunta final |

## Interacción observada: rangos por sesión y memoria en línea

Con una semilla, los rangos por sesión suman 0,0055 al núcleo con memoria en línea, 0,0004 al
de memoria congelada y 0,0010 al núcleo sin memoria. Con los rangos, aprender en línea queda
0,0065 por encima de la memoria congelada (0,0013 sin ellos). Puede ser una semilla afortunada:
la campaña v10 lo repite con tres semillas en los tres núcleos antes de darlo por cierto.

**Tabla completa de la v10 (22 de noviembre, tarde).** Ganancia de la mediana, rangos por sesión y
modo de memoria, IC de evaluación con la semilla 42:

| Receta | En línea | Congelada | Sin memoria |
| --- | ---: | ---: | ---: |
| Sin ganancia, sin rangos | 0,1107 | 0,1094 | 0,1049 |
| Sin ganancia, con rangos | 0,1163 | 0,1098 | 0,1059 |
| Con ganancia, sin rangos | 0,1126 | 0,1068 | 0,1057 |
| Con ganancia, con rangos (núcleo v10) | 0,1132 | 0,1117 | 0,1100 |

Quitar la ganancia al núcleo v10 suma 0,0031 en línea (t = 3,0) y resta 0,0020 con memoria congelada
(t = −2,8) y 0,0041 sin memoria (t = −4,3). Quitar los rangos resta 0,0006 en línea (t = −0,5), 0,0050
con memoria congelada y 0,0043 sin memoria. La ganancia ayuda a los modelos que no escriben y perjudica
al que escribe, y los rangos al revés según haya ganancia. El trío sin ganancia de la v10 es, tensor a
tensor, el trío con rangos de la v8 (misma receta y misma vista): no es una réplica independiente. No
hay explicación medida de la interacción; está encargada a Gemini y la acotan las semillas 43 y 44, que
ya incluyen el trío sin ganancia.

**La v10 no lo repite con la semilla 42 (22 de noviembre).** El núcleo de la v10 es la receta
con rangos de la v8 más la ganancia aprendida de la mediana, con otra versión del código y un
tope de épocas más alto que ninguno de los dos alcanza. Diferencia pareada del IC por sesión,
memoria en línea menos memoria congelada:

| Par | Validación | Calibración (t) | Evaluación (t) | Evaluación por trimestres |
| --- | ---: | ---: | ---: | --- |
| v8 sin rangos | +0,0013 | +0,0052 (2,8) | +0,0013 (1,2) | +0,0012, −0,0013, +0,0042, +0,0012 |
| v8 con rangos | +0,0041 | +0,0068 (3,1) | +0,0065 (5,4) | +0,0064, +0,0090, +0,0069, +0,0037 |
| v10 (rangos y ganancia de la mediana) | +0,0010 | +0,0070 (2,1) | +0,0014 (1,2) | +0,0050, +0,0047, +0,0001, −0,0041 |

- En los tres pares la ventaja de escribir en línea es mayor en calibración, las 61 sesiones
  que siguen a la validación, que en la evaluación del año siguiente.
- De la v8 con rangos a la v10, el núcleo en línea baja 0,0031 (0,1163 a 0,1132) y los de
  memoria congelada y sin memoria suben 0,0019 y 0,0041. O la ganancia de la mediana ayuda a
  los controles y perjudica al que escribe, o es la variación de un ajuste a otro. Lo separa el
  trío sin ganancia de la v10 (`r4_nogain`, en la cola del origen 2013) y lo acotan las semillas
  43 y 44. Decidido antes de ver ese resultado: el trío sin ganancia se ajusta también con las
  semillas 43 y 44, porque es la única diferencia de receta con el mejor núcleo medido.
- En la v10 la ventaja decae a lo largo del año de evaluación (pendiente de −0,012 por año,
  t = −3,1) y acaba negativa en el último trimestre; en los dos pares de la v8 no decae. Es una
  observación hecha después de mirar, sobre una semilla. **Comprobación fijada antes de tener
  los datos:** con las semillas 43 y 44 de la v10, la pendiente de esa diferencia dentro del año
  de evaluación será negativa en las dos. Si lo es, la memoria que escribe se degrada con el
  recorrido y procede medir un reinicio periódico de su estado con la herramienta de los gemelos
  de escritura, sin reajustar. Si no lo es, se descarta como ruido.

## Propuestas externas pendientes de medir

La cuarta ronda de Gemini (22 de noviembre) propone, por orden de ganancia esperada: reajuste
hasta el año anterior, memoria rápida compartida por sesión con etiquetas maduras, escritura
supervisada por activo, segundo momento de Adam de 0,95, claves aprendidas en el banco
episódico y mezcla del agente con escala conservada. Sus rangos de ganancia son conjeturas
suyas, no medidas.

### Segundo enjambre de Gemini (22 de noviembre)

Doce trabajadores con verificación cruzada; se cortó por cuota antes de escribir su síntesis.
Sus entregas están archivadas fuera del repositorio y sus 229 pruebas pasan ejecutadas aquí.
Lo que sigue separa lo que una medida propia ha confirmado de lo que queda como propuesta.

| Propuesta | Estado | Decisión |
| --- | --- | --- |
| Auditoría de 17 módulos (15 fallos) | Confirmados: sus pruebas fallan contra el código original y pasan con los parches | Integrados en `v11-candidatos`. No cambian ningún resultado de la v10, salvo dos variantes que se leen con cautela: el dropout de modalidades y de fusión (la repetición usaba otras máscaras) y la media lenta de pesos con 0,9998 (sin calentamiento arrastra los pesos iniciales durante épocas) |
| Opciones de receta: normalización en la rama densa, recorte por grupo, recencia por tasa de aprendizaje | Integradas apagadas; con ellas apagadas el ajuste es idéntico bit a bit | Se miden en la criba de la base nueva. La recencia como peso de la pérdida no sirve: el recorte de gradiente la anula |
| Emitir sin grafo los tramos que se repiten | Integrada apagada; predicciones y gradientes idénticos bit a bit en CPU (cuatro variantes, dos cabezas) | Medir la ganancia de tiempo en GPU antes de adoptarla |
| Subir la paciencia de la parada | Refutada su urgencia (P-1) | Se conserva la paciencia de 3 |
| La memoria en línea no aprende asociaciones, acumula | Coherente con las trazas (F-2) | Prueba causal encargada al programa de post-entrenamiento: la misma red evaluada sin escribir |
| Escritura supervisada dentro del núcleo (M4 del primer enjambre) | El enjambre afirma que colapsa a cero; sin medida propia | No se implementa. Las memorias supervisadas van como lectores (H1-b, H4-a, H4-b) |
| Diez cambios de arquitectura (recorte suave, escala por rasgo, rangos de fundamentales, conjunto implícito, separar gráficos y rasgos técnicos, incrustaciones periódicas, precios por parches, atención por factores, fichas de ausencia, tronco ancho) | Propuestas con su control y su criterio de abandono. Su diagnóstico de la atención entre activos coincide en parte con lo medido (C5-p) | Tres ya están en la criba de la base nueva, apagadas por defecto: el recorte suave, la escala por rasgo y la atención después de la rama técnica. El resto, candidatas de una en una si la criba deja unidades de cómputo |
| Rasgos técnicos: iliquidez de Amihud con días sin volumen, precio de cierre cero y medias diluidas al empezar | La iliquidez queda sesgada en las filas sin volumen (2,6 % en EE. UU. y 3,9 % en China, D-1); el modelo recibe aparte la fracción de días sin negociar. El cierre cero no ocurre en estos datos (D-1) y la dilución afecta a los dos primeros meses de cada activo con modelo de mercado, un 1,6 % de las filas | Versión nueva de los rasgos, fuera de la campaña en curso: exige volver a preparar las entradas de cada origen y su efecto esperado es pequeño |
| Error típico con bloques centrado en la media global | Medido con los contrastes de la v10: infla el error entre un 0 % y un 4,7 % (mediana 1,8 %) respecto a centrar en cada bloque, y un 0,7 % en el nivel del núcleo. No se dan el 18 % ni el 86 % de su ejemplo | Se deja como está: es conservador y pequeño. El resumen entre orígenes lleva aparte el error entre orígenes (I-1) |
| EE. UU. y China como bloques independientes en la conjunta | Medido (I-1): la correlación del mismo día entre mercados es de +0,008 de mediana en las diferencias pareadas y de +0,01 a +0,09 en niveles | Los bloques independientes valen: el error queda subestimado en torno a un 1 % |
| Promoción con una semilla y sin corrección múltiple | Cierto; la criba solo decide qué se sigue midiendo | La evidencia sale de los orígenes limpios, con sus contrastes pareados |
| Doce hallazgos en el núcleo agéntico | Comprobados uno a uno por el programa de post-entrenamiento: cinco son reales y están corregidos con una prueba que antes fallaba (la realimentación usaba otra correlación que la métrica declarada, un eje mal documentado, el orden de las semillas sin validar, un contexto que no cabía en la mezcla y la frontera causal por mercado); el resto son propiedades declaradas del diseño o no se reproducen | Los arreglos están en la rama `postentrenamiento-v10` y entran en la base nueva con ella |
| Kit de Colab | Suponía subir las copias empaquetadas; su verificador dejó objeciones | No se usa. El kit propio se apoya en el flujo de segundo equipo ([guía](../engineering/colab-gpu-training.md)) |

### Tercer enjambre de Gemini (22 de noviembre)

Ocho trabajadores y un verificador, con entregas incrementales, y un encargo directo sin enjambre
para rasgos nuevos. Como en el segundo, lo que sigue separa lo medido aquí de lo propuesto.

| Pieza | Estado | Decisión |
| --- | --- | --- |
| Agregador causal en línea para el núcleo agéntico | Sus pruebas pasan aquí; su verificador deja objeciones (un mutante tautológico, defectos de contrato). Con datos reales no mejora a la media de rangos (A-1) | No se adopta. El experimento con órganos de memoria como componentes y la comprobación de las siete carencias que señala en el combinador van al programa de post-entrenamiento |
| Memoria supervisada con etiquetas maduras (regla delta con compuerta, multiescala) | Solo datos sintéticos: empata con la memoria por activo a igual semivida; el subespacio contextual gana si la deriva depende del estado y diluye con poca señal | Medida con datos reales encargada al programa de post-entrenamiento, primero sin subespacio |
| Ganancia de la mediana y memoria en línea | Derivación (la cabeza y la ganancia comparten una simetría de escala que, sin decaimiento de pesos, deja sin freno a la escala de la mediana) y microrréplica en CPU con el predictor real y datos sintéticos. Sin medida con datos reales. En la red tabular un decaimiento de 0,01 no ayudó (H5-a), pero esa red no tiene ganancia ni memoria | Candidata a la segunda criba de la base nueva: decaimiento de pesos de 0,01 sobre el núcleo con ganancia, en los tres modos de memoria. Antes, su auditoría de trazas sobre los ajustes de la v10, que no usa GPU |
| Objetivo operable (primera pasada de tres) | Señala en las herramientas de medida: la rejilla del factor acaba en 3, el factor queda en 0,01 sin aviso si no hay sesiones de calibración y falta el IC operable del conjunto de semillas | Se comprueban con pruebas y se corrigen antes del informe de la criba |
| Auditoría de código (16 hallazgos, 13 parches) | Primera pasada. Seis están en la rama de post-entrenamiento y diez en herramientas propias (informe, contrastes, guiones de Colab, guardia, búsqueda) | Los de post-entrenamiento, enviados a ese programa. Los propios se integran con sus pruebas cuando cierre su verificador; la cota del origen en 2023 del guion de Colab entra en cualquier caso |
| Estado del arte tabular, Colab con RTX PRO 6000, macro con compuerta | En curso | — |
| Rasgos nuevos (encargo directo): contabilidad dinámica y anuncios, familias de Alpha158 que faltan, recuento de noticias, transformaciones relativas al sector | Código y pruebas con datos sintéticos, en curso. Cobertura medida de los hechos contables en 300 activos: del 13 % al 55 % según el concepto, sin cuenta de resultados ni acciones en circulación | Se calculan con los datos reales y se miden con árboles y con la red tabular, familia a familia, antes de tocar las entradas de la campaña |

## Criba de la base nueva: atención entre activos sobre el trío sin ganancia

Los dos cambios que ayudan a la memoria en línea se midieron por separado y nunca juntos: quitar
la ganancia de la mediana (+0,0031, t = 3,0) y mezclar los activos de la sesión (+0,0021, t = 2,3).
Los dos pierden con la memoria congelada y sin memoria. La criba v11 lleva desde el 22 de noviembre
la mezcla sobre el trío sin ganancia, con su control `self_only` y, en un nivel aparte, sus dos
modos de memoria. Regla de lectura, escrita antes de ajustar: pasa a ser la base de la receta final
si supera al trío sin ganancia en evaluación y en decisión y su control queda por debajo; con una
semilla es una criba y no evidencia, que saldrá de los orígenes limpios.

## Segunda criba de la base nueva: reglas de lectura, escritas antes de ajustar

La segunda criba (`us-campaign-v11-screen2.json`, copia b443f803) ajusta cinco modelos en 2013 con
la semilla 42. Con una semilla es una criba; lo que salga solo decide qué entra en los ajustes
finales, y allí lleva su propia declaración.

- **Los dos términos de IC** (`titans_mac_nogain_icboth`, `titans_mac_icboth`). Predicción: IC operable
  al menos 0,0030 por encima de su base y etiqueta devuelta a menos de 0,0010 de la etiqueta de su
  base. Si se cumple sobre el trío sin ganancia, es el objetivo de los ajustes finales y se ajustan
  sus dos modos de memoria. Si el operable no sube 0,0015, el término se queda fuera.
- **Decaimiento de pesos de 0,01** (`titans_mac_wd1e2`, `titans_mac_nogain_wd1e2`). Predicción del
  tercer enjambre: con la ganancia, el núcleo en línea sube al menos 0,0020 en evaluación y su
  pérdida asociativa baja; sin la ganancia no cambia. Se adopta con la ganancia solo si además
  alcanza al trío sin ganancia (0,1163); si no, queda como explicación del mecanismo y no como receta.
- **La combinada** (`cross_asset_attention_nogain_icboth`). Es la receta de un solo modelo para los
  ajustes finales si supera en evaluación y en decisión a sus dos padres, el mismo modelo sin atención
  y la atención sin los dos términos (primera criba), en la etiqueta devuelta y en lo operable.
