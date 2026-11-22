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

## Estado medido (22 de noviembre de 2026, origen 2013, semilla 42, IC de evaluación)

| Pieza | Medida | Lectura |
| --- | --- | --- |
| Receta original con rasgos | 0,0776 | Punto de partida |
| Base con término de IC y rama densa | 0,1107 | El término aporta 0,007 y la rama 0,008 |
| Base con rangos por sesión en la rama | 0,1163 | +0,0055 (t = 4,4); supera a los árboles (0,1122; p = 0,017) |
| Memoria: sin ella, congelada, en línea | 0,1049, 0,1094, 0,1107 | La memoria aporta 0,006, pero aprender en línea solo 0,0013 (t = 1,2) |
| Lectores episódicos y núcleo cognitivo | igual que su núcleo | Con el término de IC en su pérdida, +0,0008 a +0,0012 |
| Atención entre activos | +0,0002 | Sin efecto |
| Escala de la predicción | dispersión 1,1 a 1,9 frente a 0,14 | El término infla la mediana; la ganancia aprendida la corrige (tabular) |

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

## Interacción observada: rangos por sesión y memoria en línea

Con una semilla, los rangos por sesión suman 0,0055 al núcleo con memoria en línea, 0,0004 al
de memoria congelada y 0,0010 al núcleo sin memoria. Con los rangos, aprender en línea queda
0,0065 por encima de la memoria congelada (0,0013 sin ellos). Puede ser una semilla afortunada:
la campaña v10 lo repite con tres semillas en los tres núcleos antes de darlo por cierto.

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
| Diez cambios de arquitectura (recorte suave, escala por rasgo, rangos de fundamentales, conjunto implícito, separar gráficos y rasgos técnicos, incrustaciones periódicas, precios por parches, atención por factores, fichas de ausencia, tronco ancho) | Propuestas con su control y su criterio de abandono | Candidatas a la criba en Colab, de una en una |
| Iliquidez de Amihud con días sin volumen | Confirmado en el código y en los datos (D-1) | Versión nueva de los rasgos, fuera de la campaña en curso |
| Error típico con bloques centrado en la media global | Confirmado en el código; infla el error cuando los bloques tienen medias distintas (conservador) | Corregir al escribir el informe de varios orígenes, con el error entre orígenes aparte |
| EE. UU. y China como bloques independientes en la conjunta | Confirmado en el código; su efecto depende de una correlación que no se ha medido | Medir la correlación diaria entre mercados con los resultados de la conjunta |
| Promoción con una semilla y sin corrección múltiple | Cierto; la criba solo decide qué se sigue midiendo | La evidencia sale de los orígenes limpios, con sus contrastes pareados |
| Doce hallazgos en el núcleo agéntico | Sin medida propia | Los comprueba el programa de post-entrenamiento con una prueba por hallazgo |
| Kit de Colab | Suponía subir las copias empaquetadas; su verificador dejó objeciones | No se usa. El kit propio se apoya en el flujo de segundo equipo ([guía](../engineering/colab-gpu-training.md)) |
