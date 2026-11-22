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

**Experimento H1-a (en curso).** Con los 46 rasgos y una red congelada: reestimar en línea la
última capa por mínimos cuadrados con olvido exponencial y contracción hacia los pesos
ajustados, con las etiquetas maduras; lo mismo sobre las entradas; una memoria por activo del
residuo maduro; el placebo de etiquetas barajadas, y el reajuste de la red con datos más
recientes como referencia. Guion: `tab_mem.py` en los registros de la investigación.

### H2. Dos años de obsolescencia de los pesos externos cuestan orden

**Hipótesis.** Reajustar con datos hasta el año anterior, a épocas fijas, o continuar en
línea con etiquetas maduras, recupera varios milésimos de IC. Se mide con H1-a (referencia de
reajuste) y, en los Titans, comparando en un mismo año las predicciones de orígenes distintos.

### H3. La macro solo puede ordenar a través de la sensibilidad de cada activo

**Hipótesis.** Los índices macro son iguales para todos los activos de una sesión. Solo
cambian el orden si se cruzan con la sensibilidad de cada activo a sus variaciones. Una
herramienta que calcule esas sensibilidades con datos pasados (o una memoria por activo que
las aprenda de las etiquetas maduras) es la forma de que el modelo aprenda cómo se relacionan.
Sin medir todavía.

### H4. Los lectores no aportan por sus claves y por lo que guardan

**Hipótesis.** Las claves son una proyección aleatoria de las entradas crudas, el banco tiene
1.024 entradas (menos de un cuarto de una sesión) y la admisión por sorpresa lo llena de
saltos irrepetibles. Claves desde el estado del núcleo, valor igual al residuo maduro del
padre y admisión de lo reciente deberían cambiarlo. Antes hay que separar cuánto de la mejora
de los lectores con el término de IC viene del banco y cuánto de sus capas ajustadas.

### H5. Optimización en un recorrido cronológico no estacionario

**Hipótesis.** Con 400 a 1.000 actualizaciones por época, el segundo momento de Adam
(0,999) tarda de una a dos épocas en adaptarse; 0,95 lo haría seguir el régimen. Acumular
sesiones enteras en lugar de 8.192 filas daría pasos más estables.

## Registro de experimentos

| Id | Fecha | Pregunta | Variable | Control | Resultado | Decisión |
| --- | --- | --- | --- | --- | --- | --- |
| H1-a | 22-11 | ¿Aprender de las etiquetas maduras añade orden sobre una red congelada? | Cabeza reestimada en línea | Red estática; placebo | En curso | — |

## Propuestas externas pendientes de medir

La cuarta ronda de Gemini (22 de noviembre) propone, por orden de ganancia esperada: reajuste
hasta el año anterior, memoria rápida compartida por sesión con etiquetas maduras, escritura
supervisada por activo, segundo momento de Adam de 0,95, claves aprendidas en el banco
episódico y mezcla del agente con escala conservada. Sus rangos de ganancia son conjeturas
suyas, no medidas.
