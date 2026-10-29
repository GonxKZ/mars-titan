# Escalera para reutilizar S en la segunda ola

Estado: declarada el 29 de octubre de 2026, antes de cualquier resultado de S. Está
implementada en `src/mars_titan/evaluation/reusable_holdout.py` y comprobada con series
escritas en las pruebas. No se ha aplicado a ninguna criba, porque ninguna se ha ejecutado.
El aprendizaje sigue bloqueado y la reserva de 2024 sigue cerrada.

## Problema

La [criba de variantes](variant-screening-v1.md) juzga cada variante de la primera ola en S,
los años de evaluación 2019 a 2022 de sus cuatro orígenes. La segunda ola volverá a usar ese
mismo S con modelos que pueden diseñarse o ajustarse después de ver los resultados de la
primera. Cada vez que una decisión depende de lo visto en S, las siguientes cifras de S dejan
de ser una estimación sin sesgo, y con suficientes intentos un modelo puede parecer mejor solo
por haberse adaptado a S.

## Mecanismo

La escalera de Blum y Hardt (2015) limita lo que S revela. Mantiene el mejor envío publicado
y, para cada envío nuevo, solo publica su puntuación si mejora claramente a la del mejor. Si
no mejora, el tablero repite el valor anterior. El artículo demuestra para la versión con paso
fijo (figura 1, teorema 3.1) que el error del tablero crece con el logaritmo del número de
envíos y no con su número, y propone la versión sin parámetros (figura 2) para la práctica.
MARS-TITAN usa esta segunda con dos cambios declarados.

Para un envío t con serie por sesión $s_t$ y el mejor publicado $b$, en las sesiones $C$ que
definen los dos,

$$
\bar d=\frac{1}{|C|}\sum_{i\in C}\bigl(s_{t,i}-s_{b,i}\bigr),\qquad
\text{mejora}\iff \bar d>\widehat{\operatorname{se}}(\bar d),
$$

y entonces el tablero publica $[\bar s_t]_{0{,}001}$, la media del envío en sus sesiones
definidas redondeada a milésimas, y el envío pasa a ser el mejor. La puntuación es el IC por
sesión de la criba, en el que mayor es mejor, así que la condición es la del artículo con la
pérdida cambiada de signo. Sin un mejor inicial, el primer envío se publica siempre, como con
$R_0=\infty$.

| Elemento | Artículo | MARS-TITAN | Motivo |
|---|---|---|---|
| Error típico | $\operatorname{std}(\ell_t-\ell_{t-1})/\sqrt n$ | Bootstrap circular por bloques de 16 días UTC, con las 2.000 réplicas y la semilla de la criba | Las sesiones consecutivas no son independientes. Con bloques de un día el bootstrap reproduce la fórmula del artículo, como comprueba una prueba |
| Sesiones | Las mismas n filas para todos | Las que definen a la vez el envío y el mejor | El IC de una sesión no existe si las predicciones son constantes |
| Redondeo | Múltiplos de 1/n | Múltiplos de 0,001 | Con unas 1.000 sesiones en S es el mismo orden |
| Mejor inicial | Ninguno | El mejor modelo ya publicado por la primera ola | Sus cifras ya se conocen y la segunda ola debe superarlas |

## Qué se publica y qué se sella

La [declaración](../../configs/experiments/reusable-holdout-v1.json) separa dos partes. La
pública solo contiene el nombre de cada envío, el valor del tablero y si mejoró. La sellada
guarda su IC exacto, las sesiones definidas y comunes, la diferencia emparejada, su error
típico y el mejor contra el que se comparó. Las cifras selladas se abren una sola vez, después
del último envío de la segunda ola y antes de la confirmación en E, con su fecha y su commit.

El orden de los envíos es el orden en que el motor sella el análisis de cada modelo, y los que
se sellan a la vez siguen el orden del catálogo. La escalera no sustituye al contraste de la
criba ni a la confirmación. Solo controla qué cifras de S pueden influir en la elección de los
siguientes modelos.

## Límites

- Con envíos adaptativos el contraste de un error típico es heurístico. Blum y Hardt lo sitúan
  aproximadamente en un nivel de 0,15. No es una prueba con tamaño garantizado.
- La escalera solo protege lo que la segunda ola reutiliza. Las cifras de la primera ola ya
  publicadas siguen a la vista.
- El análisis de la segunda ola todavía no existe. Su integración en el motor de la criba
  llegará con el protocolo de esa ola, que deberá llamar a `run_ladder` con las series por
  sesión que ya calcula `variant_screening_analysis` y escribir la parte sellada fuera del
  informe público.

## Comprobación técnica

Las pruebas de `tests/evaluation/test_reusable_holdout.py` comprueban que el valor publicado es
el múltiplo más cercano de la resolución, que el error típico es la dispersión de las medias del
bootstrap con los sorteos de `circular_block_counts`, que las sesiones de un mismo día UTC se
sortean juntas, también cuando `run_ladder` agrupa los instantes de decisión, que con bloques de
un día se acerca a $\operatorname{std}/\sqrt n$, que un envío peor o con una mejora menor que su
error típico deja el tablero como estaba y conserva el mejor, que un empate exacto no cuenta
como mejora, que la diferencia usa solo las sesiones comunes, que el mejor inicial se publica y
se compara como los demás, que la parte pública solo trae los campos declarados y que dos
ejecuciones iguales dan el mismo tablero. También rechazan envíos sin nombre, con máscaras de
otra forma o tipo, valores no finitos en sesiones definidas, nombres repetidos, días
desordenados, envíos sin ninguna sesión común con el mejor o con un solo día común, opciones
fuera de su tipo y declaraciones que no describen este mecanismo.

## Referencia

Blum, A. y Hardt, M. (2015). The Ladder: A reliable leaderboard for machine learning
competitions. En F. Bach y D. Blei (Eds.), *Proceedings of the 32nd International Conference
on Machine Learning* (Vol. 37, pp. 1006–1014). PMLR.
https://proceedings.mlr.press/v37/blum15.html
