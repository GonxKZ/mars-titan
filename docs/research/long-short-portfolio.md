# Cartera larga y corta por cuartiles

Estado: regla declarada el 9 de octubre de 2026 antes de cualquier predicción real,
implementada y comprobada con datos sintéticos. No se ha ejecutado con predicciones
de la campaña ni con la edición real de precios. El bloqueo de aprendizaje sigue
vigente y la reserva de 2024 sigue cerrada.

La [cartera del protocolo](protocol.md#métricas-e-interpretación) traduce las
predicciones residuales a un rendimiento porcentual. Es un análisis financiero
secundario. Sirve para describir y contrastar brazos, pero no para elegir modelos,
que se eligen con el MAE residual de validación. La declaración está en la sección
`long_short` de la [comparación](../../configs/evaluation/historical-masked-2000-comparison.json)
(versión 4) y la calculan `evaluation/long_short.py` y
`evaluation/long_short_comparison.py`.

## Regla

- **Sesión y señal.** Cada sesión es un mercado y un instante de decisión. La
  señal es la predicción puntual emitida en esa decisión, la misma que puntúa la
  comparación. Nunca se mezclan Estados Unidos y China en una cartera, porque sus
  monedas y calendarios son distintos. Con el modelo conjunto se forman carteras
  por mercado.
- **Selección.** Con $n\ge 30$ filas evaluadas en la sesión, $k=\lfloor 0{,}25\,n\rfloor$.
  Una fila es larga si como mucho $k$ filas tienen una predicción mayor o igual
  que la suya, y corta si como mucho $k$ la tienen menor o igual. Un grupo empatado
  que cruza la frontera queda fuera entero. Por eso una predicción constante,
  como la del control cero, no abre posiciones.
- **Pesos.** Cada larga pesa $+0{,}5/k$ y cada corta $-0{,}5/k$ del capital de la
  sesión: exposición neta cero y bruta uno. Lo que no se ejecuta queda en efectivo
  y no se renormaliza, así que la exposición real se publica aparte.
- **Ejecución.** Entrada en la apertura de la sesión siguiente y salida en su
  cierre, partiendo de efectivo en cada sesión. Es el horizonte del objetivo
  residual, que mide la apertura a cierre de $t+1$. La decisión solo usa
  información disponible al cierre de $t$.

## Precios y reglas de mercado

Los precios proceden de la [edición sin ajustar reconstruida](../data/unadjusted-prices.md)
mediante `simulation/session_prices.py`, con las reglas de la cinta de la RL:

- Solo se ejecutan filas verificadas con volumen positivo y apertura en la rejilla
  de cotización. Nada se rellena. Una sesión con dividendo y split a la vez no se
  ejecuta, porque la edición no permite saber qué precio corresponde a cada base.
- Dividendos y splits de la fecha ex no cambian una posición de apertura a cierre:
  los dos precios comparten base y quien compra en la apertura ex no cobra el
  dividendo.
- En China se aplican las [reglas de las acciones A](../engineering/china-market-rules.md)
  (`cn_a_share_v1`). La referencia de la banda es el último cierre negociado
  verificado anterior, menos el dividendo y dividida por el split de esa apertura.
  Una compra no se ejecuta con la apertura en el límite superior o por encima, ni
  una venta en corto con la apertura en el inferior o por debajo. Sin referencia o
  sin tablero acreditado no se opera. El timbre se cobra en cada venta según su
  fecha.

## Rendimiento y costes

Para una fila ejecutada con peso $w_i$, apertura $o_i$ y cierre $c_i$:

$$
r_s=\sum_i w_i\Bigl(\frac{c_i}{o_i}-1\Bigr),\qquad
N_s=\sum_i |w_i|\Bigl(1+\frac{c_i}{o_i}\Bigr),\qquad
r^{\mathrm{neto}}_s=r_s-\kappa N_s-T_s,
$$

donde $N_s$ es el nocional negociado de entrada y salida, $\kappa$ el coste por lado
(0, 5, 10 y 20 puntos básicos, escenarios ilustrativos del protocolo) y $T_s$ el
timbre chino de las ventas. Cada sesión parte de efectivo, así que todas las
operaciones de apertura y cierre pagan coste.

## Estadísticos e incertidumbre

Sobre la serie diaria de rendimientos netos se informan el rendimiento acumulado
y anualizado (riqueza compuesta desde 1), la media diaria, la volatilidad anualizada
con $\sqrt{A}$, el Sharpe sin tipo libre de riesgo, el drawdown máximo y la rotación
media $\overline{N}$. La anualización usa $A=252$ sesiones en Estados Unidos y $A=243$
en China. Una sesión con pérdida del 100 % o mayor arruina la serie. El Sharpe no
está definido con volatilidad nula.

Las semillas de un brazo se promedian sesión a sesión antes de calcular
estadísticos, lo que equivale a repartir el capital entre las tres. Cada semilla
conserva además su resumen. La incertidumbre usa el bootstrap circular por bloques
de sesiones del mercado con la longitud de bloque, réplicas y semilla de la
comparación. Todas las series se remuestrean con los mismos índices y en el orden
de cada réplica, porque el drawdown depende del recorrido. Los contrastes usan las
familias de la comparación, con la corrección por máximo estudentizado dentro de
cada familia, coste y estadístico. El informe indica qué sentido es mejor en cada
estadístico. La volatilidad y la rotación no tienen un sentido preferido.

## Simplificaciones declaradas

El informe copia estas simplificaciones tal cual, junto con la declaración:

- T+1 impide vender en la misma sesión lo comprado en la apertura. La regla liquida
  al cierre, así que el resultado chino es una cota sin esa fricción.
- Se supone préstamo disponible sin coste para las cortas. En acciones A el
  préstamo solo existe para una lista de valores y no hay datos de disponibilidad.
- La salida al cierre se supone ejecutada. En China se cuentan las salidas con el
  cierre en la banda que la bloquearía.
- Pesos continuos, sin lotes, participación ni impacto.
- Coste por lado igual en ambos mercados y fechas, sin préstamo de valores.
- La población está condicionada a seguir cotizando en marzo de 2025, sin bajas.
- El efectivo no remunera y no se resta un tipo libre de riesgo.

## Informe

`long_short.json` publica, por mercado, la declaración, las simplificaciones, las
filas por motivo de ejecución en cada ventana (ejecutable, sin fila, sin verificar,
sin volumen, fuera de rejilla, evento ambiguo, tablero sin reglas o sin cierre de
referencia), las filas seleccionadas que no se llenaron, los estadísticos por
semilla y de la media con su intervalo y los contrastes por coste. `sessions.parquet`
conserva por sesión, brazo y semilla el rendimiento bruto, el nocional, el timbre,
la exposición real, las filas seleccionadas y llenadas, las salidas en banda y el
rendimiento neto con cada coste.

## Comprobaciones

Las pruebas (`tests/evaluation/test_long_short.py`,
`tests/evaluation/test_long_short_comparison.py` y
`tests/simulation/test_session_prices.py`) calculan a mano la selección con empates
y con diez activos, el libro con timbre, el Sharpe, la volatilidad y el drawdown,
la ruina, el emparejamiento y el orden de las réplicas, una predicción con
conocimiento perfecto del rendimiento (bruto 0,03 y rotación 2 por sesión), los
motivos de no ejecución, las bandas chinas y la media de semillas. La mutación
dirigida de la regla, el libro, los estadísticos y el remuestreo mató todos los
defectos introducidos, después de añadir pruebas para los que sobrevivieron en la
primera pasada (redondeo de $k$, drawdown sin capital inicial, réplicas desordenadas
y primera semilla en lugar de la media).

## Qué no permite afirmar

El rendimiento de una cartera con costes ilustrativos no es una estimación de lo que
se obtendría operando. No incluye préstamo, impacto ni restricciones de capacidad, y
la población tiene sesgo de supervivencia. Un Sharpe alto en el periodo de evaluación
no prueba que la señal sea útil fuera de él. El resultado se interpreta siempre junto
con la pregunta predictiva principal.
