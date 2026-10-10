# Coste de la comparación y de la cartera a escala real

Medido el 9 de octubre de 2026 con `scripts/benchmark_evaluation_scale.py` sobre datos
sintéticos. No hay predicciones de ningún modelo ni datos de mercado. Las formas son las
de la campaña A en el ámbito US: las filas de calibración y evaluación de cada ventana
del [informe de vistas](../../data/campaign-a-views-20261009.json), el protocolo US v2 y
los 23 brazos de la comparación declarada, con 64 series de brazo y semilla (tres
semillas en casi todos los brazos, una en Ridge y ninguna en el control cero).

## Condiciones

- CPU de 16 hilos compartida con otras cargas. La carga media fue de 8 a 11 en las medidas
  de dos ventanas, 14 en la comparación completa y 17 a 18 en la cartera completa.
  `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` y `CUDA_VISIBLE_DEVICES=-1`.
- Cada medida es un proceso aparte bajo `/usr/bin/time` y dentro de `memslot heavy`
  (límite de 8 GiB). Una sola repetición por tamaño, porque cada una ocupa la plaza de
  memoria pesada que comparten los demás trabajos.
- Cada ventana escribe tres variantes de predicción que los brazos comparten en ciclo,
  para no ocupar 64 veces el disco. La lectura, la comprobación de huellas y la
  puntuación se repiten igualmente para cada brazo y semilla.
- La comparación se mide con la configuración de versión 1, sin estratos ni ablación,
  porque los estratos necesitan las muestras reales de las vistas. La cartera usa la
  versión 4 completa y una edición sintética con 4.126 activos, precios en céntimos
  verificados y sin eventos.

## Resultados

| Medida | Ventanas | Filas de evaluación | Filas de calibración | Tiempo real | CPU de usuario | Pico de memoria |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Comparación walk-forward | 2 | 2.048.955 | 509.476 | 265 s | 240 s | 1,54 GiB |
| Comparación walk-forward | 19 | 12.826.460 | 3.087.269 | 1.678 s | 1.515 s | 1,90 GiB |
| Cartera larga y corta | 2 | 2.048.955 | | 110 s | 105 s | 1,43 GiB |
| Cartera larga y corta | 19 | 12.826.460 | | 860 s | 738 s | 1,96 GiB |

Los registros completos están en los cuatro JSON de esta carpeta. El informe de la
comparación completa con su tabla por sesión ocupó 105 MB y el de la cartera 5,4 MB.
La comparación usó 4.762 días de remuestreo, bloques de 16 días y 2.000 réplicas.

## Lectura

- El tiempo crece de forma casi lineal con las filas leídas: multiplicar por 6,26 las
  filas de la medida de dos ventanas da 1.659 s frente a 1.678 s medidos. La puntuación
  de cada brazo, semilla y ventana domina el coste, unos 2 µs por fila y serie. El
  remuestreo sobre 4.762 días es una parte pequeña.
- La memoria crece poco con las ventanas, porque cada ventana se puntúa por separado y
  solo se conservan los estadísticos por sesión.
- La cartera cuesta la mitad que la comparación. Lee solo las columnas de evaluación y
  carga los precios de cada activo una vez por ventana.

## Extrapolación a la campaña A v2

No son medidas. Suponen el mismo coste por fila y serie.

- Ámbito conjunto US+CN con 19 ventanas: unos 15,4 millones de filas de evaluación
  (12,8 en US y unos 2,6 en CN), un 20 % más que la medida US. Unos 34 minutos para la
  comparación y 17 para la cartera.
- Ámbitos US y CN de los tres controles separados con sus brazos conjuntos prestados
  (18 series): unos 8 y 1,3 minutos.
- Comparaciones de los adaptadores de las cinco referencias (102 series en total sobre
  el ámbito conjunto): alrededor de una hora.

Las cifras dependen de la carga concurrente y no son un límite del equipo. Ninguna
justifica por ahora otra implementación, frente a las horas de GPU del ajuste.

## No medido

- Los estratos por presencia de modalidades, que leen las muestras de las vistas reales.
- La ablación de modalidades, que depende de su propia etapa.
- La lectura en formato compacto que estudia el presupuesto de disco, todavía sin
  decidir. Todas las lecturas pasan por `walk_forward_comparison._read_predictions`.
