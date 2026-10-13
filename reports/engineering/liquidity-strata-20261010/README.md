# Estratos de liquidez: comprobación con datos reales y mutación

Recibos del 10 de octubre de 2026 para [#532](https://github.com/GonxKZ/mars-titan/issues/532).
La declaración, la asignación y las métricas están en la
[sección de estratos de liquidez](../../../docs/research/metrics.md#estratos-de-liquidez-y-métricas-cuadráticas).
Ningún paso lee objetivos, predicciones ni el año sellado, no se ha usado la GPU y no se ha
ejecutado ningún paso de optimizador.

## Comprobación con datos reales

[`real-data.json`](real-data.json) es el recibo de `benchmarks/liquidity_strata_real.py` sobre
la edición sin ajustar `1ac37278…`, con el código del commit `2cd8a352`. Después solo cambian
los docstrings del propio script y esta documentación.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python \
  benchmarks/liquidity_strata_real.py --edition <edición sin ajustar> --output <recibo>.json
```

- **Población.** Asignando cada fila de la edición como sesión publicada de una decisión, de
  2000 a 2023 US tiene un 85,7 % de filas `liquid`, un 1,8 % `low_price`, un 3,2 %
  `thin_volume`, un 0,26 % en los dos y un 9,1 % sin clasificar (16,1 millones de filas). Desde
  2018 la parte sin clasificar sube a entre el 13,5 % y el 16,2 % de cada año. En CN el 90,3 %
  es `liquid`, el 2,6 % `thin_volume`, 14 filas `low_price` y el 7,1 % queda sin clasificar.
- **Motivos.** Las filas sin clasificar de US se reparten entre 1.400.714 sin cierre negociado
  verificado y 68.225 con la ventana de volumen incompleta. En CN son 188.991 y 14.866.
- **Causalidad.** 400 activos elegidos con la semilla 20261010 se truncaron en una sesión al
  azar. Las 778.631 filas anteriores al corte conservan su código en todos los casos.
- **KGJI.** Sus 251 filas de 2022 caen en `low_price_and_thin_volume`, con cierres entre
  0,0001 y 0,01 USD y una mediana de volumen de cero títulos. Es el activo que dominaba el MSE
  del diagnóstico de #16 con una sola fila.
- **Límite de la cartera.** El 1 % del volumen de la sesión publicada por su último cierre
  negociado tiene una mediana de 46.021 USD en `liquid`, 445 USD en `low_price`, 32 USD en
  `thin_volume` y cero en los dos a la vez. La etapa de políticas no puede abrir posiciones
  relevantes en esas filas.

## Coste

| Medida | US | CN |
| --- | ---: | ---: |
| Filas por mes (todas las decisiones de todos los activos) | 79.838 a 84.040 | 13.770 a 16.200 |
| Asignación con una instancia por mes | 16,8 a 20,6 s | 3,4 a 6,5 s |
| Asignación con una instancia compartida, meses siguientes | 0,18 a 0,23 s | 0,016 a 0,019 s |
| Caché de la instancia compartida | 96,5 MB | 17,4 MB |
| Códigos iguales en los dos modos | sí | sí |

Puntuar los cinco estratos de un modelo con 625.000 filas sintéticas tardó 0,45 s frente a
0,25 s de la ventana completa, y elegir las 100 filas extremas de 100.000, 0,019 s. El recorrido
completo de la edición llevó 45 s y el proceso alcanzó 1.073 MiB. Es el mejor de tres en CPU
con dos hilos y una carga media del equipo de 25 por otros procesos al empezar, así que las
cifras solo ordenan el coste. No se ha medido la energía por falta de instrumento.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 36 mutaciones aplicadas de una en una sobre una copia
aislada de `src`, `tests` y `configs` en el commit `2cd8a352`, con las pruebas importando la
copia y la copia sin mutar en verde (62 pruebas, más las dos de la retención). Las 36 hicieron
fallar alguna prueba:

- En la sesión publicada: una decisión en el mismo instante tratada como no publicada, leer la
  sesión siguiente y tomar una fila posterior.
- En el precio y el volumen: contar un cierre rellenado o sin verificar, no llevar el precio a
  su múltiplo, umbrales inclusivos, la media en lugar de la mediana, una ventana que mira hacia
  delante, dar código a una fila sin clasificar e intercambiar precio y volumen.
- En la asignación: perder el motivo de una fila ausente, admitir activos de otro mercado y
  volver a leer la edición en cada llamada.
- En las filas extremas: no repartir el peso de la sesión entre sus filas, quedarse con las
  menores, ignorar la ponderación por mercado o el número de filas, una fila de más y conservar
  el prefijo del mercado en el activo.
- En la declaración y el informe: no exigir la métrica cuadrática, el MAE fuera del primer
  puesto, no comprobar los estratos, extremos sin ordenar, filas listadas sin límite, ventanas
  de hasta un año, códigos sin reordenar, no informar la métrica cuadrática, ignorar la
  edición o los nombres de los estratos, agregados sin la edición en su identidad y una matriz
  que acepta el MSE sin estratos.
- En la retención y la cartera: retener sin la edición, construir un asignador por ventana y
  tomar el límite del volumen del día de ejecución.

La mutación cubre los fragmentos elegidos, no garantiza que cualquier otro cambio se detecte.
