# Retención e interferencia: comprobación con datos reales y mutación dirigida

Recibos del 10 de octubre de 2026 para [#54](https://github.com/GonxKZ/mars-titan/issues/54).
No se ha ajustado ni evaluado ningún modelo, no se ha leído ninguna predicción ni objetivo y
no se ha ejecutado ningún paso de optimizador. La definición del análisis está en
[métricas](../../../docs/research/metrics.md#retención-e-interferencia-en-las-revisitas-de-régimen).
Su ejecución científica queda pendiente de las predicciones de la campaña.

## Comprobación con datos reales

[`real-data.json`](real-data.json) es el recibo de `benchmarks/retention_interference_real.py`
sobre la edición preparada v3.1 desde 2000, con el código del commit `373739e2`. Los cambios
posteriores de la rama solo añaden pruebas y documentación, y las huellas de los módulos del
recibo coinciden con las del commit final.

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 uv run --no-sync python \
  benchmarks/retention_interference_real.py --prepared <edición preparada v3.1> \
  --calendar <calendario nuevo>.json --output <recibo nuevo>.json
```

- **Calendario.** 6.037 sesiones de US, del 3 de enero de 2000 al 29 de diciembre de 2023, y
  4.372 de CN, del 4 de enero de 2006 al 29 de diciembre de 2023. Solo las 63 primeras de cada
  mercado quedan sin clasificar.
- **Causalidad.** Recortar la historia el 15 de septiembre de 2008 y el 16 de marzo de 2020 en
  US, y el 16 de septiembre de 2008 y el 12 de junio de 2015 en CN, deja idénticas todas las
  rutas, activos y rendimientos anteriores (2.188, 5.082, 658 y 2.293 sesiones).
- **Paridad con la codificación.** 40 sesiones al azar por mercado, recalculadas con
  `price_windows.window_rows` y `RegimeRule.state`, coinciden en ruta, activos y rendimientos.
- **Clases.** Las sesiones de evaluación de los protocolos v2 se reparten como recoge la
  [tabla de métricas](../../../docs/research/metrics.md#recuentos-con-el-calendario-real). Con
  la historia desde el calentamiento solo hay 12 entradas en regímenes nuevos en US y 10 en
  CN.
- **Recuperación.** Sobre la rejilla real se inyectó un beneficio conocido por clase: 0,1 en
  regímenes nuevos, 0,3 en revisitas cortas, 0,2 en revisitas largas y 0 en tramos asentados.
  En los ocho pares declarados y en los dos mercados, retención e interferencia coinciden con
  el valor calculado a mano con error menor que 10⁻¹². Un par idéntico da cero en los cuatro
  estadísticos. Con esos beneficios, los pares del banco y del Transformer en línea mantienen
  la retención y detectan la interferencia, y los dos pares de pesos rápidos dejan la
  retención sin decidir por falta de regímenes nuevos. Son valores inyectados para comprobar
  el cálculo, no resultados de ningún modelo.

## Coste

| Medida | Valor |
| --- | ---: |
| Construir el calendario de US y CN | 125,0 s |
| Pico de RSS del proceso | 1.782,8 MiB |
| Informe de US, 8 pares más el idéntico, 2.000 réplicas (mediana de 3) | 0,249 s |
| Informe de CN, mismas condiciones | 0,163 s |
| Recorte en el 16 de marzo de 2020 en US, 5.082 sesiones | 39,4 s |

CPU AMD Ryzen 9 8945HS, NumPy 2.5.3, Python 3.12.14, dos hilos y una carga media del equipo de
17,8 por otros procesos al empezar, así que los tiempos solo ordenan el coste. El calendario
se calcula una vez por edición y el análisis trabaja sobre las series por sesión que la
comparación ya tiene en memoria, sin GPU ni nuevas predicciones. El recorte en una fecha
recalcula el prefijo completo y solo sirve para la comprobación de causalidad. No se ha
medido la energía, por falta de instrumento, ni el informe completo de la comparación, que
necesita las predicciones de la campaña.

## Mutación dirigida

[`mutations.json`](mutations.json) recoge 39 mutaciones aplicadas de una en una sobre una
copia aislada de `src`, `tests` y `configs` en el commit `b21e5599`, con las pruebas importando
la copia. Cada entrada guarda el fragmento original y el mutado. Las 39 hicieron fallar
`tests/evaluation/test_retention_interference.py`, `test_regime_calendar.py` o
`test_walk_forward_retention.py`, con la copia sin mutar en verde (64 pruebas):

- En las clases: sesiones sin clasificar que cortan rachas, la ausencia con una sesión de más o
  contada desde la primera aparición, la frontera de las sesiones de entrada, todas las rachas
  como nuevas, un placebo con rutas posteriores o igual a la ruta real, los comienzos de
  historia intercambiados, un calentamiento de un mes menos y una historia que ignora el
  calentamiento.
- En los estadísticos: la ausencia larga estricta, la retención frente a los tramos
  asentados, el signo de la plasticidad y del beneficio, contar sesiones sin métrica y
  definirlas solo con la memoria.
- En el bootstrap y las decisiones: otra semilla, bloques de un día, la frontera de las series
  cortas, ignorar los recuentos del placebo, confirmar con un solo intervalo, el sentido de la
  interferencia y de la retención e intervalos marginales en lugar de simultáneos.
- En la declaración y las curvas: una ausencia larga fuera de los tramos, pares repetidos,
  curvas desde la posición cero o con sesiones sin métrica.
- En el calendario: admitir activos a los que falta una sesión, anclar en una sesión ausente,
  marcar presentes las sesiones ausentes, leer una edición que considera el test final, no
  comprobar su marca, admitir instantes repetidos y sobrescribir un calendario.
- En la comparación: el MSE en lugar del MAE, las ventanas de calibración, admitir un
  calendario sin sección declarada y no validar la declaración.

Una primera pasada sobre el commit `373739e2` dejó vivas siete de estas mutaciones: el signo de
la plasticidad, la métrica definida solo con la memoria, ignorar los recuentos del placebo, el
sentido de la interferencia, leer una edición que considera el test final, admitir instantes
repetidos y no validar la declaración al cargar la comparación. El commit `b21e5599` añade una
prueba para cada caso y la segunda pasada las detecta todas. La mutación cubre los fragmentos
elegidos, no garantiza que cualquier otro cambio se detecte.

## Desviaciones respecto al plan de la tarea

La tarea preveía `configs/memory/retention.yaml`, `tests/memory/test_retention_protocol.py` y
`reports/memory/retention.md`. El análisis consume las series por sesión de la comparación,
así que su declaración vive en las dos comparaciones de la campaña, con su huella, y su código
en `evaluation/`. Las pruebas están en `tests/evaluation/` y este directorio sustituye al
informe previsto.

El plan también incluía secuencias sintéticas A, B, A evaluadas con estados de memoria.
Necesitan estados entrenados y, en Titans, adaptar los pesos rápidos a una señal sintética, así
que quedan para después del desbloqueo. Las pruebas cubren el evaluador con series de
beneficio escritas a mano y este recibo lo comprueba sobre las rutas reales.
