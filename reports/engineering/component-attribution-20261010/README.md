# Brazos que faltan para la atribución por componentes

Generado el 10 de octubre de 2026 con `evaluation/comparison_matrix.py missing` sobre la
[matriz de la campaña A](../../../configs/evaluation/comparison-matrix-a.json). No lee
datos, no ajusta nada y no declara ningún brazo en el plan de la campaña. Es material
para decidir qué brazos se añaden, con su coste a la vista.

```bash
uv run --no-sync python -m mars_titan.evaluation.comparison_matrix missing \
  --matrix configs/evaluation/comparison-matrix-a.json \
  --hours reports/engineering/component-attribution-20261010/arm-hours-a-v2-projected.json \
  --output <informe nuevo>
```

## Horas usadas

[`arm-hours-a-v2-projected.json`](arm-hours-a-v2-projected.json) son horas proyectadas, no
medidas. Salen de la proyección de la campaña A v2 de #431 (commit 2656031f): 16.000
muestras-época por segundo, inferencia a tres veces ese caudal y 10 épocas efectivas, con
los recuentos de filas de su informe de ventanas. Con ese supuesto cada brazo neuronal o
lector del ámbito conjunto cuesta unas 94 h, porque el caudal es el mismo para todos. Los
brazos de B6 solo predicen validación, calibración y evaluación con tres semillas, unas
0,46 h sin contar el calentamiento. Cuando se generó, ningún caudal estaba medido en esta
edición. El 10 de octubre se midieron en `cuda:0` los brazos de integración y dos lectores
en una ventana ([coste](../../../docs/engineering/mars-titan-extensions.md#coste-medido-de-los-brazos-nuevos)).
Los lectores ajustaron unas 1.100 filas por segundo sin pasos de optimizador, muy por debajo
del supuesto si una muestra equivale a una fila. El documento de horas no se ha regenerado
con esa medida. Cuando exista el informe de `throughput`, basta con regenerar el documento
de horas y repetir la orden.

Como referencia, la misma proyección da 3.314 h neuronales para la campaña A v2 frente a
unas 580 h útiles en cuatro semanas. Cada lector candidato añadiría cerca de un 3 % a esa
cifra.

## Recuento

De los 362 contrastes de la matriz, 225 solo usan modelos de la campaña, 112 esperan modelos
condicionados o derivados con plan propio y 25 necesitan algún candidato. Ninguno queda
sin brazo con nombre.

El recuento se repitió el 10 de octubre de 2026 al declarar `transformer_compact_online` en la
comparación (#443). Antes figuraba como brazo condicionado y 15 contrastes lo esperaban. Ahora
13 de ellos solo usan brazos de la campaña y los otros 2 siguen esperando otro brazo con plan propio.
Con eso quedaban 196 y 135. Se repitió otra vez al declarar en la comparación los brazos de
integración `mars_titan_m1_k4_first_read`, `mars_titan_b6` y `mars_titan_b6_bias` (#452). Esos
tres brazos ya no esperan, y 23 contrastes más pasan a usar solo brazos de la campaña. Los que
todavía esperan necesitan además un modelo condicionado o derivado. La última repetición es del
mismo día, al declarar los cuatro modelos B6 enrutados de #20 (`mars_titan_b6_regime`,
`mars_titan_b6_calendar`, `mars_titan_b6_regime_banks` y `mars_titan_b6_calendar_banks`) con los
componentes `routing_slots` y `regime_information`. Añaden 6 contrastes que solo usan modelos de
la campaña. En el documento de horas figuran con las mismas 0,46 h proyectadas que los otros
dos modelos B6, porque predicen las mismas filas.

| Brazos con plan propio | Contrastes que esperan |
| --- | ---: |
| `titans_reference_mac` (referencia pública, sin brazo previsto tras #434) | 2 |
| `<brazo>__chain` | 66 |
| `<brazo>__frozen_parent` | 44 |
| `<brazo>__full_continuation` | 44 |

## Candidatos

El nivel 1 tiene código y desbloquea contrastes por sí solo. El 2 también los desbloquea,
pero necesita un cambio de código. El 3 solo sirve junto con otros. Dentro de cada nivel
se ordena por contrastes desbloqueados por cada 100 h. «Solo» supone que llegan los brazos
con plan propio.

| Nivel | Brazo | Qué permite medir | Cambio necesario | Horas proyectadas | Desbloquea solo |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `mars_titan_m2_k4` | Interacción de K = 4 con la escritura por error | Ninguno, otro brazo de la sección `mars_titan` | 94,4 | 3 |
| 2 | `mars_titan_m0_b6_bias` | Efecto de B6 sobre el lector sin contenido | B6 sobre la salida del lector | 0,5 | 2 |
| 2 | `mars_titan_m1_b6_bias` | Efecto de B6 sobre M1 | Igual que el anterior | 0,5 | 1 |
| 2 | `mars_titan_m3_without_error` | Dejar fuera la escritura por error en M3 y repartirla con Shapley frente a anomalía y relevancia | Política de escritura con pesos (0, 1/2, 1/2) | 94,4 | 5 |
| 2 | `mars_titan_m0_frozen_core` | El lector sin contenido sobre Titans congelado | Lector sobre un padre `mac_frozen` | 94,4 | 3 |
| 2 | `mars_titan_m3_frozen_core` | Dejar fuera la actualización en inferencia en el MARS-TITAN completo | Lector sobre un padre `mac_frozen` | 94,4 | 2 |
| 2 | `mars_titan_m1_frozen_core` | Efecto de actualizar Titans con el banco M1 | Lector sobre un padre `mac_frozen` | 94,4 | 1 |
| 2 | `mars_titan_m3_disabled_core` | Dejar fuera la lectura de memoria en el MARS-TITAN completo | Lector sobre un padre `mac_disabled` | 94,4 | 1 |

Los lotes que solo sirven juntos son tres:

| Brazos | Qué permite medir | Horas proyectadas |
| --- | --- | ---: |
| `mars_titan_m0_b6_bias` y `mars_titan_m1_b6_bias` | Si B6 y el banco se solapan | 0,9 |
| `mars_titan_m0_frozen_core` y `mars_titan_m1_frozen_core` | Si la memoria de Titans y el banco se solapan, con su interacción y su reparto de Shapley | 188,8 |
| `mars_titan_m3_disabled_core` y `mars_titan_m3_frozen_core` | Lectura de memoria dentro del modelo completo | 188,8 |

## Lectura

- El único candidato sin código nuevo es `mars_titan_m2_k4`. Responde si las vueltas
  extra del lector rinden más cuando el banco elige por error.
- Los dos de B6 son baratos porque solo predicen, pero B6 no se combina hoy con el banco y
  sus horas no incluyen el calentamiento ni la lectura del lector.
- Saber si la memoria de Titans y el banco episódico se solapan exige el lote de núcleo
  congelado (unas 189 h proyectadas) y cambiar el padre que admite el lector, que hoy solo
  acepta `mac_online`.
- Los dos brazos con plan propio que más contrastes esperan son el control en línea y los
  de la integración. El control en línea no tiene coste estimado porque su regla de
  actualización aún no está fijada. La referencia pública de Titans costaría, como un
  ajuste de `titans_mac_online`, unas 94 h proyectadas, pero la revisión de #434
  recomendó conservar el núcleo y no hay brazo previsto. Sus dos contrastes quedan
  declarados con esa condición.
- El juego de Shapley con las ocho piezas de Titans no se puede completar con ningún
  brazo: 236 de sus 256 coaliciones activan una pieza sin aquella de la que depende.

## Coste de evaluar la matriz

`benchmarks/comparison_matrix.py` prepara un informe walk-forward sintético del ámbito US
con las sesiones reales de las 19 ventanas, los 23 brazos y las 65 series de brazo y
semilla de la campaña A, con siete activos por sesión (597.625 filas de sesión y 48 MB de
tabla). El coste de la matriz depende de las sesiones, los brazos y las familias, no de
los activos, porque solo lee estadísticos por sesión. Los valores son sintéticos y no
dicen nada del mercado.

```bash
uv run --no-sync python benchmarks/comparison_matrix.py prepare --root <raíz nueva> --scope US
uv run --no-sync python benchmarks/comparison_matrix.py measure --root <raíz> --scope US \
  --output <salida nueva>
```

La evaluación completa, medida antes de declarar el control en línea (183 contrastes estimables en 20 familias, vistas en bruto y
calibrada con todas las métricas, ECE del signo y escritura del informe de 5,6 MB) tardó
152 s de reloj y 266 s de CPU de usuario, con un pico de 1,44 GiB. Se ejecutó una vez con
NumPy 2.5.3, PyArrow 25.0.1, dos hilos, sin GPU y con la CPU compartida (carga media
cercana a 23 en 16 núcleos), así que la cifra no es un límite del equipo. Está en
[`matrix-evaluation-cost-us.json`](matrix-evaluation-cost-us.json).

Un perfil con cProfile de la misma evaluación (163 s bajo el perfilador) atribuye 147 s a
las vistas de predicción, de los que unos 97 s son la generación de los índices del
remuestreo por bloques en `paired_comparisons.py`. Cada familia y cada métrica repiten esos
sorteos con la misma semilla y el mismo número de días. Un prototipo que calcula los
recuentos con diferencias acumuladas dio los mismos enteros y entre 1,0 y 3,3 veces menos
tiempo en esa función, y reutilizar los sorteos entre familias evitaría casi todas las
9.280 llamadas. Como la matriz se evalúa una vez por ámbito y tarda del orden de una
décima parte de la comparación walk-forward a la misma escala, el ahorro sería de uno o
dos minutos por evaluación. No se ha cambiado `paired_comparisons.py` en esta tarea.

## Archivos

- [`missing-arms.json`](missing-arms.json): informe completo, reproducible con la orden
  anterior. Una prueba lo regenera y exige que coincida.
- [`arm-hours-a-v2-projected.json`](arm-hours-a-v2-projected.json): horas por brazo del
  ámbito US+CN y sus padres, con su procedencia.
- [`matrix-evaluation-cost-us.json`](matrix-evaluation-cost-us.json): tiempo, CPU y
  memoria de la evaluación de la matriz con las sesiones del ámbito US.
