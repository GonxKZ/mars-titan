# Decisión sobre la destilación temporalmente válida

Autor: Gonzalo García Lama. Decisión: 13 de octubre de 2026. Evidencia de MT-055 (#57).

## Decisión

La destilación con profesor `mars_titan_m1` y alumno `titans_mac_online` queda implementada, declarada y sin activar. De sus dos condiciones de activación, la de coste se cumple en la vista medida, donde el profesor predice entre 4,6 y 6,8 veces más lento que el alumno. La otra, que el profesor mejore al alumno directo en la validación de la campaña A, no se puede comprobar mientras rija el bloqueo de aprendizaje. Hasta entonces no se genera ninguna salida del profesor ni se ajusta ningún alumno.

El par que preveía la issue, profesor con K = 4 y alumno con K = 1, se descarta con la medida. K = 4 costó entre 0,71 y 1,31 veces lo que K = 1 al predecir, así que no habría inferencia que ahorrar.

El [diseño, las salidas permitidas y las comprobaciones](../../docs/engineering/temporal-distillation.md) están en su documento de ingeniería y la [declaración](../../configs/evaluation/temporal-distillation.json) fija el contraste antes de ejecutar nada.

## Cortes y salidas

- Profesor: `mars_titan_m1`, ajustado con el tramo de ajuste de la ventana y elegido con su validación, como en la campaña.
- Salidas: una pasada cronológica del estado elegido sobre ese mismo tramo, con banco vacío y memoria rápida inicial. Son objetivos dentro de la muestra, sin etiquetas, y solo entran en la pérdida del alumno.
- Alumno: `titans_mac_online` con la pérdida de su receta mezclada con la distancia L1 a las salidas del profesor, peso 0,25 o 0,5 elegido con la validación de la ventana y calibración propia.
- Comparación: alumno destilado frente a directo y frente al profesor sobre la evaluación de las ventanas walk-forward. La reserva de 2024 no se abre.

## Coste medido

Inferencia en `cuda:0` sobre la ventana `fold-012` de 64 activos de EE. UU., en tres rondas sin pasos de optimizador ([recibo](../engineering/temporal-distillation-cost-20261013.json), con el detalle en el [documento de ingeniería](../../docs/engineering/temporal-distillation.md#coste)):

- Alumno `titans_mac_online`: de 1.678 a 2.474 filas/s.
- Profesor `mars_titan_m1`: de 298 a 511 filas/s, entre 4,6 y 6,8 veces el coste del alumno en la misma ronda.
- `mars_titan_m1_k4`: de 278 a 421 filas/s, entre 0,71 y 1,31 veces el coste de M1.
- Salidas del profesor: una pasada por el tramo de ajuste, unas 0,76 h de GPU por millón de filas con la mediana medida.
- Alumnos destilados: dos ajustes por ventana y semilla, uno por peso, cada uno con el coste del alumno directo, más su calibración.

Durante la medida había otros trabajos de CPU en el equipo, así que los caudales absolutos son bajos y los cocientes también pueden cambiar sin esa carga. La medida del 10 de octubre en la vista conjunta US+CN, con cohortes mucho mayores y código anterior, dio un cociente de 1,78 entre profesor y alumno. El presupuesto en horas se fijará con los recuentos de la campaña A y una medida nueva en su vista antes de generar ningún objetivo, como exige la declaración.

## Calidad y calibración

No hay ningún resultado. El bloqueo de aprendizaje impide ajustar el profesor y los alumnos, y la calibración propia del alumno solo se puede medir después de ajustarlo.

## Qué haría cambiar la decisión

- Se activa si el profesor mejora al alumno directo en la validación de la campaña A, con el límite superior del intervalo del 95 % por debajo de cero, y una medida con el código vigente en la vista de la campaña mantiene el cociente de inferencia en 1,5 o más. Entonces se fija el presupuesto, se conecta la generación por ventana y se ajustan los dos alumnos.
- Se descarta con esa evidencia si el profesor no mejora al alumno directo o si el cociente baja de 1,5 en la vista de la campaña.
- Una vez activada, el alumno destilado se descarta si no mejora al directo con el límite superior por debajo de cero o si el ahorro de inferencia no compensa la generación de salidas y los ajustes adicionales. En cualquier caso se informan los dos pesos, el alumno directo y el profesor.
