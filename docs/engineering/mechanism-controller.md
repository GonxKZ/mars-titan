# Recepción de resultados para el controlador de mecanismos

La integración de un controlador de mecanismos sigue en curso en [#701](https://github.com/GonxKZ/mars-titan/issues/701). Está implementado el acumulador de feedback de PT2. Todavía no existe el ejecutor común de rutas, contexto PT4 y calibración con confirmación en `FinancialSession`. Este componente no activa una política ni modifica las variantes por defecto.

[`CohortFeedback`](../../src/mars_titan/calibration/cohort_feedback.py) recibe cohortes completas de los cinco cuantiles financieros y conserva sus identidades y fechas de madurez por fila. Su calibrador es el [`OnlineConformal`](../../src/mars_titan/calibration/online_conformal.py) existente. La madurez la aporta el contrato temporal que use el consumidor. El acumulador no la sustituye por un día añadido a la fecha de decisión.

`emit` devuelve los cuantiles en el orden recibido, el registro de emisión y un snapshot nuevo. Internamente ordena las filas por identidad, junto con sus cuantiles y madurez. No lee etiquetas ni modifica el snapshot original. Las propuestas descartadas no se publican por el mero hecho de calcularlas.

`feedback` recibe resultados tipados con mercado, decisión, identidad y disponibilidad. Rechaza valores futuros, etiquetas anteriores a su madurez, duplicados e identidades desconocidas. Valida cada entrada antes de acceder a su valor numérico. Si falla una entrada posterior del lote, el snapshot de partida permanece intacto.

Las etiquetas parciales quedan pendientes hasta completar la cohorte más antigua de cada mercado. Las posteriores pueden recibir feedback, pero esperan su turno para actualizar la corrección. La actualización usa los cuantiles y la corrección realmente guardados al emitir, como ya hacía PT2. El reloj global y los relojes por mercado no retroceden.

## Exclusiones y recuperación

Una exclusión o finalización necesita un estado identificado y una huella de evidencia. El acumulador verifica su formato, no su autenticidad. El adaptador de sesión deberá verificar la prueba externa antes de aceptarla.

Si una fila queda excluida, se cancela la cohorte completa al alcanzar su madurez. No se inventan etiquetas para las demás filas ni se actualizan correcciones, errores o número de cohortes maduras con esa cancelación. El registro distingue etiquetas válidas recibidas, exclusiones y filas aún sin resultado. Las métricas del ejecutor deben conservar las pérdidas válidas de las otras predicciones, aunque esa cohorte ya no sirva para actualizar PT2. Los resultados posteriores de filas canceladas no se vuelven a enviar al acumulador de calibración.

`OnlineConformal.cancel` permite retirar únicamente la cohorte pendiente más antigua, tras su madurez y respetando el reloj. Las llamadas anteriores a emisión, maduración, exportación y restauración conservan su comportamiento y formato. La cobertura marginal de las cohortes completas no acredita cobertura de las canceladas ni de los intervalos que seleccione una política.

El estado del acumulador tiene versión y política propias, huella SHA-256 y una representación JSON con copias de cuantiles y metadatos. La restauración comprueba correspondencia de cohortes, identidades, orden, fechas y resultados parciales. La huella detecta cambios accidentales, no autentica un origen hostil.

Los límites por defecto son 32.768 filas pendientes y 8 MiB de JSON canónico. El máximo configurable de JSON es 16 MiB. Cada mercado conserva el límite de 64 cohortes de PT2 y el formato admite hasta ocho mercados. Los identificadores tienen como máximo 128 bytes. El presupuesto de JSON no representa un límite del pico transitorio de RAM. No se conserva un historial ilimitado de cohortes resueltas.

## Comprobación y coste

Las [pruebas y medidas](../../reports/engineering/cohort-feedback-20261021/README.md) usan fixtures identificados, sin corpus real ni pasos de optimizador. Cubren recepción dividida, exclusiones, permutaciones, recuperación, aislamiento y rechazos. La mutación dirigida y CRAP se registran con su alcance.

El acumulador y la serialización trabajan en CPU con NumPy. Se midieron propuestas completas con 64 y 256 filas y una o cuatro cohortes pendientes. La copia explícita de contenedores redujo el coste observado frente a las copias recursivas, sin cambiar los valores ni los estados exportados. No se ha medido aquí el predictor, el recorrido CUDA ni la escritura atómica en disco.

Para repetir la selección técnica en el entorno del proyecto:

```bash
PYTHONPATH=src:. uv run --no-sync pytest -q tests/calibration/test_cohort_feedback.py tests/calibration/test_online_conformal_cancellation.py tests/calibration/test_online_conformal.py
```

Para repetir la medida se necesita una ruta de salida cuyo directorio exista y cuyo archivo no exista todavía:

```bash
PYTHONPATH=src:. uv run --no-sync python reports/engineering/cohort-feedback-20261021/benchmark.py --output /tmp/mars-titan-cohort-feedback-cost.json
```

El siguiente tramo debe conectar estos snapshots con la preparación, resolución, cierre de fase, verificación, recuperación y poda de `FinancialSession`. También faltan el catálogo cerrado de acciones y sus puntos de decisión, las trazas de intento, propuesta y confirmación, el ajuste con cortes disjuntos y los controles de utilidad. La campaña completa y el test final permanecen cerrados. No hay una mejora predictiva medida por este componente.
