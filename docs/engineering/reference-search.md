# Búsqueda y continuaciones de las referencias

La búsqueda aplica doce configuraciones a cada una de las familias RNN, LSTM,
GRU y DLinear. No es un recorrido de todas las combinaciones posibles. Cada
nivel de anchura, dropout, pérdida y tasa de aprendizaje aparece cuatro veces.
Cada profundidad aparece seis veces. Las familias reciben el mismo diseño,
definido antes de consultar sus resultados.

| Parámetro | Valores |
| --- | --- |
| Anchura | 32, 64, 128 |
| Profundidad | 1, 2 |
| Dropout de fusión | 0, 0,1, 0,2 |
| Pérdida | MAE, MSE, Huber |
| Tasa de aprendizaje | 0,0001, 0,0003, 0,001 |

El equilibrio es marginal, no una afirmación de ortogonalidad entre todas las
interacciones. Huber conserva un umbral de 0,01. Cada ejecución admite hasta
30 épocas completas y paciencia de cinco épocas sin mejora del MAE por sesión.
La reserva final permanece cerrada.

## Búsqueda, finalistas y controles

La búsqueda utiliza la semilla 42. Se elige una configuración por familia y
brazo mediante el MAE de sesión de las predicciones de validación del estado
seleccionado. Un empate se resuelve por identificador, de forma determinista.
La selección sigue siendo un resultado de desarrollo, no una prueba independiente
de superioridad.

El caso ganador con semilla 42 se reutiliza. Las semillas 43 y 44 se ajustan con
esa configuración y la misma población. Desde cada finalista se ejecutan dos
continuaciones de cinco épocas, con MAE y MSE, tasa 0,0001 y un optimizador
AdamW nuevo. Mantienen arquitectura, datos, semilla y tamaño de lote. Las
continuaciones no cambian retrospectivamente el ganador de la búsqueda.

Un brazo con cuatro familias contiene 48 casos de búsqueda, ocho finalistas
nuevos y 24 continuaciones. Son 80 entrenamientos distintos, además de las
referencias tabulares que se ejecutan por separado. La configuración multimercado
declara US, CN y la unión, esta última con ponderación natural y equilibrada
durante el ajuste. Requiere 320 ejecuciones neurales y datos admisibles en ambos
mercados. No sustituye un brazo chino ausente por datos estadounidenses.

## Ejecución y recuperación

[La configuración estadounidense](../../configs/baselines/scientific-search-us.json)
y [la multimercado](../../configs/baselines/scientific-search-multimarket.json)
declaran una ventana de 64 sesiones. El manifiesto debe acreditar ese contexto y
una cobertura completa del universo seleccionado. Los presupuestos de memoria
y el lote de 512 necesitan margen real de GPU. No se interpreta ese lote como
el óptimo de rendimiento sin medirlo sobre la población preparada.

`training.reference_search` recibe `--config`, `--manifest` y `--output`.
`--resume` continúa un estudio existente tras contrastar configuración, código,
población y artefactos. Los casos terminados se verifican sin volver a ajustarlos.
La selección y los orígenes de las continuaciones aparecen en el progreso antes
de iniciar esas continuaciones.

El resumen conserva el estado de cada intento, la configuración resuelta,
huellas y la puntuación de selección. Un fallo de ejecución o de integridad
detiene el estudio y queda identificado. Una salida corrupta no cuenta como
un caso completado. Los fallos no se convierten en resultados numéricos ni se
ocultan descartando ese modelo.

Las pruebas de este controlador usan datos sintéticos y comprobaciones de
recuperación. No acreditan que haya terminado la campaña científica del corpus.
La preparación actual todavía necesita resolver las publicaciones contables
chinas antes de ejecutar los brazos CN y mixto con cuatro modalidades.

La [evidencia de verificación](../../reports/resources/reference-search-quality.json)
registra el recorrido funcional, cobertura y mutaciones dirigidas, con esos
límites de interpretación.
