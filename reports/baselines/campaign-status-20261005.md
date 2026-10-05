# Estado de las campañas el 5 de octubre de 2026

Las referencias y continuaciones neuronales terminaron sus 160 casos. La segunda etapa terminó 1.352 casos: 68 referencias tabulares, 528 ajustes y 756 evaluaciones. Se conservaron las cuatro ventanas temporales y el test final cerrado. Estos recuentos describen ejecución, no una conclusión sobre calidad predictiva.

La [instantánea comprobada](campaign-status-20261005.json) registra la revisión científica `39ef706`, las huellas de los recibos terminados y la reanudación del RL financiero sintético. En ese corte había 47 de 48 casos confirmados y la fase auxiliar estaba en marcha. El plan crece al preparar auditoría y las variantes auxiliares que cumplan su condición. El denominador describe las fases ya preparadas y puede ampliarse.

La supervisión se corrigió en [#218](https://github.com/GonxKZ/mars-titan/pull/218) y la lectura de recibos en [#222](https://github.com/GonxKZ/mars-titan/pull/222). La primera evita confundir procesos CUDA descendientes con cargas ajenas. La segunda distingue el registro financiero del predictivo, valida sus transiciones y conserva la diferencia entre pausa, fallo y finalización. Los módulos operativos se cargan por separado y mantienen el ejecutable, configuración, importaciones y checkpoints científicos originales.

El caso interrumpido antes de arrancar el siguiente proceso nativo conservaba contadores cero y carecía de ledger y salida. Se comprobó el orden de publicación duradera del ledger antes del lanzamiento. La recuperación ordinaria retomó el trabajo sin reconstruir contabilidad ni modificar el diario. El avance posterior atravesó el final del piloto y la entrada en los ajustes principales, sin reinicios externos del servicio.

El análisis agregado de las evaluaciones ya tiene artefactos locales. Su revisión estadística y publicación son tareas distintas. Las métricas reservadas de calibración y auditoría no se incluyen en esta instantánea ni se usan para diseñar cambios del candidato.

La revisión histórica con macro incompleta conserva sus 216 ajustes y su informe separado. La investigación del candidato MARS-TITAN sigue en diseño. Los controles técnicos de [preparación experimental](../../docs/engineering/experimental-controls.md) no se presentan como entrenamiento de ese modelo ni como resultados financieros históricos.
