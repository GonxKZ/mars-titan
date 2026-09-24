# Evidencias e informes

Esta carpeta contiene plantillas, informes y resúmenes revisados. Las predicciones completas, los checkpoints y las trazas grandes permanecen fuera de Git, en `artifacts/` o en los directorios locales de ejecución bajo `data/interim/`. Las curvas resumidas pueden acompañar a sus informes.

La [edición de 470 muestras](baselines/post-scan-reference-study.md) reúne 96 ejecuciones neuronales y cuatro controles tabulares. Las ediciones de [405 muestras](baselines/streaming-reference-study.md) y [295 muestras](baselines/verified-reference-study.md) conservan otros 100 ajustes cada una. Sus tablas y figuras mantienen resultados desfavorables y distinguen los límites de estas poblaciones de la campaña global pendiente.

- [Ficha de experimento](experiment-record.md): pregunta, configuración, comprobaciones, resultados y límites.
- [Ficha de literatura](literature-note.md): lectura verificable y decisión que fundamenta.
- [Matriz de conclusiones](conclusion-matrix.md): objetivo, evidencia y alcance de la respuesta.
- [Preparación de etiquetas residuales](resources/residual-preparation.md): comparación de motores con tiempos, memoria y paridad exacta.
- [Construcción directa de lotes](resources/corpus-batches.md): recorrido CPU, propiedad de buffers, recuperación y comparación con la referencia.

Las figuras de resultados deberán generarse desde artefactos identificados, con ejes, unidades, periodo, tamaño de muestra e intervalos cuando corresponda. No se versionan tablas con cifras de ejemplo que puedan confundirse con resultados.
