# Verificación del contraste multisemilla de cuantiles

El [protocolo v2](../../../docs/experiments/quantile-routing-v2.md) conserva los cinco modelos de la ronda anterior y fija cinco semillas antes de ejecutar. El analizador promedia errores por sesión y mantiene separada la dispersión entre semillas. No convierte semillas en fechas ni mide un ensemble de predicciones. La ejecución científica de esta ronda todavía no está completada.

La [verificación focalizada](verification.json) registra 104 casos aprobados, sin omisiones en esta selección. Incluye pruebas calculables de medias y desviación típica, semillas ausentes o repetidas, población y sesiones incompatibles, máscaras distintas, presupuesto de actualizaciones, bloqueo de aprendizaje, control unitario dentro de cada semilla, lectura de archivos y hashes, y la orden de análisis. Las nuevas preparaciones se comprobaron en CUDA con la semilla 43, sin pasos de optimizador. Las pruebas anteriores de una semilla también pasan. No es toda la batería del repositorio.

Los [once mutantes dirigidos](mutations.json) fueron detectados y la selección volvió a pasar al restaurar las fuentes. Comprueban, entre otros fallos, usar solo la última semilla, aceptar solo el primer control unitario, omitir la igualdad de actualizaciones y perder la identidad de sesiones o las semillas del protocolo. El resultado cubre esos once mutantes, no todos los fallos posibles.

La [medida de cobertura y CRAP](quality.json) usa coverage.py 7.16.1 con ramas y Radon 6.0.1. La convención es `C² × (1 − cobertura de sentencias)³ + C`. El nuevo analizador cubre 100 de 102 sentencias y 32 de 34 ramas. Las dos sentencias no ejecutadas pertenecen a la entrada directa del módulo. Con el ejecutor reutilizado, el alcance total es de 240 de 273 sentencias y 68 de 86 ramas. Su CLI conserva zonas sin cobertura en esta selección y el CRAP máximo es 72. La medida no sustituye las pruebas de comportamiento ni acredita ausencia de errores.

Las primeras pruebas se ejecutaron sin la variable de CUBLAS requerida y fallaron antes de ajustar. También se corrigió la huella del protocolo del fixture, que debe ser la del JSON canónico y no la de sus bytes. La ejecución final conserva las guardias y pasa con el entorno declarado. No se relajaron los contratos para obtener un resultado favorable.

El analizador histórico de una semilla permanece separado. Los módulos del predictor y sus gradientes no cambian en esta ronda. El nuevo experimento no abre la campaña completa ni el test final de 2024.
