# Comparación por sesiones de los controles E2

`analyze_spaced_runs` compara las políticas de retención de una ventana y semilla con unos mismos parámetros seleccionados. Admite el lector estándar y el regional, con configuraciones distintas. No reentrena cada política ni sustituye ese contraste de aprendizaje. La adaptación asociativa de Titans conserva su política, mientras que los parámetros compartidos permanecen congelados durante la ventana.

Las configuraciones [estándar](../../configs/experiments/spaced-utility-comparison-v1.json) y [regional](../../configs/experiments/spaced-regional-utility-comparison-v1.json) añaden el esquema 3 de `SpacedStudy`. Conservan los seis controles, la capacidad, la semilla y el presupuesto de sus perfiles anteriores. Estos siguen intactos y no adquieren campos nuevos en su identidad declarativa. El esquema nuevo incluye un `reader_contract` explícito y nueve opciones de análisis. Permanece desactivado por defecto y sin acceso al test final.

El controlador existente fija la configuración y su huella en `request.study` antes de la primera fase. La carga y la recuperación comprueban esa identidad. Cambiar el margen, la semilla o cualquier otra opción impide adoptar la salida anterior. El comparador exige esa declaración previa y rechaza una configuración añadida retrospectivamente. Los hashes permiten comprobar los documentos, pero no autentican un entrenamiento ni una fecha independiente.

## Datos y comprobaciones

Cada entrada es el `run.json` completado de uno de los seis controles. Deben compartir vista, manifiesto original, protocolo, código, enlace nativo, dispositivo, fases, índices y selección de parámetros. Se comprueban las identidades del padre, del lector y del consumidor. La política de retención y la presencia de evidencia pueden variar conforme al control declarado. El estado inicial debe reiniciarse al comenzar cada fase.

Las predicciones conservan su número de filas, tamaño y huella. El análisis necesita los Parquet originales materializados y no regenera archivos liberados con otro código. Verifica también las métricas por sesión y la evidencia mediante el comprobador del controlador. Recalcula las puntuaciones desde las filas y coteja las métricas almacenadas, con tolerancias relativas de `1e-12` y absolutas de `1e-14`. Las claves, los conteos y las fronteras temporales se comparan exactamente.

La [lectura por sesiones](../engineering/prediction-file-scoring.md) reúne los activos del mismo mercado e instante aunque crucen lotes. La huella común incluye sus identificadores y objetivos. Igualar solo el número de filas no basta. Las observaciones deben pertenecer a la fase declarada y ser anteriores a 2024. La disponibilidad histórica de las entradas originales requiere su auditoría separada.

Se conservan los cinco cuantiles cuando la cabeza original los emite. Una salida parcial o distinta entre controles se rechaza. El predictor residual cero se calcula por su definición y no recibe una cabeza de incertidumbre. Las métricas de dirección mantienen la convención existente sobre objetivos nulos, abstenciones y fallos.

El código principal está en [spaced_comparison.py](../../src/mars_titan/evaluation/spaced_comparison.py). La comprobación de fuentes está en [spaced_sources.py](../../src/mars_titan/evaluation/spaced_sources.py) y la declaración en [spaced_study.py](../../src/mars_titan/memory/spaced_study.py). No se cargan pesos, optimizadores ni memorias para realizar el análisis. Las fuentes se comprueban de nuevo antes de publicar `sessions.parquet` y `comparison.json` en un destino nuevo.

## Contrastes y alcance estadístico

La métrica principal es MAE residual por sesión. La configuración propuesta da el mismo peso a cada sesión y declara el margen de desarrollo `0.00001`. Ese valor no es un umbral financiero validado. La alternativa de ponderación por mercado se declara antes de ejecutar y cambia la identidad del estudio.

El contraste principal es `utility_minus_baseline`. La diferencia se calcula como MAE de utilidad menos MAE de referencia. Un valor negativo indica menor error. El porcentaje de mejora utiliza el signo contrario y queda sin definir cuando el MAE de referencia es cero.

La propuesta declara confianza del 95 %, 2.000 réplicas, bloques circulares de 16 días observados, sensibilidad a 8 y 32 y semilla `20261027`. Todas las sesiones y activos de un mismo día UTC se mantienen juntos en cada réplica. No se remuestrean filas independientes. Estos valores no garantizan precisión estadística y no se elige la longitud de bloque por el resultado más favorable. Si faltan días suficientes, se informa el motivo y no se fabrica un intervalo.

El contraste principal utiliza su intervalo marginal. Los secundarios forman una familia por métrica, con los intervalos simultáneos del comparador existente. En las métricas puntuales hay diez contrastes: cuatro controles frente a la referencia, utilidad frente a esos cuatro y referencia y utilidad frente al predictor residual cero. Las métricas cuantílicas tienen ocho, sin atribuir cuantiles al cero. Las familias de distintas métricas no reciben una corrección conjunta.

Se informan MAE, MSE, RMSE, dirección y Rank IC. Si existen cuantiles, se conservan sus puntuaciones y se añaden contrastes de pinball y Brier del signo. El mínimo declarado para Rank IC es de tres activos. Los casos indefinidos conservan su motivo y no se sustituyen por valores favorables.

El resultado principal distingue intervalo por debajo del margen negativo, por encima del margen positivo, dentro del margen o inconcluyente. Esa descripción se refiere a la ventana analizada. Un intervalo que incluye cero no demuestra equivalencia, y un menor error en esta ventana no acredita generalización entre semillas ni superioridad sobre otros modelos.

## Ejecución protegida

La [orden](../../scripts/analyze_spaced_windows.py) recibe `--study`, `--partition`, `--output` y un `--run variante=informe` por control. Solo admite validación, calibración o evaluación. Rechaza variantes duplicadas, desconocidas o ausentes. La protección de aprendizaje se comprueba antes de leer los informes o crear salidas y continúa vigente para evaluaciones científicas.

```bash
PYTHONPATH=src:. uv run --no-sync python scripts/analyze_spaced_windows.py --help
```

La salida conserva las puntuaciones por sesión, los contrastes, sus intervalos, las opciones declaradas y las huellas de fuentes y código. El campo interno `seconds` empieza después de la comprobación inicial de fuentes. La medida del recorrido completo de la API se realiza por separado e incluye esa comprobación. El análisis no selecciona ni promociona automáticamente una variante.

## Comprobaciones técnicas

La [verificación del 27 de octubre de 2026](../../reports/engineering/spaced-comparison-20261027/verification.json) reúne 250 casos CPU distintos, 95 nuevos y 155 regresiones o guardas. La batería conjunta aprobó 248 y se añadieron dos casos adversariales sobre el mismo producto. Los tres casos de argumentos inválidos de la CLI se repitieron tras reforzar su frontera y no se suman. No hay omisiones en las ejecuciones válidas de esta selección y no se ha ejecutado la suite completa.

Las pruebas incluyen las salidas generadas por el runner sobre fuentes manuales, para ambos contratos y los seis controles. Comprueban la declaración anterior a las emisiones, el rechazo al cambiar el margen, la recuperación de fases y la paridad con los perfiles anteriores. También cubren poblaciones distintas con el mismo conteo, fechas fuera de fase en todos los controles, cambios de fuente, promedio por mercado, signos de la diferencia, denominador cero, margen, cuantiles y aislamiento de RNG.

Las [quince mutaciones dirigidas](../../reports/engineering/spaced-comparison-20261027/mutations.json) se detectan con controles verdes antes y después, sin omisiones ni intentos de pasos de optimizador en las fases finales. Se conservan dos intentos previos de validación del arnés. El primero esperaba una excepción distinta y el segundo dejó que el bloqueo convirtiera un fallo de argumentos en una omisión. Se corrigió la expectativa y se reforzó la prueba de frontera, sin modificar el producto ni retirar su protección. El total final no suma repeticiones y no acredita detectar cualquier mutante posible.

La [cobertura y el CRAP](../../reports/engineering/spaced-comparison-20261027/quality.json) utilizan coverage.py 7.16.1 y Radon 6.0.1. Los cuatro módulos recorridos cubren 284/286 sentencias y 64/66 ramas, con CRAP máximo 44. La convención de CRAP usa cobertura de sentencias. Las condiciones booleanas compuestas no se instrumentan como todas sus combinaciones.

Estas comprobaciones utilizan fixtures y no ejecutan entrenamientos ni evaluaciones financieras reales. La comparación científica y las comprobaciones CUDA de los modelos siguen pendientes.
