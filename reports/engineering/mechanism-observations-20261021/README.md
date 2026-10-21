# Observaciones del prefijo para el selector de mecanismos

El extractor calcula y sella los 21 rasgos declarados por el [núcleo del selector](../mechanism-policy-20261021/README.md). Reutiliza las filas CPU admitidas por la sesión, sin releer el corpus ni ejecutar codificadores. Se comprueba con fixtures. Todavía no activa la selección automática en `FinancialSession`, no ajusta coeficientes ni aporta resultados predictivos nuevos. Continúa [#701](https://github.com/GonxKZ/mars-titan/issues/701).

## Datos y disponibilidad

[`extract_observation`](../../../src/mars_titan/memory/mechanism_observations.py) recibe las filas, su cohorte declarada, consumidor, estado anterior, corte y contexto identificado. Comprueba inmutabilidad, formas, FP32, finitud, rellenos, máscaras y huellas de las cinco fuentes. Rechaza duplicados, filas ausentes, identidades ajenas y disponibilidad posterior al corte. La edición histórica mantiene la frontera de 2000 y el test final de 2024 sigue cerrado. El límite lógico es 64 MiB de inputs, aunque las filas compartan memoria física. También se aplica el límite de flujos configurado en el predictor.

Para cada mercado se conservan las fracciones de presencia y la media de `log1p(RMS)` de cada fuente. El denominador es el total de filas de esa cohorte. Una fuente ausente aporta cero y conserva su fracción de presencia. No se retira la fila. Los precios y gráficos son obligatorios en el contrato histórico actual, por lo que sus fracciones son uno en las entradas admitidas. En fundamentales y macro, el RMS resume el vector codificado con valores, máscaras y edades. No es una medida física de un indicador ni una estimación de utilidad financiera.

El RMS se calcula en FP64 a partir de entradas FP32 acotadas por el esquema. Sus cuadrados y su suma caben en FP64, incluido el mayor valor finito de FP32. En las ventanas de precios se usan solo los cinco canales numéricos. Se excluye el sexto canal de presencia y se conservan los huecos con relleno cero dentro de la ventana completa.

La edad de inputs es la máxima diferencia entre el corte y la disponibilidad agregada de las filas, expresada en segundos y transformada con `log1p`. No indica la edad individual de cada noticia o concepto contable. El contexto anterior usa días y su fecha confirmada. El banco aporta solo el recuento del snapshot identificado. La calibración aporta ocupación global de su cola y media y máximo absoluto de las correcciones estáticas y vigentes del mercado. No se consultan valores episódicos, proyecciones, etiquetas pendientes ni feedback del evento actual. Se mantiene `predict_all_then_feedback_v1`.

Estas comprobaciones respetan la disponibilidad declarada. No convierten en perfectos los vintages o los sellos aproximados de las fuentes anteriores. El extractor no modifica el estado de contexto, calibración o banco ni retiene los arrays de entrada al devolver.

## Sellado y conversión

[`MechanismObservation`](../../../src/mars_titan/models/titans/mechanism_observation.py) guarda los valores FP64, esquema y orden, agregación, ausencia de normalización previa, fechas, mercado, política de inputs, versiones e identidades. Las huellas de filas y flujos vinculan el vector con las observaciones de la cohorte. La huella del productor identifica la implementación. El JSON está acotado a 64 KiB y se conserva como texto inmutable. `record()` devuelve una copia y `restore()` comprueba su huella y coherencia física y temporal.

La conversión a tensor exige dispositivo y precisión explícitos. Se comprueba antes de convertir que cada valor tenga representación finita en FP32 o FP64. No se publica infinito por un cast que desborde. La conversión FP32 puede redondear los valores y no sustituye el registro FP64 original. El futuro enlace debe conservar también el vector efectivamente consumido por la política y conciliar sus identidades con la sesión. Un SHA detecta cambios, pero no autentica un host que pueda modificar arbitrariamente código y archivos.

## Comprobaciones

La [verificación](verification.json) conserva por separado la referencia y la versión seleccionada. Incluye cálculos manuales, ausencias, mercados separados, permutación y partición, disponibilidad futura, fronteras temporales, propiedad de memoria, límites de bytes, corrupción, desbordamientos y recuperación del siguiente vector desde la sesión nativa en CPU y CUDA. El paso usa `restore_state` para recuperar el objeto auxiliar desde su representación exportada.

Las reglas históricas se comprueban también con filas cuyo hash es válido pero cuyo relleno, máscara por concepto o edad son incorrectos. Se rechazan antes de devolver una observación. Los fixtures contienen embeddings episódicos que fallan si se intenta consultarlos. Una prueba de referencias débiles comprueba que el resultado no conserva los arrays de entrada.

La [mutación de la versión seleccionada](mutations.json) incluye la retirada de la validación histórica agrupada, además de errores de mercado, denominador, presencia, precio, precisión, fecha, huellas, propiedad de memoria, presupuesto, calibración y supervisión. La [cobertura y CRAP](quality.json) indican los recorridos y rechazos ejecutados. Son evidencia técnica, no resultados financieros ni una demostración de ausencia de todos los errores.

La selección final pasó 277 casos, 24 marcados CUDA, sin fallos ni omisiones. Incluye 67 casos del extractor, cuatro CUDA, y los controles anteriores del consumidor, adaptador, catálogo y políticas, junto con los contratos de los benchmarks. No es toda la suite del proyecto. Hay una advertencia conocida del modo de diagnóstico de sincronización de PyTorch. Las veinte mutaciones dirigidas de la versión seleccionada se detectaron. Sus referencias inicial y restaurada pasaron 63 casos CPU cada una. La campaña anterior de diecinueve mutantes corresponde a la referencia y no se suma como otra prueba independiente de la candidata.

La cobertura de los dos módulos nuevos registra 223/228 sentencias y 105/110 ramas en once funciones, con coverage.py 7.16.1 y Radon 6.0.1. CRAP utiliza cobertura de sentencias. Su máximo es 48 en `extract_observation`, que reúne varias comprobaciones y la agregación. Quedan fronteras sin recorrer. Una puntuación o la detección de los fallos elegidos no sustituyen revisar las hipótesis ni prueban la admisibilidad de las fuentes originales.

## Coste y cambio conservado

El [primer protocolo](cost-protocol.json) fijó ocho condiciones con 64 y 256 filas, uno o dos mercados, FP32/FP64, dos calentamientos y doce repeticiones. Se midieron extracción y sellado, y el recorrido completo que añade el tensor CUDA. Los recorridos son inclusivos y no se suman. El consumidor está en `cuda:0` y sus verificaciones forman parte de ambos tiempos. No se ejecuta la predicción del modelo. Las modalidades son pequeñas, el banco está vacío y la calibración no tiene pendientes. No se midió energía ni se controlaron temperatura, frecuencia o carga gráfica ajena.

Se usó la RTX 4070 Laptop con PyTorch 2.14.0+cu130, NumPy 2.5.3 y dos hilos CPU. Las ventanas tienen 64 × 5 valores y las otras fuentes dimensiones 2, 2, 3 y 3, con H = 32. Se comprobaron ventanas de seis canales en las pruebas, pero no se midió aquí su coste. La preparación y construcción quedan fuera de las latencias internas y dentro del tiempo de proceso. No hubo pruebas, mutaciones o perfiles propios concurrentes con las medidas de coste.

El perfil inicial registró 5.120 llamadas a `validate_historical_vectors` en veinte extracciones de 256 filas. Acumularon 0,728 de 1,604 s instrumentados. Se repetían las mismas comprobaciones NumPy y la lectura de una representación inmutable. Esos tiempos de perfil no son las latencias del benchmark.

El [protocolo de optimización](optimization-protocol.json) se fijó antes de probar bloques de 16 y 64 filas. La candidata conserva las comprobaciones por fila de identidad, fecha, propiedad, forma, finitud y huella. Agrupa únicamente la validación histórica de las modalidades distintas de precios. Sus condiciones son comparaciones por elemento o fila unidas por `all` y `any`, de modo que verificar bloques equivale a verificar cada fila. No cambia el cálculo del RMS ni el orden de reducción de las medias. No se retiran reglas ni se añade una caché entre eventos.

Las dos candidatas pasaron sus pruebas y reprodujeron exactamente los registros de 66 condiciones, incluyendo tamaños a ambos lados del borde de bloque. Solo cambia la huella del productor y, por ello, la exterior que la contiene. La [comparación](optimization-comparison.json) conserva ambas referencias, medidas antes y después de las candidatas. La regla seleccionó 16 filas. Su media de ratios de p50 frente a la media de las dos referencias es 0,509, una reducción observada cercana al 49 %. Los p95 de las ocho condiciones también quedan por debajo de sus referencias. No se atribuye ese porcentaje al predictor ni a una sesión financiera completa.

| Filas | Mercados | Precisión | Referencia, media de p50 (ms) | Bloques de 16, p50 (ms) |
| ---: | --- | --- | ---: | ---: |
| 64 | US | FP32 | 12,952 | 6,745 |
| 64 | CN + US | FP32 | 12,819 | 7,892 |
| 256 | US | FP32 | 36,956 | 13,217 |
| 256 | CN + US | FP32 | 36,197 | 14,367 |
| 64 | US | FP64 | 13,319 | 7,381 |
| 64 | CN + US | FP64 | 13,919 | 7,273 |
| 256 | US | FP64 | 32,506 | 18,402 |
| 256 | CN + US | FP64 | 32,112 | 17,409 |

Hubo deriva entre referencias. El ratio de p50 posterior respecto a inicial va de 0,765 a 1,032. Se conserva esa variación y no se presenta una precisión inferencial del porcentaje de aceleración. Los percentiles tienen la resolución limitada de doce repeticiones. El bloque de 64 obtuvo una media de ratios de 0,548 y no se eligió.

El buffer nuevo copia como máximo 16 filas de las modalidades distintas de precios. Con las anchuras máximas del esquema son aproximadamente 0,5 MiB de arrays FP32 y máscara, además de temporales del validador. Es un intercambio explícito de memoria transitoria por menos llamadas. No es una reducción medida de toda la RAM. Los recibos registran RSS de proceso y VRAM del asignador de PyTorch, que tampoco representan toda la memoria del dispositivo.

El [coste de proceso](process-cost.json) pasa de 9,80 s y 9,13 s en las referencias a 6,18 s con bloques de 16. El pico de RSS de la candidata es 1.050.578.944 bytes, cercano al de las referencias. Incluye imports, construcción y todas las condiciones. No se observa un ahorro de RAM atribuible al extractor. En un segundo [perfil instrumentado](profiles-summary.json), las veinte llamadas de 256 filas realizan 320 validaciones históricas en lugar de 5.120. Las verificaciones del consumidor y las comprobaciones y cálculos por fila siguen ocupando tiempo. No se retiraron para obtener la reducción y no se afirma haber alcanzado un límite del equipo.

Los [artefactos externos](external-artifacts.json) identifican logs, XML, perfiles, fuentes de referencia y paridades. Se conservan los fallos de preparación: uso incorrecto del retorno de un validador, fixtures de calibración sin resellar, colecciones de batches anidadas, estado exportado confundido con su objeto, dimensiones no admitidas y un resumen de cobertura solicitado antes de terminar su proceso. Se corrigieron las llamadas y se esperó el mismo proceso. El primer smoke del benchmark omitía declarar `mac_online` y falló antes de medir. La ejecución válida y los protocolos fijados son posteriores. Esos fallos no se cuentan como detecciones de mutantes.

Para reproducir las pruebas en el entorno del proyecto:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 MARS_TITAN_REQUIRE_CUDA=1 uv run --no-sync pytest -q tests/memory/test_mechanism_observations.py tests/memory/test_financial_mechanisms.py tests/models/titans/test_financial_adapter.py tests/models/titans/test_mechanism_catalog.py tests/models/titans/test_mechanism_policy.py tests/models/titans/test_mechanism_losses.py tests/models/titans/test_refinement_policy.py tests/tooling/test_regional_benchmarks.py
```

Para repetir el coste se necesita una ruta de salida nueva cuyo directorio exista:

```bash
PYTHONPATH=src:. CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run --no-sync python reports/engineering/mechanism-observations-20261021/benchmark.py --rows 64 256 --warmup 2 --repetitions 12 --output /tmp/mars-titan-observation-cost.json
```

Faltan la selección automática, las trazas de la elección y su publicación recuperable, la colección de alternativas y el ajuste protegido. También falta medir el recorrido integrado con estados poblados y datos utilizables. La campaña completa y el test final permanecen cerrados. No hay una mejora predictiva medida por este componente.
