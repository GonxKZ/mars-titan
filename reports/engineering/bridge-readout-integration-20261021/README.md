# Integración de las vistas compartidas en la sesión financiera

`EpisodicReadoutConfig.evidence_bridge` conecta el puente con el lector, la cabeza existente, `FrozenFinancialConsumer` y la sesión nativa. La opción está desactivada por defecto. `RegionalEpisodeRead` y `BridgedReadoutResult` conservan las trazas regionales sin reinterpretar `EpisodeRead.weights[B,k]`. La [especificación vigente](../../../docs/experiments/shared-evidence-bridge-v1.md) describe la integración y sus límites.

Las proyecciones comunes y el refinador parten de la inicialización anterior. Las proyecciones pasan al puente y se retiran del padre. No quedan parámetros duplicados para ese cálculo. Los pesos regionales son adicionales. El lector de H = 32 pasa de 7.361 a 19.649 parámetros, con el mismo predictor padre. Esa diferencia de capacidad se declara y no se presenta como una comparación de calidad con igual número de parámetros.

## Evidencia técnica

La [verificación](verification.json) conserva las tandas por separado. La regresión de compatibilidad anterior a dividir la validación de trazas pasó 373 casos. Después, la selección final pasó 298 casos, 33 marcados CUDA, sin fallos, errores ni omisiones. Incluye el puente, el lector anterior, el consumidor congelado, las rutas y la referencia GRU. El único aviso corresponde al modo experimental de depuración de sincronizaciones de Torch en una prueba anterior. Las tandas se solapan y no se suman como casos distintos.

Antes de editar se capturaron 36 casos del lector anterior, en CPU y CUDA, con K = 1, 2 y 4, selección por paso o fija, memoria vacía y control sin banco. Con la ampliación apagada se cargaron sus checkpoints y se conservaron exactamente identidad, estados y lecturas. Es paridad de esos casos del lector, no una autorización para reanudar una sesión histórica con otras fuentes. La nueva revisión de código tiene otra huella de ejecución y no redefine el baseline científico seleccionado.

Con el puente activo se comprobaron gradientes, recuperación de parámetros, selección inicial reutilizada, ausencia de parámetros comunes duplicados, RNG y formas. La misma cabeza escalar o de cuantiles consume el estado refinado. La mediana sigue siendo la emisión puntual cuando existen cuantiles. K lecturas no introducen K actualizaciones de MAC.

La sesión nativa confirmó la siguiente emisión, el estado rápido y la traza tras recuperar un corte posterior a la confirmación de feedback, en CPU y CUDA. La maduración no se aplicó dos veces. La traza contiene IDs comunes, pesos, vistas y lecturas derivadas, con un grafo compartido. No copia los valores fuente ni las etiquetas. Su presupuesto se comprueba antes de emitir y sus campos, tamaños, ausencia, selección fija y contadores se contrastan al confirmar o recuperar.

Una comprobación nativa adicional en FP32 usó los mismos cuatro flujos técnicos en CPU y CUDA. Con tolerancias fijadas en el script de `rtol=2e-5` y `atol=1e-6`, la diferencia máxima fue de 4,66 × 10⁻¹⁰. Ese caso no representa todas las formas o precisiones del sistema.

La [mutación dirigida](mutations.json) detectó los 14 cambios seleccionados. Las referencias inicial y restaurada aprobaron 56 casos CPU cada una. Se alteraron identidad apagada, inicialización emparejada, selección fija, tipo del resultado, validación del hijo antes de copiar al padre, posiciones duplicadas, sellos de configuración, rechazo de consumidores incompatibles, copia y separación del grafo, mezcla de snapshots y conteos de evidencia. No es una puntuación sobre todos los mutantes posibles.

La [cobertura y CRAP](quality.json) usan coverage.py 7.16.1, Radon 6.0.1 y el resumen común del proyecto. La selección final midió 737/810 sentencias y 232/296 ramas en cinco módulos. Se conservó la medida anterior de la validación monolítica y se separaron cabecera, tensores y secuencia de bloques. Sus CRAP finales son 33,16, 25,54 y 15,05. Quedan condiciones y rutas sin recorrer. Estos valores no acreditan ausencia de errores ni una cobertura global del proyecto.

## Coste nativo observado

El [protocolo original](cost-protocol.json) fijó 4, 64 y 256 consultas, K = 1 y 4, lector anterior y puente, dos calentamientos y cinco repeticiones. Se usaron FP64, `cuda:0`, H = 32, bloques físicos de 64 y M1 con capacidad 256. El puente usa tres regiones y dos ciclos. Los activos replican un fixture de precios, con etiquetas manuales. El generador marca presente un canal gráfico constante de valor `3.0` y ausentes noticias, fundamentales y macro. No contiene gráficos reales ni historias bursátiles independientes y no evalúa FinMultiTime.

La revisión inicial describía todas las modalidades distintas de precios como ausentes. Esa descripción se corrige tras inspeccionar `flow_source`, sin modificar protocolos, entradas, artefactos ni medidas. La [corrección de procedencia](../bridge-controller-20261021/data-description-correction.json) conserva el generador y su huella.

La primera pasada alcanzó su límite de 600 segundos con diez de doce condiciones completas y salida 124. Se preservó su JSON parcial, su fuente y el registro de terminación. Una [continuación registrada antes de ejecutarse](continuation-protocol.json) midió solo las dos condiciones restantes, de 256 consultas y K = 4. Conservó el cálculo, precisión, calentamiento y repeticiones. Añadió al script únicamente filtros explícitos de tamaños y K y metadatos de la selección.

Dentro de cada pareja se comprobó igualdad del banco elegible, codec, parámetros del padre y configuración común del lector. La continuación construye nuevos fixtures y tiene otras huellas de inputs y banco. Los parámetros del padre coinciden, pero no se mezclan las pasadas como si fueran un único conjunto idéntico para comparar K. El [resumen de coste](cost.json) mantiene ambas ejecuciones y sus identidades.

Estas son las medianas de emisión, verificación, traza y confirmación nativa, en segundos:

| Consultas | K | Lector anterior | Puente | Pasada |
| --- | ---: | ---: | ---: | --- |
| 4 | 1 | 0,402 | 0,450 | Primera, incompleta |
| 4 | 4 | 0,615 | 0,672 | Primera, incompleta |
| 64 | 1 | 1,323 | 1,158 | Primera, incompleta |
| 64 | 4 | 1,247 | 1,220 | Primera, incompleta |
| 256 | 1 | 3,892 | 5,064 | Primera, incompleta |
| 256 | 4 | 2,106 | 2,755 | Continuación |

El puente añade coste en varias condiciones. Los cambios de signo con 64 consultas y la variación entre pasadas impiden afirmar una aceleración general. Con cinco muestras, los p95 y p99 del JSON son descriptivos. No se seleccionó retrospectivamente una condición favorable ni se sustituyó el resultado incompleto por la continuación.

La maduración posterior se midió aparte. Con 256 consultas, sus medianas anterior/puente fueron 1,536/1,647 s para K = 1 y 0,752/0,775 s para K = 4 en la continuación. La suma serial combina esas dos fases consecutivas, sin sumar intervalos inclusivos anidados.

El lector anterior conserva su control actual, sin vistas regionales. El puente archiva evidencia adicional. Con 256 consultas, el control ocupa 1.773 bytes en la referencia, frente a 339.748 bytes con K = 1 y 1.339.760 con K = 4 en el puente. La diferencia medida incluye esa instrumentación y E/S. No se atribuye todo el coste adicional a la atención o a los mensajes entre regiones. Faltan un perfil del recorrido y un control de instrumentación comparable para separar esas causas.

Las latencias incluyen consulta del banco, preparación, núcleo, lectura, cabeza, validación, publicación y retirada de artefactos. La creación de inputs y la preparación anterior del banco quedan fuera de esas llamadas y dentro del proceso. No son los aproximadamente 1,7 ms del componente aislado ni latencias por activo individual. El JSON conserva el caudal y los picos asignados por Torch, que no son la VRAM total.

Se usaron una RTX 4070 Laptop de 8.188 MiB, driver 595.91.07, Torch 2.14.0+cu130, CUDA 13.0 y dos hilos CPU. TF32 y autocast estaban desactivados. Antes de las pasadas había 5.959 MiB ocupados por el sistema y otras aplicaciones y una utilización del 16 % y 11 %. No se detuvieron otras cargas. No hubo pruebas, mutaciones ni consultas propias en paralelo con los cronometrajes.

Los [registros de proceso](process-cost.json) conservan 600,47 segundos y salida 124 en la primera pasada y 144,65 segundos y salida 0 en la continuación. Esta última midió 140,41 segundos internos. Sus picos RSS externos fueron 1.546.040 y 1.652.064 KiB. No se midieron energía ni contadores de caché, ocupación o transferencias físicas.

## Reproducción y límites

Con el entorno uv CUDA y la extensión episódica nativa configurados:

```bash
evidence_dir="$(mktemp -d)"
mkdir "$evidence_dir/fixtures"
uv run --no-sync python reports/engineering/bridge-readout-integration-20261021/benchmark.py --output "$evidence_dir/cost.json" --storage-root "$evidence_dir/fixtures"
```

El script exige una salida nueva. Los filtros `--rows 256 --refinements 4` reproducen la selección de la continuación. Cada informe conserva el HEAD padre `6f0e18a6f7884d436054946cb23dd0d12782fa62` y las huellas del código efectivamente ejecutado antes de su commit. [external-artifacts.json](external-artifacts.json) identifica XML, muestras, fuentes y el inventario externo de los archivos nativos. Los artefactos pesados permanecen fuera de Git.

Las rutas, el controlador y la declaración anterior de variantes que todavía esperan pesos de otra forma rechazan esta ampliación hasta disponer de un contrato propio. La integración técnica no completa esos consumidores, la clasificación sectorial histórica, las ablaciones con igual presupuesto ni la evaluación predictiva. No se han ejecutado entrenamientos, postentrenamientos, RL, normalización o solver protegidos, ni se ha abierto el test final. No hay una mejora de predicción o del estado del arte medida.
