# Dust: verificación técnica y agrupación de probes

El paquete opcional de Dust para `transformer_direct` escalar supera 70 casos técnicos y cinco mutaciones dirigidas. En el perfil sintético, chunk 4 incumple la tolerancia de gradientes y chunk 8 supera una comprobación posterior con la misma referencia. Este informe no contiene ajuste sobre datos financieros ni una medida de MAE financiero.

La implementación revisada corresponde al commit `cd44119df34886e093bb34f99edc1556ad8f122a`. El [paquete](https://github.com/GonxKZ/mars-titan/tree/cd44119df34886e093bb34f99edc1556ad8f122a/src/mars_titan/training/dust) adapta el [código mínimo Dust](https://github.com/qlabs-eng/dust/tree/b20f7c03eac630b6441ba6f254128c1761dc45ad), con licencia MIT. El estimador cubre proyecciones, sesgos, LayerNorm y QKV del predictor completo. Devuelve gradientes y contadores sin ejecutar `backward`, asignar `.grad` o aplicar un optimizador. Es una variante no canónica y no cambia el ajuste predeterminado.

## Verificación

Pasaron 46 casos del adaptador y estimador, con siete CUDA, y 24 casos de entradas, con tres CUDA. Son 70 casos únicos, diez CUDA, sin fallos ni omisiones. Las repeticiones usadas en mutación no se añaden a ese total.

Coverage 7.16.1 registra 428 de 451 sentencias (94,90 %) y 113 de 136 ramas (83,09 %) en los cuatro archivos del paquete, sin nuevas exclusiones. Con Radon 6.0.1, el CRAP máximo es 23,4365 en `DustEstimator.load_state_dict`, con complejidad 19. Se usa `CRAP = v²(1−c)³+v`, donde `c` es cobertura de sentencias en el rango de la función. Las ramas se informan aparte. Estos valores son diagnósticos del alcance ejercitado.

| Mutación | Comportamiento alterado | Resultado |
| --- | --- | --- |
| D1 | Signo del gradiente de cabeza | Detectada |
| D2 | División doble por población | Detectada |
| D3 | Admisión de Dropout no nulo | Detectada |
| D4 | Aceptar un booleano como versión entera del checkpoint | Detectada |
| D5 | Omitir la restauración del RNG privado tras un probe fallido | Detectada |

Cada detección tiene una fase original G1 aprobada, un fallo semántico CALL en M1 y una restauración G2 aprobada, sin errores de colección ni omisiones. D1 y D2 se ejecutaron en el primer lote y D3–D5 en la continuación v3. El primer intento de D3 se conserva como incompleto porque el formato esperado de `DID NOT RAISE` no coincidía con pytest. No se cuenta como otra detección. El alcance son estas cinco mutaciones, no todas las alteraciones posibles.

## Perfil sintético

El modelo tiene dimensión 32, una capa, cuatro cabezas, multiplicador de feedforward 2 y contexto de 64 sesiones. Recibe 32 filas con anchuras 6, 384, 512, 78 y 420 para precios, noticias, gráficos, fundamentales y macro. Los inputs proceden de una fixture determinista con máscaras. El objetivo es `linspace(-0.02,0.02,32)` en FP32 y no representa un resultado financiero.

Se fijan población 16, sigma 0,01, semilla privada 73, inicialización del modelo 42 y pérdida MAE. El generador privado vuelve al mismo estado antes de cada estimación. La tolerancia es `rtol=2e-4, atol=2e-6`. Se usa una RTX 4070 Laptop GPU, 8.188 MiB reportados, controlador 595.91.07, PyTorch 2.14.0+cu130 y CUDA 13.0. La política es `fp32_strict`, con TF32 desactivado, dos hilos CPU, uno interop y límite del asignador Torch de 1 GiB.

| Chunk | Filas físicas máximas | Resultado | Llamadas completas / atención local / cabeza por estimación |
| --- | ---: | --- | ---: |
| 2 | 64 | Calentamiento y tres medidas con paridad | 106 / 25 / 8 |
| 4 | 128 | Falla en el primer calentamiento | 54 / 13 / 4 |
| 8 | 256 | Comprobación posterior con paridad | 28 / 7 / 2 |
| 16 | 512 | No ejecutado, supera `max_batch=256` | No medido |

Cada estimación completada acumula 6.720 exposiciones a filas en recorridos completos del modelo, a partir de las mismas 32 filas lógicas. La agrupación reduce el número de llamadas completas y conserva ese volumen. Las atenciones locales y pérdidas de cabeza tienen contadores separados.

Chunk 4 incumple la tolerancia en dos de 32 componentes de `price_encoder.blocks.0.norm1.bias`, con diferencia absoluta máxima `2.8028152883052826e-6`. No se amplían las tolerancias ni se admite ese chunk. La primera ejecución termina en error después de cinco estimaciones y conserva sus registros.

La comprobación de chunk 8 usa una sola referencia de chunk 2, sincronizada y sin tiempos, seguida de un calentamiento y tres medidas de chunk 8. Comprueba las mismas fuentes, configuración, tensores y estado inicial del RNG privado que el perfil anterior. Al completar la comprobación, pesos, buffers, inputs, configuración y RNG global permanecen iguales.

| Chunk | Pared, mediana [mínimo, máximo] en ms | Intervalo CUDA, mediana en ms | Pico asignado / reservado en MiB |
| --- | ---: | ---: | ---: |
| 2 | 1600,015 [1390,531, 1643,335] | 1599,772 | 44,243 / 60,000 |
| 8 | 480,665 [383,458, 736,923] | 480,491 | 58,382 / 60,000 |

Las cifras proceden de dos ejecuciones bajo concurrencia técnica declarada (`technical_contended`). Cada mediana usa tres medidas con probes idénticos. El intervalo CUDA incluye huecos entre lanzamientos y no es tiempo exclusivo de kernels. Los tiempos por muestra excluyen preparación, transferencias, reinicio del RNG y auditorías. Los procesos supervisados duraron 34,354 s y 30,553 s. El JSON conserva también el tiempo interno del proceso y las mediciones RSS muestreadas y máximas de vida por separado.

Estos tres valores por configuración no acreditan una aceleración global ni su variación en cargas reales. Quedan pendientes el coste del ajuste completo y las comparaciones con optimizadores y actualizaciones emparejados sobre ventanas financieras cortas. No se midió energía, no se abre el test final y no se atribuye superioridad predictiva.

El [resumen curado](summary.json) contiene los SHA de fuentes, pruebas y recibos, los 15 resultados de fases válidas de mutación, las muestras de tiempo y los contadores completos de ambas ejecuciones.
