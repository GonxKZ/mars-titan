# Lectura contextual de bancos históricos

Se integró una variante opcional de lectura de los tres bancos neuronales de PT5, con mezcla uniforme, constante y por consulta. La escritura usa el gradiente de la función leída respecto a una traslación común de sus matrices. La [especificación](../../../docs/experiments/bank-readout-v1.md) separa sus estados, ecuaciones y límites. Las [ocho configuraciones](../../../configs/experiments/bank-readout-v1.json) reutilizan el constructor cronológico e incluyen réplicas y combinaciones con el mismo grafo regional. Permanecen desactivadas y sin entrenar.

El código comprobado es `c7143fba31667861cab347a1e13d6f6c348ce806`. La recuperación de parámetros, matrices históricas y siguiente actualización se comprueba en componentes y sesión nativa. Los bancos no son episodios ni regiones anatómicas. Su máscara controla lectura, no borrado ni parada de escritura.

## Comprobaciones

La selección final pasó 402 casos en 297,92 segundos, 75 marcados CUDA, sin fallos, errores ni omisiones. Incluye 54 casos nuevos, 18 CUDA. Se conserva un aviso de PyTorch sobre su detector experimental de sincronizaciones. No es toda la suite ni una campaña científica. Ninguna prueba aplica pasos de optimizador o estima etiquetas residuales.

Las catorce mutaciones dirigidas fueron detectadas. La referencia inicial y restaurada pasaron 36 casos CPU. Se comprobaron máscaras, ausencia total, rutas uniforme/constante, gradiente común, gradientes de direcciones, redundancia, RNG, carga parcial de hijo/padre, operador equivalente falso y contadores. No se acredita el conjunto de todas las mutaciones posibles.

coverage.py 7.16.1 midió ramas y Radon 6.0.1 calculó complejidad y CRAP mediante `scripts/summarize_python_quality.py`. En cuatro módulos se cubren 1131/1329 sentencias y 407/586 ramas. La función de probabilidades tiene 21/21 sentencias, 10/10 ramas y CRAP 14. El informe conserva las rutas no ejercitadas y los valores altos, incluido CRAP 665,75 en `FinancialConfig.__post_init__`. Es una señal para ampliar comprobaciones, no una demostración de ausencia de errores.

La [captura previa](baseline.json) comprueba igualdad bit a bit de parámetros, identidad, estados y lecturas en ocho condiciones pequeñas CPU/CUDA y FP32/64, con None y EMA. No generaliza esa igualdad a todos los tamaños. Las pruebas del Jacobiano escalar completo coinciden con una derivación independiente y conservan un contraejemplo de amplificación. No hay comprobación Lean ni garantía global de estabilidad.

## Coste de la preparación financiera

RTX 4070 Laptop de 8188 MiB, controlador 595.91.07, PyTorch 2.14.0+cu130, CUDA 13.0, Python 3.12.14 y dos hilos CPU. cuda:0 explícito, TF32/autocast desactivados. El componente se midió en FP32/64 y el predictor en FP32. El [protocolo](cost-protocol.json) fija tres calentamientos y veinte repeticiones, con orden secuencial exploratorio. No hubo otras cargas propias durante las medidas y no se detuvieron aplicaciones ajenas.

La tabla utiliza la primera ejecución completa con 256 activos lógicos, procesados en cuatro bloques de 64 de la receta. Incluye validación de entradas, transferencias CPU/GPU, codificadores internos, MAC y cabeza. No incluye codificadores externos, publicación nativa ni conversión final de la salida a Python. Las modalidades tienen dimensiones pequeñas y ausencias explícitas de los fixtures, no todo FinMultiTime.

| Variante | p50 (ms) | p95 (ms) | p99 (ms) | Pico asignado (bytes) |
|---|---:|---:|---:|---:|
| `ema_weight_mean` | 28.370 | 30.288 | 31.300 | 177116672 |
| `ema_uniform` | 43.337 | 47.233 | 49.726 | 176147968 |
| `ema_constant` | 49.651 | 58.878 | 64.996 | 176147968 |
| `ema_query` | 46.793 | 50.002 | 50.096 | 176147968 |
| `replica_uniform` | 47.543 | 49.539 | 49.856 | 176147968 |
| `replica_query` | 46.683 | 50.261 | 52.699 | 176147968 |
| `ema_uniform_graph` | 66.535 | 70.198 | 73.335 | 176324096 |
| `ema_query_graph` | 55.259 | 68.356 | 73.105 | 176324096 |

Las lecturas por funciones cuestan más en esta ejecución que la red de matrices medias. Esa referencia calcula otra función en el caso no lineal. Los lectores añaden 3D parámetros y tres booleanos compartidos, sin nuevas matrices por flujo. La memoria transitoria sí cambia. En la lectura aislada FP32 de 256 filas, la media de matrices alcanzó 117656576 bytes asignados frente a 101539840 de `query`. No es el coste de actualización ni la VRAM total.

El proceso completo de esa medida tardó 19,936 s y 23,84 s según la medida externa. El pico RSS de Python fue 1665724416 bytes y el externo 1678828 KiB. Incluyen fixtures, inicialización y profiler. No se midió energía ni ocupación del hardware. Las colas con veinte repeticiones son descriptivas. Las [capturas del profiler](profiles.json) se hicieron separadas de las latencias.

## Alternativas no promovidas

Se probaron dos alternativas aisladas de agrupación de matrices, sin modificar la referencia publicada. Ambas conservaron paridad financiera con diferencia máxima absoluta 1,1921e-7 dentro de las tolerancias previas. También pasaron 54 casos técnicos. Redujeron medias geométricas de p50 para 64/256 activos entre un 6,1 % y un 11,7 % según alternativa y repetición, con hasta 50193920 bytes adicionales de pico. No satisficieron el límite de empeoramiento del 5 % en las condiciones de una fila.

La segunda alternativa mantuvo el bucle original en ese tamaño. La variación medida, de hasta 54 %, impide atribuir todas las diferencias al algoritmo. No se elige la mejor repetición ni se relajan márgenes después. Se conserva el bucle de referencia y se guardan políticas, código alternativo, cifras y hashes en [optimization.json](optimization.json) y los artefactos externos. No se presenta una aceleración incorporada al predictor.

Se conservan dos tentativas previas incompletas del script: argumento `warmup` inexistente y petición de 256 filas contra `max_batch=128`. Se corrigió el benchmark para respetar bloques de 64, sin ampliar límites ni omitir activos. Los errores no son resultados de entrenamiento ni datos de la tabla.

## Lo que sigue pendiente

No se entrenó el lector ni se midió una mejora predictiva. Sus direcciones tienen gradientes externos comprobados, pero siguen en sus valores iniciales. Recuperar una lectura o pasar una máscara no demuestra utilidad financiera ni memoria semántica resistente a cualquier lesión. Los [antecedentes consultados](source_audit.md) no establecen esa conclusión ni prioridad científica para esta adaptación.

La cobertura temporal y la recuperación técnica no sustituyen el estudio con las mismas filas, cortes, controles fuertes, MAE por sesión e incertidumbre por bloques. Siguen pendientes la especialización aprendida y la evaluación autónoma del controlador. El lector as-of anterior tampoco se ha integrado en su ajuste protegido. Continúan cerrados la campaña completa y el test final.

Los recibos legibles por máquina separan [pruebas](verification.json), [calidad](quality.json), [mutación](mutations.json), [coste](cost.json), [proceso](process-cost.json) y [artefactos externos](external-artifacts.json). Las muestras, predicciones técnicas por fila y capturas pesadas se conservan fuera de Git.

Con el entorno local CUDA y la extensión nativa compatibles ya configurados, los puntos de entrada comprobados son:

```bash
uv run --no-sync pytest -q tests/models/titans/test_bank_readout.py tests/models/titans/test_financial_bank_readout.py tests/memory/test_bank_readout_session.py
uv run --no-sync python reports/engineering/bank-readout-20261021/benchmark.py --output /tmp/mars-bank-readout-cost.json
```

La salida del benchmark debe ser una ruta nueva. Un archivo con estado `running_not_complete` corresponde a una ejecución parcial y no acredita una medida completa.
