# Lectura y simulación de episodios

La medición del 24 de septiembre de 2026 recorre 24 576 muestras de un mundo sintético de 128 activos y 256 sesiones. Cada muestra contiene 64 sesiones de precios y las representaciones analíticas de noticias, gráficos, fundamentales y macro. No ejecuta codificadores neuronales ni entrenamiento en GPU.

El recorrido lee todas las modalidades desde Parquet, calcula una referencia fija sobre el evento ficticio y simula 192 decisiones con posiciones persistentes. Comprueba las huellas de todas las entradas y exige resultados contables idénticos entre las opciones de concurrencia. Se midieron tres procesos por opción, con un calentamiento en cada proceso. Los hilos numéricos se limitaron a uno y las lecturas internas de Arrow no abrieron hilos adicionales.

| Trabajadores | Mediana del recorrido | Desviación entre repeticiones | Muestras/s | Pico de RAM del proceso |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,778 s | 0,030 s | 31 606 | 161,5 MiB |
| 2 | 0,617 s | 0,016 s | 39 837 | 184,4 MiB |
| 4 | 0,544 s | 0,007 s | 45 142 | 234,3 MiB |
| 8 | 0,560 s | 0,012 s | 43 868 | 303,7 MiB |

Cuatro trabajadores redujeron la mediana del recorrido un 30 % respecto a uno y aumentaron el pico de RAM unos 73 MiB. El tiempo total de proceso, que incluye importaciones y calentamiento, pasó de 1,935 a 1,475 s. Ocho trabajadores no aportaron una mejora en esta medición. Los resultados no fijan la concurrencia de un entrenamiento neuronal que todavía no se ha medido.

Se decodificaron 58 392 576 bytes de entradas por recorrido. El informe conserva bytes observados por el sistema operativo, latencias p50, p95 y p99 por cohorte, cada repetición, versiones, equipo y huellas de código. Los contadores de lectura física y los bytes solicitados al sistema tienen significados distintos. La caché del sistema estaba caliente y otras cargas seguían activas. No se vaciaron cachés ni se detuvieron procesos.

Con cuatro trabajadores, la lectura y preparación ocuparon una mediana de 0,173 s y la simulación 0,358 s. Un perfil instrumentado separado localiza trabajo en lectura de grupos, construcción de observaciones y validación contable. Sus tiempos incluyen el coste del perfilador y no sustituyen las medidas anteriores.

La implementación mantiene Parquet y la lectura por columnas y cohortes. No hay evidencia de que descomprimir domine el recorrido completo, por lo que no se añade una copia Arrow IPC. Tampoco se añade un lector propio, un kernel CUDA ni un motor C++ antes de medir el recorrido con padres y codificadores reales. C++20 y la configuración CMake existente quedan disponibles para un candidato posterior con referencia numérica y comprobación de recuperación.

La eliminación previa de una importación innecesaria de Torch en el generador analítico sí cambió la ruta usada por la CLI. Su [benchmark](../../reports/episodes-import-benchmark.json) registró 3,09 frente a 1,97 s de mediana de proceso y 617,8 frente a 133,6 MiB de pico, con los 48 bloques Parquet idénticos por SHA. Esa optimización no acelera por sí sola los codificadores ni el entrenamiento.

Los resultados y límites de esta medición están en [episode-pipeline-benchmark.json](../../reports/episode-pipeline-benchmark.json). GPU, transferencias, energía y coste monetario figuran como no medidos. El código no los estima a partir de una muestra aislada de potencia o de la velocidad de una operación.

```bash
uv run --extra reinforcement python benchmarks/episode_pipeline.py \
  --work artifacts/benchmarks/episode-pipeline \
  --output artifacts/benchmarks/episode-pipeline.json
```

La carpeta de trabajo debe ser nueva. El máximo es ocho trabajadores, una tarea pendiente por trabajador y 8 MiB de caché por lector. Esta carga permanece por debajo de los límites de memoria de las ejecuciones científicas, pero no demuestra el consumo de una campaña neuronal completa.
