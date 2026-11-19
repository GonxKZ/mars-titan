# Kernels de Titans en la RTX 5050 Laptop de PC2

La ruta CUDA SM 12.0 fusiona por capa la escritura de pesos y de momentum de Titans-MAC. Conserva los redondeos intermedios FP32 y las reducciones de PyTorch en el backward. Los lectores episódicos y los puentes conservan el producto de puntuación en FP64 y seleccionan los primeros ocho vecinos con desempate estable. Las demás arquitecturas siguen la implementación de PyTorch en entrenamiento. La agrupación de las copias de parámetros al anfitrión conserva los SHA-256 individuales.

## Cómputo y tiempo de entrenamiento

Con PyTorch 2.14.0+cu130 y Triton 3.8.0 se comparó la revisión científica del calentamiento `e23a4403` con `559bfa0f`. El tramo de entrenamiento del origen 2020, semilla 42, aportó una sesión de 559 filas, 555 etiquetas maduras y 1.024 episodios anteriores a la decisión. En cada una de las 20 variantes se intercalaron 24 pares de actualizaciones después de cuatro de calentamiento. Cada implementación mantuvo sus propios parámetros y AdamW. Coincidieron los bits de las salidas, los gradientes, los parámetros y el optimizador en los pasos 4 y 28. La suma de las medianas para una actualización por variante dio un factor de 1,09×. El lote ya estaba en la GPU.

| Medida | Referencia | Kernels PC2 | Factor |
|---|---:|---:|---:|
| Escritura de memoria, 1.024 flujos | 847,10 µs | 333,76 µs | 2,54× |
| Ocho vecinos, 512 consultas por 1.024 episodios | 219,88 µs | 20,39 µs | 10,79× |
| Titans-MAC, pasos cronológicos 72–104, dos repeticiones | 18,78 s | 18,50 s | 1,02× |
| M3 con cuatro refinamientos, primeros 24 pasos | 29,69 s | 30,31 s | 0,98× |

El pico temporal de la escritura pasó de 48 a 32 MiB. El de la selección de vecinos pasó de 8,01 a 0,03 MiB. En M3 con cuatro refinamientos, desde la entrada al entrenador hasta volver a leer el checkpoint se midieron 141,11 y 136,48 s, una vez por versión. La medición cubre el inicio de una ventana. Se corrigió una caída de WSL al final de pytest retrasando la importación de Triton hasta el primer kernel compatible. La batería de 784 comprobaciones y dos repeticiones adicionales de 460 cerraron con código cero. Los tiempos por operación y variante y los resultados de cada repetición cronológica están en [el recibo de PC2](kernels.json).

## Parejas medidas con el calentamiento residente

En dos ventanas cortas sobre una sola GPU, Transformer con DLinear dio 1,78× y Transformer con M1 dio 1,51× durante el entrenamiento. Los checkpoints de cada modelo coincidieron con los de su ejecución en serie. La comprobación del proyecto exige 12 GiB de RAM disponibles para dos trabajos y PC2 llegó a 433 MiB de RAM libre en Windows durante el ensayo. El [selector de parejas](../../../scripts/run_pc2_gpu_pairs.py) comprueba ese margen, la VRAM y el estado de las dependencias. Si encuentra presión pide una parada recuperable. La ruta con entradas preparadas utiliza un trabajador mientras no haya medidas equivalentes de parejas bajo esa revisión.

## Reproducción por equipo

Desde la raíz, `PYTHONPATH=src python benchmarks/titans_cuda_updates.py salida.json` compara las operaciones en caliente. En otra GPU NVIDIA con Triton se puede pasar `--probe-device` para una prueba local. El archivo resultante identifica si la arquitectura está habilitada en producción. Ese modificador solo actúa en el proceso del microbenchmark. Ampliar la lista de entrenamiento exige verificar también los gradientes, el optimizador y las ventanas de ese equipo. [Programa de medida](../../../benchmarks/titans_cuda_updates.py)

La preparación y el coste de la lectura integrada en PC1 están descritos en [el ensayo de la RTX 4070](../comparison-efficiency-20261118/README.md).
