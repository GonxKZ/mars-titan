# Campaña TITANS en la DGX (EE. UU., 4.198 activos)

Documento de trabajo de la campaña en la NVIDIA DGX Spark (GB10, SM 12.1, 48 SM, 121,6 GiB de
memoria unificada, 273 GB/s, CPU Grace con 20 núcleos Arm). Recoge el contrato operativo, las
medidas que motivan cada decisión y el estado de cada pieza. Se actualiza a medida que llegan
resultados. El test final de 2024 permanece cerrado.

## Contrato operativo

- **Memoria: 64 GiB como máximo.** En la GB10 las reservas CUDA no se cargan al cgroup de
  memoria: un proceso con `MemoryMax=4G` reservó 6 GiB en la GPU y `memory.current` marcó
  0,35 GiB (prueba del 20 de noviembre). Por eso el tope se aplica en dos partes: `MemoryMax`
  del scope de systemd para la parte del anfitrión y el tope del asignador de PyTorch
  (`bound_vram`) para la GPU. La suma, con el contexto CUDA de cada proceso, no supera 64 GiB.
- **Datos.** Los 703 volúmenes del paquete y sus cuatro archivos concatenados coinciden con sus
  huellas. Los manifiestos guardan rutas absolutas del equipo donde se generaron; esas rutas
  existen en la DGX como directorios reales porque el código rechaza enlaces simbólicos en
  cualquier componente.
- **Entorno.** CPython 3.12.3 del sistema (el intérprete gestionado por uv no trae
  `os.pidfd_open`, que usa el supervisor de GPU), PyTorch 2.14.0+cu130, Triton 3.8.0 y el banco
  episódico nativo compilado contra LibTorch con CUDA.
- **Paridad entre plataformas.** Las pruebas de huellas fijadas en x86_64 fallan en ARM64 porque
  la inicialización aleatoria de PyTorch en CPU y el BLAS producen otros bits. Los resultados de
  la DGX no se presentan como reproducciones bit a bit de PC1 o PC2.

## Medidas de partida en la GB10

| Pieza | Medida | Consecuencia |
| --- | --- | --- |
| Verificación de huellas al abrir cada trabajo | Decenas de GB por trabajo | Caché de huellas con firma `ctime` |
| Estadísticas de escala, ventana 2023 | 14,04 M filas a ~21.000 filas/s en un hilo (~11 min) | Índices y estadísticas compartidos por vista |
| Memoria de Titans, tramo de 8 pasos, B = 4.096 | Referencia 105,8 ms sin grafo, 259,3 ms con grafo | Limitada por ancho de banda: leer y escribir W y S |
| Gradiente analítico compilado | 29,8 ms sin grafo, 98,0 ms con grafo | ×3,6 y ×2,6 sobre la referencia |

El suelo físico de una escritura de memoria con 4.096 flujos es leer y escribir W₁, W₂, S₁ y S₂
una vez (≈512 MB), unos 1,9 ms a 273 GB/s. La referencia queda a unas siete veces ese suelo.

## Gradiente asociativo cerrado (`memory_inner_gradient: analytic_v1`)

`associative_gradient.py` escribe en forma cerrada las derivadas que la ruta histórica obtiene
con `torch.autograd.grad(create_graph=True)`. El metagradiente exterior es la derivada de primer
orden de ese grafo explícito. Comprobado en FP64 durante seis pasos encadenados con error relativo
máximo de 9·10⁻¹⁵ en estados y metagradientes, y en las pruebas
`tests/models/titans/test_associative_gradient.py` con cinco configuraciones de memoria y MAC.
Sin declararlo, la identidad y el cálculo no cambian.

## Escritura de memoria fusionada (`memory_inner_gradient: fused_v1`)

`fused_memory_write.py` ejecuta el paso de escritura completo de la memoria de profundidad 2 y
dimensión 64 en un programa de Triton por flujo: lee W₁, W₂, S₁ y S₂ una vez y escribe W′, S′ y
el residuo. La inversa recalcula la pasada directa del flujo y aplica el VJP cerrado (GELU″ =
φ·(2 − h²) y la doble derivada de la capa de normalización), sin operaciones atómicas. Cubre
profundidad 2, dimensión 64, sin recorte ni plasticidad; la configuración rechaza el resto.

| B = 4.096 flujos, FP32 | Referencia PyTorch | `fused_v1` | Suelo a 273 GB/s |
| --- | --- | --- | --- |
| Sin grafo | 9,4 ms | 2,7 ms | 2,0 ms |
| Directa + inversa | 34,1 ms | 8,0 ms | ≈5,1 ms |

Paridad con la referencia: error relativo ≤ 1,6·10⁻¹⁵ en FP64 y ≤ 6·10⁻⁷ en FP32, en estados y en
los nueve cotangentes, con y sin capa de normalización.

**Revisión externa con Gemini (`gemini-4-neon-xhigh` vía `agy`).** Derivó de forma independiente
la pasada directa y el VJP y confirmó los nueve cotangentes. Propuso un parche completo: kernel
persistente con 4 CTA por SM, inversa en cinco fases con dos matrices vivas, políticas de
desalojo de L2, `eps` como escalar de ejecución y la forma algebraica de la inversa de la capa de
normalización sin 1/r ni r³. Medido en la GB10, el parche no mejora (8,03 ms frente a 7,97 ms en
directa + inversa, 2,83 ms frente a 2,68 ms sin grafo) y empeora la paridad en FP64 de 10⁻¹⁵ a
10⁻¹⁰, porque Triton pasa `eps` como FP32. No se aplica. También se midió por separado igualar
los warps de la inversa a los de la directa (recálculo idéntico bit a bit): con 4 warps la inversa
tarda 8,04–8,14 ms frente a 7,52–7,98 ms con 8, con la misma paridad en FP64, así que se mantienen
8. Los riesgos numéricos señalados (|h| > 1,8·10¹⁹, σ(z) > 10¹³, |k| ≫ 8) quedan fuera del
dominio: las claves están normalizadas y la memoria pasa por la capa de normalización.

## Tubería de entradas

- **Copia empaquetada verificada por tramo** (`training_event_cache`): ajuste, validación,
  calibración y evaluación se reproducen desde Arrow con mapeo de memoria, con la huella del
  índice, de la especificación y del bloque. Ningún tramo medido toca Parquet tras la primera
  pasada. La reserva final nunca se empaqueta.
- **Especificación de entradas junto al índice**: se decodifica una vez por tramo y se guarda
  con la identidad del corpus, del índice, de la política y de la representación; los trabajos
  siguientes la reconstruyen sin abrir los miles de archivos Parquet del primer evento.
- **Validación en paralelo** (`prepare_workers`): las comprobaciones de cada bloque (finitud,
  ventanas de precio, vectores históricos, identificadores y huella) se reparten entre hilos y
  se entregan en el orden de entrada; el resultado es idéntico al secuencial.
- **Preparación previa** (`scripts/dgx_prepare_inputs.py`): índices, escalas, especificaciones y
  copias de cada origen antes de lanzar la cola, con las mismas funciones que el trabajo.
- **Tope de GPU por trabajo** (`budget.cuda_gib`): el asignador de PyTorch de cada trabajo no
  puede reservar más; con `MemoryMax` del servicio para el anfitrión, la suma queda acotada.

## Pruebas de humo (origen 2005, Titans-MAC, una época)

| Ejecución | Configuración | Resultado |
| --- | --- | --- |
| Base | Receta original, bloques de 1.024, gradiente por autograd, Parquet | ~5,4 sesiones/s (~10.000 filas/s), GPU al 26 % |
| Humo 1 | Bloques de 8.192, `analytic_v1`, copia del ajuste | 19 sesiones/s (~27.000 filas/s), 387 s, RSS 9,7 GB, CUDA 4,8 GB |
| Humo 2 | `fused_v1`, copias de los cuatro tramos, 3 hilos de validación | 439 s de trabajo (123 s de ajuste con dos validaciones), RSS 7,5 GiB, CUDA 4,0 GiB |

El trabajo del humo 2 dedicaba la mayor parte de su tiempo fuera del ajuste a decodificar desde
Parquet el primer evento de cada tramo para su especificación (visto con py-spy); esa lectura ya
no existe en los trabajos.

## Tamaño de los orígenes

Filas con etiqueta por tramo (suma sobre activos). El ajuste crece de 1,31 M filas (2005) a
13,78 M (2023), 10,5 veces; la suma de los 19 orígenes equivale a 94,9 orígenes de 2005.

| Origen | Ajuste | Validación | Calibración | Evaluación | Activos con ajuste |
| --- | --- | --- | --- | --- | --- |
| 2005 | 1.313.128 | 194.655 | 100.689 | 414.341 | 1.521 |
| 2010 | 3.602.637 | 274.686 | 139.027 | 570.313 | 2.155 |
| 2015 | 6.667.897 | 354.810 | 181.845 | 754.028 | 2.753 |
| 2020 | 10.722.076 | 476.924 | 244.312 | 1.015.804 | 3.728 |
| 2023 | 13.775.654 | 521.928 | 259.215 | 1.040.641 | 4.174 |

## Protocolo de aprendizaje

- **Parada adaptativa** por meseta de validación (`validation_plateau`) con paciencia y mínimo de
  épocas, en lugar del presupuesto fijo de 10/5 épocas que impusieron dos portátiles de 8 GB.
- **Continuación encadenada** entre ventanas: cada ventana parte del estado elegido de la
  anterior con el mismo modelo y semilla (`run_titans_window(initial=...)`). El código exige que
  la validación del padre acabe antes que la de la continuación, así que no hay fuga. El estado
  heredado se mide en validación y puede elegirse si ninguna época lo mejora.
- **Búsqueda de hiperparámetros declarada antes de ejecutar**
  (`configs/experiments/dgx-us-lr-search-v1.json`): ritmos 3·10⁻⁴, 10⁻³ y 3·10⁻³ en los orígenes
  2005–2007 con semilla 42 y meseta de validación. Criterio fijado de antemano: menor media del
  MAE por sesión de validación de la época elegida; con menos de un 0,5 % de diferencia relativa
  se conserva 10⁻³, el valor del protocolo. Solo usa validación de orígenes de desarrollo. El
  lector usa el mismo ritmo que su padre, como en el protocolo.
- **Lector en la DGX** (`configs/titans/dgx-us-reader-v1.json`): bloques de 3.584 filas. El lector
  con puente de evidencias más exigente necesita 972 MiB para repetir un bloque diferenciable
  con el banco lleno, dentro del máximo de 1 GiB; el gradiente del segmento es la media sobre
  etiquetas sea cual sea el bloque.

## Pendiente de medir

El reparto de tiempo de un paso de ajuste completo (kernels, CPU y huecos) con nsys, las épocas
hasta la meseta con y sin continuación, el coste por época de cada familia (redes de áreas,
atención entre activos, lectores) y el número de trabajos concurrentes que cabe en 64 GiB.
