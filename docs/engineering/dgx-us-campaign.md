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
  índice y de la especificación. Cada tramo tiene una sola copia, guardada con el bloque máximo
  y comprimida con zstd, que sirve a cualquier tamaño de bloque. Ningún tramo medido toca
  Parquet tras la primera pasada. La reserva final nunca se empaqueta.
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
- **Búsqueda declarada antes de ejecutar** (`configs/experiments/dgx-us-lr-search-v1.json`,
  69 trabajos en los orígenes 2005–2007 con semilla 42 y meseta de validación). Nivel *search*:
  AdamW a 3·10⁻⁴, 10⁻³ y 3·10⁻³; media exponencial de pesos (δ = 0,995) y AdamW sin calendario a
  10⁻³ y 3·10⁻³; retirada de modalidades (ModDrop) de 0,15 y 0,3; dropout de fusión de 0,1 y
  decaimiento de pesos desacoplado de 0,1. Nivel *chains*: continuación 2005→2006→2007 sin y con
  contracción y ruido. Nivel *probes*: una época de cada familia para medir su coste. Nivel
  *readers*: el lector m3, el núcleo cognitivo con herramientas y dos lectores exigentes.
  Criterio fijado de antemano: menor media del MAE por sesión de validación de la época
  elegida; con menos de un 0,5 % de diferencia relativa se conserva 10⁻³ sin más, la receta del
  protocolo. Solo usa validación de orígenes de desarrollo.
- **Lector en la DGX** (`configs/titans/dgx-us-reader-v1.json`): bloques de 3.584 filas. El lector
  con puente de evidencias más exigente necesita 972 MiB para repetir un bloque diferenciable
  con el banco lleno, dentro del máximo de 1 GiB; el gradiente del segmento es la media sobre
  etiquetas sea cual sea el bloque.

## Decisión de la búsqueda (20 de noviembre de 2026)

Criterio de la descripción de `dgx-us-lr-search-v1.json`, aplicado por código
(`decision.py`, sin intervención) sobre el MAE por sesión de validación de la época elegida de
cada ajuste, semilla 42:

| Variante | 2005 | 2006 | 2007 | Media | Frente a AdamW 10⁻³ | Época elegida (de las recorridas) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `titans_mac` | 0,014131 | 0,012411 | 0,012743 | 0,013095 | +0,000 % | 2/5, 4/7, 2/5 |
| `titans_mac_lr3e-4` | 0,014154 | 0,012429 | 0,012753 | 0,013112 | -0,131 % | 5/8, 2/5, 5/8 |
| `titans_mac_lr3e-3` | 0,014147 | 0,012435 | 0,012759 | 0,013114 | -0,144 % | 2/5, 1/4, 3/6 |
| `titans_mac_ema` | 0,014165 | 0,012413 | 0,012754 | 0,013111 | -0,121 % | 5/8, 7/10, 2/5 |
| `titans_mac_ema_lr3e-3` | 0,014182 | 0,012443 | 0,012763 | 0,013129 | -0,262 % | 2/5, 1/4, 2/5 |
| `titans_mac_sf` | 0,014147 | 0,012423 | 0,012751 | 0,013107 | -0,092 % | 6/9, 2/5, 3/6 |
| `titans_mac_sf_lr3e-3` | 0,014152 | 0,012416 | 0,012757 | 0,013109 | -0,105 % | 2/5, 2/5, 1/4 |
| `titans_mac_mdrop` | 0,014121 | 0,012417 | 0,012749 | 0,013096 | -0,005 % | 4/7, 2/5, 3/6 |
| `titans_mac_fdrop` | 0,014132 | 0,012421 | 0,012761 | 0,013105 | -0,073 % | 4/7, 3/6, 1/4 |
| `titans_mac_mdrop30` | 0,014142 | 0,012411 | 0,012765 | 0,013106 | -0,083 % | 2/5, 5/8, 3/6 |
| `titans_mac_wd` | 0,014115 | 0,012426 | 0,012750 | 0,013097 | -0,017 % | 6/9, 1/4, 2/5 |

Mejor variante: `titans_mac` (+0,000 % frente a AdamW 10⁻³); receta elegida con el margen declarado de 0,5 %: `titans_mac`.

| Continuación | 2006 | 2007 | Media | Época elegida (de las recorridas) |
| --- | ---: | ---: | ---: | --- |
| `titans_mac` | 0,012411 | 0,012743 | 0,012577 | 4/7, 2/5 |
| `titans_mac_chain` | 0,012423 | 0,012747 | 0,012585 | 0/3, 0/3 |
| `titans_mac_chain_sp` | excluido | excluido | — | titans_mac_chain_sp/2005 falló: La contracción de F3 solo se aplica a una continuación desde su padre |

Mejor opción: `titans_mac` (+0,000 % frente al ajuste desde cero); elegida con el margen de 0,5 %: `titans_mac`.

- **Receta de la campaña: AdamW 10⁻³** (`configs/titans/dgx-us-core-v1.json`), sin
  continuación entre orígenes. Ninguna variante mejora la receta del protocolo; ModDrop 0,15 y el
  decaimiento de pesos quedan a menos de un 0,02 % y las demás variantes de optimizador por
  detrás.
- **La continuación no ayuda en validación**: la cadena eligió en 2006 y 2007 su estado heredado
  (época 0), es decir, ninguna época de ajuste en la ventana nueva mejoró el estado del origen
  anterior, y el ajuste desde cero queda un 0,066 % por delante.
- `titans_mac_chain_sp` no pudo evaluarse: su receta con contracción y ruido se declaró también
  para el primer origen, que no tiene padre, y `run_titans_window` lo rechaza («La contracción de
  F3 solo se aplica a una continuación desde su padre»). Es un defecto de la declaración de la
  búsqueda, no del código; la variante queda excluida con su causa y sus orígenes 2006 y 2007
  bloqueados.
- **Determinismo de extremo a extremo**: `titans_mac_chain/2005`, que no tiene padre, reproduce
  exactamente `titans_mac/2005` (0,014131074, época 2 de 5) en dos trabajos distintos.
- Los niveles *probes* y *readers* de la búsqueda no se ejecutan: la campaña completa mide el coste
  de los lectores en su primer nivel y el plazo manda.

## Planificación dentro del plazo

Con 3 trabajos a la vez, un ajuste de Titans-MAC de 2005 tarda de media 10,3 min (22 ajustes de
la búsqueda) y el coste crece con las filas de ajuste: los 19 orígenes suman 94,9 orígenes de
2005. Solo Titans-MAC con 3 semillas son unas 49 h de un trabajador, unas 16 h de pared; los 11
modelos con ajuste propio, más los 11 lectores, superan con holgura los días que quedan si los
lectores y las redes cuestan como un ajuste. Por eso:

- **La paciencia no se reduce.** Simulada sobre las 22 historias reales de la búsqueda con la regla
  exacta de `selection.advance_selection`, una paciencia de 2 cambia la época elegida en 7 de
  22 ajustes y empeora su MAE de validación entre un 0,09 % y un 0,23 %; con paciencia 1, en 10 de
  22. La ganancia del ajuste sobre su estado inicial es del orden del 0,5 %, así que esas épocas
  tardías importan.
- **Semilla antes que nivel** (`execution.order: seed_tier`): cada semilla recorre todos los
  niveles antes de la siguiente. Si el plazo (`budget.deadline_utc`) se agota, faltan las
  últimas semillas de todos los modelos, no niveles enteros: el cerebro holográfico, los puentes
  y los refinamientos tienen al menos una semilla completa en los 19 orígenes. Los trabajos, sus
  dependencias y sus identidades son los mismos que con el orden por niveles.
- El coste real de los lectores y de las redes se mide en la propia campaña (los lectores m0–m3
  van en el primer nivel, detrás de su padre) y se reparte la memoria según lo medido.

## Regularización declarada

`FinancialConfig.fusion_dropout` (0,1 o 0,2, la rejilla de las referencias) activa el dropout de
las capas de la fusión multimodal. `FinancialConfig.modality_dropout` (hasta 0,5) retira en ajuste
cada modalidad presente distinta del precio, fila a fila, por la misma ruta que una ausencia
real: bit de presencia falso y entradas a cero (ModDrop, Neverova et al., 2015). Sin declararlas,
la identidad y el cálculo no cambian. Las máscaras salen de un generador propio sembrado con la
semilla del predictor, la época del ajuste y la huella del lote, así que la pasada sin grafo, su
repetición con grafo de la acumulación y una reanudación sortean exactamente las mismas
máscaras, y el generador global no se toca. Las pruebas demuestran la reanudación exacta con el
grafo del tramo y con la repetición por bloques.

## Ejecución y presupuesto de memoria

- **Copia fija del código.** La campaña corre desde un `git worktree` en el commit autorizado:
  `scripts/dgx_campaign.sh` antepone el `src` de su copia a la instalación editable. El primer
  lanzamiento de la búsqueda corría desde el árbol de desarrollo y se invalidó a los seis
  minutos al cambiar el código: los trabajos en marcha se detuvieron con «El código o la
  configuración numérica cambiaron durante el recorrido» y los siguientes no tuvieron permiso.
  Ese estado se conserva como registro; la búsqueda se relanzó en un estado nuevo.
- **Un único presupuesto de 64 GB.** La memoria CUDA no se carga al cgroup, así que el tope es
  `mars-titan.slice` (anfitrión) más `budget.cuda_gib` por trabajo. Con tres trabajos de 8 GiB el
  slice tiene 40 GiB: búsqueda 25 GiB y preparación 15 GiB. Antes de este cambio, la preparación
  de entradas murió por OOM a 56 GB porque la bajada de su límite no se aplicó, y en ese rato el
  total de la máquina superó los 64 GB pedidos.
- **Dónde se va el tiempo.** Un perfil con py-spy de un trabajo en ajuste muestra que el hilo
  principal pasa la mayor parte del tiempo en Python: validaciones e integridad por paso,
  bucles de emisión y observación, la repetición con grafo de la acumulación (26 %) y la norma
  del recorte de gradiente (10 %, que sincroniza por `error_if_nonfinite=True`). Por eso cada
  trabajo deja huecos en la GPU y varios trabajos a la vez escalan: con 2 trabajos cada época de
  2005 tarda 55–65 s y la GPU está al 72–84 %; con 3, 70–80 s, un 20 % más de caudal total y un
  84 % de SM ocupados.
- **Trazas de aprendizaje.** `execution.learning_traces` lleva las trazas de #448 a todos los
  ajustes y lectores de la campaña: cada 8 pasos del optimizador, la pérdida, las normas del
  gradiente y de los pesos y la razón de actualización por grupo de parámetros, la norma total
  frente al umbral de recorte y la tasa de aprendizaje; dentro de la memoria de Titans, las
  puertas α, η y θ con su saturación, la pérdida asociativa y los pesos rápidos. Se acumulan en
  el dispositivo sin sincronizar y no entran en la identidad del ajuste.
- **MPS.** El servidor MPS funciona en la GB10 y no cambia el resultado (misma huella SHA-256 de
  gradientes con y sin MPS), pero con dos trabajos no mejoró el tiempo por época (60–65 s frente
  a 55–65 s): el límite está en el hilo de Python de cada trabajo, no en la concurrencia de
  kernels. Sigue activo para la búsqueda mientras se mide con más trabajos.

## Lanzamiento de la campaña completa (20 de noviembre de 2026)

- **Qué corre.** `mars-titan-campana.service` en `mars-titan.slice`, desde el `git worktree`
  fijo en c1ecc639 con la declaración 393fc98e: 1254 trabajos, orden `seed_tier`, plazo
  2026-12-02T18:00Z y permiso expreso registrado con nota. La comprobación inicial de la cola
  arrancó durante unos 6 s cada una de las 11 arquitecturas con ajuste propio, con datos reales y
  la copia empaquetada verificada, y todas quedaron «bien». Los lectores no pasan por esa
  comprobación porque necesitan a su padre ajustado: arrancan en cuanto este termina.
- **Reproducibilidad entre estados.** `titans_mac/2005/seed-42` de la campaña, que graba trazas
  de aprendizaje, reproduce exactamente el de la búsqueda, que no las grababa: mismo MAE de
  validación (0,014131073766871805, época 2) y las tres tablas de predicciones idénticas fila a
  fila (194 655, 100 689 y 414 341 filas). Las trazas no tocan la aritmética del ajuste. En
  tiempo, 534 s frente a 449 s en la búsqueda (pausado y reanudado), con otra mezcla de trabajos
  simultáneos: eso no aísla el coste de las trazas, que se mide con 2006 y 2007.
- **Reparto de los 64 GB.** Slice de 40 GiB: preparación 15 GiB y campaña 25 GiB. A eso se suman
  3 trabajos × 8 GiB de CUDA. La campaña no declara `MemoryHigh`.
- **`MemoryHigh` por debajo de `MemoryMax` estrangula.** La preparación tenía `MemoryHigh=13G` y
  un conjunto de trabajo anónimo de 13,96 GiB, sin caché que liberar. El núcleo la detenía en cada
  asignación: `memory.pressure` *full* del 83,9 % y 1,2 millones de eventos `high`. La copia de
  evaluación de 2016 llevaba 1,47 GB en 28 min, cuando la de 2015 tardó 4 min. Con `MemoryHigh`
  igual a `MemoryMax` (el mismo presupuesto) la presión bajó al 12,8 % y escribió 670 MB en
  2,5 min. Regla: en unidades con memoria anónima, el presupuesto es `MemoryMax`; un `MemoryHigh`
  menor no ahorra memoria, solo frena.
- **Primeras medidas con 3 trabajos.**
  - Uso de GPU entre el 59 % y el 96 % (media de unos 82 %).
  - CPU ociosa al 60 % y espera de E/S del 6 %.
  - 7,1 GiB de memoria anónima entre los tres trabajos. El resto del cgroup es caché de página
    de las copias empaquetadas.
  - Cada intento de la cola guarda `resources.peak_rss_bytes`. Los 9,8 GiB de pico de un ajuste
    de Titans-MAC incluyen las páginas mapeadas de la copia.
- **Costes de la búsqueda por trabajo, origen 2005, con 3 trabajos a la vez.**
  - Titans-MAC: 9,0 min de media entre 2005 y 2007.
  - Redes de áreas, grupos de activos y atención entre activos: de 3,1 a 4,3 min.
  - Controles con memoria desactivada o congelada: de 2,7 a 2,8 min.
- **Cuando termine la preparación**, sus 15 GiB pasan a la campaña: 4 trabajos con 32 GiB de
  anfitrión y 4 × 8 GiB de CUDA, siempre que la memoria anónima de los orígenes tardíos lo permita.

## Proyección fusionada, continuaciones y copias empaquetadas (20 de noviembre de 2026)

### Proyección del codec en los lectores

- Con py-spy, el einsum del codec era entre el 37 % y el 42 % del tiempo de `mars_titan_m1` y
  `mars_titan_m2` en 2006. En aarch64, einsum suma con FMA y el núcleo nativo, que multiplica y
  suma por separado, nunca se activaba. bf8b2472 añade `project_codec_block_fused`
  ([codec episódico](episodic-codec.md#aritmética-fusionada-en-aarch64)).
- **Datos reales.** Se recorrieron los 2.202 bloques de los cuatro tramos de 2005 (3.306.229
  filas), con el mismo camino de datos que el lector. Einsum y el núcleo dieron los mismos bits en
  todos los campos de `EpisodicEncoding`. La codificación completa, con 4 hilos y la campaña en
  marcha, mejoró así:
  - validación, de 29,0 a 18,0 s;
  - calibración, de 26,4 a 17,2 s;
  - evaluación, de 42,6 a 28,2 s;
  - ajuste, de 71,2 a 50,6 s.

  Son entre 1,41 y 1,61 veces más rápido. El resumen está en
  `~/.local/state/mars-titan/dgx-titans-us-v31/monitor/codec-real-check-2005.json`.

### Continuaciones de estado

| Estado | Commit | Motivo | Trabajos heredados |
| --- | --- | --- | --- |
| `todos` | c1ecc639 | Lanzamiento | — |
| `todos-bf8b2472` | bf8b2472 | Proyección fusionada | 25 |
| `todos-f3802dd4` | f3802dd4 | Ruta real del padre heredado | 26; los 3 lectores fallidos empiezan de nuevo |

- **Padres enlazados.** La herencia enlaza los resultados terminados, y el lector verifica a su
  padre sin admitir enlaces intermedios. Los tres lectores de 2007 de `todos-bf8b2472` fallaron
  con «El destino incluye un enlace intermedio». Desde f3802dd4, la campaña pasa la ruta real.
  Con el estado real, los tres lectores aceptan su padre heredado.
- **Huella binaria del enlace nativo.** La identidad del lector incluye `native_sha256`, la
  huella del archivo `.so`. Compilar de nuevo el enlace en otro `worktree`, aunque las fuentes
  no cambien, cambia esa huella. Por eso `mars_titan_m1/2008/seed-42`, pausado en
  `todos-bf8b2472`, falló al reanudar con «La identidad de la ventana ha cambiado». Si las
  fuentes nativas no cambian, la continuación reutiliza el mismo binario.
- **Permisos de la carpeta de una copia.** `stream` exige carpetas que solo la cuenta pueda
  escribir. Crear `packed-inputs/2018/train` con `mkdir -p` (0775) hizo fallar la preparación de
  2018. Se corrigió con `chmod 700`.
- **Reinicio declarado.** `inherit --restart-paused` deja fuera los trabajos pausados, que
  empiezan de nuevo en el destino. Sirve cuando el código nuevo cambia la identidad de sus
  ajustes y el punto de recuperación ya no puede reanudarse.

### Una copia empaquetada para todos los bloques

La telemetría de la cola mostró que el primer lector de cada origen reconstruía las cuatro copias:

| Trabajo | Duración | Reconstrucción | Fracción |
| --- | --- | --- | --- |
| `mars_titan_m0/2005/seed-42` | 577 s | 375 s | 65 % |
| `mars_titan_m0/2006/seed-42` | 788 s | 230 s | 29 % |
| `mars_titan_m0/2007/seed-42` | 728 s | 455 s | 63 % |
| `mars_titan_m0/2008/seed-42` | 798 s | 510 s | 64 % |

Mientras tanto, los demás lectores del origen leían Parquet (`busy_fallbacks`).

- **Causa.** La copia exigía el bloque del consumidor: 8.192 filas en los ajustes y 3.584 en los
  lectores. Las copias de 2008 con uno y otro bloque tienen la misma huella SHA-256, porque
  ningún instante supera 3.584 filas: solo cambiaba el recibo. Aun así, cada cambio entre ajuste
  y lector reescribía la copia completa. En toda la campaña suman del orden de 30 a 45 horas de
  trabajo.
- **Corrección.** Una copia por tramo, guardada con el bloque máximo (8.192). Un instante del
  lector original tiene como mucho 8.192 entradas, así que ninguno ocupa más de una fila
  guardada. `split_events` lo comprueba en cada copia. La reproducción divide cada instante
  desde su primera fila, como el lector original, con cualquier tamaño de bloque. Si una copia
  tuviera instantes divididos, solo serviría a los bloques que dividen el suyo. Los demás leerían
  el original (`block_fallbacks`), sin reconstruirla.
- **Compresión zstd.** En la copia de 2008, zstd de nivel 1 ocupa el 0,418 del tamaño sin
  comprimir, frente al 0,470 de lz4, y descomprime a 2,5 GB/s. Las copias de los 19 orígenes y
  los cuatro tramos ocuparían unos 600 GiB con lz4 y no cabían, junto con las salidas de la
  campaña, en el disco libre. Con zstd ocupan unos 535 GiB.
- **Coste del cambio.** La huella de la implementación entra en el recibo de cada copia y en la
  identidad de los ajustes que la leen. Por eso todas las copias se reconstruyen una vez, y los
  trabajos pausados empiezan de nuevo (`--restart-paused`).

### Perfil de los trabajos: verificación sin relectura y filas de la vista

Perfil con py-spy 0.4.4 (`record --nonblocking --idle --threads`, 60 s a 40 Hz, sin pausar los
procesos) de dos trabajos en marcha, y muestreo por segundo de los bytes leídos (`rchar`):

| Trabajo | Hilo | SHA-256 de la copia | Filas de la vista |
| --- | --- | --- | --- |
| `mars_titan_m3/2009/seed-42` (lector) | principal | 19,5 % | 15,1 % |
| `titans_mac/2010/seed-42` (ajuste) | lectura anticipada | 21,9 % | — |

En el ajuste, el hilo principal pasó el 21,3 % de la ventana esperando el primer bloque de la
pasada, mientras la lectura anticipada calculaba la huella. En los dos trabajos, la lectura
anticipada pasa más del 74 % del tiempo esperando a que el consumidor libere sitio en la cola:
cuando hay datos, la entrada no limita.

- **Huella en cada pasada.** `verified_signature` volvía a calcular el SHA-256 de la copia entera
  al abrirla en cada pasada. Son entre 9 y 12 GiB por pasada en 2009-2011, a 2,1 GB/s desde la
  caché de páginas o a unos 0,45 GB/s desde el disco, y cada trabajo hace de 9 a 18 pasadas.
- **Corrección.** Tras renombrar la copia, el constructor comprueba la huella sobre el archivo
  ya en su sitio. El sello publica la firma de ese momento: dispositivo, inodo, tamaño y ctime.
  Una pasada que encuentra esa misma firma no vuelve a leer el archivo. Escribirlo,
  sustituirlo o cambiar sus permisos mueve el ctime, que una cuenta sin privilegios no puede
  fijar; es el criterio de la caché de huellas de las entradas del corpus. Con otra firma, se
  lee entero y se exige la misma huella (`rehashed_copies` en el recibo). Con otro contenido,
  la copia se rechaza (`invalid_copies`) y se reconstruye. La construcción también deja de
  releer la copia para verificar sus filas: lee el archivo dos veces en lugar de tres.
- **Filas de la vista.** `view_rows` abría el Parquet de cada uno de los 4.198 activos y sus
  etiquetas en cada llamada. Cada época de validación y cada tramo final la llaman. En la vista
  de 2010, la primera llamada tarda 19,6 s en validación y 5,3 s en evaluación; repetida, 0,005
  y 0,038 s. Ahora se lee una vez por objeto de vista y tramo, y cada llamada devuelve copias.
- **Coste del cambio.** La huella de la implementación cambia. Las copias se reconstruyen y los
  trabajos pausados empiezan de nuevo. Los terminados se heredan: ni `event_archive` ni
  `candidate_walk_forward` están en sus listas de código. La herencia usa `--finish-active`,
  así que los trabajos activos terminan con su versión.
- **Sin cambiar.** El sello SHA-256 de cada lote en CPU (`CPUDecisionBatch._digest`, 8 % del hilo
  principal del lector) es la garantía de que el lote no cambia. El hilo principal del ajuste
  está limitado por Python: la GPU está al 17 %, `backward` ocupa el 11 % y la norma del
  recorte el 5 %, junto con las comprobaciones de finitud. Queda pendiente medirlo con nsys.

## Campaña v2: selección por IC de rangos y numérica declarada (20 de noviembre de 2026)

`configs/experiments/dgx-us-campaign-v2.json` sustituye a la v1, que se detuvo con 41 trabajos
completados. La v1 queda como registro de la variante «selección por MAE»; su estado no se borra.

### Por qué se detuvo la v1

- **Los lectores se emitían sin entrenar.** La regla de la v1 exige una mejora de 10⁻⁵ en el MAE por
  sesión de validación. Las épocas de un lector mejoran entre 10⁻⁶ y 10⁻⁵, así que casi ninguna
  cuenta y se emite el estado inicial, que el lector conserva como candidato. MAE de validación de
  `mars_titan_m1`, semilla 42:

  | Origen | Época 0 | Época 1 | Época 2 | Época 3 | Elegida |
  | --- | ---: | ---: | ---: | ---: | --- |
  | 2006 | 0,0124111 | 0,0124128 | 0,0124122 | 0,0124118 | 0 |
  | 2008 | 0,0126519 | 0,0126432 | 0,0126423 | 0,0126417 | 3 de 6, con el MAE aún bajando |
  | 2010 | 0,0198602 | 0,0198531 | 0,0198536 | 0,0198535 | 0, aunque la época 1 mejora 7,1·10⁻⁶ |

  Por eso `m1`, `m2` y `m3` coinciden hasta el cuarto decimal en los seis orígenes medidos (IC de
  rangos 0,0436 y MAE un 0,204 % mejor que la predicción cero en evaluación). La mitad del cómputo
  de la campaña comparaba lectores sin ajustar.
- **El MAE no ve el IC.** Con errores gaussianos, predecir con correlación ρ reduce el MAE frente a
  la predicción cero en un factor √(1 − ρ²) ≈ 1 − ρ²/2. Medido con la semilla 42 en los orígenes
  2005–2012: el MAE por sesión de evaluación mejora un 0,18 % (Titans-MAC) y un 0,20 % (lectores)
  sobre la predicción cero, con un IC de rangos diario de 0,041 a 0,044 (t ≈ 8,5). Un IC de 0,040 y
  uno de 0,055 difieren en 7·10⁻⁴ de MAE relativo, menos que el margen del 0,5 % de la búsqueda.
- **La búsqueda miró donde no había datos.** Entre 2000 y 2008 ninguna fila tiene noticias ni
  fundamentales; en 2012 las tienen el 13,2 % y el 55,9 %. La búsqueda de la v1 (orígenes
  2005–2007) no podía medir nada multimodal.
- **La identidad numérica dependía de la afinidad del proceso.** PyTorch cuenta los núcleos de la
  máscara de afinidad al arrancar, y `cpu_threads` e `interop_threads` forman parte de la identidad
  numérica de cada ajuste. El 20 de noviembre se fijó el hilo principal de cada trabajo en los diez
  núcleos X925 con un temporizador externo: los trabajos que arrancaron ya fijados registraron 10
  hilos en lugar de 20 y, al reanudar, fallaron con «El código o la configuración numérica cambiaron
  durante el recorrido». El temporizador se retiró. Quedan fuera de cualquier comparación los
  trabajos de la v1 con 10 hilos: `titans_mac/2013`, `titans_mac_disabled/2006`,
  `titans_mac_frozen/2006` y `titans_mac_frozen/2007` (completados) y los cuatro en curso o
  pausados de ese intervalo. Son válidos los 37 completados con 20 hilos: `titans_mac` y
  `mars_titan_m0` de 2005 a 2012 y `m1`, `m2` y `m3` de 2005 a 2011.

### Reglas de la v2

- **Selección y parada por IC de rangos** (`session_rank_ic`), una enmienda declarada del
  protocolo: cada ajuste y cada lector minimizan la pérdida (1 − IC) / 2 del IC de rangos medio por
  sesión de su validación, con al menos 30 activos por sesión, paciencia 3 y sin mejora mínima. La
  validación se puntúa también en el estado inicial, el MAE se sigue midiendo en cada época y la
  serie por sesión del IC queda en el informe del ajuste. La ventana vuelve a calcular la
  puntuación del estado elegido con las tablas que confirma. Un modelo puede declarar su propia
  regla: `titans_mac_mae` es el mismo núcleo elegido por el MAE del protocolo.
- **Numérica.** `fp32_ieee_attention_v1` deja activo solo el núcleo matemático de la atención de
  PyTorch. Con tensores FP32 en CUDA, `scaled_dot_product_attention` elegía el núcleo de memoria
  eficiente de CUTLASS (`fmha_cutlassF_f32`), que no pasa por cuBLAS: medido en la GB10, su error
  máximo frente a FP64 es de 5·10⁻⁷, el doble que el del núcleo matemático (2,5·10⁻⁷).
  `torch_deterministic_algorithms_v2` activa el determinismo obligatorio de PyTorch sin el relleno
  de la memoria sin inicializar. La identidad de cada trabajo registra las dos políticas.
- **Hilos declarados** (`execution.threads`: 20 intra y 20 interop). Cada trabajo los fija al
  arrancar, siempre de forma explícita, y los exporta a OpenMP y OpenBLAS, así que su identidad
  no depende de dónde se ejecute (véase el relanzamiento del 20 de noviembre).
- **Presupuesto de GPU real.** Las ventanas de ajuste y de lector llamaban a `require_cuda()` con
  su valor por defecto, los 6 GiB del portátil de 8 GB, que rebajaba en silencio el tope de la
  campaña: con 8 GiB declarados, cada trabajo de la v1 tuvo 6. Con la atención en el núcleo
  matemático, `area_network` y `area_network_asset_groups` fallaron en 2013 con «6.00 GiB
  allowed». Ahora la ventana exige CUDA sin fijar un tope propio (`require_window_cuda`) y la
  campaña declara 16 GiB de GPU y 24 GiB de memoria por trabajo, con 4 trabajadores.
- **Bloques de orígenes.** Las copias empaquetadas de los 19 orígenes suman unos 570 GiB y no
  caben a la vez en el disco. La cola se ejecuta por bloques (`--origin`), empezando por
  2013–2016, y las copias de un bloque se borran cuando termina. `budget.min_free_gib` sube a 60.
- **Variantes de una sola variable.** Además de los 22 modelos de la v1, 31 variantes del núcleo
  `titans_mac`, cada una en su propio nivel. Una prueba
  (`tests/training/test_dgx_campaign_v2.py`) exige que cada variante difiera de su control en una
  única variable declarada.

  | Familia | Variantes | Control |
  | --- | --- | --- |
  | Regla de selección | `mae` | `titans_mac` |
  | Gradiente interno | `autograd` | `titans_mac` |
  | Memoria | `cascade` (PT5), `huber`, `eta015` | `autograd` |
  | Memoria | `stability` (PT1) | `eta015` |
  | Memoria | `timescales` | `titans_mac` |
  | Fusión | `film` (PT6), `wide` | `titans_mac` |
  | Fusión y memoria | `cascade_film` | `cascade` |
  | Atención | `diffattn` | `titans_mac` |
  | Capacidad | `l2`, `p16` | `titans_mac` |
  | Entradas y objetivo | `nols`, `aux` | `titans_mac` |
  | Entradas y objetivo | `rank`, `scalar` | `nols` |
  | Entradas y objetivo | `ic` | `scalar` |
  | Entradas y objetivo | `ic_rank` | `ic` |
  | Regularización | `mdrop`, `fdrop`, `wd` | `titans_mac` |
  | Optimizador | `ema`, `sf`, `lr3e-4`, `lr3e-3` | `titans_mac` |
  | Modalidades | `macro` (macro como condicionante) | `titans_mac` |
  | Modalidades | `no_charts`, `no_macro`, `no_news`, `no_fundamentals` | `titans_mac` |

  `macro` (`presence_modulation: macro_film_v1`) usa los 140 indicadores de la cesta (70 series y
  70 fórmulas) como condicionante: el bloque macro es el mismo para todos los activos de una
  sesión, así que sumado a la fusión desplaza todas las filas a la vez y no puede cambiar su
  orden. La modulación reescala cada coordenada de los cuatro bloques propios del activo con
  1 + tanh de una proyección lineal del bloque macro, más un desplazamiento; empieza en cero y
  una modalidad ausente sigue anulada. Las cuatro retiradas (`removed_modalities`) quitan una
  modalidad del predictor en el ajuste y al emitir, por la ruta de una ausencia real: miden qué
  aporta cada una cuando el modelo puede adaptarse a no tenerla. El vector del gráfico es un
  ResNet18 de la misma ventana de precios y ocupa el 85 % de las muestras.

- **Criba fijada antes de ver resultados.** Las variantes se ajustan primero con la semilla 42 en
  los orígenes 2013–2016 (`--seed 42`). Una variante se promueve al resto de orígenes y semillas
  si la media de su IC de rangos de validación supera a la de su control en al menos 0,003. Una
  modalidad aporta si retirarla baja esa media en al menos 0,003 y es prescindible si la
  diferencia queda dentro de ±0,003. Solo se usa validación.
- **Un origen cada vez.** `scripts/dgx_campaign_phases.sh` recorre un archivo de fases
  (`configs/experiments/phases/`), cada una una orden `run` limitada con `--origin`, `--seed` y
  `--tier`. Con trabajadores repartidos entre varios orígenes, cada uno leía una copia distinta de
  17 a 21 GiB y la caché de página del grupo no las contenía: presión de E/S del 53 % y GPU al
  16 %. Con los cinco trabajadores en el mismo origen, la GPU sube al 64–90 % y la presión de E/S
  baja al 11 %. La comprobación inicial solo se hace en la primera fase.
- **La comprobación inicial detectó tres declaraciones inválidas** antes de gastar cómputo: el
  objetivo de IC solo sustituye a la cabeza escalar (de ahí `scalar` como su control) y la caja de
  PT1 exige una puerta de momento inicial menor que su techo de 0,3 (de ahí `eta015`).

### Relanzamiento del 20 de noviembre: hilos, ajustes sin informe y memoria del controlador

La campaña v2 se relanzó en el commit `10cd0c47` tras tres hallazgos:

- **Los hilos entre operaciones no estaban fijados.** Mientras no se fijan de forma explícita,
  PyTorch responde a `get_num_interop_threads()` con los núcleos de la máscara de afinidad del
  hilo que pregunta. Con 20 declarados en un equipo de 20 núcleos, `apply_threads` no fijaba
  nada, y al sujetar después el hilo principal de un trabajo a los diez núcleos X925 su identidad
  numérica pasaba a 10 hilos: todos los trabajos fallaban en su siguiente guardado («El código o
  la configuración numérica cambiaron durante el recorrido») o en su siguiente intento («La
  identidad de la ventana ha cambiado»). Ningún trabajo del lanzamiento anterior llegó a
  completarse. `apply_threads` fija ahora siempre los dos valores y una prueba lo comprueba
  cambiando la afinidad después. Con el arreglo, el hilo principal de cada trabajo vuelve a
  sujetarse a los núcleos rápidos: sin sujetarlo pasaba el 36 % del tiempo en los A725, cuya
  capacidad es el 72 % de la de un X925.
- **Un ajuste que moría antes de su primer estado no podía reintentarse.** La carpeta del ajuste
  existía y el informe no: ni podía empezar ni continuar («El informe de la ejecución no es
  regular»). `resumable_run` retira esa carpeta y las tres ventanas que ajustan (Titans, lectores
  y postentrenamiento) empiezan de nuevo.
- **La memoria que libera la GPU no vuelve sola al sistema.** El controlador abierto de NVIDIA
  (590.48.01) guarda las páginas de sistema que libera en reservas propias
  (`NVreg_EnableSystemMemoryPools`, 0x211 por defecto: 4 KiB, 64 KiB y 2 MiB) y solo las devuelve
  bajo presión de memoria, con sus reductores `nv-sysmem-alloc-node-*`. Esas páginas no cuentan
  ni como libres ni como disponibles: tras parar seis trabajos quedaron 74 GiB ocupados sin
  ningún proceso, que los trabajos siguientes reutilizaron. Consecuencias para este equipo de
  memoria unificada: el pico de memoria de GPU alcanzado sigue pesando sobre producción hasta que
  la presión lo recorta, así que la campaña no pasa de 4 trabajos a la vez; crear un contexto de
  CUDA fuera de MPS puede fallar por memoria con poca memoria libre aunque haya mucha
  disponible; y cualquier tarea de datos que escriba muchos gigas (restaurar, empaquetar, copiar)
  se ejecuta en una unidad con `MemoryHigh` y ritmo limitado, porque su caché de página, sumada a
  la memoria de GPU, llegó a detener a los contenedores de producción el 30 % del tiempo durante
  unos minutos.

Con cuatro trabajos en el mismo origen y el hilo principal sujeto, una época de `titans_mac` en el
origen 2013 tarda unos 8 minutos. El IC de rangos de validación de su época 3 es 0,02154, el
mismo valor que en el lanzamiento anterior con otro commit: el recorrido es reproducible.

## Campaña conjunta de EE. UU. y China (20 de noviembre de 2026)

### Datos

La copia de datos para la DGX (`MARS-TITAN-DGX-US-CN-completo.zip`, 169,32 GB en Drive) se
descargó truncada: el zip, generado al vuelo, se cortó a los 71,85 GB y no tiene directorio
central. Sus 1.190 entradas completas se leyeron por su posición con un índice propio, sin
ejecutar nada del paquete, y contienen todo lo que entrena:

| Grupo | Contenido | Destino bajo `/home/gonzalo` |
| --- | --- | --- |
| `Datos multimodales US` | kit de EE. UU. ya restaurado (idéntico: mismo `DATOS-US.json`) | sin cambios |
| `A00-macroeconomia` | series macro de EE. UU. y China | `history-2000-20261007/full-macro` |
| `A01-preparados-v31` | precios y contextos preparados | `materialization/prepared-v3.1` |
| `A02-etiquetas-v31` | etiquetas y catálogo de la edición | `materialization/targets-v3.1`, `edition-catalog-20261013` |
| `A03-muestras-codificadas-v31` | muestras codificadas de China y cachés de vectores | `materialization/encoded-history-v3.1` |
| `A04`–`A06` | vistas de EE. UU. (informe), China (13 cortes) y conjuntas (19 cortes) | `campaign-a-v2-views-v31-20261013/views` |

Cada volumen se comprobó con la huella SHA-256 de su manifiesto antes de descomprimirlo y los
157.846 archivos se incorporaron sin sobrescribir ninguno (no existía ninguno). Los originales
`B01`–`B06` (series, noticias, imágenes, tablas, externos y embeddings) quedan comprimidos en
`/home/gpic/TITANS-IA/MARS-TITAN-DGX-US-CN`, comprobados con el CRC-32 del zip y su SHA-256.
Falta el resto de `B07-procesados` (intermedios y ediciones anteriores) y lo que venga detrás:
no entrena nada de la edición v3.1. Del repositorio del paquete no se tomó código: su rama
`main` no añade la campaña conjunta al motor.

La edición histórica v3.1 tiene 4.198 activos y 15.874.289 muestras en EE. UU. y 810 activos y
2.824.687 muestras en China. En el origen 2013 la vista conjunta tiene 6.080.386 filas de ajuste
(5.353.824 de EE. UU.); en el de 2023, 16.204.399.

### Motor y declaración

`comparison_campaign` acepta el mercado `US+CN`. Su cohorte es la vista completa de cada origen,
byte a byte (`cohort.rule: entire_source_view`, `write_full_view`): no se elige ningún activo, así
que ni los objetivos ni la cobertura intervienen. La cohorte por huella sigue siendo solo de
EE. UU. Cada mercado conserva su calendario, los dos comparten ventana y cada sesión de la
selección es un mercado en un instante. `configs/experiments/dgx-us-cn-campaign-v1.json` declara
los mismos modelos, recetas y reglas que la v2 de EE. UU.: la diferencia entre un modelo de la
conjunta y el mismo modelo, origen y semilla de EE. UU. es el mercado añadido. No hay criba
propia: se ajustan los niveles principales y después las variantes que promueva la criba de
EE. UU.

Antes de encolar la campaña se validó el recorrido completo con el motor y una vista reducida (45
activos de cada mercado del origen 2013, dos épocas como máximo): la comprobación inicial, el
ajuste de `titans_mac` y el lector `mars_titan_m1` terminaron y los tres tramos se emitieron con
filas de los dos mercados (evaluación: 11.295 de EE. UU. y 10.665 de China). Es una prueba de
funcionamiento, no un resultado. La validación a tamaño completo como quinto trabajo se detuvo:
con cuatro trabajos de EE. UU. en marcha provocaba intercambio a disco.

### Relevo de campañas

`scripts/dgx_campaign_relay.sh` recorre un plan cuyas fases pertenecen a varias campañas, cada
una con la copia fija de su commit autorizado: las dos campañas se alternan en lugar de competir
por la GPU, la memoria unificada y la caché de página de sus orígenes. El avance se guarda junto
al plan, que se puede ampliar por el final. Una línea `limpia|<estado>/packed-inputs/<año>`
retira las copias empaquetadas de un origen cuando sus fases terminan, y una línea
`prepara|<copia fija>|<estado>|<receta>|<año>` las construye antes de su primera fase con un solo
proceso (`scripts/dgx_prepare_inputs.py origin --archive` de la copia fija, con las rutas de
índices, copias y huellas que usa la cola de ese estado). La guardia de producción
ya no para el relevo: crea un archivo de pausa, los trabajos paran en su siguiente barrera y, con
diez lecturas seguidas de memoria y disco recuperados, la pausa se retira y la fase se repite
desde sus checkpoints. Una fase que falla nada más arrancar detiene el relevo sin consumir el
plan. Tras un reinicio del equipo, `mars-titan-arranque.service` vuelve a crear MPS, la guardia,
la afinidad y el relevo, que continúa por la fase que tocaba.

El plan del día 20 alternaba la criba de EE. UU. y la conjunta en 2013–2016, seguía con los 22
modelos principales de 2023 a 2017 y terminaba con las semillas 43 y 44 de 2021–2023: 692
trabajos, con una estimación de unos 3 por hora. No llegó a anotar ninguna fase: la revisión del
día 21 lo reordena con el coste ya medido.

### Revisión del 21 de noviembre: coste medido, orden del plan, modalidades y disco

**Coste medido.** Con los 20 primeros trabajos de EE. UU. terminados (origen 2013, semilla 42,
cuatro en paralelo, todos al primer intento), un trabajo ocupa de media 1,8 horas de un
trabajador, no la hora de `titans_mac`:

| Nivel | Modelos y horas de cada uno | Horas de trabajador |
| --- | --- | --- |
| núcleo | `titans_mac` 1,25; `m0` 0,63; `m1` 8,30; `m2` 2,35; `m3` 2,54 | 15,1 |
| cognitivos | `m3_k4` 2,15; puerta constante 2,02; cognitivo 2,01 | 6,2 |
| controles | desactivado 0,70; congelado 0,92 | 1,6 |
| redes entre activos | ocho modelos, de 0,70 a 1,58 | 8,9 |
| refinamientos | `m3_k2` 2,09 | 2,1 |
| puentes | anillo 2,59; completo y propio, en pausa tras 3,2 y 2,3 horas | unas 9 |

`mars_titan_m1` es el caso aparte: 45,8 horas de CPU en 8,3 de reloj (cinco o seis núcleos
ocupados de continuo), cuando `m2` y `m3` gastan 1,7. No se ha perfilado todavía; mientras
tanto es el camino crítico de cada fase, porque empieza cuando termina el ajuste de su padre.

El coste crece con las filas de ajuste del origen: 5,35 millones en 2013 y 13,78 en 2023 en
EE. UU.; 6,08 y 16,20 en la conjunta. Con esa escala el plan del día 20 pedía unas 2.400 horas
de trabajador y hasta el plazo caben unas 1.100 (281 horas de reloj con cuatro trabajadores, sin
descontar pausas ni la construcción de las copias).

**Orden.** El plan se reordenó ese día con el coste ya medido y, tras leer el origen 2013, se
sustituyó por el de las campañas v3 y conjunta v2 (sección siguiente), que es el vigente
(`configs/experiments/phases/dgx-relay-plan-20261121.txt`). La primera fase conjunta a tamaño
completo arrancó el 21 de noviembre a las 00:49 UTC, tras pausar en su barrera los cuatro
trabajos de EE. UU. que estaban en marcha: su copia empaquetada de 2013 se construyó en 15
minutos (19,1 GiB de ajuste y 3,7 de validación) y una época de `titans_mac` son 552 pasos y
unos 8 minutos, frente a 371 pasos en EE. UU.

**Memoria de la conjunta.** En esa primera media hora `area_network` ocupó 12,9 GiB de GPU
(7,9 a 8,7 en EE. UU.) y cada trabajo 6 GiB de memoria de proceso (2,5 a 3 en EE. UU.): los
cuatro arrancaron sin copia empaquetada y esperaron a que uno la construyera. Al continuar desde
su punto guardado, con la copia ya hecha, vuelven a 2,5–3 GiB, y los trabajos nuevos arrancan con
esa cifra. Con tres redes entre activos y `titans_mac` a la vez, la conjunta pediría 15 GiB más
de GPU que el peor rato medido en EE. UU. (35,4 GiB), así que sus niveles con atención entre
activos van en una fase propia con 3 trabajadores, después de los demás con 4. Se puede relajar
cuando el monitor, que ahora muestrea también la unidad del relevo, tenga la GPU real de esos
modelos.

Los 6 GiB por trabajo del primer arranque crecen con las filas del origen: quien construye la
copia llegó a 9,2 GiB y los tres que esperaban a 6, con 6,08 millones de filas. Con los 16,2
millones de la conjunta de 2023 y cuatro trabajos, la suma rondaría el tope de 72 GiB del grupo.
Por eso el plan prepara las copias de cada origen antes de su primera fase: un solo proceso las
construye, los trabajos las encuentran hechas y arrancan con 2,5–3 GiB. La preparación se
comprobó en seco con la vista conjunta de 2013: reutiliza los mismos índices que los trabajos
(`indices/4429006f47ecff46`) sin crear ninguno. La guardia guarda además una lectura por minuto
de memoria disponible, disco libre y presión de memoria de producción (`guardia/historia.log`).

**Modalidades por mercado.** Presencia medida en las muestras codificadas de la edición v3.1
(columna `presence`, orden del contrato de máscaras):

| Modalidad | EE. UU. (15.874.289 muestras) | China (2.824.687 muestras) |
| --- | --- | --- |
| precios, gráficos y macro | 100 % | 100 % |
| noticias | 17,5 % (ninguna en 2008, 13,2 % en 2012, 35,4 % en 2023) | 13,1 % (ninguna hasta 2018, 34,1 % en 2020, 54,6 % en 2023) |
| fundamentales | 38,1 % (ninguno en 2008, 54 % desde 2012; 2.264 activos) | 0,7 % (49 activos; 4,5 % en 2022 y 6,0 % en 2023) |

La campaña conjunta es multimodal con los mismos cinco bloques y sus bits de presencia, pero en
los orígenes 2013–2016 China aporta precios, gráficos y macro: sus noticias solo entran en el
ajuste de los orígenes cuyo tramo de ajuste alcanza 2020, y sus fundamentales casi no existen en
la edición.

Todos los orígenes ajustan desde 2000 (2006 en China): el origen es el año que se evalúa, no el
primero que se usa. Lo que existe desde 2000 son los precios, los gráficos y la macro, que pasa
de 59 indicadores observados de 140 en 2000 a 87 en 2006, 109 en 2012 y 140 en 2023. Las
noticias y los fundamentales de EE. UU. empiezan en 2009 (0,6 % y 3,9 % de las muestras de ese
año; 13,2 % y 55,9 % en 2012) y las noticias de China en 2020 (34,1 %), como recoge la auditoría
de [cobertura histórica](../data/historical-coverage.md): la copia de FinMultiTime no trae
noticias de EE. UU. anteriores a abril de 2009 ni de China anteriores a marzo de 2020. Antes de
esas fechas las modalidades van enmascaradas con su motivo (`no_admissible_value` o
`source_missing`), no rellenadas.

**Primera lectura del aprendizaje (EE. UU., 2013, semilla 42).** El IC de rangos de validación
por época de `titans_mac` oscila entre 0,008 y 0,043 (0,018, 0,016, 0,022, 0,008, 0,024, 0,043,
0,029, 0,032 y 0,021), mientras que con la memoria congelada queda entre 0,036 y 0,048 desde la
segunda época y con la memoria desactivada entre 0,032 y 0,042. La época elegida de `titans_mac`
es un pico: su evaluación baja a 0,031, frente a 0,048 y 0,045 de los controles. Con esta receta
la escritura en línea de la memoria añade varianza al orden entre activos y no señal. Los
lectores sí son estables (`m3`: 0,055 a 0,056 en todas sus épocas; evaluación 0,053). Es un
origen y una semilla; las variantes de la regla de escritura son las que deben corregirlo, y por
eso el plan las adelanta a las redes de la conjunta. La criba se leerá con un contraste pareado
por sesión entre cada variante y su control, además del umbral de +0,003: elegir la mejor época
de una curva ruidosa favorece a las variantes inestables.

**Disco.** Entre las 16:20 y las 00:30 UTC el disco libre bajó de 274 a 95 GiB sin que el
entrenamiento creciera: otra sesión de trabajo del equipo escribió 238 GiB de compilaciones bajo
`~/.claude/jobs` y la caché de construcción de Docker llegó a 107 GB. Nada de eso se ha tocado.
Se retiraron las copias empaquetadas de EE. UU. de 2015 a 2017 (94 GiB; se reconstruyen cuando
llegue su fase) y la guardia de producción tiene ahora una válvula: con el entrenamiento ya
parado y menos de 35 GiB libres retira las copias empaquetadas de los estados que nombra el
plan, y después pide 110 GiB libres, en lugar de 70, para retirar la pausa.

## Campañas v3 y conjunta v2: segunda ronda programada (21 de noviembre de 2026)

`configs/experiments/dgx-us-campaign-v3.json` y `configs/experiments/dgx-us-cn-campaign-v2.json`
continúan las dos campañas por herencia (`inherit`): ninguna receta cambia, los trabajos
terminados se enlazan y los pausados se trasladan. Cambian los niveles, para poder dejar fuera
de una fase los modelos que no aportan, y se añaden los 18 modelos de la segunda ronda, fijados
antes de leer la criba de variantes. El relevo recorre el plan sin supervisión y deja un informe
por origen.

### Qué enseñó el origen 2013

Contrastes pareados por sesión de EE. UU. con la semilla 42 (`scripts/dgx_screening_report.py`).
La diferencia de decisión reúne validación y calibración (187 sesiones); la de evaluación, las 251
sesiones de 2013, que no deciden.

| Modelo | Control | Decisión | t | Evaluación | t |
| --- | --- | --- | --- | --- | --- |
| `mars_titan_m0` (lector sin banco) | `titans_mac_frozen` | +0,0068 | 4,2 | +0,0055 | 4,0 |
| `mars_titan_m1`, `m2` y `m3` | `mars_titan_m0` | −0,0003 a −0,0001 | −1,9 a −0,3 | hasta +0,0002 | menos de 1 |
| `mars_titan_cognitive` | `mars_titan_m3` | −0,0016 | −1,4 | −0,0005 | −0,5 |
| `cross_asset_attention` | `titans_mac` | +0,0083 | 1,6 | +0,0178 | 3,2 |
| `cross_asset_attention_self` | `cross_asset_attention` | −0,0220 | −2,5 | −0,0311 | −4,3 |
| `titans_mac_disabled` | `titans_mac` | +0,0053 | 1,4 | +0,0139 | 4,0 |
| `titans_mac_frozen` | `titans_mac` | +0,0032 | 0,5 | +0,0163 | 2,9 |

Tres lecturas. El lector supera a los controles con margen estadístico, pero sus niveles de
memoria episódica no cambian las predicciones: `m3` admite menos del 0,5 % de las muestras y `m1`,
que cuesta 8,3 horas de un trabajador por origen, coincide con el lector sin banco. La atención
entre activos sí aporta (su control que solo se atiende a sí mismo se hunde). Y la escritura en
línea de la memoria del núcleo empeora la evaluación frente a desactivarla o congelarla. La
media de rangos de dos modelos ya ajustados, `mars_titan_m0` y `titans_mac_disabled`, elegida
solo con validación y calibración, sube el IC de evaluación de 0,0530 a 0,0591 y su ICIR de 0,78
a 0,87: las combinaciones de modelos distintos son la mejora más segura.

### Niveles y herencia

Pasan a niveles propios, sin cambiar de receta: `mars_titan_m1` (`ladder_m1`), `mars_titan_m2`
(`ladder`), `mars_titan_m3_k4` y el control de puerta constante (`cognitive_controls`) y las
seis redes de áreas y de grupos, que no superan a sus controles (`networks_area`). Quedan en
`core` el núcleo, el lector sin banco y `m3`; en `cognitive`, el cognitivo; en `networks`, la
atención entre activos y su control. La herencia solo traslada los modelos cuya declaración no
cambia: en EE. UU. pasaron 10 terminados y 4 pausados y en la conjunta 1 y 2; los resultados de
los modelos movidos siguen en los estados anteriores, que quedan cerrados y se conservan porque
los nuevos los enlazan. Un origen cuesta así unas 11 horas de trabajador en sus niveles
principales en lugar de 42.

### Segunda ronda

| Nivel | Modelos | Control | Pregunta |
| --- | --- | --- | --- |
| `r2_xattn` | atención entre activos con la memoria desactivada y congelada | los controles del núcleo | ¿se suman las dos mejoras del núcleo? |
| `r2_reader_xattn` | `mars_titan_m0_xattn` | `mars_titan_m0` | ¿mejora el lector sobre un núcleo con atención entre activos? |
| `r2_disabled` | sobre la memoria desactivada: objetivo de IC por sesión, entradas en rangos, macro como condicionante y media exponencial de pesos, con sus controles de cabeza escalar y sin escalas | cadena hasta `titans_mac_disabled` | ¿qué cambios de la criba sirven sin la memoria en línea? |
| `r2_capacity` | tamaño oculto de 128 y de 32 | el de 64 con el mismo gradiente por autograd | ¿sobra o falta capacidad? |
| `ventana_2016`, `ventana_2023` | el núcleo y sus dos controles ajustados desde abril de 2009 | los mismos con el ajuste desde 2000 | ¿ayudan los años sin noticias ni fundamentales? |

Cada modelo nuevo de ajuste se construyó en CPU con su receta antes de declararlo; el lector
sobre la atención entre activos se valida al ejecutarse. La escritura fusionada solo cubre la
dimensión 64, así que los tamaños 128 y 32 usan el gradiente por autograd y su control también.

### Ajustar desde 2000

Se mantiene el ajuste desde 2000 como base: son el 23 % de las muestras de EE. UU., traen dos
regímenes que no vuelven a aparecer (2000–2002 y 2008) y en este problema, con IC de pocas
centésimas, la varianza pesa más que el sesgo. Las modalidades ausentes van enmascaradas, así
que el modelo aprende a predecir con precios, gráficos y macro cuando no hay más. El riesgo es
que el mercado de 2000–2008 ya no se parezca al de hoy. No se supone: `walk_forward.fit_months`
acota el ajuste a los meses finales del tramo, y los niveles `ventana_2016` y `ventana_2023`
comparan el núcleo y sus controles ajustados desde abril de 2009 con los ajustados desde 2000,
con la misma validación. Van en una fase propia al final de su origen porque su tramo de ajuste
tiene otra copia empaquetada.

### Informes y promoción sin supervisión

El relevo admite una línea `orden|<ejecutable>|<argumentos>`, que ejecuta una herramienta entre
dos fases y nunca detiene el entrenamiento. Con ella el plan llama a dos herramientas, instaladas
en `~/.local/share/mars-titan/herramientas`:

- `dgx_screening_report.py` escribe `report.md` y `report.json` por origen y campaña en
  `~/.local/state/mars-titan/informes`: IC por sesión de cada modelo y mercado, contrastes
  pareados con el control declarado en `configs/experiments/dgx-us-campaign-v3-contrasts.json`,
  combinaciones por media de rangos elegidas con validación y calibración, el agregado de los
  orígenes de la criba y la comparación de la conjunta con EE. UU. en las sesiones comunes.
- `dgx_campaign_promotion.py` aplica, al terminar el origen 2013 de EE. UU., la regla fijada de
  antemano y edita solo las fases posteriores a la suya: en 2014–2016 siguen las 12 variantes con
  mayor t pareada, los contrastes de modalidades y los controles que necesitan; a los orígenes
  finales de las dos campañas pasan las que superan a su control en al menos 0,003 con t de al
  menos 2. Si falla, no toca el plan, que es válido sin promoción, y deja el error registrado.

El veredicto de la criba sigue siendo la media de los orígenes 2013–2016: lo da el informe
agregado de 2016.

### Justificación de cada decisión

Cada decisión se apoya en una medida de esta campaña y, cuando existe, en el estudio del estado
del arte del 20 de noviembre (22 artículos de 2025 y 2026 leídos a texto completo: Titans, MIRAS,
ATLAS, HOPE, LaCT, las críticas al aprendizaje en tiempo de prueba, escalado y conjuntos, y
predicción de la sección cruzada de rentabilidades). Ninguna cifra externa se toma como ganancia
esperada: fija qué se contrasta.

| Área | Decisión | Medida propia | Respaldo externo | Dónde se contrasta |
| --- | --- | --- | --- | --- |
| Ventana de datos | Ajustar desde 2000 | 2000–2008 es el 23 % de las muestras de EE. UU.; sin noticias ni fundamentales | Ventanas crecientes en Gu, Kelly y Xiu (RFS 2020) | `ventana_2016` y `ventana_2023` |
| Modalidades | Las cinco, con máscaras; el macro también como condicionante | Presencia por año y mercado (tabla anterior) | El macro es constante en la sesión y no puede ordenar activos por sí solo | `v_no_*`, `v_macro`, `titans_mac_disabled_macro` |
| Mercados | EE. UU. y EE. UU. con China, emparejados | Conjunta frente a EE. UU. en sus sesiones: +0,0080 con la memoria congelada (t = 2,1) | — | Informe de cada origen (`--compare`) |
| Métrica y selección | IC de rangos por sesión; decisión con validación y calibración | La mejora del MAE crece con el cuadrado del IC y no distingue 0,040 de 0,055 | — | Contraste pareado por sesión |
| Memoria en línea del núcleo | No depender de ella: los modelos finales se cruzan con la memoria desactivada | Desactivarla o congelarla mejora la evaluación en EE. UU. (t = 4,0 y 2,9) y la decisión en la conjunta (t = 4,0 y 4,1) | Liu et al. (ICML 2026): la escritura con pérdida clave-valor equivale a una atención lineal aprendida | Variantes de la regla de escritura y `r2_*` |
| Regla de escritura | Contrastar puerta de momento, caja de estabilidad, Huber, cascada y escalas de tiempo | Curva de validación inestable (0,008 a 0,043 por época) | La puerta de olvido y el momento son las palancas mayores de las ablaciones de Titans; Huber es la idea de Yaad (MIRAS) | `v_eta015`, `v_stability`, `v_huber`, `v_cascade`, `v_timescales` |
| Memoria episódica del lector | No repetir `m1` ni `m2` | Diferencias pareadas con el lector sin banco de 0,0003 como mucho; `m1` cuesta 8,3 horas | — | `ladder` solo en 2013 |
| Lector en dos etapas | Mantenerlo en el nivel principal | +0,0068 sobre el control congelado (t = 4,2) | — | Todos los orígenes |
| Atención entre activos | Cruzarla con la memoria desactivada y con el lector | +0,0178 en evaluación (t = 3,2); su control sin vecinos pierde 0,0311 | MASTER (AAAI 2024, arXiv:2312.15235) | `r2_xattn`, `r2_reader_xattn` |
| Capacidad | Probar 128 y 32, pero crecer con conjuntos | El núcleo tiene 186.119 parámetros | Con datos fijos, un conjunto alcanza mejor pérdida que un modelo mayor (Kim et al., 2025, arXiv:2509.14786); la «virtud de la complejidad» está discutida (Kelly, Malamud y Zhou, 2024; Nagel, 2025) | `r2_capacity` y combinaciones |
| Objetivo | Contrastar el IC por sesión y las entradas en rangos, también sin memoria en línea | — | Las pérdidas de ranking mueven poco el IC en la literatura revisada: se mide, no se supone | `v_ic`, `v_rank`, `r2_disabled` |
| Regularización y optimizador | Media exponencial de pesos, AdamW sin calendario, dropout, decaimiento y dos tasas | La búsqueda de la v1 los juzgó por MAE en años sin noticias | — | Criba y `titans_mac_disabled_ema` |
| Combinaciones | Media de rangos elegida con validación y calibración; semillas 43 y 44 en 2023 | Evaluación de 0,0530 a 0,0591 con dos modelos | Conjuntos de semillas y de familias (Lobacheva et al., 2020, arXiv:2007.08483) | Informe de cada origen |
| Numérica | FP32 IEEE, determinismo e hilos declarados, sin cambios | El recorrido se repite bit a bit | — | Identidad de cada trabajo |

### Lo que queda fuera de este plazo y por qué

- **Rasgos de precio de horizonte largo y de liquidez (252 a 512 sesiones) e intensidad de
  noticias.** Es la mejora con más respaldo (Gu, Kelly y Xiu, 2020; Jiang, Kelly y Xiu, 2023) y la
  única que llegaría a todas las filas, pero cambia la representación de la edición de datos y
  sus huellas: necesita la cadena de preparación del equipo de origen y una edición nueva.
- **Un vector por noticia con agregación aprendida y recencia, y un codificador de texto
  financiero sin anticipación.** Hoy se promedian por igual las noticias de cinco sesiones con un
  MiniLM general. Requiere la caché de vectores por noticia y volver a codificar el texto.
- **Regla Omega y Muon interno (ATLAS), retenciones de MIRAS, memoria continua de HOPE, memoria
  multicabezal de 64 por cabeza.** Piden código y núcleos nuevos. Sus ganancias publicadas son
  pequeñas, proceden de un solo grupo y de una ejecución por configuración, y aquí la memoria en
  línea todavía no supera a la congelada: primero debe ganar alguna regla de la criba.
- **Tamaño oculto de 128 con la memoria en línea.** El estado de memoria crece con el cuadrado de
  la dimensión y la escritura fusionada solo cubre 64; por eso la capacidad se contrasta con la
  memoria desactivada.

### Plan vigente

| Bloque | Qué recorre | Fin estimado |
| --- | --- | --- |
| 1 | Origen 2013: la conjunta termina sus niveles y la segunda ronda; EE. UU. termina sus puentes, la segunda ronda y las 31 variantes; promoción e informes | 22 de noviembre |
| 2 | Origen 2023 de las dos campañas: niveles principales, segunda ronda, variantes promovidas, semillas 43 y 44 de los modelos que entran en las combinaciones y contraste de la ventana | 25 de noviembre |
| 3 | Orígenes 2014–2016: resto de la criba de EE. UU. y conjunta emparejada | 28 de noviembre |
| 4 | Orígenes 2022 → 2017 con los niveles del 2023, hasta donde llegue el plazo | 2 de diciembre |

El 2 de diciembre a las 18:00 UTC la cola deja de lanzar trabajos por su fecha límite declarada
y un temporizador para el relevo; los estados, los resultados y los informes se conservan.

### Primeras lecturas de la conjunta

Con los tres primeros modelos del origen 2013 (semilla 42), la escritura en línea vuelve a
perder: desactivar la memoria da +0,0172 de IC de decisión (t = 4,0) y congelarla +0,0134
(t = 4,1). En las sesiones de EE. UU., los modelos ajustados con EE. UU. y China mejoran a los
ajustados solo con EE. UU.: +0,0080 de decisión con la memoria congelada (t = 2,1; +0,0066 en
evaluación) y +0,0111 en evaluación con el núcleo (t = 2,9). El IC de evaluación de China es
menor que el de EE. UU. (0,028 frente a 0,054 con la memoria congelada), con 810 activos por
sesión y sin noticias ni fundamentales en ese origen. Son tres modelos y una semilla.

## Etiqueta, rasgos técnicos y tercera ronda (21 de noviembre de 2026)

### Qué resta la etiqueta

La etiqueta de la edición histórica (`data.budget_targets.residual_targets`) es

    y(t+1) = r(t+1) − alfa(t) − beta(t) · m(t+1)

donde `r` y `m` son las rentabilidades de apertura a cierre del activo y del factor de mercado
(SPY o CSI 300) y `alfa` y `beta` salen de un OLS con intercepto sobre las 252 sesiones del
calendario que terminan en la sesión de la predicción, con 126 pares como mínimo. Las dos se
conocen al predecir. Reconstruida con los precios preparados, la etiqueta del origen 2013 de
EE. UU. coincide en todas sus filas con un error mediano de 10⁻¹⁷. El predictor, que ve 64
sesiones, no puede calcular ese alfa: sin más información solo lo aproxima.

Llamando `z = r − beta · m` al rendimiento intradía cubierto de mercado, que es lo que se puede
operar, la etiqueta es exactamente `y = z − alfa`. De ahí salen dos consecuencias:

- `−alfa`, sin ajustar nada, ordena la etiqueta con un IC de rangos de 0,084, 0,099 y 0,081 en
  validación, calibración y evaluación de 2013 (t de Newey-West de 19,5 en evaluación), por
  encima de todos los modelos de la campaña sin rasgos (0,031 a 0,053). El alfa de 252 sesiones
  es en buena parte error de estimación, y la etiqueta lo resta entero.
- Con `z` el signo se invierte: `+alfa` tiene 0,045, porque una parte del alfa intradía
  persiste. Un modelo que solo aprendiera `−alfa` ordenaría bien la etiqueta y mal lo operable.

Por eso cada informe mide las dos cosas. La predicción de un modelo para `z` es la suya para la
etiqueta más alfa, en unidades de retorno. Lo comprueba un modelo de árboles con los 46 rasgos:
ajustado con la etiqueta en unidades de retorno tiene un IC de 0,109 con la etiqueta y de 0,083
con `z` al sumarle alfa; ajustado directamente con `z` tiene 0,086 con `z` y los mismos 0,109
con la etiqueta al restarle alfa. Un solo modelo en unidades de retorno, como los de la campaña
con su cabeza de cuantiles, sirve para los dos objetivos, y no hace falta otra campaña con otra
etiqueta.

### Los modelos sin rasgos frente a las referencias (EE. UU., origen 2013, semilla 42)

| Predicción | IC de evaluación | t NW | IC sin alfa | Tercio líquido | Líquido sin alfa |
| --- | --- | --- | --- | --- | --- |
| −alfa y +alfa, sin ajustar nada | 0,0813 | 19,5 | 0,0450 | 0,0510 | 0,0341 |
| Árboles con los 46 rasgos (`lightgbm_technical`) | 0,1122 | 29,3 | 0,0837 | 0,0678 | 0,0537 |
| Lectores MARS-TITAN (`mars_titan_m0`) | 0,0530 | 10,6 | 0,0487 | 0,0323 | 0,0385 |
| `titans_mac_frozen` | 0,0475 | 9,2 | 0,0484 | 0,0240 | 0,0360 |
| `cross_asset_attention` | 0,0490 | 9,3 | 0,0463 | 0,0208 | 0,0334 |
| `titans_mac_disabled` | 0,0451 | 13,7 | 0,0482 | 0,0105 | 0,0301 |
| `titans_mac` | 0,0312 | 16,9 | 0,0463 | 0,0085 | 0,0344 |

«IC sin alfa» es el de la predicción más alfa con `z`. Los modelos sin rasgos apenas superan en
`z` a `+alfa` (0,046 a 0,049 frente a 0,045). Con el contraste SPA de Hansen ningún modelo
supera a los árboles (p = 1,00) y el conjunto de confianza de modelos al 90 % solo los contiene
a ellos: con la información que reciben, los modelos de la campaña sin rasgos quedan lejos de
un modelo sencillo que sí ve el alfa.

### Experimentos tabulares: qué información cuenta

Árboles de gradiente (LightGBM, 400 árboles salvo que se diga, número elegido en validación)
ajustados solo con rasgos técnicos causales, con los mismos tramos que la campaña. Son baratos
(minutos de CPU) y dicen qué información hay antes de gastar GPU.

| Rasgos y objetivo | Validación | Calibración | Evaluación |
| --- | --- | --- | --- |
| −alfa solo | 0,0840 | 0,0986 | 0,0813 |
| 30 rasgos de `long_horizon_v1` | 0,1093 | 0,1167 | 0,1106 |
| Los mismos sin el alfa ni su t | 0,0869 | 0,0966 | 0,0877 |
| 46 rasgos de `long_horizon_v2` | 0,1107 | 0,1178 | 0,1136 |
| v2 con 1.200 árboles y pérdida de Huber en unidades de retorno | 0,1099 | 0,1201 | 0,1122 |
| v2 más etiqueta realizada de 63 y 252 sesiones y alfa de 126 | 0,1118 | 0,1179 | 0,1144 |
| v2 más horquilla y liquidez (Roll, rango, sesiones sin negociar de 252) | 0,1112 | 0,1179 | 0,1145 |
| v2 más estado del mercado (rendimientos y volatilidad del índice) | 0,1107 | 0,1162 | 0,1139 |
| v2 más calendario (día de la semana y del mes, fin de mes) | 0,1115 | 0,1247 | 0,1151 |
| v2 más todo lo anterior (68 rasgos) | 0,1127 | 0,1250 | 0,1143 |
| v2 ajustado con `z`, medido con `z` | 0,0807 | 0,0764 | 0,0869 |

Lo que se concluye, con sus medidas:

- El alfa es la pieza principal (quitarlo cuesta 0,023 de IC), pero los demás rasgos añaden
  0,03 sobre `−alfa`: hay señal real además del desplazamiento.
- La v2 añade 16 rasgos a la v1 y gana 0,003 en evaluación. Los grupos probados después añaden
  0,001 o menos, salvo el calendario en calibración (61 sesiones de otoño de 2012), que no se
  repite en validación ni en evaluación. La v2 queda como versión definitiva.
- Contraer el alfa por su error típico, `−alfa · s² / (tau² + s²)`, o dividirlo por él, empeora
  entre 0,001 y 0,006 frente a `−alfa` sin más. La idea era razonable y la medida la descarta.
- El recorte de `learning_scale` a 5 escalas robustas afecta al 2,2 % de las filas del alfa y
  no cambia su IC (0,0813 antes y después).
- En el tercio más líquido de cada sesión los árboles conservan 0,068 con la etiqueta y 0,054
  con `z`: la señal no vive solo en los valores pequeños.
- En China el archivo del factor (CSI 300) no está en este equipo. La media intradía de 252
  sesiones, que no necesita factor, tiene allí un IC de 0,059 en evaluación, igual que un alfa
  calculado frente a un factor construido con los propios activos (0,058). En EE. UU., donde se
  puede comprobar, ese factor ponderado por volumen reproduce el alfa verdadero con una
  correlación de 0,999. La v2 incluye esa media en porcentaje para que los activos de China
  reciban la información aunque falte el archivo.

### Rasgos técnicos `long_horizon_v1` y `long_horizon_v2`

`mars_titan.data.technical_features` calcula por activo, con los precios disponibles hasta cada
sesión, rasgos acotados a ±8. La v1 tiene 30: el alfa, su t, la beta y la volatilidad residual
del OLS de la etiqueta; rentabilidades de 5 a 252 sesiones y el momento 12-1 en unidades de
volatilidad; volatilidad anual y sus cocientes; distancia al máximo y al mínimo de 252 sesiones;
volumen en dólares, su tendencia, volumen anómalo e iliquidez; los movimientos de la última
sesión y de las 21 anteriores; y tres indicadores de historia suficiente. La v2 conserva esas
30 columnas bit a bit y añade 16: medias intradía y nocturna de 5, 63 y 252 sesiones (la de 252
también en porcentaje), la etiqueta ya realizada de las últimas 1, 5 y 21 sesiones, el alfa de
21 y 63 sesiones, la fracción de sesiones sin negociación y la posición del cierre en el rango.

Una vista los declara con `input_extensions: {"technical": "long_horizon_v2"}` y
`CorpusDataset` los añade al bloque del gráfico al leer (512 → 558 columnas). Como los declara
el manifiesto, la vista tiene otra huella y sus índices y copias empaquetadas son propios. Una
campaña los declara para todas sus cohortes con `inputs`.

Fiabilidad de los rasgos, que son la entrada más importante de la tercera ronda:

- Las pruebas comprueban que el alfa y la beta son los de la implementación de referencia de la
  etiqueta, que la etiqueta realizada es la etiqueta de la sesión anterior y que ninguna fila
  cambia al alterar precios o factor posteriores.
- `tests/data/technical_features_reference.py` es una segunda implementación, escrita por otro
  modelo (Gemini 4 Neon) a partir de la especificación en prosa y sin ver la primera. Coinciden
  en los precios reales de 150 activos (618.223 filas) con una diferencia máxima de 2,6·10⁻⁶,
  y una prueba lo mantiene.
- Leída a través de `CorpusDataset`, la etiqueta se reconstruye con los rasgos con un error
  máximo de 8·10⁻¹⁰.

### Rama de los rasgos técnicos (`technical_branch_v1`)

Con los rasgos en el bloque del gráfico, el núcleo con la memoria desactivada sube en el origen
2013 de un IC de validación de 0,042 a 0,091 en cinco épocas (0,071, 0,081, 0,085, 0,090 y
0,091), y con la memoria congelada a 0,086 en cuatro. Supera a `−alfa` (0,084) y se queda
lejos de los árboles con los mismos rasgos (0,110). Los rasgos entran por la misma proyección
lineal que las 512 dimensiones de la incrustación del gráfico, y los árboles parten cada rasgo
por umbrales, cosa que una proyección lineal no hace.

`models.titans.technical_branch` lee las columnas técnicas aparte, tras MAC y la mezcla entre
activos y antes de la cabeza: codificación lineal a trozos de cada rasgo ya normalizado
(Gorishniy, Rubachev y Babenko, 2022), con bordes fijos en los cuantiles de la normal entre
±5, una incrustación por rasgo, una capa oculta y un residuo que se suma a la representación
de cada flujo con una puerta que empieza en cero. Un predictor que la declara empieza con el
mismo tronco y las mismas predicciones que su control (62.001 parámetros más sobre 189.063), y
un lector ajustado sobre el núcleo recibe el residuo en la representación de trabajo. Sin
declararla, la identidad y el cálculo no cambian. Las campañas `dgx-us-campaign-v5.json` y
`dgx-us-cn-campaign-v4.json` añaden a las de con rasgos los niveles `r3_branch` (los tres
ajustes del núcleo con la rama como único cambio) y `r3_branch_more` (atención entre activos y
lectores sobre el núcleo con rama).

Piloto en el origen 2013 (semilla 42), IC de validación por época, con los rasgos v2:

| Modelo | Época 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| `titans_mac_disabled` | 0,071 | 0,081 | 0,085 | 0,090 | 0,091 |
| `titans_mac_disabled_branch` | 0,072 | 0,080 | 0,087 | 0,088 | |
| `titans_mac_frozen` | 0,033 | 0,075 | 0,085 | 0,086 | |
| `titans_mac_frozen_branch` | 0,081 | 0,082 | | | |
| `titans_mac_branch` (memoria en línea) | 0,068 | | | | |

Sin rasgos, los mismos modelos no pasaron de 0,042 (desactivada) y 0,048 (congelada). La rama
acelera el arranque con la memoria congelada y, a igualdad de épocas, no separa a la variante
sin memoria (0,088 frente a 0,090 en la cuarta). No es todavía una mejora demostrada: el piloto
sigue y la orden de promoción decide con todas las épocas del origen 2013.

Otra idea medida y descartada: readaptar cada sesión, con las etiquetas ya realizadas de las 5,
21 o 63 sesiones anteriores, una corrección lineal de la predicción de los árboles (la versión
barata de una memoria de régimen supervisada). Empeora en todos los casos, entre 0,000 y 0,021
de IC, con t de hasta −4,4: las pendientes recientes de los rasgos son ruido en este problema.

### Herramientas nuevas

- `scripts/dgx_tabular_reference.py` ajusta los árboles con los rasgos que declara la cohorte
  de un origen y escribe sus predicciones con el formato de la campaña en un estado de
  referencia (`--reference` del informe). Cada tabla lleva así su referencia sencilla con la
  misma información.
- `scripts/dgx_screening_report.py --offset` da, por modelo, el IC con la etiqueta y con `z`,
  los dos en el tercio más líquido, la t de Newey-West y la combinación con `−alfa`. Con
  `--inference MODELO` escribe `sessions.csv` y añade la inferencia de
  `scripts/dgx_ic_inference.py`: intervalos por bootstrap estacionario, Diebold-Mariano, el
  contraste SPA de Hansen y el conjunto de confianza de modelos de Hansen, Lunde y Nason. Ese
  módulo lo escribió Gemini desde una especificación y se comprobó aquí con simulaciones
  propias: tamaño del SPA y de Diebold-Mariano bajo la nula al 5 %, 5,5 % y 5,3 %; el mejor
  modelo queda en el conjunto y el peor sale en 100 de 100 simulaciones; la t de Newey-West
  coincide con un cálculo directo.
- El relevo construye por adelantado, mientras entrena una fase, las copias de la siguiente
  línea «prepara» (`MARS_TITAN_RELAY_AHEAD`, `scripts/dgx_campaign_ahead.sh`). Cada
  preparación dejaba la GPU parada unos 40 minutos en el origen 2013 y más en los posteriores,
  y el plan tiene 29.

### Propuestas de un segundo modelo y qué se ha hecho con cada una

Se pidió a Gemini 4 Neon, con un resumen anónimo de los hallazgos, que propusiera mejoras con
fuentes. Lo que propuso y lo que se ha decidido, siempre con una medida:

| Propuesta | Decisión | Motivo |
| --- | --- | --- |
| Rama tabular con codificación lineal a trozos | Hecha, como variante de un solo cambio | El núcleo con los rasgos por la vía lineal se queda en 0,091 frente a 0,110 de los árboles |
| Conexión directa de `−alfa` a la salida, o ajustar `z` | No hace falta | En unidades de retorno, ajustar la etiqueta o `z` da el mismo IC en las dos (0,109 y 0,083 a 0,086): se mide con la predicción más alfa |
| Medir `z` como objetivo económico principal | Hecho en todos los informes | La etiqueta premia `−alfa`, que con `z` tiene IC negativo |
| Pérdida de IC por sesión sobre rangos gaussianos | Ya estaba en la criba (`v_ic`, `v_ic_rank`) | Se mide con rasgos junto al resto de variantes |
| Macro por FiLM en lugar de concatenado | Ya estaba en la criba (`v_macro`) | Igual |
| Mezcla de rangos con los árboles | Hecha en los informes | La referencia tabular entra en las combinaciones elegidas en validación y calibración |
| Más rasgos: tira y afloja, quiebre del alfa, horquilla, calendario | Probados en árboles | Añaden 0,001 o menos sobre la v2; no entran |
| Memoria compartida de régimen con ajuste supervisado por sesión | Pendiente | Es otro mecanismo de memoria; antes hay que ver qué hace la memoria actual con los rasgos |
| Inferencia con dependencia y contraste múltiple (HAC, bootstrap, SPA, MCS) | Hecha | `dgx_ic_inference.py` y `--inference` |
| Estratos de liquidez | Hechos | Tercio más líquido en los informes |
| 2013 como origen de desarrollo y 2014–2023 limpios | Adoptado | Las decisiones de hoy miraron la evaluación de 2013 |
| Carteras por deciles con rotación y costes | Herramienta escrita y probada, sin integrar | Queda para la lectura económica final |

### Protocolo de evidencia

El origen 2013 es el de desarrollo: la versión de los rasgos, la rama y las herramientas se han
decidido con su validación y su calibración, y su evaluación se ha mirado. Los orígenes 2014 a
2023 no tienen todavía ningún resultado con rasgos: son la evidencia fuera de muestra y se leen
con el diseño ya fijado. El test final de 2024 sigue cerrado.

### Plan vigente (novena revisión del 21 de noviembre)

Las campañas con rasgos son `dgx-us-campaign-v5.json` y `dgx-us-cn-campaign-v4.json`, en el
commit 7c8f4f94. El plan recorre, con una fase cada vez y la preparación de la siguiente
adelantada:

| Bloque | Qué recorre |
| --- | --- |
| 1 | Origen 2013. Sin rasgos, lo que falta para emparejar cada modelo. Con rasgos: modelos principales con y sin rama, criba de las 31 variantes y de la segunda ronda, promoción, semillas 43 y 44, modelos secundarios y la conjunta |
| 2 | Origen 2023, los modelos ajustados con todos los datos: con rasgos en EE. UU. y en la conjunta (más las variantes promovidas), su pareja sin rasgos en EE. UU. y la ventana desde 2009 |
| 3 | Orígenes 2014 a 2022 con rasgos en EE. UU., con un informe que reúne los orígenes recorridos |
| 4 | Semillas 43 y 44 del origen 2023 con rasgos y la pareja conjunta sin rasgos |
| 5 | Si queda plazo: orígenes 2014 a 2022 de la conjunta con rasgos y la ventana en 2016 |

La rama técnica entra en el origen 2013 de EE. UU. La orden de promoción que sigue al primer
informe con rasgos (`dgx_campaign_promotion.py --candidates r3_`) la añade a los demás orígenes
y a la conjunta solo si supera a sus controles en validación y calibración de 2013 por 0,003
de IC con t de 2. La misma regla, con las 31 variantes de la criba, decide cuáles pasan a los
orígenes 2018 y 2023.

Cada origen lanza antes su referencia tabular en segundo plano, y cada informe lleva las dos
medidas (etiqueta y `z`), el tercio más líquido, la t de Newey-West y la inferencia frente a
`lightgbm_technical`. El plan está en `configs/experiments/phases/dgx-relay-plan-20261121.txt`.

## Agente cognitivo con herramientas

`src/mars_titan/evaluation/cognitive_agent.py` lleva a la evaluación la cadena de Karpathy
(núcleo cognitivo → herramientas → bucle de realimentación → agente que funciona) sobre las
emisiones de la campaña, sin entrenar nada nuevo ni abrir 2024:

- **Núcleo**: Titans-MAC, cuya memoria aprende en tiempo de prueba.
- **Herramientas**: modelos de la campaña agrupados por capacidad (recuperación episódica,
  bucle de refinamiento, puerta metacognitiva, puente de evidencias, pares entre activos,
  áreas holográficas).
- **Bucle**: AdaNormalHedge.TV por mercado con la pérdida (1 − IC)/2 de las sesiones cuyas
  etiquetas ya maduraron, con la madurez real de cada etiqueta de la vista (calibración y
  después S). Abstención con margen de fiabilidad madura.
- **Enrutado por régimen**: generalistas y especialistas que solo despiertan en su régimen
  (dispersión transversal reciente alta o baja × mercado al alza o a la baja, calculado solo con
  sesiones maduras), con la garantía de arrepentimiento de AdaNormalHedge para opciones que
  duermen.
- **Medida**: núcleo solo; agente; agente sin cada herramienta; núcleo con cada herramienta;
  media uniforme; realimentación desordenada en el tiempo; herramienta placebo (núcleo
  permutado entre activos en cada sesión); agente sin abstención; valor de Shapley exacto de
  cada herramienta (hasta 8), cuya suma es, sesión a sesión, la ganancia del agente. Las
  semillas se promedian por sesión, los orígenes forman una serie de días UTC y cada familia
  recibe intervalos por bootstrap circular por bloques de días, intervalos simultáneos y
  valores p de Romano y Wolf y de Holm. El informe guarda la serie por sesión de cada variante,
  la cobertura y el peso de cada herramienta por mes y por régimen.

Las pruebas cubren la validación del diseño, la causalidad (cambiar etiquetas futuras no cambia
ninguna decisión anterior a su madurez), el control uniforme, el placebo, la abstención, la
eficiencia y la simetría de Shapley y un mundo con dos regímenes en el que solo el enrutado
acierta la herramienta de cada régimen. La prueba de humo con las emisiones reales de 2005 de
la búsqueda (las «herramientas» son allí variantes de optimizador, solo para probar la
tubería) tardaba 58 s por origen y semilla con 4 herramientas y 16 subconjuntos. Tipificar
cada opción y medir su IC por sesión una sola vez por origen y semilla (`Options`), y leer la
vista una sola vez para la madurez, la deja en 22,6 s con los mismos resultados: pesos y
abstenciones idénticos bit a bit a `online_mixture` y la predicción combinada con una
diferencia de redondeo de BLAS por debajo de 10⁻¹⁴.

El informe añade el cómputo registrado del núcleo y de cada herramienta (`costs`: segundos de
pared de todos los intentos de sus trabajos en la cola de la campaña, sin contar en un lector el
ajuste de su padre) y, con Shapley, la mejora de IC de cada herramienta por hora de cómputo
propia: la pregunta de Karpathy de si un núcleo pequeño con herramientas compensa su coste.

### Multimodalidad: juego de modalidades

Las modalidades que pueden faltar en la edición con máscaras son noticias y fundamentales;
precios, gráficos y macro están en todas las filas, así que retirarlos mediría entradas fuera
de distribución. Para cada modelo declarado en `modalities` (núcleo, recuperación episódica,
núcleo cognitivo y áreas holográficas), cada coalición de modalidades leídas es la evaluación
del mismo estado elegido con las demás leídas como una ausencia real (`data.modality_ablation`:
bit de presencia falso, relleno de ausencia, disponibilidad nula, cero noticias admitidas),
también en el calentamiento, de modo que la memoria rápida y el banco episódico ven la misma
ausencia. El informe da el valor de Shapley exacto de cada modalidad, el efecto total (que es
su suma, sesión a sesión) y la interacción de Möbius de la coalición completa, positiva si
noticias y fundamentales se complementan y negativa si son redundantes, en una familia propia
con bootstrap por bloques y Romano y Wolf, y la proporción de filas que cambia cada ablación.

Las ablaciones las ejecuta `python -m mars_titan.training.campaign_ablation` sobre el estado
de la campaña (`--declaration --machine --state --views --models --variants --workers
--cuda-gib`), con el traslado de cada familia (`carry_titans`, `carry_mars_titan_arm`), sin
escribir en los trabajos: cada ablación queda en
`<estado>/ablations/<variante>/<modelo>/<origen>/seed-<semilla>/` con su recibo, que el agente
comprueba contra el informe del trabajo antes de leerla.

Al preparar esta orden apareció un fallo latente en la unión de dos piezas: con la caché de
entradas de la campaña declarada (`MARS_TITAN_TRAIN_ARCHIVE`), `window_indices.phase_sources`
entregaba también a una ablación la copia empaquetada de la lectura canónica, cuya clave no
distingue variantes de lectura, y la ablación reproducía las entradas sin ablacionar sin
avisar. Una prueba de regresión lo demostró (diferencia relativa máxima de 1,7·10⁻³ frente a la
ventana con la ausencia real) antes de corregirlo: un corpus con variante de lectura lee
siempre Parquet. El archivo corregido no entra en la identidad de los índices ni de las copias,
así que lo ya preparado sigue siendo válido.

## Análisis tras la campaña

Las tres órdenes leen el estado de la campaña sin escribir en sus trabajos y no abren 2024.

1. **Métricas de error de todos los modelos.** Con la cola pausada (la exportación comprueba
   las huellas de lo que copia), `comparison_campaign export-results --machine mars_titan
   --declaration configs/experiments/dgx-us-campaign-v1.json --state <estado> --views <vistas>
   --destination <zip>` y después `python -m mars_titan.evaluation.comparison_campaign_analysis
   --bundle <zip> --output <carpeta>`; con `--allow-partial` solo describe lo terminado, sin
   inferencia. Cada predicción se puntúa por sesión con `forecast_scores.score_sessions`, en
   lectura bruta y con la escala calibrada en la calibración común: MAE y MSE, acierto de
   dirección y precisión al alza y a la baja, IC de rangos y de Pearson por sesión, pérdida
   pinball de los cinco cuantiles, cobertura y anchura de los intervalos del 80 y el 95 %,
   interval score, Brier del signo implícito y riesgo selectivo, con las curvas de aprendizaje
   de cada ajuste, los recursos de cada trabajo y las familias de contrastes declaradas.
2. **Ablaciones de modalidades.** `python -m mars_titan.training.campaign_ablation --declaration
   configs/experiments/dgx-us-campaign-v1.json --machine mars_titan --state <estado> --views
   <vistas> --models titans_mac mars_titan_m3 mars_titan_cognitive area_network --workers 3
   --cuda-gib 8`: 4 modelos × 3 variantes × 19 orígenes × 3 semillas = 684 evaluaciones con el
   estado elegido de cada trabajo, sin reentrenar.
3. **Agente cognitivo.** `python -m mars_titan.evaluation.cognitive_agent --declaration
   configs/experiments/dgx-us-cognitive-agent-v1.json --state <estado> --views <vistas> --output
   <carpeta> --processes <n>`: el informe con todas las familias, la serie por sesión de cada
   variante y la telemetría por origen y semilla.

## Pendiente de medir

El reparto de tiempo de un paso completo en la GPU con nsys, las épocas hasta la meseta con y
sin continuación en 2006 y 2007, el coste por época de cada familia (redes de áreas, atención
entre activos, lectores y núcleo cognitivo) y el número de trabajos concurrentes que cabe en
64 GiB cuando termine la preparación de entradas.
