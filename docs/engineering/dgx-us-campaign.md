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
completados. La v1 queda como registro del brazo «selección por MAE»; su estado no se borra.

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
retira las copias empaquetadas de un origen cuando sus fases terminan. La guardia de producción
ya no para el relevo: crea un archivo de pausa, los trabajos paran en su siguiente barrera y, con
diez lecturas seguidas de memoria y disco recuperados, la pausa se retira y la fase se repite
desde sus checkpoints. Una fase que falla nada más arrancar detiene el relevo sin consumir el
plan. Tras un reinicio del equipo, `mars-titan-arranque.service` vuelve a crear MPS, la guardia,
la afinidad y el relevo, que continúa por la fase que tocaba.

El plan (`configs/experiments/phases/dgx-relay-plan-20261120.txt`), en orden de prioridad:

| Bloque | Campaña | Orígenes | Semillas | Modelos |
| --- | --- | --- | --- | --- |
| A | EE. UU. | 2013–2016 | 42 | los 53 (criba) |
| B | conjunta | 2013–2016 | 42 | los 22 principales |
| C | las dos | 2023 → 2017 | 42 | los 22 principales |
| D | las dos | 2021–2023 | 43 y 44 | núcleo, sus lectores y sus controles |

A y B se alternan origen a origen. Las variantes que promueva la criba se añaden a C y D al
analizar A y B. Son 692 trabajos; con unos 3 trabajos por hora en los orígenes pequeños y menos
en los grandes, el plazo del 2 de diciembre puede cortar el final del plan, que por eso está
ordenado por prioridad.

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
