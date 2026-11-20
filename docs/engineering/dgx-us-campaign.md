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

## Pendiente de medir

El reparto de tiempo de un paso completo en la GPU con nsys, las épocas hasta la meseta con y
sin continuación en 2006 y 2007, el coste por época de cada familia (redes de áreas, atención
entre activos, lectores y núcleo cognitivo) y el número de trabajos concurrentes que cabe en
64 GiB cuando termine la preparación de entradas.
