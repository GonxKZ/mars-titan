# Programa de post-entrenamiento de MARS-TITAN sobre la base v10

Programa del 22 de noviembre de 2026, rama `postentrenamiento-v10` sobre 6c19d20c. Recorre todo lo
que el repositorio hace después del ajuste principal (continuaciones, adaptadores, memorias y
correctores con etiquetas maduras, calibración, combinación en línea y semillas), lo contrasta con
la literatura primaria, corrige lo que faltaba con pruebas que fallaban antes, mide en CPU lo que
se puede medir con emisiones reales y deja preparadas, con órdenes exactas, las etapas que
necesitan la GPU. No abre el test final de 2024. Cada cifra lleva el estado, el origen y la semilla
que la sostienen; lo que no se ha medido se marca como hipótesis. Las mediciones de otros (el
laboratorio tabular y el plan del relevo del coordinador) se citan como suyas y no se repiten aquí.

## Resumen

- **Medido (v8, 2013, semilla 42, evaluación de 251 sesiones, contrastes pareados por sesión con HAC).**
  - Las dos memorias supervisadas con etiquetas maduras suben el IC de rangos de los 13 modelos: la combinada, de +0,0020 a +0,0046.
  - Sus placebos son negativos, así que la información se mide como brazo − placebo: +0,0026 (t 3,7) por activo y +0,0035 (t 5,6) por vecinos sobre titans_mac.
  - Con el padre que ya escribe en su memoria, la memoria por activo sobrevive el 94–98 % en el par base. En el par con rangos, el de la base v10, sobrevive el 76 % (t −3,6 frente al control congelado). Sección 5.1.
- **Correcciones con pruebas que fallaban antes.**
  - Las cinco reales de la auditoría del segundo enjambre: H-01, H-08(1), H-09, H-11(i) y H-12(b).
  - El arranque de los gemelos en CUDA.
  - Las del programa (sección 4).
  - El resto de la auditoría no se confirma o es una propiedad del diseño declarado (sección 7.2).
- **Medido (gemelos, v8, 2013, semilla 42; sección 6.5).** Escribir en evaluación con la misma red:
  - par base: N = +0,00003 (t 0,6);
  - par con rangos: N = +0,00113 (t 3,5), el 17 % de la ventaja, sin decaer en 2013. Es también el par sin ganancia de la v10.
- **Medido (órganos, v10, 2013, semilla 42; sección 5.4).** Los 8 brazos mejoran a su padre en evaluación: de +0,0014 a +0,0031 (t 1,6–2,9); 3 de 8 pasan Holm.
- **Medido (2013; secciones 5.5 a 5.7).**
  - Los órganos como componente de ridge_cov_active suman +0,0011 sobre la media de rangos (t 2,1 en la exploración; 1,7 en la prueba de S2).
  - La memoria con el residuo operable gana en lo operable (+0,0011 a +0,0014) y pierde en la etiqueta.
  - D1 es la memoria por activo con otro nombre.
- **En curso.** El par de la v10 en GPU; S2 en los orígenes limpios, por el relevo.
- **Hipótesis, no medido.**
  - El coste de las memorias por origen en la campaña (≈ 1 h 20 min de CPU).
  - Cualquier efecto en otros orígenes, semillas o en la v10.


## 1. Datos, protocolo y lectura de los resultados

- **Tarea.** Para cada activo, el residuo del retorno de apertura a cierre de la sesión siguiente
  frente al mercado, y = r^OC_{i,t+1} − α̂ − β̂·r^OC_{m,t+1}; la decisión se toma 5 minutos después del cierre y la etiqueta madura en la decisión siguiente
  (`matures_at`). Métrica primaria: IC de rangos por sesión, con al menos 30 activos.
- **Origen 2013 (`fold-008`).** Ajuste hasta abril de 2012, validación de abril a octubre de 2012,
  calibración de octubre de 2012 a enero de 2013 y evaluación en 2013. La identidad de la vista de
  cada trabajo es la cohorte que escribe su estado (`<estado>/cohorts/2013/manifest.json`,
  f5f4d478… en la v8 y en la v10); la vista base de `views/US/fold-008` no lo es (ab18f946…), y los
  pilotos lo comprueban antes de leer la madurez.
- **Emisiones leídas (solo lectura).** v8: `dgx-titans-us-v31/v8/semillas-42,43,44/todos-114bcf4e`
  (origen 2013, semilla 42; 1.139.633 filas en 437 sesiones: 125 de validación, 61 de calibración
  y 251 de evaluación). v8-ext1: `…/v8-ext1/semillas-42,43,44/todos-8e532348`. Estado antiguo sin
  rasgos: `…/todas/semillas-42,43,44/todos-5a6d1d94` (orígenes 2005 a 2013). La v10
  (`…/v10/semillas-42,43,44/todos-3d10a30f`) tenía a las 05:24 UTC sus 19 trabajos de 2013 en
  pausa y ninguno terminado: sus emisiones todavía no se pueden medir.
- **Contrastes.** Siempre pareados por sesión (variante menos padre en la misma sesión), con t
  HAC de Newey y West (1994) con el retardo automático del informe de criba
  (`scripts/dgx_screening_report.py`). Todo hiperparámetro se elige con la calibración (empates
  al valor menor) y la evaluación se lee una vez.
- **Placebos.** Memoria por activo: la lectura de otro activo (desarreglo con semilla fija).
  Vecinos: los valores del banco permutados entre sus filas en cada cohorte. Un efecto solo cuenta
  si su placebo no lo reproduce.

## 2. Inventario (fase 0) y skills aplicadas

Piezas posteriores al ajuste principal que hay en el repositorio; el detalle está en la matriz de la sección 3:
- Continuaciones entre orígenes (`continues` en la campaña DGX, nuevo aquí, y la cadena de la búsqueda).
- Lectores sobre el códec congelado (m0, m3, cognitivo).
- Adaptadores del paquete `posttraining/`, que no están conectados a la cola DGX.
- Correctores asociativos B6 y E1 (solo GPU).
- Las dos memorias supervisadas con etiquetas maduras, nuevas aquí: por activo y por vecinos.
- Calibración: CQR estática, PT2 en línea y reajuste periódico.
- Combinación en línea con AdaNormalHedge.TV: árbitros de versión y de modelos, y el agente cognitivo con herramientas, placebo, contexto y Shapley.
- Media de semillas (`seed_mean_v1`).
- Los gemelos de escritura (sección 6.5).

| Skill | Dónde se aplicó | Qué cambió por aplicarla |
|---|---|---|
| test-driven-development | H-01, mecanismo de los gemelos, arreglo de CUDA, H-11, H-09, H-12 y los cambios de la sección 4 | Cada prueba se vio fallar antes del código. Excepciones registradas: parte del estudio de memorias y la orden de los gemelos se escribieron antes que sus pruebas; se compensó con mutantes |
| mutation-testing | Todos los arreglos | Mutantes a mano muertos (sección 4); uno equivalente con estos límites (≤ frente a <, porque 128 no es múltiplo de 5) |
| systematic-debugging | Primer lanzamiento en GPU; paradas por presión de producción | La causa se reprodujo en la GPU con una unidad mínima antes de tocar código. La historia de la guardia (56 minutos malos en 4 h) llevó a congelar la unidad en lugar de pararla |
| verification-before-completion | Relevo, integración en v11, HEAD limpio | Clones nuevos con `git am`; integración simulada sobre v11 (78 de 79 pruebas); el error del GRU episódico se comprobó en HEAD limpio |
| testing-anti-patterns | Dobles de prueba | Sin mocks del módulo probado. El doble del asignador de CUDA reproduce un contrato medido en la GPU real |
| statistical-power | Gemelos y memorias | MDE solo con validación y calibración (sección 5.3); sin potencia observada |
| experimental-design | Gemelos | Factorial 2×2 con control inverso y paridad; semillas como réplicas con la sesión como unidad; Holm entre los dos pares |
| code-review | Revisión propia | Encontró la fuga de `MARS_TITAN_SHARED_INDICES` y que una frontera causal global rechazaría ventanas válidas de EE. UU. con China |
| commit-work | Todos los commits | Diff cacheado revisado; un commit por arreglo; mensajes en ficheros |
| performance-profiling | Gemelos y memorias | Separar la preparación en CPU de la fase de GPU (bytes leídos, %CPU, RSS, cpu.stat). El tiempo de sistema de las memorias queda como mejora pendiente |
| peer-review | Auditoría del enjambre | Estructura afirmación → evidencia → prueba → veredicto (sección 7.2); no cambió código |
| python-advanced, risk-metrics-calculation, backtesting-frameworks | Lectura | Sin cambio visible. La lista de sesgos de backtesting (anticipación, selección) ya la cubrían la madurez y la selección en calibración |


## 3. Matriz de taxonomía (fase 1)

Veredictos: **mantener** (lo actual es lo mejor respaldado o no hay motivo medido para tocarlo),
**corregir** (un fallo o un hueco que se arregla aquí), **ablación** (una sola variable frente a su
control, con su placebo cuando lo hay) y **descartar** (no aplica a esta tarea o ya se midió en
contra). Las referencias están en la sección 9.

| Dimensión | Estado en el repositorio | Opción mejor respaldada | Veredicto |
|---|---|---|---|
| Arquitectura del adaptador o corrector | Lectores episódicos sobre el códec congelado (m0, m3, cognitivo), matriz de adaptadores del paquete `posttraining/` (no conectada a la campaña DGX), correctores asociativos B6 y E1 (solo GPU) y, desde este programa, dos correctores sin parámetros sobre predicciones emitidas | Con una señal tan débil, primero un corrector no paramétrico que interpola la predicción con una lectura de memoria (kNN-LM) y no ajusta pesos; si muestra señal, un adaptador de bajo rango (LoRA) sobre el padre congelado | **Ablación**: los dos correctores sobre emisiones reales (5.1). Mantener los lectores de la v10. Los adaptadores del paquete quedan fuera hasta que un corrector sin parámetros sobreviva a su placebo (hipótesis) |
| Representación y claves | Lectores: estado oculto del códec. Vecinos: proyección aleatoria de signos (64 dimensiones, semilla 53) de las 92 entradas del laboratorio (46 rasgos normalizados y sus rangos gaussianos por sesión). Memoria por activo: el activo | Proyección aleatoria fija (Achlioptas, 2003): conserva distancias y no tiene parámetros que sobreajustar. El laboratorio del coordinador midió claves de proyección por encima de las activaciones ocultas | **Mantener**. Claves aprendidas: solo como ablación en GPU, detrás de las de la sección 6 |
| Operaciones | Núcleo MAC con memoria neuronal en línea (gradiente interno fusionado), 4 fichas persistentes, oculto 64 y una capa; correctores: EWMA por activo, k vecinos por coseno con softmax de temperatura τ, combinación z + κ·z(lectura) | Titans (memoria que aprende en evaluación) y kNN-LM (interpolación con una lectura) | **Mantener**. La atención entre activos la mide la v10 (sin emisiones terminadas) |
| Activaciones | GELU y SiLU en la memoria y la fusión | Sin evidencia ni hipótesis que justifique tocarlas en un post-entrenamiento | **Mantener** |
| Normalización | Escala robusta de aprendizaje (mediana y MAD del ajuste, recorte a 5), LayerNorm residual de la memoria; en los correctores, tipificación por sesión de la predicción y rangos gaussianos centrados de la etiqueta | Normalización de sección cruzada para un objetivo de ordenación: en la v8, los rangos por sesión de los rasgos fueron la única variante por encima de su control (+0,0042 de IC de decisión, t 2,65; +0,0055 en evaluación, t 4,86; plan del relevo, origen 2013, semilla 42) | **Mantener**. Límite documentado: la tipificación de los correctores usa solo las filas con etiqueta de cada sesión (sección 8) |
| Regularización y anclaje | Decaimiento de pesos (0 en la v10), decaimiento anclado L2-SP (`AnchoredAdamW`: el ancla son los pesos al construir el optimizador, que en una continuación son los del padre), contracción con ruido solo en continuación, proximal en memorias; no hay EWC | L2-SP y contracción con ruido atacan la pérdida de generalización del arranque en caliente; EWC protege una tarea anterior que aquí no hay que conservar | **Ablación** en GPU (sección 6): decaimiento 0,1 hacia cero frente a hacia el padre, una sola variable. **Descartar** EWC |
| Inicialización | Salidas residuales a cero (lectores, atención entre activos), fichas persistentes normales; la continuación parte del estado elegido del padre | Arranque en caliente regularizado (fila anterior). Única medición previa: en la búsqueda (2006 y 2007, semilla 42, criterio MAE) la cadena eligió el estado heredado y quedó un 0,066 % detrás del ajuste desde cero | **Ablación** en GPU: continuación frente a ajuste desde cero con la parada por IC de la v10 |
| Pérdidas | Pinball de cinco cuantiles más término de IC por sesión de peso 3 (8 filas mínimas); destilación opcional | Pinball para cuantiles; el término de IC lo mide la propia v10 con sus brazos `_noic` | **Mantener**. Destilación: **descartar** mientras no haya un maestro mejor que el alumno |
| Optimización | AdamW, tasa 1e-3, recorte de norma 1, acumulación de 8.192 filas (3d10a30f) | El coordinador midió que quitar la acumulación no acelera (85–102 frente a 107–126 pasos por minuto) y el laboratorio no vio efecto de variantes del optimizador | **Mantener** |
| Programa de tasa | Tasa constante; AdamW sin calendario (Defazio et al., 2024) implementado en F3 de #749 | Ya se ensayó en un piloto y no ganó su criterio de validación (`docs/research/comparison-training-review-20261114.md`) | **Mantener**; **descartar** el cambio en esta fase |
| Continuación entre orígenes | `continuation: previous_origin` (cadena propia) y, desde este programa, `continues` (parte de otro ajuste) | Continuar con regularización hacia el padre o con contracción y ruido (L2-SP; Ash y Adams, 2020) | **Ablación** en GPU (sección 6): cuatro brazos frente al ajuste desde cero de la v10 |
| Repaso | `posttraining/replay_continuation.py` (paquete fuera de línea) | El repaso contra el olvido (Rolnick et al., 2019) solo tiene sentido si la continuación pierde por olvido | **Hipótesis**: solo tras medir la continuación |
| Media de pesos (SWA/EMA) | `weight_average` en la receta cronológica (F2 de #749); su documento dice que no se ha ajustado ni evaluado con datos financieros | SWA (Izmailov et al., 2018) y EMA de pesos (Morales-Brotons et al., 2024) reducen la varianza del recorrido | **Ablación** en GPU, detrás de la continuación (no declarada aquí) |
| Conjuntos de semillas | `evaluation/seed_ensemble.py` (`seed_mean_v1`) | Conjuntos profundos (Lakshminarayanan et al., 2017) | **No medible hoy** en 2013 (faltan 43 y 44 en la v8); la v10 declara las tres semillas |
| Aprendizaje en línea con etiquetas maduras | Memorias C, PT2 en línea, árbitros en línea (AdaNormalHedge), agente cognitivo | Combinación en línea con realimentación retardada (AdaNormalHedge, Luo y Schapire, 2015) y conformal adaptativo (ACI, PID) | **Ablación** sobre emisiones (5.1 a 5.3) con control uniforme y placebo |
| Calibración conformal | CQR estática, PT2 en línea, recalibración periódica, conformal ponderado por regímenes | CQR (Romano et al., 2019); ACI (Gibbs y Candès, 2021); PID (Angelopoulos et al., 2023); HopCPT (Auer et al., 2023) | **Ablación**: piloto de escala e intervalos (5.2) con la regla declarada antes de leerlo |
| Adaptación en evaluación sin etiquetas | La memoria en línea de Titans ya aprende en evaluación | TTT (Sun et al., 2020) generaliza esa idea; Tent (Wang et al., 2021) minimiza la entropía de una clasificación, que aquí no existe | **Descartar** Tent; mantener la memoria del núcleo |
| Canal de datos | Vistas por cohorte con identidad, copias empaquetadas verificadas, calentamiento de 12 meses cuyas filas no se emiten, etiqueta madura en la decisión siguiente | — | **Mantener**. Consecuencia comprobada: una memoria supervisada integrada en el modelo empezaría en frío (lo mide el arranque en frío de la adenda, 5.1) |
| Generalización y evaluación | Controles congelado y desactivado por arquitectura, placebos, selección en calibración, una lectura de la evaluación, t HAC por sesión, familias de contrastes de la criba | HAC de Newey y West (1994); placebo propio por mecanismo; potencia declarada | **Corregir**: cada piloto informa sus orígenes, semillas y efecto mínimo detectable (5.3) |
| Ingeniería computacional | FP32 IEEE estricto y algoritmos deterministas en la ruta de modelo; los correctores calculan cosenos en FP32 y pesos y sumas en FP64 | Selección exacta y determinista de los k mayores | **Corregido** aquí: selección exacta sin ordenar el banco y lectura en hilos con resultado idéntico (4) |

## 4. Cambios de código (fase 2)

Veinticuatro commits locales en `postentrenamiento-v10`. Cada cambio de comportamiento se escribió con su
prueba antes y se vio fallar; donde la prueba se escribió después del código se compensó con
mutaciones (rompiendo a propósito una guarda o una fórmula y comprobando que alguna prueba falla).

| Commit | Qué cambia | Pruebas y mutaciones |
|---|---|---|
| d20b0cb2 | `memory/asset_residual_memory.py`: memoria por activo del residuo de rango maduro, con la definición del laboratorio tabular (Z, rangos gaussianos G, residuo E = G − (Z·G/n)Z, EWMA por activo, salida Z + κ·z(lectura)) | 23: igualdad con la definición del laboratorio con una sesión de retraso; el futuro (etiquetas y entradas) no cambia ninguna emisión anterior, bit a bit; una etiqueta que madura justo en la decisión se lee y una posterior no; placebo de otro activo; pausa y restauración a mitad del recorrido sin cambios; estado manipulado rechazado; NaN con sus coordenadas; test final cerrado |
| 35b487f4 | `memory/neighbour_residual_memory.py`: memoria por vecinos del mismo residuo (claves de proyección, banco de las W cohortes maduras, k vecinos por coseno, softmax con τ, valores recortados a ±3) | Igualdad con la lectura del laboratorio (cosenos FP32 frente a FP64, tolerancia 2·10⁻⁵); causalidad por perturbación del futuro, bit a bit; placebo de valores permutados reproducible; pausa y restauración; banco vacío y banco menor que k |
| 4a1a6734, 614e38c6 | `evaluation/posttraining_pilots.py` y sus declaraciones: escala e intervalos (CQR estática, PT2 en línea, regímenes) y memoria por activo sobre trabajos terminados, con la madurez de la vista de cada trabajo | Carga ordenada con la madurez de la vista; filas sin etiqueta o con objetivo distinto fallan con sus coordenadas; la tasa en línea y la configuración de memoria se eligen sin leer la evaluación; efecto plantado encontrado y su placebo no |
| 961bb9b7 | `training/comparison_campaign.py`: `continues: <padre>`. Una continuación parte del estado elegido de otro ajuste `mac_online` de su equipo en el origen anterior, con su misma semilla y su misma arquitectura; sin trabajo en el primer origen | 128 pruebas del archivo en verde (cuatro nuevas: plan y bloqueo hasta que termina el padre, guardas del padre, arquitectura de la receta y ejecución desde el padre); 10 mutantes de esas guardas y de la planificación muertos |
| 342061ee, a73b4971 | `configs/experiments/dgx-us-campaign-v10-post1.json` y tres recetas de continuación (decaimiento, L2-SP y contracción) que solo cambian su opción de ajuste respecto de la receta del núcleo de 3d10a30f (huella 3b88760f…) | 5 pruebas de la declaración y las recetas; una prueba con pasos reales del optimizador comprueba que el decaimiento anclado lleva los pesos hacia el estado elegido del padre (el mutante con ancla cero muere) |
| d7333251 | Orden de los pilotos de la v8: primero el núcleo y sus lectores | Mismo conjunto de trabajos |
| e4a7501f | Vecinos sin las filas del propio activo (`exclude_own`), para separar la historia del mismo activo de la de activos parecidos | 6 pruebas nuevas (lectura frente a la referencia con exclusión, contratos, un banco solo del propio activo lee cero, pausa y restauración, identidad, estado restaurado incoherente); 9 mutantes muertos |
| 689273d4 | Adenda del estudio de memorias: vecinos de otros activos con su placebo y su combinación, corrección en unidades de retorno sumada a los cinco cuantiles con la recalibración estática ajustada en calibración, y arranque en frío de la memoria por activo | 7 pruebas nuevas (entre ellas: la corrección conserva el orden de los cuantiles y el IC; el arranque en frío equivale a la repetición directa; la elección no cambia al barajar los objetivos de evaluación dentro de cada sesión); 9 mutantes muertos |
| 703fb445 | Selección exacta de los k vecinos sin ordenar el banco y lectura en varios hilos (sección 5.1, coste); el método pasa a v2 porque los empates en el k-ésimo coseno van ahora a la fila más antigua, como en la definición | Empates con claves canónicas (fallaba antes con y sin exclusión); selección idéntica a `np.argsort` estable con empates, filas excluidas y menos de k candidatas finitas; resultado idéntico con 1 y 3 hilos en la memoria y en el estudio; 9 mutantes muertos |
| 3e81573a | Declaración de la adenda sobre la v8 (`configs/experiments/posttraining-memories-v10-adenda.json`), escrita antes de leer ningún resultado del estudio | La comprueba `checked_study` |
| 5287c130, 954de171 | Orden de los pilotos (primero la réplica entre orígenes) y declaración del estudio de memorias sobre las emisiones de la v10 (`posttraining-memories-v10-on-v10.json`, 19 modelos), escrita cuando ninguno de sus trabajos de 2013 había terminado | La comprueba `checked_study` |
| c42fbaa8 | H-01: el bucle del agente cognitivo aprende con la correlación de la métrica declarada (rangos), no con Pearson | La prueba fallaba: una herramienta con Pearson alto y rangos negativos se llevaba el 99,96 % del peso. Una opción constante pierde 1/2 y no da NaN. 3 mutantes muertos |
| 5b448f66 | Gemelos de escritura: `carry_titans(..., evaluated_variant)` repite la propia ventana del ancla con su estado elegido y la otra política de escritura. Carga todos los tensores (parámetros y búferes) y comprueba bytes y contratos | Oráculo con el registrador de gradientes: cada gemelo repite bit a bit la ventana ajustada con la otra variante. Los búferes de learning_scale viajan. Cualquier otro cambio de contrato se rechaza. 7 mutantes muertos |
| 6f4876cb | `training/write_twins.py`: la orden `gemelos` (GPU, paridad bit a bit antes de cualquier gemelo, reanudación con huellas, picos de memoria) y `analizar` (CPU: N, D, A, I con HAC y TTWS con error delta y Fieller) | 19 pruebas, escritas después del código, sin TDD estricto; lo compensan 12 mutantes muertos |
| 4ea5e27e, f4de9cac | Declaraciones de los gemelos (v8 base, v8 con rangos y v10 con tres semillas), con la potencia medida solo en validación y calibración | La comprueba `checked_declaration` |
| e9ff43ba | Los gemelos inicializan CUDA antes de poner a cero sus máximos: el primer lanzamiento en la GB10 fallaba a los 30 s con «Invalid device argument» | La prueba usa un doble del asignador con el contrato medido en la GPU y falló igual antes del arreglo. 3 mutantes muertos |
| d94c7807 | H-11(i): la declaración del agente rechaza un contexto cuyas copias (1 + len(REGIMES) por opción) no caben en las 128 de la mezcla | La prueba fallaba (no se rechazaba). Límite exacto 24/25 modelos. 3 mutantes muertos |
| c76d7133 | H-09: la media de semillas rechaza una semilla con cuantiles cruzados | La prueba fallaba. Los empates siguen admitidos. 3 mutantes muertos |
| 803f6c7e | H-12(b): la calibración de cada ventana debe haber madurado, por mercado, antes de su primera decisión evaluada | La prueba fallaba. La prueba antigua de «etiquetas que nunca maduran» usaba un escenario sin causalidad y ahora retrasa solo la evaluación. 4 mutantes muertos |
| fc9fc2fb | H-08(1): el eje «al alza» del régimen es el signo del residuo medio, no la dirección del mercado (solo documentación) | Sin cambio de comportamiento |

El contrato causal común a los dos correctores: la lectura en la decisión t solo usa cohortes cuya
etiqueta maduró como mucho en t, escritas una vez y en orden de emisión, con reloj monótono por
mercado; las claves solo usan entradas de la propia decisión; cualquier valor no finito, cohorte
fuera de orden, etiqueta que madura antes de su decisión o fecha del test final falla con mercado,
fila e instante.

## 5. Pilotos con emisiones reales (fase 3)

Todo en CPU, en unidades acotadas (`mars-titan-post-*`, una cada vez, 8 GiB y 300 % de CPU), con
las emisiones de solo lectura y una orden reproducible por piloto:

```bash
cd /home/gpic/TITANS-IA/MARS-TITAN-postentrenamiento
export PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/gpic/.local/share/mars-titan/dgx-us-py312/bin/python
estados=/home/gpic/.local/state/mars-titan/dgx-titans-us-v31
# Estudio de memorias y su adenda (v8, origen 2013, semilla 42)
$py -m mars_titan.evaluation.posttraining_pilots memorias \
    --declaration configs/experiments/posttraining-memories-v10-adenda.json \
    --state v8=$estados/v8/semillas-42,43,44/todos-114bcf4e \
    --output <informe> --digest-cache <caché privada de huellas> \
    --inputs-cache <caché del laboratorio tabular> --inputs-names <names.json del laboratorio> --threads 3
# Pilotos por trabajo (escala e intervalos, memoria por activo)
$py -m mars_titan.evaluation.posttraining_pilots pilotos \
    --declaration configs/experiments/posttraining-pilots-v10.json \
    --state v8=$estados/v8/semillas-42,43,44/todos-114bcf4e \
    --state v8_ext1=$estados/v8-ext1/semillas-42,43,44/todos-8e532348 \
    --state antiguo=$estados/todas/semillas-42,43,44/todos-5a6d1d94 \
    --output <informe> --digest-cache <caché privada de huellas>
```

La caché de huellas debe ser privada (la orden rechaza una dentro de un estado de campaña). El
estudio de memorias sobre la v10 es la misma orden con
`configs/experiments/posttraining-memories-v10-on-v10.json` y
`--state v10=$estados/v10/semillas-42,43,44/todos-3d10a30f`, cuando sus 19 trabajos de 2013
estén terminados.

### 5.1 Estudio de memorias y su adenda (medido)

Medido en la v8, origen 2013, semilla 42, con 13 modelos. La evaluación son 251 sesiones; el padre es la emisión de cada modelo. Cada celda da el incremento pareado por sesión sobre el IC de rangos del padre, en unidades de 10⁻⁴, con su t HAC entre paréntesis. «Activo − placebo» y «vecinos − placebo» restan su placebo (información); «filas propias» es vecinos menos vecinos de otros activos.

| modelo | IC padre | activo | activo−placebo | vecinos | vecinos−placebo | otros activos | filas propias | comb. | comb.−activo | comb.−vecinos | arranque en frío |
|---|---|---|---|---|---|---|---|---|---|---|---|
| titans_mac | 0,1107 | +18,6 (2,7) | +25,9 (3,7) | +21,0 (3,3) | +34,6 (5,6) | +17,2 (2,8) | +3,7 (5,8) | +33,7 (2,6) | +15,1 (2,3) | +12,7 (1,8) | +3,9 (2,0) |
| titans_mac_frozen | 0,1094 | +19,0 (2,9) | +27,5 (4,1) | +24,0 (1,9) | +67,8 (5,8) | +16,0 (2,7) | +7,9 (1,2) | +33,9 (2,7) | +14,9 (2,3) | +9,9 (2,8) | +3,7 (2,2) |
| titans_mac_disabled | 0,1049 | +19,8 (3,1) | +28,5 (4,4) | +26,0 (2,0) | +66,2 (5,7) | +16,6 (2,7) | +9,4 (1,4) | +35,1 (2,8) | +15,2 (2,3) | +9,1 (2,6) | +3,6 (1,9) |
| titans_mac_xrank | 0,1163 | +8,9 (2,6) | +10,9 (3,1) | +13,9 (2,4) | +29,8 (5,0) | +10,3 (1,8) | +3,6 (5,5) | +19,7 (2,2) | +10,8 (1,8) | +5,8 (1,7) | +1,9 (2,2) |
| titans_mac_frozen_xrank | 0,1098 | +17,7 (2,7) | +26,6 (4,2) | +18,2 (3,2) | +32,0 (5,8) | +14,1 (2,5) | +4,1 (6,6) | +30,6 (2,6) | +12,9 (2,2) | +12,4 (1,9) | +3,5 (2,2) |
| titans_mac_disabled_xrank | 0,1059 | +15,3 (2,3) | +24,0 (3,6) | +19,6 (1,7) | +62,6 (5,8) | +12,3 (1,1) | +7,3 (6,2) | +23,7 (1,4) | +8,4 (0,7) | +4,2 (0,6) | +4,0 (2,2) |
| titans_mac_noic | 0,1038 | +19,3 (2,2) | +33,1 (3,8) | +27,8 (1,3) | +196,9 (11,0) | +29,7 (2,4) | −1,9 (−0,2) | +45,9 (2,5) | +26,6 (2,3) | +18,1 (2,8) | +2,6 (1,4) |
| mars_titan_m0 | 0,1108 | +18,6 (2,7) | +25,9 (3,7) | +20,9 (3,3) | +34,5 (5,6) | +17,2 (2,8) | +3,7 (5,8) | +33,6 (2,6) | +15,0 (2,3) | +12,7 (1,8) | +3,9 (2,0) |
| mars_titan_m3 | 0,1108 | +18,6 (2,7) | +25,9 (3,7) | +20,9 (3,3) | +34,5 (5,6) | +17,2 (2,8) | +3,7 (5,8) | +33,6 (2,6) | +15,0 (2,3) | +12,7 (1,8) | +3,9 (2,0) |
| mars_titan_cognitive | 0,1107 | +18,6 (2,7) | +25,9 (3,7) | +21,0 (3,3) | +34,6 (5,6) | +17,2 (2,8) | +3,7 (5,8) | +33,7 (2,6) | +15,1 (2,3) | +12,7 (1,8) | +3,9 (2,0) |
| mars_titan_m0_u2 | 0,1108 | +18,6 (2,7) | +25,9 (3,7) | +20,9 (3,3) | +34,5 (5,6) | +17,2 (2,8) | +3,7 (5,8) | +33,6 (2,6) | +15,0 (2,3) | +12,7 (1,8) | +3,9 (2,0) |
| mars_titan_m0_noic | 0,1044 | +19,9 (2,3) | +35,4 (4,3) | +33,6 (2,6) | +105,1 (9,1) | +25,5 (2,1) | +8,2 (6,5) | +43,3 (2,3) | +23,4 (2,1) | +9,6 (1,4) | +5,6 (2,0) |
| mars_titan_m3_noic | 0,1042 | +20,2 (2,4) | +35,5 (4,4) | +34,1 (2,7) | +104,9 (9,1) | +26,0 (2,1) | +8,1 (6,5) | +44,1 (2,4) | +23,9 (2,1) | +10,0 (1,5) | +5,6 (2,0) |

Lectura:
- Los placebos son negativos y significativos (en titans_mac: por activo −0,00073 (t −3,1), por vecinos −0,00136 (t −5,9)). Una lectura sin información diluye el padre, así que la información de cada memoria es brazo − placebo, no brazo − padre.
- **Memoria por activo** en titans_mac: +0,00186 (t 2,7) sobre el padre; la configuración del laboratorio (h = 21, κ = 0,1) es también la elegida en calibración.
- **Memoria por vecinos**: +0,00210 (t 3,3). Con solo otros activos, +0,00172 (t 2,8). Las filas del propio activo añaden +0,00037 (t 5,8).
- **Combinada**: +0,00337 (t 2,6). Supera a cada memoria sola: +0,00151 (t 2,3) frente a la de activo y +0,00127 (t 1,8) frente a la de vecinos.
- **Acuerdo** entre las dos lecturas (correlación por sesión, evaluación): 0,37; con vecinos de otros activos, 0,28.
- **Arranque en frío** (la memoria por activo vacía al empezar la evaluación): +0,00039 (t 2,0) frente a la memoria caliente. Empezar en frío no cuesta nada medible, y podría ganar algo (t 2,0).
- **Corrección de los cinco cuantiles en unidades de retorno**:
  - Sobre los cuantiles crudos, baja el pinball (t −9,9) y el MAE (t −19,3), pero empeora la puntuación de intervalo al 80 % (t +8,3) y al 95 % (t +7,6).
  - Tras la CQR estática ajustada en calibración, empeora el pinball (t +3,6) y la puntuación de intervalo.
  - No se adopta para intervalos.

**Supervivencia cuando el padre ya escribe** (la pregunta del coordinador). La diferencia es el contraste pareado del incremento del padre en línea menos el del control; «sobrevive» es el cociente de incrementos.

| Padre en línea | Brazo | Incremento del padre en línea | Control congelado: incremento · diferencia (t) · sobrevive | Control desactivado: incremento · diferencia (t) · sobrevive |
|---|---|---|---|---|
| titans_mac | por activo, h = 21 y κ = 0,1 fijos | +18,6 | +19,0 · −0,4 (t −0,4) · 98 % | +19,8 · −1,2 (t −1,0) · 94 % |
| titans_mac | por activo − su placebo, fijos | +25,9 | +27,5 · −1,6 (t −1,2) · 94 % | +28,5 · −2,6 (t −1,7) · 91 % |
| titans_mac | por vecinos, κ elegida | +21,0 | +24,0 · −3,0 (t −0,5) · 87 % | +26,0 · −5,1 (t −0,7) · 81 % |
| titans_mac | por vecinos − su placebo, κ elegida | +34,6 | +67,8 · −33,2 (t −5,5) · 51 % | +66,2 · −31,6 (t −5,2) · 52 % |
| titans_mac | combinada, κ elegidas | +33,7 | +33,9 · −0,2 (t −0,1) · 99 % | +35,1 · −1,4 (t −0,6) · 96 % |
| titans_mac_xrank | por activo, h = 21 y κ = 0,1 fijos | +13,5 | +17,7 · −4,2 (t −3,6) · 76 % | +15,3 · −1,8 (t −1,2) · 88 % |
| titans_mac_xrank | por activo − su placebo, fijos | +22,2 | +26,6 · −4,3 (t −3,2) · 84 % | +24,0 · −1,8 (t −1,2) · 93 % |
| titans_mac_xrank | por vecinos, κ elegida | +13,9 | +18,2 · −4,3 (t −3,4) · 77 % | +19,6 · −5,7 (t −0,9) · 71 % |
| titans_mac_xrank | por vecinos − su placebo, κ elegida | +29,8 | +32,0 · −2,2 (t −1,7) · 93 % | +62,6 · −32,8 (t −6,1) · 48 % |
| titans_mac_xrank | combinada, κ elegidas | +19,7 | +30,6 · −10,9 (t −3,1) · 64 % | +23,7 · −4,0 (t −0,4) · 83 % |

- El par con rangos es el que aplica a la base v10. Con parámetros fijos, la memoria por activo sobrevive el 76 % frente al congelado (t −3,6) y el 88 % frente al desactivado. En el par base sobrevive entera (94–99 %, |t| ≤ 1,7).
- Las filas «vecinos − placebo» con κ distinta en el padre en línea y en el control (0,1 frente a 0,2) mezclan la dilución del placebo, que crece con κ. Ahí la comparación limpia de vecinos es el incremento sobre el padre.

**Coste medido.** 1 h 23 min de reloj con 3 hilos, CPUQuota 300 % y 3,4 GiB de RSS. La lectura por vecinos de los 13 modelos tomó 3.736 s (los vecinos se calculan una vez para todos), la de otros activos 957 s, y cada modelo 16–19 s más. El tiempo de sistema (6.484 s frente a 2.710 s de usuario) indica asignaciones por bloque mejorables.

**Integración** como órganos lectores de la base nueva: `INTEGRACION-MEMORIAS.md`, en la carpeta del informe. Recoge los puntos de entrada, qué lee cada memoria, el contrato causal, el coste, lo que falta para ejecutarlas como trabajo de la campaña y las réplicas.

### 5.2 Pilotos de escala, intervalos y agente cognitivo

Siguen pendientes. Los 77 trabajos de `posttraining-pilots-v10.json` y los dos agentes (`posttraining-agent-v8-2013.json` y `posttraining-agent-antiguo-2005-2011.json`) van en una sola unidad de CPU después de los gemelos (una unidad a la vez).

### 5.3 Efecto mínimo detectable

Con 251 sesiones de evaluación de un origen y una semilla, el EE HAC de un incremento de memoria es de 0,0006 a 0,0013 (MDE ≈ 2,8 · EE: 0,002 a 0,0036). El de una diferencia de supervivencia es ≈ 0,0001 por activo y ≈ 0,0006 por vecinos.

En los gemelos, el MDE esperado de D es ≈ 0,0030 (par base) y ≈ 0,0047 (par con rangos). Se calculó solo con validación y calibración; no se usa potencia observada.


### 5.4 Órganos de memoria sobre la v10 (medido, 2013, semilla 42)

Cada modelo y semilla emite dos brazos fila a fila (`memory_organs emitir`, 2c0d7bc8), en una raíz con la forma de un estado de campaña:
- **combined**: z + κa·u_activo + κv·u_vecinos, con h, κa y κv elegidos en la calibración del propio origen;
- **asset_lab**: h = 21 y κ = 0,1 fijos.

La señal se suma a la predicción y a sus cinco cuantiles, multiplicada por la dispersión de la predicción en la sesión. Las claves salen de la vista de cada origen (9f5efd92). La criba lee la raíz con `--reference <raíz> --decision-split calibration` (b340d59f).

Evaluación: 251 sesiones; calibración: 61. Contrastes pareados por sesión con t HAC. lightgbm_technical de dgx-us-v4-dbae075f.

| Modelo | Brazo | Elección | IC cal. | IC eval. | − padre, eval. | − lightgbm, eval. | − lightgbm, cal. |
|---|---|---|---|---|---|---|---|
| titans_mac | padre | — | +0,1231 | +0,1132 | — | +0,0009 (t +0,5) | +0,0030 (t +0,9) |
| titans_mac | combined | h 21, κa 0.05, κv 0.1 | +0,1253 | +0,1154 | +0,0023 (t +2,8) | +0,0032 (t +1,6) | +0,0052 (t +1,7) |
| titans_mac | asset_lab | h 21, κa 0.1 | +0,1238 | +0,1146 | +0,0014 (t +2,3) | +0,0024 (t +1,2) | +0,0037 (t +1,2) |
| titans_mac_frozen | padre | — | +0,1161 | +0,1117 | — | -0,0005 (t -0,2) | -0,0040 (t -1,8) |
| titans_mac_frozen | combined | h 21, κa 0.1, κv 0.1 | +0,1197 | +0,1149 | +0,0031 (t +2,5) | +0,0026 (t +1,5) | -0,0004 (t -0,2) |
| titans_mac_frozen | asset_lab | h 21, κa 0.1 | +0,1174 | +0,1137 | +0,0019 (t +2,9) | +0,0014 (t +0,8) | -0,0027 (t -1,4) |
| titans_mac_disabled | padre | — | +0,1123 | +0,1100 | — | -0,0023 (t -1,0) | -0,0078 (t -3,6) |
| titans_mac_disabled | combined | h 21, κa 0.1, κv 0.2 | +0,1176 | +0,1128 | +0,0029 (t +1,6) | +0,0006 (t +0,3) | -0,0025 (t -0,7) |
| titans_mac_disabled | asset_lab | h 21, κa 0.1 | +0,1140 | +0,1118 | +0,0019 (t +2,8) | -0,0004 (t -0,2) | -0,0061 (t -2,7) |
| cross_asset_attention | padre | — | +0,1229 | +0,1152 | — | +0,0030 (t +1,6) | +0,0027 (t +1,3) |
| cross_asset_attention | combined | h 21, κa 0.05, κv 0.1 | +0,1256 | +0,1175 | +0,0023 (t +2,5) | +0,0053 (t +3,0) | +0,0055 (t +2,2) |
| cross_asset_attention | asset_lab | h 21, κa 0.1 | +0,1235 | +0,1169 | +0,0016 (t +2,4) | +0,0046 (t +2,6) | +0,0034 (t +1,5) |

Lectura:
- **Frente al padre.** Los 8 brazos lo mejoran en evaluación: de +0,0014 a +0,0031 (t 1,6–2,9). Con Holm al 5 % pasan 3 de 8. Los contrastes están muy correlacionados, así que Holm es conservador.
- **Frente a lightgbm_technical.** Solo cross_asset_attention lo supera con t ≥ 2: combined +0,0053 (t 3,0) y asset_lab +0,0046 (t 2,6). El núcleo titans_mac con combined queda en +0,0032 (t 1,6).
- **Calibración de combined.** Es dentro de muestra: h y las κ se eligen ahí entre 80 combinaciones. El sesgo de selección esperado ronda 2,4 errores estándar (cota de Bailey y López de Prado), así que solo cuenta la evaluación.
- **Placebos** (no se emiten). Placebo − padre es negativo: de −0,0008 a −0,0044 (t −3,8 a −8,9). Por eso la lectura es brazo − padre y no brazo − placebo.
- **Comprobaciones.**
  - El IC de la emisión fila a fila es el del estudio (|Δ| máx. 0).
  - Filas, muestras y objetivos de cada tabla son los del padre.
  - Cambiar las etiquetas desde la sesión 20 no cambia ninguna señal hasta su cierre (prueba de causalidad).
  - Las claves se escalan solo con las filas de ajuste.
- **Coste.** 32 min 14 s, con la GPU y pruebas en paralelo. La pasada por vecinos son 1.583 s y el RSS máximo 2,6 GiB. La pasada por vecinos bajó de 691 a 242 s, medido en validación con 3 modelos.
- **Relevo.** Lo lanza la línea `orden|…` de `memory_organs linea`. La declaración v2 (55cfc250) añade el trío sin ganancia en la misma pasada.

### 5.5 Órganos como componente del agregador causal (medido en 2013; S2 confirmatorio)

El agregador en línea del tercer enjambre (`evaluation/online_aggregator.py`) se adoptó con sus 22 pruebas y 3 nuevas.
- Las pruebas nuevas matan mutantes que sobrevivían. Hay además una prueba de la rejilla operable declarada.
- No se adoptan la abstención ni los pesos por régimen.
- La barrera causal usa la madurez exacta de la vista.
- Los hiperparámetros, el mejor componente y los factores de lo operable se eligen solo en calibración.

**Exploratorio** (v10, 2013, semilla 42, evaluación). El listón del máximo de 24 ensayos es t ≈ 1,98.
- **Etiqueta.** ridge_cov_active con {titans_mac_nogain, cross_asset_attention, lightgbm_technical, señal del brazo combinado} da 0,1196.
  - Frente a la media de rangos de los tres modelos (0,1185): +0,0011 (t 2,1).
  - Frente al brazo combinado solo: +0,0014 (t 1,7).
  - Señal − placebo: +0,0012 (t 2,1).
  - La señal recibe un 10 % del peso en ridge y un 1 % en hedge.
- **Operable.** Frente a la media de rangos (0,0877): +0,0012 (t 1,2). Frente al brazo combinado solo: +0,0031 (t 3,7).

**S2 (registro confirmatorio, añadido de las 20:05Z).** En los orígenes limpios no hay atención entre activos, así que S2 es ridge_cov_active sobre {titans_mac_nogain, lightgbm_technical, señal}, frente a la media de rangos de los dos modelos (S1).
- La orden es `organ_aggregation agregar`, con la declaración `posttraining-aggregator-s2.json`.
  - La exige por su huella; con ella van la rejilla y los factores (0,001–100) y la semilla del placebo, 20261122.
  - Espera las entradas del relevo o sale con el código 75.
- Prueba con 2013 (desarrollo, no confirmatoria):

| Evaluación | Etiqueta | Operable |
|---|---|---|
| S1: media de rangos − el mejor por decisión (titans_mac_nogain) | +0,0017 (t 2,0) | +0,0056 (t 5,3) |
| S2 − S1 | +0,0011 (t 1,7) | +0,0004 (t 0,3) |
| S2 − su placebo | +0,0010 (t 1,6) | +0,0010 (t 1,3) |

### 5.6 Memoria por activo con el residuo operable; rejilla ancha (medido en 2013)

- **Rejilla ancha.** Con factores de 0,001 a 100, la tabla operable de 2013 sale idéntica byte a byte a la de 0,01–3. Ningún factor cae en un borde.
- **Memoria con el residuo operable (O) frente al de la etiqueta (L).** Con los parámetros del laboratorio (h 21, κ 0,1), en los 7 modelos:
  - en lo operable, O gana de +0,0011 a +0,0014 (t de 4,3 a 5,3; pasa Holm en los 7);
  - en la etiqueta, O pierde de −0,0006 a −0,0009 (t de −1,8 a −2,8).
- **titans_mac_nogain.** En lo operable, O da 0,0864 y L 0,0852.
  - Con h y κ elegidos en calibración (5; 0,05) baja a 0,0845, +0,0008 sobre lightgbm_technical (t 0,4).
  - En calibración, sus variantes O quedan fuera de la frontera de Pareto de los dos IC.

### 5.7 D1, la memoria supervisada del tercer enjambre (medido en 2013)

- **Comparación (P1).** D1 con r = 0 (semivida 28, de calendario 84) frente a la memoria por activo (h 21), en los tres modelos.
- **Diferencia D1 − memoria por activo.** Como mucho +0,0001 (t ≤ 1,5), con κ fijo o elegido en calibración.
- **Lecturas.** Las dos correlacionan entre 0,982 y 0,983. Las dos superan a sus placebos (etiquetas barajadas y mapa de lectura) por +0,0022 a +0,0027 (t 3,2–4,2).
- **Ablación L = 2** (esperar una sesión más): +0,0001 (t ≤ 1,1).
- **Veredicto.** Es la misma media exponencial del residuo maduro con otro nombre.
  - Solo cambia que no normaliza por el peso, que decae por calendario y que recorta en 3.
  - No se adopta. P2 (r = 4) queda aparcado por decisión del coordinador.

### 5.8 Núcleo agéntico sobre las emisiones (A1–A4; protocolo fijado, sin medir)

Son medidas descriptivas y secundarias sobre las emisiones ya hechas, sin reentrenar nada. No
entran en la familia primaria.
- La declaración es `configs/experiments/posttraining-agent-core.json` (huella 7019689c…).
- La orden es `python -m mars_titan.evaluation.agent_core medir`, con su línea en
  `nucleo/linea-a.txt`.
- Se fijaron en git (a6242492) antes de medir ningún origen.

- **A1, abstención selectiva de S1 y S2.**
  - Por activo, se abstiene donde más discrepan los rangos tipificados de la red y de los
    árboles. Por sesión, donde menos se parecen sus órdenes.
  - La cobertura se elige en calibración entre {0,5; …; 1}, por el IC medio con los abstenidos a
    cero. La regla está escrita antes de mirar la evaluación.
  - En evaluación se informan:
    - el IC del conjunto cubierto;
    - el IC con lo abstenido a cero, frente a dos controles de igual cobertura en cada sesión:
      menor |z| y azar (20 sorteos);
    - la curva de riesgo y cobertura;
    - los tercios de liquidez.
- **A2, intervalos conformes al 80 % y al 95 %, en la etiqueta.** Se comparan cuatro sistemas:
  - los cuantiles del núcleo tal cual;
  - esos cuantiles conformalizados en calibración (CQR);
  - la regresión cuantílica del sistema combinado (media de rangos y anchura del núcleo),
    ajustada en validación;
  - esa misma regresión conformalizada en calibración.

  Se mide la cobertura y la anchura en evaluación: en total, por tercio de liquidez, por decil de
  volatilidad de 21 sesiones y por trimestre.
- **A3, Shapley exacto del IC medio de evaluación.**
  - Jugadores: padre, árboles, señal del órgano, placebo (la señal barajada dentro de cada sesión)
    y, si existe, la atención. En 2013 existe y tiene las mismas filas.
  - Hay dos juegos, la media de rangos y ridge_cov_active, y cada sesión da su t HAC.
  - En la media de rangos el placebo no es un jugador nulo: con su mismo peso diluye. El cero solo
    se exige en ridge.
- **A4, lo operable con costes.**
  - Carteras de colas al 10 %, de S1, de S2 y de cada modelo solo, sin optimizar nada contra la
    evaluación.
  - Se informan:
    - la rotación diaria;
    - el coste de equilibrio por unidad de rotación y el margen frente a 5 pb;
    - el Sharpe neto a 5, 10 y 15 pb;
    - Sortino, caída máxima y CVaR al 5 %;
    - el DSR con 4 ensayos correlacionados, una cota indulgente;
    - la monotonía por deciles.
  - Sin AUM no hay impacto de mercado.
- **Covariables.**
  - Alfa, liquidez y volatilidad son las de la criba del día de la decisión.
  - Un oráculo compara el alfa con `KnownOffset`.
  - La madurez es la de la vista del padre.

**Resultado de 2013** (medido; descriptivo y secundario).
- Fuente: v10, semilla 42, evaluación de 251 sesiones, t HAC por sesión; `nucleo/2013`, copia
  e8910918, 1.264 s de CPU.
- Órgano: el de `organos/v10-nogain` (declaración nogain-v1), como en la prueba de S2 de 2013, porque
  `organos/v10-v2` de 2013 no existía. Los orígenes del plan leen `organos/v10-v2`, cuya lectura por
  vecinos es conjunta con más modelos. Por eso 2013 queda como desarrollo y no entra en la serie por
  origen.
- A1 se cierra como negativo (decisión del coordinador del 23-11) y no se encadena como contraste.
  En los orígenes sin atención, A3 tiene cuatro componentes y no se compara con el de 2013.
- Oráculo: los IC de S1 y S2 reproducen los de la orden de S2, 0,1180 / 0,0886 y 0,1191 / 0,0890.

- **A1, abstención.**
  - La regla de calibración elige cobertura 1, es decir, no abstenerse, en los ocho casos: dos
    sistemas × dos modos × por activo o por sesión. El Δ de evaluación es 0 por construcción.
  - Curva en evaluación (sin t):
    - Al 50 % de cobertura, el IC del conjunto cubierto de S1 sube de 0,1180 a 0,1645 en la
      etiqueta, pero quitar las de menor |z| da 0,1628 (+0,0017). En lo operable, 0,1173 frente a
      0,1180 (−0,0007).
    - La discrepancia entre la red y los árboles apenas mejora a la magnitud. Poner a cero lo
      abstenido siempre baja el IC: 0,1066 al 50 %.
  - El IC se concentra en el tercio menos líquido: 0,1675, 0,1005 y 0,0757 en la etiqueta.
- **A2, intervalos.**
  - Al 80 %:
    - los cuantiles del núcleo cubren 0,889, con anchura media 0,0552;
    - con CQR, 0,831 (0,0482);
    - la regresión cuantílica del sistema combinado con CQR, 0,828 (0,0374): un 32 % más
      estrecha que el núcleo crudo y un 22 % más que su CQR.
  - Al 95 %: 0,966, 0,961 y 0,961.
  - Todos cubren al menos lo nominal en total, por tercio de liquidez y por trimestre (el peor
    trimestre, 0,809 al 80 %).
  - El decil de volatilidad extremo no: el combinado con CQR da 0,687 al 80 % y 0,896 al 95 %;
    el núcleo con CQR, 0,772 y 0,917.
- **A3, Shapley del IC medio.**
  - Etiqueta, ridge (conjunto 0,1196):
    - padre +0,0391 (t 20,9), árboles +0,0370 (t 19,6), atención +0,0383 (t 19,5);
    - señal del órgano +0,0049 (t 3,0);
    - placebo +0,0003 (t 1,1), compatible con cero.
  - En la media de rangos el placebo diluye, −0,0108 (t −13,7), como se declaró.
  - En lo operable el placebo no sale cero, +0,0090 (t 7,1).
    - Su factor elegido es k = 0,005, así que su predicción operable es casi el alfa conocido, que
      informa de lo operable.
    - La señal del órgano, +0,0093, no se distingue de él.
- **A4, colas al 10 % en lo operable.**
  - S1:
    - +58,8 pb por sesión en bruto, con rotación de 1,57 (sobre 4);
    - coste de equilibrio de 37,4 pb por unidad de rotación, 7,5 veces el de 5 pb;
    - Sharpe anual bruto 18,5, y neto 16,0, 13,6 y 11,1 a 5, 10 y 15 pb.
  - Diferencias diarias en bruto: S1 − padre −2,8 pb (t −2,4), S1 − árboles +5,1 pb (t 3,7) y
    S2 − S1 −2,8 pb (t −1,8). En las colas, el padre solo gana a la mezcla.
  - Los Sharpe son de residuos de una sesión en un universo de supervivientes (D-2), sin impacto
    ni AUM. No son Sharpe operables.

### 5.9 Post-entrenamientos de los ajustes finales (diseño, sin GPU)

Están en `docs/research/posttraining-final-fits-design.md`:
- C1, destilación del conjunto hacia el núcleo con un profesor fuera de ajuste y cronológico;
- C2, ajuste fino de recencia con 63 y 252 sesiones frente a no tocar;
- C3, cabeza de calibración posterior frente a CQR. C3 ya lo mide A2 en CPU.

### 5.10 D, las cribas de la v11: regla fijada antes de sus emisiones

Las dos cribas de la v11 (`todos-7823a21b` y `todos-b443f803`, 2013, semilla 42) aún no tienen
emisiones. La regla queda escrita antes:

- **Órganos.** Los de la línea de la v11 del plan v36 (`posttraining-memory-organs-v11.json`, en
  su raíz propia), sin cambios.
- **Mejor modelo solo de cada criba.** Es el de mayor IC medio por sesión en **calibración**, en la
  etiqueta (con min_assets 30), entre los modelos de la criba con sus tres tramos completos.
  - Si dos quedan a menos de 0,0005, se toma el que va antes en la lista de la declaración de
    órganos de la v11.
  - La evaluación no interviene.
- **S1 y S2.** S1 es la media de rangos de ese modelo y lightgbm_technical. S2 es ridge_cov_active
  con además la señal de su órgano combinado. La declaración es la de S2 cambiando solo el padre
  y el órgano: las rejillas, la semilla del placebo y los modos quedan iguales.
- **Lectura.** Es exploratoria y va fuera de la familia primaria. Se informan:
  - S1 menos el mejor modelo, S2 menos S1 y S2 menos el placebo, pareadas por sesión con t HAC;
  - el mismo contraste frente al S1 y el S2 de la v10 en 2013.

### 5.11 F, fiabilidad de las emisiones (F1, F2) y F6, el cerebro holográfico: protocolo fijado

Encargo del coordinador del 23 de noviembre de 2026 (línea F): más del 90 % de precisión y
fiabilidad, sin trampas ni falsear nada. Las medidas son descriptivas y secundarias, fuera de la
familia primaria.
- La declaración es `configs/experiments/posttraining-reliability.json` (huella cdb1c511…).
- La orden es `python -m mars_titan.evaluation.reliability medir`, con las entradas, la espera
  (código 75) y la publicación atómica de S2 y de A. Cada orden mide un mercado.
- Se fijó en git antes de medir ningún origen.
- 2013 es desarrollo y solo sirve para elegir. Los orígenes limpios no se miran hasta que el relevo
  los entrene, y 2024 sigue sellado.

**Expectativa previa** (R-2 del cuaderno): la razón de información diaria en lo operable es 0,65
(0,66 en el tercio líquido).
- Con ella, la fracción esperada de sesiones positivas es Φ(0,65) ≈ 0,74, y la de ventanas de 5
  sesiones Φ(0,65·√5) ≈ 0,93.
- Un 90 % diario pide 1,28.

**Predictores.** S1, S2, los árboles y el padre, construidos como en S2 (semilla 42, alineación
estricta y factor operable k elegido en calibración).
- En lo operable va además un control: el alfa conocido solo como puntuación, que ya ordena y + α
  por sí mismo. En R-3r da un IC de 0,047 con todos los nombres y de 0,037 en el tercio líquido.
- Lo que aporte cada predictor se mide como mejora sobre él: la diferencia pareada del IC por
  sesión, con su t de Newey-West y su intervalo al 95 %, y la fracción de sesiones y de ventanas
  con mejora > 0.
- El tercio líquido se toma ordenando la liquidez dentro de cada sesión, sin filtrar por su signo
  ni tomar logaritmos (el tropiezo corregido en R-2).

**Regla del artefacto** (fijada antes de medir): el tope es el cuantil 0,99 de |α| conocido en
calibración.
- R-1 vio aciertos del 90–97 % en las mayores |predicción|. Eran cortos con alfas patológicos.
- Las filas sobre el tope no cuentan en F2. En F1, la curva selectiva se da con ellas y sin ellas,
  junto con la parte de nombres elegidos que lo superan.

- **F1, fiabilidad,** en la etiqueta y en lo operable, con todas las filas y con el tercio líquido:
  - la fracción de sesiones con IC de rangos > 0 y con decil alto − decil bajo > 0, con intervalos
    de Wilson al 95 %;
  - la misma fracción en ventanas de 5 y de 21 sesiones puntuadas seguidas de cada mercado, sin
    solaparse; la última ventana incompleta se descarta;
  - la curva de precisión selectiva: acierto de signo y respecto a la mediana de los
    k = max(1, round(n·c/2)) nombres más altos y más bajos de cada sesión, con
    c ∈ {1; 0,5; 0,2; 0,1; 0,05; 0,02; 0,01};
  - los intervalos de A2 al 80 % y al 95 % en la etiqueta (núcleo tal cual, CQR y combinado):
    cobertura y anchura en total, por tercio de liquidez, por decil de volatilidad realizada de 21
    sesiones conocida antes de la sesión y por trimestre;
  - el acierto de signo donde el intervalo del padre excluye el cero.
- **F2, confianza direccional calibrada,** solo en lo operable y sin las filas sobre el tope:
  - P(y + α > 0) sale de una regresión isótona creciente sobre la puntuación operable, ajustada en
    la primera mitad, en el tiempo, de las sesiones de calibración.
  - El umbral de confianza max(P, 1 − P) se busca en {0,55; 0,6; …; 0,95; 0,975; 0,99} con
    control de riesgo (Learn-then-Test) en la segunda mitad de esas sesiones.
    - Un umbral es válido si dos pruebas de «precisión < 0,90» dan p ≤ 0,05/11: la binomial por
      filas y la normal por conglomerados de sesión.
    - La segunda impide que unas pocas sesiones enteras pasen por muchas filas independientes.
    - Se elige el umbral válido más bajo.
  - En evaluación se informan:
    - la cobertura y la precisión, de Wilson y por conglomerados de sesión;
    - el Brier frente al de la tasa base;
    - la tabla de calibración.
  - Si ningún umbral pasa, la cobertura es 0 y se dice tal cual.
- **Reglas de entrega,** leídas mecánicamente por la orden (`verdicts`), sin juicio después de
  medir:
  - **F1.** Una fiabilidad por sesión ≥ 0,90 está demostrada si llegan a 0,90 dos límites
    inferiores: el de Wilson y el del arranque estacionario por bloques (media de 5 sesiones, 2.000
    remuestras). El segundo hace falta porque Wilson supone sesiones independientes.
    - En ventanas basta el de Wilson.
    - Si solo llega el punto, la fiabilidad es compatible.
  - **Atribución en lo operable.** Se atribuye al modelo si su mejora sobre el alfa solo tiene
    media > 0 y t ≥ 2. Si no, es persistencia del alfa.
  - **F2** entra como capa de confianza de S2 (operable, todas las filas) si se cumplen todas:
    - hay un umbral válido en calibración;
    - en evaluación cubre ≥ 0,001 de las filas en ≥ 10 sesiones;
    - su precisión es ≥ 0,90, con el límite inferior por conglomerados de sesión ≥ 0,85;
    - cubre más que el alfa solo con su propio umbral.
- **Piezas añadidas tras la primera lectura de 2013 (desarrollo), fijadas antes de medirlas:**
  - **Confianza de signo que emite S2.** Es la P calibrada de F2 sobre lo operable de S2 con todas
    las filas, emitida por fila en calibración y en evaluación (`s2-confianza-<tramo>.parquet`).
    - La isótona lleva bloques de al menos el 1 % de las filas de ajuste. En la primera lectura,
      un bloque extremo de pocas filas daba P = 1 a 348 filas de evaluación, que acertaban 0,59.
    - Se informan su rango real, el ECE y el MCE (10 intervalos) en la mitad de ajuste, en la de
      riesgo y en evaluación, y la frase expresa de si hay o no casos al 90 %.
  - **Intervalos adaptativos** sobre los cuantiles del padre en la etiqueta, en este orden:
    1. CQR;
    2. CQR con ACI por sesión (γ = 0,05; cada mercado con su α; solo aprende de sesiones con
       etiquetas maduras);
    3. CQR por decil de volatilidad dentro de la sesión (Mondrian);
    4. Mondrian con ACI por decil.
  - **Regla de R-1 para los intervalos:** a menos de 0,02 de la nominal en total, y ningún decil
    de volatilidad ni mercado más de 0,05 por debajo.
    - Se entrega el primero que pasa; si ninguno pasa, se dice.
    - Los grupos de Mondrian son los mismos deciles que mira la regla. Pasarla en calibración sale
      por construcción; lo que se mide es si se sostiene en evaluación.
  - **Intervalo de la entrega fijado con 2013.** Con la segunda lectura de 2013 (desarrollo), y de
    acuerdo con el coordinador, se fija Mondrian con ACI como intervalo de la entrega en los
    orígenes limpios.
    - La regla de R-1 es su confirmación en cada origen; los demás candidatos se informan como
      comparación.
    - Se añade el nivel del 90 %, el que se leerá como «fiabilidad del 90 %». Se construye desde la
      banda de los cuantiles 0,10–0,90 con el mismo método y se mide igual.
- **Primera lectura de 2013** (desarrollo; declaración b3d399ef…, US, semilla 42):
  - **F1.** Sesiones con IC > 0 con todos los nombres:
    - S2 da 0,956 [0,923; 0,975] en la etiqueta y 0,944 [0,909; 0,966] en lo operable, con una
      mejora sobre el alfa solo de +0,0440 (t 10,2).
    - En ventanas de 5 sesiones, 1,000 [0,929; 1,000].
  - **Tercio líquido operable:** 0,741 [0,683; 0,791] a diario y 0,940 [0,838; 0,979] en
    ventanas de 5, con una mejora sobre el alfa de +0,0278 (t 4,3).
  - **Precisión selectiva,** lejos del 90 %: las colas del 1 % operables aciertan 0,588.
  - **F2:** ningún umbral pasa, y la cobertura con precisión ≥ 0,90 es cero. No entra como
    selector.
  - **Intervalos:**
    - el 80 % del núcleo cubre 0,889;
    - la CQR cubre 0,831, de 0,772 a 0,885 por decil de volatilidad;
    - al 95 %, 0,966 y 0,961.
- **Segunda lectura de 2013** (declaración 5fb27d72…):
  - **Confianza de signo de S2.** En evaluación su rango es [0,391; 0,570] y su confianza máxima
    0,609, así que no hay casos al 90 %. El ECE es 0,011 en la mitad de riesgo y 0,028 en
    evaluación.
  - **Intervalos al 80 %:**
    - la CQR (0,831) y Mondrian (0,830) no pasan por la cobertura total;
    - CQR+ACI pasa con 0,803, pero su peor decil es 0,762;
    - Mondrian+ACI pasa con 0,803 y deciles de 0,801 a 0,805.
  - **Intervalos al 95 %:** pasan los cuatro. Mondrian+ACI cubre 0,951, con deciles de 0,950 a
    0,952.
- **F6, el cerebro holográfico,** con la regla acordada con el coordinador antes de medir. Se mide
  tras F1 y F2, en 2013 y sin GPU.
  1. **H, el holográfico, por semilla.** Es ridge_cov_active de {padre estático
     titans_mac_disabled_nogain, señal de su brazo combinado}, sin árboles y con la configuración
     de S2. H3 es la media de rangos por sesión de las semillas 42, 43 y 44.
     - Antes de mezclar los brazos de la semilla 42 (`organos/v10-nogain`) con los de la 43 y la
       44 (`organos/v10-v2`) se comprueba que comparten declaración.
  2. **C3, el núcleo.** Es la media de rangos de titans_mac_nogain en esas tres semillas: la pareja
     exacta del estático, porque solo cambia la memoria en línea. titans_mac va como lectura
     secundaria.
  3. **El conjunto C3⊕H3.** Es la media de rangos con pesos 1/2 fijos; no se elige nada.
  4. **Criterio para que H entre como segundo cerebro.**
     - IC(C3⊕H3) − IC(C3) ≥ +0,0020 con t HAC ≥ 2 en la etiqueta, pareado por sesión en
       evaluación.
     - Además, ese contraste debe ser > 0 en lo operable.
     - Si no pasa, H se queda como órgano del agregador.
  5. **Control a igual número de redes: C3⊕E3,** con E3 el estático sin órganos. Si C3⊕H3 no
     supera a C3⊕E3 con t ≥ 2, la ganancia viene de la segunda red y no de los órganos, y se dice
     así.
  6. **«Se equivoca en sitios distintos».** Se informan la correlación de los IC por sesión de C3 y
     H3 y la correlación de rangos media entre sus predicciones dentro de cada sesión, sin umbral.
  7. **Dónde se decide.** 2013 decide. Los orígenes limpios confirman con la misma regla, sin volver
     a elegir.
  - **Orden y declaración.**
    - La declaración es `configs/experiments/posttraining-holographic.json` (huella 0316226e…).
    - La orden es `python -m mars_titan.evaluation.holographic medir`, con una raíz de órganos
      por semilla (`--organs SEMILLA=RAÍZ`).
    - Las raíces `organos/v10-nogain` y `organos/v10-v2` solo difieren en la descripción y en la
      lista de modelos. La orden lo comprueba antes de mezclarlas, y también que cada órgano es el
      del ajuste declarado del estático.
  - **Resultado de F6 en 2013** (desarrollo; declaración 0316226e…, US). H se queda como
    órgano del agregador.
    - La regla da IC(C3⊕H3) − IC(C3) = −0,0002 (t −0,24; [−0,0019; +0,0015]) en la etiqueta y
      +0,0009 (t 0,70) en lo operable, lejos de +0,0020 con t ≥ 2.
    - H3 solo queda por debajo del núcleo: −0,0069 (t −3,8).
    - El control C3⊕H3 − C3⊕E3 da +0,0018 (t 2,8): los órganos hacen del estático mejor compañero,
      pero no lo bastante para sumar al núcleo. H3 − E3 da +0,0019 (t 1,5).
    - Correlaciones: la de los IC por sesión de C3 y H3 es 0,94 (0,78 en lo operable); la de rangos
      dentro de la sesión, 0,85 (0,77).

### 5.12 P, la capa de cartera para las medidas netas: protocolo y resultado en 2013

Es el encargo del coordinador del 23 de noviembre: las medidas netas de una cartera sencilla sobre
la puntuación operable de S2.
- La declaración es `configs/experiments/posttraining-portfolio-layer.json` (huella 828ee84c…),
  fijada antes de escribir el código y de medir.
- Es descriptivo y secundario.
- Tiene la misma separación que F: 2013 es desarrollo, los orígenes limpios confirman y 2024 sigue
  sellado.

**Por qué** (R-3r del cuaderno):
- La señal persiste: guarda un 73 % al día siguiente y un 53 % a cinco sesiones.
- Los libros diarios de deciles de A4 rotan en torno a 1,5 veces al día.
- Los brutos de A4 (unos 56 pb diarios en S2) incluyen los nombres con |α| patológico. Esta capa
  los excluye.

**Universo y puntuación.**
- La puntuación es la operable de S2 (k·p + α), con las entradas y la configuración de S2 (semilla
  42). Se tipifica en rangos dentro de cada sesión.
- Se excluyen las filas con |α| sobre el tope de F (cuantil 0,99 de calibración).
- La lectura principal es el tercio más líquido de cada sesión, donde de 2 a 5 pb por lado son
  plausibles. Todos los nombres van como lectura secundaria.

**Libro.**
- Largo el decil alto y corto el bajo, con pesos iguales dentro de cada pata, neutral en dinero y
  con exposición bruta 2.
- El rendimiento de la sesión es el objetivo operable y + α de sus nombres.

**Suavizado y banda.**
- Media exponencial por activo del rango tipificado: s̃ₜ = λ·zₜ + (1 − λ)·s̃ₜ₋₁. Si el activo no
  estaba en la sesión anterior, s̃ₜ = zₜ. La rejilla es λ ∈ {1; 0,6; 0,4; 0,25; 0,15}.
- La banda de no operar es una histéresis de rangos: un nombre entra en el decil extremo y solo
  sale si abandona el extremo q_sal, con q_sal ∈ {0,1 (sin banda); 0,2; 0,3}.
- (λ, q_sal) se elige en calibración, por el resultado neto medio a 3 pb por lado, en cada alcance.
  La rejilla es pequeña porque calibración tiene unas 60 sesiones.

**Costes y rotación.**
- El coste de la sesión es c·Σ|wₜ − wₜ₋₁| (por lado), con c ∈ {2; 3; 5} pb.
- La rotación es Σ|Δw| por sesión.

**Control.**
- El mismo libro con la señal diaria sin suavizar ni banda (λ = 1, q_sal = 0,1) y los mismos costes.
- Como referencia va el libro del alfa solo.

**Medidas en evaluación.**
- Por coste se informan:
  - el neto medio en pb por sesión y la rotación;
  - el Sharpe anualizado (√252) y la caída máxima;
  - el coste de equilibrio.
- Por horizonte h ∈ {1; 5; 21} sesiones, en ventanas sin solapar: la fracción de ventanas con neto
  > 0 (Wilson y arranque estacionario por bloques, como en F) y su neto medio.
- La diferencia capa − control por sesión a cada coste, pareada y con t de Newey-West.

**Regla de entrega** (alcance líquido). La capa entra si, en evaluación, se cumplen las dos:
- su neto a 3 pb por lado supera al del control con t ≥ 2;
- su neto a 5 pb por lado es > 0 con t ≥ 2.

Si no, no entra, y las cifras se dan tal cual.

**Límites declarados:**
- La ejecución es al precio de la decisión, con coste fijo por lado y sin impacto de mercado.
- No se modela si los cortos están disponibles.
- Los costes son los mismos en todo el tercio líquido.

**Detalles fijados antes de medir** (en la declaración, con el código):
- En calibración, a igualdad gana lo más sencillo: λ mayor y, después, salida menor.
- Cada tramo empieza sin cartera y con el suavizado reiniciado.
- Una sesión con menos nombres que min_assets en el universo, o con una pata vacía, no tiene
  posiciones y paga el cierre.
- Los deciles salen del percentil de punto medio del rango suavizado.
- La orden es `python -m mars_titan.evaluation.portfolio_layer medir`, con las mismas entradas
  que S2.

**Resultado en 2013** (desarrollo; US; semilla 42; huella 828ee84c…; `cartera/2013`). La regla no
se cumple: la capa no entra.
- Tercio líquido (lectura principal):
  - En calibración ganó λ = 1 con salida 0,2, es decir, la banda sin suavizado. Ningún punto con
    suavizado superó a la señal diaria.
  - La capa rota 1,20 por sesión, frente a 1,82 del control, y su bruto baja de 23,6 a 21,1 pb.
  - Capa − control: −0,60 pb a 3 pb (t −0,70; intervalo [−2,30; 1,09]) y +0,65 pb a 5 pb
    (t 0,75).
  - Neto de la capa a 3 pb: 17,5 pb por sesión (t 5,3), con Sharpe 4,6 y caída máxima de 292 pb.
  - Sesiones con neto > 0: 0,633 [0,572; 0,691] (bloques [0,574; 0,693]).
  - Ventanas de 5 con neto > 0: 0,80 [0,67; 0,89]. Ventanas de 21: 0,91 [0,62; 0,98], con solo
    11 ventanas.
  - Frente al alfa solo: +4,1 pb a 3 pb (t 1,2).
- Todos los nombres (lectura secundaria):
  - En calibración ganó el propio control (λ = 1, salida 0,1), así que capa y control coinciden.
  - Neto a 3 pb: 48,8 pb (t 15,9). Sesiones con neto > 0: 0,892 [0,848; 0,925] (bloques
    [0,853; 0,932]). Todas las ventanas de 5 y de 21 dan neto > 0.
  - Frente al alfa solo: +18,6 pb (t 7,4).
  - No es una cifra operable: los nombres ilíquidos pagan el mismo coste por lado que los líquidos
    (límite declarado).
- Lectura: en 2013 la señal diaria ya es lo mejor de la rejilla. La banda cambia rotación por bruto
  sin ganar a 3 pb, y solo queda por delante a partir de unos 5 pb, sin significación.
- P mide un solo mercado por orden; con dos, falla con su motivo. Así el libro, la rotación y las
  ventanas no mezclan sesiones de dos mercados.

**Corrección por la auditoría** (23 de noviembre), en la precisión selectiva de F1:
- Con n impar y c = 1, round(n/2) podía meter la fila mediana en las dos colas.
- Ahora k = min(max(1, round(n·c/2)), ⌊n/2⌋), y esa fila no cuenta en ninguna.
- El efecto en la columna «Todo» de 2013 es como mucho una fila de unas 2.700 por sesión. No se
  rehace la medida; los orígenes limpios ya usan la corrección.

### 5.13 F7, las métricas de error: protocolo (fijado antes de medir)

Es el encargo del dueño, transmitido por el coordinador el 23 de noviembre: «mejorar las métricas de
error, es decir que sea el menor, de todas las métricas existentes calculadas siempre».
- La declaración es `configs/experiments/posttraining-error-metrics.json` (huella
  606db891…).
- La separación es la de F: 2013 es desarrollo, los orígenes limpios confirman y 2024 sigue sellado.

**Punto de partida, con su escala.**
- En 2013 el núcleo de la criba (titans_mac) da un MAE de 0,011089, frente a 0,011169 de predecir
  cero (−0,7 %). Con su factor LAD de calibración (0,76) baja a 0,011056 (−1,0 %).
- El padre de S2 (titans_mac_nogain, semilla 42) emite una amplitud unas 15 veces mayor de la que
  le corresponde:
  - MAE 0,027454 (+146 % sobre cero), con factor LAD 0,066;
  - cobertura del 80 % de 0,889 y ECE de 0,148.
  - Con su factor LAD baja a 0,011034 (−1,2 %).
- Con un IC de Pearson ρ, el mejor predictor lineal reduce el error cuadrático en ρ² y el RMSE en
  1 − √(1 − ρ²). Con ρ = 0,14, eso es un 2 % y un 1 %, y el MAE se mueve en el mismo orden.
- El informe da esa cota con el ρ medido y no promete reducciones que el ruido de la etiqueta no
  permite.

**Orden y nivel** (adenda del coordinador, antes de medir). El término de orden cuesta error de
nivel. En la criba de 2013, semilla 42:
- titans_mac: MAE 0,011089 con IC 0,1132;
- titans_mac_noic: MAE 0,010996 (−0,8 %) con IC 0,1077;
- titans_mac_nogain: el mejor IC (0,1163) y el peor MAE (0,027454).

La calibración es donde se pueden tener las dos cosas: el orden del modelo de orden y el nivel
recalibrado. Por eso el noic entra como referencia de nivel. Si la versión entregada del modelo de
orden lo iguala sin perder IC, esa es la entrega.

**Qué se mide.** En evaluación, con `forecast_scores.score_sessions` (las métricas que el
repositorio calcula siempre, cada sesión con el mismo peso):
- MAE y RMSE de la predicción puntual.
- Pinball de cada cuantil emitido (0,025; 0,1; 0,5; 0,9; 0,975) y su media.
- Error de cobertura |cobertura − nominal| de los intervalos centrales del 80 y del 95 %.
- ECE (10 tramos) de la probabilidad de subida implícita en los cuantiles, con la regla del
  repositorio.
- Fuera de la regla, como información: puntuación de intervalo, Brier de signo, frecuencia de cada
  nivel e IC de rangos.

**Predictores.** Todos en la etiqueta y en lo operable, sobre las filas de S2 de cada modo:
- el padre (titans_mac_nogain, semilla 42), con punto y cuantiles;
- las semillas: la media de 42, 43 y 44 del padre (`seed_ensemble.seed_mean`; los cuantiles, con
  la media de Vincent);
- S1 y S2, con sus puntuaciones de sesión;
- dos referencias, si están emitidas en el origen (si no, el informe dice que faltan):
  - el núcleo de la criba (titans_mac, semilla 42), por la cifra de partida;
  - el modelo entrenado solo con pinball (titans_mac_noic, semilla 42), que da el mejor nivel con
    estos datos.

En lo operable:
- La puntuación es k·p + α, con el k del criterio del agregador en calibración. Para el padre es el
  de S2; para las semillas y el núcleo se calcula con la misma función.
- Los cuantiles son α + q.

S1 y S2 emiten combinaciones de puntuaciones z por sesión, sin unidades de rendimiento. Su versión
sin calibrar no tiene MAE ni cuantiles, así que solo se miden calibradas y frente a los controles.

**Controles:**
- Cero: predicción puntual 0. Sus cuantiles son la climatología de calibración: los cuantiles del
  objetivo por mercado, con cada sesión con el mismo peso.
- En lo operable, además, alfa solo: predicción α, con α + la climatología de la etiqueta como
  cuantiles.

Cómo se compara con un control:
- En las pérdidas (MAE, RMSE, pinball) se da la mejora relativa 1 − M/M_control.
- En los errores de calibración (cobertura, ECE) se da la diferencia, porque la climatología está
  calibrada en promedio por construcción.

**Calibración.** Se ajusta en calibración, solo con las etiquetas maduras antes de la primera sesión
de evaluación, y se aplica sin cambios. No toca el orden dentro de la sesión.
- Punto: λ·s, con λ el factor LAD del repositorio (`scale_diagnostics.market_shrinkage`: la mediana
  ponderada de y/s por mercado, con cada sesión con el mismo peso, en [0, 1]). Un λ = 0 no se
  entrega, porque rompería el orden.
- Cuantiles:
  - el centro pasa a λ·s;
  - las anchuras de la cabeza (q_τ − q_0,5) se multiplican por un factor por nivel s_τ ≥ 0, el
    minimizador exacto de la pinball de calibración (el cuantil ponderado del cociente entre residuo
    y anchura);
  - si dos niveles se cruzan, se ordenan por fila; ese reordenamiento nunca sube la pinball sumada;
  - S1 y S2 usan las anchuras del padre, y las semillas las suyas.
- Hay tres versiones: sin calibrar; «punto» (λ·s con los cuantiles emitidos); y «completa» (punto y
  cuantiles).

**Regla**, por predictor y modo, en evaluación:
- Una versión calibrada entra si no empeora ninguna métrica de la regla frente a la sin calibrar.
  Las de la regla son MAE, RMSE, pinball de cada nivel y su media, error de cobertura del 80 y del
  95 % y ECE. La comparación es estricta, sin tolerancia.
- Se entrega la más completa que pase: la «completa»; si no, la «punto»; si no, la sin calibrar.
- S1 y S2 no tienen versión sin calibrar. Se comparan con el control de su modo (cero en la
  etiqueta, alfa solo en lo operable), y si no pasa ninguna versión no se entregan en unidades de
  rendimiento.
- Cada diferencia de MAE, error cuadrático y pinball media lleva su t de Newey-West, pareada por
  sesión, para que se vea si es ruido.
- La versión entregada se elige con 2013 y queda fijada. En los orígenes limpios la regla solo
  confirma, y si no confirma se dice qué métrica empeora.
- Lectura de nivel: la versión entregada iguala al noic si su MAE y su pinball media no son
  mayores que los del noic tal como se emite. El IC de rangos no cambia con la calibración, por
  construcción (λ > 0), y se comprueba.
- No se usa la isótona de la mediana: sus bloques empatan el orden de la sesión y perderían IC.
- El agregador ya exige que toda etiqueta de calibración madure antes de la primera decisión
  evaluada. El filtro de madurez de F7 es una segunda guarda.
- La orden es `python -m mars_titan.evaluation.error_metrics medir`, con las entradas de S2 y las
  carpetas de las semillas y las referencias. `informe --summary` rehace el resumen sin medir.

**Resultado en 2013** (desarrollo; US; semilla 42; huella 42fae0ce…; `errores/2013`; pilotos-b
congelado y descongelado).

*Escala.*
- El IC de Pearson medio en la etiqueta va de 0,075 (S1) a 0,174 (noic). La cota del RMSE sobre
  predecir cero va, por tanto, del 0,3 al 1,5 %.
- Las rebajas medidas quedan dentro de la cota: el RMSE baja del 0,2 % (S2) al 1,1 % (noic) y el
  MAE del 0,5 al 1,8 %.

*Punto (escala LAD).* Entra para el padre, las semillas, el noic, S1 y S2 en los dos modos, y para
el núcleo en lo operable.
- Padre: λ = 0,066. Su MAE pasa de 0,027449 a 0,011034, es decir, de −145,7 % a +1,2 % sobre
  cero; el RMSE mejora un 0,8 %.
- S2: λ = 0,0014. Sobre cero, el MAE mejora un 0,5 % y el RMSE un 0,2 % en la etiqueta. En lo
  operable mejora un 0,2 % sobre cero y un 1,8 % sobre alfa solo.
- El núcleo (titans_mac) se queda sin calibrar en la etiqueta. Con λ = 0,76 el MAE baja
  (t −12,2), pero el error cuadrático sube 3,7·10⁻⁷ (t 1,7), y la regla es estricta.

*Cuantiles (versión completa).* Solo entran para el núcleo en lo operable: la pinball baja
4·10⁻⁵ (t −13,7) y el error de cobertura del 80 % pasa de 0,053 a 0,014.
- La cabeza del padre sin ganancia no sirve para recentrar. Sus anchuras van atadas a una mediana
  unas 15 veces más amplia, y recentradas suben el error de cobertura del 80 % de 0,089 a 0,161. A
  S1 y S2, que usan esas anchuras, les pasa lo mismo.
- El noic ya está calibrado: pinball 0,002702 (un 16,9 % mejor que la climatología) y ECE 0,0065.
  La recalibración empeora sus colas bajas y su ECE.

*Nivel del noic.* Las t de este punto son descriptivas, fuera del protocolo.
- En la etiqueta no lo iguala ningún modelo de orden calibrado: el padre queda un 0,3 % peor en
  MAE (t 5,2) y S2 un 1,0 % peor (t 9,6). Al tipificar por sesión, S2 pierde la amplitud
  transversal: su IC de Pearson es 0,078, frente a 0,158 del padre.
- En lo operable S2 sí lo iguala, con más IC de rangos (0,089 frente a 0,079):
  - MAE −4·10⁻⁶ (t −0,6);
  - error cuadrático −1,7·10⁻⁶ (t −2,9).
- El noic con su propia escala LAD sigue algo por delante: MAE 0,010965, frente a 0,010973 de S2.

*Lectura.* La pieza que entra es la escala LAD del punto. Los cuantiles de la entrega tienen que
salir de una cabeza bien escalada (el noic o el núcleo), no de la del padre sin ganancia.

**F7-b, la entrega con las anchuras del modelo solo pinball** (adenda fijada antes de medir de
nuevo 2013, con las tres condiciones del coordinador). Está en la sección `delivery` de la
declaración.

*La candidata:*
- el orden de S2;
- el nivel con la escala LAD de S2;
- las anchuras de la cabeza del noic (o del núcleo, si no hay noic), centradas en λ·S2 y con su
  factor de pinball por nivel;
- encima, el Mondrian+ACI de F, con la misma configuración que la de F.

*Lo que ya entra:* el punto λ·S2, los cuantiles de la cabeza del padre tal como se emiten y el
Mondrian+ACI de F sobre esa cabeza.

*Decisión en calibración:*
- Se parte en dos mitades en el tiempo: la primera ajusta λ, los factores y las puntuaciones
  conformes, y la segunda puntúa.
- La candidata se queda si no empeora la pinball media ni el error de cobertura al 80, 90 y 95 %.

*Regla en evaluación:*
- Todo se ajusta de nuevo con la calibración entera.
- La candidata entra si no empeora ninguna métrica de F7 frente a lo que ya entra: pinball de cada
  nivel y media, ECE en la etiqueta y en lo operable, y error de cobertura de los intervalos
  entregados al 80, 90 y 95 %. El MAE y el RMSE son iguales por construcción.

*La capa:*
- Sigue encima salvo que, sin ella, las bandas de la candidata no sean peores en error de
  cobertura al 80 y al 95 % en las dos lecturas.
- El 90 % la necesita siempre, porque la cabeza no tiene los niveles 0,05 y 0,95.

*Procedencia.* El informe dice de qué modelo sale cada pieza:
- orden: S2;
- nivel: la escala LAD;
- anchuras: el noic, o el núcleo de respaldo;
- cobertura: el Mondrian+ACI de F.

Esto obliga a que la base emita un modelo solo pinball por origen y semilla.

*Comprobación.* Lo que ya entra reproduce exactamente, en la prueba, las coberturas del
Mondrian+ACI de F.

**Resultado de F7-b en 2013** (huella 606db891…; `errores/2013`). La lectura sin F7-b queda
en `errores/2013-093a0caa`. **Entra.**
- En calibración (segunda mitad), la candidata gana en todo:
  - pinball media 0,002974, frente a 0,004796;
  - errores de cobertura de 0,0029, 0,0036 y 0,0017, frente a 0,0104, 0,0073 y 0,0049.
- En evaluación no empeora ninguna métrica:
  - pinball media 0,002723, frente a 0,004767 (−43 %), y mejor en cada nivel;
  - ECE 0,028, frente a 0,148 (en lo operable, 0,030 frente a 0,177);
  - cobertura de 0,7996, 0,9003 y 0,9499 (errores de 0,0004, 0,0003 y 0,0001), frente a 0,8034,
    0,9022 y 0,9515;
  - al 90 %, los deciles de volatilidad quedan entre 0,899 y 0,901.
- Con la misma cobertura, los intervalos son más estrechos: al 90 % la anchura media es 0,0466,
  frente a 0,0607 (−23 %), y la puntuación de intervalo 0,0746, frente a 0,0872 (−14 %).
- La capa hace falta: sin ella, el error de cobertura sube a 0,0155 al 80 % y a 0,0051 al 95 %.
- La candidata no llega a la cabeza del propio noic, porque su centro es λ·S2: pinball 0,002723
  frente a 0,002702, y ECE 0,028 frente a 0,0065.

### 5.14 WP5, los órganos episódicos nuevos del holográfico: revisión y protocolo (fijado antes de medir)

**Encargo.** El coordinador pide verificar e integrar el parche WP5 de Gemini (neon xhigh), que
trae tres órganos nuevos para el cerebro holográfico, apagados por defecto y seleccionables con el
campo `episodic_organs` de la declaración de órganos. Es código ajeno sin revisar: si algo no pasa
la revisión, no se integra.
- **Banco denso:** vecinos por coseno sobre las filas maduras de las últimas sesiones, pesos
  softmax(coseno/τ), media y dispersión ponderadas del residuo y − p recortado.
- **Memoria holográfica con regla delta:** claves activo ⊛ régimen observable por convolución
  circular, escritura del residuo que la memoria aún no predice y lectura por correlación circular.
- **Traza por activo:** media exponencial del residuo maduro, con semivida en sesiones maduras.

**Revisión (antes de integrar).**
- **Aplicación.** El parche, hecho sobre bad39c05, se aplica limpio sobre 0fe44aa1, primero en una
  copia aparte. Toca `memory_organs.py` (+188) y añade `memory/episodic_organs.py` y dos ficheros
  de pruebas.
- **Causalidad.** Es la de las memorias existentes:
  - antes de leer la cohorte (mercado, t), solo maduran cohortes de ese mercado con madurez ≤ t;
  - la madurez de una cohorte es la máxima de sus filas;
  - toda etiqueta madura después de su decisión, así que nunca se lee la misma sesión;
  - nada madura en 2024.
- **Régimen y selección.** El régimen de la holográfica sale de las medianas, dentro de la sesión,
  de return_63 y volatility_21_over_252 de las claves, escaladas con las filas de ajuste. τ, la
  semivida y κ se eligen solo por el IC medio de calibración.
- **Placebos:**
  - el banco, con sus valores permutados en cada cohorte;
  - la holográfica, con una memoria paralela que escribe los residuos permutados entre pares;
  - la traza, con sus lecturas permutadas entre los activos de la sesión.
- **Pruebas.** Pasan las 63 del parche y las de los módulos relacionados (278 en total).
- **Mutantes míos sobre lo nuevo: once, todos muertos.**
  - Madurez adelantada en los tres órganos.
  - Placebo igual al real: en el banco y en la holográfica, al escribir y al leer.
  - Elección en evaluación, en el banco y en la traza.
  - Régimen con la mediana de todas las filas, futuro incluido.
  - Tipificación con la escala de toda la serie.
  - Lectura desalineada una fila al emitir.
- **Identidad con el campo ausente.** Se emitió el estado sintético de las pruebas con el código
  anterior y con el nuevo: los 25 archivos son iguales (16 byte a byte; los 9 JSON, iguales salvo
  la huella `code.memory_organs` y los tiempos). Con el campo presente, los brazos existentes salen
  byte a byte iguales y la tabla del estudio conserva sus filas.
- **Integración:** 70277226 (el módulo) y 15a11b44 (la emisión). Las declaraciones, las variantes
  de F6 y `episodic_contrasts` van en fa49edf2.

**Protocolo.**
1. **Emisión.**
   - Declaración `posttraining-memory-organs-wp5-v10.json` (6c862b76…): la de v2 para
     titans_mac_nogain y su estático titans_mac_disabled_nogain, con la configuración de los tres
     órganos que propuso el parche, sin tocarla.
   - Origen 2013, US, semillas 42, 43 y 44, raíz `organos/wp5-v10`.
   - Una unidad por semilla, con pilotos-b congelado.
2. **Identidad en datos reales.** Los brazos existentes de los dos modelos (parquet de calibración
   y de evaluación, y el estado por activo) deben salir byte a byte iguales a los publicados: la
   semilla 42 en `organos/v10-nogain` y la 43 y la 44 en `organos/v10-v2`. Si alguno difiere, no
   se lee nada más y se dice.
   - Este paso recorre, con el campo presente, el mismo código de los brazos existentes. Con el
     campo ausente solo cambian ramas que no se ejecutan, y eso ya está comprobado en el estado
     sintético.
3. **Cada órgano frente a su padre y a su placebo** (`episodic_contrasts`; descriptivo).
   - En evaluación se miden órgano − padre, placebo − padre y órgano − placebo, por semilla y en la
     media de semillas, con la t de Newey-West de la criba.
   - El placebo falla como debe si no gana al padre con t ≥ 2. Si le gana, es una alarma: el
     órgano no se lee hasta saber por qué.
   - Hay señal propia si el órgano gana al padre y a su placebo, las dos veces con t ≥ 2.
   - El padre de la regla es el estático. titans_mac_nogain va como segundo padre, para leer el
     órgano sobre el núcleo.
4. **La regla de F6 sin cambiarla, una vez por órgano.**
   - Declaraciones `posttraining-holographic-wp5-<brazo>.json` (9ea3c63c… del banco, 8272fe12… de la
     holográfica y 1053e40e… de la traza). Solo cambian el órgano de F6 por
     titans_mac_disabled_nogain__<brazo>.
   - H entra como segundo cerebro si IC(C3⊕H3) − IC(C3) ≥ +0,0020 con t ≥ 2 en la etiqueta y es
     > 0 en lo operable. El control a igual número de redes es C3⊕E3.
   - La vara del coordinador es la correlación de los IC por sesión de C3 y H3 en la etiqueta: por
     debajo de 0,94, la señal es distinta. Con el órgano combinado de F6 fue 0,936.
   - Se miden tres variantes a la vez. Si entra alguna, se dice también si pasaría con la
     corrección de Bonferroni para tres (t ≥ 2,39). La confirmación sigue siendo de los orígenes
     limpios, con la misma regla.
5. **Lo de siempre.** 2013 es desarrollo y solo elige, los orígenes limpios no se miran y 2024
   sigue sellado.

**Coste.** Se midió con datos sintéticos del tamaño real (500 activos, 64 dimensiones, un hilo):
- el banco denso tarda 0,31 s por cohorte con el banco lleno;
- la holográfica, 0,12 s por sesión.

Lo caro de cada semilla es rehacer los brazos existentes, de 30 a 80 min según la carga.

## 6. Etapas con GPU (fase 4)

### 6.1 Qué queda preparado

- **Continuación desde los padres de la v10.** `configs/experiments/dgx-us-campaign-v10-post1.json`
  (fuente: la declaración v10 de 3d10a30f, huella 836c79a6…; cohorte de 4.198 activos, vista
  completa; parada de la v10: meseta del IC de validación con paciencia 3 y tope de 48 épocas; test
  final cerrado). Vuelve a ajustar `titans_mac` de la v10 en 2012 con su misma receta
  (`configs/titans/dgx-us-core-u2-ic3-dense-gr-v9.json`, huella 3b88760f…, con `accumulation_rows`
  de 8.192) y continúa desde su estado elegido en 2013 con cuatro brazos que solo cambian el
  ajuste: `titans_mac_cont` (la misma receta), `titans_mac_cont_wd` (decaimiento 0,1 hacia cero),
  `titans_mac_cont_l2sp` (decaimiento 0,1 hacia el padre) y `titans_mac_cont_sp` (contracción 0,6
  con ruido 0,01). Se vuelve a ajustar el padre en este estado porque la v10 no tiene 2012 y porque
  `inherit` trasladaría la campaña entera y cerraría su permiso.
- **Adaptadores sobre los padres de la v10.** Los únicos que la campaña DGX sabe ejecutar son los
  lectores (`reader`: m0, m3, cognitivo), ya declarados en la v10. Los adaptadores del paquete
  `posttraining/` no están conectados a la cola DGX; no se declaran aquí (sección 3).
- **Comprobado en CPU sin GPU.** El plan da 18 trabajos (6 del padre en 2012 y 2013 y 12
  continuaciones en 2013, cada una tras `titans_mac/2012/seed-s`); las continuaciones quedan
  bloqueadas hasta que termina su padre; la ejecución de un trabajo de continuación carga el estado
  elegido del padre (prueba de `execute_job` con datos sintéticos); y con pasos reales del
  optimizador el decaimiento anclado lleva los pesos hacia ese estado.

### 6.2 Órdenes

Desde una copia fija (`git worktree`) en el commit que integre esta rama, con el lanzador de la
campaña (`scripts/dgx_campaign.sh`, que fija cuBLAS determinista y antepone el `src` de la copia):

```bash
copia=/home/gpic/TITANS-IA/MARS-TITAN-campana-<commit>
estado=/home/gpic/.local/state/mars-titan/dgx-titans-us-v31/v10-post1/semillas-42,43,44/todos-<commit>
vistas=/home/gonzalo/.local/state/mars-titan/campaign-a-v2-views-v31-20261013/views/US
decl=configs/experiments/dgx-us-campaign-v10-post1.json
cd "$copia"
./scripts/dgx_campaign.sh plan --machine mars_titan --declaration "$decl"          # 18 trabajos
./scripts/dgx_campaign.sh prepare --declaration "$decl" --state "$estado" --views "$vistas"
./scripts/dgx_campaign.sh doctor --machine mars_titan --declaration "$decl" --state "$estado" --views "$vistas"
./scripts/dgx_campaign.sh authorize --declaration "$decl" --state "$estado" --views "$vistas" \
    --note "Post-entrenamiento v10: continuaciones desde titans_mac/2012"
```

En el plan del relevo, con su formato (`prepara|…` construye antes los índices y las copias
empaquetadas de un origen con `scripts/dgx_prepare_inputs.py`, como en la v10):

```text
prepara|<copia>|<estado>|configs/titans/dgx-us-core-v2.json|2012
<copia>|<estado>|<vistas>|configs/experiments/dgx-us-campaign-v10-post1.json|1 --seed 42 --origin 2012 --tier core
prepara|<copia>|<estado>|configs/titans/dgx-us-core-v2.json|2013
<copia>|<estado>|<vistas>|configs/experiments/dgx-us-campaign-v10-post1.json|4 --seed 42 --tier post
```

La segunda fase no filtra por origen a propósito: el filtro de orígenes exige que el padre de cada
trabajo esté en el bloque («El bloque de orígenes deja fuera al padre…»), y con `--tier post` las
continuaciones solo esperan a que `titans_mac/2012/seed-42` esté terminado. `titans_mac/2013` de
este estado no hace falta para los contrastes (se compara con el de la v10); ejecutarlo
(`--seed 42 --origin 2013 --tier core`) solo sirve para comprobar que el padre vuelve a dar las
mismas emisiones que en la v10.

### 6.3 Coste esperado (estimación, no medición de estas recetas)

Medido en la v8 sobre la GB10 (origen 2013, semilla 42): `titans_mac` 2,06 h de reloj en dos
intentos (0,65 + 1,41 h), RSS de unos 20 GiB, memoria CUDA asignada máxima 2,4 GiB y reservada
hasta 7,4 GiB; `titans_mac_frozen` 2,50 h. Con la parada por meseta de la v10, un ajuste de 2012
(menos datos de ajuste que 2013) debería costar del orden de 1,5 a 2 h y una continuación entre 1
y 2 h: si ninguna época mejora la validación se detiene a las tres épocas (hipótesis). La semilla
42 son 5 trabajos, de 5 a 10 horas de GPU con un trabajador; las otras dos semillas, 10 trabajos
más.

### 6.4 Orden recomendado y regla de decisión (declarada antes de ejecutar)

1. Semilla 42: el padre en 2012 y las cuatro continuaciones en 2013.
2. Contrastes pareados por sesión con t HAC frente a `titans_mac/2013/seed-42` de la v10 (desde
   cero), con `scripts/dgx_screening_report.py --compare <estado de la v10>` y un archivo de
   controles que empareje cada brazo con `titans_mac`; además `titans_mac_cont_l2sp` frente a
   `titans_mac_cont_wd` (una sola variable: el destino del decaimiento).
3. Un brazo pasa a las semillas 43 y 44 si en la 42 su diferencia de decisión (validación) no es
   negativa y la de evaluación es positiva con t ≥ 2. Se adopta solo si las tres semillas juntas
   mantienen t ≥ 2. Si ninguno pasa, se registra como negativo y la continuación no entra en la
   campaña final.
4. Riesgos: que el padre vuelto a ajustar no sea idéntico al de la v10 (se comprueba con el paso
   opcional de 6.2); que la parada elija el estado inicial del padre en 2013 (el brazo sin cambios
   lo mide y el informe de cada trabajo registra la época elegida); un solo origen de desarrollo.

### 6.5 Gemelos de escritura: la misma red, evaluada sin escribir

**Pregunta** (encargo del coordinador, 06:10 UTC). La memoria en línea de `titans_mac` gana a su control congelado en validación (de +0,0005 a +0,0041 según la receta). ¿Esa ventaja viene de escribir en la memoria durante la evaluación o de los parámetros que deja un ajuste con escritura?

**Diseño.** Factorial 2×2 ajuste × evaluación sobre la propia ventana fold-008 (origen 2013) y la vista de los trabajos:

| Celda | Ajuste | Evaluación | Cómo se obtiene |
|---|---|---|---|
| oo | mac_online | escribe | emitida por titans_mac; se regenera para la paridad |
| ff | mac_frozen | no escribe | emitida por titans_mac_frozen; se regenera para la paridad |
| of | mac_online | no escribe | gemelo mac_frozen con el estado elegido de titans_mac (todos sus tensores) |
| fo | mac_frozen | escribe | gemelo mac_online con el estado elegido de titans_mac_frozen (control a) |

- Contrastes por sesión, sobre el IC de rangos con al menos 30 activos y con HAC de Newey-West (Bartlett, retardo automático):
  - N = IC(oo) − IC(of): lo que aporta escribir en evaluación a la red ajustada escribiendo.
  - D = IC(oo) − IC(ff): la ventaja total.
  - A = IC(fo) − IC(ff): escribir con una red ajustada sin escritura.
  - I = N − A.
- TTWS = N̄/D̄, con error delta e intervalo de Fieller de la covarianza HAC conjunta. Solo se interpreta si D̄ es positivo con |t| ≥ 1,96.
- Control b (paridad): cada ancla se regenera con su propia variante, y sus tres tramos deben salir bit a bit iguales a los emitidos; si no, la orden se detiene con la primera fila distinta y no se interpreta nada.
- Las semillas se promedian sesión a sesión como réplicas; la unidad es la sesión.
- Potencia, medida solo con validación y calibración:
  - Par base: D = +0,00125 en validación (EE HAC 0,00152, 125 sesiones); MDE esperado ≈ 0,0030 con las 251 sesiones de evaluación.
  - Par con rangos: D = +0,00412 (EE 0,00237); MDE ≈ 0,0047.
  - Las N de los dos pares se leen juntas con Holm.

**Mecanismo y verificación.** `carry_titans(..., evaluated_variant)` carga el state_dict completo y comprueba que el gemelo conserva sus bytes. Solo pueden cambiar la variante y el modo de MAC en los contratos.
- La afirmación de Gemini sobre los búferes es cierta: `copy_paired_parameters` copia parámetros y no los búferes de learning_scale ni head.output_scale. En la campaña no cambia nada y el gemelo no usa esa función.
- En CPU, con el registrador de gradientes, cada gemelo repite bit a bit la ventana ajustada con la otra variante. Es el oráculo N = D = A, I = 0, TTWS = 1.
- El relevo está comprobado con clones nuevos de 114bcf4e y 3d10a30f y `git am` del parche (`gemelos-escritura.patch`, sha256 fa8263f9…): pasan 171 pruebas.

**Ejecución.** Código del modelo de 114bcf4e (el que ajustó la v8) más el parche, en la GB10 con el MPS de la campaña, CUBLAS :4096:8 y FP32 estricto. La unidad tiene MemoryMax 12G y CPUQuota 200 %, y se congela mientras la presión de producción supera 2.

**Resultados** (medido; v8, 2013, semilla 42; paridad bit a bit en los cuatro anclas; t HAC entre paréntesis).

| Par | Tramo | N | D | A | I | TTWS [Fieller 95 %] | Pendiente de N por año |
|---|---|---|---|---|---|---|---|
| base | evaluación | +0,00003 (0,58) | +0,00133 (1,23) | +0,00168 (2,67) | −0,00165 (−2,56) | no interpretable | +0,00009 (0,60) |
| base | calibración | +0,00018 (3,16) | +0,00523 (2,79) | +0,00415 (4,55) | −0,00397 (−4,29) | 0,034 [0,011; 0,125] | — |
| rangos | evaluación | +0,00113 (3,52) | +0,00649 (5,24) | +0,00233 (4,36) | −0,00120 (−2,34) | 0,175 [0,078; 0,313] | −0,00006 (−0,06) |
| rangos | calibración | +0,00205 (5,63) | +0,00680 (3,15) | +0,00391 (5,07) | −0,00187 (−2,38) | 0,30 [0,16; 0,80] | — |

- **Holm declarado** sobre las dos N de evaluación: pasa la del par con rangos (p 0,0004); la del base no (p 0,56).
- **N del par con rangos por trimestre de 2013:** +0,0008 (1,5), +0,0016 (2,0), +0,0015 (2,3) y +0,0006 (1,1). No decae.
- **Lectura:**
  - En el par base, escribir en evaluación no aporta: N tiene un MDE de 0,00013, y la ventaja de la red en línea está en sus parámetros.
  - En el par con rangos, escribir aporta el 17 % de la ventaja.
  - A > 0 en los dos pares: escribir ayuda también a una red ajustada sin escritura.
  - I < 0: el ajuste con escritura ya incorpora parte de lo que aporta escribir.
- **El par sin ganancia de la v10 es este par con rangos:**
  - los estados elegidos coinciden tensor a tensor (60 de 60);
  - comparten receta (72f06e0f) y vista (f5f4d478);
  - las tres tablas de predicciones tienen la misma huella.
  - No se repite en GPU.
- **Coste:** recorridos de 3.782, 2.360, 2.651 y 2.037 s (base) y 3.407, 2.381, 1.583 y 1.270 s (rangos). RSS 10 GB y GPU reservada 4,6 GB.
- **Par de la v10** (titans_mac, receta con ganancia; 2013, semilla 42, evaluación de 251 sesiones,
  t HAC; medido el 22-11):
  - N = +0,0012 (t 2,45; IC [+0,0002; +0,0022]; MDE80 0,0014). Tiene el mismo signo en validación
    (+0,0012, t 1,4) y en calibración (+0,0005, t 0,4), y no decae (−0,0017 al año, t −1,2).
  - D = +0,0014 (t 1,2), no detectado. Cae a lo largo del año (−0,0124 al año, t −4,1; el 4.º
    trimestre da −0,0039).
  - A = +0,0004 (t 0,9) e I = +0,0008 (t 1,1).
  - Lectura: en la v10, escribir en evaluación suma poco, pero suma, y sin coste. Una sola semilla.
- **N con las semillas 43 y 44** (aaa64e35, `posttraining-write-twins-v10-n.json`): solo la paridad
  en línea y la red evaluada sin escribir. Los dos trabajos de la semilla 42 se reutilizan. En GPU
  desde el 23-11 a las 00:38Z.
- **Brazos con reinicio** (ae7eb3fa, f611f7a5; aprobados por el coordinador):
  - R0 evalúa desde el estado inicial de la memoria y del contexto al abrirse la decisión, sin calcular el calentamiento.
  - R63 vuelve además a ese estado cada 63 instantes medidos.
  - Antes de ningún brazo, la calibración se regenera sin reinicio y debe salir bit a bit igual.
  - Regla: R0 y R63 sobre el par cuya N decaiga en evaluación (pendiente negativa con |t| > 2). Si no decae ninguno, solo R0 sobre el par con rangos: responde si las escrituras del calentamiento aportan algo.
  - Coste estimado (hipótesis): 25–32 min de GPU. Medido: 13 min de paridad de calibración y 18 min
    de R0.
- **R0 sobre el par con rangos de la v8** (la regla dio este caso, porque N no decae en ningún par;
  2013, semilla 42, evaluación de 251 sesiones, t HAC, IC del 95 %):
  - N = +0,0011 (t 3,52; IC [+0,0005; +0,0018]).
  - R0 − oo = −0,0001 (t −1,16; IC [−0,0003; +0,0001]; MDE80 0,0003): lo escrito en el calentamiento
    no aporta nada detectable.
  - R0 − of = +0,0010 (t 3,25; IC [+0,0004; +0,0016]): R0 conserva en torno al 90 % de N.
  - Las pendientes no son distintas de cero.
  - Lectura: la ganancia de escribir sale de lo que se escribe en la propia evaluación. Reiniciar ni
    ayuda ni perjudica más allá de ±0,0003, así que la receta no necesita reinicio.


## 7. Documentos de Gemini y auditoría del segundo enjambre

### 7.1 Documentos de diseño del primer enjambre

Los dos documentos del primer enjambre (`memoria-nucleo.md` y `memoria-vecinos.md`) se trataron
como entrada no fiable. Sus ganancias son conjeturas suyas y no se usan. Lo comprobado:

- **La memoria de Titans nunca escribe etiquetas.** `training/titans_walk_forward.py` declara
  `MEMORY_POLICY = "reset_each_pass_then_input_warmup_v1"` (línea 97) y
  `labels="compared_with_the_issued_prediction_after_maturity_never_written"` (línea 149): la
  memoria en línea aprende de las entradas y la etiqueta madura solo se compara con lo emitido.
- **Calentamiento supervisado causal (a).** Las filas de calentamiento no se emiten:
  `financial_run.py` marca `warmup = event.at < source.phase.decision_start` y salta esas filas
  (`if warmup: … continue`, línea 684). Una memoria supervisada integrada en el modelo empezaría,
  por tanto, sin ninguna etiqueta madura en la primera decisión. No se implementa en GPU; su coste
  se acota con el arranque en frío de la adenda (5.1), que vacía la memoria por activo en la
  primera sesión de evaluación.
- **Corrección en unidades de retorno sumada a los cinco cuantiles (b).** Implementada como
  `shift_quantiles`: q_j + s_t·(κ_a·a + κ_v·v) con s_t la dispersión de la predicción en la sesión;
  conserva el orden de los cuantiles y da los mismos rangos de sesión que la combinación tipificada
  (prueba). Se mide en la adenda con la recalibración estática ajustada solo en calibración.
- **Puntos de conexión (c).** `FrozenFinancialConsumer.prepare` existe
  (`models/titans/frozen_financial.py`, línea 507, con la marca de calentamiento) y
  `ChronologicalInference._labels` (`financial_run.py`, líneas 824–863) resuelve las etiquetas
  maduras contra la predicción emitida y llama a `signal.settle(matured)` (línea 856): una memoria
  supervisada que se asentara ahí solo vería etiquetas maduras, como la señal de fiabilidad.
- **Selección exacta de los k mayores.** El documento propone, para GPU, empaquetar (coseno,
  rango) en un entero de 64 bits para que no haya empates. La versión de CPU de este programa
  cumple el mismo contrato por otro camino (cota de los máximos de bloque y orden estable:
  coseno descendente y, a igual coseno, la fila más antigua) y se prueba igual a `np.argsort`
  estable. Su crítica a `episodic_readout.py` (líneas 442–443, puntuaciones en FP64 y orden
  estable) solo vale si ese lector se escalara a un banco de 264.474 filas: su configuración
  admite como mucho 1.024 episodios.
- **Ablaciones del lector por vecinos.** La primera (vecinos del propio activo frente a los de
  otros) es la exclusión de la adenda (`exclude_own`). Las otras cuatro (decaimiento por edad,
  claves aprendidas, distancia L2 frente a coseno y ventanas de 126 y 252 sesiones) no se han
  ejecutado. La cita de TabReD (ICLR 2025) no se ha comprobado aquí.

### 7.2 Auditoría del segundo enjambre (H-01…H-12 de `agente.md`)

Cada afirmación se contrastó con el código. Si era un defecto, con una prueba que fallaba antes del arreglo.

| Hallazgo | Veredicto | Evidencia o arreglo |
|---|---|---|
| H-01 | Real: el bucle aprendía con Pearson aunque la métrica declarada es de rangos. La parte de NaN no se confirma | c42fbaa8. La prueba fallaba: una herramienta con Pearson alto y rangos negativos se llevaba el 99,96 % del peso. Una opción constante pierde 1/2 sin NaN |
| H-02 | Propiedad del diseño | La pérdida (1 − IC)/2 está documentada en `online_arbiters` |
| H-03 | No reproducido | Es la construcción publicada de AdaNormalHedge.TV (w(0,0) = 0,1978 para una copia nueva). El peso de una opción en una sesión solo suma sus copias despiertas |
| H-04, H-05 | Propiedad del diseño | El agente ordena (métrica de IC); la CQR en línea es la declarada en PT2 |
| H-06 | Solo aplica a la declaración antigua `dgx-us-cognitive-agent-v1.json` | Las declaraciones del post-entrenamiento tienen un modelo por herramienta |
| H-07 | No aplica a estos datos | Retraso medido d = 1 en las 312 sesiones de calibración y evaluación de la v8 en 2013 |
| H-08 | (1) Real, de documentación. Lo demás es diseño | fc9fc2fb: el eje «al alza» es el signo del residuo medio, no la dirección del mercado |
| H-09 | Real en parte: no se validaba el orden de cada semilla | c76d7133. La varianza epistémica no es un defecto del método declarado (media de Vincent) |
| H-10 | Regla del proyecto | Fallar con un mensaje que diga dónde y por qué |
| H-11 | (i) Real; (ii) diseño documentado; (iii) no es un defecto; (iv) diseño | d94c7807. (iii): el paso descendente de Romano y Wolf es bilateral (`stepdown`), y la dirección solo marca «favorable»: una interacción negativa sale con su intervalo y su p bilateral |
| H-12 | (b) Real; (a), (c) y (d), diseño; (e) menor, sin llamadas en producción | 803f6c7e: la calibración de cada ventana madura, por mercado, antes de su primera decisión evaluada |


### 7.3 Tercer enjambre: entregas y auditoría de código

Cada hallazgo se reprodujo con una prueba que fallaba antes del arreglo.

| Hallazgo | Veredicto | Commit |
|---|---|---|
| H03: `read()` de la memoria por activo con un divisor mínimo de 1e-12 | Real. Tras 45 semividas daba 0,0275 en vez de 0,9674. Sin efecto en 2013: el hueco máximo es de 126 sesiones | e72f82cc |
| H14: una semilla emitida a medias no se rehacía; temporales con nombre fijo | Real | b872cc88 |
| H01: los vecinos rechazaban una madurez en 2024 | Real en sintético. No ocurre con la vista (`target_after_cutoff`) | dbdf70a0 |
| H04: `emit()` y `restore()` de la memoria por activo | Real. Un rechazo dejaba rastro y se aceptaban estados firmados inválidos | 23ac080b |
| H15: el análisis de los gemelos puntuaba todos los mercados; la paridad fallaba con NaN idénticos | Real. Sin efecto en el par de la v10, que solo tiene US | 8c89db99 |
| H02: claves de vecinos fuera del rango de FP32; `restore()` de vecinos | Real. Las filas nominales no cambian bit a bit | 9577db90 |

Las siete carencias que el informe del agregador atribuye a los árbitros (f611f7a5):
- **Pérdida lineal ciega a la covarianza.** Límite de diseño, que cubre el agregador adoptado.
- **Árbitro con Pearson.** Es coherente con su métrica `session_ic`. Los agentes declaran el IC de rangos y aprenden con rangos.
- **`mature_mixture`.** O(T·N); solo `_reliability` es cuadrática, y con T ≈ 437 no pesa.
- **`regimes()` sin filtrar las sesiones puntuadas.** Sin efecto: ninguna sesión de 2013 tiene menos de 30 filas.
- **Las otras tres.** Son el alcance declarado o las cubre el agregador.

## 8. Limitaciones y riesgos

- **Universo de supervivientes (D-2 del cuaderno del coordinador).** De los 4.202 activos de EE. UU.,
  4.200 llegan al final de 2023. La etiqueta resta el alfa de cada activo y lo operable no, así que
  el IC operable es la medida más expuesta al sesgo; los contrastes entre modelos, que comparten
  universo, lo están menos que los niveles.
- **Un origen y una semilla.** Todo lo medido en CPU sobre Titans es la v8 en 2013 con la semilla
  42 (y, para la memoria por activo, la réplica entre orígenes del estado antiguo sin rasgos, también
  con la semilla 42). Ninguna cifra se generaliza sin más orígenes ni semillas.
- **Modelos muy correlacionados.** Los trece modelos del estudio comparten datos y casi toda la
  arquitectura: sus contrastes no son independientes y no se suman como evidencia.
- **Claves de los vecinos fuera del modelo.** Las 92 entradas salen de la caché del laboratorio
  tabular (en la carpeta temporal de la sesión, con su huella SHA-256 en cada informe). Coinciden
  fila a fila con las emitidas, pero no son las entradas que ve el modelo.
- **Tipificación con las filas etiquetadas.** Los correctores tipifican la predicción y ordenan
  la etiqueta solo con las filas de la sesión que tienen etiqueta; una fila sin etiqueta no entra
  en esas estadísticas.
- **Selección por calibración con 61 sesiones.** La elección de κ y de la semivida usa pocas
  sesiones; el contraste primario es siempre la configuración del laboratorio, fijada antes de ver
  estas emisiones.
- **Primera ejecución del estudio detenida.** La declaración original
  (`posttraining-memories-v10.json`) se ejecutó con el método de vecinos v1 y se detuvo sin
  escribir informe; la adenda (v2) se declaró antes de leer ningún resultado. Los vecinos de v1 y
  v2 solo difieren en los empates en el k-ésimo coseno.
- **Desliz de TDD.** Parte del estudio de memorias se escribió antes que su prueba; se compensó con
  mutaciones (9 muertas en esa entrega) y queda registrado aquí.
- **Lo que no se pudo medir.**
  - El conjunto de semillas en 2013: faltan la 43 y la 44 en la v8.
  - B6 y E1: necesitan las claves del códec en GPU.
  - Cualquier emisión de la v10: a las 08:18 UTC sus 21 trabajos de 2013 seguían sin ejecutar (`declared_not_executed`). El plan del relevo pone las tres semillas del núcleo y de sus controles en 2013 antes que cualquier criba.
- **Gemelos con un origen y una semilla en la v8.** La inferencia es dentro de la red ajustada, no entre ajustes. La v10 con tres semillas es la réplica declarada.

## 9. Referencias

Comprobadas el 22 de noviembre de 2026 en el resumen del artículo o del editor; tratadas como datos.

- Achlioptas (2003). Database-friendly random projections: Johnson-Lindenstrauss with binary coins. *Journal of Computer and System Sciences* 66(4). doi:10.1016/S0022-0000(03)00025-4.
- Angelopoulos, Candès y Tibshirani (2023). Conformal PID control for time series prediction. arXiv:2307.16895.
- Ash y Adams (2020). On warm-starting neural network training. NeurIPS 2020. arXiv:1910.08475.
- Auer, Gauch, Klotz y Hochreiter (2023). Conformal prediction for time series with modern Hopfield networks (HopCPT). arXiv:2303.12783.
- Behrouz, Zhong y Mirrokni (2025). Titans: learning to memorize at test time. arXiv:2501.00663.
- Defazio et al. (2024). The road less scheduled. arXiv:2405.15682.
- Fieller (1954). Some problems in interval estimation. *Journal of the Royal Statistical Society, Series B* 16(2):175–185. doi:10.1111/j.2517-6161.1954.tb00159.x.
- Freund, Schapire, Singer y Warmuth (1997). Using and combining predictors that specialize. STOC 1997, 334–343. doi:10.1145/258533.258616.
- Gibbs y Candès (2021). Adaptive conformal inference under distribution shift. arXiv:2106.00170.
- Gneiting y Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* 102(477):359–378 (puntuación de intervalo).
- Hinton, Vinyals y Dean (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Hu et al. (2021). LoRA: low-rank adaptation of large language models. arXiv:2106.09685.
- Izmailov et al. (2018). Averaging weights leads to wider optima and better generalization (SWA). arXiv:1803.05407.
- Jegadeesh (1990). Evidence of predictable behavior of security returns. *Journal of Finance* 45(3):881–898.
- Khandelwal et al. (2019). Generalization through memorization: nearest neighbor language models (kNN-LM). arXiv:1911.00172.
- Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks (EWC). *PNAS*. arXiv:1612.00796.
- Lakshminarayanan, Pritzel y Blundell (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. arXiv:1612.01474.
- Li, Grandvalet y Davoine (2018). Explicit inductive bias for transfer learning with convolutional networks (L2-SP). ICML 2018. arXiv:1802.01483.
- Luo y Schapire (2015). Achieving all with no parameters: AdaNormalHedge. arXiv:1502.05934.
- Morales-Brotons, Vogels y Hendrikx (2024). Exponential moving average of weights in deep learning. arXiv:2411.18704.
- Newey y West (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*. doi:10.2307/2297912.
- Romano y Wolf (2016). Efficient computation of adjusted p-values for resampling-based stepdown multiple testing. *Statistics & Probability Letters* 113:38–40. doi:10.1016/j.spl.2016.02.012.
- Rolnick et al. (2019). Experience replay for continual learning. arXiv:1811.11682.
- Romano, Patterson y Candès (2019). Conformalized quantile regression. arXiv:1905.03222.
- Sun et al. (2020). Test-time training with self-supervision for generalization under distribution shifts. arXiv:1909.13231.
- Wang et al. (2021). Tent: fully test-time adaptation by entropy minimization. arXiv:2006.10726.
- Wortsman et al. (2022). Model soups. arXiv:2203.05482.
- Holm (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2):65–70.

