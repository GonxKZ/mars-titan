# Memoria redundante por áreas (B1)

`redundant_area_memory_v1` es la variante B1 del estudio de variantes de [#742](https://github.com/GonxKZ/mars-titan/issues/742), dentro de la familia de [#750](https://github.com/GonxKZ/mars-titan/issues/750). Reparte la corrección madura de B6 en A áreas, cada una con una permutación fija de la clave, una proyección DPFP y su propia memoria asociativa lineal. La lectura es la media de las áreas activas. Está implementada y comprobada técnicamente con datos construidos a mano. No se ha ejecutado sobre ventanas de datos reales y no hay ninguna medida de su utilidad predictiva.

Cada área recibe la misma asociación madura y calcula pesos propios sobre rasgos distintos. El estado no conserva copias del conjunto de datos. Esta separación no implementa todavía comunicación entre áreas ni reproduce un mecanismo biológico.

## Dónde se inserta

B1 es una corrección madura, igual que B6 ([contrato de B6](../engineering/mars-titan-extensions.md#corrección-asociativa-b6-en-la-sesión)). Suma a la predicción del Titans-MAC `mac_online` congelado la lectura de las áreas. Escribe el objetivo de corrección `etiqueta − núcleo` cuando madura la etiqueta. Este valor no es el error de la predicción final, que sería `etiqueta − emitida`. Tiene estado propio y conserva los pesos rápidos de Titans. La combinación con el banco episódico sigue pendiente y se rechaza para evitar mezclar sus objetivos de escritura.

No es un cambio de `FinancialConfig`. `FinancialPredictor` no recibe etiquetas por contrato, y una memoria que solo escribe resultados maduros necesita el recorrido que los entrega en orden. Por eso B1 entra como opción del componente `associative_memory` de la [declaración de MARS-TITAN](../../configs/titans/mars-titan-extensions.json), con el bloque `areas`, y la ventana `run_correction_window` la ejecuta sobre el padre de la misma ventana y semilla sin código nuevo. Sin el bloque `areas` la identidad, la huella y el cálculo de B6 no cambian.

## Fuentes

- Danihelka, Wayne, Uria, Kalchbrenner y Graves (2016), [*Associative Long Short-Term Memory*](https://proceedings.mlr.press/v48/danihelka16.html), PMLR 48:1986–1994, sección 3, ecuaciones 6 a 8. Se contrastaron en el PDF publicado.
- Schlag, Irie y Schmidhuber (2021), [*Linear Transformers Are Secretly Fast Weight Programmers*](https://proceedings.mlr.press/v139/schlag21a.html), PMLR 139:9355–9366. Se contrastaron la regla delta, la normalización por suma, DPFP (ecuación 37) y el listado 1 del suplemento. La ecuación indexa con i + j y el listado usa `roll(j)`, equivalente a i − j. La implementación conserva el orden del listado. Los dos órdenes difieren por una permutación fija de coordenadas.
- Nadaraya (1964), [*On Estimating Regression*](https://epubs.siam.org/doi/abs/10.1137/1109020), y Watson (1964), [*Smooth Regression Analysis*](https://www.jstor.org/stable/25049340), como antecedentes del control no paramétrico. Se verificaron sus datos bibliográficos. La retención y el límite de capacidad de este control son decisiones del proyecto.

## Por qué una permutación sola no basta

En Danihelka et al. la ligadura es elemento a elemento y el valor no se permuta. El ruido de cada copia cae en otras coordenadas y la media lo reduce. Con un estado matricial completo y ligadura por producto exterior ocurre lo contrario. Si Q es ortogonal y A₀ = 0, escribir con las claves Qk da Q·A en las dos reglas:

$$
A'^{+}=\rho A'+\eta\,Qk\,(v-k^\mathsf{T}Q^\mathsf{T}A')^\mathsf{T}=Q\,A^{+}\quad\text{(delta)},\qquad
(I+\eta QGQ^\mathsf{T})A'^{+}=\rho A'+\eta QK^\mathsf{T}WV\ \Rightarrow\ A'^{+}=QA^{+}\quad\text{(proximal)}.
$$

La lectura (Qq)ᵀ(QA) = qᵀA es la misma en todas las áreas. Una permutación es ortogonal, así que A áreas permutadas con clave lineal son A copias exactas de una sola memoria. `test_permuting_a_linear_key_leaves_every_read_unchanged` lo comprueba a 10⁻¹³ con las dos reglas. Con valor escalar tampoco ayuda una traza vectorial: decodificar un escalar con un funcional lineal vuelve a ser invariante.

DPFP rompe esa equivalencia porque multiplica coordenadas vecinas de r = [ReLU(x), ReLU(−x)]. Al permutar la clave antes de proyectar, cada área multiplica otros pares, así que sus regresiones son distintas y su media es un conjunto de regresiones sobre rasgos cuadráticos distintos. Es la proyección de una de las dos fuentes y no añade parámetros.

## Ecuaciones, módulo y pruebas

Sea k ∈ ℝ⁶⁴ la clave del codec normalizada en L2 y FP64, la misma de B6, π_s la permutación fija del área s y ν el número de desplazamientos.

| Ecuación | Módulo | Prueba |
| --- | --- | --- |
| r(x) = [ReLU(x), ReLU(−x)], DPFP_ν(x)_{(j−1)·2d+i} = r_i · r_{(i−j) mod 2d}, j = 1…ν | `memory/area_memory.py:dpfp` | `test_dpfp_matches_equation_37_and_the_listing_order` |
| φ_s(k) = DPFP_ν(P_s k)/‖DPFP_ν(P_s k)‖, o cero si la norma es cero | `AreaMemory.features` | `test_area_write_and_read_follow_the_proximal_equation_per_area`, `test_a_key_without_neighbour_pairs_neither_reads_nor_writes` |
| (I + η Φ_sᵀWΦ_s) A_s⁺ = ρA_s + η Φ_sᵀWv, con W = I/n en la cohorte | `AssociativeMemory._proximal` de B6 en cada área | la misma prueba, frente a `torch.linalg.solve` |
| y(q) = (1/\|S\|) Σ_{s∈S} φ_s(q)ᵀ A_s, con S las áreas activas | `AreaMemory.read` y `read_areas` | `test_lesion_reads_the_renormalized_mean_of_the_active_areas` |
| emitida = núcleo + y(q) con el estado anterior al evento, escritura de `etiqueta − núcleo` después de emitir | `CorrectionInference` | `test_area_emissions_follow_the_independent_recurrence` |
| ŷ(q) = Σ_i w_i v_i, w_i ∝ exp(qᵀk_i/τ) ρ^(c−c_i) | `memory/kernel_memory.py:KernelMemory.read` | `test_read_is_the_weighted_mean_of_the_mature_episodes`, `test_kernel_emissions_follow_the_nadaraya_watson_recurrence` |

La referencia de la ventana se escribe aparte en `tests/memory/test_area_correction_window.py`: parte de las predicciones del núcleo y de las claves del codec, construye los rasgos con un bucle sobre la ecuación 37 y resuelve cada escritura proximal con su sistema lineal. Las 20 emisiones de los seis eventos de cuatro flujos coinciden a 10⁻¹³.

La regla delta también está admitida. Escribe fila a fila y, como en B6, su resultado depende del orden canónico de la cohorte. Una tasa pequeña puede reducir esa sensibilidad, pero no demuestra invariancia. Los modelos declarados usan la regla proximal, que es su paso implícito sobre la cohorte e invariante al orden. Con ‖φ‖ = 1 se mantiene la cota η ≤ 2 − λ de la regla delta en cada área.

## Controles

Todos están implementados y declarados en [redundant-area-memory-v1.json](../../configs/experiments/redundant-area-memory-v1.json), con la receta de B6 ([η = 0,05 o 0,25 y λ = 0,01](../../configs/titans/mature-correction-historical-masked.json)) salvo Nadaraya-Watson, que tiene la suya.

| Modelo | Configuración | Qué descarta |
| --- | --- | --- |
| `area_memory` | A = 4, ν = 1, permutaciones con semilla 0 | Variante |
| `single_area_same_coordinates` | A = 1, ν = 4, sin permutar | Que repartir en áreas promediadas aporte frente a una regresión conjunta con las mismas 512 coordenadas de estado |
| `identical_areas` | A = 4, ν = 1, sin permutar | Que las permutaciones aporten. Con cuatro copias iguales la media es exacta y lee lo mismo que una sola área con ν = 1 |
| `shuffled_labels` | Como la variante, con los valores barajados dentro de cada escritura | Que se use la asociación entre clave y error. Conserva los valores de la cohorte y su media, así que deja intacto el sesgo |
| `zero_memory` | Predicciones del padre | Que corregir aporte algo. Con η = 0 la ventana las reproduce exactamente (`test_zero_rate_areas_reproduce_the_core_inference`) |
| `nadaraya_watson` | τ = 0,1 o 0,5, λ = 0,01, hasta 65.536 episodios | Que resumir asociaciones en estados fijos compita con guardar los episodios y promediarlos por parecido |
| `linear_codec` | B6 proximal con la clave del codec | Que la proyección por áreas aporte frente a la corrección lineal ya declarada |

El barajado usa un generador PCG64 con la semilla de las áreas y el primer identificador canónico de la escritura, y se aplica en orden canónico. No guarda estado de RNG y no depende del orden de entrega de las filas (`test_shuffled_labels_write_a_fixed_permutation_of_the_cohort_values`). El ensemble de semillas como control de la redundancia pertenece a otra familia del estudio.

## Desviaciones respecto a las fuentes

- Danihelka et al. usan trazas complejas con ligadura elemento a elemento y valores vectoriales. Aquí cada área es una memoria asociativa real con ligadura por producto exterior y valor escalar, y la permutación actúa a través de DPFP. La media de la ecuación 7 se conserva, restringida a las áreas activas.
- Schlag et al. aprenden β en cada paso. Aquí η es fijo por receta y caso de búsqueda, y no hay parámetros entrenables.
- La ecuación 29 normaliza φ por su suma. Aquí se normaliza en L2 para que ‖φ‖ = 1, η tenga la escala de B6 y se cumpla la cota de las claves de `AssociativeMemory`. Los productos se calculan con mantisas y exponentes relativos antes de normalizar. Así se conserva la dirección cuando todos los productos son menores que el mínimo representable, como en los casos con coordenadas de 10⁻²⁰⁰ o 10⁻³⁰⁰. Esto sigue siendo aritmética FP64, sin certificación de error. Una proyección nula no aporta una asociación nueva. La retención y el contador siguen actualizándose según la regla.
- La regla declarada es la proximal por cohorte, el paso implícito de la regla delta, y no la escritura fila a fila.
- El encargo hablaba de estados D×D. Con un valor escalar cada área guarda 2dν × 1 coordenadas. Un estado D×D solo tendría sentido con valores vectoriales y, con clave lineal, sus áreas seguirían siendo copias exactas.
- Nadaraya-Watson añade retención por escritura y una capacidad. Con λ = 0,01 y 256 activos, el factor temporal de los episodios descartados al llenar 65.536 cae por debajo del 8 % del de la última escritura. El factor de similitud puede compensarlo. La capacidad puede borrar casos raros relevantes y no tiene una garantía de conservación predictiva.

## Causalidad y estado

B1 hereda el orden de B6. Todas las lecturas de un evento usan el estado confirmado antes de él, las etiquetas que maduran en el evento se escriben después de emitir, con el valor `etiqueta − núcleo` y en orden canónico de disponibilidad, decisión e identificador. `canonical_order` rechaza una etiqueta que madura después del corte o una ya aplicada, y las pruebas lo fuerzan con las dos memorias. Alterar las etiquetas que maduran en el tercer instante solo cambia emisiones posteriores, y alterar las entradas del último instante no cambia ninguna anterior (`test_future_labels_and_inputs_never_change_earlier_emissions`). Las predicciones, parámetros y buffers del padre no cambian ni reciben gradientes.

El estado de las áreas se exporta con la exportación de B6 de cada área, su contador, su cursor y una huella común. Nadaraya-Watson exporta claves, valores, la escritura de cada muestra y los contadores. `MatureCorrection.restore` rechaza otra configuración, una huella alterada o estados inconsistentes. `FinancialSession` guarda esa exportación en cada generación. Se comprobaron con el enlace nativo CPU los cortes, la recuperación y la igualdad de emisiones entre sesión y ventana. En la ventana cada tramo empieza vacío y un tramo interrumpido se repite desde el principio.

La corrección hereda el ajuste acotado del padre cuando declara `fit_months`. Las filas de validación, calibración y evaluación coinciden. Kalman conserva su rechazo de esa opción hasta acotar también su tramo de estimación de ruido. Las nuevas variantes no amplían las reglas de la campaña histórica ni activan aprendizaje.

## Coste medido

Esta tabla conserva la medida anterior a la incorporación de B2. La [medida conjunta posterior](area-query-bridge-v1.md#coste-y-comprobaciones) usa otra revisión identificada en sus recibos y no sustituye estos datos.

La medida técnica del 26 de octubre usa claves de 64 coordenadas, valor escalar, dos hilos, FP64 y CPU. Cada estado se carga antes del cronómetro. Se miden cohortes de 64, 256 y 1024, con tres calentamientos y veinte repeticiones. El control Nadaraya-Watson tiene su capacidad de 65.536 muestras llena. No se han medido GPU ni calidad predictiva.

| Memoria | Lectura de 256 claves, p50 | Escritura de 256 resultados, p50 | Bytes numéricos residentes |
| --- | ---: | ---: | ---: |
| B6 lineal | 0,052 ms | 0,377 ms | 512 |
| Cuatro áreas | 3,302 ms | 6,079 ms | 6.144 |
| Un área de igual número de coordenadas | 1,955 ms | 11,904 ms | 4.608 |
| Áreas idénticas | 2,886 ms | 6,475 ms | 6.144 |
| Etiquetas barajadas | 3,603 ms | 5,753 ms | 6.144 |
| Nadaraya-Watson | 116,610 ms | 36,731 ms | 34.603.008 |

El control de área única iguala 512 coeficientes, no el coste de factorización ni los índices. Los bytes excluyen objetos Python y temporales de las operaciones. La ejecución completa duró 26,2 s y alcanzó 713 MB de RAM. Los modelos se midieron por bloques y el equipo conservaba cargas ajenas. No deben interpretarse pequeñas diferencias como aceleraciones reproducibles. Los valores individuales, las huellas de código y el método de percentiles quedan en el [recibo de coste](../../reports/engineering/area-memory-20261026/cost.json).

En la versión anterior al ajuste de precisión de DPFP se comparó además una lectura batched con la referencia, alternando el orden durante cuarenta repeticiones. A 256 claves redujo la mediana de lectura de 3,931 a 3,462 ms, pero el ciclo de lectura y escritura solo pasó de 10,604 a 10,295 ms. No se adoptó: el beneficio del ciclo era pequeño y la referencia conserva una lectura más directa. La [prueba de esa alternativa](../../reports/engineering/area-memory-20261026/batched-read-probe.json) corresponde a aquella versión. No mide la implementación final ni una aceleración del modelo completo.

## Comprobaciones técnicas

La batería final de esta variante aprobó 182 casos CPU, sin fallos ni omisiones. Quedaron deseleccionados los dos casos CUDA. Las 17 pruebas de la guarda de aprendizaje habían pasado en la batería anterior al último ajuste de DPFP. El [recibo de verificación](../../reports/engineering/area-memory-20261026/verification.json) separa ambas ejecuciones y conserva las huellas de sus salidas. Las pruebas de ventanas usan un optimizador de registro que no cambia los pesos del padre. No son entrenamientos ni evaluaciones científicas.

Se detectaron las siete [mutaciones dirigidas](../../reports/engineering/area-memory-20261026/mutations.json), cada una entre dos referencias verdes. Cubren dimensiones, dispositivo, cursor, pérdida de rasgos pequeños, admisión de etiquetas futuras, media de áreas activas y control de etiquetas barajadas. No hubo pasos de optimizador en esas fases.

Coverage.py 7.16.1 midió 246 de 260 sentencias y 65 de 78 ramas en los dos módulos nuevos. El [informe de calidad](../../reports/engineering/area-memory-20261026/quality.json) usa Radon 6.0.1 y CRAP = C² × (1 − cobertura de sentencias)³ + C. El máximo es 25,86 en la validación del estado de Nadaraya-Watson. Se publica aparte la [cobertura de integración](../../reports/engineering/area-memory-20261026/quality-integration.json). Estos valores describen las rutas comprobadas y no demuestran que el proyecto carezca de errores.

## Pendiente

- Ejecutar B1 y sus controles en la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)) cuando se levante el bloqueo de aprendizaje. La validación elige η o τ, así que la ventana comprueba el bloqueo antes de leer ninguna fuente.
- Comprobar la interoperabilidad del padre en `cuda:0` con el estado CPU cuando la GPU quede libre.
- El [puente B2](area-query-bridge-v1.md) está conectado con sus controles y comprobado en CPU. Su interoperabilidad CUDA y evaluación predictiva siguen pendientes.
- B4 (estudio de lesiones) y B3 (claves ligadas a roles) siguen pendientes. La máscara técnica de áreas no completa B4.
