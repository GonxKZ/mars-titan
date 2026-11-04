# Puerta de reescalado por regiones en fecha

`pit_region_gate_v1` es la variante C4 del estudio de variantes ([#742](https://github.com/GonxKZ/mars-titan/issues/742), [#751](https://github.com/GonxKZ/mars-titan/issues/751)). Es una etapa de salida. Agrupa los activos de cada mercado por la correlación de sus residuos ya maduros y reescala la predicción emitida con un factor por región. No cambia pesos, recetas ni la identidad del predictor. Está implementada en `src/mars_titan/evaluation/pit_region_gate.py` y comprobada con pruebas técnicas sobre historiales y paneles sintéticos. No se ha evaluado con datos y no hay ninguna mejora medida.

## Motivación y fuentes

La decisión 9 del estudio fija cómo se definen las regiones. Primero modalidad y mercado, después clústeres por correlación de la ventana pasada. El GICS actual solo sirve como control de fuga. Los cambios de GICS de marzo de 2023 movieron Visa, Mastercard, PayPal y Fiserv de Tecnología de la Información a Financieras, y catorce empresas del S&P 500 cambiaron de sector ([S&P DJI, Indexology](https://www.indexologyblog.com/2023/03/16/2023-gics-changes-sp-global-1200-impact-analysis/)). Las fichas sectoriales de FinMultiTime son actuales (`SectorMap` declara `point_in_time=False`), así que una región «tecnología» con la etiqueta de hoy usaría información posterior en todo el periodo desde 2000. No disponemos de una clasificación con fecha de vigencia verificada. Por eso la variante usa solo correlaciones pasadas.

Agrupar acciones por la correlación de sus rentabilidades con un enlace jerárquico procede de [Mantegna (1999)](https://doi.org/10.1007/s100510050929), que construye una jerarquía de activos a partir de la distancia derivada de ρ. MASTER usa un vector de estado del mercado para reescalar características mediante una puerta ([Li et al., 2024](https://arxiv.org/abs/2312.15235)). Un preprint de 2026 sobre volatilidad realizada observa que las variables de régimen ayudan más cuando solo gobiernan la puerta que cuando entran como características ([Borde, 2026](https://ideas.repec.org/p/arx/papers/2608.12251.html)). Esa evidencia es sobre volatilidad, no sobre residuos de rentabilidad, y solo motiva la forma de la variante: la región decide la escala de la predicción, no añade otra entrada.

El resumen de la fase 0 en #742 muestra predicciones mucho más amplias de lo que justifica su ordenación, y A1 corrige esa amplitud con un factor λ común por mercado. C4 pregunta si ese exceso de amplitud depende del grupo de activos. Si una región tiene residuos que el modelo ordena peor, su λ debe ser menor. Como el factor cambia entre regiones dentro de una misma sesión, C4 puede cambiar también el IC por sesión, cosa que A1 no hace.

## Regla

Sea $t_G$ la primera emisión de la calibración (G) de un mercado g. Para cada activo a se toman sus residuos con madurez $\le t_G$ en los W últimos instantes de madurez del mercado (W = 126 por defecto, unos seis meses de la validación que precede a G). Cada serie se estandariza con su media y su desviación típica en la ventana. La correlación de un par usa solo las madureces comunes:

$$
\rho_{ab}=\frac{\sum_{t\in C_{ab}} z_{a,t}\,z_{b,t}}{\sqrt{\sum_{t\in C_{ab}} z_{a,t}^2\;\sum_{t\in C_{ab}} z_{b,t}^2}},\qquad
\rho_{ab}=0 \text{ si } |C_{ab}|<m.
$$

Las regiones salen del enlace medio sobre $1-\rho$, cortado en k grupos (k = 8 y m = 60 por defecto). Se numeran desde 1 por el menor activo que contienen, para que la numeración no dependa del orden interno de SciPy. Un activo con menos de m etiquetas o con residuos constantes va a la región 0. Después, con las filas de G de cada región r,

$$
\lambda_{g,r}=\arg\min_{\lambda\in[0,1]}\sum_{i\in G_{g,r}} w_i\,|y_i-\lambda x_i|,\qquad \hat y_i=\lambda_{g,r(i)}\,x_i,
$$

donde $w_i$ es el peso 1/n de la sesión y λ se calcula con la mediana ponderada de A1 (`scale_diagnostics.shrinkage_factor`). Un activo que no estaba en la calibración, o una región sin factor, usa el factor del mercado de A1, y un mercado sin factor predice cero, igual que en A1. Regiones y factores quedan congelados antes de abrir la evaluación.

## Controles

| `regions` | Regiones | Para qué sirve |
| --- | --- | --- |
| `residual_correlation` | Clústeres de la correlación pasada | La variante |
| `permuted` | Las mismas etiquetas y tamaños repartidas al azar entre los activos con historia, con `SeedSequence([seed, mercado])` | Separa el efecto de agrupar bien del de tener k factores en lugar de uno |
| `market` | Una sola región por mercado | Reproduce A1 bit a bit. Es el modelo plano |
| `current_sector` | El sector de las fichas actuales, numerado por su nombre | Mide cuánto aportaría una etiqueta con fuga. Nunca es una evaluación válida por sí misma |

La región 0 no se permuta, porque reúne los activos sin evidencia y su tamaño no lo decide el agrupamiento. Solo el control sectorial recibe los sectores. Cualquier otra combinación se rechaza.

La variante se descarta si no mejora al control permutado en el contraste principal de la criba, que tras la fase 0 es el IC por sesión. El MAE por sesión frente a A1 se informa como contraste secundario. Si `current_sector` mejora mucho más que la variante, la diferencia apunta a la fuga de la etiqueta actual, no a una ventaja de los sectores. El contraste y la familia de corrección los fija el protocolo de la criba ([#746](https://github.com/GonxKZ/mars-titan/issues/746)).

## Correspondencia entre ecuaciones, módulo y pruebas

Las pruebas están en `tests/evaluation/test_pit_region_gate.py`. El historial sintético tiene doce activos y dos factores latentes, de modo que los grupos correctos se conocen de antemano. La correlación, el agrupamiento y la permutación están en `src/mars_titan/data/correlation_regions.py`, que también usa C3 para sus sectores estadísticos, y sus pruebas propias en `tests/data/test_correlation_regions.py`.

| Elemento | Código | Prueba |
| --- | --- | --- |
| $\rho_{ab}$ con madureces comunes y mínimo m | `correlation_regions.correlation` | `test_correlation_matches_pearson_when_fully_observed_and_ignores_thin_pairs`, en las pruebas del módulo común |
| Clústeres numerados por su menor activo | `residual_regions` | `test_planted_groups_are_recovered_and_numbered_by_their_first_asset` |
| Solo madureces $\le t_G$ | `residual_regions` | `test_regions_only_read_labels_matured_by_the_cutoff` |
| Región 0 | `residual_regions` | `test_short_or_constant_histories_go_to_region_zero` |
| $\lambda_{g,r}$ con los pesos de A1 | `fit` | `test_each_region_gets_the_a1_factor_of_its_own_rows` |
| $\hat y_i$ y factor del mercado | `apply` | `test_application_scales_each_row_and_unseen_assets_use_the_market_factor` |
| Paridad con A1 | `fit` y `apply` | `test_the_market_control_reproduces_a1_bit_for_bit` |
| Control permutado | `correlation_regions.permuted_regions` | `test_the_permuted_control_keeps_sizes_and_changes_the_assignment` |
| Control sectorial | `sector_regions` | `test_the_current_sector_control_is_the_only_mode_that_reads_sectors` |

## Causalidad, estado y paridad

- Las regiones solo leen etiquetas con madurez $\le t_G$. La prueba añade etiquetas extremas que maduran un día después del corte y las regiones no cambian. Las mismas etiquetas sí cambian el reparto cuando el corte las alcanza, lo que muestra que el filtro es lo que las deja fuera.
- λ solo usa filas de G. El registro guarda la mayor madurez de esas etiquetas y `apply` rechaza un tramo con alguna emisión anterior.
- `apply` no lee los objetivos de la evaluación.
- Dos etiquetas del mismo activo con la misma madurez se rechazan, porque su orden sería arbitrario. Cualquier instante en 2024 se rechaza, también en el historial.
- El activo se lee de la identidad `mercado/activo/instante` de cada fila y se rechaza una identidad con otra forma o con un mercado distinto del de la fila.
- El estado es un registro JSON congelado con la configuración, las regiones, los factores por región y por mercado y la madurez del ajuste. `stage_sha256` lo identifica y la ida y vuelta por JSON reproduce las salidas bit a bit.
- La etapa se registra en `evaluation/output_stages.py`. Sin etapa declarada, la salida es la del modelo.

## Desviaciones y límites

- **Puerta ajustada en la salida, no aprendida dentro del modelo.** La propuesta del estudio describe una puerta por región que reescala el estado del modelo. Dentro del modelo, la puerta necesitaría regiones estables durante todo el ajuste. Los clústeres de una ventana móvil cambian de numeración y de miembros entre orígenes, y alinearlos en el tiempo es otro problema con sus propias decisiones. La etapa de salida usa una sola asignación por origen, fijada antes de G, y no exige otro ajuste del modelo. Una puerta aprendida queda como ampliación posible si esta versión supera a sus controles.
- No hay sector en fecha. Si se verificara una clasificación con fecha de vigencia, entraría como otro valor de `regions` con su propio control.
- El corte se hace por número de grupos y no por altura del dendrograma. Con un mercado de pocos activos con historia, varios grupos pueden quedar con un solo activo, y su λ se ajusta con pocas filas.
- Las regiones se calculan una vez por origen con la ventana previa a G. No se actualizan durante la evaluación.
- La lectura de los Parquet y la alineación del historial con la vista de cada ventana las hará la criba.

## Coste medido

Sin tensores ni parámetros entrenables no hay forward ni backward. `benchmarks/pit_region_gate_cost.py` mide la etapa en CPU con los paneles sintéticos de `output_stage_cost` y un historial de 126 madureces por activo con ocho factores latentes, de semilla fija, que solo fijan formas. En los tres casos el agrupamiento forma ocho regiones del mismo tamaño que los grupos plantados. El [recibo](../../reports/engineering/pit-region-gate-20261027/cpu-cost.json) guarda medianas de cinco repeticiones con dos hilos de BLAS, en un Ryzen 9 8945HS con otras sesiones compartiendo la máquina (carga media de 30 al empezar y la swap llena). Esa carga limita la comparación y las cifras no son cotas superiores.

| Caso | Filas evaluadas | Filas de G | Lectura de activos | Regiones | Ajuste | Control permutado | Emisión | Pico de NumPy en el ajuste |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 activos, 63 sesiones | 16.128 | 16.128 | 0,021 s | 0,055 s | 0,082 s | 0,085 s | 0,005 s | 7,8 MB |
| 256 activos, 252 sesiones | 64.512 | 16.128 | 0,115 s | 0,051 s | 0,088 s | 0,087 s | 0,024 s | 7,8 MB |
| 4.200 activos, 252 sesiones | 1.058.400 | 264.600 | 1,47 s | 7,23 s | 10,84 s | 10,51 s | 0,226 s | 1.071,8 MB |

El ajuste incluye las regiones y los factores, y se hace una vez por origen y mercado. Con 256 activos, la población de la criba, cuesta menos de una décima de segundo y no merece optimización. Con 4.200 activos domina la correlación. Un perfil aparte con la misma forma midió 0,12-0,18 s en cada uno de los tres productos de matrices y entre 2 y 4 s en cada pasada elemento a elemento sobre las matrices de 4.200 × 4.200 (raíz, división y recorte), con la misma presión de memoria. El enlace medio de SciPy tardó 0,7-0,8 s. Esas pasadas y el pico de 1,07 GB se podrían reducir con operaciones en el sitio. No se ha hecho porque la criba no usa esa población.

No se midieron la GPU, la lectura de Parquet ni la energía.

## Mutaciones dirigidas

Se aplicaron 20 mutaciones a la lógica crítica y se revirtieron después. Todas hacen fallar alguna prueba. Cinco sobrevivían al principio y motivaron pruebas nuevas:

- Exigir más madureces comunes que el mínimo en un par. Ahora un par con exactamente m madureces comunes debe aportar su correlación.
- Usar el enlace simple. La prueba compara con una referencia directa de SciPy sobre residuos sin estructura, donde los dos enlaces reparten distinto.
- Numerar las regiones por su último activo, que la misma referencia distingue.
- Guardar la menor madurez de G en lugar de la mayor. Ahora una emisión entre ambas se rechaza.
- Cortar el historial en la última emisión de G en lugar de la primera. Ahora etiquetas extremas que maduran dentro de G no cambian las regiones.

Las demás cubren el corte de madurez, el extremo de la ventana, el mínimo de etiquetas, los residuos constantes, el centrado, el orden de la numeración, la región 0 en la permutación, la semilla, el orden de los sectores, los pesos de sesión, el factor del mercado para activos nuevos, la comprobación de madurez en la emisión, las madureces duplicadas y la exclusividad del control sectorial.

## Pendiente

- Aplicar la etapa en el análisis de la criba sobre los Parquet de cada trabajo (#746).
- Cualquier evaluación con datos, bloqueada mientras rija el bloqueo de aprendizaje.
