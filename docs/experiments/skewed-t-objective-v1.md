# t de Student asimétrica con β-NLL

Este documento describe L2 de la segunda ola de variantes de [#782](https://github.com/GonxKZ/mars-titan/issues/782), que se implementa en [#787](https://github.com/GonxKZ/mars-titan/issues/787). `skewed_t_objective_v1` es un componente opcional de `FinancialConfig` (`skewed_t_objective`) que añade a la cabeza escalar una escala, una asimetría y unos grados de libertad por fila y sustituye la pérdida del ajuste por la NLL de una t de Student asimétrica. Es un objetivo de salida del marco común de `output_objectives`, como F4, F1 y A4 de [#749](https://github.com/GonxKZ/mars-titan/issues/749), y no se puede declarar a la vez que ellos.

Está integrada en el predictor, en el recorrido cronológico y en la escritura de predicciones. Se ha comprobado con pruebas en CPU sobre fixtures. La comprobación CUDA está pendiente porque la GPU está ocupada. No se ha entrenado ni evaluado con datos financieros y nada de lo que sigue es una mejora observada.

## Motivación

F1 ([escala de Laplace](laplace-scale-head-v1.md)) aprende una escala por fila con una distribución simétrica de colas fijas. Los residuos diarios suelen tener colas más pesadas que las de una Laplace y caídas que no son el reflejo de las subidas. L2 deja libres el peso de las colas (ν) y la asimetría (ξ) para contrastar si esa forma mejora la predicción emitida y sus niveles con las mismas filas, la misma ponderación y el mismo recorrido que F1. Michańków (2025) compara redes que predicen los parámetros de la normal, la t y la t asimétrica en seis índices bursátiles e informa de que la LSTM con la t asimétrica es la mejor en varios de sus criterios. Ese resultado es suyo, con otras series, otras métricas y otra arquitectura, y aquí no se hereda.

## Ecuaciones

Sea h el estado codificado que ya recibe la cabeza escalar y μ su salida. Tres proyecciones lineales de h dan

$$\sigma=\operatorname{softplus}(a_\sigma)+\sigma_{\min},\qquad \xi=\xi_{\max}^{\tanh(a_\xi)},\qquad \nu=\nu_{\min}+\operatorname{softplus}(a_\nu),\qquad a=Wh+c.$$

Con z = (y − μ)/σ, la densidad es la de la ecuación (7) de [Michańków (2025)](https://arxiv.org/abs/2508.18921v2), que parte la t de Student estándar t_ν en cero con la transformación de Fernández y Steel:

$$f(y)=\frac{2}{\sigma(\xi+1/\xi)}\begin{cases}t_\nu(\xi z), & z<0,\\ t_\nu(z/\xi), & z\ge 0.\end{cases}$$

La masa por debajo de μ es 1/(1 + ξ²), así que ξ > 1 alarga la cola derecha, ξ < 1 la izquierda y ξ = 1 da la t simétrica. μ es la moda de la distribución. Con x = ξz si z < 0 y x = z/ξ si z ≥ 0, la NLL por fila es la de la ecuación (16) del artículo:

$$\operatorname{NLL}=\log(\xi+1/\xi)-\log 2+\log\sigma-\log t_\nu(x).$$

El ajuste la pondera en cada fila con la escala sin gradiente, como la β-NLL de [Seitzer et al. (2022)](https://arxiv.org/abs/2203.09168v2) que usa F1:

$$\mathcal L=\frac1N\sum_i \operatorname{sg}(\sigma_i)\,\operatorname{NLL}_i.$$

Por fila, antes de dividir por N, el gradiente de la posición es

$$\frac{\partial\mathcal L}{\partial\mu}=-s\,\frac{(\nu+1)\,x}{\nu+x^2},\qquad s=\begin{cases}\xi, & z<0,\\ 1/\xi, & z\ge 0.\end{cases}$$

Es la puntuación de la t, que no depende de la escala de la fila y está acotada por s(ν + 1)/(2√ν). Para errores pequeños crece como el del error cuadrático y para errores grandes decae como 1/x, así que una fila muy alejada influye menos en μ que con el error absoluto, cuyo gradiente es siempre ±1. Sin el peso sg(σ), el gradiente de μ llevaría un factor 1/σ que daría más peso a las filas con escala pequeña, que es lo que la β-NLL corrige. Con las demás salidas fijas, el peso positivo no cambia los ceros de la derivada de escala por fila. Eso no garantiza el mismo óptimo de una red con parámetros compartidos.

Los pesos W empiezan en cero y los sesgos en softplus⁻¹(`initial_scale` − σ_min), 0 y softplus⁻¹(`initial_dof` − ν_min), de modo que todas las filas empiezan con σ = `initial_scale`, ξ = 1 y ν = `initial_dof`. A diferencia de F1, el gradiente de μ ya difiere del control L1 en la inicialización, porque la puntuación de la t no es el signo del error. Con W = 0 el tronco solo recibe ese gradiente de μ. El de σ, ξ y ν le llega cuando W deja de ser cero. Las pruebas comprueban los dos casos.

La declaración usa σ_min = 10⁻⁶ y `initial_scale` = 0,01 en unidades del residuo diario, los mismos valores que F1, con ν_min = 2, `initial_dof` = 5 y ξ_max = 3. Son valores declarados, no ajustados con datos.

## Punto y cuantiles emitidos

La predicción emitida es la mediana de la distribución ajustada. F1 emite la mediana de su Laplace, que es su μ, y la cabeza de cuantiles emite su nivel central. El control L1 estima también la mediana condicional. Con la mediana, el contraste entre L2 y F1 cambia la familia de la distribución y no el funcional del punto. El artículo no dice qué punto informa y observa que la asimetría separa la media de la moda.

Con `rows`, el recorrido escribe además los cuantiles exactos de la distribución en los cinco niveles de `quantile_head_v1`. Con T_ν⁻¹ la cuantílica de la t estándar,

$$z_\tau=\begin{cases}T_\nu^{-1}\!\left(\tau(1+\xi^2)/2\right)/\xi, & \tau<1/(1+\xi^2),\\ \xi\,T_\nu^{-1}\!\left(1-(1-\tau)(1+\xi^2)/(2\xi^2)\right), & \tau\ge 1/(1+\xi^2),\end{cases}\qquad q_\tau=\mu+\sigma z_\tau.$$

La mediana emitida es q₀,₅, el mismo valor que el nivel central de la fila. PyTorch no tiene la cuantílica de la t, así que `skewed_t_quantiles` usa `scipy.special.stdtrit` en float64 sobre la CPU. En 2.000 pares aleatorios de ν y nivel, la distribución de SciPy recupera el nivel pedido con un error absoluto máximo de 9,7·10⁻¹⁴. Con ξ = 1, `stdtrit(ν, 0,5)` vale exactamente cero y la mediana coincide bit a bit con μ.

La pérdida solo usa las salidas [μ, σ, ξ, ν], así que el punto emitido no necesita gradiente. A cambio, cada preparación copia las salidas a la CPU para calcular la mediana. En GPU eso supone una sincronización por bloque, cuyo coste aún no se ha medido. El objetivo declara `scalar_point` falso, porque su punto ya no es la cabeza escalar sin transformar, y el lector episódico lo rechaza como padre.

## Desviaciones y límites

- El artículo no declara las activaciones de su capa de salida (sección 3.4). softplus, tanh y las cotas σ_min, ξ_max y ν_min son decisiones propias. El artículo solo pide ν > 0. Aquí ν > 2 permite una varianza finita, como exige su ecuación (10). En coma flotante, la suma de softplus puede redondear al suelo. El resultado se acota además por ν_min·(1 + ε), con ε del tipo de cálculo. La identidad declara esta protección numérica y la inicialización conserva sus valores anteriores.
- El artículo parametriza la escala con σ² y aquí con σ. La distribución es la misma.
- La β-NLL no está en el artículo de Michańków. Seitzer et al. la plantean para la gaussiana con el peso sg(σ^{2β}). Aquí el peso es sg(σ), el que quita a μ el factor de la escala de cada fila, igual que sg(b) en F1. Es una adaptación propia.
- La NLL conserva la constante −log 2 de la ecuación (16). No cambia ningún gradiente.
- σ, ξ y ν comparten el estado codificado con μ. Unos parámetros con su propio tronco serían otro contraste.
- L2 se rechaza con la cabeza de cuantiles, con una receta distinta de `mae`, con destilación, en el postentrenamiento y en el ajuste Dust inicial, que no tienen una pérdida declarada para ella. También la rechaza el lector episódico.
- El artículo evalúa con LPS, CRPS, PIT y VaR sobre índices. La comparación de este proyecto usa el residuo por sesión y las métricas de la criba.

## Correspondencia entre ecuaciones, módulo y prueba

| Ecuación o regla | Módulo | Prueba |
| --- | --- | --- |
| La densidad (7) integra uno y deja 1/(1 + ξ²) por debajo de μ | `models/titans/skewed_t_objective.py` | `test_the_density_integrates_to_one_with_the_declared_mass_below_the_location`, con `scipy.integrate.quad` |
| NLL (16) = −log f | `skewed_t_nll` | `test_the_nll_is_the_negative_log_density_of_equation_seven`, contra `scipy.stats.t` |
| Gradientes de la NLL completa | `skewed_t_nll` | `test_the_full_nll_gradients_match_finite_differences`, con `gradcheck` en FP64 |
| β-NLL con el peso fijo | `skewed_t_beta_nll` | `test_the_beta_nll_holds_the_scale_weight_fixed` |
| ∂L/∂μ cerrado y su cota | `skewed_t_beta_nll` | `test_the_location_gradient_is_the_bounded_score_of_the_t` |
| σ, ξ y ν en sus rangos, W = 0 al empezar | `SkewedT` | `test_parameters_start_at_their_declared_values_without_global_rng`, `test_the_parameters_stay_inside_their_declared_ranges` |
| Cuantiles q_τ, que invierten la distribución | `skewed_t_quantiles` | `test_quantiles_invert_the_distribution_function`, `test_symmetric_quantiles_are_those_of_the_student_t` |
| El punto emitido es la mediana ajustada | `SkewedT.forward` | `test_the_emitted_point_is_the_median_of_the_fitted_distribution` |
| La cabeza recibe la puntuación de la t y el tronco solo el gradiente de μ con W = 0 | `FinancialPredictor._prepare` | `test_gradients_reach_the_three_projections_and_the_location_gets_the_t_score` |

Las pruebas del predictor están en `tests/models/titans/test_skewed_t_objective.py` y las del recorrido en `tests/training/test_financial_run_skewed_t.py`. Con un optimizador que solo registra gradientes, el recorrido comprueba que cada actualización usa la β-NLL con cuatro salidas por etiqueta y que el gradiente del sesgo de la cabeza escalar es la suma de las puntuaciones de las filas que maduran en ella, con W nulo y sin anular. También comprueba que la repetición por bloques reproduce los gradientes, que alterar el sufijo futuro del corpus no cambia los gradientes anteriores al corte, que cada fila emitida lleva como predicción la mediana de sus cinco niveles con intervalos asimétricos y que la identidad del recorrido registra el objetivo.

## Paridad, estado y recuperación

Sin declarar `skewed_t_objective`, `FinancialConfig.identity()` no cambia, el predictor no crea el módulo y `PreparedDecisions.outputs` vale None. Con L2, la identidad registra `output` y `output_objective` como cualquier objetivo de salida, los parámetros de la referencia conservan sus valores iniciales con la misma semilla, el generador global no avanza y la predicción inicial coincide bit a bit con la del control, porque ξ = 1. `PreparedDecisions.outputs` contiene [μ, σ, ξ, ν] por fila. Solo se añaden `output_objective.weight` y `output_objective.bias`, 195 parámetros con H = 64, que pertenecen al papel `shared` del optimizador. No hay estado por flujo. Los parámetros viajan en el `state_dict` del checkpoint y la prueba de ida y vuelta reproduce las salidas y las medianas tras cargarlo.

## Controles

La [declaración](../../configs/experiments/skewed-t-objective-v1.json) parte de la receta histórica con máscaras, con la cabeza escalar y la pérdida `mae`. Tiene tres modelos `mac_online`, desactivados por defecto y sin acceso al test final:

| Modelo | Papel | Cambio del predictor |
| --- | --- | --- |
| `absolute_error_control` | Referencia L1 | Ninguno |
| `laplace_scale_head` | Control | F1 con los valores de su propia declaración |
| `skewed_t_objective` | Variante | L2 con los valores de este documento |

El contraste principal es L2 frente a F1, que comparten ponderación, recorrido y funcional del punto. L2 se descarta si no mejora a F1 en ese contraste. El contraste con el control L1 se informa aparte. Los niveles emitidos de L2 y F1 se compararán además con los de la referencia de cuantiles B de la criba (`quantile_reference`, con su cabeza y pinball), con la pinball media y la cobertura de los intervalos centrales del 80 % y el 95 %. El registro de L2 en el catálogo de la criba, con su métrica y su presupuesto, está pendiente, como el de las demás piezas de la segunda ola.

## Coste medido

La medida CPU del 30 de octubre de 2026 recorrió 8 instantes de 256 activos con las formas de la edición v3.1, cuatro hilos, un calentamiento y siete repeticiones con el orden de los modelos rotado, sin optimizador. Al terminar, ningún parámetro había cambiado. La carga del equipo estuvo entre 9 y 15, con otras pruebas en marcha.

| Modelo | Parámetros | Avance con grafo (s) | Retropropagación (s) | Inferencia (s) | Pico del asignador |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control L1 | 185.859 | 0,783 | 0,481 | 0,741 | 504,9 MB |
| F1 | 185.924 | 0,780 (×0,997) | 0,482 (×1,001) | 0,748 (×1,010) | +32 KB |
| L2 | 186.054 | 0,775 (×0,990) | 0,489 (×1,016) | 0,756 (×1,021) | +94 KB |

Son medianas. Las diferencias de tiempo frente al control no superan el 2,1 % y quedan dentro de la dispersión entre repeticiones (el avance del control osciló entre 0,749 y 0,854 s). L2 añade 195 parámetros, tres proyecciones de H = 64 a 1 con sus sesgos. El avance incluye la mediana calculada con SciPy. Los cuantiles de los niveles se calculan al escribir las predicciones y no entran en esta medida. El [recibo](../../reports/engineering/skewed-t-objective-20261030/cost-cpu.json) conserva las muestras, las versiones y la huella de las fuentes. Para repetirla:

```bash
CUDA_VISIBLE_DEVICES=-1 uv run python -m benchmarks.objective_variants_cost \
  --rows 256 --instants 8 --repetitions 7 --warmup 1 --threads 4 \
  --models reference_mae laplace_scale_head_v1 skewed_t_objective_v1 \
  --output "${MARS_SKEWED_T_COST:?}"
```

## Comprobación y trabajo pendiente

La [verificación integrada](../../reports/engineering/skewed-t-objective-20261030/verification.json) registra 510 casos CPU aprobados sin omisiones y 33 casos CUDA deseleccionados. Incluye ecuaciones, gradientes, filas emitidas, recuperación, consumidores, ventanas, el registro común y el módulo de determinismo. El módulo L2 cubre 98/98 sentencias y 20/20 ramas. Se midieron las 20 funciones cambiadas de producción, con CRAP máximo 32,62. Ruff, formato y el comprobador pasan. La regresión histórica conserva 2.051 aprobados y 42 omitidos en training, y 5.872 aprobados, 124 omitidos y un fallo del fixture de determinismo en el resto. Ese fallo se corrigió en #819 y su módulo pasa en la selección integrada. No se presenta la regresión completa como repetida.

Ocho casos nuevos reprodujeron un redondeo de los grados de libertad al suelo, también a ν = 2, en FP32 y FP64. La protección ν_min·(1 + ε) corrige ese extremo y se declara en la identidad. La revisión de la ponderación limita la afirmación sobre los ceros de la derivada a cada fila con las demás salidas fijas. No se deduce de ella el mismo óptimo de los parámetros compartidos.

La misma regresión declara después ocho casos de saturación en `cuda:0`. La recogida comprueba que este alcance tiene ahora 41 casos CUDA pendientes, incluidos los 33 anteriores. El módulo repite 45 casos CPU aprobados con nueve CUDA deseleccionados. La función nueva de prueba recorre sus 9/9 sentencias en CPU y tiene CRAP 5. Esta medida no comprueba la ejecución en GPU.

Las 52 [mutaciones dirigidas iniciales](../../reports/engineering/skewed-t-objective-20261030/mutations.json) se detectaron con referencias verdes antes y después de cada una, sin intentos de optimizador y con la protección intacta. Cubren configuración, identidad, inicialización, densidad, peso detenido de la β-NLL, inversión de la distribución, punto emitido, consumidores y coste. Dos recetas que actúan en módulos modificados por `develop` se repitieron sobre la integración y sustituyen sus resultados anteriores. Otras dos detectan la retirada de la cota numérica o de su identidad. También se ejecutó la eliminación de L2 de la lista opcional de Dust. Sobrevive porque el módulo `output_objective` sigue activando la segunda guarda en los modelos admitidos. Se declara equivalente bajo ese contrato, sin contarla como detectada ni atribuir una puntuación global de mutación.

La [medida posterior a la corrección](../../reports/engineering/skewed-t-objective-20261030/cost-cpu-numerical-floor.json), en `202175bb2`, mantiene 256 activos, ocho instantes y siete repeticiones, con un calentamiento y dos hilos. Las medianas en segundos para el recorrido completo son:

| Modelo | Avance con grafo | Retropropagación | Inferencia |
| --- | ---: | ---: | ---: |
| Control L1 | 2,093 | 0,907 | 1,867 |
| F1 | 1,936 | 1,011 | 1,596 |
| L2 | 1,945 | 0,995 | 1,837 |

El proceso completo duró 144,84 s y alcanzó 1.850.544 KiB de RSS. La medida interna dura 135,54 s e incluye los tres modelos y su preparación. La carga media pasó de 10,43 a 17,74 con la regresión CPU y otras aplicaciones activas. Cambian los hilos y la carga respecto a la primera pasada, así que los tiempos no acreditan una aceleración ni aíslan el coste de la cota. No son latencias de una predicción individual. Ningún parámetro cambió y no hubo optimizador. El perfilador consultó CUDA, ocultada para esta medida CPU, y registró el código 100. Recogió los eventos CPU y el proceso terminó correctamente. La ruta GPU sigue sin comprobarse.

Queda pendiente:

- La comprobación en `cuda:0`, que espera a que la GPU quede libre. La prueba de paridad CUDA de los objetivos de salida ya recorre F1 y L2.
- El coste de la sincronización por bloque en GPU.
- El registro en el catálogo de la criba.
- Cualquier evaluación predictiva, que sigue bloqueada.
