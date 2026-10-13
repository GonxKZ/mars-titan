# Propuestas arquitectónicas posteriores a Titans

Estado: propuestas con revisión adversarial, pilotos diseñados y predicciones escritas antes de ver ningún dato. No hay resultados predictivos. El bloqueo de aprendizaje impide ajustar, pilotar o evaluar cualquiera de ellas. La revisión bibliográfica que las motiva está en [memoria posterior a Titans](post-titans-memory-review.md), la derivación matemática de PT1 en el [certificado de contracción](titans-memory-certificate.md), los diagnósticos en la [especificación de diagnósticos](memory-diagnostics.md) y la búsqueda de precedentes en el [registro de novedad](novelty-ledger.md#búsqueda-de-precedentes-del-9-y-10-de-octubre-de-2026).

Cada propuesta sigue el mismo esquema: propiedad de los datos que la motiva, mecanismo y propiedad que garantiza, separación entre fuente publicada y derivación propia, punto de inserción, control emparejado y alternativa trivial, métrica y regla de decisión declaradas de antemano, coste dentro de las cuatro semanas de GPU, riesgos, piloto con datos reales, predicción escrita y revisión adversarial. Una propuesta que no tiene una propiedad de los datos a la que responder se descarta aunque sea interesante.

## Propiedades del problema que gobiernan las decisiones

Estas propiedades proceden del protocolo, del objetivo residual y de la literatura financiera revisada. No se han medido con datos nuevos durante esta revisión.

| Propiedad | Evidencia | Consecuencia de diseño |
| --- | --- | --- |
| Relación señal/ruido muy baja | Con 18,1 millones de observaciones diarias de EE. UU., el mejor R² fuera de muestra está entre −0,03 % y +0,01 % (Rahimikia et al., 2025). | Capacidad adicional aumenta varianza antes que señal. Las propuestas deben acotar el daño posible y compararse con controles del mismo coste. |
| Colas pesadas y saltos | Resultados trimestrales, crisis y límites de precio en China. | Una sola sesión no debería mover sin límite el estado rápido ni una corrección aprendida. |
| No estacionariedad y cambios de régimen | Deterioro temporal documentado por Rahimikia et al. (2025) y reentreno periódico en Guijarro-Ordonez et al. (2026). | Adaptación con olvido explícito y diagnóstico de recuperación tras eventos fechados de antemano. |
| Estructura transversal | El objetivo resta el mercado con una beta de 252 sesiones, pero sectores y estilos siguen correlacionados. | Las etiquetas de una misma sesión no son evidencias independientes. Hoy el núcleo no tiene interacción entre activos. |
| Etiquetas que maduran con retraso | El objetivo de $t$ se conoce al cierre de $t+1$ y las escrituras maduras esperan al cursor confirmado. | Toda adaptación con etiquetas debe usar solo resultados maduros y medir su sensibilidad al retraso. |
| Modalidades ausentes con máscara | Noticias dispersas, fundamentales trimestrales y huecos de la edición histórica desde 2000. | La fusión recibe presencia explícita. Cualquier cambio debe conservar la equivalencia con todas las modalidades presentes. |
| Presupuesto fijo | 4 semanas de RTX 4070 de 8 GB (unas 672 h) ya comprometidas en gran parte por la campaña A. La hipótesis de la auditoría, sin medir, es de unas 85 h por familia, configuración y semilla para 19 ventanas. | Las propuestas del núcleo deben costar poco o reutilizar predicciones ya emitidas. |

## Niveles

**Núcleo de campaña.** PT1, PT2 y PT3 caben en el presupuesto con un control emparejado y tienen una regla de decisión cerrada. PT2 y PT3 trabajan sobre predicciones ya emitidas y su coste es de CPU. PT1 exige reentrenar Titans y se pilota antes de cualquier ampliación.

**Línea posterior.** PT4 a PT7 tienen un plan de piloto pero no entran en la campaña de cuatro semanas. PT8 recoge las ideas descartadas y su motivo.

---

## PT1. Memoria de Titans acotada y contractiva

**Datos que la motivan.** Colas pesadas, cambios de régimen y entrenamiento con truncación 8 de una memoria que se usa durante cientos de sesiones. Con las puertas actuales, una secuencia admisible de puertas puede amplificar el estado aunque cada paso sea estable (Proposición 3 del certificado).

**Mecanismo.** Dos piezas desactivables por separado dentro de `NeuralMemory`:

1. Puertas en una caja certificada: $\alpha=\alpha_{lo}+(1-\alpha_{lo})\,\sigma(A x)$, $\eta=\eta_{hi}\,\sigma(e x)$ y $\theta=\theta_{max}\,\sigma(t x)$ con $(\alpha_{lo},\eta_{hi},\theta_{max})=(1/500,\,3/10,\,1/10)$.
2. Escritura interna recortada: cada fila del gradiente interno de cada capa se recorta a norma $G$. Con una capa y clave unitaria equivale a la pérdida de Huber con $\delta=G/2$.

**Propiedad garantizada.** Con una capa, el Teorema 4 da contracción uniforme $\rho=0{,}9997$ en la norma $P_0\otimes I$ para cualquier secuencia de claves y puertas, y el Teorema 6 da contracción incremental con la escritura recortada. Con cualquier profundidad, la Proposición 5 acota el estado: $\lVert S\rVert\le\theta_{max}G/(1-\eta_{hi})$ por fila y $\lVert W\rVert\le\max(\lVert W_0\rVert,\,\theta_{max}G/((1-\eta_{hi})\alpha_{lo}))$. La memoria de dos capas con LayerNorm residual no tiene certificado de contracción y se diagnostica con D5.

**Fuente y derivación.** Ecuaciones 1 a 3 de Titans (Behrouz et al., 2025). La pérdida de Huber procede de Yaad en MIRAS (Behrouz et al., 2026) y la acotación por olvido de la modificación σ (Ioannou y Kokotovic, 1984). Las funciones de Lyapunov cuadráticas comunes y las LMI en vértices son estándar (Boyd et al., 1994). Son derivación propia la reducción a bloques 2 × 2, la condición necesaria y suficiente, las cajas certificadas, el contraejemplo y la contracción incremental con recorte.

**Inserción.** `src/mars_titan/models/titans/config.py` (nueva extensión de `MemoryConfig` con identidad propia que solo aparece si se declara), `neural_memory.py` (cálculo de puertas y recorte tras `autograd.grad`), `financial.py` (campo de `FinancialConfig`) y la receta de campaña correspondiente. Desactivada, la salida debe ser idéntica bit a bit a la actual.

**Control emparejado.** Titans-MAC de la receta con las mismas semillas, filas, ventanas y presupuesto. **Alternativa trivial.** Reducir `theta_max` y fijar `gate_bias` con $\eta$ bajo, sin caja ni recorte. Si esa alternativa iguala a PT1 en error y diagnósticos, la caja no aporta nada que no dé un paso menor.

**Métrica y regla de decisión.** No inferioridad. Métrica primaria: diferencia relativa de MAE residual por sesión, PT1 menos control, emparejada, con bootstrap circular por bloques de 16 sesiones sobre las ventanas de validación. Se conserva PT1 si el límite superior del intervalo del 95 % es menor o igual que +0,2 % y además, en los eventos de D2, el percentil 95 del MAE por sesión no empeora más de un 1 %. Criterios secundarios, que no deciden por sí solos: exponente de Lyapunov máximo estimado en D5 y norma máxima del estado. Con el [control de varianza](#pt7-ensemble-de-semillas-como-control-de-varianza-protocolo), PT1 solo se conserva si además la diferencia mínima interpretable entre semillas del control es menor que +0,2 %, y el abandono por el límite de +0,5 % exige una banda por debajo de ese límite. La alternativa trivial iguala a PT1 en error si el intervalo de su diferencia relativa queda dentro de la banda.

**Coste.** Una configuración más de Titans por semilla. El sobrecoste por paso debe medirse antes de lanzarla. Con la hipótesis de 85 h por semilla y 19 ventanas, una semilla completa ocuparía el 13 % de las cuatro semanas. Por eso PT1 se pilota primero.

**Riesgos.** $\eta_{hi}=0{,}3$ reduce la persistencia del momentum. $\alpha_{lo}=1/500$ limita la semivida a unas 346 sesiones. Si la señal útil de la memoria dependía de amplificaciones transitorias, PT1 la eliminaría, y eso también sería un resultado.

**Piloto con datos reales.** Hasta 64 activos de EE. UU. con todas las modalidades disponibles, ventanas de validación 2018, 2019 y 2020 (la última incluye la caída de marzo de 2020), entrenamiento expansivo desde 2000, una semilla para PT1, otra para su alternativa trivial y tres para el control con el mismo número de actualizaciones (la 42 se empareja con PT1, y la 43 y la 44 forman la banda de PT7). Presupuesto máximo declarado: 41 h de GPU para PT1, su alternativa trivial y la semilla 42 del control, más las 27 h de la reserva de PT7 para las otras dos, según la [medida del control](#pt7-ensemble-de-semillas-como-control-de-varianza-protocolo). Métricas: las de la regla más D1, D2 y D5. Éxito: no inferioridad en las tres ventanas y ningún empeoramiento del exponente de Lyapunov. Abandono: límite superior por encima de +0,5 % en el piloto, o D2 muestra que la variante online se comporta igual que la congelada.

**Predicción escrita de antemano.** El MAE no cambiará de forma detectable (diferencia relativa entre −0,2 % y +0,2 %). La norma máxima del estado bajará al menos a la mitad en el evento de 2020. El exponente de Lyapunov estimado del control será negativo en la mayor parte de las sesiones, de modo que la garantía de PT1 tendrá más valor como seguro que como mejora media.

**Revisión adversarial.** El contraejemplo más fuerte es que el control ya sea estable en los datos: las puertas aprendidas podrían no visitar nunca la región peligrosa y PT1 solo añadiría restricciones. El diseño lo distingue con D5 y con la fracción de pasos fuera de la caja en D2. Si esa fracción es casi nula, PT1 se registra como garantía sin efecto observable. La segunda objeción es que la garantía cubre la memoria lineal y la campaña usa dos capas. Se responde con la cota de estado, que sí cubre las dos capas, y con D5. La tercera es que un $\theta$ pequeño haga lo mismo. Para eso está la alternativa trivial.

---

## PT2. Calibración conformal en línea con etiquetas maduras

**Datos que la motivan.** No estacionariedad y retraso de las etiquetas. La calibración CQR actual se ajusta una vez en la partición de calibración y queda congelada, así que no puede seguir un cambio de volatilidad dentro de un año. Su registro ya declara `coverage_guaranteed=False` porque la dependencia temporal rompe la intercambiabilidad.

**Mecanismo.** Para cada mercado $g$ y cada intervalo central (80 % y 95 % con los cuantiles 0,025, 0,1, 0,9 y 0,975), se sigue en línea el cuantil de la puntuación CQR $E_i=\max(q_{lo,i}-y_i,\,y_i-q_{hi,i})$. La corrección vigente $Q_{g}$ ensancha o estrecha el intervalo emitido, $[q_{lo}-Q_g,\,q_{hi}+Q_g]$, y se guarda con cada predicción. Cuando madura la cohorte $m$ de un mercado, con errores $\mathrm{err}_i=\mathbf{1}\{y_i\notin[q_{lo,i}-Q_{g,i},\,q_{hi,i}+Q_{g,i}]\}$ calculados con la corrección realmente emitida,

$$
Q_{g}\leftarrow Q_{g}+\gamma\,\big(\overline{\mathrm{err}}_m-a\big),\qquad a=1-\text{cobertura nominal}.
$$

La mediana no cambia, así que el MAE tampoco.

**Propiedad.** Si las puntuaciones están acotadas por $B$ y la madurez retrasa como mucho $d$ cohortes, la cobertura media sobre $M$ cohortes cumple $\big|\frac1M\sum_m\overline{\mathrm{err}}_m-a\big|\le\frac{2B+\gamma(1+d)}{\gamma M}$. Es la garantía de largo plazo del seguimiento de cuantiles (Angelopoulos et al., 2023), con el término de retraso de las versiones multipaso (Wang y Hyndman, 2024, y Szabadváry, 2024) y promediada por cohortes. Es una garantía marginal a largo plazo, no condicional por activo ni por sesión.

**Fuente y derivación.** ACI (Gibbs y Candès, 2021), seguimiento de cuantiles y control PID conformal (Angelopoulos et al., 2023), regla con errores maduros de Wang y Hyndman (2024) y CQR (Romano et al., 2019). Es una adaptación. La parte propia es la actualización por cohortes de mercado con la corrección emitida guardada en cada predicción y su cota, una variante menor.

**Inserción.** Nuevo módulo `src/mars_titan/calibration/online_conformal.py` junto a `conformal_quantiles.py`, con estado por mercado exportable y cursor de cohorte. La comparación (`evaluation/walk_forward_comparison.py`) declararía un modelo de calibración en línea frente a la CQR estática existente. Sin el modelo, las salidas no cambian.

**Control emparejado.** CQR estática actual con las mismas predicciones. **Alternativa trivial.** Recalibrar la CQR estática una vez al año con la ventana de calibración más reciente. Si iguala a PT2, la adaptación dentro del año no aporta.

**Métrica y regla de decisión.** Primaria: desviación absoluta media de la cobertura respecto al nominal por año y mercado, promediando los intervalos de 80 % y 95 %. Se conserva PT2 si reduce esa desviación al menos un 25 % frente a la CQR estática con intervalo del 95 % que excluye cero, y el interval score emparejado por sesión no empeora más de un 1 %. Abandono si el interval score empeora más de un 1 % o si la desviación no baja.

**Coste.** Solo CPU sobre predicciones ya emitidas. Minutos por ventana. Sin GPU.

**Riesgos.** $\gamma$ es un hiperparámetro y debe fijarse antes de mirar el test, en la partición de calibración de cada ventana. Una corrección por mercado no arregla una mala cobertura concentrada en activos pequeños. Un $\gamma$ grande produce intervalos que oscilan.

**Piloto con datos reales.** Sobre las predicciones de validación ya emitidas por la campaña (todas las ventanas de validación, todos los activos), sin reentrenar nada. $\gamma\in\{0{,}005,\,0{,}02\}$ declarados de antemano, elegido el mejor en la primera mitad del periodo de validación y evaluado en la segunda. Éxito y abandono como la regla.

**Predicción escrita de antemano.** La CQR estática quedará infracubierta en 2008 y 2020 y sobrecubierta en años tranquilos. PT2 reducirá la desviación media de cobertura entre un 30 % y un 60 %. El interval score mejorará poco (menos de un 2 %) porque la anchura óptima la fija sobre todo el cuantil base, no la corrección.

**Revisión adversarial.** El contraejemplo más fuerte es una secuencia en la que la volatilidad cambia más rápido que la madurez de las etiquetas. Con retraso de una sesión, PT2 siempre llega tarde a un salto y lo corrige después. La garantía es de largo plazo y no protege la sesión del salto. La alternativa trivial de recalibración anual separa lo que aporta la adaptación dentro del año. La evidencia externa es favorable pero no financiera en su mayor parte: Manokhin (2026) observa en 2.217 series que los métodos adaptativos en línea superan a los estáticos en un 9 a 33 % de Winkler relativo, y Gibbs y Candès (2024) la aplican a volatilidad bursátil.

**Implementación (10 de octubre de 2026).** Implementada y comprobada sin entrenar en [su documento](../engineering/online-conformal-calibration.md). Dos correcciones de esta propuesta quedan fijadas antes de ver datos. La tasa no tenía unidad: ahora γ = κ B̂, con B̂ la mayor puntuación absoluta de calibración del mercado y el intervalo, y {0,005, 0,02} son los valores de κ. La alternativa trivial de recalibrar una vez al año coincide con el control, porque la CQR estática ya se ajusta en cada ventana anual, así que pasa a ser una recalibración mensual con filas maduras.

---

## PT3. Corrección madura bayesiana con ruido de cohorte correlacionado

**Datos que la motivan.** Señal muy débil, etiquetas que maduran por cohortes de miles de activos a la vez, correlación residual entre activos de una misma sesión y colas pesadas. B6 ya escribe una corrección lineal $A$ con resultados maduros mediante la regla delta o la proximal, pero ambas tratan cada etiqueta de la cohorte como evidencia independiente y usan un paso fijo. Su documento deja pendiente «RLS con olvido».

**Mecanismo.** Una tercera regla de B6, `kalman`, sobre el mismo contrato de claves, valores, cursor y madurez. El estado es la media $A\in\mathbb{R}^{d\times m}$ y la covarianza $P\in\mathbb{R}^{d\times d}$ compartida por las columnas. Antes de cada cohorte, paseo aleatorio $P^-=P+qI$. La cohorte de $n$ resultados con claves $K$ y valores $Y$ tiene ruido $\Sigma=\sigma^2[(1-\varrho)I+\varrho\mathbf{1}\mathbf{1}^\top]$, cuya inversa es cerrada:

$$
\Sigma^{-1}=\frac{1}{\sigma^2(1-\varrho)}\Big(I-g\,\mathbf{1}\mathbf{1}^\top\Big),\qquad g=\frac{\varrho}{1-\varrho+n\varrho}.
$$

La actualización en forma de información es

$$
P^{+}=\big((P^-)^{-1}+K^\top\Sigma^{-1}K\big)^{-1},\qquad
A^{+}=A+P^{+}K^\top\Sigma^{-1}\big(Y-KA\big),
$$

con $K^\top\Sigma^{-1}K=\frac{1}{\sigma^2(1-\varrho)}\big(K^\top K-g\,(K^\top\mathbf1)(K^\top\mathbf1)^\top\big)$, que cuesta $O(nd^2+d^3)$ con $d=64$. Con $\varrho=0$ se recupera RLS por bloques con paseo aleatorio. Con $\varrho\to1$ la media común de la cohorte cuenta como una sola observación. Un recorte de Huber opcional reescala las filas con residuo estandarizado grande, $w_i=\min(1,\,c/|e_i|)$, y la matriz ponderada conserva la forma cerrada. La lectura devuelve $k^\top A$ y su varianza predictiva $k^\top Pk+\sigma^2$.

**Propiedad.** La ganancia de cada dirección depende de la incertidumbre acumulada en esa dirección y no de un paso fijo. Con claves anisótropas, RLS converge en las direcciones poco excitadas donde LMS es lento. La correlación de cohorte limita el tamaño efectivo de una sesión a como mucho $1/\varrho$ observaciones de su componente común. Con $q>0$ la covarianza no colapsa y la regla sigue un parámetro que deriva.

**Fuente y derivación.** Filtro de Kalman (Kalman, 1960), modelos lineales dinámicos (West y Harrison, 1997) y efectos temporales aleatorios de datos de panel (Baltagi, 2021). Fentazi et al. (2026) muestran con un arnés sin fuga que un banco de RLS cerrado supera a tres de cuatro métodos publicados de adaptación con etiquetas retrasadas, a una fracción del coste. Es una adaptación. Son propios la combinación concreta para cohortes maduras de B6, la forma cerrada de la actualización ponderada y su contrato de madurez. La aproximación de aplicar la medida al estado actual y no al de la decisión se declara como desviación, con error acotado por el ruido de proceso acumulado durante el retraso.

**Inserción.** `src/mars_titan/memory/associative_memory.py` (regla `kalman` con identidad propia, exportación de $P$), `MatureCorrection` sin cambios de contrato y un modelo nuevo en `configs/titans/mars-titan-extensions.json`. Las reglas delta y proximal conservan su identidad y su salida.

**Control emparejado.** B6 proximal con las mismas claves, cohortes y predicciones base, y el núcleo sin corrección. **Alternativa trivial.** La misma regla con $\varrho=0$, que separa lo que aporta la correlación de cohorte de lo que aporta RLS.

**Métrica y regla de decisión.** Primaria: diferencia relativa de MAE por sesión de B6 Kalman frente a B6 proximal, emparejada y con bootstrap por bloques. Se conserva PT3 si el límite superior del 95 % es menor que cero, o si es menor o igual que +0,1 % y la cobertura del intervalo $\pm1{,}645$ desviaciones predictivas de la corrección está entre el 85 % y el 95 %. D3 debe conservar al menos la mitad de la ganancia con un retraso adicional de una sesión. Abandono si empeora más de un 0,1 % o si la ganancia desaparece con el retraso.

**Coste.** CPU sobre predicciones y claves ya emitidas. Con $d=64$ y cohortes de hasta 8.192 resultados, del orden de milisegundos por cohorte. Sin GPU.

**Riesgos.** $\sigma^2$, $q$ y $\varrho$ deben fijarse con la ventana de entrenamiento de cada etapa. Un $q$ grande persigue ruido. Con claves de rango deficiente, $P$ crece sin límite en las direcciones no excitadas si $q>0$. La implementación debe acotarla o declarar ese crecimiento.

**Piloto con datos reales.** Sobre predicciones de validación ya emitidas por Titans-MAC en todas las ventanas y activos, con las claves del codec. $\sigma^2$ y $\varrho$ estimados por momentos con los residuos de la ventana de entrenamiento y $q\in\{10^{-6},10^{-5}\}$ declarados. Éxito y abandono como la regla.

**Predicción escrita de antemano.** La corrección apenas cambiará el MAE (diferencia entre −0,1 % y +0,1 % frente al núcleo y frente a B6 proximal). $\varrho$ estimado quedará entre 0,02 y 0,15 porque la beta ya resta el mercado. Lo más probable es que PT3 solo aporte una varianza predictiva útil para la calibración, no un menor error.

**Revisión adversarial.** El contraejemplo más fuerte es la baja señal: si el residuo no tiene estructura lineal estable en las claves del codec, cualquier corrección persigue ruido y la mejor regla es no corregir. El control sin corrección lo detecta. La segunda objeción es que RLS con un paso bien elegido sea equivalente a la regla proximal. La alternativa con $\varrho=0$ y la comparación con proximal lo separan.

**Implementación (10 de octubre de 2026).** Implementada y comprobada sin entrenar en [su documento](../engineering/kalman-associative-memory.md). Antes de ver datos quedan fijadas tres precisiones. q pasa a expresarse como fracción de σ², q = κσ² con κ ∈ {10⁻⁶, 10⁻⁵}, porque la escala del residuo solo se conoce al estimar σ². La varianza inicial es p0 = 0,1 σ². El crecimiento de P en direcciones no excitadas se declara con su cota λ_max(P_t) ≤ p0 + qt en lugar de acotarlo.

---

## PT4. Memoria de mercado compartida (línea posterior)

**Datos.** Estructura transversal. El núcleo procesa cada activo como un flujo independiente y no tiene ningún mecanismo para que lo ocurrido en un activo informe a otro, salvo el banco episódico.

**Mecanismo.** Una segunda memoria neuronal por mercado, leída por todos los flujos con el estado de la sesión anterior y escrita una vez por sesión, después de emitir todas las predicciones, con la media de los gradientes internos de los flujos de esa sesión. No usa etiquetas, igual que la memoria de Titans.

**Fuente y derivación.** La memoria y su regla son las de Titans. La lectura compartida con escritura por gradiente medio de la sesión es propia. No encontramos precedente en la búsqueda del 9 de octubre de 2026 con las consultas del registro.

**Por qué no entra en la campaña.** Exige que todos los activos de una sesión estén en el mismo lote o acumular el gradiente entre lotes antes de escribir, y cambia la selección de estado por flujo. Es la propuesta más arquitectónica y la más costosa de implementar bien.

**Piloto.** 64 activos de EE. UU. en un solo lote, ventanas 2018 a 2020, control Titans-MAC con los mismos pesos lentos iniciales y memoria de mercado desactivada. Éxito: mejora de MAE con límite superior por debajo de −MID frente al control y frente al control sectorial, con la banda entre semillas de PT7. Abandono: una mejora que no supera la banda en alguno de los dos contrastes, o fuga detectada por la prueba de causalidad intrasesión. Los tres modelos se ajustan sin acumulación por bloques, así que el control de PT4 no es el del piloto común y ajusta sus propias semillas 43 y 44 con el mismo procedimiento que la 42. Presupuesto: 8 h de GPU y 27 h para esas dos semillas, estimadas con la medida del control común porque la receta sin acumulación no se ha medido.

**Predicción.** Mejora pequeña concentrada en días de eventos sectoriales, nula en promedio.

**Revisión adversarial.** Un factor sectorial explícito como entrada consigue lo mismo sin memoria. Ese control es obligatorio antes de atribuir nada a la memoria compartida.

**Implementación.** Implementada y comprobada sin entrenar como `session_context` (`market_memory_v1` y el control `sector_mean_v1` con las fichas de sectores de FinMultiTime), con lectura solo de sesiones confirmadas, escritura una vez por sesión independiente de la partición en bloques, arranque idéntico al control y la comparación declarada sin ejecutar. No admite todavía la acumulación por bloques de la receta base ([#456](https://github.com/GonxKZ/mars-titan/issues/456), [especificación](../engineering/market-memory.md)).

## PT5. Memoria multiescala (línea posterior)

**Datos.** No estacionariedad con varias escalas: eventos de días, ciclos de meses y regímenes de años. Una sola tasa de olvido por fila elige una escala.

**Mecanismo.** Cascada de variables acopladas con constantes de tiempo geométricas sobre los pesos rápidos (Benna y Fusi, 2016), que produce olvido aproximadamente de ley de potencias. Kaplanis et al. (2018) ya la usaron en aprendizaje por refuerzo continuo. Es una adaptación.

**Alternativa trivial obligatoria.** Banco de integradores exponenciales independientes con las mismas escalas y el mismo número de estados. Si iguala a la cascada, el acoplamiento no aporta.

**Piloto.** 64 activos, ventanas 2018 a 2020, tres modelos (control, cascada de tres niveles, banco EMA de tres escalas), 9 h de GPU. Éxito: la cascada mejora al banco EMA con límite superior por debajo de −MID, la banda entre semillas del control de PT7, que comparte el control, las filas y las ventanas. Abandono en otro caso.

**Predicción.** El banco EMA igualará a la cascada. Multiplica por tres el estado por flujo.

**Implementación.** Implementada y comprobada sin entrenar como `memory_multiscale` (`cascade` y la alternativa `ema_bank`), con g = 1/16, contracción uniforme demostrada dentro de la caja de PT1 y la comparación declarada sin ejecutar ([#457](https://github.com/GonxKZ/mars-titan/issues/457), [especificación](../engineering/multiscale-memory.md)).

## PT6. Fusión condicionada por la presencia de modalidades (línea posterior)

**Datos.** Modalidades ausentes con máscara. La fusión actual concatena las codificaciones multiplicadas por presencia y los bits de presencia, y aplica una capa lineal. Puede desplazar la salida según el patrón, pero no reescalar la contribución de una modalidad según la ausencia de otra.

**Mecanismo.** Modulación FiLM (Perez et al., 2018) de cada codificación con escala y desplazamiento que dependen solo del vector de presencia. Con todas las modalidades presentes debe reducirse a la identidad. Hay muchos precedentes en la literatura de modalidades ausentes. Es una adaptación.

**Alternativa trivial.** Una capa de fusión más ancha con el mismo número de parámetros.

**Piloto.** Requiere reentrenar. 64 activos, ventanas 2018 a 2020, 6 h de GPU. Métrica: MAE en las filas con alguna modalidad ausente. Éxito: la modulación mejora a la fusión más ancha con límite superior por debajo de −MID, con la banda de PT7 calculada en esas mismas filas. Abandono si la alternativa trivial la iguala, es decir, si la mejora no supera la banda. Además, en todas las filas FiLM no debe empeorar al control más de un 0,2 %, con la misma no inferioridad que PT1: conserva solo si la MID de todas las filas queda por debajo de +0,2 % y abandona si el límite superior pasa de +0,5 % con la MID por debajo de ese límite. El piloto usa los 64 activos y el control de PT7, así que comparte sus semillas. Con otro control necesitaría sus propias tres semillas.

**Predicción.** Sin diferencia detectable, porque el Transformer que sigue a la fusión ya puede modelar esas interacciones.

**Implementación.** Implementada y comprobada sin entrenar como `presence_modulation` (`film_v1` y la alternativa `wide_fusion_v1`), con identidad exacta en las filas completas, arranque idéntico al control y la comparación declarada sin ejecutar ([#458](https://github.com/GonxKZ/mars-titan/issues/458), [especificación](../engineering/presence-modulation.md)).

## PT7. Ensemble de semillas como control de varianza (protocolo)

No es arquitectura, pero condiciona la interpretación de todo lo anterior. Está declarado e implementado desde el 10 de octubre de 2026 ([#459](https://github.com/GonxKZ/mars-titan/issues/459)) y no se ha ejecutado.

**Datos.** Con señal tan débil, la varianza entre semillas puede ser del mismo orden que cualquier diferencia entre arquitecturas. Bouthillier et al. (2021) muestran que la inicialización basta para cambiar la conclusión de una comparación, y Rahimikia et al. (2025) y Gu et al. (2020) promedian semillas por ese motivo.

**Regla.** El control de los pilotos (`titans_mac_online` con la receta de 2000) se ajusta con las semillas 42, 43 y 44 sobre las mismas filas, ventanas y presupuesto, y cada semilla repite los dos casos de búsqueda y la selección con validación. La campaña A ajusta las semillas 43 y 44 solo con el caso elegido por la 42. Aquí eso daría a la 42 la ventaja de elegir el mejor de dos casos con su propia validación, y la banda mezclaría esa ventaja con la dispersión entre semillas. La propuesta conserva su semilla 42 y se compara con el control de esa misma semilla. En cada réplica del bootstrap circular de la regla, con los mismos días, se toma la mayor diferencia relativa del MAE por sesión entre dos semillas del control, con la de menor MAE como base. Su percentil 95 es la diferencia mínima interpretable (MID). Una mejora exige el límite superior del intervalo del 95 % por debajo de −MID. Una igualdad con la alternativa trivial exige el intervalo entero dentro de [−MID, MID]. En la no inferioridad de PT1 y de PT6 frente al control, la regla solo conserva si la MID queda por debajo de +0,2 % y solo abandona por el límite de +0,5 % si la MID queda por debajo de ese límite. Si no, informa de que el piloto no permite interpretar el margen y nunca lo ensancha. La [declaración](../../configs/evaluation/memory-pilot-seed-control.json) fija semillas, remuestreo, cuantil y reglas de PT1 y de su alternativa trivial, de PT4 frente al control y al control sectorial, de PT5 y de PT6 frente a la rama ancha y frente al control. Cada regla enlaza la declaración de su propuesta, que a su vez nombra sus reglas, y `evaluation/seed_variance_control.py` las aplica. `proposal_models` rechaza una propuesta con otro modelo, receta o variante de control, y una que cambie la receta común sin ajustar sus propias semillas, como PT4.

**Fuente y derivación.** Agarwal et al. (2021) muestran que con tres ejecuciones los intervalos bootstrap sobre ejecuciones se quedan cortos. Por eso las semillas no se remuestrean: la banda usa la mayor diferencia entre ellas dentro de cada réplica sobre los días. Lakens (2017) fija los límites de equivalencia con el menor efecto de interés antes de ver datos. Aquí ese límite es la banda entre semillas, una adaptación propia. Exigir el intervalo del 95 % dentro de la banda es más estricto que el TOST de Lakens, que usa el del 90 %.

**Control que la descarta.** Cada regla informa también de la decisión que tomaría sin banda. Si en todas las reglas evaluadas coinciden, las dos semillas añadidas no han cambiado ninguna conclusión y PT7 se retira de los pilotos siguientes. Lo calcula `discard_verdict` con los informes de las reglas.

**Coste.** Medido el 13 de octubre de 2026 en `cuda:0` ([recibo](../../reports/engineering/memory-pilot-seed-control-cost-20261013.json)) con el ajuste cronológico de `titans_mac_online` y la receta de 2000, sin pasos de optimizador, sobre una vista de la edición v3.1 con los 64 activos de EE. UU. con más filas de ajuste (ventana `fold-012`, como mucho 64 flujos por instante). Tres pasadas dieron entre 1.221 y 1.526 filas por segundo en el ajuste y entre 1.849 y 2.666 en la inferencia, con otras pruebas de CPU en marcha en el equipo. Una semilla del control en el piloto ajusta 55 millones de filas (64 activos en todas las sesiones desde 2000 hasta cada validación, 30 épocas y dos casos de búsqueda) y predice 6,2 millones. Con la pasada más lenta son 13,5 h de GPU por semilla y con la mediana, 11,0 h. La reserva de PT7 para las semillas 43 y 44 es de 27 h, el doble de la cota más lenta redondeado hacia arriba. La estimación anterior de unas 4 h no se basaba en ninguna medida y quedaba muy corta. Por el mismo motivo, el presupuesto de PT1 pasa de 6 a 41 h para PT1, su alternativa trivial y la semilla 42 del control, que cuestan por paso lo mismo que el control dentro de la dispersión medida en #453. Es una cota de cómputo: no incluye lecturas de disco, índices, normalizadores ni reanudaciones. Los presupuestos de PT4, PT5 y PT6 (8, 9 y 6 h) se escribieron antes de esta medida y tampoco cubren tres modelos con este coste por semilla, así que se recalcularán con su propia medida antes de lanzarlos. PT5 y PT6 comparten la reserva de 27 h con PT1. PT4 necesita otra igual para su control sin acumulación.

**Predicción.** La MID superará el margen de +0,2 % de PT1, de modo que el piloto no podrá conservar PT1 por no inferioridad y la regla lo informará como no interpretable.

**Revisión adversarial.** Con tres semillas la banda solo acota por debajo la cola de la dispersión, así que es una condición mínima y no una garantía. Supone además que la propuesta y su alternativa dispersan entre semillas como el control. Una propuesta que reduzca o aumente esa dispersión no queda bien medida. La banda también recoge el ruido de las sesiones, de modo que con pocas sesiones tiende a ser ancha y a dejar reglas sin interpretar. Esa es la respuesta correcta cuando el piloto no tiene potencia, pero hace que el piloto decida poco. La alternativa más sencilla sería promediar tres semillas en cada modelo, como Gu et al. (2020). Triplicaría el coste de cada propuesta, mientras que PT7 solo paga las semillas del control una vez para todos los pilotos que comparten control, filas y ventanas. Las reglas que comparten la banda no son independientes entre sí, y ninguna corrige por multiplicidad.

## PT8. Descartadas y motivo

| Idea | Motivo del descarte |
| --- | --- |
| Varios pasos internos por token al estilo DeltaProduct | Su motivación es el seguimiento de estados en lenguajes formales (Siems et al., 2025). Nuestro flujo tiene un token por sesión y el coste crece con el número de pasos. |
| Actualización por fragmentos grandes al estilo LaCT | Agrupa muchos tokens por actualización (Zhang et al., 2026). Con un token por sesión y emisión diaria, agrupar sesiones retrasaría la escritura y rompería el orden de disponibilidad. |
| Optimizador interno Muon o Newton-Schulz (ATLAS) | La ablación de ATLAS mejora la perplejidad al quitar Muon y los revisores lo señalaron. Sin evidencia en series. |
| HOPE y memoria continua de Nested Learning | Sin código oficial, sin series temporales y con mayor memoria reconocida por los autores. |
| Memory Caching | Diseñado para secuencias largas de lenguaje. Nuestras secuencias por flujo son de cientos de pasos y el estado ya es pequeño. |
| Recuperación con modelos de lenguaje (FinSeer, StockMem, MemCast) | Riesgo de contaminación por el preentrenamiento y ganancias pequeñas con controles justos (TS-Memory frente a LoRA igualado). |
| Sustituir la memoria por Mamba u otro SSM | Sin evidencia de ventaja en retornos de acciones. Rahimikia et al. (2025) encuentran que los modelos fundacionales de series rinden peor que el cero. |
| Más capacidad por neurona (dendritas activas, KAN, MoE fino) | Con R² cercano a cero la capacidad no es el cuello de botella. Cualquier prueba exigiría el control de más parámetros con el mismo presupuesto, que el piloto no puede pagar. Se detalla en la revisión. |
| Ampliar K a más de 4 refinamientos | El cómputo en inferencia se satura o empeora en problemas difíciles y de regresión (Muennighoff et al., 2025, y Gema et al., 2025). |

## Componentes que conviene retirar o simplificar

Son recomendaciones condicionadas a los diagnósticos. Ninguna se aplica en esta revisión.

1. **Banco episódico compartido de 1.024 episodios.** Con 4.000 a 5.000 etiquetas maduras por sesión, una sola sesión supera su capacidad y el muestreo por depósito conserva sobre todo episodios antiguos. Antes de ampliar el factorial con variantes del banco, D8 debe mostrar que recupera episodios recientes y útiles. Si no, la opción más sencilla es un banco por mercado con capacidad de varias sesiones o retirar el banco de la comparación principal.
2. **Refinamientos K = 4.** K = 1 ya es el principal. La literatura de cómputo en inferencia (Bansal et al., 2022, y Gema et al., 2025) indica que más pasos pueden empeorar fuera del régimen de entrenamiento. Si el presupuesto aprieta, K = 4 es el primer modelo prescindible.
3. **Interpretación de C en CM-v1.** C penaliza el radio numérico de $R^\top JR$ de cada paso. La Proposición 3 muestra que una cota por paso no controla los productos. C puede conservarse como experimento declarado, pero no debe presentarse como garantía de estabilidad. Si se busca esa garantía, PT1 la da por construcción en la memoria lineal.
4. **Regla delta de B6.** Su coste es unas diez veces el de la proximal en cohortes grandes y RLS la generaliza. Si PT3 no es peor que proximal, la regla delta puede salir de la campaña.
5. **Truncación 8.** No se recomienda ampliarla sin D7, porque su coste crece linealmente con el horizonte.

## Referencias

Las entradas completas están en [`post-titans.bib`](../references/post-titans.bib) y en los catálogos existentes.

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. y Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems, 34*. https://arxiv.org/abs/2108.13264
- Angelopoulos, A. N., Candès, E. J. y Tibshirani, R. J. (2023). Conformal PID control for time series prediction. *Advances in Neural Information Processing Systems, 36*, 23047–23074. https://doi.org/10.52202/075280-1000
- Baltagi, B. H. (2021). *Econometric analysis of panel data* (6.ª ed.). Springer. https://doi.org/10.1007/978-3-030-53953-5
- Bansal, A., Schwarzschild, A., Borgnia, E., Emam, Z., Huang, F., Goldblum, M. y Goldstein, T. (2022). End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking. *Advances in Neural Information Processing Systems, 35*, 20232–20242.
- Behrouz, A., Razaviyayn, M., Zhong, P. y Mirrokni, V. (2026). It's all connected: A journey through test-time memorization, attentional bias, retention, and online optimization. *International Conference on Learning Representations*.
- Behrouz, A., Zhong, P. y Mirrokni, V. (2025). Titans: Learning to memorize at test time. *Advances in Neural Information Processing Systems, 38*.
- Benna, M. K. y Fusi, S. (2016). Computational principles of synaptic memory consolidation. *Nature Neuroscience, 19*(12), 1697–1706. https://doi.org/10.1038/nn.4401
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G. y Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*. https://arxiv.org/abs/2103.03098
- Boyd, S., El Ghaoui, L., Feron, E. y Balakrishnan, V. (1994). *Linear matrix inequalities in system and control theory*. SIAM. https://doi.org/10.1137/1.9781611970777
- Fentazi, M. R., Ameur, M. y Ksentini, A. (2026). *Training on the future: A delay-aware audit of test-time adaptation for time-series forecasting* (arXiv:2610.12232v1). https://arxiv.org/abs/2610.12232v1
- Gema, A. P. et al. (2025). Inverse scaling in test-time compute. *Transactions on Machine Learning Research*.
- Gibbs, I. y Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems, 34*.
- Gu, S., Kelly, B. y Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies, 33*(5), 2223–2273.
- Guijarro-Ordonez, J., Pelger, M. y Zanotti, G. (2026). Deep learning statistical arbitrage. *Management Science, 72*(9), 7502–7549. https://doi.org/10.1287/mnsc.2022.03132
- Ioannou, P. A. y Kokotovic, P. V. (1984). Instability analysis and improvement of robustness of adaptive control. *Automatica, 20*(5), 583–594. https://doi.org/10.1016/0005-1098(84)90009-8
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering, 82*(1), 35–45. https://doi.org/10.1115/1.3662552
- Kaplanis, C., Shanahan, M. y Clopath, C. (2018). Continual reinforcement learning with complex synapses. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80, 2497–2506.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362. https://doi.org/10.1177/1948550617697177
- Manokhin, V. (2026). *Report the floor: A training-free conformal interval is a mandatory baseline for probabilistic time-series forecasting* (arXiv:2606.09473v1). https://arxiv.org/abs/2606.09473v1
- Muennighoff, N. et al. (2025). s1: Simple test-time scaling. *Proceedings of EMNLP 2025*, 20275–20321. https://doi.org/10.18653/v1/2025.emnlp-main.1025
- Perez, E., Strub, F., de Vries, H., Dumoulin, V. y Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). https://doi.org/10.1609/aaai.v32i1.11671
- Rahimikia, E., Ni, H. y Wang, W. (2025). *Re(visiting) time series foundation models in finance* (arXiv:2511.18578v1). https://arxiv.org/abs/2511.18578
- Romano, Y., Patterson, E. y Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems, 32*.
- Siems, J. et al. (2025). DeltaProduct: Improving state-tracking in linear RNNs via Householder products. *Advances in Neural Information Processing Systems, 38*, 153738–153782. https://doi.org/10.52202/085713-5141
- Szabadváry, J. H. (2024). Adaptive conformal inference for multi-step ahead time-series forecasting online. *Proceedings of COPA 2024*, PMLR 230, 250–263.
- Wang, X. y Hyndman, R. J. (2024). *Online conformal inference for multi-step time series forecasting* (arXiv:2410.13115v2). https://arxiv.org/abs/2410.13115v2
- West, M. y Harrison, J. (1997). *Bayesian forecasting and dynamic models* (2.ª ed.). Springer. https://doi.org/10.1007/b98971
- Zhang, T. et al. (2026). Test-time training done right. *International Conference on Learning Representations*.
