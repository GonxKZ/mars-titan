# Diagnósticos de memoria para la arquitectura MARS-TITAN

Estado: especificación para otro trabajo de implementación. Nada de lo descrito se ha ejecutado con datos. Las medidas requieren modelos ya ajustados, así que dependen del levantamiento del bloqueo de aprendizaje. Ninguna usa el test de 2024 y ninguna ejecuta pasos de optimizador. Todas son pasadas hacia delante, productos Jacobiano-vector o derivadas locales sobre pesos congelados.

Las métricas existentes (MAE residual, interval score, Brier, ECE, cartera y RL) dicen si el sistema predice mejor. No dicen por qué ni si la memoria hace lo que su diseño supone. Estos diagnósticos responden a esa segunda pregunta y alimentan las reglas de decisión de las [propuestas posteriores a Titans](post-titans-proposals.md). Su base matemática para la memoria lineal está en el [certificado de contracción](titans-memory-certificate.md).

## Reglas comunes

- Se calculan sobre las ventanas de validación del recorrido por etapas, con los mismos activos, filas y semillas que la comparación principal. Las fechas de corte se fijan antes de mirar resultados.
- Cada diagnóstico registra versión del código, huella de la configuración, semilla, ventana, activos y coste medido. Los resultados se guardan como recibos JSON en `reports/` y nunca se usan para seleccionar hiperparámetros sobre el test.
- Las cifras por activo se agregan por sesión antes de promediar, para que los días con muchos activos no dominen. La incertidumbre usa bootstrap circular por bloques de 16 sesiones, como el protocolo de la comparación.
- Los módulos propuestos son rutas futuras. Hoy existen `models/titans/transition_jacobian.py` y `models/titans/local_control.py`, que ya calculan Jacobianos densos de la transición y productos Jacobiano-vector en dimensiones pequeñas.

## D1. Curva de influencia y retención

**Pregunta.** ¿Cuánto tiempo afecta una observación a la memoria y a la predicción? ¿Coincide con la semivida que declaran las puertas?

**Definición.** Para un flujo y una sesión $t_0$, la influencia a distancia $\tau$ es

$$
I(\tau)=\frac{\lVert J_{z,t_0+\tau\leftarrow t_0}\,u\rVert}{\lVert u\rVert},\qquad
I_y(\tau)=\Big|\frac{\partial\hat y_{t_0+\tau}}{\partial x_{t_0}}\,u\Big|,
$$

donde $u$ es la dirección del token fusionado observado en $t_0$ normalizada y $J$ el producto de Jacobianos de la transición rápida. Se calcula con productos Jacobiano-vector hacia delante, sin materializar matrices de $2LD^2$ filas.

**Salida.** Curvas medianas y cuantiles 10 y 90 de $I(\tau)$ e $I_y(\tau)$ para $\tau\in\{1,2,4,8,16,32,64,128,256,512\}$, por mercado y año. Semivida empírica $\tau_{1/2}$ y su comparación con la semivida de las puertas $\ln2/-\ln(1-\bar\alpha)$.

**Uso.** Si $I_y(\tau)$ es despreciable para $\tau>8$, la memoria no transporta información más allá del horizonte de truncación con el que se entrena y su ventaja frente a un contexto de ocho sesiones sería difícil de defender. Si es grande para $\tau>8$, el entrenamiento nunca ve esas dependencias y conviene medir la sensibilidad a la truncación (D7).

**Coste estimado.** Un JVP por paso cuesta del orden de una pasada hacia delante adicional. Con 64 flujos de muestra y 512 pasos por ventana, menos de 0,1 h de GPU por ventana y configuración.

## D2. Comportamiento ante cambios de régimen

**Pregunta.** ¿La memoria reacciona a un cambio de régimen como se espera y recupera el error previo?

**Definición.** Estudio de eventos alrededor de fechas fijadas por una cronología externa antes de mirar resultados: quiebra de Lehman (15/09/2008), caída de marzo de 2020 (24/02/2020 a 23/03/2020) y el ciclo de subidas de tipos de 2022 (16/03/2022). Para China se usan las sesiones de junio a agosto de 2015. En una ventana de ±60 sesiones se registran el MAE residual por sesión relativo a la mediana previa, las medias de $\alpha$, $\eta$ y $\theta$, la norma del estado $\lVert W_t\rVert_F$, la magnitud de escritura $\lVert W_t-W_{t-1}\rVert_F$, el tiempo de recuperación hasta volver al 110 % del MAE previo y la fracción de pasos con puertas fuera de la caja certificada.

**Uso.** Compara memoria online, congelada y desactivada con las mismas filas. Si la variante online no se recupera antes que la congelada, la adaptación interna no aporta en el momento en que más debería. También permite ver si PT1 frena la reacción en el evento.

**Coste estimado.** Reutiliza las predicciones ya emitidas más las estadísticas de estado, que deben registrarse durante la pasada. Menos de 0,05 h de GPU por evento y configuración si se registran en la misma pasada.

## D3. Sensibilidad al retraso de las etiquetas

**Pregunta.** ¿Las correcciones con resultados maduros (B6, PT2 y PT3) dependen de recibir la etiqueta lo antes posible?

**Definición.** Se repite la escritura madura con retrasos adicionales $d\in\{0,1,2,5\}$ sesiones sobre el mínimo causal. Nunca se usa un retraso menor que el mínimo. Se registran el MAE, la cobertura y el interval score de cada variante frente al mismo control sin corrección.

**Uso.** Una mejora que aparece con $d=0$ y desaparece con $d=1$ es frágil y probablemente explota autocorrelación de muy corto plazo. Fentazi et al. (2026) muestran con un arnés sin fuga que casi todas las adaptaciones publicadas pierden su ventaja con el retraso causal, mientras que un banco de RLS cerrado resiste. Este diagnóstico decide si PT3 se comporta como ese banco.

**Coste estimado.** Solo CPU sobre predicciones ya emitidas. Minutos por ventana.

## D4. Ablaciones declaradas

**Pregunta.** ¿Qué componente aporta cada parte del resultado?

**Definición.** Con los mismos pesos lentos cuando la variante lo permite, o con ajustes emparejados cuando no, se comparan:

| Ablación | Qué aísla | ¿Necesita reentrenar? |
| --- | --- | --- |
| `memory_mode` online, frozen y disabled | Adaptación interna frente a lectura fija y frente a no leer | No para frozen y disabled con los mismos pesos |
| `persistent_tokens = 0` | Memoria persistente del artículo | Sí |
| `gate_bias` declarado frente a v1 | Inicialización de las puertas | Sí |
| `memory_residual_layer_norm` | Forma $x+\mathrm{LN}(\mathrm{MLP}(x))$ | Sí |
| PT1 completo, solo caja y solo recorte | Las dos piezas de PT1 por separado | Sí, solo si PT1 completo pasa su regla |
| Banco episódico desactivado | Banco frente a memoria neuronal | No |
| K = 1 frente a 2 y 4 | Refinamientos de lectura | No |
| B6 delta, proximal y PT3 Kalman | Regla de corrección madura | No |

**Uso.** Atribuye resultados y alimenta la decisión de retirar componentes. Las ablaciones sin reentreno se calculan para todas las ventanas. Las que exigen reentreno se limitan a las ventanas del piloto.

## D5. Espectro y Jacobiano de la recurrencia sobre datos reales

**Pregunta.** ¿La memoria de dos capas con LayerNorm residual se comporta como contractiva sobre los datos, aunque el certificado no la cubra?

**Definición.** Para una muestra de flujos y sesiones se calcula el Jacobiano $J_t=\partial z_t/\partial z_{t-1}$ de la transición rápida completa. Con $D\le16$ se usa la forma densa de `transition_jacobian.py`. Con $D=64$ se usan iteraciones de potencia con JVP y VJP. Se registran:

- radio espectral y norma espectral de cada $J_t$,
- crecimiento de productos $\gamma_L=\lVert J_{t+L}\cdots J_{t+1}\rVert^{1/L}$ para $L\in\{8,32,128,512\}$,
- exponente de Lyapunov máximo estimado por reortogonalización QR sucesiva (método de Benettin),
- norma inducida por $P_0\otimes I$ de la caja certificada, para la parte lineal cuando se use una memoria de una capa.

**Uso.** Un exponente positivo en datos reales es una señal de alarma aunque el error sea bueno. Si $\gamma_8<1$ pero $\gamma_{512}>1$, la truncación de 8 oculta una inestabilidad a largo plazo. Estos valores son medidas en coma flotante y no certifican cotas, igual que advierte `transition_jacobian.py`.

**Coste estimado.** Con $D=64$ y 20 iteraciones de potencia, unas 40 pasadas hacia delante y hacia atrás por punto. Con 32 flujos y 64 sesiones por ventana, del orden de 0,2 h de GPU por ventana y configuración.

## D6. Capacidad asociativa efectiva

**Pregunta.** ¿Cuántas asociaciones recientes puede recuperar la memoria sobre las claves que realmente produce el codificador?

**Definición.** En una sesión $t$ y para las claves $k_{t-a}$ escritas hace $a$ pasos, el error de recuerdo relativo es $e(a)=\lVert M_t(k_{t-a})-v_{t-a}\rVert/\lVert v_{t-a}\rVert$, con $M_t$ la memoria en $t$ y sin escribir nada nuevo. La capacidad efectiva es el mayor $N$ con mediana de $e(a)<0{,}5$ para todo $a\le N$. Se acompaña de la dimensión efectiva de las claves (razón de participación de su covarianza en una ventana de 256 sesiones) y del rango estable de cada capa, $\lVert W\rVert_F^2/\lVert W\rVert_2^2$.

**Uso.** Si la dimensión efectiva de las claves es muy inferior a $D$, la capacidad está limitada por el codificador y ampliar $D$ o cambiar la regla de escritura no ayudará. ATLAS (Behrouz et al., 2026) acota la capacidad de memorias matriciales y profundas, pero con claves independientes. Este diagnóstico mide la que se obtiene con claves financieras correlacionadas.

**Coste estimado.** Lecturas sin escritura. Menos de 0,05 h de GPU por ventana.

## D7. Sensibilidad a la truncación del gradiente exterior

**Pregunta.** ¿Entrenar con truncación 8 deja sin aprender dependencias que la memoria sí transporta?

**Definición.** Con los pesos ajustados, se compara la norma del gradiente exterior respecto a los parámetros lentos calculada con truncación 8, 32 y 128 sobre los mismos lotes de validación, sin aplicar el gradiente. Se registra el coseno entre las direcciones y la razón de normas.

**Uso.** Si el coseno entre truncación 8 y 128 es bajo, la dirección de aprendizaje cambia al ver más pasado y la truncación es una limitación real. Si es alto, ampliarla no justifica su coste. Es una derivada sobre datos de validación, no un ajuste.

**Coste estimado.** Con 128 pasos de retropropagación, del orden de 16 veces una pasada truncada a 8. Unas 0,5 h de GPU por ventana con 32 flujos.

## D8. Banco episódico y edad de lo recuperado

**Pregunta.** ¿El banco compartido de 1.024 episodios recupera episodios recientes o casi solo antiguos?

**Definición.** Distribución de la antigüedad de los episodios recuperados por sesión, fracción de admisiones rechazadas y relación entre la capacidad y las etiquetas maduras de cada sesión (entre 4.000 y 5.000 en la campaña completa). Se acompaña de la utilidad de lectura frente a un banco aleatorio del mismo tamaño.

**Uso.** Si una sola sesión supera la capacidad del banco, el muestreo por depósito conserva sobre todo episodios antiguos y el banco no puede funcionar como memoria de corto plazo. Es la base de la recomendación de simplificación de la propuesta principal.

**Coste estimado.** Solo registros de la pasada y CPU.

## D9. Calibración a lo largo del tiempo

**Pregunta.** ¿La cobertura de los intervalos se mantiene por año, mercado y régimen?

**Definición.** Cobertura empírica, interval score y PIT de los cuantiles por año y mercado, para el núcleo sin calibrar, con CQR estática y con PT2. Desviación absoluta media de la cobertura respecto al nominal y su peor año.

**Uso.** Es la métrica principal de PT2 y un control secundario de PT1 y PT3.

**Coste estimado.** Solo CPU.

## Orden recomendado

D4 y D8 se obtienen casi gratis si la pasada registra las estadísticas necesarias. Conviene implementarlos primero. D1 y D5 dan la evidencia más directa sobre la memoria y deben preceder a cualquier propuesta que modifique su dinámica. D3 y D9 deciden PT2 y PT3. D6 y D7 son más caros y se aplican a las ventanas del piloto.

## Referencias

- Behrouz, A., Li, Z., Kacham, P., Daliri, M., Deng, Y., Zhong, P., Razaviyayn, M. y Mirrokni, V. (2026). ATLAS: Learning to optimally memorize the context at test time. *Proceedings of the 43rd International Conference on Machine Learning*, PMLR 306, 7361–7390.
- Fentazi, M. R., Ameur, M. y Ksentini, A. (2026). *Training on the future: A delay-aware audit of test-time adaptation for time-series forecasting* (arXiv:2610.12232v1). https://arxiv.org/abs/2610.12232v1
