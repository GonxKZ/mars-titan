# Puerta E4 por evidencia de cambio de régimen

`regime_evidence_gate_v1` regula cuánto escribe la memoria de corrección madura A de B6 en cada mercado. Cuando el error de las predicciones emitidas se mantiene en su escala habitual, A escribe con una fracción de su tasa η. Cuando ese error crece de forma sostenida, la puerta se abre y A se adapta más deprisa. La [declaración de ventanas](../../configs/experiments/regime-evidence-gate-window-v1.json) conecta la puerta al recorrido de corrección con su control emparejado. La sesión durable `FinancialSession` también admite la puerta y publica su estado en cada generación confirmada. Hay comprobaciones CPU con fixtures y pesos fijados a mano, sin entrenar ni ejecutar una comparación predictiva.

## Motivación y fuentes

Behrens et al. (2007) mostraron que las personas aprenden más deprisa en entornos volátiles y relacionaron esa tasa con la volatilidad estimada ([DOI:10.1038/nn1954](https://doi.org/10.1038/nn1954)). Nassar et al. (2010) describieron una regla delta aproximadamente bayesiana cuya tasa aumenta con la probabilidad de un punto de cambio ([DOI:10.1523/JNEUROSCI.0822-10.2010](https://doi.org/10.1523/JNEUROSCI.0822-10.2010)). Liakoni et al. (2021) proponen una sorpresa basada en el factor de Bayes que regula cuánto pesa una observación nueva frente a lo aprendido en entornos volátiles ([DOI:10.1162/neco_a_01352](https://doi.org/10.1162/neco_a_01352)). El detector es el CUSUM unilateral de Page (1954, [DOI:10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100)). Las cuatro referencias se han comprobado en Crossref. No se han reproducido sus modelos ni sus experimentos.

Ninguno de esos trabajos propone esta puerta. El CUSUM no estima una probabilidad de cambio ni un factor de Bayes. Es una aproximación barata y recuperable de la misma idea: acumular evidencia antes de cambiar la tasa de aprendizaje. La regla y sus valores son una propuesta del proyecto.

## Ecuaciones

Para el mercado \(m\) y la publicación \(t\), los resultados que maduran en ese evento tienen error \(e_i=y_i-\widehat y_i^{emitida}\). Se usa el punto final emitido, no el núcleo sin corrector ni la sorpresa asociativa de Titans. Con la media cuadrática \(r_{m,<t}^2\) anterior a la cohorte:

\[
z_i=\frac{e_i}{\max(r_{m,<t},\epsilon)},\qquad
y_t=\frac{1}{n_t}\sum_i z_i^2-1,
\]

\[
S_t=\min\bigl(h,\max(0,S_{t-1}+y_t-k)\bigr),\qquad
g_t=g_{\min}+(1-g_{\min})\min\!\left(1,\frac{S_t}{h}\right).
\]

Mientras el mercado no acumula `min_history` errores, \(y_t\) no se suma y \(S\) sigue en cero. La primera cohorte del mercado fija \(r^2\) con su media cuadrática. Después, todos los errores la actualizan de forma exponencial con retención \(\rho=2^{-1/\text{semivida}}\), medida en publicaciones:

\[
r_{m,\le t}^2=\rho\,r_{m,<t}^2+(1-\rho)\,\frac{1}{n_t}\sum_i e_i^2.
\]

Con errores en su escala habitual, \(\mathbb E[z^2]\approx 1\) y la deriva \(k\) hace bajar \(S\). Un aumento sostenido de la varianza lo acumula hasta \(h\). Los valores declarados son \(h=4\), \(k=0{,}5\), \(g_{\min}=0{,}25\), semivida de 63 publicaciones, historial mínimo de cuatro errores y suelo \(10^{-8}\). Son valores de desarrollo, sin ajuste ni validación predictiva.

El CUSUM de Page no tiene techo. Aquí \(S\) se acota en \(h\) porque la puerta ya está abierta del todo en ese punto. Sin el techo, un cambio largo acumula tanta evidencia que la puerta tarda decenas de publicaciones en cerrarse cuando el error vuelve a su nivel. Con el techo, \(S\) baja al menos \(k-y_t\) por publicación desde \(h\). Es una desviación deliberada del detector original y tiene su propia mutación dirigida.

## Dónde actúa la puerta

La regla proximal de B6 resuelve \(A'=(I+\eta G)^{-1}(\rho_A A+\eta C)\), con \(G=\sum_i w_i k_ik_i^\top\), \(C=\sum_i w_i k_iv_i^\top\) y pesos que suman uno. E4 sustituye \(\eta\) por \(\eta g\) en la escritura de ese evento. Si el evento mezcla filas de US y CN, la tasa es \(\eta\bar g\), con \(\bar g=\sum_i w_ig_i\), y los pesos pasan a \(w_ig_i/\bar g\). Por fila equivale a \(\eta w_ig_i\), y la [prueba con dos mercados](../../tests/memory/test_regime_gate_correction.py) lo compara con la solución directa del sistema. La configuración de A no cambia, así que la retención \(\rho_A\) y la tasa de las escrituras siguientes se mantienen.

El valor escrito sigue siendo etiqueta menos núcleo. La puerta solo cambia la tasa. La decisión se toma con los resultados que maduran en el evento y se aplica después de emitir todas sus predicciones, de modo que solo afecta a las predicciones siguientes. Las disponibilidades de la cohorte se comprueban antes de leer el primer valor financiero.

E4 admite únicamente la corrección proximal simple. La regla delta tiene una cota de estabilidad en \(\eta\|k\|^2\), kalman no usa η, y la mezcla E1 y las áreas B1, con o sin su puente B2, escriben estados acoplados. Con las claves ligadas a roles de B3, A va envuelta en el operador de roles y una escritura con otra tasa perdía esa envoltura. La ventana aceptaba esa combinación hasta el 30 de octubre y el estado resultante no se podía recuperar. Ahora `check_correction` la rechaza igual que las demás, en la ventana y en la sesión. Esas combinaciones quedan rechazadas hasta tener su propia regla de tasa. E3 y E4 tampoco se combinan, porque no existe una identidad ni un control conjuntos.

## Control que la descarta

El control `constant` escribe siempre con la misma puerta y calcula el mismo estadístico, sin usarlo. `matched_control` construye ese control con la puerta media por fila que la variante `cusum` realizó en la validación de su misma ventana. La validación precede a la calibración y a la evaluación, así que el control no usa información de los tramos que se comparan. Una puerta constante igual a uno reproduce B6 bit a bit.

La comparación tiene tres modelos con el mismo padre, receta, casos de búsqueda y filas: B6 sin puerta, la variante `cusum` y el control emparejado. Si la variante no mejora al control en MAE por sesión, con emparejamiento temporal e incertidumbre por bloques, el momento en que se abre la puerta no aporta nada. Una mejora frente a B6 que también obtenga el control solo indicaría que η era demasiado alta. La puerta media de la evaluación puede diferir de la de validación, así que ambos modelos no tienen garantizada la misma tasa media en el tramo evaluado. Ese valor se registra en cada fase.

## Objeciones

El estadístico no distingue un cambio en la dinámica de un aumento del ruido. Piray y Daw (2024, [DOI:10.1038/s41467-024-53459-z](https://doi.org/10.1038/s41467-024-53459-z)) separan volatilidad y estocasticidad y muestran que deben mover la tasa en sentidos opuestos. Si el error crece porque el mercado está más ruidoso, abrir la puerta haría que A persiguiera ruido. El control emparejado no resuelve esta ambigüedad, solo permite detectar si la puerta empeora el resultado.

La media de \(z^2\) es sensible a colas pesadas. Un único error extremo puede abrir la puerta de todo un mercado durante varias publicaciones. Un dato erróneo tendría el mismo efecto. Una versión robusta, con mediana o recorte, sería una variante distinta con su propia identidad. No se ha implementado.

La puerta es por mercado. Un cambio que afecte solo a un sector se diluye en la media del mercado. Llevarla a sector o activo multiplica el estado y reduce las filas por cohorte, y necesitaría otra comparación.

## Estado y recuperación

`RegimeEvidenceGate.step` devuelve un `GateDecision` y otro estado. No modifica el anterior. Cada mercado conserva cuenta, media cuadrática, CUSUM, filas escritas y suma de puertas. La publicación vacía avanza el corte sin cambiar estadísticas. El estado se exporta con huella SHA-256 y `restore` rechaza cambios, configuraciones distintas y valores incoherentes, como un CUSUM fuera de \([0,h]\).

`RegimeGateCorrectionInference` extiende la inferencia de B6. Guarda el punto emitido de `run.pending` antes de que la resolución lo retire y sustituye solo la escritura de A. `run_correction_window` acepta `regime_gate=None` por defecto y mantiene entonces su identidad. Con una configuración válida registra la puerta y las huellas de sus fuentes en la petición, la identidad y `selected.json`. `carry_correction` la restaura del ancla y rechaza identidades contradictorias antes de abrir los tramos. Cada fase reinicia la escala y el CUSUM, como B6 reinicia A, y una fase interrumpida se repite desde su inicio.

La comprobación de la corrección y la escritura con la tasa η·g de cada fila están en `regime_evidence_gate` (`check_correction` y `gated_write`). La ventana y la sesión llaman a las mismas funciones.

E3 y E4 comparten el mismo punto de inserción. `WRITE_FACTORS` en `mars_titan_correction` declara para cada factor su módulo, su clase de inferencia y su política de memoria. Las ventanas, la identidad y el traslado tratan ambos factores con el mismo código. Las identidades de E3 no cambian.

## Sesión durable

`FinancialSession` acepta `regime_gate` sobre la misma corrección proximal simple. Cada generación nativa confirmada es una publicación E4, también la que no madura ningún resultado. La cohorte son los resultados que el ejecutor nativo acepta para predicciones pendientes ya emitidas, y el error se mide contra el punto final emitido, como en la ventana. La puerta de cada mercado escala la tasa de las escrituras de A en esa misma generación. El estado siguiente se prepara con la generación y se confirma con ella, así que un corte antes o después de confirmar se recupera con la misma puerta y la misma predicción siguiente. La sesión empieza sin estado, igual que cada fase de la ventana.

Al cargar una generación, `regime_gate_session.verify` exige que el número de publicaciones coincida con la generación y el corte con el suyo. `verify_confirmed` exige que los errores acumulados coincidan con los resultados aplicados por el ejecutor. El código de `regime_gate_session` y de `regime_evidence_gate` forma parte de la identidad del modelo, y la configuración entra en el contrato. Cambiarla durante la sesión o al reanudar se rechaza. E3 y E4 tampoco se combinan aquí.

La ventana solo publica en los eventos con etiquetas y la sesión en todas las generaciones, así que sus contadores de publicaciones y sus cortes difieren aunque las estadísticas coincidan. La [prueba de paridad](../../tests/memory/test_regime_gate_session.py) compara las 20 emisiones, la matriz A final y el estado de cada mercado de las dos implementaciones sobre los mismos seis eventos. Sin la puerta, las emisiones cambiarían desde el tercer evento.

## Comprobaciones

La [verificación del 28 de octubre](../../reports/engineering/regime-gate-20261028/verification.json) registra 451 pruebas CPU superadas, 71 propias y el resto regresiones de B6, E3, áreas, kalman, lesiones y las guardas del bloqueo. La única omisión es la guarda que impide preparar ventanas B6 de la campaña. No hay pasos de optimizador ni datos reales. Las pruebas cubren las ecuaciones con valores literales, la apertura y el cierre de la puerta, la independencia entre mercados, la invariancia al orden de las filas, la disponibilidad comprobada antes de leer valores, la recuperación con huella, la igualdad bit a bit con B6 cuando la puerta vale uno, la tasa η·g de cada escritura, la ponderación por fila con dos mercados, el control emparejado y la regeneración desde el ancla.

Se detectan las [11 mutaciones dirigidas](../../reports/engineering/regime-gate-20261028/mutations.json) con referencias verdes antes y después: puntuar la etiqueta sin el punto emitido, escribir sin la puerta, repartir la puerta media entre mercados, conservar la puerta entre fases, ignorar las huellas de las fuentes, tomar el control de la evaluación, quitar el techo del CUSUM, estandarizar con la propia cohorte, dejar que el control siga al detector, perder la puerta al regenerar y combinar E3 con E4. La [cobertura](../../reports/engineering/regime-gate-20261028/quality.json) de los dos módulos nuevos es de 231/238 sentencias y 83/88 ramas, con CRAP máximo 21. Las líneas sin recorrer son validaciones defensivas que ninguna prueba alcanza sin fabricar estados imposibles. El [diagnóstico de `mars_titan_correction`](../../reports/engineering/regime-gate-20261028/quality-support.json) se limita a esta batería.

Una prueba añadida para cubrir validaciones encontró un error real. `(e/r) ** 2` lanzaba `OverflowError` con errores enormes en vez del `ValueError` declarado. Ahora se calcula z·z, que da infinito y se rechaza con su mensaje. El producto es además el cuadrado correctamente redondeado de IEEE 754. `** 2` pasa por `pow` de la biblioteca C, que se desvió una unidad en la última posición en 1.696 de dos millones de valores aleatorios comprobados con aritmética racional. No había resultados anteriores de E4 que dependieran de esa diferencia. Las mutaciones se repitieron sobre la fuente corregida.

## Coste

La [medida de coste](../../reports/engineering/regime-gate-20261028/cost.json) recorre 64, 256, 1.024 y 8.192 filas de prueba repartidas entre US y CN, con dos calentamientos y siete repeticiones. El modo sin puerta solo escribe A con la regla proximal y clave de 64 coordenadas, como B6. Los modos `cusum` y `constant` añaden la puerta, la escritura con la tasa η·g, la exportación del estado, su JSON y la recuperación. El evento medido aumenta el error un 50 % y la puerta `cusum` queda en 0,39, abierta solo en parte. La tabla da medianas en milisegundos:

| Filas | Sin puerta | `cusum` | Constante |
| ---: | ---: | ---: | ---: |
| 64 | 0,311 | 0,671 | 0,664 |
| 256 | 0,541 | 1,332 | 1,226 |
| 1.024 | 1,300 | 5,718 | 4,380 |
| 8.192 | 16,132 | 29,696 | 29,983 |

El coste añadido crece con las filas porque la puerta recorre los resultados en Python. El estado serializado ocupa menos de 820 bytes con dos mercados. Las ventanas no exportan ni recuperan el estado en cada evento, solo al cerrar la fase, así que esta medida sobrestima el coste por evento del recorrido real. La carga media del equipo estuvo entre 26 y 30 durante la medida, con otras pruebas en marcha, y el RSS máximo del proceso fue de 646.180 KiB. No se mide la ventana completa, VRAM ni energía. Siete repeticiones no permiten estimar colas p95 o p99 estables.

## Pendiente

Falta comprobar CUDA cuando haya admisión GPU y ejecutar la comparación con datos reales, que sigue bloqueada. Nada de lo anterior acredita que E4 reduzca el error.
