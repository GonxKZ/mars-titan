# Jerarquía fija del puente episódico

La variante opcional `shared_evidence_bridge_hierarchy_v1` de [#719](https://github.com/GonxKZ/mars-titan/issues/719) está implementada y cuenta con pruebas, mutación dirigida y medidas técnicas sobre fixtures. La revisión aún no está cerrada. El experimento financiero no se ha ejecutado. La opción permanece desactivada por defecto. El [informe de evidencia](../../reports/engineering/hierarchical-bridge-20261022/README.md) reúne recibos, fuentes y límites.

La jerarquía amplía el puente episódico posterior a MAC. Conserva un `EpisodeSnapshot`, una selección de candidatos y una proyección común de valores por lectura. Sus cuatro vistas no son cuatro fuentes independientes ni copias persistentes de los episodios. La fusión de cinco modalidades anterior a MAC y el contrato de contexto, selección y refinamiento de #709 son otros componentes. La elección adaptativa de configuraciones de #711 también queda fuera.

## Configuración y estados

La configuración es `EvidenceBridgeConfig.hierarchy`, opcional y de tipo `BridgeHierarchyConfig`. El caso principal es:

```python
BridgeHierarchyConfig(
    topology="tree",
    local_steps=2,
    feedback=True,
    max_node_updates_per_row=988,
)
```

Solo se admiten `tree` y `star`, `local_steps` en {1,2,4}, `feedback` booleano y una cuota entera entre 1 y 1596. Los enteros no admiten booleanos. Las dos rondas son fijas y la configuración rechaza campos desconocidos. No hay listas arbitrarias de nodos o instrucciones ejecutables.

El contrato exige cuatro regiones, `topology="self"` y `cycles=0` en la configuración exterior del puente. `hierarchy.topology` identifica el grafo de siete estados. `recurrence` e `intervention` deben ser None. La combinación con intervenciones fijas v1 se rechaza expresamente, sin ignorar máscaras o cortes. Su compatibilidad sigue pendiente como extensión posterior. La máscara directa de disponibilidad del lector sí pertenece al contrato jerárquico.

Los nodos 0–3 son las vistas, 4–5 los grupos y 6 la raíz. Las parejas de pertenencia son {0,1} y {2,3}. Cada estado tiene H componentes. El estado inicial es `X⁰=(R⁰,0,0,0)`, con cuatro vistas y tres auxiliares. Grupos y raíz se reinician una vez por lectura y se conservan entre sus dos rondas. No pasan a otra fila, lectura K, ruta o decisión.

Las vistas conservan consultas, atenciones y matrices receptoras `W_r` propias. Los grupos y la raíz no consultan de nuevo el snapshot, no reciben etiquetas distintas y no tienen matrices aprendibles, sesgos o sorteos de parámetros propios. Sus transiciones usan la identidad, sin tanh. La salida conserva la proyección de las cuatro vistas finales, `output(final_views.reshape(B, 4*H))`. No añade estados auxiliares a la cabeza.

## Grafo efectivo y orden de cálculo

Los dos soportes tienen seis pares conectados en ambos sentidos y siete autoaristas, diecinueve entradas antes de aplicar disponibilidad:

| Topología | Pares conectados |
| --- | --- |
| `tree` | (0,4), (1,4), (2,5), (3,5), (4,6), (5,6) |
| `star` | (0,6), (1,6), (2,6), (3,6), (4,6), (5,6) |

La adyacencia usa `[receptor, emisor]`. Para la disponibilidad local `a₀,…,a₃`, cada grupo está presente si alguna de sus dos vistas lo está y la raíz si alguna de las cuatro lo está. Se utiliza esa misma regla en ambas topologías. Un autoenlace no convierte en presente a un nodo ausente.

Para el soporte A y la máscara de siete nodos a, se define un único operador por fila:

$$
B_{ij}=a_i a_j A_{ij},\qquad
G_{ij}=\frac{B_{ij}}{\max(1,\sum_k B_{ik})}.
$$

Una fila ausente queda a cero. Todas las fases usan las filas receptoras del mismo G normalizado sobre el soporte completo. No hay grafos ascendentes y descendentes diferentes. Una vista ausente tampoco conserva su residual.

Para el estado X al comienzo de una fase, sean `m_i=Σ_j G_ij x_j` y:

$$
\phi_i(m)=
\begin{cases}
\tanh(mW_i),&i\in\{0,1,2,3\},\\
m,&i\in\{4,5,6\}.
\end{cases}
$$

Con ganancia fija `g=bridge_gain`, en [0,1], la fase T_S actualiza simultáneamente los nodos de S:

$$
(T_S X)_i=
\begin{cases}
a_i[(1-g)x_i+g\phi_i(m_i)],&i\in S,\\
a_i x_i,&i\notin S.
\end{cases}
$$

La autoarista participa en m_i y el residual es otro término. Ambos deben aparecer en las derivadas. Cada fase lee el estado anterior a esa fase. No se actualizan todos los nodos en sitio.

Con L=`local_steps`, una ronda es `H_L=(T_local)^L ∘ T_group ∘ T_root ∘ T_group`, leída de derecha a izquierda. Se ejecutan dos rondas. El primer grupo observa las vistas y la raíz anterior. La raíz lee esos grupos actualizados. La segunda fase de grupo lee la nueva raíz y las vistas aún sin actualizar. Después se realizan L pasos locales síncronos con los auxiliares fijados para esa parte local. Es un intercambio grupo-raíz-grupo, no dos barridos puramente direccionales.

En el árbol, una contribución puede recorrer vista → grupo → raíz → otro grupo → vista dentro de una ronda. En la estrella, las vistas leen la raíz directamente. Los grupos anteriores a la actualización de raíz pueden influir en ella. Los estados finales de grupo calculados después de la última raíz ya no vuelven a esa raíz y no afectan a la salida final de vistas de star. Se archivan como estados reales, sin atribuirles un camino causal inexistente ni exigir una derivada no nula de esa salida respecto de ellos.

Las cadencias por nodo y ronda son L para una vista, dos para un grupo y una para la raíz. El caso principal L2 tiene dos frecuencias distintas. L4 tiene tres, pero son cadencias internas de cálculo. Ninguno completa por sí solo el contrato de tres ritmos globales de #709.

## Control sin realimentación y comparación plana

Con `feedback=False`, se normaliza el soporte y se sustituyen sus cuatro filas locales por filas identidad: una vista activa se lee solo a sí misma con peso uno. Una fila ausente sigue nula. Las filas de grupos y raíz permanecen iguales. Ese es el operador efectivo que debe devolverse como `hierarchy_weights`.

El control conserva los siete estados, sus formas, las fases y los productos densos. Sus conexiones efectivas hacia las vistas cambian y no siguen siendo las diecinueve entradas del soporte conectado. Tampoco equivale a poner a cero mensajes ajenos conservando una diagonal menor que uno. La renormalización local forma parte del control y de su identidad.

En L2, sus cuatro pasos locales corresponden algebraicamente al puente plano de cuatro vistas, topología self y J4, con los mismos parámetros, ganancia y R⁰. El control jerárquico sigue calculando ocho actualizaciones de grupo y dos de raíz. La equivalencia numérica entre implementaciones de formas diferentes requiere prueba. El coste del plano J4 sin esos auxiliares también debe conservarse. L1 corresponde a dos pasos locales y L4 a ocho. Esta última cuenta no amplía el tipo legado J=1,2,4 ni crea una acción J8.

Tree, star, control sin realimentación y plano comparten parámetros aprendibles, no estados, capacidad efectiva, gradientes o coste físico idénticos. El control directo sin mensajes y los casos de cancelación o interferencia pueden igualar o superar al árbol. Esos resultados deben poder descartar la necesidad del intercambio para el fixture, sin seleccionar una arquitectura por ese caso.

## Trabajo global y memoria

Los contadores separan R⁰ de las nuevas actualizaciones. Una lectura poblada ejecuta:

| Cuenta por fila | L1 | L2 principal | L4 |
| --- | ---: | ---: | ---: |
| `bridge_initial_views` | 4 | 4 | 4 |
| `bridge_local_updates` | 8 | 16 | 32 |
| `bridge_group_updates` | 8 | 8 | 8 |
| `bridge_root_updates` | 2 | 2 | 2 |
| `bridge_node_updates` | 18 | 26 | 42 |
| Reserva de 38 lecturas | 684 | 988 | 1596 |
| Ejecución de 22 lecturas con prefijos compartidos | 396 | 572 | 924 |

La cuota nueva cuenta `(8L+10) × lecturas`. Las 38 lecturas conservadoras cubren 34 de alternativas y cuatro de emisión. Las 22 requieren compartir los prefijos pertinentes, con 18 más cuatro. El DAG ejecutado no amplía la admisión conservadora. Las 152 vistas iniciales del máximo de 38 lecturas se registran aparte. Estas cuentas no son FLOPs ni todos los cálculos del predictor.

La reserva global debe rechazarse antes del primer MAC si no cabe. Bloques, mercados, rutas, contextos y reintentos no renuevan cuotas. L4 no aumenta una cuota declarada de 988 a 1596. Los bordes previstos son 988/987 y 1596/1595 para el evento conservador. K4 directo requiere 104/103 con L2. Si el consumidor admite el retorno identificado K1 cuando una ruta no cabe, conserva la jerarquía, L y su calendario. K no cuenta pasos locales ni actualizaciones de memoria de Titans.

La ausencia de vistas no reduce por sí sola el trabajo denso de una lectura poblada. Banco ausente, snapshot vacío y snapshot poblado con todas las vistas silenciadas conservan procedencias distintas. Los dos primeros pueden devolver cero trabajo del puente. El tercero conserva los IDs seleccionados y debe producir vistas, grupos, raíz y presencia útil nulos.

Los estados adicionales, G, máscaras, buffers, temporales, retención de autograd y trazas necesitan presupuesto propio. Los estados auxiliares finales retenidos por K lecturas añaden `B*K*3*H*element_size` bytes lógicos. Esa fórmula no mide el pico del allocator ni la VRAM física. Una medida del forward tampoco determina memoria o coste de backward. Las cotas de trabajo, memoria y serialización se comprueban por separado.

## Identidad, trazas y recuperación

`hierarchy=None`, el campo omitido y JSON null deben normalizarse como ausencia. No añaden campos de identidad arquitectónica ni buffers al checkpoint de módulo. La comprobación debe conservar inicialización, RNG y salida exacta para un mismo lote. Los sellos de fuentes y runtime sí cambian al cambiar el código. Eso no acredita recuperar una sesión histórica con otro runtime.

El contrato activo usa `shared_evidence_bridge_hierarchy_v1`, con topología, pertenencia, calendario, rondas, L, feedback y cuota identificados. Catálogo, factoría y consumidores deben distinguirlo de los lectores legado, lineal, recurrente e intervenido. La carga y la validación previa al sello contrastan buffers con la configuración declarada, no con una instancia que pueda estar alterada, antes de copiar parámetros.

El resultado activo `HierarchyBridgeRead` conserva las cuatro vistas e incorpora `group_values[B,2,H]`, `root_values[B,1,H]` y `hierarchy_weights[B,7,7]`. El campo heredado `bridge_weights[B,4,4]` es el bloque principal de G sin renormalizar. Sus filas pueden sumar menos de uno y no son una distribución completa entre vistas.

Las trazas tensorial y JSON activas deben identificar esos campos y el grafo efectivo. Grupos y raíz proceden del resultado ejecutado, no de reconstruirlos después desde las vistas finales. No añaden IDs o etiquetas de episodios. Las consultas no se presentan como capturadas si el formato no las almacena. El lector identificado permite distinguir el modo activo incluso sin banco. Las copias archivadas deben permanecer inmutables.

Sus verificadores comprueban estructura, tipos, formas, finitud, geometría y procedencia declarada. No recalculan la trayectoria numérica completa. Una alteración finita de auxiliares que conserve esos contratos puede superar esa comprobación estructural. Detectar un cambio de contenido requiere compararlo con una referencia confirmada mediante SHA-256. Esa integridad depende de la confianza en la referencia y no constituye una autenticación completa del cálculo o de su origen.

La recuperación comprueba identidad, geometría, pesos, contadores y fuentes, y reproduce la emisión siguiente con una sola confirmación del feedback. Los auxiliares no se recuperan como estado inicial de la siguiente lectura. El packet de colección puede reconstruir el desglose desde sus proyecciones y el lector sellado, sin multiplicar la historia de episodios por grupo.

## Composición con una etapa anterior a MAC

El puente jerárquico puede acompañar a un programa anterior a MAC (`RegionalExecutionConfig`) o a la familia jerárquica anterior (`RegionalHierarchyConfig`). La identidad de `FrozenMechanismConsumer` usa schema 6 y la receta `frozen_pre_mac_hierarchical_bridge_cohort_v1` cuando ambas etapas están activas. Conserva la configuración completa del predictor, el lector, el catálogo y la etapa anterior. Los casos individuales mantienen sus recetas y schema 5. Conservar esos esquemas no conserva el sello de fuentes de un runtime histórico.

La familia jerárquica anterior a MAC tiene ocho estados temporales: cinco fuentes, dos grupos y una raíz por predicción. El puente posterior tiene siete: cuatro vistas episódicas, dos grupos y una raíz por lectura. Son estados distintos, con entradas, matrices y reinicios propios. No se intercambian sus auxiliares ni se crean bancos episódicos o pesos rápidos adicionales. K conserva su significado de lectura y L el de pasos locales del puente.

Las cuotas también se mantienen separadas. La reserva durable anterior a MAC conserva nueve recursos de cohorte. Para la familia jerárquica anterior incluye una cota de 20 evaluaciones locales y diez auxiliares por preparación. El puente mantiene su límite de actualizaciones por fila del evento completo: con L2, 38 lecturas conservadoras requieren 988 actualizaciones. Las vistas iniciales y el trabajo observado del DAG se cuentan aparte. Ambos límites se comprueban antes del primer MAC. Un rechazo posterior a reservar no devuelve el cargo ni renueva el presupuesto por bloque, ruta o reapertura.

La comprobación de recuperación contrasta la identidad conjunta, pesos rápidos, estado auxiliar, recibos de cuotas y siguiente emisión con una ejecución sin corte. Un feedback ya confirmado no se aplica de nuevo al reabrir. Los auxiliares siguen reiniciándose por predicción o lectura y no se convierten en estado histórico recuperable.

La evidencia histórica del puente individual conserva su versión. Las pruebas nuevas de la composición distinguen sus dos familias anteriores a MAC y las activaciones individuales. No hay una mejora predictiva medida ni se habilita aprendizaje por activar estos componentes.

## Alcance matemático

La formulación de siete estados y su calendario es propia de este contrato. En aritmética real, medias normalizadas y mezcla convexa conservan el rango inicial acotado, y las vistas aplican tanh. Esa observación no certifica finitud en coma flotante ni contracción del intercambio. Se exige comprobar finitud antes de la activación y de la máscara, sin permitir que saturación o ausencia oculten overflow o NaN.

El Jacobiano debe incluir residual, autoaristas y las dependencias grupo-raíz-grupo. Su producto se evalúa en el estado propio de cada fase y ronda. Para derivar respecto del estado de consulta o de `W_r`, también cuenta su uso al formar R⁰ y en todas las actualizaciones locales. No se hereda la cota del [bucle plano](nonlinear-evidence-bridge-v1.md), no se detienen esos gradientes de forma implícita y no se declara una formalización Lean o una certificación numérica.

## Verificación técnica ejecutada

Los resultados de esta sección y del apartado de coste corresponden al puente individual de `006b8217d602a924fc833f23653d051655ee2ff2`. No acreditan la composición posterior de ambas etapas, que tiene su informe separado.

La selección reúne 685 IDs únicos aprobados, con 210 casos que utilizan CUDA. La tanda principal contiene 676 IDs. Los ocho casos CPU de guardas añaden ocho y la comprobación de límites de artefactos contiene cuatro, con tres solapados. No hubo fallos, errores ni omisiones en esas tandas. No se presenta el recuento como una ejecución de la suite completa del repositorio.

Se comprobaron paridad apagada, configuración, ecuaciones, gradientes, máscaras, finitud, disponibilidad, orden, bloques y RNG. La selección incluye factoría, catálogo, rechazo de geometría, cuotas antes de MAC, ambas trazas y recuperación nativa de la siguiente emisión con feedback único. Las comprobaciones negativas de artefactos incluyen contenido de caché alterado con mtime restaurado, salida privada obligatoria, rechazo de sobrescritura del recibo y rechazo de la carga desde la copia del script situada en el repositorio.

Coverage.py 7.16.1 y Radon 6.0.1 registran 1.737/2.027 sentencias y 707/926 ramas ejecutadas en once módulos. `bridge_hierarchy.py` cubre 92/92 sentencias y 38/38 ramas. La complejidad y CRAP se conservan por función y con su convención en el recibo. La cobertura completa de ese módulo no se extrapola al resto ni demuestra ausencia de defectos.

La campaña aislada detectó las diez sustituciones dirigidas por el fallo previsto en la fase de ejecución del test. Los controles inicial y restaurado aprobaron 153 casos cada uno, sin errores, omisiones o timeouts. Se comprobaron los bytes importados, se restauró la copia y las fuentes de producción quedaron intactas. Esos controles no aumentan los 685 IDs únicos de la selección. El alcance incluye identidad, orden de fases, persistencia entre rondas, normalización sin realimentación, gradientes auxiliares, cuota global, trazas, carga y doble envío de settlement. Este último se detecta por el control nativo de cursor y no certifica el registro interno de idempotencia C++. El resultado corresponde a esas diez sustituciones, sin una puntuación global de mutación del repositorio.

Los tres paneles analíticos se ejecutaron en CUDA FP64, con estados y parámetros fijados, sin optimizador:

| Panel | Observación y alcance |
| --- | --- |
| Transmisión | La señal llega a otra pareja de vistas por el árbol. Cortar los caminos declarados o anular la realimentación elimina esa llegada. Un control directo también recupera el signo sin jerarquía. |
| Cancelación | Con readout y objetivo técnico fijados, el árbol obtiene error cuadrático 0,8192438615495152 y el control sin realimentación obtiene cero. La evidencia canónica sigue presente. |
| Estados acotados y derivada | La sensibilidad simétrica local es 6,112847, conforme a 3521/576. La referencia sin realimentación es 16. Ambas superan uno y no constituyen una comparación general de estabilidad del predictor. |

Los cortes de esos paneles son referencias calculables del operador. No habilitan la composición productiva de jerarquía con intervención v1. Los paneles permiten contrastar transmisión y contraejemplos, sin acreditar que el árbol sea necesario o mejore una predicción financiera.

## Coste observado y límites

El benchmark de coste contiene 96 registros, correspondientes a 24 combinaciones distintas por etapa repetidas en orden normal e inverso, 48 registros por etapa. Las seis variantes son plano J4, tree L1/L2/L4, star L2 y control sin realimentación L2. Utiliza B=1/256, N=8/1.024, cuatro vistas, H=32 y ocho vecinos, con tres calentamientos y diez medidas por registro, en una RTX 4070 Laptop con PyTorch 2.14.0+cu130 y CUDA 13.

El componente ejecutó 624 invocaciones FP32, con referencia CPU FP64 sobre los mismos valores elevados de precisión y tolerancias atol=2e-6, rtol=1e-5. La preparación FP64 ejecutó 624 cohortes lógicas, distribuidas en bloques de hasta 128 filas iguales para todas las variantes. Las cohortes de 256 conservaron sus dos bloques, dando 936 llamadas físicas a `prepare`. Las salidas se retuvieron como tuplas, sin fusionarlas dentro del tiempo medido. El control sin realimentación se contrastó con el plano bajo las tolerancias declaradas, sin convertir esa paridad en igualdad de cómputo.

Para B=256 y N=1.024, las medianas fueron:

| Etapa y variante | Orden normal (ms) | Orden inverso (ms) |
| --- | ---: | ---: |
| Componente, plano J4 | 2,599 | 2,627 |
| Componente, tree L2 | 4,334 | 4,355 |
| Prepare en dos bloques, plano J4 | 23,603 | 19,700 |
| Prepare en dos bloques, tree L2 | 25,929 | 21,093 |

El árbol añade tiempo en esos contrastes. La variación entre órdenes se conserva y no demuestra una aceleración estable. Los percentiles p95/p99 son descriptivos con diez muestras, sin una estimación fiable de las colas. En ese componente, el pico transitorio del allocator fue 6.457.344 bytes para el plano y 7.507.968 para tree L2. No es VRAM física exclusiva ni memoria de un único modelo de producción.

La preparación mide las llamadas completas a `FrozenFinancialConsumer.prepare` y su sincronización. Excluye construcción del fixture, transferencias iniciales, encoding del corpus, colección, persistencia y feedback. No se midieron energía o tráfico físico de transferencias. Forward y backward no quedan igualados por estos resultados.

Se conservan dos intentos incompletos, detenidos por las guardas de trabajo de 32 MiB. No cuentan como pasadas. La ejecución completa divide en bloques la referencia CPU y la preparación de forma declarada, sin elevar cuotas, retirar filas ni cambiar las guardas. El [script reproducible](../../benchmarks/hierarchical_bridge_cuda.py) solo imprime el protocolo sin `--run`. La carga se ejecuta desde una copia externa, con salida privada nueva y sin sobrescribir recibos.

El experimento financiero no se ha ejecutado. Estos resultados no seleccionan la jerarquía ni acreditan mejora predictiva, ahorro global de memoria o superioridad arquitectónica. La [verificación posterior de la composición](../../reports/engineering/composed-hierarchy-20261023/README.md) conserva sus pruebas, cuotas y coste como un contraste técnico separado.

## Antecedentes y simplificaciones

[HRM v4](https://arxiv.org/html/2506.21734v4), revisado el 30 de septiembre de 2026, sección 2, combina dos módulos recurrentes con cadencias distintas. [TRM v1](https://arxiv.org/html/2510.04871v1), del 6 de octubre de 2025, secciones 3.3 y 4.1–4.3, ofrece una recurrencia compartida más sencilla y ablaciones de sus estados. Sus críticas preceden a HRM v4 y no se presentan como una revisión de esa versión. El presente contrato no importa supervisión intermedia, gradientes truncados o parada aprendida de esos trabajos.

[FractalNet v4](https://arxiv.org/html/1605.07648v4), del 26 de mayo de 2017, sección 3, define una expansión estructural autosemejante y entrenamiento con drop-path. Este árbol fijo no reproduce ese procedimiento. [SEAL v2](https://arxiv.org/html/2506.10943v2), del 18 de septiembre de 2025, secciones 3.1 y 5, genera propuestas, adapta pesos y usa una valoración posterior, con límites de olvido y coste. Elegir aquí una topología fija no reproduce SEAL ni aprende una política de J.

La evidencia de [Roy y colaboradores, 2022](https://www.nature.com/articles/s41467-022-29384-4.pdf), sobre conjuntos de engramas distribuidos, y el [resumen de Schmitt y colaboradores, 2017](https://www.nature.com/articles/nature22073), sobre amplificación selectiva de conectividad, motivan controles de cooperación y comunicación. No demuestran recuerdos completos replicados en cada región ni convierten esta partición en una jerarquía cerebral o memoria holográfica. Las vistas y grupos son posiciones computacionales, sin especialización aprendida acreditada. Los antecedentes no validan resultados financieros ni superioridad de esta variante.
