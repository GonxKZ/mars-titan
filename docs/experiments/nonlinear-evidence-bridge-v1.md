# Puente episódico con recurrencia no lineal

`shared_evidence_bridge_nonlinear_v1` amplía el puente de vistas sobre un único `EpisodeSnapshot`. Conserva una selección de episodios común, una proyección de valores por lectura y consultas, atenciones y transformaciones propias por región. Los puentes comunican estados derivados del mismo registro. No añaden bancos regionales ni copias persistentes del episodio. Su utilidad predictiva está sin medir. La [verificación técnica](../../reports/engineering/nonlinear-evidence-bridge-20261022/README.md) separa pruebas, mutación y coste del componente.

## Actualización y activación

El puente forma R⁰ mediante las atenciones y transformaciones existentes. Para cada paso j, todas las regiones usan el mismo estado anterior:

$$
M^j=G R^j,\qquad U^j_r=M^j_rW_r,
\qquad R^{j+1}_r=a_r[(1-g)R^j_r+g\tanh(U^j_r)].
$$

G es el grafo de mensajes normalizado después de anular emisores y receptores ausentes. W_r es la matriz `region_value` ya existente de la región receptora. Se reutiliza entre pasos. La transición no añade parámetros, sorteos ni estado persistente entre predicciones. Los pasos intermedios forman parte del cómputo y, cuando corresponde, de su grafo de autograd. Rᴶ se devuelve en `BridgeRead` y sus valores pueden archivarse en la traza existente. La proyección de salida existente consume Rᴶ. Cada región conserva los parámetros propios del puente original, sin sustituirlos por copias del episodio.

`BridgeRecurrenceConfig` declara explícitamente `steps` en {1,2,4} y `max_node_updates_per_row` entre 1 y 1216. El campo opcional `EvidenceBridgeConfig.recurrence` activa este recorrido. Requiere `cycles=0`. Las topologías existentes son `self`, `ring` y `complete`. El número J de pasos no cuenta lecturas K ni actualizaciones de pesos rápidos de Titans. Tampoco es el factor C de CM-v1.

Sin recurrencia, la configuración omite el campo opcional y mantiene el camino lineal anterior. `recurrence:null` se canoniza como ausencia. El modo activo usa una identidad propia incluso con ganancia cero. Conserva formas y parámetros emparejados, pero no puede cargar un checkpoint de otro modo, J o presupuesto como si fueran equivalentes. La identidad de configuración anterior se conserva. Las identidades del runtime que sellan fuentes cambian al cambiar su código. Una sesión sellada con fuentes históricas requiere su runtime compatible, no una identidad artificialmente conservada.

La fábrica `MarsTitanVariant` construye la configuración tipada desde `evidence_bridge`. El catálogo activo declara `reader_contract="shared_evidence_bridge_nonlinear_v1"`. Una política ligada al catálogo anterior se rechaza. El módulo nuevo participa en los sellos de las implementaciones financieras. La configuración declarativa conserva la ampliación desactivada por defecto.

## Presupuesto de la decisión y recuperación

La admisión comprueba el trabajo antes del primer MAC. Una lectura poblada forma R vistas iniciales y ejecuta R×J actualizaciones nuevas. La cuenta no representa todos los FLOPs. Las regiones ausentes no reducen el trabajo denso ejecutado. Con banco vacío no hay actualizaciones de este bucle.

El lector directo reserva todas sus lecturas K. El colector reserva además las alternativas admisibles de ambos contextos y la emisión final. La unión conservadora de las seis rutas suma 17 lecturas por contexto. El máximo es 34 de alternativas más cuatro de emisión. Para R=3 y J=2 se necesitan 228 actualizaciones nuevas. El extremo R=8, J=4 requiere 1216. Compartir prefijos reduce el trabajo real hasta nueve lecturas por contexto, pero la admisión conserva el máximo sin prefijos. Dividir el lote o cambiar el contexto no renueva la cuota.

El evento ancla reservado por el colector se reutiliza para emitir. Las alternativas no multiplican admisiones ni feedback y no avanzan por separado el estado durable del predictor. Si la reserva conjunta no cabe, se rechaza antes de ejecutar MAC. Cuando solo una ruta solicitada queda fuera del presupuesto y K1 sí cabe, el consumidor puede emitir con K1 y el motivo `reader_budget`, conservando J. Los límites de trabajo, buffers y trazas son comprobaciones distintas.

La emisión activa distingue `bridge_initial_views` y `bridge_node_updates`, además de `bridge_value_projections`. Su consumidor usa una receta y esquema propios. El packet de colección conserva su esquema y `reader_value_projections`. Sus vistas iniciales y pasos J se reconstruyen con ese contador y el lector sellado. No se almacena otra historia regional. La recuperación comprueba catálogo, parámetros, código, J y presupuesto. Los errores maduros siguen perteneciendo a la predicción realmente emitida y recibida, no a las alternativas descartadas.

## Alcance matemático y experimental

Las comprobaciones de finitud preceden a tanh para no ocultar un desbordamiento mediante saturación. El bucle conserva los gradientes de estados, mensajes y matrices regionales. La selección discreta de episodios mantiene su contrato anterior.

Tanh y la mezcla residual mantienen un rango acotado cuando R⁰ ya está en ese rango. Esto no implica contracción. El caso escalar `z_next=tanh(2*z)`, iniciado en cero, conserva estados cero y tiene derivada 16 después de cuatro pasos. Las potencias de un operador fijo no representan este Jacobiano no lineal completo.

La condición suficiente de norma de la sección siguiente no está impuesta por el módulo. Sin reinyección de evidencia, una contracción uniforme también haría que las vistas tendiesen a cero al aumentar J. No se propone una contracción fuerte como mejora automática de memoria.

La variante no reproduce una región cerebral, un recuerdo holográfico o SEAL. Las hipótesis, antecedentes y controles están en la [revisión de recurrencia regional](../research/regional-recurrence-extensions.md). La fusión anterior a MAC y CM-v1 conservan identidades separadas. Las pruebas con fixtures comprueban mecanismos, no error financiero. La campaña completa y el test final permanecen cerrados.

## Condición suficiente sobre el bucle regional

Esta derivación propia usa desigualdades de norma convencionales. No se reivindica novedad ni se atribuye a la familia 325. No se ha formalizado en Lean ni se ha certificado su evaluación en coma flotante. El alcance comprende únicamente este bucle, con sus hipótesis explícitas.

Para una fila, hay R estados regionales z_r en R^H, una máscara a_r en {0,1}, una ganancia fija g en [0,1] y matrices W_r. Los estados se actualizan de forma síncrona:

$$
F_{G,a}(z)_r=a_r\left[(1-g)z_r+g\tanh\left(\sum_s G_{rs}a_sz_sW_r\right)\right].
$$

Las filas disponibles de G contienen pesos no negativos que suman uno. Las filas ausentes son cero. Por tanto, la suma de cada fila es como mucho uno. En la implementación el grafo depende de la topología y la disponibilidad, no de z. W_r y g tampoco dependen del estado de este bucle. Esta distinción es necesaria para el Jacobiano.

### Jacobiano del bucle completo

Al apilar las regiones como vectores columna, sean D_a la máscara por bloques, W la matriz diagonal por bloques con W_r transpuesta y D_tanh la diagonal de derivadas de tanh en los valores anteriores a la activación. Entonces:

$$
D_zF=D_a[(1-g)I+gD_{\tanh}W(G\otimes I_H)D_a].
$$

La identidad residual forma parte de este Jacobiano. Si el grafo, la ganancia o las matrices dependiesen de z, aparecerían términos adicionales. El argumento siguiente no los omite por una aproximación, sino que corresponde a su ausencia en esta transición concreta. Las consultas de evidencia que forman el estado inicial son otra dependencia y no quedan cubiertas por este cálculo.

### Una norma común proporciona una condición suficiente

Definir la norma por bloques como ||z||_{max,2}=max_r||z_r||_2. Supóngase una cota real L sobre todas las normas espectrales ||W_r||_2. Tanh aplicada coordenada a coordenada es 1-Lipschitz en norma euclídea. Para dos estados y una misma topología y máscara, la desigualdad triangular y los pesos no negativos dan:

$$
\left\|\sum_sG_{rs}a_s(z_s-y_s)W_r\right\|_2
\le L\sum_sG_{rs}a_s\|z_s-y_s\|_2
\le L\|z-y\|_{max,2}.
$$

Con la parte residual completa:

$$
\|F_{G,a}(z)-F_{G,a}(y)\|_{max,2}
\le q\|z-y\|_{max,2},\qquad q=1-g+gL.
$$

Si 0<g<=1 y 0<=L<1, entonces 0<=q<1. La misma norma sirve para cualquier secuencia de grafos y máscaras que cumplan las condiciones anteriores, comparando dos trayectorias sometidas a la misma secuencia. Por inducción, después de J pasos su diferencia es como mucho q^J veces la inicial en esa norma. La conversión conservadora a norma euclídea apilada introduce sqrt(R):

$$
\|z^J-y^J\|_2\le\sqrt{R}\,q^J\|z^0-y^0\|_2.
$$

Esto no deriva de cotas individuales del radio numérico. Se ha utilizado una norma común y la actualización no lineal completa. Si las dos trayectorias reciben grafos distintos, hacen falta términos que acoten esa diferencia. Una estimación ordinaria de ||W_r||_2 no es una certificación numérica y una penalización no impone por sí sola L<1.

Aunque q<1, el factor sqrt(R) q^J puede ser mayor o igual que uno para un J finito. La conversión no garantiza contracción euclídea en ese horizonte ni demuestra que haya amplificación. La contracción obtenida inicialmente es la de la norma común por bloques.

### La contracción tiene un coste informativo

Este bucle no vuelve a inyectar evidencia ni añade un sesgo después de formar z^0. El estado cero es un punto fijo de cada F. Bajo la condición uniforme q<1, ||z^J||_{max,2}<=q^J||z^0||_{max,2}, incluso con grafos y máscaras variables. Aumentar indefinidamente J eliminaría la representación inicial. Esta condición suficiente no recomienda imponer contracción fuerte ni demuestra retención semántica.

Con g=1, una región, G=[1], W=[2] y z^0=0, todos los estados son cero y la derivada de J pasos es 2^J. El contraejemplo de las pruebas usa J=4 y obtiene 16. La cota de rango de tanh no sustituye la condición L<1.

La comparación principal mantiene J finito en {1,2,4}, sin imponer esta restricción nueva. Una normalización espectral estricta sería otra intervención, con identidad y control propios, que tendría que contrastar error, retención y coste. No se añade a CM-v1 ni se cambia B mediante esta derivación.
