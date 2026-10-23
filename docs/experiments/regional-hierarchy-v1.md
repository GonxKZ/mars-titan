# Jerarquía regional de ocho estados antes de MAC

`pre_mac_regional_hierarchy_v1` añade dos grupos y una raíz a las cinco representaciones de la fusión regional. Reutiliza los pesos de [`RegionalFusion`](../../src/mars_titan/models/titans/regional_fusion.py), con un calendario fijo anterior a MAC. La implementación está verificada técnicamente en CPU y CUDA. La [PR 723](https://github.com/GonxKZ/mars-titan/pull/723) sigue en borrador y pendiente de integrar. La matriz de 128 registros se ha ejecutado y no sostiene una aceleración consistente. No se ha evaluado una mejora predictiva.

La configuración ocupa el campo existente `FinancialConfig.regional_execution`, junto con los [programas anteriores](regional-program-v1.md) y `None`. No añade un segundo campo financiero ni acciones al catálogo de 36 combinaciones. La [declaración técnica](../../configs/experiments/regional-hierarchy-v1.json) conserva siete variantes nuevas y una referencia anterior. La familia sigue desactivada por defecto.

Esta jerarquía se ejecuta sobre modalidades antes de MAC. Es distinta de la [jerarquía episódica de #719](https://github.com/GonxKZ/mars-titan/issues/719), que combina cuatro vistas de un snapshot después de MAC y tiene siete estados. Sus recuentos, identidades y pruebas no son intercambiables.

## Estado, contexto y mensajes

El orden de las hojas es `prices`, `news`, `charts`, `fundamentals`, `macro`. Para una fila, X contiene cinco vectores de dimensión H y P sus cinco presencias. La consulta contextual se realiza una vez antes del programa:

$$
\bar x=\frac{\sum_{i=0}^{4}P_iX_i}{\max(1,\sum_{i=0}^{4}P_i)},\qquad
c=\operatorname{read\_confirmed}(\bar x,f,t,\mathcal C_{<t}).
$$

La consulta utiliza los bloques ya proyectados y el contexto confirmado del flujo f antes de t. Incluye la proyección de salida de PT4. Los tokens pendientes de otros bloques no entran en esa lectura. Con contexto desactivado, c es cero y no se prepara una escritura PT4 para esa fila. Durante las diez fases se reutiliza el mismo c.

El estado de trabajo es:

$$
Z=(R_0,R_1,R_2,R_3,R_4,G_0,G_1,S)\in\mathbb R^{8\times H},\qquad
Z^0=(P_0X_0,\ldots,P_4X_4,0,0,0).
$$

La partición principal es `((0,2),(1,3,4))`, precios y gráficos frente a noticias, fundamentales y macro. El control permutado usa `((0,1),(2,3,4))`. Son las dos únicas particiones registradas. No se presupone una especialización de esos grupos.

Cada grupo está presente si lo está alguna de sus hojas y la raíz si hay alguna hoja presente:

$$
a=(P_0,\ldots,P_4,\bigvee_{j\in G_0}P_j,\bigvee_{j\in G_1}P_j,\bigvee_{j=0}^{4}P_j).
$$

La pertenencia G en estas expresiones identifica las posiciones del grupo. El vector de disponibilidad no se deduce de que exista una autoarista. Para los emisores registrados Nᵢ, el mensaje uniforme es:

$$
m_i(Z)=a_i\frac{\sum_{j\in N_i}a_jZ_j}{\max(1,\sum_{j\in N_i}a_j)}.
$$

La máscara elimina emisores y receptores ausentes antes de normalizar. Un receptor ausente recibe cero. La geometría se fija así, con g(i) como el grupo de la hoja i:

| Topología | Emisores de Rᵢ | Emisores de Gg | Emisores de S |
| --- | --- | --- | --- |
| `tree` | Rᵢ y Gg(i) | Gg, sus hojas y S | S, G₀ y G₁ |
| `star` | Rᵢ y S | Gg y S | S, las cinco hojas, G₀ y G₁ |
| `flat_control` | Rᵢ y S | Gg y sus hojas | Las cinco hojas |

Con todas las fuentes presentes, las tres geometrías tienen 22 coeficientes booleanos activos. Sus grados y caminos son distintos. La raíz del control plano no incluye su autoestado en el promedio, aunque conserve el residual de su actualización. Los grupos del control plano se calculan y mantienen su propio estado, pero no transmiten a raíz ni hojas.

Con `feedback=False`, solo las filas receptoras de las hojas se sustituyen por identidad y se renormalizan. Así mᵢ=Rᵢ para una hoja presente. Los grupos y la raíz siguen calculándose. Permanecen la consulta c y la proyección del estado propio, por lo que este control corta la realimentación auxiliar hacia las hojas.

## Transiciones y calendario

[`RegionalHierarchy`](../../src/mars_titan/models/titans/regional_hierarchy.py) recibe el operador existente y utiliza sus normalizaciones y matrices. No registra otra copia de sus parámetros. Para un receptor auxiliar i, sea grupo o raíz:

$$
\Delta_i=W_2\operatorname{SiLU}\!\left(W_1\left(\operatorname{LN}_s(Z_i)+m_i(Z)\right)\right),\qquad
Z_i'=\operatorname{where}(a_i,Z_i+g_s\Delta_i,0).
$$

W₁ y W₂ son las dos matrices de `summary_update`, compartidas por los grupos, la raíz y ambas rondas. La ganancia gₛ es la del operador. Una fase lee el estado anterior completo y escribe el siguiente. Los dos grupos se actualizan simultáneamente. Los receptores que no intervienen en esa fase conservan su valor.

Cada paso local usa T=`summary_to_local`:

$$
u_i=c+T m_i(Z),\qquad
n_i=\operatorname{where}\!\left(P_i,\operatorname{LN}_{\ell}(R_i+u_i),0\right),\qquad
(q_i,k_i,v_i)=W_{QKV}n_i.
$$

La ruta uniforme calcula QKV completo y después anula q y k. La atención local tiene únicamente la diagonal `self`, por lo que su resultado para una hoja es vᵢ. Con Oₗ=`local_out` y la ganancia local gₗ:

$$
R_i'=\operatorname{where}(P_i,R_i+g_{\ell}O_{\ell}v_i,0).
$$

El contrato local exige `topology="self"`, `routing="uniform"`, `cycles=2`, `local_steps=2` y `summary=True`. Las ganancias conservan los valores declarados del operador. La topología de ocho posiciones pertenece a la configuración nueva y no oculta un anillo de atención entre las cinco hojas.

El programa fijo `hierarchical_group_root_v1` realiza dos rondas:

```text
grupo → raíz → grupo → local → local
```

Cada ronda actualiza dos grupos dos veces, una raíz y las cinco hojas dos veces. El total por fila es de 20 evaluaciones locales, ocho de grupo y dos de raíz. Son diez registros de fase y 30 evaluaciones de receptor. Los estados se conservan entre fases y rondas, con versiones de lectura y escritura desde 0→1 hasta 9→10. La segunda ronda empieza en la versión 5.

Los mensajes locales cambian con la hoja y el estado de cada fase. Se aplican 20 proyecciones T por fila, también en posiciones enmascaradas. `common_skip` añade una proyección de la raíz final. El programa anterior proyecta su resumen una vez por ciclo. Los grupos, por tanto, añaden trabajo aunque no añadan parámetros.

Se rechazan entradas no finitas antes de enmascararlas. También se comprueban las actualizaciones auxiliares, la normalización local, QKV antes de anular q y k, las actualizaciones locales y la suma residual de salida antes de seleccionar los bloques presentes. Las regresiones incluyen valores inválidos en receptores ausentes. Estos controles no demuestran contracción o estabilidad global. El Jacobiano pertinente incluye las diez fases, sus residuales, normalizaciones, proyecciones y parámetros compartidos.

## Salida, referencias y variantes

Con O=`readout`, las dos salidas registradas para una hoja presente son:

$$
Y_i=X_i+O(R_i^{10}-R_i^0)\quad\text{en local\_only},
$$

$$
Y_i=X_i+O(R_i^{10}-R_i^0+c+TS^{10})\quad\text{en common\_skip}.
$$

Una hoja ausente conserva exactamente Xᵢ. `common_skip` mantiene la misma fórmula y pesos entre controles, pero no obliga a que sus raíces finales coincidan. Con ese readout, cortar la realimentación local no eliminaría la vía directa de la raíz.

| Variante declarada | Topología y partición | Realimentación | Readout |
| --- | --- | --- | --- |
| `hierarchy_tree_local` | Árbol, partición principal | Sí | Local |
| `hierarchy_star_local` | Estrella, partición principal | Sí | Local |
| `hierarchy_permuted_tree_local` | Árbol, partición permutada | Sí | Local |
| `hierarchy_tree_no_feedback_local` | Árbol, partición principal | No | Local |
| `hierarchy_flat_local` | Control plano, partición principal | Sí | Local |
| `hierarchy_tree_common_skip` | Árbol, partición principal | Sí | Salto común |
| `hierarchy_flat_common_skip` | Control plano, partición principal | Sí | Salto común |
| `legacy_self_uniform_interleaved` | Programa anterior `summary_interleaved_v1` | Contrato anterior | Contrato anterior |

La referencia anterior realiza 20 evaluaciones locales y dos de resumen. Sus resúmenes se actualizan después de los pasos locales, mientras el calendario nuevo actualiza la raíz antes de ellos. El nuevo `flat_control` comparte las 30 evaluaciones del árbol y no reproduce numéricamente esa referencia de 22. Una comparación entre ambos cambia también calendario y trabajo. La razón 30/22 no es una razón de FLOPs o latencia.

El caso `regional_execution=None` conserva la ruta anterior, sin configuración, buffers o trazas nuevos. Es distinto de la configuración `legacy_self_uniform_interleaved` de la tabla. Con PT4, la ruta None consulta el contexto después de la fusión. La referencia intercalada y la familia nueva lo consultan antes del grafo. Las huellas de código cambian con una implementación y no autorizan sustituir silenciosamente un runtime histórico.

Los ocho estados son temporales y se reinician en cada preparación. El resultado del componente expone su `working_state[B,8,H]` real para las comprobaciones. El predictor financiero lo descarta después de extraer bloques y traza. Los grupos no crean episodios, etiquetas o copias persistentes de datos. Sí existen auxiliares, concatenaciones y temporales que consumen memoria.

K sigue contando refinamientos del lector episódico después de MAC. J cuenta las vueltas del puente episódico. Los factores C y M de CM-v1 mantienen su significado y su identidad. Ninguno cuenta las diez fases de esta familia. La ruta emitida conserva una transición MAC por observación y una preparación del token final para PT4. Las alternativas pueden repetir preparaciones desde el prefijo común, pero solo se confirma el estado de la emisión.

## Identidad, carga, trazas y límites

`RegionalHierarchyConfig` es una dataclass congelada con cuatro campos: `topology`, `groups`, `feedback` y `readout`. Admite únicamente las dos particiones completas y tipos exactos, incluidos índices enteros y booleanos explícitos. La identidad `pre_mac_regional_hierarchy_v1` registra además posiciones, calendario, mensajes, lectura y límite lógico de trabajo. Los parsers rechazan mezclas, campos omitidos e identidades de otra familia. La clase e identidad de `RegionalExecutionConfig` siguen separadas.

El buffer booleano de geometría se deriva de la configuración. Su payload y el del predictor se validan antes de copiar pesos. Un checkpoint válido puede reparar valores booleanos instalados alterados, sin convertir tipos o formas incompatibles en otra topología. El catálogo, el módulo físico y el consumidor deben declarar la misma identidad canónica. `FrozenFinancialConsumer` continúa rechazando programas activos y el recorrido durable utiliza `FrozenMechanismConsumer`.

El descriptor nuevo tiene `schema_version=2` y cuatro claves: `schema_version`, `execution`, `operator` y `dtype`. Los cuerpos de ejecución y operador contienen `configuration`, `hidden_size` y `max_batch`. Cada fila exporta identificadores de configuración y operador, programa, norma contextual, disponibilidad de ocho posiciones, las diez fases y las dos unidades de trabajo. Las fases proceden del recorrido ejecutado, con ronda, receptores y versiones. El esquema 1 conserva sus filas y su parser anterior.

El archivo contrasta estructura, orden, disponibilidad OR y contadores. No autentica el contenido neuronal de un contexto que no guarda ni certifica kernels. Los estados Z no se archivan como memoria persistente. Las comprobaciones de colección distinguen coherencia entre trazas y autenticación de sus entradas.

El máximo lógico del archivo es de 8192 filas. Cada exportación física respeta `max_batch`, con un máximo de 4096, y H no supera 512. La admisión del componente aplica además:

$$
\widehat W(B,H,b)=B(512Hb+1024)\leq128\ \mathrm{MiB},\qquad b\in\{4,8\}.
$$

Es una estimación conservadora de tensores lógicos. Excluye MAC, runtime, metadatos Python, JSON y workspaces opacos de bibliotecas. No certifica el pico del asignador, RSS o VRAM. Si la estimación supera el límite se rechaza la preparación antes de las fases neuronales, sin reducir silenciosamente el lote ni elevar la cota. Los dos vectores de grupo añaden `2BHb` bytes de estado lógico respecto al resumen anterior, pero esa cifra no comprende todos los temporales o activaciones de autograd.

La sesión conserva sus nueve recursos y consulta `regional_work_per_row` para esta familia. Con N filas, la cota de preparaciones es R=N sin colección, R=2N con colección sin PT4 y R=3N con colección y PT4. Cada intento reserva 20R evaluaciones locales y 10R auxiliares. Los cargos acumulados de la cohorte se contrastan con su cuota total y no se devuelven tras fallo o reapertura.

Los otros recursos mantienen su contrato: R observaciones del núcleo, R decisiones regionales, hasta R consultas contextuales, R observaciones MAC si existe el módulo, cotas de lectores existentes y hasta una confirmación contextual por mercado. Las fases no renuevan saldo por grupo, bloque, ruta o alternativa. `settlement` no reserva trabajo predictivo y el calentamiento no ejecuta colección ni lectores. Las reservas conservadoras se distinguen de las visitas observadas. La confirmación nativa y la compactación de recibos siguen el [contrato durable anterior](regional-program-v1.md#reserva-emisión-y-recuperación).

## Comprobaciones y protocolo técnico

Las [pruebas del componente](../../tests/models/titans/test_regional_hierarchy.py) recorren las 32 máscaras posibles de cinco hojas, incluida ausencia total y grupos vacíos. Contrastan una referencia desenrollada independiente, el estado real, gradientes, diferencias finitas, readouts, reinicio, partición de filas, RNG y cargas inválidas. Los falsadores de comunicación mantienen c fijo, porque la consulta PT4 puede crear por sí misma una dependencia entre modalidades.

La [fixture de colección financiera](../../tests/memory/test_regional_hierarchy_collection.py) tiene otra admisión: exige precios y gráficos presentes. De las 32 máscaras, ocho son admisibles y 24 se rechazan antes de preparar el núcleo. Las ocho filas verifican auxiliares reales, incluido el segundo grupo vacío cuando faltan noticias, fundamentales y macro. El primer grupo no puede estar vacío en esa fixture. Estos rechazos no son casos omitidos ni amplían la admisión del predictor para reproducir las 32 máscaras del componente.

Las [pruebas financieras](../../tests/models/titans/test_financial_regional_hierarchy.py), [de archivo](../../tests/models/titans/test_regional_hierarchy_archive.py) y [de sesión](../../tests/memory/test_regional_hierarchy_session.py) cubren construcción, factoría, sellado, siguiente emisión, cuotas y recuperación. Las comprobaciones técnicas incluyen CPU y CUDA. El [informe de verificación y costes](../../reports/engineering/regional-hierarchy-matrix-20261022/README.md) detalla las pruebas, la cobertura y las mutaciones dirigidas con sus ámbitos. Esta evidencia no completa el alcance científico de #709 ni sustituye la revisión e integración de la PR.

[`regional_hierarchy_matrix_cuda.py`](../../benchmarks/regional_hierarchy_matrix_cuda.py) se ha ejecutado con ocho controles, dos opciones de contexto, FP32/FP64, lotes de 1/256 y dos órdenes globales inversos. Los cuatro estratos terminaron correctamente. Son 128 registros, 256 preparaciones de prefijo, 256 calentamientos y 640 muestras, 1152 preparaciones en total. El modelo de fixture usa `mac_online`, H=32, semilla 42 y cabeza de cuantiles. Cada condición y orden reconstruye su propio prefijo de dos eventos y repite el tercero. No son sesiones financieras independientes.

El productor contrastó nombres, formas y valores exactos de los parámetros dentro de cada contexto y precisión. La geometría se comprueba según su configuración, sin exigir que coincida entre topologías. La agregación validó las configuraciones financieras reales, los manifiestos de fuentes, los registros únicos y los contadores. Las huellas de tensores incluyen buffers geométricos, por lo que no permiten reconstruir por sí solas la igualdad de pesos entre topologías. Los dos órdenes conservan sus cinco muestras por registro, con mínimo, mediana, máximo y caudal. No permiten estimar colas de latencia o intervalos de confianza.

La suma de los cuatro procesos de medida fue de 236,661 s y la de su supervisión externa de 251,995 s. Son costes completos de ejecución, separados del tiempo interior de `FinancialPredictor.prepare`. Las pruebas, las mutaciones y la ejecución instrumentada para cobertura quedan fuera de esos costes.

Los máximos registrados por el asignador Torch fueron de 99,5625 MiB asignados y 124 MiB reservados, con modelo e inputs residentes. El máximo de RSS muestreado fue de 1515,043 MiB. Este último no es un pico de RAM por condición ni el máximo de vida del proceso comunicado por otra interfaz. Los tres valores tienen ámbitos distintos.

Con PT4, FP32 y lote de 256, las medianas por preparación fueron:

| Orden global | Referencia intercalada (ms) | Árbol local (ms) |
| --- | --- | --- |
| Original | 45,092 | 36,728 |
| Inverso | 42,688 | 61,048 |

La diferencia cambia de signo al invertir el orden. Cada comparación conserva el contexto, la precisión, el lote y el orden de su referencia. Los órdenes permanecen separados y no se selecciona el más favorable. Estas medidas no sostienen una aceleración consistente ni aportan evidencia de mejora predictiva.

El modo de protocolo está comprobado y no importa Torch ni crea salida. Desde la raíz del repositorio, con el entorno del proyecto preparado:

```bash
uv run --no-sync python benchmarks/regional_hierarchy_matrix_cuda.py
```

La ejecución requiere un estrato explícito y una salida nueva fuera del repositorio. Este ejemplo ejecutaría solo FP32 con una fila:

```bash
hierarchy_artifacts=$(mktemp -d)
uv run --no-sync python benchmarks/regional_hierarchy_matrix_cuda.py \
  --run --dtype float32 --batch 1 \
  --output "$hierarchy_artifacts/float32-1"
```

Los cuatro estratos son el producto de `float32`/`float64` y `1`/`256`. Cada uno contiene 32 registros. El guion exige `cuda:0` en la RTX 4070 y no cambia a CPU si falta CUDA. Tiene un límite interno de 180 segundos por proceso, un asignador limitado a 512 MiB y una reserva global libre de 256 MiB. La caché de tensores CPU para comparaciones está acotada a 128 MiB y el JSON a 16 MiB. Estos límites no describen toda la memoria del proceso. Las ejecuciones incompletas deben conservarse como tales, sin seleccionar solo sus tiempos favorables.

Las comprobaciones usan fixtures y parámetros fijados, sin pasos de optimizador compartido ni selección financiera. La actualización asociativa de pesos rápidos de MAC mantiene su contrato separado. No se han entrenado estas variantes ni ejecutado un estudio científico. La campaña completa y el test final de 2024 siguen cerrados. [#709](https://github.com/GonxKZ/mars-titan/issues/709) conserva pendiente el selector aprendido y los tres ritmos globales y [#711](https://github.com/GonxKZ/mars-titan/issues/711) mantiene separado su ciclo exterior de propuestas. No se afirma superioridad o novedad a partir de esta jerarquía.
